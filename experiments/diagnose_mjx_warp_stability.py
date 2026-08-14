from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco  # type: ignore[import-untyped]
import numpy as np
from mujoco import mjx

from mujoco_truss_gen import (
    DomainRandomizationConfig,
    MjxNodeVelocityEnv,
    MujocoNodeVelocityCommandEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)
from mujoco_truss_gen.mujoco_model.model import MujocoModel


def _keys(seed: int, batch_size: int) -> jax.Array:
    return jax.random.split(jax.random.key(seed), batch_size)


def _scalar_at(value: Any, index: int) -> float:
    return float(np.asarray(jax.device_get(value))[index])


def _data_snapshot(state: Any) -> dict[str, np.ndarray]:
    return {
        name: np.array(jax.device_get(getattr(state.data, name)), copy=True)
        for name in ("time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart")
    }


def _capacity_defaults(
    model_source: Any,
    batch_size: int,
    *,
    contact_capacity_per_env: int | None,
    constraint_capacity: int | None,
    scale: int,
) -> tuple[int, int, dict[str, int]]:
    native_model = MujocoModel(model_source).model
    per_env_contacts = contact_capacity_per_env or max(16, 4 * int(native_model.ngeom))
    per_env_constraints = constraint_capacity or max(
        128,
        8 * (int(native_model.nv) + int(native_model.neq) + int(native_model.ntendon)),
    )
    capacities = {
        "contact_capacity_per_env": per_env_contacts * scale,
        "contact_capacity_total": per_env_contacts * batch_size * scale,
        "constraint_capacity_per_env": per_env_constraints * scale,
    }
    return (
        capacities["contact_capacity_total"],
        capacities["constraint_capacity_per_env"],
        capacities,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_source = get_mujoco_spec(args.topology, realistic=args.realistic)
    requested_integrator = {
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
    }[args.integrator]
    model_source.option.integrator = requested_integrator
    domain_randomization = (
        DomainRandomizationConfig(
            initial_translation_x_range=(-1.0, 1.0),
            initial_translation_y_range=(-1.0, 1.0),
            initial_yaw_range=(-math.pi, math.pi),
        )
        if args.randomize_pose
        else None
    )
    config = TrussEnvConfig(
        model_source,
        nsubsteps=args.nsubsteps,
        max_steps=args.episode_steps,
        speed=args.speed,
        domain_randomization=domain_randomization,
        runtime_apply_control_noise=False,
    )
    naconmax, njmax, capacities = _capacity_defaults(
        model_source,
        args.batch_size,
        contact_capacity_per_env=args.warp_contact_capacity_per_env,
        constraint_capacity=args.warp_constraint_capacity,
        scale=args.capacity_scale,
    )
    env = MjxNodeVelocityEnv(
        config,
        mjx_impl="warp",
        warp_graph_mode=args.graph_mode,
        warp_naconmax=naconmax,
        warp_njmax=njmax,
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    reset_where = jax.jit(env.reset_where)
    reset_keys = _keys(args.seed, args.batch_size)
    observation, state = reset(reset_keys)
    jax.block_until_ready(observation)
    episode_initial_state = _data_snapshot(state)
    episode_action_counts = np.zeros(args.batch_size, dtype=np.int32)
    episode_actions = np.empty(
        (args.episode_steps, args.batch_size, env.action_size), dtype=np.float32
    )

    rng = np.random.default_rng(args.seed)
    maximums = {
        "qpos": 0.0,
        "qvel": 0.0,
        "forward_velocity_raw": 0.0,
        "com_delta_x": 0.0,
    }
    capacity_high_water = {
        "contact_count": 0,
        "constraint_count_max": 0,
        "overflow": False,
    }
    first_outlier: dict[str, Any] | None = None
    completed_steps = 0
    started = time.perf_counter()
    actions_array = np.zeros((args.batch_size, env.action_size), dtype=np.float32)

    for step_index in range(args.steps):
        if step_index % args.action_hold_steps == 0:
            actions_array = rng.uniform(
                -args.speed,
                args.speed,
                size=(args.batch_size, env.action_size),
            ).astype(np.float32)
        episode_actions[episode_action_counts, np.arange(args.batch_size)] = actions_array
        episode_action_counts += 1
        actions = jnp.asarray(
            actions_array,
            dtype=jnp.float32,
        )
        step_keys = _keys(args.seed + step_index + 1, args.batch_size)
        observation, state, reward, done, info = step(step_keys, state, actions)
        jax.block_until_ready(observation)
        completed_steps = step_index + 1

        qpos = np.asarray(jax.device_get(state.data.qpos))
        qvel = np.asarray(jax.device_get(state.data.qvel))
        raw_velocity = np.asarray(jax.device_get(info["forward_velocity_raw"]))
        com_delta = np.asarray(jax.device_get(info["com_delta_x"]))
        current = {
            "qpos": float(np.max(np.abs(qpos))),
            "qvel": float(np.max(np.abs(qvel))),
            "forward_velocity_raw": float(np.max(np.abs(raw_velocity))),
            "com_delta_x": float(np.max(np.abs(com_delta))),
        }
        for name, value in current.items():
            maximums[name] = max(maximums[name], value)

        diagnostics = env.buffer_diagnostics(state)
        for name in ("contact_count", "constraint_count_max"):
            value = diagnostics[name]
            if value is not None:
                capacity_high_water[name] = max(capacity_high_water[name], int(value))
        capacity_high_water["overflow"] = bool(
            capacity_high_water["overflow"] or diagnostics["overflow"]
        )

        reward_array = np.asarray(jax.device_get(reward))
        finite_by_environment = (
            np.all(np.isfinite(qpos), axis=1)
            & np.all(np.isfinite(qvel), axis=1)
            & np.isfinite(raw_velocity)
            & np.isfinite(reward_array)
        )
        qvel_max_by_environment = np.max(np.abs(qvel), axis=1)
        extreme_by_environment = (
            np.abs(raw_velocity) >= args.velocity_threshold
        ) | (qvel_max_by_environment >= args.velocity_threshold)
        offending_environments = np.flatnonzero(
            ~finite_by_environment | extreme_by_environment
        )
        if offending_environments.size:
            environment_index = int(offending_environments[0])
            finite = bool(finite_by_environment[environment_index])
            reproduction_path = Path(args.reproduction)
            reproduction_path.parent.mkdir(parents=True, exist_ok=True)
            action_count = int(episode_action_counts[environment_index])
            np.savez_compressed(
                reproduction_path,
                initial_qpos=episode_initial_state["qpos"][environment_index],
                initial_qvel=episode_initial_state["qvel"][environment_index],
                initial_act=episode_initial_state["act"][environment_index],
                initial_ctrl=episode_initial_state["ctrl"][environment_index],
                initial_time=episode_initial_state["time"][environment_index],
                initial_qacc_warmstart=episode_initial_state["qacc_warmstart"][environment_index],
                actions=episode_actions[:action_count, environment_index],
                failing_qpos=qpos[environment_index],
                failing_qvel=qvel[environment_index],
                warp_naconmax=np.asarray(diagnostics["contact_capacity"], dtype=np.int64),
                warp_njmax=np.asarray(diagnostics["constraint_capacity"], dtype=np.int64),
            )
            first_outlier = {
                "step": completed_steps,
                "environment_index": environment_index,
                "finite": finite,
                "forward_velocity_raw": float(raw_velocity[environment_index]),
                "reward": _scalar_at(reward, environment_index),
                "done": bool(np.asarray(jax.device_get(done))[environment_index]),
                "qpos_max_abs": float(np.max(np.abs(qpos[environment_index]))),
                "qvel_max_abs": float(np.max(np.abs(qvel[environment_index]))),
                "domain_randomization": {
                    field.name: _scalar_at(
                        getattr(state.domain_randomization, field.name), environment_index
                    )
                    for field in fields(state.domain_randomization)
                },
                "buffer_diagnostics": diagnostics,
                "reproduction": str(reproduction_path),
                "reproduction_action_count": action_count,
            }
            break

        done_array = np.asarray(jax.device_get(done), dtype=bool)
        if np.any(done_array):
            reset_keys = _keys(args.seed + args.steps + step_index + 1, args.batch_size)
            observation, state = reset_where(reset_keys, state, jnp.asarray(done_array))
            jax.block_until_ready(observation)
            reset_snapshot = _data_snapshot(state)
            for name, values in episode_initial_state.items():
                values[done_array] = reset_snapshot[name][done_array]
            episode_action_counts[done_array] = 0

    return {
        "topology": args.topology,
        "realistic": args.realistic,
        "requested_integrator": args.integrator,
        "effective_integrator": int(env.mujoco_model.model.opt.integrator),
        "graph_mode": args.graph_mode,
        "batch_size": args.batch_size,
        "steps_requested": args.steps,
        "steps_completed": completed_steps,
        "episode_steps": args.episode_steps,
        "nsubsteps": args.nsubsteps,
        "seed": args.seed,
        "action_hold_steps": args.action_hold_steps,
        "randomize_pose": args.randomize_pose,
        "capacities": capacities,
        "capacity_high_water": capacity_high_water,
        "maximums": maximums,
        "first_outlier": first_outlier,
        "elapsed_seconds": time.perf_counter() - started,
    }


def replay(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.replay) as saved:
        initial = {name: saved[f"initial_{name}"] for name in ("qpos", "qvel", "act", "ctrl")}
        initial["time"] = saved["initial_time"]
        initial["qacc_warmstart"] = saved["initial_qacc_warmstart"]
        actions = np.asarray(saved["actions"])
        warp_naconmax = int(saved["warp_naconmax"])
        warp_njmax = int(saved["warp_njmax"])

    if args.implementation == "native":
        return _replay_native(args, initial, actions)

    model_source = get_mujoco_spec(args.topology, realistic=args.realistic)
    model_source.option.integrator = {
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
    }[args.integrator]
    config = TrussEnvConfig(
        model_source,
        nsubsteps=args.nsubsteps,
        max_steps=max(len(actions) + 1, args.episode_steps),
        speed=args.speed,
        runtime_apply_control_noise=False,
    )
    env_kwargs: dict[str, Any] = {"mjx_impl": args.implementation}
    if args.implementation == "warp":
        env_kwargs.update(
            warp_graph_mode=args.graph_mode,
            warp_naconmax=warp_naconmax,
            warp_njmax=warp_njmax,
        )
    env = MjxNodeVelocityEnv(config, **env_kwargs)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    observation, state = reset(_keys(args.seed, 1))

    def forward_one(data: Any, domain: Any) -> Any:
        model = env._model_for_domain(domain)
        return mjx.forward(
            model,
            data.replace(**{name: jnp.asarray(value) for name, value in initial.items()}),
        )

    data = jax.jit(jax.vmap(forward_one))(state.data, state.domain_randomization)
    state = replace(state, data=data)
    observation = env._get_obs(state)
    jax.block_until_ready(observation)

    trace_qpos = []
    trace_qvel = []
    trace_raw_velocity = []
    trace_reward = []
    first_outlier = None
    for step_index, action in enumerate(actions, start=1):
        observation, state, reward, done, info = step(
            _keys(args.seed + step_index, 1),
            state,
            jnp.asarray(action[None, :], dtype=jnp.float32),
        )
        jax.block_until_ready(observation)
        qpos = np.asarray(jax.device_get(state.data.qpos[0]))
        qvel = np.asarray(jax.device_get(state.data.qvel[0]))
        raw_velocity = float(jax.device_get(info["forward_velocity_raw"][0]))
        reward_value = float(jax.device_get(reward[0]))
        trace_qpos.append(qpos)
        trace_qvel.append(qvel)
        trace_raw_velocity.append(raw_velocity)
        trace_reward.append(reward_value)
        finite = bool(
            np.all(np.isfinite(qpos))
            and np.all(np.isfinite(qvel))
            and np.isfinite(raw_velocity)
            and np.isfinite(reward_value)
        )
        if first_outlier is None and (not finite or abs(raw_velocity) >= args.velocity_threshold):
            first_outlier = {
                "step": step_index,
                "finite": finite,
                "forward_velocity_raw": raw_velocity,
                "reward": reward_value,
                "done": bool(jax.device_get(done[0])),
                "qpos_max_abs": float(np.max(np.abs(qpos))),
                "qvel_max_abs": float(np.max(np.abs(qvel))),
                "buffer_diagnostics": env.buffer_diagnostics(state),
            }

    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        qpos=np.asarray(trace_qpos),
        qvel=np.asarray(trace_qvel),
        forward_velocity_raw=np.asarray(trace_raw_velocity),
        reward=np.asarray(trace_reward),
    )
    return {
        "topology": args.topology,
        "realistic": args.realistic,
        "implementation": args.implementation,
        "requested_integrator": args.integrator,
        "effective_integrator": int(env.mujoco_model.model.opt.integrator),
        "action_count": len(actions),
        "first_outlier": first_outlier,
        "maximums": {
            "qpos": float(np.max(np.abs(trace_qpos))),
            "qvel": float(np.max(np.abs(trace_qvel))),
            "forward_velocity_raw": float(np.max(np.abs(trace_raw_velocity))),
            "reward": float(np.max(np.abs(trace_reward))),
        },
        "trace": str(trace_path),
    }


def _replay_native(
    args: argparse.Namespace,
    initial: dict[str, np.ndarray],
    actions: np.ndarray,
) -> dict[str, Any]:
    model_source = get_mujoco_spec(args.topology, realistic=args.realistic)
    model_source.option.integrator = {
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        "euler": mujoco.mjtIntegrator.mjINT_EULER,
    }[args.integrator]
    config = TrussEnvConfig(
        model_source,
        nsubsteps=args.nsubsteps,
        max_steps=max(len(actions) + 1, args.episode_steps),
        speed=args.speed,
        runtime_apply_control_noise=False,
    )
    env = MujocoNodeVelocityCommandEnv(config)
    env.reset(seed=args.seed)
    data = env.mj_model.data
    for name, value in initial.items():
        if name == "time":
            data.time = float(value)
        else:
            getattr(data, name)[:] = value
    env.node_velocity_controller.latest_node_commands[:] = 0.0
    mujoco.mj_forward(env.mj_model.model, data)
    effective_integrator = int(env.mj_model.model.opt.integrator)

    trace_qpos = []
    trace_qvel = []
    trace_raw_velocity = []
    trace_reward = []
    first_outlier = None
    try:
        for step_index, action in enumerate(actions, start=1):
            _, reward, terminated, truncated, info = env.step(action)
            qpos = data.qpos.copy()
            qvel = data.qvel.copy()
            raw_velocity = float(info["forward_velocity_raw"])
            reward_value = float(reward)
            trace_qpos.append(qpos)
            trace_qvel.append(qvel)
            trace_raw_velocity.append(raw_velocity)
            trace_reward.append(reward_value)
            finite = bool(
                np.all(np.isfinite(qpos))
                and np.all(np.isfinite(qvel))
                and np.isfinite(raw_velocity)
                and np.isfinite(reward_value)
            )
            if first_outlier is None and (
                not finite
                or abs(raw_velocity) >= args.velocity_threshold
                or np.max(np.abs(qvel)) >= args.velocity_threshold
            ):
                first_outlier = {
                    "step": step_index,
                    "finite": finite,
                    "forward_velocity_raw": raw_velocity,
                    "reward": reward_value,
                    "done": bool(terminated or truncated),
                    "qpos_max_abs": float(np.max(np.abs(qpos))),
                    "qvel_max_abs": float(np.max(np.abs(qvel))),
                    "contact_count": int(data.ncon),
                    "constraint_count": int(data.nefc),
                }
    finally:
        env.close()

    trace_path = Path(args.trace)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        trace_path,
        qpos=np.asarray(trace_qpos),
        qvel=np.asarray(trace_qvel),
        forward_velocity_raw=np.asarray(trace_raw_velocity),
        reward=np.asarray(trace_reward),
    )
    return {
        "topology": args.topology,
        "realistic": args.realistic,
        "implementation": "native",
        "requested_integrator": args.integrator,
        "effective_integrator": effective_integrator,
        "action_count": len(actions),
        "first_outlier": first_outlier,
        "maximums": {
            "qpos": float(np.max(np.abs(trace_qpos))),
            "qvel": float(np.max(np.abs(trace_qvel))),
            "forward_velocity_raw": float(np.max(np.abs(trace_raw_velocity))),
            "reward": float(np.max(np.abs(trace_reward))),
        },
        "trace": str(trace_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress an affected topology and capture the first extreme Warp transition."
    )
    parser.add_argument("--topology", default="henneberg_n6_2tube_1")
    parser.add_argument("--realistic", action="store_true")
    parser.add_argument("--implementation", choices=("jax", "warp", "native"), default="warp")
    parser.add_argument("--integrator", choices=("implicitfast", "euler"), default="implicitfast")
    parser.add_argument(
        "--graph-mode",
        choices=("warp", "warp_staged", "warp_staged_ex"),
        default="warp_staged",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--episode-steps", type=int, default=1_000)
    parser.add_argument("--nsubsteps", type=int, default=100)
    parser.add_argument("--speed", type=float, default=0.05)
    parser.add_argument("--action-hold-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-pose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--velocity-threshold", type=float, default=100.0)
    parser.add_argument("--capacity-scale", type=int, default=1)
    parser.add_argument("--warp-contact-capacity-per-env", type=int)
    parser.add_argument("--warp-constraint-capacity", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reproduction", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.steps <= 0
        or args.episode_steps <= 0
        or args.nsubsteps <= 0
        or args.action_hold_steps <= 0
    ):
        parser.error(
            "batch size, steps, episode steps, nsubsteps, and action hold steps must be positive"
        )
    if args.reproduction is None:
        args.reproduction = args.output.with_name(f"{args.output.stem}-reproduction.npz")
    if args.trace is None:
        args.trace = args.output.with_name(f"{args.output.stem}-trace.npz")
    payload = replay(args) if args.replay is not None else run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
