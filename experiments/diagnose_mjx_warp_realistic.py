from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import mujoco  # type: ignore[import-untyped]
import numpy as np
from experiments.benchmark_mjx_env import (
    DEFAULT_WARP_GRAPH_MODES,
    _capacity_defaults,
)
from mujoco import mjx

from mujoco_truss_gen import (
    MjxNodeVelocityEnv,
    MujocoNodeVelocityCommandEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)
from mujoco_truss_gen.mjx_env import WarpGraphMode

TRACE_FIELDS = (
    "qpos",
    "qvel",
    "observation",
    "reward",
    "ctrl",
    "contact_count",
    "constraint_count",
    "done",
)
FINITE_FIELDS = ("qpos", "qvel", "observation", "reward", "ctrl")


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": list(array.shape),
        "finite": bool(np.all(finite)),
        "nonfinite_count": int(array.size - np.count_nonzero(finite)),
        "min": float(np.min(finite_values)) if finite_values.size else None,
        "max": float(np.max(finite_values)) if finite_values.size else None,
        "max_abs": float(np.max(np.abs(finite_values))) if finite_values.size else None,
        "l2": float(np.linalg.norm(finite_values)) if finite_values.size else None,
    }


def _first_nonfinite(trace: dict[str, np.ndarray]) -> dict[str, Any] | None:
    step_indices = np.asarray(trace["step_index"])
    earliest: dict[str, Any] | None = None
    for field in FINITE_FIELDS:
        values = np.asarray(trace[field])
        indices = np.argwhere(~np.isfinite(values))
        if not indices.size:
            continue
        index = indices[0]
        candidate = {
            "step": int(step_indices[index[0]]),
            "field": field,
            "environment_index": int(index[1]) if len(index) > 1 else 0,
            "element_index": [int(value) for value in index[2:]],
            "value": float(values[tuple(index)]),
        }
        if earliest is None or (candidate["step"], field) < (
            earliest["step"],
            earliest["field"],
        ):
            earliest = candidate
    return earliest


def _first_divergence(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any] | None:
    reference_steps = np.asarray(reference["step_index"])
    candidate_steps = np.asarray(candidate["step_index"])
    step_count = min(len(reference_steps), len(candidate_steps))
    earliest: dict[str, Any] | None = None
    for field in TRACE_FIELDS:
        left = np.asarray(reference[field])[:step_count]
        right = np.asarray(candidate[field])[:step_count]
        if left.shape != right.shape:
            return {
                "step": int(min(reference_steps[0], candidate_steps[0])),
                "field": field,
                "reason": "shape_mismatch",
                "reference_shape": list(left.shape),
                "candidate_shape": list(right.shape),
            }
        if field in {"contact_count", "constraint_count", "done"}:
            close = left == right
        else:
            close = np.isclose(left, right, rtol=rtol, atol=atol, equal_nan=True)
        indices = np.argwhere(~close)
        if not indices.size:
            continue
        index = indices[0]
        left_value = left[tuple(index)]
        right_value = right[tuple(index)]
        candidate_result = {
            "step": int(candidate_steps[index[0]]),
            "field": field,
            "environment_index": int(index[1]) if len(index) > 1 else 0,
            "element_index": [int(value) for value in index[2:]],
            "reference_value": left_value.item(),
            "candidate_value": right_value.item(),
            "max_abs_difference_at_step": float(
                np.max(np.abs(left[index[0]].astype(float) - right[index[0]].astype(float)))
            ),
        }
        if earliest is None or (candidate_result["step"], field) < (
            earliest["step"],
            earliest["field"],
        ):
            earliest = candidate_result
    if earliest is None and len(reference_steps) != len(candidate_steps):
        return {
            "step": int(min(reference_steps[-1], candidate_steps[-1]) + 1),
            "field": "trace_length",
            "reason": "one_backend_stopped_early",
            "reference_steps": len(reference_steps),
            "candidate_steps": len(candidate_steps),
        }
    return earliest


def _classify_warp_run(
    summary: dict[str, Any],
    *,
    capacity_divergence: dict[str, Any] | None,
    jax_divergence: dict[str, Any] | None,
) -> str:
    first_nonfinite = summary.get("first_nonfinite")
    if first_nonfinite is not None:
        return (
            "capacity_exhaustion"
            if summary["capacity_high_water"]["overflow"]
            else "numerical_or_contact_instability_without_capacity_exhaustion"
        )
    if capacity_divergence is not None:
        return "capacity_dependent_kernel_behavior"
    if jax_divergence and jax_divergence.get("field") in {"reward", "done"}:
        return "task_semantic_mismatch"
    if jax_divergence is not None:
        return "finite_numerical_divergence"
    return "no_detected_divergence"


def _mjx_counts(env: MjxNodeVelocityEnv, state: Any) -> tuple[np.ndarray, np.ndarray]:
    if env.mjx_impl == "warp":
        return (
            np.asarray(jax.device_get(state.data._impl.nacon), dtype=np.int64),
            np.asarray(jax.device_get(state.data._impl.nefc), dtype=np.int64),
        )
    return (
        np.asarray(jax.device_get(state.data.ncon), dtype=np.int64),
        np.asarray(jax.device_get(state.data.nefc), dtype=np.int64),
    )


def _append_mjx_trace(
    storage: dict[str, list[np.ndarray]],
    env: MjxNodeVelocityEnv,
    obs: Any,
    state: Any,
    reward: Any,
    done: Any,
) -> None:
    contact_count, constraint_count = _mjx_counts(env, state)
    values = {
        "qpos": np.asarray(jax.device_get(state.data.qpos)),
        "qvel": np.asarray(jax.device_get(state.data.qvel)),
        "observation": np.asarray(jax.device_get(obs)),
        "reward": np.asarray(jax.device_get(reward)),
        "ctrl": np.asarray(jax.device_get(state.data.ctrl)),
        "contact_count": contact_count,
        "constraint_count": constraint_count,
        "done": np.asarray(jax.device_get(done)),
    }
    for field, value in values.items():
        storage[field].append(value)


def _run_mjx_backend(
    implementation: Literal["jax", "warp"],
    graph_mode: WarpGraphMode,
    capacity_scale: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model_source = get_mujoco_spec("octahedron", realistic=True)
    config = TrussEnvConfig(
        model_source,
        nsubsteps=args.nsubsteps,
        max_steps=max(args.steps + 1, 10_000),
        runtime_apply_control_noise=False,
    )
    warp_naconmax = None
    warp_njmax = None
    capacities = None
    if implementation == "warp":
        warp_naconmax, warp_njmax, capacities = _capacity_defaults(
            model_source,
            args.batch_size,
            contact_capacity_per_env=args.warp_contact_capacity_per_env,
            constraint_capacity=args.warp_constraint_capacity,
            scale=capacity_scale,
        )
    env = MjxNodeVelocityEnv(
        config,
        mjx_impl=implementation,
        warp_graph_mode=graph_mode,
        warp_naconmax=warp_naconmax,
        warp_njmax=warp_njmax,
    )
    keys = jax.random.split(jax.random.key(args.seed), args.batch_size)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    obs, state = reset(keys)
    jax.block_until_ready(obs)

    initial_path = Path(args._initial_state)
    if implementation == "jax" and not initial_path.exists():
        np.savez_compressed(
            initial_path,
            qpos=np.asarray(state.data.qpos),
            qvel=np.asarray(state.data.qvel),
            act=np.asarray(state.data.act),
            ctrl=np.asarray(state.data.ctrl),
        )
    else:
        with np.load(initial_path) as initial:
            qpos = jnp.asarray(initial["qpos"])
            qvel = jnp.asarray(initial["qvel"])
            act = jnp.asarray(initial["act"])
            ctrl = jnp.asarray(initial["ctrl"])

        def forward_one(data: Any, domain: Any, qp: Any, qv: Any, ac: Any, ct: Any) -> Any:
            model = env._model_for_domain(domain)
            return mjx.forward(
                model,
                data.replace(qpos=qp, qvel=qv, act=ac, ctrl=ct),
            )

        data = jax.jit(jax.vmap(forward_one))(
            state.data,
            state.domain_randomization,
            qpos,
            qvel,
            act,
            ctrl,
        )
        state = replace(state, data=data)
        obs = env._get_obs(state)
        jax.block_until_ready(obs)

    with np.load(args._actions) as action_file:
        actions = jnp.asarray(action_file["actions"])
    storage: dict[str, list[np.ndarray]] = {field: [] for field in TRACE_FIELDS}
    step_indices: list[int] = []
    step_records: list[dict[str, Any]] = []
    high_water = {
        "contact_count_total": 0,
        "contact_count_max_per_env": 0,
        "constraint_count_max": 0,
        "overflow": False,
    }

    reward = jnp.zeros((args.batch_size,), dtype=jnp.float32)
    done = jnp.zeros((args.batch_size,), dtype=jnp.bool_)
    for step_index in range(args.steps + 1):
        _append_mjx_trace(storage, env, obs, state, reward, done)
        diagnostics = env.buffer_diagnostics(state)
        contact_count_total = int(np.sum(storage["contact_count"][-1]))
        contact_count_max = int(np.max(storage["contact_count"][-1], initial=0))
        constraint_count = int(np.max(storage["constraint_count"][-1], initial=0))
        high_water["contact_count_total"] = max(
            high_water["contact_count_total"], contact_count_total
        )
        high_water["contact_count_max_per_env"] = max(
            high_water["contact_count_max_per_env"], contact_count_max
        )
        high_water["constraint_count_max"] = max(
            high_water["constraint_count_max"], constraint_count
        )
        high_water["overflow"] = bool(high_water["overflow"] or diagnostics["overflow"])
        step_records.append(
            {
                "step": step_index,
                "fields": {field: _array_summary(storage[field][-1]) for field in FINITE_FIELDS},
                "contact_count_total": contact_count_total,
                "contact_count_max_per_env": contact_count_max,
                "constraint_count_max": constraint_count,
                "capacity_overflow": bool(diagnostics["overflow"]),
                "done_count": int(np.count_nonzero(storage["done"][-1])),
            }
        )
        step_indices.append(step_index)
        if any(not step_records[-1]["fields"][field]["finite"] for field in FINITE_FIELDS):
            break
        if step_index < args.steps:
            obs, state, reward, done, _ = step(
                keys,
                state,
                actions[step_index],
            )
            jax.block_until_ready(obs)

    trace = {field: np.stack(values) for field, values in storage.items()}
    trace["step_index"] = np.asarray(step_indices, dtype=np.int64)
    summary = {
        "backend": implementation,
        "graph_mode": graph_mode if implementation == "warp" else None,
        "capacity_scale": capacity_scale if implementation == "warp" else None,
        "capacities": capacities,
        "steps_recorded": len(step_indices),
        "first_nonfinite": _first_nonfinite(trace),
        "capacity_high_water": high_water,
        "step_records": step_records,
    }
    return summary, trace


def _run_native(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = TrussEnvConfig(
        get_mujoco_spec("octahedron", realistic=True),
        nsubsteps=args.nsubsteps,
        max_steps=max(args.steps + 1, 10_000),
        runtime_apply_control_noise=False,
    )
    with np.load(args._initial_state) as initial:
        qpos = initial["qpos"]
        qvel = initial["qvel"]
        act = initial["act"]
        ctrl = initial["ctrl"]
    with np.load(args._actions) as action_file:
        actions = action_file["actions"]

    envs = [MujocoNodeVelocityCommandEnv(config) for _ in range(args.batch_size)]
    observations = []
    for index, env in enumerate(envs):
        env.reset(seed=args.seed + index)
        env.mj_model.data.qpos[:] = qpos[index]
        env.mj_model.data.qvel[:] = qvel[index]
        env.mj_model.data.act[:] = act[index]
        env.mj_model.data.ctrl[:] = ctrl[index]
        env.node_velocity_controller.latest_node_commands[:] = 0.0
        mujoco.mj_forward(env.mj_model.model, env.mj_model.data)
        observations.append(env._get_obs())

    storage: dict[str, list[np.ndarray]] = {field: [] for field in TRACE_FIELDS}
    step_indices: list[int] = []
    step_records: list[dict[str, Any]] = []
    reward = np.zeros(args.batch_size, dtype=np.float32)
    done = np.zeros(args.batch_size, dtype=bool)
    for step_index in range(args.steps + 1):
        values = {
            "qpos": np.stack([env.mj_model.data.qpos.copy() for env in envs]),
            "qvel": np.stack([env.mj_model.data.qvel.copy() for env in envs]),
            "observation": np.stack(observations),
            "reward": reward.copy(),
            "ctrl": np.stack([env.mj_model.data.ctrl.copy() for env in envs]),
            "contact_count": np.asarray([env.mj_model.data.ncon for env in envs]),
            "constraint_count": np.asarray([env.mj_model.data.nefc for env in envs]),
            "done": done.copy(),
        }
        for field, value in values.items():
            storage[field].append(np.asarray(value))
        step_records.append(
            {
                "step": step_index,
                "fields": {field: _array_summary(values[field]) for field in FINITE_FIELDS},
                "contact_count_total": int(np.sum(values["contact_count"])),
                "contact_count_max_per_env": int(np.max(values["contact_count"], initial=0)),
                "constraint_count_max": int(np.max(values["constraint_count"], initial=0)),
                "done_count": int(np.count_nonzero(done)),
            }
        )
        step_indices.append(step_index)
        if any(not step_records[-1]["fields"][field]["finite"] for field in FINITE_FIELDS):
            break
        if step_index < args.steps:
            next_observations = []
            next_rewards = []
            next_done = []
            for env_index, env in enumerate(envs):
                obs, item_reward, terminated, truncated, _ = env.step(
                    actions[step_index, env_index]
                )
                next_observations.append(obs)
                next_rewards.append(item_reward)
                next_done.append(terminated or truncated)
            observations = next_observations
            reward = np.asarray(next_rewards)
            done = np.asarray(next_done)

    trace = {field: np.stack(values) for field, values in storage.items()}
    trace["step_index"] = np.asarray(step_indices, dtype=np.int64)
    summary = {
        "backend": "native",
        "graph_mode": None,
        "capacity_scale": None,
        "steps_recorded": len(step_indices),
        "first_nonfinite": _first_nonfinite(trace),
        "capacity_high_water": {
            "contact_count_total": int(np.max(np.sum(trace["contact_count"], axis=1), initial=0)),
            "contact_count_max_per_env": int(np.max(trace["contact_count"], initial=0)),
            "constraint_count_max": int(np.max(trace["constraint_count"], initial=0)),
            "overflow": False,
        },
        "step_records": step_records,
    }
    return summary, trace


def _run_child(args: argparse.Namespace) -> None:
    if args._backend == "native":
        summary, trace = _run_native(args)
    else:
        summary, trace = _run_mjx_backend(
            args._backend,
            cast(WarpGraphMode, args._graph_mode),
            args._capacity_scale,
            args,
        )
    np.savez_compressed(args._trace, **trace)  # type: ignore[arg-type]
    print(json.dumps(summary, sort_keys=True))


def _subprocess_case(
    script: Path,
    args: argparse.Namespace,
    *,
    backend: str,
    graph_mode: str,
    capacity_scale: int,
    initial_state: Path,
    actions: Path,
    trace: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(script),
        "--batch-size",
        str(args.batch_size),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--nsubsteps",
        str(args.nsubsteps),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--_backend",
        backend,
        "--_graph-mode",
        graph_mode,
        "--_capacity-scale",
        str(capacity_scale),
        "--_initial-state",
        str(initial_state),
        "--_actions",
        str(actions),
        "--_trace",
        str(trace),
    ]
    if args.warp_contact_capacity_per_env is not None:
        command.extend(["--warp-contact-capacity-per-env", str(args.warp_contact_capacity_per_env)])
    if args.warp_constraint_capacity is not None:
        command.extend(["--warp-constraint-capacity", str(args.warp_constraint_capacity)])
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return {
            "status": "error",
            "backend": backend,
            "graph_mode": graph_mode,
            "capacity_scale": capacity_scale,
            "error": completed.stderr.strip() or completed.stdout.strip(),
        }
    return {"status": "ok", **json.loads(completed.stdout.splitlines()[-1])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace the realistic octahedron across native MuJoCo, MJX-JAX, and MJX-Warp."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--nsubsteps", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--warp-contact-capacity-per-env", type=int)
    parser.add_argument("--warp-constraint-capacity", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/mjx_warp_realistic_diagnostic.json"),
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=Path("benchmark_results/mjx_warp_realistic_diagnostic_traces.npz"),
    )
    parser.add_argument("--_backend", choices=("native", "jax", "warp"))
    parser.add_argument("--_graph-mode", default="warp")
    parser.add_argument("--_capacity-scale", type=int, default=1)
    parser.add_argument("--_initial-state")
    parser.add_argument("--_actions")
    parser.add_argument("--_trace")
    args = parser.parse_args()

    if args._backend:
        _run_child(args)
        return
    if args.batch_size <= 0 or args.steps <= 0:
        parser.error("--batch-size and --steps must be positive")

    model_source = get_mujoco_spec("octahedron", realistic=True)
    action_shape = MujocoNodeVelocityCommandEnv(model_source).action_space.shape
    if action_shape is None:
        raise RuntimeError("Realistic octahedron action space has no fixed shape.")
    action_size = action_shape[0]
    rng = np.random.default_rng(args.seed)
    actions_array = rng.uniform(
        low=-0.01,
        high=0.01,
        size=(args.steps, args.batch_size, action_size),
    ).astype(np.float32)

    script = Path(__file__).resolve()
    summaries: dict[str, dict[str, Any]] = {}
    traces: dict[str, dict[str, np.ndarray]] = {}
    with tempfile.TemporaryDirectory(prefix="mjx-warp-diagnostic-") as directory:
        work = Path(directory)
        initial_state = work / "initial_state.npz"
        actions = work / "actions.npz"
        np.savez_compressed(actions, actions=actions_array)
        cases = [("jax", "warp", 1), ("native", "warp", 1)] + [
            ("warp", graph_mode, capacity_scale)
            for graph_mode in DEFAULT_WARP_GRAPH_MODES
            for capacity_scale in (1, 2)
        ]
        for backend, graph_mode, capacity_scale in cases:
            name = backend if backend != "warp" else f"{graph_mode}_capacity_x{capacity_scale}"
            trace_path = work / f"{name}.npz"
            print(f"[{name}] tracing realistic octahedron", flush=True)
            result = _subprocess_case(
                script,
                args,
                backend=backend,
                graph_mode=graph_mode,
                capacity_scale=capacity_scale,
                initial_state=initial_state,
                actions=actions,
                trace=trace_path,
            )
            summaries[name] = result
            if result["status"] == "ok":
                with np.load(trace_path) as loaded:
                    traces[name] = {key: loaded[key] for key in loaded.files}

    comparisons: dict[str, Any] = {}
    jax_trace = traces.get("jax")
    native_trace = traces.get("native")
    if jax_trace is not None and native_trace is not None:
        comparisons["native_vs_jax"] = _first_divergence(
            jax_trace, native_trace, rtol=args.rtol, atol=args.atol
        )
    classifications: dict[str, str] = {}
    for graph_mode in DEFAULT_WARP_GRAPH_MODES:
        base_name = f"{graph_mode}_capacity_x1"
        doubled_name = f"{graph_mode}_capacity_x2"
        base_trace = traces.get(base_name)
        doubled_trace = traces.get(doubled_name)
        capacity_divergence = None
        if base_trace is not None and doubled_trace is not None:
            capacity_divergence = _first_divergence(
                base_trace, doubled_trace, rtol=args.rtol, atol=args.atol
            )
        comparisons[f"{graph_mode}_capacity_x1_vs_x2"] = capacity_divergence
        for name in (base_name, doubled_name):
            if name not in traces or jax_trace is None:
                continue
            jax_divergence = _first_divergence(
                jax_trace, traces[name], rtol=args.rtol, atol=args.atol
            )
            comparisons[f"jax_vs_{name}"] = jax_divergence
            classifications[name] = _classify_warp_run(
                summaries[name],
                capacity_divergence=capacity_divergence,
                jax_divergence=jax_divergence,
            )

    combined_trace = {
        f"{name}__{field}": value
        for name, trace in traces.items()
        for field, value in trace.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.traces.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.traces, **combined_trace)  # type: ignore[arg-type]
    payload = {
        "schema_version": 1,
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "python": platform.python_version(),
            "jax_backend": jax.default_backend(),
            "jax_device": jax.devices()[0].device_kind,
            "model": "octahedron:realistic",
            "batch_size": args.batch_size,
            "steps": args.steps,
            "seed": args.seed,
            "nsubsteps": args.nsubsteps,
            "rtol": args.rtol,
            "atol": args.atol,
            "traces": str(args.traces),
        },
        "summaries": summaries,
        "comparisons": comparisons,
        "classifications": classifications,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.traces}")
    failures = [name for name, result in summaries.items() if result["status"] != "ok"]
    if failures:
        raise SystemExit(f"Diagnostic backend failures: {', '.join(failures)}")


if __name__ == "__main__":
    main()
