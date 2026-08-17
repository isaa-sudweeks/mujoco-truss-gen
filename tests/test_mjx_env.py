from __future__ import annotations

import importlib.util
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from mujoco_truss_gen import (
    PRESETS,
    DomainRandomizationConfig,
    MjxEnvState,
    MjxNodeVelocityEnv,
    MujocoNodeVelocityCommandEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)
from mujoco_truss_gen.mjx_env import (
    _DEFAULT_WARP_NACONMAX,
    _configure_integrator_for_backend,
    _copy_model_source_for_env,
    _warp_contact_capacity,
)


def _is_indexed_henneberg_variant(preset_name: str) -> bool:
    return (
        preset_name.startswith("henneberg_") and preset_name.rsplit("_", maxsplit=1)[-1].isdigit()
    )


CANONICAL_PRESET_NAMES = tuple(name for name in PRESETS if not _is_indexed_henneberg_variant(name))
CUDA_WARP_AVAILABLE = (
    jax.devices()[0].platform in {"cuda", "gpu"} and importlib.util.find_spec("warp") is not None
)


@dataclass(frozen=True)
class CompiledEnv:
    env: MjxNodeVelocityEnv
    reset: object
    step: object
    reset_where: object


@pytest.fixture(scope="module")
def compiled_env() -> CompiledEnv:
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            max_steps=2,
            nsubsteps=1,
            speed=0.01,
        )
    )
    return CompiledEnv(
        env=env,
        reset=jax.jit(env.reset),
        step=jax.jit(env.step),
        reset_where=jax.jit(env.reset_where),
    )


def _keys(seed: int, batch_size: int = 2) -> jax.Array:
    return jax.random.split(jax.random.key(seed), batch_size)


def _record_allclose_failure(
    failures: list[str],
    label: str,
    actual: object,
    expected: object,
    *,
    rtol: float,
    atol: float,
) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    if np.allclose(actual_array, expected_array, rtol=rtol, atol=atol, equal_nan=True):
        return

    absolute_error = np.abs(actual_array - expected_array)
    allowed_error = atol + rtol * np.abs(expected_array)
    failures.append(
        f"{label}: max_abs_error={float(np.nanmax(absolute_error)):.6g}, "
        f"max_tolerance_ratio={float(np.nanmax(absolute_error / allowed_error)):.6g} "
        f"(rtol={rtol:g}, atol={atol:g})"
    )


@pytest.mark.parametrize("preset_name", CANONICAL_PRESET_NAMES)
def test_mjx_node_velocity_env_constructs_for_canonical_abstract_presets(
    preset_name: str,
) -> None:
    env = MjxNodeVelocityEnv(get_mujoco_spec(preset_name, realistic=False))

    assert env.action_size > 0
    assert env.observation_size == 7 * env.action_size
    assert env.action_low.shape == (env.action_size,)
    assert env.action_high.shape == (env.action_size,)


@pytest.mark.parametrize(
    "preset_name",
    (
        "tetrahedron",
        "octahedron",
        "icosahedron",
        "usevitch_210272254_p1",
        "henneberg_n6_2tube_1",
        "henneberg_n8_2tube_187",
    ),
)
def test_mjx_edge_gram_rigidity_matches_native_initial_value(preset_name: str) -> None:
    env = MjxNodeVelocityEnv(get_mujoco_spec(preset_name, realistic=False))

    assert float(env._critical_eig(env._data_template)) == pytest.approx(1.0, rel=2e-4)


def test_mjx_node_velocity_env_constructs_with_sparse_mass_matrix() -> None:
    env = MjxNodeVelocityEnv(get_mujoco_spec("octahedron", realistic=True))

    assert env.mujoco_model.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert env._qM0.shape == (env.mjx_model.nv, env.mjx_model.nv)
    assert env._nominal_qm_physical.shape == (env.mjx_model.nv, env.mjx_model.nv)


@pytest.mark.parametrize(
    ("implementation", "uses_realistic_connectors", "requested", "expected"),
    (
        (
            "jax",
            True,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        ),
        (
            "warp",
            False,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        ),
        (
            "warp",
            True,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            mujoco.mjtIntegrator.mjINT_EULER,
        ),
        ("warp", True, mujoco.mjtIntegrator.mjINT_RK4, mujoco.mjtIntegrator.mjINT_RK4),
    ),
)
def test_mjx_backend_integrator_compatibility(
    implementation: str,
    uses_realistic_connectors: bool,
    requested: mujoco.mjtIntegrator,
    expected: mujoco.mjtIntegrator,
) -> None:
    model = get_mujoco_spec("tetrahedron", realistic=False).compile()
    model.opt.integrator = requested

    _configure_integrator_for_backend(
        model,
        implementation,
        uses_realistic_connectors=uses_realistic_connectors,
    )

    assert model.opt.integrator == expected


def test_mjx_backend_integrator_override_does_not_mutate_caller_model() -> None:
    caller_model = get_mujoco_spec("octahedron", realistic=True).compile()
    caller_model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    internal_model = _copy_model_source_for_env(caller_model)
    assert isinstance(internal_model, mujoco.MjModel)
    _configure_integrator_for_backend(
        internal_model,
        "warp",
        uses_realistic_connectors=True,
    )

    assert internal_model is not caller_model
    assert internal_model.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert caller_model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


def test_mjx_implementation_defaults_to_jax_and_reports_no_warp_buffers() -> None:
    env = MjxNodeVelocityEnv(get_mujoco_spec("tetrahedron", realistic=False))
    _, state = jax.jit(env.reset)(_keys(0, batch_size=1))

    assert env.mjx_impl == "jax"
    assert env.warp_graph_mode == "warp"
    assert env.warp_naconmax is None
    assert env.warp_njmax is None
    assert env.buffer_diagnostics(state) == {
        "implementation": "jax",
        "contact_count": None,
        "contact_capacity": None,
        "constraint_count_max": None,
        "constraint_capacity": None,
        "contact_capacity_reached": False,
        "constraint_capacity_reached": False,
        "overflow": False,
    }


@pytest.mark.parametrize(
    ("implementation", "configured", "expected"),
    (
        ("jax", None, None),
        ("warp", None, _DEFAULT_WARP_NACONMAX),
        ("warp", 12_345, 12_345),
    ),
)
def test_warp_contact_capacity_avoids_single_world_default(
    implementation: str,
    configured: int | None,
    expected: int | None,
) -> None:
    assert _warp_contact_capacity(implementation, configured) == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"mjx_impl": "other"}, "mjx_impl must be one of"),
        ({"warp_graph_mode": "other"}, "warp_graph_mode must be one of"),
        ({"warp_naconmax": 0}, "warp_naconmax must be a positive integer"),
        ({"warp_naconmax": 1.0}, "warp_naconmax must be a positive integer"),
        ({"warp_njmax": True}, "warp_njmax must be a positive integer"),
        (
            {"mjx_impl": "jax", "warp_naconmax": 16},
            "may only be customized when mjx_impl='warp'",
        ),
    ),
)
def test_mjx_implementation_options_are_validated(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MjxNodeVelocityEnv(get_mujoco_spec("tetrahedron", realistic=False), **kwargs)


def test_warp_backend_fails_clearly_without_active_cuda_device() -> None:
    if jax.devices()[0].platform in {"cuda", "gpu"}:
        pytest.skip("This error-path test applies only when CUDA is not active.")

    with pytest.raises(RuntimeError, match="requires the active JAX default device"):
        MjxNodeVelocityEnv(
            get_mujoco_spec("tetrahedron", realistic=False),
            mjx_impl="warp",
        )


def test_mjx_reset_is_batched_deterministic_and_initializes_actuators(
    compiled_env: CompiledEnv,
) -> None:
    keys = _keys(1)
    obs_a, state_a = compiled_env.reset(keys)
    obs_b, state_b = compiled_env.reset(keys)

    np.testing.assert_array_equal(state_a.data.qpos, state_b.data.qpos)
    np.testing.assert_array_equal(state_a.data.qvel, state_b.data.qvel)
    np.testing.assert_array_equal(obs_a, obs_b)
    assert not np.array_equal(np.asarray(state_a.data.qpos[0]), np.asarray(state_a.data.qpos[1]))
    assert obs_a.shape == (2, compiled_env.env.observation_size)
    assert obs_a.dtype == jnp.float32
    assert state_a.data.qpos.shape[0] == 2
    assert state_a.step_count.dtype == jnp.int32
    np.testing.assert_array_equal(state_a.step_count, np.zeros(2, dtype=np.int32))
    np.testing.assert_allclose(
        state_a.data.act[:, compiled_env.env._reset_act_adrs],
        state_a.data.ten_length[:, compiled_env.env._reset_tendon_ids],
        rtol=1e-6,
        atol=1e-6,
    )


def test_mjx_jitted_step_matches_node_action_and_episode_semantics(
    compiled_env: CompiledEnv,
) -> None:
    env = compiled_env.env
    keys = _keys(2)
    _, state = compiled_env.reset(keys)
    actions = jnp.stack(
        (
            jnp.linspace(-0.02, 0.02, env.action_size),
            jnp.linspace(0.02, -0.02, env.action_size),
        )
    )

    obs, state, reward, done, info = compiled_env.step(_keys(3), state, actions)

    expected_action = np.clip(np.asarray(actions[0]), -0.01, 0.01)
    expected_edge_commands = env._controller.clipped_edge_commands(
        env.mujoco_model.model, expected_action
    )
    expected_node_commands = expected_action.copy()
    expected_node_commands[env._controller.passive_node_mask] = 0.0
    np.testing.assert_allclose(
        state.data.ctrl[0, env._controller.actuator_ids],
        expected_edge_commands,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(state.node_commands[0], expected_node_commands)
    assert obs.shape == (2, env.observation_size)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert not np.any(np.asarray(done))
    np.testing.assert_array_equal(state.step_count, np.ones(2, dtype=np.int32))

    expected_info_keys = {
        "alive",
        "collapse_penalty",
        "com_delta_x",
        "critical_eig",
        "critical_eig_raw",
        "energy",
        "forward",
        "forward_velocity",
        "forward_velocity_normalized",
        "forward_velocity_normalized_raw",
        "forward_velocity_raw",
        "minimum_substep_critical_eig_raw",
        "rigidity",
        "slip",
        "substeps_executed",
        "terminated",
        "terminated_by_collapse",
        "terminated_during_substeps",
        "truncated",
    }
    assert set(info) == expected_info_keys
    assert all(value.shape == (2,) for value in info.values())
    np.testing.assert_array_equal(info["substeps_executed"], np.ones(2, dtype=np.int32))
    assert not np.any(np.asarray(info["terminated_during_substeps"]))

    _, state, _, done, info = compiled_env.step(_keys(4), state, actions)
    np.testing.assert_array_equal(state.step_count, np.full(2, 2, dtype=np.int32))
    assert np.all(np.asarray(done))
    assert not np.any(np.asarray(info["terminated"]))
    assert np.all(np.asarray(info["truncated"]))


def test_mjx_stops_after_first_collapsed_physics_substep() -> None:
    nsubsteps = 5
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            max_steps=3,
            nsubsteps=nsubsteps,
            speed=0.01,
            critical_eig_threshold=2.0,
        )
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    keys = _keys(30, batch_size=2)
    _, state = reset(keys)
    initial_time = np.asarray(state.data.time)
    timestep = float(env.mujoco_model.model.opt.timestep)
    actions = jnp.zeros((2, env.action_size), dtype=jnp.float32)

    _, next_state, _, done, info = step(keys, state, actions)

    assert np.all(np.asarray(done))
    assert np.all(np.asarray(info["terminated_by_collapse"]))
    assert np.all(np.asarray(info["terminated_during_substeps"]))
    np.testing.assert_array_equal(info["substeps_executed"], np.ones(2, dtype=np.int32))
    np.testing.assert_allclose(next_state.data.time, initial_time + timestep)
    np.testing.assert_allclose(
        info["minimum_substep_critical_eig_raw"],
        info["critical_eig_raw"],
    )


def test_mjx_substep_guard_freezes_only_collapsed_batch_elements() -> None:
    nsubsteps = 5
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            max_steps=3,
            nsubsteps=nsubsteps,
            speed=0.01,
            critical_eig_threshold=0.5,
        )
    )
    first_actuator_id = int(env._actuator_ids[0])

    def command_dependent_critical_eig(data) -> jax.Array:
        commanded = jnp.abs(data.ctrl[first_actuator_id]) > 1e-8
        return jnp.where(commanded, 0.0, 1.0)

    env._critical_eig = command_dependent_critical_eig
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    keys = _keys(31, batch_size=2)
    _, state = reset(keys)
    initial_time = np.asarray(state.data.time)
    timestep = float(env.mujoco_model.model.opt.timestep)
    action_direction = np.sign(np.asarray(env._incidence_matrix[0])) * float(env.config.speed)
    actions = jnp.asarray(np.stack((action_direction, np.zeros(env.action_size))))

    _, next_state, _, done, info = step(keys, state, actions)

    np.testing.assert_array_equal(done, np.array([True, False]))
    np.testing.assert_array_equal(info["substeps_executed"], np.array([1, nsubsteps]))
    np.testing.assert_array_equal(info["terminated_during_substeps"], np.array([True, False]))
    np.testing.assert_allclose(
        next_state.data.time,
        initial_time + timestep * np.array([1, nsubsteps]),
    )


def test_mjx_observation_and_reward_match_cpu_environment(
    compiled_env: CompiledEnv,
) -> None:
    env = compiled_env.env
    keys = _keys(5, batch_size=1)
    obs, state = compiled_env.reset(keys)
    cpu_env = MujocoNodeVelocityCommandEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            max_steps=2,
            nsubsteps=1,
            speed=0.01,
        )
    )
    try:
        cpu_env.mj_model.data = mjx.get_data(env.mujoco_model.model, state.data)[0]
        np.testing.assert_allclose(obs[0], cpu_env._get_obs(), rtol=1e-5, atol=1e-6)
        previous_com = cpu_env._center_of_mass().copy()

        action = jnp.linspace(-0.01, 0.01, env.action_size)[None, :]
        obs, state, reward, _, info = compiled_env.step(keys, state, action)
        cpu_env.mj_model.data = mjx.get_data(env.mujoco_model.model, state.data)[0]
        cpu_env.node_velocity_controller.latest_node_commands = np.asarray(state.node_commands[0])
        expected_obs = cpu_env._get_obs()
        expected_reward, expected_info, expected_terminated = cpu_env._compute_reward(
            np.asarray(action[0]), previous_com
        )

        np.testing.assert_allclose(obs[0], expected_obs, rtol=1e-5, atol=1e-6)
        assert float(reward[0]) == pytest.approx(expected_reward, rel=1e-5, abs=1e-6)
        assert bool(info["terminated"][0]) is expected_terminated
        for key, expected_value in expected_info.items():
            if isinstance(expected_value, bool):
                assert bool(info[key][0]) is expected_value
            elif np.isnan(expected_value):
                assert np.isnan(np.asarray(info[key][0]))
            else:
                assert float(info[key][0]) == pytest.approx(expected_value, rel=1e-5, abs=1e-6)
    finally:
        cpu_env.close()


@pytest.mark.parametrize("preset_name", ("tetrahedron", "octahedron"))
def test_native_mujoco_and_mjx_contact_free_euler_rollouts_match(
    preset_name: str,
) -> None:
    action_count = 4
    config = TrussEnvConfig(
        get_mujoco_spec(preset_name, realistic=False),
        max_steps=action_count,
        nsubsteps=2,
        speed=0.01,
    )
    native_env = MujocoNodeVelocityCommandEnv(config)
    mjx_env = MjxNodeVelocityEnv(config)

    try:
        native_env.reset(seed=2026)
        native_model = native_env.mj_model.model
        native_data = native_env.mj_model.data

        # Run MJX from the exact native model and state rather than performing a
        # separate MJX reset, which could introduce different random perturbations.
        assert native_env.mj_model.xml == mjx_env.mujoco_model.xml
        assert native_env.nsubsteps == mjx_env.config.nsubsteps
        assert native_model.opt.timestep == pytest.approx(
            mjx_env.mujoco_model.model.opt.timestep,
            rel=0.0,
            abs=0.0,
        )
        contact_disable_bit = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        native_model.opt.disableflags |= contact_disable_bit
        # MJX and native MuJoCo use materially different implicitfast paths. Use
        # their common explicit integrator to isolate contact-free dynamics parity.
        native_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        mujoco.mj_forward(native_model, native_data)
        assert len(native_data.contact) == 0
        mjx_env.mjx_model = mjx.put_model(native_model)
        assert int(mjx_env.mjx_model.opt.disableflags) & contact_disable_bit
        assert int(mjx_env.mjx_model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_EULER)
        mjx_data = mjx.put_data(native_model, native_data)
        state = MjxEnvState(
            data=jax.tree.map(lambda value: value[None, ...], mjx_data),
            step_count=jnp.zeros((1,), dtype=jnp.int32),
            node_commands=jnp.asarray(
                native_env.node_velocity_controller.latest_node_commands[None, :]
            ),
            domain_randomization=jax.tree.map(
                lambda value: value[None], mjx_env._nominal_domain_randomization_state()
            ),
        )

        np.testing.assert_allclose(state.data.qpos[0], native_data.qpos, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(state.data.qvel[0], native_data.qvel, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            mjx_env._get_obs(state)[0],
            native_env._get_obs(),
            rtol=1e-5,
            atol=1e-6,
        )

        action_axis = np.linspace(-1.0, 1.0, mjx_env.action_size, dtype=np.float32)
        actions = np.stack(
            (
                0.75 * config.speed * action_axis,
                -0.5 * config.speed * action_axis,
                0.25 * config.speed * np.sin(np.pi * action_axis),
                np.zeros_like(action_axis),
            )
        ).astype(np.float32)
        mjx_step = jax.jit(mjx_env.step)
        failures: list[str] = []

        for step_index, action in enumerate(actions, start=1):
            native_obs, native_reward, native_terminated, native_truncated, _ = native_env.step(
                action
            )
            mjx_obs, state, mjx_reward, mjx_done, mjx_info = mjx_step(
                _keys(step_index, batch_size=1),
                state,
                jnp.asarray(action[None, :]),
            )
            assert len(native_data.contact) == 0
            context = f"{preset_name=}, {step_index=}"

            _record_allclose_failure(
                failures,
                f"qpos ({context})",
                state.data.qpos[0],
                native_data.qpos,
                rtol=2e-3,
                atol=2e-5,
            )
            _record_allclose_failure(
                failures,
                f"qvel ({context})",
                state.data.qvel[0],
                native_data.qvel,
                rtol=2e-2,
                atol=2e-4,
            )
            _record_allclose_failure(
                failures,
                f"observation ({context})",
                mjx_obs[0],
                native_obs,
                rtol=2e-2,
                atol=2e-4,
            )
            _record_allclose_failure(
                failures,
                f"reward ({context})",
                mjx_reward[0],
                native_reward,
                rtol=2e-2,
                atol=2e-4,
            )
            mjx_terminated = bool(mjx_info["terminated"][0])
            mjx_truncated = bool(mjx_info["truncated"][0])
            if mjx_terminated is not native_terminated:
                failures.append(
                    f"termination ({context}): MJX={mjx_terminated}, native={native_terminated}"
                )
            if mjx_truncated is not native_truncated:
                failures.append(
                    f"truncation ({context}): MJX={mjx_truncated}, native={native_truncated}"
                )
            if bool(mjx_done[0]) is not (native_terminated or native_truncated):
                failures.append(
                    f"done ({context}): MJX={bool(mjx_done[0])}, "
                    f"native={native_terminated or native_truncated}"
                )
            _record_allclose_failure(
                failures,
                f"node controller output ({context})",
                state.node_commands[0],
                native_env.node_velocity_controller.latest_node_commands,
                rtol=1e-6,
                atol=1e-7,
            )
            _record_allclose_failure(
                failures,
                f"actuator controller output ({context})",
                state.data.ctrl[0],
                native_data.ctrl,
                rtol=1e-4,
                atol=1e-5,
            )

        assert not failures, "Native MuJoCo/MJX parity failures:\n" + "\n".join(failures)
    finally:
        native_env.close()


def test_mjx_reset_where_only_resets_masked_elements(
    compiled_env: CompiledEnv,
) -> None:
    env = compiled_env.env
    _, state = compiled_env.reset(_keys(6))
    actions = jnp.full((2, env.action_size), 0.005, dtype=jnp.float32)
    _, stepped_state, _, _, _ = compiled_env.step(_keys(7), state, actions)

    obs, merged_state = compiled_env.reset_where(_keys(8), stepped_state, jnp.array([True, False]))

    assert obs.shape == (2, env.observation_size)
    np.testing.assert_array_equal(merged_state.step_count, np.array([0, 1]))
    np.testing.assert_array_equal(
        merged_state.node_commands[0], np.zeros(env.action_size, dtype=np.float32)
    )
    np.testing.assert_array_equal(merged_state.node_commands[1], stepped_state.node_commands[1])
    np.testing.assert_array_equal(merged_state.data.qpos[1], stepped_state.data.qpos[1])
    assert not np.array_equal(
        np.asarray(merged_state.data.qpos[0]), np.asarray(stepped_state.data.qpos[0])
    )


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="CUDA Warp dependencies are unavailable")
@pytest.mark.parametrize("graph_mode", ("warp", "warp_staged", "warp_staged_ex"))
def test_warp_graph_modes_step_and_selectively_reset(graph_mode: str) -> None:
    batch_size = 2
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            max_steps=4,
            nsubsteps=2,
            critical_eig_threshold=2.0,
        ),
        mjx_impl="warp",
        warp_graph_mode=graph_mode,
        warp_naconmax=128 * batch_size,
        warp_njmax=256,
    )
    assert env.mujoco_model.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert int(env.mjx_model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    reset_where = jax.jit(env.reset_where)

    obs, state = reset(_keys(100, batch_size=batch_size))
    actions = jnp.linspace(-0.01, 0.01, env.action_size)[None, :].repeat(batch_size, axis=0)
    next_obs, stepped_state, reward, done, info = step(
        _keys(101, batch_size=batch_size), state, actions
    )
    reset_obs, merged_state = reset_where(
        _keys(102, batch_size=batch_size),
        stepped_state,
        jnp.array([True, False]),
    )

    assert (
        obs.shape
        == next_obs.shape
        == reset_obs.shape
        == (
            batch_size,
            env.observation_size,
        )
    )
    expected_commands = jnp.where(env._passive_node_mask[None, :], 0.0, actions)
    np.testing.assert_allclose(stepped_state.node_commands, expected_commands, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(merged_state.step_count, np.array([0, 1]))
    np.testing.assert_array_equal(merged_state.data.qpos[1], stepped_state.data.qpos[1])
    assert bool(jnp.all(jnp.isfinite(next_obs)))
    assert bool(jnp.all(jnp.isfinite(reward)))
    assert done.shape == (batch_size,)
    assert info["terminated"].shape == (batch_size,)
    assert bool(jnp.all(info["terminated_during_substeps"]))
    np.testing.assert_array_equal(info["substeps_executed"], np.ones(batch_size, dtype=np.int32))
    diagnostics = env.buffer_diagnostics(stepped_state)
    assert diagnostics["contact_capacity"] == 128 * batch_size
    assert diagnostics["constraint_capacity"] == 256
    assert diagnostics["overflow"] is False


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="CUDA Warp dependencies are unavailable")
def test_warp_default_contact_capacity_reaches_runtime_buffer() -> None:
    batch_size = 2
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            max_steps=2,
            nsubsteps=1,
            speed=0.01,
        ),
        mjx_impl="warp",
        warp_graph_mode="warp_staged",
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    _, state = reset(_keys(105, batch_size=batch_size))
    actions = jnp.zeros((batch_size, env.action_size), dtype=jnp.float32)
    _, state, _, _, _ = step(_keys(106, batch_size=batch_size), state, actions)

    diagnostics = env.buffer_diagnostics(state)
    assert env.warp_naconmax == _DEFAULT_WARP_NACONMAX
    assert diagnostics["contact_capacity"] == _DEFAULT_WARP_NACONMAX
    assert diagnostics["overflow"] is False


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="CUDA Warp dependencies are unavailable")
def test_warp_and_jax_one_step_behavior_match() -> None:
    config = TrussEnvConfig(
        get_mujoco_spec("tetrahedron", realistic=False),
        max_steps=2,
        nsubsteps=1,
        speed=0.01,
    )
    jax_env = MjxNodeVelocityEnv(config, mjx_impl="jax")
    warp_env = MjxNodeVelocityEnv(
        config,
        mjx_impl="warp",
        warp_naconmax=256,
        warp_njmax=256,
    )
    keys = _keys(110, batch_size=1)
    jax_obs, jax_state = jax.jit(jax_env.reset)(keys)
    warp_obs, warp_state = jax.jit(warp_env.reset)(keys)
    action = jnp.linspace(-0.01, 0.01, jax_env.action_size)[None, :]

    jax_result = jax.jit(jax_env.step)(keys, jax_state, action)
    warp_result = jax.jit(warp_env.step)(keys, warp_state, action)

    np.testing.assert_allclose(warp_obs, jax_obs, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(warp_result[0], jax_result[0], rtol=2e-2, atol=2e-4)
    np.testing.assert_allclose(warp_result[2], jax_result[2], rtol=2e-2, atol=2e-4)
    np.testing.assert_array_equal(warp_result[3], jax_result[3])
    np.testing.assert_array_equal(
        warp_result[4]["terminated"],
        jax_result[4]["terminated"],
    )
    np.testing.assert_array_equal(
        warp_result[4]["truncated"],
        jax_result[4]["truncated"],
    )
    np.testing.assert_allclose(
        warp_result[1].node_commands,
        jax_result[1].node_commands,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        warp_result[1].data.ctrl,
        jax_result[1].data.ctrl,
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.cuda
@pytest.mark.skipif(not CUDA_WARP_AVAILABLE, reason="CUDA Warp dependencies are unavailable")
def test_warp_domain_randomization_is_independent_and_finite() -> None:
    batch_size = 4
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            domain_randomization=DomainRandomizationConfig(
                body_mass_multiplier_range=(0.8, 1.2),
                body_inertia_multiplier_range=(0.8, 1.2),
                actuator_gain_multiplier_range=(0.75, 1.25),
                actuator_bias_multiplier_range=(0.75, 1.25),
                geom_friction_slide_range=(0.4, 1.2),
                gravity_z_range=(-10.5, -8.8),
            ),
        ),
        mjx_impl="warp",
        warp_naconmax=128 * batch_size,
        warp_njmax=256,
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    obs, state = reset(_keys(120, batch_size=batch_size))

    masses = np.asarray(state.domain_randomization.body_mass_multiplier)
    assert np.unique(masses).size > 1
    assert np.all((masses >= 0.8) & (masses <= 1.2))
    assert np.all(
        (np.asarray(state.domain_randomization.gravity_z) >= -10.5)
        & (np.asarray(state.domain_randomization.gravity_z) <= -8.8)
    )
    actions = jnp.zeros((batch_size, env.action_size), dtype=jnp.float32)
    for step_index in range(10):
        obs, state, reward, _, _ = step(
            _keys(121 + step_index, batch_size=batch_size),
            state,
            actions,
        )
    assert bool(jnp.all(jnp.isfinite(obs)))
    assert bool(jnp.all(jnp.isfinite(reward)))
    assert env.buffer_diagnostics(state)["overflow"] is False


def test_mjx_domain_randomization_reset_is_deterministic_and_batched() -> None:
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            domain_randomization=DomainRandomizationConfig(
                body_mass_multiplier_range=(0.5, 1.5),
                body_inertia_multiplier_range=(2.0, 2.0),
                dof_armature_range=(0.02, 0.02),
                dof_frictionloss_range=(0.03, 0.03),
                actuator_dynprm_multiplier_range=(4.0, 4.0),
                geom_friction_slide_range=(0.25, 0.25),
                geom_friction_torsional_range=(0.05, 0.05),
                geom_friction_rolling_range=(0.01, 0.01),
                tendon_stiffness_range=(0.4, 0.4),
                tendon_damping_range=(0.5, 0.5),
                tendon_armature_range=(0.06, 0.06),
                tendon_frictionloss_range=(0.07, 0.07),
                gravity_z_range=(-4.0, -4.0),
            ),
        )
    )
    reset = jax.jit(env.reset)

    _, state_a = reset(_keys(30, batch_size=4))
    _, state_b = reset(_keys(30, batch_size=4))
    _, state_c = reset(_keys(31, batch_size=4))

    for actual, expected in zip(
        jax.tree.leaves(state_a.domain_randomization),
        jax.tree.leaves(state_b.domain_randomization),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(
        np.asarray(state_a.domain_randomization.body_mass_multiplier),
        np.asarray(state_c.domain_randomization.body_mass_multiplier),
    )
    np.testing.assert_allclose(state_a.domain_randomization.body_inertia_multiplier, 2.0)
    np.testing.assert_allclose(state_a.domain_randomization.dof_armature, 0.02)
    np.testing.assert_allclose(state_a.domain_randomization.dof_frictionloss, 0.03)
    np.testing.assert_allclose(state_a.domain_randomization.actuator_dynprm_multiplier, 4.0)
    np.testing.assert_allclose(state_a.domain_randomization.geom_friction_slide, 0.25)
    np.testing.assert_allclose(state_a.domain_randomization.geom_friction_torsional, 0.05)
    np.testing.assert_allclose(state_a.domain_randomization.geom_friction_rolling, 0.01)
    np.testing.assert_allclose(state_a.domain_randomization.tendon_stiffness, 0.4)
    np.testing.assert_allclose(state_a.domain_randomization.tendon_damping, 0.5)
    np.testing.assert_allclose(state_a.domain_randomization.tendon_armature, 0.06)
    np.testing.assert_allclose(state_a.domain_randomization.tendon_frictionloss, 0.07)
    np.testing.assert_allclose(state_a.domain_randomization.gravity_z, -4.0)


def test_mjx_warns_when_dof_damping_multiplier_targets_all_zeros() -> None:
    with pytest.warns(UserWarning, match="dof_damping_multiplier_range has no effect"):
        MjxNodeVelocityEnv(
            TrussEnvConfig(
                get_mujoco_spec("octahedron", realistic=False),
                domain_randomization=DomainRandomizationConfig(
                    dof_damping_multiplier_range=(0.8, 1.2)
                ),
            )
        )


def test_mjx_domain_randomization_reset_where_only_resamples_masked_elements() -> None:
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            domain_randomization=DomainRandomizationConfig(
                body_mass_multiplier_range=(0.5, 1.5),
                gravity_z_range=(-11.0, -8.0),
            ),
        )
    )
    reset = jax.jit(env.reset)
    reset_where = jax.jit(env.reset_where)

    _, state = reset(_keys(32, batch_size=3))
    _, merged = reset_where(_keys(33, batch_size=3), state, jnp.array([True, False, True]))

    assert not np.array_equal(
        np.asarray(state.domain_randomization.body_mass_multiplier[0]),
        np.asarray(merged.domain_randomization.body_mass_multiplier[0]),
    )
    np.testing.assert_array_equal(
        merged.domain_randomization.body_mass_multiplier[1],
        state.domain_randomization.body_mass_multiplier[1],
    )
    assert not np.array_equal(
        np.asarray(state.domain_randomization.gravity_z[2]),
        np.asarray(merged.domain_randomization.gravity_z[2]),
    )


def test_mjx_initial_pose_randomization_is_rigid_jitted_and_masked() -> None:
    config = TrussEnvConfig(
        get_mujoco_spec("tetrahedron", realistic=False),
        domain_randomization=DomainRandomizationConfig(
            initial_translation_x_range=(0.5, 0.5),
            initial_translation_y_range=(-0.25, -0.25),
            initial_yaw_range=(np.pi / 2.0, np.pi / 2.0),
        ),
    )
    env = MjxNodeVelocityEnv(config)
    nominal_env = MjxNodeVelocityEnv(
        TrussEnvConfig(get_mujoco_spec("tetrahedron", realistic=False))
    )
    keys = _keys(41, batch_size=3)
    _, state = jax.jit(env.reset)(keys)
    _, nominal_state = jax.jit(nominal_env.reset)(keys)

    np.testing.assert_allclose(state.domain_randomization.initial_translation_x, 0.5)
    np.testing.assert_allclose(state.domain_randomization.initial_translation_y, -0.25)
    np.testing.assert_allclose(state.domain_randomization.initial_yaw, np.pi / 2.0)
    positions = np.asarray(state.data.xpos[:, env._node_body_ids])
    nominal_positions = np.asarray(nominal_state.data.xpos[:, nominal_env._node_body_ids])
    np.testing.assert_allclose(positions[..., 2], nominal_positions[..., 2], atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1),
        np.linalg.norm(nominal_positions[:, :, None] - nominal_positions[:, None, :], axis=-1),
        atol=2e-6,
    )

    _, merged = jax.jit(env.reset_where)(
        _keys(42, batch_size=3), state, jnp.array([True, False, True])
    )
    np.testing.assert_array_equal(merged.data.qpos[1], state.data.qpos[1])
    np.testing.assert_array_equal(
        merged.domain_randomization.initial_yaw[1],
        state.domain_randomization.initial_yaw[1],
    )


def _fixed_randomized_env(*, realistic: bool) -> MjxNodeVelocityEnv:
    return MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=realistic),
            domain_randomization=DomainRandomizationConfig(
                body_mass_multiplier_range=(2.0, 2.0),
                body_inertia_multiplier_range=(3.0, 3.0),
                dof_damping_multiplier_range=(4.0, 4.0),
                dof_armature_range=(0.02, 0.02),
                dof_frictionloss_range=(0.03, 0.03),
                actuator_gain_multiplier_range=(5.0, 5.0),
                actuator_bias_multiplier_range=(6.0, 6.0),
                actuator_dynprm_multiplier_range=(7.0, 7.0),
                geom_friction_slide_range=(0.25, 0.25),
                geom_friction_torsional_range=(0.05, 0.05),
                geom_friction_rolling_range=(0.01, 0.01),
                tendon_stiffness_range=(0.4, 0.4),
                tendon_damping_range=(0.5, 0.5),
                tendon_armature_range=(0.06, 0.06),
                tendon_frictionloss_range=(0.07, 0.07),
                gravity_z_range=(-4.0, -4.0),
            ),
        )
    )


def _cpu_randomized_mjx_model(*, realistic: bool) -> mjx.Model:
    model = get_mujoco_spec("tetrahedron", realistic=realistic).compile()
    data = mujoco.MjData(model)
    model.body_mass[:] *= 2.0
    model.body_inertia[:] *= 3.0
    model.dof_damping[:] *= 4.0
    model.dof_armature[:] = 0.02
    model.dof_frictionloss[:] = 0.03
    model.actuator_gainprm[:] *= 5.0
    model.actuator_biasprm[:] *= 6.0
    model.actuator_dynprm[:] *= 7.0
    model.geom_friction[:, 0] = 0.25
    model.geom_friction[:, 1] = 0.05
    model.geom_friction[:, 2] = 0.01
    model.tendon_stiffness[:] = 0.4
    model.tendon_damping[:] = 0.5
    model.tendon_armature[:] = 0.06
    model.tendon_frictionloss[:] = 0.07
    model.opt.gravity[2] = -4.0
    mujoco.mj_setConst(model, data)
    return mjx.put_model(model)


def test_mjx_domain_randomization_model_patch_matches_cpu_setconst_for_abstract_model() -> None:
    env = _fixed_randomized_env(realistic=False)
    domain = replace(
        env._nominal_domain_randomization_state(),
        body_mass_multiplier=jnp.asarray(2.0),
        body_inertia_multiplier=jnp.asarray(3.0),
        dof_damping_multiplier=jnp.asarray(4.0),
        dof_armature=jnp.asarray(0.02),
        dof_frictionloss=jnp.asarray(0.03),
        actuator_gain_multiplier=jnp.asarray(5.0),
        actuator_bias_multiplier=jnp.asarray(6.0),
        actuator_dynprm_multiplier=jnp.asarray(7.0),
        geom_friction_slide=jnp.asarray(0.25),
        geom_friction_torsional=jnp.asarray(0.05),
        geom_friction_rolling=jnp.asarray(0.01),
        tendon_stiffness=jnp.asarray(0.4),
        tendon_damping=jnp.asarray(0.5),
        tendon_armature=jnp.asarray(0.06),
        tendon_frictionloss=jnp.asarray(0.07),
        gravity_z=jnp.asarray(-4.0),
    )
    actual = env._model_for_domain(domain)
    expected = _cpu_randomized_mjx_model(realistic=False)

    for field in (
        "body_mass",
        "body_subtreemass",
        "body_inertia",
        "body_invweight0",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "dof_invweight0",
        "dof_M0",
        "tendon_invweight0",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_dynprm",
        "geom_friction",
        "tendon_stiffness",
        "tendon_damping",
        "tendon_armature",
        "tendon_frictionloss",
    ):
        np.testing.assert_allclose(
            getattr(actual, field),
            getattr(expected, field),
            rtol=1e-6,
            err_msg=field,
        )
    np.testing.assert_allclose(actual.opt.gravity, expected.opt.gravity)


def test_mjx_domain_randomization_realistic_rollout_is_finite() -> None:
    env = _fixed_randomized_env(realistic=True)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    obs, state = reset(_keys(34, batch_size=2))
    actions = jnp.zeros((2, env.action_size), dtype=jnp.float32)
    obs, state, reward, done, info = step(_keys(35, batch_size=2), state, actions)

    assert obs.shape == (2, env.observation_size)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert all(value.shape == (2,) for value in info.values())
    assert np.all(np.isfinite(np.asarray(state.data.qpos)))
    assert np.all(np.isfinite(np.asarray(obs)))
    assert np.all(np.isfinite(np.asarray(reward)))


def test_mjx_domain_randomization_batch_elements_diverge_under_same_actions() -> None:
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            domain_randomization=DomainRandomizationConfig(
                body_mass_multiplier_range=(0.5, 2.0),
                gravity_z_range=(-12.0, -6.0),
            ),
        )
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    _, state = reset(_keys(36, batch_size=2))
    state = replace(
        state,
        data=jax.tree.map(lambda value: value.at[1].set(value[0]), state.data),
    )
    actions = jnp.full((2, env.action_size), 0.005, dtype=jnp.float32)
    _, state, _, _, _ = step(_keys(37, batch_size=2), state, actions)

    assert not np.array_equal(np.asarray(state.data.qpos[0]), np.asarray(state.data.qpos[1]))


def test_mjx_control_noise_is_explicitly_keyed() -> None:
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            control_noise_std=0.1,
            runtime_apply_control_noise=True,
        )
    )
    commands = jnp.zeros(len(env._controller.actuator_ids))

    noisy_a = env._apply_control_noise(jax.random.key(10), commands)
    noisy_b = env._apply_control_noise(jax.random.key(10), commands)
    noisy_c = env._apply_control_noise(jax.random.key(11), commands)

    np.testing.assert_array_equal(noisy_a, noisy_b)
    assert not np.array_equal(np.asarray(noisy_a), np.asarray(noisy_c))
    assert np.all(np.asarray(noisy_a) >= np.asarray(env._ctrl_low))
    assert np.all(np.asarray(noisy_a) <= np.asarray(env._ctrl_high))


def test_mjx_nonfinite_terminal_diagnostics_keep_reward_finite(
    compiled_env: CompiledEnv,
) -> None:
    _, state = compiled_env.reset(_keys(12, batch_size=1))
    data = jax.tree.map(lambda value: value[0], state.data)
    data = data.replace(xpos=data.xpos.at[compiled_env.env._node_body_ids[0]].set(jnp.nan))

    reward, info, terminated = compiled_env.env._compute_reward(
        data,
        jnp.zeros(compiled_env.env.action_size),
        jnp.zeros(3),
    )

    assert bool(terminated)
    assert bool(info["terminated_by_collapse"])
    assert np.isnan(np.asarray(info["critical_eig_raw"]))
    assert float(info["critical_eig"]) == pytest.approx(0.0)
    assert np.isfinite(np.asarray(reward))


def test_mjx_env_rejects_unsupported_configuration_and_models(tmp_path: Path) -> None:
    spec = get_mujoco_spec("tetrahedron", realistic=False)
    MjxNodeVelocityEnv(
        TrussEnvConfig(
            spec,
            domain_randomization=DomainRandomizationConfig(body_mass_multiplier_range=(1.0, 1.0)),
        )
    )

    with pytest.raises(ValueError, match="model_factory"):
        MjxNodeVelocityEnv(
            TrussEnvConfig(
                spec,
                domain_randomization=DomainRandomizationConfig(model_factory=lambda _rng: spec),
            )
        )

    root = ET.fromstring(spec.to_xml())
    actuator = root.find("actuator")
    assert actuator is not None
    ET.SubElement(
        actuator,
        "general",
        name="bisector_act_unknown",
        joint="node_1_x",
        ctrlrange="-1 1",
    )
    model_path = tmp_path / "unsupported_internal.xml"
    model_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    with pytest.raises(ValueError, match="not owned"):
        MjxNodeVelocityEnv(model_path)

    compiled_model_without_xml_metadata = get_mujoco_spec("tetrahedron", realistic=False).compile()
    with pytest.raises(ValueError, match="control-graph metadata"):
        MjxNodeVelocityEnv(compiled_model_without_xml_metadata)


def test_realistic_mjx_env_jitted_rollout_and_cpu_diagnostics_match() -> None:
    config = TrussEnvConfig(
        get_mujoco_spec("tetrahedron", realistic=True),
        max_steps=2,
        nsubsteps=1,
        speed=0.01,
    )
    env = MjxNodeVelocityEnv(config)
    keys = _keys(20)
    initial_obs, state = jax.jit(env.reset)(keys)

    assert initial_obs.shape == (2, env.observation_size)
    assert np.all(np.isfinite(np.asarray(state.data.ctrl)))
    assert env._angle_bisector_controller.enabled

    actions = jnp.zeros((2, env.action_size), dtype=jnp.float32)
    obs, stepped_state, reward, done, info = jax.jit(env.step)(keys, state, actions)
    assert obs.shape == (2, env.observation_size)
    assert reward.shape == (2,)
    assert done.shape == (2,)
    assert all(value.shape == (2,) for value in info.values())
    assert np.all(np.isfinite(np.asarray(stepped_state.data.qpos)))
    assert np.all(np.isfinite(np.asarray(stepped_state.data.ctrl)))

    cpu_env = MujocoNodeVelocityCommandEnv(config)
    try:
        cpu_env.mj_model.data = mjx.get_data(env.mujoco_model.model, state.data)[0]
        np.testing.assert_allclose(initial_obs[0], cpu_env._get_obs(), rtol=1e-5, atol=1e-6)
        expected_critical_eig = cpu_env.mj_model.collapse_check()
        actual_critical_eig = env._critical_eig(jax.tree.map(lambda value: value[0], state.data))
        assert float(actual_critical_eig) == pytest.approx(
            expected_critical_eig, rel=1e-5, abs=1e-6
        )
    finally:
        cpu_env.close()

    reset_obs, reset_state = jax.jit(env.reset_where)(
        _keys(21), stepped_state, jnp.array([True, False])
    )
    assert reset_obs.shape == (2, env.observation_size)
    np.testing.assert_array_equal(reset_state.step_count, np.array([0, 1]))


def test_mjx_env_validates_leading_batch_shapes(compiled_env: CompiledEnv) -> None:
    env = compiled_env.env
    with pytest.raises(ValueError, match="leading batch dimension"):
        env.reset(jax.random.key(0))

    _, state = env.reset(_keys(13))
    with pytest.raises(ValueError, match="actions must have shape"):
        env.step(_keys(14), state, jnp.zeros((2, env.action_size + 1)))
    with pytest.raises(ValueError, match="mask must have shape"):
        env.reset_where(_keys(15), state, jnp.array([True]))
