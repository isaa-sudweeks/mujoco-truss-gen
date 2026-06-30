from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mujoco import mjx

from mujoco_truss_gen import (
    PRESETS,
    DomainRandomizationConfig,
    MjxNodeVelocityEnv,
    MujocoNodeVelocityCommandEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)


def _is_indexed_henneberg_variant(preset_name: str) -> bool:
    return (
        preset_name.startswith("henneberg_") and preset_name.rsplit("_", maxsplit=1)[-1].isdigit()
    )


CANONICAL_PRESET_NAMES = tuple(name for name in PRESETS if not _is_indexed_henneberg_variant(name))


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


@pytest.mark.parametrize("preset_name", CANONICAL_PRESET_NAMES)
def test_mjx_node_velocity_env_constructs_for_canonical_abstract_presets(
    preset_name: str,
) -> None:
    env = MjxNodeVelocityEnv(get_mujoco_spec(preset_name, realistic=False))

    assert env.action_size > 0
    assert env.observation_size == 7 * env.action_size
    assert env.action_low.shape == (env.action_size,)
    assert env.action_high.shape == (env.action_size,)


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
        "rigidity",
        "slip",
        "terminated",
        "terminated_by_collapse",
        "truncated",
    }
    assert set(info) == expected_info_keys
    assert all(value.shape == (2,) for value in info.values())

    _, state, _, done, info = compiled_env.step(_keys(4), state, actions)
    np.testing.assert_array_equal(state.step_count, np.full(2, 2, dtype=np.int32))
    assert np.all(np.asarray(done))
    assert not np.any(np.asarray(info["terminated"]))
    assert np.all(np.asarray(info["truncated"]))


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
    with pytest.raises(ValueError, match="DomainRandomizationConfig"):
        MjxNodeVelocityEnv(TrussEnvConfig(spec, domain_randomization=DomainRandomizationConfig()))

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
