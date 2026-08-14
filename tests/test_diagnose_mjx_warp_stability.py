from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiments import diagnose_mjx_warp_stability as diagnostic

from mujoco_truss_gen import (
    DomainRandomizationConfig,
    MjxNodeVelocityEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)


def _keys(seed: int, batch_size: int) -> jax.Array:
    return jax.random.split(jax.random.key(seed), batch_size)


def test_reproduction_round_trip_preserves_batch_pressure_inputs(tmp_path: Path) -> None:
    batch_size = 3
    env = MjxNodeVelocityEnv(
        TrussEnvConfig(
            get_mujoco_spec("tetrahedron", realistic=False),
            max_steps=4,
            nsubsteps=1,
            speed=0.01,
            domain_randomization=DomainRandomizationConfig(
                gravity_z_range=(-9.9, -9.7),
                initial_translation_x_range=(-0.1, 0.1),
            ),
            runtime_apply_control_noise=False,
        )
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    _, state = reset(_keys(1, batch_size))
    actions = np.linspace(
        -0.01,
        0.01,
        num=batch_size * env.action_size,
        dtype=np.float32,
    ).reshape(batch_size, env.action_size)
    reproduction_path = tmp_path / "batch-reproduction.npz"

    diagnostic._save_batch_reproduction(
        reproduction_path,
        state,
        actions,
        environment_index=2,
        failing_qpos=np.asarray(state.data.qpos[2]),
        failing_qvel=np.asarray(state.data.qvel[2]),
        warp_naconmax=32_768,
        warp_njmax=512,
    )
    reproduction = diagnostic._load_batch_reproduction(reproduction_path)

    assert reproduction["environment_index"] == 2
    assert reproduction["actions"].shape == (1, batch_size, env.action_size)
    np.testing.assert_array_equal(reproduction["actions"][0], actions)
    assert reproduction["initial"]["qpos"].shape[0] == batch_size
    assert reproduction["domain_randomization"]["gravity_z"].shape == (batch_size,)

    _, reset_state = reset(_keys(99, batch_size))
    restored_state = diagnostic._restore_mjx_state(env, reset_state, reproduction)
    np.testing.assert_allclose(restored_state.data.qpos, state.data.qpos)
    np.testing.assert_allclose(restored_state.data.qvel, state.data.qvel)
    np.testing.assert_allclose(
        restored_state.domain_randomization.gravity_z,
        state.domain_randomization.gravity_z,
    )

    step_keys = _keys(2, batch_size)
    _, expected_state, expected_reward, _, _ = step(step_keys, state, jnp.asarray(actions))
    _, replayed_state, replayed_reward, _, _ = step(
        step_keys,
        restored_state,
        jnp.asarray(reproduction["actions"][0]),
    )
    np.testing.assert_allclose(replayed_state.data.qpos, expected_state.data.qpos, rtol=1e-5)
    np.testing.assert_allclose(replayed_state.data.qvel, expected_state.data.qvel, rtol=1e-5)
    np.testing.assert_allclose(replayed_reward, expected_reward, rtol=1e-5)


def test_replay_rejects_legacy_single_environment_artifact(tmp_path: Path) -> None:
    reproduction_path = tmp_path / "legacy.npz"
    np.savez_compressed(reproduction_path, initial_qpos=np.zeros(1))

    with pytest.raises(ValueError, match="shared Warp batch"):
        diagnostic._load_batch_reproduction(reproduction_path)
