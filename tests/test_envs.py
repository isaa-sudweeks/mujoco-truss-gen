from __future__ import annotations

import numpy as np

from mujoco_truss_gen import (
    MujocoRelativeObsEnv,
    MujocoTrussEnv,
    MujocoVelocityCommandEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)


def test_generated_spec_runs_in_all_builtin_envs() -> None:
    spec = get_mujoco_spec("octahedron", realistic=False)

    for env_cls in (MujocoTrussEnv, MujocoRelativeObsEnv, MujocoVelocityCommandEnv):
        env = env_cls(TrussEnvConfig(spec, max_steps=3, nsubsteps=1, speed=0.01))
        try:
            obs, _ = env.reset(seed=123)
            assert env.observation_space.contains(obs)

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)

            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert "critical_eig" in info
        finally:
            env.close()


def test_env_accepts_xml_path(tmp_path) -> None:
    spec = get_mujoco_spec("octahedron", realistic=False)
    xml_path = tmp_path / "model.xml"
    xml_path.write_text(spec.to_xml(), encoding="utf-8")

    env = MujocoTrussEnv(xml_path, max_steps=1)
    try:
        env.reset(seed=7)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, _, truncated, _ = env.step(action)
        assert truncated
    finally:
        env.close()
