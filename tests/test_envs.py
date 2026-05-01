from __future__ import annotations

from copy import deepcopy

import numpy as np

from mujoco_truss_gen import (
    PRESETS,
    MujocoRelativeObsEnv,
    MujocoTrussEnv,
    MujocoVelocityCommandEnv,
    TrussEnvConfig,
    get_icosahedron_definition,
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


def test_builtin_presets_compile() -> None:
    assert {"octahedron", "icosahedron"} <= set(PRESETS)

    for preset_name in PRESETS:
        get_mujoco_spec(preset_name, realistic=False).compile()
        get_mujoco_spec(preset_name, realistic=True).compile()


def test_icosahedron_definition_shape() -> None:
    node_dict, triangle_dict = get_icosahedron_definition()

    assert len(node_dict) == 12
    assert len(triangle_dict) == 20
    for triangle_nodes in triangle_dict.values():
        assert len(triangle_nodes) == 4
        assert triangle_nodes[3] in triangle_nodes[:3]


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


def test_custom_dictionary_spec_compiles_and_runs() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    spec = get_mujoco_spec(node_dict, triangle_dict, realistic=False)
    env = MujocoTrussEnv(TrussEnvConfig(spec, max_steps=2, nsubsteps=1, speed=0.01))
    try:
        obs, _ = env.reset(seed=11)
        assert env.observation_space.contains(obs)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert not truncated
        assert "critical_eig" in info
    finally:
        env.close()


def test_generation_does_not_mutate_custom_dictionaries() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
        "node_4": [0.4, -0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
        "triangle_2": ["node_1", "node_4", "node_2", "node_1"],
    }
    original_nodes = deepcopy(node_dict)
    original_triangles = deepcopy(triangle_dict)

    get_mujoco_spec(node_dict, triangle_dict, realistic=False).compile()
    assert node_dict == original_nodes
    assert triangle_dict == original_triangles

    get_mujoco_spec(node_dict, triangle_dict, realistic=True).compile()
    assert node_dict == original_nodes
    assert triangle_dict == original_triangles
