from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np
import pytest

from mujoco_truss_gen import (
    PRESETS,
    MujocoRelativeObsEnv,
    MujocoTrussEnv,
    MujocoVelocityCommandEnv,
    TrussEnvConfig,
    get_edge_index,
    get_icosahedron_definition,
    get_mujoco_spec,
    get_route_lengths,
    save_xml,
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
    assert {"octahedron", "icosahedron", "tetrahedron"} <= set(PRESETS)

    for preset_name in PRESETS:
        get_mujoco_spec(preset_name, realistic=False).compile()

    for preset_name in ("octahedron", "icosahedron", "solar_array"):
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


def test_save_xml_relative_path_uses_current_working_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    spec = get_mujoco_spec("octahedron", realistic=False)

    xml_path = save_xml(spec, "generated/octahedron.xml")

    assert xml_path == tmp_path / "generated" / "octahedron.xml"
    assert xml_path.exists()
    assert xml_path.read_text(encoding="utf-8").lstrip().startswith("<mujoco")


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


def test_custom_triangle_validation_reports_unknown_nodes() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_missing", "node_1"],
    }

    with pytest.raises(ValueError, match="Triangle 'triangle_1' references unknown node"):
        get_mujoco_spec(node_dict, triangle_dict)


def test_custom_triangle_validation_reports_bad_passive_node() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
        "node_4": [0.4, -0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_4"],
    }

    with pytest.raises(ValueError, match="passive node 'node_4' must be one"):
        get_mujoco_spec(node_dict, triangle_dict)


def test_custom_node_validation_reports_bad_position() -> None:
    node_dict = {
        "node_1": [0.0, 0.0],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    with pytest.raises(ValueError, match="Node 'node_1' position"):
        get_mujoco_spec(node_dict, triangle_dict)


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


def test_routed_shape_spec_compiles_and_runs() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.8, 0.8, 0.2],
        "node_4": [0.0, 0.8, 0.2],
    }
    shape_dict = {
        "quad_1": {
            "route": ["node_1", "node_2", "node_3", "node_4", "node_1"],
            "active_edges": [["node_1", "node_2"], ["node_4", "node_1"]],
        },
    }

    assert get_route_lengths(node_dict, shape_dict) == {"quad_1": 3.2}
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=False)
    assert get_edge_index(spec).shape == (2, 8)

    env = MujocoTrussEnv(TrussEnvConfig(spec, max_steps=2, nsubsteps=1, speed=0.01))
    try:
        obs, _ = env.reset(seed=13)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (2,)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, _, _, _, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert "critical_eig" in info
    finally:
        env.close()


def test_routed_shape_generation_does_not_mutate_custom_dictionaries() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.8, 0.8, 0.2],
        "node_4": [0.0, 0.8, 0.2],
    }
    shape_dict = {
        "quad_1": {
            "route": ["node_1", "node_2", "node_3", "node_4", "node_1"],
            "active_edges": [["node_1", "node_2"], ["node_4", "node_1"]],
        },
    }
    original_nodes = deepcopy(node_dict)
    original_shapes = deepcopy(shape_dict)

    get_mujoco_spec(node_dict, shape_dict, realistic=False).compile()

    assert node_dict == original_nodes
    assert shape_dict == original_shapes


def test_tetrahedron_route_constraints_start_satisfied() -> None:
    spec = get_mujoco_spec("tetrahedron", realistic=False)
    model = spec.compile()
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    assert data.nefc >= 2
    np.testing.assert_allclose(data.efc_pos[:2], [0.0, 0.0], atol=1e-8)


def test_actuator_names_are_edge_based() -> None:
    model = get_mujoco_spec("tetrahedron", realistic=False).compile()

    actuator_names = {model.actuator(index).name for index in range(model.nu)}

    assert actuator_names == {"act_12", "act_34", "act_23", "act_14"}
