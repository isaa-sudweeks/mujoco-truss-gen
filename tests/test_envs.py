from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from itertools import combinations

import mujoco
import numpy as np
import pytest

from mujoco_truss_gen import (
    HENNEBERG_PRESET_SPECS,
    HENNEBERG_PRESET_VARIANT_COUNTS,
    HENNEBERG_RIGIDITY_THRESHOLD,
    PRESETS,
    AccelerometerConfig,
    DomainRandomizationConfig,
    MujocoModel,
    MujocoNodeVelocityCommandEnv,
    MujocoRelativeObsEnv,
    MujocoTrussEnv,
    MujocoVelocityCommandEnv,
    NodeVelocityController,
    TrussEnvConfig,
    TrussPhysicalParameters,
    get_edge_index,
    get_edge_types,
    get_henneberg_routed_graph_definition,
    get_icosahedron_definition,
    get_mujoco_spec,
    get_networkx_graph,
    get_node_features,
    get_preset_definition,
    get_route_lengths,
    get_usevitch_graph_definition,
    save_xml,
    view_graph,
)
from mujoco_truss_gen.mujoco_model import presets as preset_module
from mujoco_truss_gen.mujoco_model.constants import (
    ACTIVE_NODE_MASS,
    ACTUATOR_CTRL_RANGE,
    EDGE_TENDON_WIDTH,
    HINGE_DAMPING,
    NODE_RADIUS,
    PASSIVE_NODE_MASS,
)
from mujoco_truss_gen.mujoco_model.io_viewer import (
    NodeVelocityViewerState,
    _apply_terminal_command,
)


def _default_compile_preset_names() -> list[str]:
    return [
        preset_name
        for preset_name in PRESETS
        if not _is_indexed_henneberg_variant(preset_name)
    ]


def _is_indexed_henneberg_variant(preset_name: str) -> bool:
    if not preset_name.startswith("henneberg_"):
        return False
    suffix = preset_name.rsplit("_", maxsplit=1)[-1]
    return suffix.isdigit()


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

    for preset_name in _default_compile_preset_names():
        get_mujoco_spec(preset_name, realistic=False).compile()

    for preset_name in ("octahedron", "icosahedron", "solar_array"):
        get_mujoco_spec(preset_name, realistic=True).compile()

    get_mujoco_spec("tetrahedron", realistic=True).compile()


def test_usevitch_graph_presets_compile_and_match_partitions() -> None:
    expected_presets = {
        "usevitch_1514879",
        "usevitch_210272254_p1",
        "usevitch_210272254_p2",
        "usevitch_212365307",
        "usevitch_54501547959",
        "usevitch_64702095263",
        "usevitch_49530656767",
        "usevitch_53827448765",
        "usevitch_54364254015",
        "usevitch_44565393342",
        "usevitch_44968308287",
        "usevitch_44137822173",
        "usevitch_60202270686_p1",
        "usevitch_60202270686_p2",
        "usevitch_60243677150_p1",
        "usevitch_60243677150_p2",
        "usevitch_60243677150_p3",
    }

    assert expected_presets <= set(PRESETS)

    for preset_name in expected_presets:
        node_dict, triangle_dict = get_preset_definition(preset_name)
        assert len(triangle_dict) == len(node_dict) - 2
        edge_keys = {
            tuple(sorted(edge))
            for triangle_nodes in triangle_dict.values()
            for edge in combinations(triangle_nodes[:3], 2)
        }
        edge_lengths = []
        for from_node, to_node in edge_keys:
            edge_length = np.linalg.norm(
                np.asarray(node_dict[to_node]) - np.asarray(node_dict[from_node])
            )
            edge_lengths.append(edge_length)
            assert np.isfinite(edge_length)
            assert edge_length > 1e-8
        assert np.mean(edge_lengths) == pytest.approx(1.0)
        for from_node, to_node in combinations(node_dict, 2):
            node_distance = np.linalg.norm(
                np.asarray(node_dict[to_node]) - np.asarray(node_dict[from_node])
            )
            assert np.isfinite(node_distance)
            assert node_distance > 1e-8
        get_mujoco_spec(preset_name, realistic=False).compile()

    nodes, triangles = get_usevitch_graph_definition(60243677150, partition_index=3)
    assert len(nodes) == 9
    assert len(triangles) == 7


def test_henneberg_graph_generation_counts_match_reference() -> None:
    graphs_by_node_count = preset_module._henneberg_graphs_by_node_count(8)

    assert {node_count: len(graphs) for node_count, graphs in graphs_by_node_count.items()} == {
        4: 1,
        5: 1,
        6: 4,
        7: 26,
        8: 374,
    }

    assert [
        (
            node_count,
            len(graphs),
            sum(preset_module._minimum_trail_count(graph) == 1 for graph in graphs),
            sum(
                graph.number_of_edges() % 2 == 0
                and preset_module._minimum_trail_count(graph) == 2
                for graph in graphs
            ),
            sum(
                graph.number_of_edges() % 3 == 0
                and preset_module._minimum_trail_count(graph) == 3
                for graph in graphs
            ),
        )
        for node_count, graphs in graphs_by_node_count.items()
        if node_count >= 5
    ] == [
        (5, 1, 1, 0, 0),
        (6, 4, 2, 1, 1),
        (7, 26, 10, 0, 3),
        (8, 374, 85, 190, 89),
    ]
    assert HENNEBERG_PRESET_VARIANT_COUNTS == {
        (5, 1): 1,
        (6, 1): 2,
        (6, 2): 1,
        (6, 3): 1,
        (7, 1): 10,
        (7, 3): 3,
        (8, 1): 85,
        (8, 2): 190,
        (8, 3): 89,
    }


def test_henneberg_routed_graph_presets_are_rigid_equal_route_covers() -> None:
    expected_presets = {
        f"henneberg_n{node_count}_{tube_count}tube"
        for node_count, tube_count in HENNEBERG_PRESET_SPECS
    }
    expected_presets.update(
        f"henneberg_n{node_count}_{tube_count}tube_{preset_index}"
        for (node_count, tube_count), variant_count in HENNEBERG_PRESET_VARIANT_COUNTS.items()
        for preset_index in range(1, variant_count + 1)
    )
    assert expected_presets <= set(PRESETS)

    for node_count, tube_count in HENNEBERG_PRESET_SPECS:
        preset_name = f"henneberg_n{node_count}_{tube_count}tube"
        indexed_preset_name = f"{preset_name}_1"
        node_dict, shape_dict = get_preset_definition(preset_name)
        indexed_node_dict, indexed_shape_dict = get_preset_definition(indexed_preset_name)
        direct_nodes, direct_shapes = get_henneberg_routed_graph_definition(
            node_count,
            tube_count,
            preset_index=1,
        )
        scaled_nodes, scaled_shapes = get_preset_definition(preset_name, scale=2.5)

        assert indexed_node_dict == node_dict
        assert indexed_shape_dict == shape_dict
        assert node_dict == direct_nodes
        assert shape_dict == direct_shapes
        assert scaled_shapes == shape_dict
        for node_name, position in node_dict.items():
            np.testing.assert_allclose(scaled_nodes[node_name], np.asarray(position) * 2.5)

        assert len(node_dict) == node_count
        assert len(shape_dict) == tube_count
        route_lengths = [len(shape["route"]) - 1 for shape in shape_dict.values()]
        if tube_count > 1:
            assert len(set(route_lengths)) == 1

        route_edges = []
        for shape in shape_dict.values():
            route = shape["route"]
            assert shape["active_edges"] == [
                [from_node, to_node]
                for from_node, to_node in zip(route, route[1:], strict=False)
            ]
            for from_node, to_node in zip(route, route[1:], strict=False):
                from_index = int(from_node.removeprefix("node_")) - 1
                to_index = int(to_node.removeprefix("node_")) - 1
                route_edges.append(tuple(sorted((from_index, to_index))))

        assert len(route_edges) == 3 * node_count - 6
        assert len(set(route_edges)) == len(route_edges)

        coordinates = np.array(
            [node_dict[f"node_{index}"] for index in range(1, node_count + 1)],
            dtype=float,
        )
        assert np.all(np.isfinite(coordinates))
        edge_lengths = preset_module._edge_lengths(coordinates, tuple(sorted(route_edges)))
        assert np.min(edge_lengths) > 1e-8
        assert np.mean(edge_lengths) == pytest.approx(1.0)

        wcri = preset_module._worst_case_rigidity_index(coordinates, tuple(sorted(route_edges)))
        rank = preset_module._rigidity_matrix_rank(coordinates, tuple(sorted(route_edges)))
        assert wcri > HENNEBERG_RIGIDITY_THRESHOLD
        assert rank == 3 * node_count - 6

        get_mujoco_spec(preset_name, realistic=False).compile()


def test_henneberg_indexed_presets_can_select_distinct_configurations() -> None:
    first_nodes, first_shapes = get_preset_definition("henneberg_n6_1tube_1")
    second_nodes, second_shapes = get_preset_definition("henneberg_n6_1tube_2")

    assert first_shapes != second_shapes or first_nodes != second_nodes
    get_mujoco_spec("henneberg_n6_1tube_2", realistic=False).compile()

    high_index_nodes, high_index_shapes = get_preset_definition("henneberg_n8_2tube_190")
    high_index_edges = tuple(
        sorted(
            tuple(
                sorted(
                    (
                        int(from_node.removeprefix("node_")) - 1,
                        int(to_node.removeprefix("node_")) - 1,
                    )
                )
            )
            for shape in high_index_shapes.values()
            for from_node, to_node in zip(shape["route"], shape["route"][1:], strict=False)
        )
    )
    high_index_coordinates = np.array(
        [high_index_nodes[f"node_{index}"] for index in range(1, 9)],
        dtype=float,
    )
    assert [len(shape["route"]) - 1 for shape in high_index_shapes.values()] == [9, 9]
    assert preset_module._rigidity_matrix_rank(high_index_coordinates, high_index_edges) == 18


def test_henneberg_preset_definitions_are_deep_copied() -> None:
    _, mutated_shapes = get_preset_definition("henneberg_n6_2tube")
    mutated_shapes["path_1"]["route"][0] = "mutated"
    mutated_shapes["path_1"]["active_edges"][0][0] = "mutated"

    _, fresh_shapes = get_preset_definition("henneberg_n6_2tube")

    assert fresh_shapes["path_1"]["route"][0] != "mutated"
    assert fresh_shapes["path_1"]["active_edges"][0][0] != "mutated"


def test_usevitch_embedding_distance_matrix_matches_paper_algorithm() -> None:
    rng = np.random.default_rng(123)
    edges = ((0, 1), (2, 3))

    distances = preset_module._random_usevitch_distance_matrix(4, edges, rng)

    np.testing.assert_allclose(distances, distances.T)
    np.testing.assert_allclose(np.diag(distances), np.zeros(4))
    assert 0.0 <= distances[0, 1] < 1.0
    assert 0.0 <= distances[2, 3] < 1.0
    assert distances[0, 2] == pytest.approx(10.0)
    assert distances[0, 3] == pytest.approx(10.0)
    assert distances[1, 2] == pytest.approx(10.0)
    assert distances[1, 3] == pytest.approx(10.0)


def test_usevitch_embedding_candidates_normalize_mean_edge_length() -> None:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 4.0, 0.0],
        ]
    )
    edges = ((0, 1), (1, 2))

    normalized = preset_module._normalize_usevitch_candidate_edge_lengths(coordinates, edges)
    edge_lengths = preset_module._edge_lengths(normalized, edges)

    np.testing.assert_allclose(edge_lengths, [2.0 / 3.0, 4.0 / 3.0])
    assert np.mean(edge_lengths) == pytest.approx(1.0)


def test_usevitch_embedding_search_avoids_near_singular_presets() -> None:
    for graph_label in preset_module.USEVITCH_GRAPH_LABELS:
        node_count = preset_module._node_count_from_usevitch_label(graph_label)
        edges = tuple(sorted(preset_module._usevitch_edges_from_label(graph_label, node_count)))
        coordinates = preset_module._best_usevitch_embedding(graph_label, node_count, edges)

        wcri = preset_module._worst_case_rigidity_index(coordinates, edges)

        assert wcri > 1e-4


def test_usevitch_embedding_search_skips_fallback_when_default_is_rigid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_search(
        node_count: int,
        edges: tuple[tuple[int, int], ...],
        rng: np.random.Generator,
        distance_parameters: tuple[float, float, float],
        best_coordinates: np.ndarray | None,
        best_wcri: float,
        best_edge_rms_error: float,
    ) -> tuple[np.ndarray | None, float, float]:
        calls.append(distance_parameters)
        return np.ones((node_count, 3), dtype=float), 2e-4, 0.0

    preset_module._best_usevitch_embedding.cache_clear()
    monkeypatch.setattr(preset_module, "_search_usevitch_mds_parameters", fake_search)
    try:
        preset_module._best_usevitch_embedding(1514879, 7, ((0, 1),))
    finally:
        preset_module._best_usevitch_embedding.cache_clear()

    assert calls == [preset_module.USEVITCH_MDS_DEFAULT_DISTANCE_PARAMETERS]


def test_builtin_preset_definitions_support_unit_scale() -> None:
    unscaled_nodes, unscaled_structure = get_preset_definition("octahedron")
    same_nodes, same_structure = get_preset_definition("octahedron", scale=1.0)
    scaled_nodes, scaled_structure = get_preset_definition("octahedron", scale=2.5)

    assert same_nodes == unscaled_nodes
    assert same_structure == unscaled_structure
    assert scaled_structure == unscaled_structure
    for node_name, position in unscaled_nodes.items():
        np.testing.assert_allclose(scaled_nodes[node_name], np.asarray(position) * 2.5)


def test_scaled_named_preset_compiles() -> None:
    get_mujoco_spec("tetrahedron", scale=0.5, realistic=False).compile()
    get_mujoco_spec("octahedron", scale=2.5, realistic=False).compile()
    get_mujoco_spec("octahedron", scale=2.0, realistic=True).compile()


def test_model_records_initial_bounding_box_diagonal() -> None:
    base_model = MujocoModel(get_mujoco_spec("octahedron", scale=1.0, realistic=False))
    scaled_model = MujocoModel(get_mujoco_spec("octahedron", scale=2.5, realistic=False))

    assert base_model.initial_bounding_box_diagonal > 0.0
    assert base_model.initial_bounding_box_dimensions.shape == (3,)
    np.testing.assert_allclose(
        scaled_model.initial_bounding_box_diagonal,
        2.5 * base_model.initial_bounding_box_diagonal,
    )
    active_axes = list(base_model.axis_indices)
    np.testing.assert_allclose(
        scaled_model.initial_bounding_box_dimensions[active_axes],
        2.5 * base_model.initial_bounding_box_dimensions[active_axes],
    )


@pytest.mark.parametrize("realistic", [False, True])
def test_relative_obs_normalizes_by_initial_bounding_box_dimensions(
    realistic: bool,
) -> None:
    raw_env = MujocoRelativeObsEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=realistic),
            normalize_observations=False,
        )
    )
    normalized_env = MujocoRelativeObsEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=realistic),
            normalize_observations=True,
        )
    )
    try:
        raw_obs = raw_env._get_obs()
        normalized_obs = normalized_env._get_obs()

        axis_divisors = raw_env.mj_model.initial_bounding_box_dimensions[
            list(raw_env.mj_model.axis_indices)
        ]
        divisors = np.tile(axis_divisors, len(raw_env.mj_model.node_names))
        node_obs_size = divisors.size

        np.testing.assert_allclose(
            normalized_obs[:node_obs_size],
            raw_obs[:node_obs_size] / divisors,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized_obs[node_obs_size : 2 * node_obs_size],
            raw_obs[node_obs_size : 2 * node_obs_size] / divisors,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized_obs[2 * node_obs_size :],
            raw_obs[2 * node_obs_size :],
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        raw_env.close()
        normalized_env.close()


@pytest.mark.parametrize("realistic", [False, True])
def test_base_obs_normalizes_coordinate_channels_by_initial_bounding_box_dimensions(
    realistic: bool,
) -> None:
    raw_env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=realistic),
            normalize_observations=False,
        )
    )
    normalized_env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=realistic),
            normalize_observations=True,
        )
    )
    try:
        raw_obs = raw_env._get_obs()
        normalized_obs = normalized_env._get_obs()

        coordinate_divisors = raw_env.mj_model.initial_bounding_box_dimensions[[0, 2]]
        np.testing.assert_allclose(
            normalized_obs[:-4],
            raw_obs[:-4],
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized_obs[-4:-2],
            raw_obs[-4:-2] / coordinate_divisors,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            normalized_obs[-2:],
            raw_obs[-2:] / coordinate_divisors,
            rtol=1e-6,
            atol=1e-6,
        )
    finally:
        raw_env.close()
        normalized_env.close()


def test_forward_reward_is_scaled_by_initial_bounding_box_diagonal() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", scale=2.0, realistic=False),
            forward_weight=3.0,
            energy_weight=0.0,
            alive_bonus=0.0,
            rigidity_weight=0.0,
            slip_weight=0.0,
            critical_eig_threshold=0.0,
            max_forward_velocity=None,
        )
    )
    try:
        env.mj_model.get_forward_velocity = lambda: 6.0
        env.mj_model.collapse_check = lambda: 1.0
        env.mj_model.get_slip_penalty = lambda height: 0.0

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        reward, info, terminated = env._compute_reward(action)

        expected_forward = 3.0 * 6.0 / env.mj_model.initial_bounding_box_diagonal
        np.testing.assert_allclose(info["forward"], expected_forward)
        np.testing.assert_allclose(reward, expected_forward)
        assert not terminated
    finally:
        env.close()


def test_collapse_terminal_step_suppresses_unstable_positive_forward_reward() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            forward_weight=3.0,
            energy_weight=0.0,
            alive_bonus=0.25,
            rigidity_weight=5.0,
            slip_weight=0.5,
            critical_eig_threshold=0.03,
            max_forward_velocity=1.0,
            collapse_penalty=2.0,
        )
    )
    try:
        env.mj_model.get_node_position_matrix = lambda: np.array(
            [[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            dtype=float,
        )
        env.mj_model.collapse_check = lambda: 0.0
        env.mj_model.get_slip_penalty = lambda height: 50.0

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        reward, info, terminated = env._compute_reward(action, np.zeros(3, dtype=float))

        assert terminated
        assert info["terminated_by_collapse"]
        assert info["forward_velocity_raw"] > env.config.max_forward_velocity
        assert info["forward_velocity"] == pytest.approx(0.0)
        assert info["forward"] == pytest.approx(0.0)
        assert info["alive"] == pytest.approx(0.0)
        assert info["rigidity"] == pytest.approx(0.0)
        assert info["slip"] == pytest.approx(0.0)
        assert info["collapse_penalty"] == pytest.approx(-2.0)
        assert reward == pytest.approx(-2.0)
    finally:
        env.close()


def test_nonfinite_terminal_diagnostics_do_not_make_reward_nonfinite() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            forward_weight=3.0,
            energy_weight=0.0,
            alive_bonus=0.25,
            rigidity_weight=5.0,
            slip_weight=0.5,
            critical_eig_threshold=0.03,
            max_forward_velocity=1.0,
            collapse_penalty=1.5,
        )
    )
    try:
        env.mj_model.get_node_position_matrix = lambda: np.array(
            [[np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0]],
            dtype=float,
        )
        env.mj_model.collapse_check = lambda: np.nan
        env.mj_model.get_slip_penalty = lambda height: pytest.fail(
            "terminal velocity shaping should not compute slip"
        )

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        reward, info, terminated = env._compute_reward(action, np.zeros(3, dtype=float))

        assert terminated
        assert info["terminated_by_collapse"]
        assert np.isnan(info["critical_eig_raw"])
        assert info["critical_eig"] == pytest.approx(0.0)
        assert info["forward_velocity"] == pytest.approx(0.0)
        assert info["forward"] == pytest.approx(0.0)
        assert info["rigidity"] == pytest.approx(0.0)
        assert info["slip"] == pytest.approx(0.0)
        assert info["collapse_penalty"] == pytest.approx(-1.5)
        assert np.isfinite(reward)
        assert reward == pytest.approx(-1.5)
    finally:
        env.close()


def test_nonfinite_nonterminal_velocity_and_slip_are_sanitized_for_reward() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            forward_weight=3.0,
            energy_weight=0.0,
            alive_bonus=0.0,
            rigidity_weight=0.0,
            slip_weight=0.5,
            critical_eig_threshold=0.03,
            max_forward_velocity=None,
        )
    )
    try:
        env.mj_model.get_node_position_matrix = lambda: np.array(
            [[np.nan, 0.0, 0.0], [np.nan, 0.0, 0.0]],
            dtype=float,
        )
        env.mj_model.collapse_check = lambda: 1.0
        env.mj_model.get_slip_penalty = lambda height: np.inf

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        reward, info, terminated = env._compute_reward(action, np.zeros(3, dtype=float))

        assert not terminated
        assert np.isnan(info["forward_velocity_raw"])
        assert info["forward_velocity"] == pytest.approx(0.0)
        assert info["forward"] == pytest.approx(0.0)
        assert info["slip"] == pytest.approx(0.0)
        assert np.isfinite(reward)
        assert reward == pytest.approx(0.0)
    finally:
        env.close()


def test_nonterminal_forward_reward_clips_normalized_com_velocity() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            forward_weight=4.0,
            energy_weight=0.0,
            alive_bonus=0.0,
            rigidity_weight=0.0,
            slip_weight=0.0,
            critical_eig_threshold=0.03,
            max_forward_velocity=0.25,
        )
    )
    try:
        env.mj_model.get_node_position_matrix = lambda: np.array(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=float,
        )
        env.mj_model.collapse_check = lambda: 1.0
        env.mj_model.get_slip_penalty = lambda height: 0.0

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        reward, info, terminated = env._compute_reward(action, np.zeros(3, dtype=float))

        expected_forward = env.config.forward_weight * env.config.max_forward_velocity
        expected_physical_velocity = (
            env.config.max_forward_velocity * env.mj_model.initial_bounding_box_diagonal
        )
        assert not terminated
        assert info["forward_velocity_normalized_raw"] > env.config.max_forward_velocity
        assert info["forward_velocity_normalized"] == pytest.approx(
            env.config.max_forward_velocity
        )
        assert info["forward_velocity"] == pytest.approx(expected_physical_velocity)
        assert info["com_delta_x"] == pytest.approx(1.0)
        assert info["forward"] == pytest.approx(expected_forward)
        assert reward == pytest.approx(expected_forward)
    finally:
        env.close()


def test_max_forward_velocity_none_disables_com_velocity_clipping() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            forward_weight=1.0,
            energy_weight=0.0,
            alive_bonus=0.0,
            rigidity_weight=0.0,
            slip_weight=0.0,
            critical_eig_threshold=0.03,
            max_forward_velocity=None,
        )
    )
    try:
        env.mj_model.get_node_position_matrix = lambda: np.array(
            [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
            dtype=float,
        )
        env.mj_model.collapse_check = lambda: 1.0
        env.mj_model.get_slip_penalty = lambda height: 0.0

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, info, terminated = env._compute_reward(action, np.zeros(3, dtype=float))

        assert not terminated
        assert info["forward_velocity_raw"] > 1.0
        assert info["forward_velocity"] == pytest.approx(info["forward_velocity_raw"])
        assert info["forward_velocity_normalized"] == pytest.approx(
            info["forward_velocity_normalized_raw"]
        )
    finally:
        env.close()


def test_normalized_velocity_clipping_is_scale_invariant() -> None:
    rewards = []
    for scale in (0.5, 2.0):
        env = MujocoTrussEnv(
            TrussEnvConfig(
                get_mujoco_spec("octahedron", scale=scale, realistic=False),
                forward_weight=3.0,
                energy_weight=0.0,
                alive_bonus=0.0,
                rigidity_weight=0.0,
                slip_weight=0.0,
                critical_eig_threshold=0.0,
                max_forward_velocity=0.25,
            )
        )
        try:
            env.mj_model.get_forward_velocity = lambda env=env: (
                2.0 * env.mj_model.initial_bounding_box_diagonal
            )
            env.mj_model.collapse_check = lambda: 1.0
            env.mj_model.get_slip_penalty = lambda height: 0.0

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            reward, info, terminated = env._compute_reward(action)

            assert not terminated
            assert info["forward_velocity_normalized_raw"] == pytest.approx(2.0)
            assert info["forward_velocity_normalized"] == pytest.approx(0.25)
            rewards.append(reward)
        finally:
            env.close()

    assert rewards == pytest.approx([0.75, 0.75])


def test_reset_reinitializes_scaled_tendon_actuator_lengths() -> None:
    for realistic in (False, True):
        model = MujocoModel(get_mujoco_spec("octahedron", scale=2.5, realistic=realistic))
        model.reset(np.random.default_rng(7))

        for actuator_id in model.external_actuator_ids:
            if model.model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_TENDON:
                continue
            if model.model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_INTEGRATOR:
                continue

            act_adr = model.model.actuator_actadr[actuator_id]
            tendon_id = model.model.actuator_trnid[actuator_id, 0]
            assert act_adr >= 0
            assert model.data.act[act_adr] == pytest.approx(model.data.ten_length[tendon_id])


def test_scaled_abstract_preset_keeps_control_values_unscaled() -> None:
    root = ET.fromstring(get_mujoco_spec("octahedron", scale=2.5, realistic=False).to_xml())

    tendon = root.find(".//tendon/spatial[@name='tendon_node_1_node_2']")
    assert tendon is not None
    np.testing.assert_allclose(_xml_vector(tendon.get("range", "")), [0.5, 5.0])

    actuator = root.find(".//actuator/general[@name='act_12']")
    assert actuator is not None
    np.testing.assert_allclose(
        _xml_vector(actuator.get("ctrlrange", "")),
        ACTUATOR_CTRL_RANGE,
    )
    np.testing.assert_allclose(_xml_vector(actuator.get("actrange", "")), [0.0, 3.0])


def test_scaled_realistic_preset_scales_edge_tendon_upper_range() -> None:
    root = ET.fromstring(get_mujoco_spec("octahedron", scale=2.5, realistic=True).to_xml())

    tendon = root.find(".//tendon/spatial[@name='tendon_node_1_node_2']")
    assert tendon is not None
    np.testing.assert_allclose(_xml_vector(tendon.get("range", "")), [0.5, 5.0])


def test_env_domain_randomization_model_factory_rebuilds_on_reset() -> None:
    calls = []

    def model_factory(rng: np.random.Generator):
        calls.append(rng)
        scale = 0.5 if len(calls) == 1 else 2.0
        return get_mujoco_spec("octahedron", scale=scale, realistic=False)

    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            domain_randomization=DomainRandomizationConfig(model_factory=model_factory),
        )
    )
    try:
        env.reset(seed=1)
        small_extent = env.mj_model.model.stat.extent

        env.reset(seed=2)
        large_extent = env.mj_model.model.stat.extent

        assert len(calls) == 2
        assert large_extent > small_extent
    finally:
        env.close()


def test_env_runtime_domain_randomization_restores_nominals_between_resets() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            domain_randomization=DomainRandomizationConfig(
                body_mass_multiplier_range=(2.0, 2.0),
                body_inertia_multiplier_range=(3.0, 3.0),
                geom_friction_slide_range=(0.25, 0.25),
                gravity_z_range=(-4.0, -4.0),
            ),
        )
    )
    try:
        nominal_mass = env.mj_model.model.body_mass.copy()
        nominal_inertia = env.mj_model.model.body_inertia.copy()

        _, info = env.reset(seed=1)
        np.testing.assert_allclose(env.mj_model.model.body_mass, nominal_mass * 2.0)
        np.testing.assert_allclose(env.mj_model.model.body_inertia, nominal_inertia * 3.0)
        np.testing.assert_allclose(env.mj_model.model.geom_friction[:, 0], 0.25)
        assert env.mj_model.model.opt.gravity[2] == pytest.approx(-4.0)
        assert info["domain_randomization"]["body_mass_multiplier"] == pytest.approx(2.0)

        env.reset(seed=2)
        np.testing.assert_allclose(env.mj_model.model.body_mass, nominal_mass * 2.0)
        np.testing.assert_allclose(env.mj_model.model.body_inertia, nominal_inertia * 3.0)
    finally:
        env.close()


def test_preset_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="scale must be greater than zero"):
        get_preset_definition("octahedron", scale=0.0)

    with pytest.raises(ValueError, match="scale is only supported"):
        get_mujoco_spec({"node_1": [0.0, 0.0, 0.1]}, {}, scale=2.0)


def test_generated_world_uses_professional_scene_defaults() -> None:
    spec = get_mujoco_spec("tetrahedron", realistic=False)
    root = ET.fromstring(spec.to_xml())

    option = root.find("./option")
    assert option is not None
    assert option.get("integrator") == "implicitfast"
    assert spec.compile().opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    ground = root.find("./worldbody/geom[@name='ground']")
    assert ground is not None
    assert ground.get("type") == "plane"
    assert ground.get("material") == "ground_grid"

    ground_texture = root.find("./asset/texture[@name='ground_checker']")
    assert ground_texture is not None
    assert ground_texture.get("builtin") == "checker"

    skybox = root.find("./asset/texture[@name='skybox']")
    assert skybox is not None
    assert skybox.get("type") == "skybox"

    light_names = {light.get("name") for light in root.findall("./worldbody/light")}
    assert {"key", "fill"} <= light_names


def test_generated_spec_uses_firehose_steel_and_black_materials() -> None:
    root = ET.fromstring(get_mujoco_spec("octahedron", realistic=True).to_xml())

    firehose_material = root.find("./asset/material[@name='blue_firehose']")
    assert firehose_material is not None
    assert firehose_material.get("texture") is None
    np.testing.assert_allclose(
        _xml_vector(firehose_material.get("rgba", "")),
        [0.0, 0.1804, 0.3647, 1.0],
    )
    assert float(firehose_material.get("reflectance", "nan")) == pytest.approx(0.01)
    assert float(firehose_material.get("specular", "nan")) == pytest.approx(0.08)

    for tendon in root.findall(".//tendon/spatial"):
        if tendon.get("name", "").startswith("Perimeter_Constraint_"):
            assert tendon.get("material") is None
            assert float(tendon.get("width", "nan")) == pytest.approx(0.0001)
            np.testing.assert_allclose(
                _xml_vector(tendon.get("rgba", "")),
                [0.0, 0.0, 0.0, 0.0],
            )
        else:
            assert tendon.get("material") == "blue_firehose"

    rod_geom = next(
        body.find("./geom")
        for body in root.findall(".//body")
        if body.get("name", "").startswith("rod_")
    )
    assert rod_geom is not None
    assert rod_geom.get("material") == "connector_steel"

    node_geom = root.find(".//body[@name='node_1']/geom")
    assert node_geom is not None
    assert node_geom.get("material") == "node_black"
    np.testing.assert_allclose(
        _xml_vector(node_geom.get("rgba", "")),
        [0.18, 0.18, 0.18, 1.0],
    )


def test_realistic_spec_adds_accelerometers_to_each_generated_node() -> None:
    root = ET.fromstring(get_mujoco_spec("octahedron", realistic=True).to_xml())

    node_site_names = {
        site.get("name")
        for site in root.findall(".//body/site")
        if site.get("name", "").startswith("node_")
    }
    accelerometers = root.findall("./sensor/accelerometer")

    assert {sensor.get("site") for sensor in accelerometers} == node_site_names
    assert {sensor.get("name") for sensor in accelerometers} == {
        f"accel_{node_name}" for node_name in node_site_names
    }


def test_realistic_accelerometer_config_is_passed_to_mujoco() -> None:
    spec = get_mujoco_spec(
        "octahedron",
        realistic=True,
        accelerometer_config=AccelerometerConfig(
            noise=0.03,
            cutoff=25.0,
            nsample=3,
            delay=0.01,
            name_prefix="imu_accel",
        ),
    )
    root = ET.fromstring(spec.to_xml())
    sensor = root.find("./sensor/accelerometer")

    assert sensor is not None
    assert sensor.get("name", "").startswith("imu_accel_node_")
    assert float(sensor.get("noise", "nan")) == pytest.approx(0.03)
    assert float(sensor.get("cutoff", "nan")) == pytest.approx(25.0)
    assert int(sensor.get("nsample", "0")) == 3
    assert float(sensor.get("delay", "nan")) == pytest.approx(0.01)
    spec.compile()


def test_realistic_accelerometers_can_be_disabled() -> None:
    root = ET.fromstring(
        get_mujoco_spec("octahedron", realistic=True, accelerometer_config=None).to_xml()
    )

    assert root.find("./sensor/accelerometer") is None


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


def test_triangle_node_masses_follow_active_and_passive_roles() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    root = ET.fromstring(get_mujoco_spec(node_dict, triangle_dict, realistic=False).to_xml())

    passive_geom = root.find(".//body[@name='node_1']/geom")
    active_geom = root.find(".//body[@name='node_2']/geom")
    assert passive_geom is not None
    assert active_geom is not None
    assert float(passive_geom.get("mass", "nan")) == pytest.approx(PASSIVE_NODE_MASS)
    assert float(active_geom.get("mass", "nan")) == pytest.approx(ACTIVE_NODE_MASS)


def test_physical_parameters_override_generated_truss_values() -> None:
    params = TrussPhysicalParameters(
        node_radius=0.12,
        active_node_mass=0.33,
        passive_node_mass=0.44,
        abstract_actuator_kp=1234.0,
        actuator_dampratio=0.75,
        actuator_ctrl_range=[-0.2, 0.2],
        default_actuator_range=[0.1, 1.7],
        tendon_range_max_factor=3.0,
        edge_tendon_width=0.08,
        perimeter_constraint_tendon_width=0.002,
        tendon_constraint_solref=[0.03, 0.8],
        tendon_constraint_solimp=[0.85, 0.93, 0.002],
    )
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [0.8, 0.0, 0.2],
        "node_3": [0.4, 0.7, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    root = ET.fromstring(
        get_mujoco_spec(
            node_dict,
            triangle_dict,
            realistic=False,
            physical_params=params,
        ).to_xml()
    )

    passive_geom = root.find(".//body[@name='node_1']/geom")
    active_geom = root.find(".//body[@name='node_2']/geom")
    edge_tendon = root.find(".//tendon/spatial[@name='tendon_node_1_node_2']")
    perimeter_tendon = root.find(".//tendon/spatial[@name='Perimeter_Constraint_0']")
    actuator = root.find(".//actuator/general")
    constraint = root.find(".//equality/tendon[@name='Perimeter_Constraint_0']")

    assert passive_geom is not None
    assert active_geom is not None
    assert edge_tendon is not None
    assert perimeter_tendon is not None
    assert actuator is not None
    assert constraint is not None

    assert float(passive_geom.get("mass", "nan")) == pytest.approx(0.44)
    assert float(active_geom.get("mass", "nan")) == pytest.approx(0.33)
    np.testing.assert_allclose(_xml_vector(active_geom.get("size", "")), [0.12])
    np.testing.assert_allclose(_xml_vector(edge_tendon.get("range", "")), [0.5, 2.4])
    assert float(edge_tendon.get("width", "nan")) == pytest.approx(0.08)
    assert float(perimeter_tendon.get("width", "nan")) == pytest.approx(0.002)
    np.testing.assert_allclose(_xml_vector(actuator.get("ctrlrange", "")), [-0.2, 0.2])
    np.testing.assert_allclose(_xml_vector(actuator.get("actrange", "")), [0.1, 1.7])
    np.testing.assert_allclose(_xml_vector(actuator.get("gainprm", ""))[:1], [1234.0])
    expected_kv = 2.0 * params.actuator_dampratio * math.sqrt(
        params.abstract_actuator_kp * params.realistic_actuator_nominal_mass
    )
    np.testing.assert_allclose(
        _xml_vector(actuator.get("biasprm", ""))[1:3],
        [-1234.0, -expected_kv],
        rtol=1e-5,
    )
    np.testing.assert_allclose(_xml_vector(constraint.get("solref", "")), [0.03, 0.8])
    np.testing.assert_allclose(
        _xml_vector(constraint.get("solimp", ""))[:3],
        [0.85, 0.93, 0.002],
    )


def test_custom_triangle_nodes_are_lifted_above_ground() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, -0.8],
        "node_2": [0.8, 0.0, -0.4],
        "node_3": [0.4, 0.7, -0.6],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }

    for realistic in (False, True):
        env = MujocoTrussEnv(
            TrussEnvConfig(
                get_mujoco_spec(node_dict, triangle_dict, realistic=realistic),
                max_steps=1,
                nsubsteps=1,
            )
        )
        try:
            for seed in range(5):
                env.reset(seed=seed)
                node_z = env.mj_model.get_node_position_matrix()[:, 2]
                assert float(np.min(node_z)) >= NODE_RADIUS
        finally:
            env.close()


def test_custom_routed_shape_nodes_are_lifted_above_ground() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, -0.8],
        "node_2": [0.8, 0.0, -0.4],
        "node_3": [0.8, 0.8, -0.6],
        "node_4": [0.0, 0.8, -0.5],
    }
    shape_dict = {
        "quad_1": {
            "route": ["node_1", "node_2", "node_3", "node_4", "node_1"],
            "active_edges": [["node_1", "node_2"], ["node_4", "node_1"]],
        },
    }

    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec(node_dict, shape_dict, realistic=False),
            max_steps=1,
            nsubsteps=1,
        )
    )
    try:
        for seed in range(5):
            env.reset(seed=seed)
            node_z = env.mj_model.get_node_position_matrix()[:, 2]
            assert float(np.min(node_z)) >= NODE_RADIUS
    finally:
        env.close()


def test_builtin_presets_reset_with_nodes_above_ground() -> None:
    for preset_name in _default_compile_preset_names():
        realistic_values = (False, True) if preset_name != "tetrahedron" else (False,)
        for realistic in realistic_values:
            env = MujocoTrussEnv(
                TrussEnvConfig(
                    get_mujoco_spec(preset_name, realistic=realistic),
                    max_steps=1,
                    nsubsteps=1,
                )
            )
            try:
                for seed in range(10):
                    env.reset(seed=seed)
                    node_z = env.mj_model.get_node_position_matrix()[:, 2]
                    assert float(np.min(node_z)) >= NODE_RADIUS
            finally:
                env.close()


def test_realistic_octahedron_starts_above_collapse_threshold() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=True),
            max_steps=1,
            nsubsteps=1,
        )
    )
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, _, info = env.step(action)

        assert info["critical_eig"] > env.config.critical_eig_threshold
        assert not terminated
    finally:
        env.close()


def test_octahedron_stays_finite_and_above_ground_under_zero_action() -> None:
    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec("octahedron", realistic=False),
            max_steps=250,
            nsubsteps=1,
            speed=0.01,
        )
    )
    try:
        obs, _ = env.reset(seed=23)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        ground_contact_tolerance = 0.015

        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(action)
            node_z = env.mj_model.get_node_position_matrix()[:, 2]

            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)
            assert np.isfinite(info["critical_eig"])
            assert float(np.min(node_z)) >= NODE_RADIUS - ground_contact_tolerance
            assert not terminated
            assert not truncated
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


def test_realistic_angle_bisector_controller_aligns_connector_rods() -> None:
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

    env = MujocoTrussEnv(
        TrussEnvConfig(
            get_mujoco_spec(node_dict, triangle_dict, realistic=True),
            max_steps=2,
            nsubsteps=1,
            speed=0.01,
        )
    )
    try:
        env.reset(seed=17)
        controller = env.mj_model.angle_bisector_controller

        assert controller.enabled
        assert {target.node_name for target in controller.targets} == {
            "node_1",
            "node_2",
            "node_1_tri_triangle_2",
            "node_2_tri_triangle_2",
        }
        assert env.action_space.shape == (4,)
        assert env.mj_model.model.nu == 8
        assert all(
            name.startswith("bisector_act_") for name in env.mj_model.internal_actuator_names
        )
        assert not any(
            name.startswith("bisector_act_") for name in env.mj_model.external_actuator_names
        )

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        env.step(action)

        for target in controller.targets:
            node_pos = env.mj_model.data.site_xpos[target.node_site_id]
            neighbor_a = env.mj_model.data.site_xpos[target.neighbor_site_ids[0]]
            neighbor_b = env.mj_model.data.site_xpos[target.neighbor_site_ids[1]]
            dir_a = _unit(neighbor_a - node_pos)
            dir_b = _unit(neighbor_b - node_pos)
            bisector = _unit(dir_a + dir_b)

            tip_site_id = mujoco.mj_name2id(
                env.mj_model.model,
                mujoco.mjtObj.mjOBJ_SITE,
                f"tip_site_{target.node_name}",
            )
            rod_direction = _unit(env.mj_model.data.site_xpos[tip_site_id] - node_pos)

            assert float(np.dot(rod_direction, bisector)) == pytest.approx(-1.0, abs=1e-4)
    finally:
        env.close()


def test_realistic_node_box_face_normal_points_to_connector_ball() -> None:
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

    root = ET.fromstring(get_mujoco_spec(node_dict, triangle_dict, realistic=True).to_xml())

    for node_name in ("node_1", "node_2", "node_1_tri_triangle_2", "node_2_tri_triangle_2"):
        node_body = root.find(f".//body[@name='{node_name}']")
        assert node_body is not None

        box_geom = node_body.find("./geom[@type='box']")
        assert box_geom is not None
        face_normal = _quat_rotate_x(_xml_vector(box_geom.get("quat", "1 0 0 0")))

        tip_site = node_body.find(f"./body[@name='rod_{node_name}']/site")
        assert tip_site is not None
        connector_direction = _unit(_xml_vector(tip_site.get("pos", "")))

        assert float(np.dot(face_normal, connector_direction)) == pytest.approx(1.0)


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
    root = ET.fromstring(spec.to_xml())
    assert root.find(".//equality/tendon[@name='Route_Length_Constraint_quad_1']") is None
    actuator_names = {actuator.get("name") for actuator in root.findall(".//actuator/general")}
    assert actuator_names == {"act_12", "act_23", "act_34", "act_14"}

    env = MujocoTrussEnv(TrussEnvConfig(spec, max_steps=2, nsubsteps=1, speed=0.01))
    try:
        obs, _ = env.reset(seed=13)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (4,)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, _, _, _, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert "critical_eig" in info
    finally:
        env.close()


def test_realistic_routed_shape_clones_nodes_and_adds_bisector_controller() -> None:
    node_dict, shape_dict = get_preset_definition("tetrahedron")
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=True)
    root = ET.fromstring(spec.to_xml())

    route_sites = {
        spatial.get("name"): [site.get("site") for site in spatial.findall("site")]
        for spatial in root.findall("./tendon/spatial")
        if spatial.get("name", "").startswith("route_")
    }
    assert route_sites == {
        "route_path_1": ["node_1", "node_2", "node_4", "node_3"],
        "route_path_2": [
            "node_2_route_path_2_0",
            "node_3_route_path_2_1",
            "node_1_route_path_2_2",
            "node_4_route_path_2_3",
        ],
    }
    assert {
        body.get("name")
        for body in root.findall("./worldbody/body")
        if body.get("name", "").startswith("connector_ball_")
    } == {
        "connector_ball_node_1",
        "connector_ball_node_2",
        "connector_ball_node_3",
        "connector_ball_node_4",
    }

    model = MujocoModel(spec)
    controller = model.angle_bisector_controller
    assert controller.enabled
    assert {target.node_name for target in controller.targets} == {
        "node_1",
        "node_2",
        "node_3",
        "node_4",
        "node_2_route_path_2_0",
        "node_3_route_path_2_1",
        "node_1_route_path_2_2",
        "node_4_route_path_2_3",
    }
    assert len(model.external_actuator_names) == 6
    assert all(not name.startswith("bisector_act_") for name in model.external_actuator_names)
    for tendon_name, edge_length in model.get_edge_length_dict().items():
        if tendon_name.startswith("tendon_"):
            assert edge_length == pytest.approx(1.0, abs=0.05)


def test_realistic_routed_passive_cylinders_face_connector_rods() -> None:
    root = ET.fromstring(get_mujoco_spec("tetrahedron", realistic=True).to_xml())
    passive_nodes = set()
    for spatial in root.findall("./tendon/spatial"):
        if not spatial.get("name", "").startswith("route_"):
            continue
        sites = spatial.findall("site")
        passive_nodes.update((sites[0].get("site"), sites[-1].get("site")))
    routed_nodes = [
        body
        for body in root.findall("./worldbody/body")
        if body.get("name", "").startswith("node_")
        and body.find(f"./body[@name='rod_{body.get('name')}']") is not None
    ]
    assert routed_nodes

    for node_body in routed_nodes:
        node_name = node_body.get("name")
        assert node_name is not None

        hinge_joint = node_body.find(f"./joint[@name='{node_name}_z_hinge']")
        assert hinge_joint is not None
        assert float(hinge_joint.get("damping", "nan")) == pytest.approx(HINGE_DAMPING)
        angular_hinge = node_body.find(f"./joint[@name='{node_name}_angular_hinge']")
        roll_hinge = node_body.find(f"./joint[@name='{node_name}_roll_hinge']")

        rod_body = node_body.find(f"./body[@name='rod_{node_name}']")
        assert rod_body is not None
        assert rod_body.find("./joint") is None

        node_geom = node_body.find("./geom")
        assert node_geom is not None
        if node_name in passive_nodes:
            assert node_geom.get("type") == "cylinder"
            size = _xml_vector(node_geom.get("size", ""))
            assert float(size[0]) == pytest.approx(EDGE_TENDON_WIDTH)
            face_normal = _quat_rotate_z(_xml_vector(node_geom.get("quat", "1 0 0 0")))
        else:
            assert node_geom.get("type") == "box"
            face_normal = _quat_rotate_x(_xml_vector(node_geom.get("quat", "1 0 0 0")))

        tip_site = rod_body.find(f"./site[@name='tip_site_{node_name}']")
        assert tip_site is not None
        connector_direction = _unit(_xml_vector(tip_site.get("pos", "")))

        assert float(np.dot(face_normal, connector_direction)) == pytest.approx(
            1.0,
            abs=1e-5,
        )
        angular_actuator = root.find(
            f"./actuator/general[@name='bisector_angular_act_{node_name}']"
        )
        roll_actuator = root.find(
            f"./actuator/general[@name='bisector_roll_act_{node_name}']"
        )
        if node_name in passive_nodes:
            assert angular_hinge is not None
            assert float(angular_hinge.get("damping", "nan")) == pytest.approx(
                HINGE_DAMPING
            )
            assert angular_actuator is not None
            assert angular_actuator.get("joint") == f"{node_name}_angular_hinge"
            assert roll_hinge is None
            assert roll_actuator is None
        else:
            assert angular_hinge is not None
            assert float(angular_hinge.get("damping", "nan")) == pytest.approx(
                HINGE_DAMPING
            )
            assert angular_actuator is not None
            assert angular_actuator.get("joint") == f"{node_name}_angular_hinge"
            assert roll_hinge is not None
            assert float(roll_hinge.get("damping", "nan")) == pytest.approx(
                HINGE_DAMPING
            )
            assert roll_actuator is not None
            assert roll_actuator.get("joint") == f"{node_name}_roll_hinge"


def test_realistic_routed_connector_rods_start_on_angle_bisectors() -> None:
    node_dict, shape_dict = get_preset_definition("tetrahedron")
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=True)
    model = MujocoModel(spec)
    model.apply_angle_bisector_control()
    mujoco.mj_forward(model.model, model.data)

    for target in model.angle_bisector_controller.targets:
        node_pos = model.data.site_xpos[target.node_site_id]
        neighbor_site_ids = target.neighbor_site_ids
        if len(neighbor_site_ids) == 2 and len(target.neighbor_candidate_site_ids) >= 2:
            neighbor_site_ids = tuple(
                sorted(
                    target.neighbor_candidate_site_ids,
                    key=lambda site_id: float(
                        np.linalg.norm(model.data.site_xpos[site_id] - node_pos)
                    ),
                )[:2]
            )
        neighbor_positions = [
            model.data.site_xpos[site_id] for site_id in neighbor_site_ids
        ]
        if len(neighbor_positions) == 1:
            target_direction = _unit(node_pos - neighbor_positions[0])
            expected_dot = 1.0
        else:
            bisector = _unit(
                _unit(neighbor_positions[0] - node_pos) + _unit(neighbor_positions[1] - node_pos)
            )
            target_direction = -bisector
            expected_dot = 1.0
        tip_site_id = mujoco.mj_name2id(
            model.model,
            mujoco.mjtObj.mjOBJ_SITE,
            f"tip_site_{target.node_name}",
        )

        if target.angular_actuator_id is not None:
            parent_xmat = model.data.xmat[target.parent_body_id].reshape(3, 3)
            angle = float(model.data.ctrl[target.actuator_id])
            angular_angle = float(model.data.ctrl[target.angular_actuator_id])
            yawed_rod = _rotate_about_axis(
                target.initial_rod_vector,
                target.hinge_axis,
                angle,
            )
            yawed_angular_axis = _rotate_about_axis(
                target.angular_hinge_axis,
                target.hinge_axis,
                angle,
            )
            rod_direction = _unit(
                parent_xmat
                @ _rotate_about_axis(yawed_rod, yawed_angular_axis, angular_angle)
            )
            assert float(np.dot(rod_direction, target_direction)) == pytest.approx(
                1.0,
                abs=1e-4,
            )
        else:
            rod_direction = _unit(model.data.site_xpos[tip_site_id] - node_pos)
            planar_target = target_direction - target.hinge_axis * float(
                np.dot(target_direction, target.hinge_axis)
            )
            planar_target = _unit(planar_target)
            assert float(np.dot(rod_direction, planar_target)) == pytest.approx(
                expected_dot,
                abs=1e-4,
            )
            assert abs(float(np.dot(rod_direction, target.hinge_axis))) < 1e-6


def test_realistic_routed_plane_candidates_use_edge_tendon_duplicate_neighbors() -> None:
    node_dict, shape_dict = get_preset_definition("tetrahedron")
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=True)
    root = ET.fromstring(spec.to_xml())
    model = MujocoModel(spec)
    edge_neighbors = _edge_tendon_neighbors(root)

    checked_duplicate_neighbors = False
    for target in model.angle_bisector_controller.targets:
        if len(target.neighbor_candidate_site_ids) < 2:
            continue

        candidate_names = tuple(
            model.model.site(site_id).name for site_id in target.neighbor_candidate_site_ids
        )
        assert candidate_names == edge_neighbors[target.node_name]
        checked_duplicate_neighbors = checked_duplicate_neighbors or any(
            "_route_" in name for name in candidate_names
        )

    assert checked_duplicate_neighbors


def test_realistic_routed_active_rods_follow_live_nearest_neighbor_plane() -> None:
    node_dict, shape_dict = get_preset_definition("tetrahedron")
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=True)
    model = MujocoModel(spec)
    controller = model.angle_bisector_controller
    target = next(
        target
        for target in controller.targets
        if target.angular_actuator_id is not None
        and len(target.neighbor_site_ids) == 2
        and len(target.neighbor_candidate_site_ids) >= 2
    )

    moved_site_id = target.neighbor_candidate_site_ids[0]
    moved_node_name = model.model.site(moved_site_id).name
    slide_joint_id = mujoco.mj_name2id(
        model.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{moved_node_name}_z",
    )
    assert slide_joint_id >= 0

    model.data.qpos[int(model.model.jnt_qposadr[slide_joint_id])] += 0.35
    mujoco.mj_forward(model.model, model.data)
    controller.update(model.model, model.data)

    node_pos = model.data.site_xpos[target.node_site_id]
    nearest_site_ids = sorted(
        target.neighbor_candidate_site_ids,
        key=lambda site_id: float(np.linalg.norm(model.data.site_xpos[site_id] - node_pos)),
    )[:2]
    neighbor_a = model.data.site_xpos[nearest_site_ids[0]]
    neighbor_b = model.data.site_xpos[nearest_site_ids[1]]
    plane_normal = _unit(np.cross(neighbor_a - node_pos, neighbor_b - node_pos))

    parent_xmat = model.data.xmat[target.parent_body_id].reshape(3, 3)
    angle = float(model.data.ctrl[target.actuator_id])
    angular_angle = float(model.data.ctrl[target.angular_actuator_id])
    yawed_rod = _rotate_about_axis(target.initial_rod_vector, target.hinge_axis, angle)
    yawed_angular_axis = _rotate_about_axis(
        target.angular_hinge_axis,
        target.hinge_axis,
        angle,
    )
    rod_direction = _unit(
        parent_xmat @ _rotate_about_axis(yawed_rod, yawed_angular_axis, angular_angle)
    )

    assert abs(float(np.dot(rod_direction, plane_normal))) < 1e-6


def test_realistic_routed_roll_hinges_are_active_node_controlled_only() -> None:
    root = ET.fromstring(get_mujoco_spec("tetrahedron", realistic=True).to_xml())
    passive_nodes = set()
    for spatial in root.findall("./tendon/spatial"):
        if not spatial.get("name", "").startswith("route_"):
            continue
        sites = spatial.findall("site")
        passive_nodes.update((sites[0].get("site"), sites[-1].get("site")))

    for node_body in root.findall("./worldbody/body"):
        node_name = node_body.get("name")
        if not node_name or not node_name.startswith("node_"):
            continue
        if node_body.find(f"./body[@name='rod_{node_name}']") is None:
            continue

        roll_hinge = node_body.find(f"./joint[@name='{node_name}_roll_hinge']")
        roll_actuator = root.find(
            f"./actuator/general[@name='bisector_roll_act_{node_name}']"
        )
        if node_name in passive_nodes:
            assert roll_hinge is None
            assert roll_actuator is None
        else:
            assert roll_hinge is not None
            assert roll_actuator is not None
            assert roll_actuator.get("joint") == f"{node_name}_roll_hinge"


def test_realistic_routed_roll_hinge_tracks_live_nearest_neighbor_plane() -> None:
    node_dict, shape_dict = get_preset_definition("tetrahedron")
    spec = get_mujoco_spec(node_dict, shape_dict, realistic=True)
    model = MujocoModel(spec)
    controller = model.angle_bisector_controller
    target = next(
        target
        for target in controller.targets
        if target.roll_actuator_id is not None
        and target.roll_hinge_axis is not None
        and len(target.neighbor_site_ids) == 2
        and len(target.neighbor_candidate_site_ids) >= 2
    )

    moved_site_id = target.neighbor_candidate_site_ids[0]
    moved_node_name = model.model.site(moved_site_id).name
    slide_joint_id = mujoco.mj_name2id(
        model.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{moved_node_name}_z",
    )
    assert slide_joint_id >= 0

    model.data.qpos[int(model.model.jnt_qposadr[slide_joint_id])] += 0.35
    mujoco.mj_forward(model.model, model.data)
    controller.update(model.model, model.data)

    node_pos = model.data.site_xpos[target.node_site_id]
    nearest_site_ids = sorted(
        target.neighbor_candidate_site_ids,
        key=lambda site_id: float(np.linalg.norm(model.data.site_xpos[site_id] - node_pos)),
    )[:2]
    neighbor_a = model.data.site_xpos[nearest_site_ids[0]]
    neighbor_b = model.data.site_xpos[nearest_site_ids[1]]
    plane_normal = _unit(np.cross(neighbor_a - node_pos, neighbor_b - node_pos))

    parent_xmat = model.data.xmat[target.parent_body_id].reshape(3, 3)
    target_normal_parent = parent_xmat.T @ plane_normal
    angle = float(model.data.ctrl[target.actuator_id])
    angular_angle = float(model.data.ctrl[target.angular_actuator_id])
    roll_angle = float(model.data.ctrl[target.roll_actuator_id])
    yawed_angular_axis = _rotate_about_axis(
        target.angular_hinge_axis,
        target.hinge_axis,
        angle,
    )
    pitched_roll_axis = _rotate_about_axis(
        target.roll_hinge_axis,
        target.hinge_axis,
        angle,
    )
    pitched_roll_axis = _rotate_about_axis(
        pitched_roll_axis,
        yawed_angular_axis,
        angular_angle,
    )
    pitched_normal = _rotate_about_axis(
        target.hinge_axis,
        yawed_angular_axis,
        angular_angle,
    )
    rolled_normal = _unit(
        _rotate_about_axis(pitched_normal, pitched_roll_axis, roll_angle)
    )

    assert abs(float(np.dot(rolled_normal, target_normal_parent))) == pytest.approx(
        1.0,
        abs=1e-4,
    )


def test_node_velocity_controller_maps_node_commands_to_edge_commands() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [2.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_3"],
            "active_edges": [["node_1", "node_2"]],
        },
    }
    model = MujocoModel(get_mujoco_spec(node_dict, shape_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )

    np.testing.assert_array_equal(controller.passive_node_names, ["node_1", "node_3"])
    np.testing.assert_allclose(
        controller.incidence_matrix,
        np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]),
    )

    edge_commands = controller.transform(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(controller.latest_node_commands, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(edge_commands, [2.0, -2.0])


def test_node_velocity_controller_clips_edge_commands() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [2.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_3"],
            "active_edges": [["node_1", "node_2"]],
        },
    }
    model = MujocoModel(get_mujoco_spec(node_dict, shape_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )

    np.testing.assert_allclose(
        controller.clipped_edge_commands(model.model, np.array([0.0, 2.0, 0.0])),
        [ACTUATOR_CTRL_RANGE[1], ACTUATOR_CTRL_RANGE[0]],
    )


def test_node_velocity_controller_uses_first_conflicting_route_orientation() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2"],
            "active_edges": [["node_1", "node_2"]],
        },
        "path_2": {
            "route": ["node_2", "node_1"],
            "active_edges": [["node_2", "node_1"]],
        },
    }
    model = MujocoModel(get_mujoco_spec(node_dict, shape_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )

    assert [(edge.from_node, edge.to_node) for edge in controller.edges] == [("node_1", "node_2")]


def test_node_velocity_command_env_steps_with_node_actions() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [2.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_3"],
            "active_edges": [["node_1", "node_2"]],
        },
    }
    env = MujocoNodeVelocityCommandEnv(
        TrussEnvConfig(
            get_mujoco_spec(node_dict, shape_dict, realistic=False),
            max_steps=2,
            nsubsteps=1,
            speed=0.01,
        )
    )
    try:
        obs, _ = env.reset(seed=13)
        assert env.observation_space.contains(obs)
        assert env.action_space.shape == (3,)

        action = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        obs, _, _, _, info = env.step(action)

        assert env.observation_space.contains(obs)
        assert "critical_eig" in info
        np.testing.assert_allclose(
            env.node_velocity_controller.latest_node_commands,
            [0.0, 0.01, 0.0],
        )
        np.testing.assert_allclose(env.mj_model.get_external_ctrl(), [0.01, -0.01])
    finally:
        env.close()


def test_triangle_control_graph_matches_between_abstract_and_realistic_models() -> None:
    abstract_model = MujocoModel(get_mujoco_spec("octahedron", realistic=False))
    realistic_model = MujocoModel(get_mujoco_spec("octahedron", realistic=True))

    assert abstract_model.control_graph.control_node_names == (
        realistic_model.control_graph.control_node_names
    )
    assert [
        (edge.from_node, edge.to_node, edge.type)
        for edge in abstract_model.control_graph.edges
    ] == [
        (edge.from_node, edge.to_node, edge.type)
        for edge in realistic_model.control_graph.edges
    ]
    assert abstract_model.control_graph.passive_control_node_names == (
        realistic_model.control_graph.passive_control_node_names
    )
    assert [
        (edge.from_node, edge.to_node)
        for edge in abstract_model.control_graph.actuator_edges
    ] == [
        (edge.from_node, edge.to_node)
        for edge in realistic_model.control_graph.actuator_edges
    ]

    abstract_env = MujocoNodeVelocityCommandEnv(
        TrussEnvConfig(get_mujoco_spec("octahedron", realistic=False), max_steps=1)
    )
    realistic_env = MujocoNodeVelocityCommandEnv(
        TrussEnvConfig(get_mujoco_spec("octahedron", realistic=True), max_steps=1)
    )
    try:
        assert abstract_env.action_space.shape == realistic_env.action_space.shape
        assert abstract_env.observation_space.shape == realistic_env.observation_space.shape
    finally:
        abstract_env.close()
        realistic_env.close()


def test_routed_control_graph_matches_between_abstract_and_realistic_models() -> None:
    abstract_model = MujocoModel(get_mujoco_spec("tetrahedron", realistic=False))
    realistic_model = MujocoModel(get_mujoco_spec("tetrahedron", realistic=True))

    assert abstract_model.control_graph.control_node_names == (
        realistic_model.control_graph.control_node_names
    )
    assert [
        (edge.from_node, edge.to_node, edge.type)
        for edge in abstract_model.control_graph.edges
    ] == [
        (edge.from_node, edge.to_node, edge.type)
        for edge in realistic_model.control_graph.edges
    ]
    assert abstract_model.control_graph.passive_control_node_names == (
        realistic_model.control_graph.passive_control_node_names
    )
    assert [
        (edge.from_node, edge.to_node)
        for edge in abstract_model.control_graph.actuator_edges
    ] == [
        (edge.from_node, edge.to_node)
        for edge in realistic_model.control_graph.actuator_edges
    ]


def test_control_graph_maps_abstract_duplicates_to_shared_physical_nodes() -> None:
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
    abstract_model = MujocoModel(get_mujoco_spec(node_dict, triangle_dict, realistic=False))
    realistic_model = MujocoModel(get_mujoco_spec(node_dict, triangle_dict, realistic=True))

    assert abstract_model.control_graph.control_node_to_physical_node["node_1"] == "node_1"
    assert (
        abstract_model.control_graph.control_node_to_physical_node["node_1_tri_triangle_2"]
        == "node_1"
    )
    assert realistic_model.control_graph.control_node_to_physical_node["node_1"] == "node_1"
    assert (
        realistic_model.control_graph.control_node_to_physical_node["node_1_tri_triangle_2"]
        == "node_1_tri_triangle_2"
    )

    abstract_features = get_node_features(abstract_model, graph_view="control")
    np.testing.assert_allclose(abstract_features[0], abstract_features[3])


def test_control_graph_gnn_edges_include_connector_edge_types() -> None:
    model = MujocoModel(get_mujoco_spec("tetrahedron", realistic=True))

    edge_index = get_edge_index(model, graph_view="control")
    edge_types = get_edge_types(model, graph_view="control")

    assert edge_index.shape[1] == len(edge_types)
    assert {"actuated", "connector"} == set(edge_types.tolist())

    connector_pairs = {
        (edge.from_node, edge.to_node)
        for edge in model.control_graph.edges
        if edge.type == "connector"
    }
    assert ("node_1", "node_1_route_path_2_2") in connector_pairs
    assert ("node_4", "node_4_route_path_2_3") in connector_pairs


def test_get_networkx_graph_returns_named_control_graph_with_edge_types() -> None:
    graph = get_networkx_graph(get_mujoco_spec("tetrahedron", realistic=True))

    assert "node_1" in graph.nodes
    assert "node_1_route_path_2_2" in graph.nodes
    edge_types = {edge_data["type"] for _, _, edge_data in graph.edges(data=True)}
    assert edge_types == {"actuated", "connector"}
    assert graph.has_edge("node_1", "node_1_route_path_2_2")


def test_view_graph_renders_without_showing_window() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax, graph = view_graph(
        get_mujoco_spec("octahedron", realistic=False),
        graph_view="control",
        layout="physical",
        show=False,
    )

    try:
        assert graph.number_of_nodes() == 12
        assert ax.get_title() == "Control graph"
        assert fig.axes == [ax]
    finally:
        plt.close(fig)


def test_triangle_node_velocity_controller_uses_actuator_edges_only() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [0.5, 0.8, 0.2],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
    }
    model = MujocoModel(get_mujoco_spec(node_dict, triangle_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )

    assert controller.node_names == ["node_1", "node_2", "node_3"]
    assert controller.passive_node_names == ["node_1"]
    assert len(controller.edges) == len(model.control_graph.actuator_edges)
    assert len(controller.edges) < len(model.control_graph.edges)
    np.testing.assert_allclose(
        controller.incidence_matrix,
        np.array([[-1.0, 1.0, 0.0], [1.0, 0.0, -1.0]]),
    )

    edge_commands = controller.transform(np.array([1.0, 2.0, 3.0]))

    np.testing.assert_allclose(controller.latest_node_commands, [0.0, 2.0, 3.0])
    np.testing.assert_allclose(edge_commands, [2.0, -3.0])


def test_node_velocity_command_env_supports_triangle_and_routed_control_graphs() -> None:
    for preset_name in ("octahedron", "tetrahedron"):
        abstract_env = MujocoNodeVelocityCommandEnv(
            TrussEnvConfig(get_mujoco_spec(preset_name, realistic=False), max_steps=2)
        )
        realistic_env = MujocoNodeVelocityCommandEnv(
            TrussEnvConfig(get_mujoco_spec(preset_name, realistic=True), max_steps=2)
        )
        try:
            abstract_obs, _ = abstract_env.reset(seed=13)
            realistic_obs, _ = realistic_env.reset(seed=13)
            assert abstract_env.action_space.shape == realistic_env.action_space.shape
            assert abstract_env.observation_space.shape == realistic_env.observation_space.shape
            assert abstract_env.observation_space.contains(abstract_obs)
            assert realistic_env.observation_space.contains(realistic_obs)

            abstract_action = np.zeros(abstract_env.action_space.shape, dtype=np.float32)
            realistic_action = np.zeros(realistic_env.action_space.shape, dtype=np.float32)
            abstract_obs, _, _, _, abstract_info = abstract_env.step(abstract_action)
            realistic_obs, _, _, _, realistic_info = realistic_env.step(realistic_action)

            assert abstract_env.observation_space.contains(abstract_obs)
            assert realistic_env.observation_space.contains(realistic_obs)
            assert "critical_eig" in abstract_info
            assert "critical_eig" in realistic_info
        finally:
            abstract_env.close()
            realistic_env.close()


def test_node_velocity_viewer_state_tracks_sliders_and_tendon_readouts() -> None:
    state = NodeVelocityViewerState(
        ["node_1", "node_2"],
        ["tendon_node_1_node_2"],
        ["node_1"],
        speed=0.01,
    )

    state.set_node_command("node_1", 0.01)
    state.set_node_command("node_2", 0.02)
    state.set_edge_commands(np.array([0.015]))

    np.testing.assert_allclose(state.node_commands, [0.0, 0.01])
    np.testing.assert_allclose(state.edge_commands, [0.015])


def test_node_velocity_terminal_commands_update_node_commands() -> None:
    node_dict = {
        "node_1": [0.0, 0.0, 0.2],
        "node_2": [1.0, 0.0, 0.2],
        "node_3": [2.0, 0.0, 0.2],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_3"],
            "active_edges": [["node_1", "node_2"]],
        },
    }
    model = MujocoModel(get_mujoco_spec(node_dict, shape_dict, realistic=False))
    controller = NodeVelocityController(
        model.model,
        model.xml,
        model.node_names,
        model.site_to_node,
        model.external_actuator_ids,
    )
    node_commands = np.zeros(len(controller.node_names), dtype=float)

    assert not _apply_terminal_command("set node_2 0.02", controller, node_commands, 0.01)
    np.testing.assert_allclose(node_commands, [0.0, 0.01, 0.0])
    np.testing.assert_allclose(controller.latest_edge_commands, [0.01, -0.01])

    assert not _apply_terminal_command("add 1 -0.005", controller, node_commands, 0.01)
    np.testing.assert_allclose(node_commands, [0.0, 0.005, 0.0])

    assert not _apply_terminal_command("set node_1 0.01", controller, node_commands, 0.01)
    np.testing.assert_allclose(node_commands, [0.0, 0.005, 0.0])

    assert not _apply_terminal_command("zero", controller, node_commands, 0.01)
    np.testing.assert_allclose(node_commands, [0.0, 0.0, 0.0])
    assert _apply_terminal_command("quit", controller, node_commands, 0.01)


def test_realistic_logical_gnn_edge_index_matches_abstract_graph() -> None:
    abstract_model = get_mujoco_spec("octahedron", realistic=False)
    realistic_model = get_mujoco_spec("octahedron", realistic=True)

    abstract_edge_index = get_edge_index(abstract_model)
    realistic_edge_index = get_edge_index(realistic_model, graph_view="logical")

    assert realistic_edge_index.shape == abstract_edge_index.shape
    assert realistic_edge_index.shape == (2, 24)
    assert (
        int(np.max(realistic_edge_index))
        < get_node_features(
            realistic_model,
            graph_view="logical",
        ).shape[0]
    )


def test_realistic_logical_gnn_node_features_support_mean_aggregation() -> None:
    model = MujocoModel(get_mujoco_spec("octahedron", realistic=True))

    logical_features = get_node_features(model, graph_view="logical", aggregation="mean")
    physical_positions = model.get_node_position_dict()
    physical_velocities = model.get_node_velocity_linear_dict()
    node_1_instances = [
        node_name
        for node_name in model.node_names
        if node_name == "node_1" or node_name.startswith("node_1_tri_")
    ]
    expected_node_1 = np.concatenate(
        [
            np.mean([physical_positions[node_name] for node_name in node_1_instances], axis=0),
            np.mean([physical_velocities[node_name] for node_name in node_1_instances], axis=0),
        ]
    )

    assert logical_features.shape == (6, 6)
    np.testing.assert_allclose(logical_features[0], expected_node_1)


def test_realistic_logical_gnn_node_features_support_connector_ball_aggregation() -> None:
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
    model = MujocoModel(get_mujoco_spec(node_dict, triangle_dict, realistic=True))

    logical_features = get_node_features(
        model,
        graph_view="logical",
        aggregation="connector_ball",
    )
    connector_ball_id = mujoco.mj_name2id(
        model.model,
        mujoco.mjtObj.mjOBJ_BODY,
        "connector_ball_node_1",
    )
    node_3_id = model.node_body_ids["node_3"]
    expected_node_1 = np.concatenate(
        [model.data.xpos[connector_ball_id], model.data.cvel[connector_ball_id][3:]]
    )
    expected_node_3 = np.concatenate([model.data.xpos[node_3_id], model.data.cvel[node_3_id][3:]])

    assert logical_features.shape == (4, 6)
    np.testing.assert_allclose(logical_features[0], expected_node_1)
    np.testing.assert_allclose(logical_features[2], expected_node_3)


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

    get_mujoco_spec(node_dict, shape_dict, realistic=True).compile()

    assert node_dict == original_nodes
    assert shape_dict == original_shapes


def test_tetrahedron_routed_shape_has_no_route_constraints() -> None:
    spec = get_mujoco_spec("tetrahedron", realistic=False)
    model = spec.compile()
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    assert data.nefc == 0


def test_actuator_names_are_edge_based() -> None:
    model = get_mujoco_spec("tetrahedron", realistic=False).compile()

    actuator_names = {model.actuator(index).name for index in range(model.nu)}

    assert actuator_names == {"act_12", "act_24", "act_34", "act_23", "act_13", "act_14"}


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    return (
        vector * math.cos(angle)
        + np.cross(axis, vector) * math.sin(angle)
        + axis * float(np.dot(axis, vector)) * (1.0 - math.cos(angle))
    )


def _edge_tendon_neighbors(root: ET.Element) -> dict[str, tuple[str, ...]]:
    neighbors: dict[str, list[str]] = {}
    for spatial in root.findall("./tendon/spatial"):
        if not spatial.get("name", "").startswith("tendon_"):
            continue
        sites = [
            site_ref.get("site")
            for site_ref in spatial.findall("site")
            if site_ref.get("site")
        ]
        if len(sites) != 2:
            continue
        site_a, site_b = sites
        neighbors.setdefault(site_a, [])
        neighbors.setdefault(site_b, [])
        if site_b not in neighbors[site_a]:
            neighbors[site_a].append(site_b)
        if site_a not in neighbors[site_b]:
            neighbors[site_b].append(site_a)
    return {node_name: tuple(node_neighbors) for node_name, node_neighbors in neighbors.items()}


def _xml_vector(value: str) -> np.ndarray:
    return np.fromstring(value, sep=" ", dtype=float)


def _quat_rotate_x(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
    )


def _quat_rotate_z(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]
    )
