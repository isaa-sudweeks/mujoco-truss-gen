from __future__ import annotations

import sys
from io import StringIO
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from mujoco_truss_gen import get_mujoco_spec, stl_to_shape_dict
from mujoco_truss_gen.mesh_import import stl as stl_import


def _install_fake_trimesh(monkeypatch: pytest.MonkeyPatch, mesh: object) -> None:
    fake_trimesh = SimpleNamespace(load_mesh=lambda filename: mesh)
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)


def test_stl_to_shape_dict_returns_routed_shape_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.1],
                [1.0, 1.0, 0.1],
            ]
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [1, 3, 2],
            ]
        ),
    )
    _install_fake_trimesh(monkeypatch, mesh)

    node_dict, shape_dict = stl_to_shape_dict("two_triangles.stl")

    assert list(node_dict) == ["node_1", "node_2", "node_3", "node_4"]
    assert all(isinstance(position, list) and len(position) == 3 for position in node_dict.values())
    assert shape_dict

    route_edges = set()
    active_edges = set()
    for shape in shape_dict.values():
        route = shape["route"]
        assert len(route) >= 2
        for edge in zip(route, route[1:], strict=False):
            route_edges.add(tuple(edge))
        for edge in shape["active_edges"]:
            active_edges.add(tuple(edge))

    assert active_edges == route_edges
    get_mujoco_spec(node_dict, shape_dict, realistic=False).compile()


def test_stl_to_shape_dict_reports_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.1],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )
    _install_fake_trimesh(monkeypatch, mesh)
    stream = StringIO()

    stl_to_shape_dict("progress.stl", progress_stream=stream)

    output = stream.getvalue()
    assert "Loading STL mesh" in output
    assert "Loaded mesh with 3 vertices and 1 faces" in output
    assert "Extracted 3 unique graph edges" in output
    assert "Generated" in output


def test_stl_to_shape_dict_can_disable_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.1],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )
    _install_fake_trimesh(monkeypatch, mesh)
    stream = StringIO()

    stl_to_shape_dict("quiet.stl", verbose=False, progress_stream=stream)

    assert stream.getvalue() == ""


def test_stl_to_shape_dict_can_open_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.1],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )
    _install_fake_trimesh(monkeypatch, mesh)
    preview_calls = []

    def record_preview(*args: Any, block: bool) -> None:
        preview_calls.append((args, block))

    monkeypatch.setattr(stl_import, "_preview_routed_graph", record_preview)

    stl_to_shape_dict("preview.stl", preview=True, preview_block=False)

    assert len(preview_calls) == 1
    assert preview_calls[0][1] is False


def test_stl_to_shape_dict_skips_preview_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.1],
            ]
        ),
        faces=np.array([[0, 1, 2]]),
    )
    _install_fake_trimesh(monkeypatch, mesh)

    def fail_preview(*args: Any, block: bool) -> None:
        raise AssertionError("preview should be disabled by default")

    monkeypatch.setattr(stl_import, "_preview_routed_graph", fail_preview)

    stl_to_shape_dict("no_preview.stl")


def test_preview_uses_external_process_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    commands = []

    monkeypatch.setattr(stl_import, "_requires_external_preview_process", lambda: True)
    monkeypatch.setattr(stl_import, "_preview_python_executable", lambda: "/usr/bin/python3")
    monkeypatch.setattr(stl_import, "_write_preview_payload", lambda *args, block: "payload.npz")
    monkeypatch.setattr(
        stl_import.subprocess,
        "run",
        lambda command, check: commands.append(command),
    )

    stl_import._preview_routed_graph(
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
        {(0, 1)},
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
        {(0, 1)},
        {"node_1": [0.0, 0.0, 0.0], "node_2": [1.0, 0.0, 0.0]},
        {"path_1": {"route": ["node_1", "node_2"], "active_edges": [["node_1", "node_2"]]}},
        block=True,
    )

    assert len(commands) == 1
    assert commands[0][0] == "/usr/bin/python3"
    assert commands[0][-1] == "payload.npz"


def test_preview_payload_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = []

    monkeypatch.setattr(
        stl_import,
        "_render_routed_graph_preview",
        lambda *args, block: rendered.append((args, block)),
    )

    payload_path = stl_import._write_preview_payload(
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
        {(0, 1)},
        [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])],
        {(0, 1)},
        {"node_1": [0.0, 0.0, 0.0], "node_2": [1.0, 0.0, 0.0]},
        {"path_1": {"route": ["node_1", "node_2"], "active_edges": [["node_1", "node_2"]]}},
        block=False,
    )

    stl_import._render_preview_payload(payload_path)

    assert not payload_path.exists()
    assert len(rendered) == 1
    assert rendered[0][1] is False


def test_routed_paths_are_balanced_by_length() -> None:
    vertices = [
        np.array([float(index), 0.0, 0.0]) for index in range(10)
    ]
    vertices.extend([np.array([4.0, 1.0, 0.0]), np.array([4.0, -1.0, 0.0])])
    edges = {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (4, 10),
        (4, 11),
    }

    trails = stl_import._decompose_component_edges(edges, vertices)
    lengths = [
        sum(
            stl_import._edge_length(stl_import._edge_key(first, second), vertices)
            for first, second in zip(trail, trail[1:], strict=False)
        )
        for trail in trails
    ]

    assert len(trails) == 2
    assert max(lengths) - min(lengths) <= 1.0


def test_stl_to_shape_dict_applies_scale_offset_and_vertex_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.001, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        faces=np.array([[0, 2, 3], [1, 2, 3]]),
    )
    _install_fake_trimesh(monkeypatch, mesh)

    node_dict, shape_dict = stl_to_shape_dict(
        "merged.stl",
        merge_tolerance=0.01,
        scale=2.0,
        offset=[0.0, 0.0, 0.1],
    )

    assert len(node_dict) == 3
    assert node_dict["node_1"] == [0.0, 0.0, 0.1]
    assert shape_dict


def test_stl_to_shape_dict_target_node_count_decimates_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = SimpleNamespace(
        vertices=np.array(
            [
                [0.0, 0.0, 0.1],
                [0.1, 0.0, 0.1],
                [1.0, 0.0, 0.1],
                [0.0, 1.0, 0.1],
            ]
        ),
        faces=np.array([[0, 1, 3], [1, 2, 3]]),
    )
    _install_fake_trimesh(monkeypatch, mesh)

    node_dict, shape_dict = stl_to_shape_dict("decimated.stl", target_node_count=3)

    assert len(node_dict) == 3
    assert shape_dict


def test_target_node_count_preserves_spatial_extent(monkeypatch: pytest.MonkeyPatch) -> None:
    vertices = np.array(
        [
            [np.cos(angle), np.sin(angle), 0.0]
            for angle in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
        ]
    )
    faces = np.array([[0, index, index + 1] for index in range(1, 31)])
    _install_fake_trimesh(monkeypatch, SimpleNamespace(vertices=vertices, faces=faces))

    node_dict, shape_dict = stl_to_shape_dict("ring.stl", target_node_count=8)
    simplified = np.array(list(node_dict.values()))
    radii = np.linalg.norm(simplified[:, :2], axis=1)

    assert len(node_dict) == 8
    assert np.min(radii) > 0.9
    assert shape_dict


def test_stl_to_shape_dict_missing_trimesh_reports_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "trimesh", None)

    with pytest.raises(ImportError, match='pip install "mujoco-truss-gen\\[mesh\\]"'):
        stl_to_shape_dict("missing_dependency.stl")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"target_edge_length": 0.1, "target_node_count": 10}, "either target_edge_length"),
        ({"target_edge_length": 0.0}, "target_edge_length"),
        ({"target_node_count": 1}, "target_node_count"),
        ({"scale": 0.0}, "scale"),
        ({"merge_tolerance": -1.0}, "merge_tolerance"),
    ],
)
def test_stl_to_shape_dict_rejects_invalid_options(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    match: str,
) -> None:
    mesh = SimpleNamespace(vertices=np.zeros((3, 3)), faces=[[0, 1, 2]])
    _install_fake_trimesh(monkeypatch, mesh)

    with pytest.raises(ValueError, match=match):
        stl_to_shape_dict("invalid.stl", **kwargs)


@pytest.mark.parametrize(
    "mesh, match",
    [
        (SimpleNamespace(vertices=np.empty((0, 3)), faces=np.empty((0, 3))), "vertices"),
        (SimpleNamespace(vertices=np.zeros((3, 3)), faces=np.empty((0, 3))), "faces"),
    ],
)
def test_stl_to_shape_dict_rejects_empty_meshes(
    monkeypatch: pytest.MonkeyPatch,
    mesh: object,
    match: str,
) -> None:
    _install_fake_trimesh(monkeypatch, mesh)

    with pytest.raises(ValueError, match=match):
        stl_to_shape_dict("empty.stl")
