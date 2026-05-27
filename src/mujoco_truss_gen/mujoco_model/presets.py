from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy

import numpy as np

from mujoco_truss_gen.mujoco_model.model_types import NodeDict, ShapeDict, TriangleDict


def _validate_scale(scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be greater than zero.")
    return scale


def _scale_node_dict(node_dict: NodeDict, scale: float) -> NodeDict:
    scale = _validate_scale(scale)
    return {
        node_name: [scale * coordinate for coordinate in position]
        for node_name, position in node_dict.items()
    }


def get_octahedron_definition(scale: float = 1.0) -> tuple[NodeDict, TriangleDict]:
    node_dict = {
        "node_1": [0.0, 0.0, 0.1],
        "node_2": [1.0, 0.0, 0.1],
        "node_3": [0.5, 0.8660, 0.1],
        "node_4": [0.5, -0.2887, 0.9165],
        "node_5": [0.0, 0.5774, 0.9165],
        "node_6": [1.0, 0.5774, 0.9165],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_4", "node_1"],
        "triangle_2": ["node_1", "node_5", "node_3", "node_1"],
        "triangle_3": ["node_3", "node_6", "node_2", "node_6"],
        "triangle_4": ["node_4", "node_6", "node_5", "node_6"],
    }
    return _scale_node_dict(node_dict, scale), triangle_dict


def get_icosahedron_definition(scale: float = 1.0) -> tuple[NodeDict, TriangleDict]:
    phi = (1.0 + 5.0**0.5) / 2.0
    vertex_scale = 0.5
    z_offset = 0.95
    vertices = [
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    node_dict = {
        f"node_{index}": [vertex_scale * x, vertex_scale * y, vertex_scale * z + z_offset]
        for index, (x, y, z) in enumerate(vertices, start=1)
    }
    triangle_dict = {
        f"triangle_{index}": [
            *(f"node_{vertex_index + 1}" for vertex_index in face),
            f"node_{face[0] + 1}",
        ]
        for index, face in enumerate(faces, start=1)
    }

    return _scale_node_dict(node_dict, scale), triangle_dict


def get_solar_array_definition(scale: float = 1.0) -> tuple[NodeDict, TriangleDict]:
    node_dict = {
        "node_1": [0.0, 0.0, 0.1],
        "node_2": [1.0, 0.0, 0.1],
        "node_3": [0.5, 0.8660, 0.1],
        "node_4": [0.5, -0.2887, 0.9165],
        "node_5": [0.0, 0.5774, 0.9165],
        "node_6": [1.0, 0.5774, 0.9165],
        "node_7": [0.0, 0.0, 1.8330],
        "node_8": [1.0, 0.0, 1.8330],
        "node_9": [0.5, 0.8660, 1.8330],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_4", "node_1"],
        "triangle_2": ["node_1", "node_5", "node_3", "node_1"],
        "triangle_3": ["node_3", "node_6", "node_2", "node_6"],
        "triangle_4": ["node_4", "node_6", "node_8", "node_8"],
        "triangle_5": ["node_5", "node_6", "node_9", "node_9"],
        "triangle_6": ["node_5", "node_4", "node_7", "node_7"],
        "triangle_7": ["node_7", "node_8", "node_9", "node_9"],
    }
    return _scale_node_dict(node_dict, scale), triangle_dict


def get_tetrahedron_definition(scale: float = 1.0) -> tuple[NodeDict, ShapeDict]:
    node_dict = {
        "node_1": [0.0, 0.0, 0.1],
        "node_2": [1.0, 0.0, 0.1],
        "node_3": [0.5, 0.8660, 0.1],
        "node_4": [0.5, np.sqrt(3) / 6, 0.1 + np.sqrt(2 / 3)],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_4", "node_3"],
            "active_edges": [
                ["node_1", "node_2"],
                ["node_4", "node_3"],
            ],
        },
        "path_2": {
            "route": ["node_2", "node_3", "node_1", "node_4"],
            "active_edges": [
                ["node_2", "node_3"],
                ["node_1", "node_4"],
            ],
        },
    }

    return _scale_node_dict(node_dict, scale), shape_dict


PRESETS: dict[str, Callable[[float], tuple[NodeDict, TriangleDict | ShapeDict]]] = {
    "octahedron": get_octahedron_definition,
    "icosahedron": get_icosahedron_definition,
    "solar_array": get_solar_array_definition,
    "tetrahedron": get_tetrahedron_definition,
}


def get_preset_definition(
    structure_type: str,
    scale: float = 1.0,
) -> tuple[NodeDict, TriangleDict | ShapeDict]:
    try:
        preset = PRESETS[structure_type]
    except KeyError as exc:
        known_presets = ", ".join(sorted(PRESETS))
        message = f"Unknown structure type: {structure_type}. Known presets: {known_presets}"
        raise ValueError(message) from exc

    node_dict, structure_dict = preset(scale)
    return node_dict, deepcopy(structure_dict)
