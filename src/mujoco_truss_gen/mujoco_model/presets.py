from __future__ import annotations

from mujoco_truss_gen.mujoco_model.model_types import NodeDict, TriangleDict


def get_octahedron_definition() -> tuple[NodeDict, TriangleDict]:
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
    return node_dict, triangle_dict
