from __future__ import annotations

from mujoco_truss_gen.mujoco_model.bodies import (
    build_world,
    create_node_bodies,
    create_triangle_bodies,
)
from mujoco_truss_gen.mujoco_model.builders import (
    build_abstract_shapes,
    build_abstract_triangle,
    build_realistic_triangle,
    build_shapes,
    build_triangle,
    get_mujoco_spec,
)
from mujoco_truss_gen.mujoco_model.geometry import get_perimeter, get_route_lengths
from mujoco_truss_gen.mujoco_model.gnn_utilities import get_edge_index, get_node_features
from mujoco_truss_gen.mujoco_model.io_viewer import save_xml, view
from mujoco_truss_gen.mujoco_model.model import MujocoModel
from mujoco_truss_gen.mujoco_model.presets import (
    PRESETS,
    get_icosahedron_definition,
    get_octahedron_definition,
    get_preset_definition,
)

__all__ = [
    "MujocoModel",
    "PRESETS",
    "build_abstract_shapes",
    "build_abstract_triangle",
    "build_realistic_triangle",
    "build_shapes",
    "build_triangle",
    "build_world",
    "create_node_bodies",
    "create_triangle_bodies",
    "get_edge_index",
    "get_icosahedron_definition",
    "get_mujoco_spec",
    "get_node_features",
    "get_octahedron_definition",
    "get_perimeter",
    "get_preset_definition",
    "get_route_lengths",
    "save_xml",
    "view",
]
