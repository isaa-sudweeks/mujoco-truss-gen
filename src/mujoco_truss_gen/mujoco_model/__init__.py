from __future__ import annotations

from mujoco_truss_gen.mujoco_model.bodies import build_world, create_node_bodies, create_triangle_bodies
from mujoco_truss_gen.mujoco_model.builders import (
    build_abstract_triangle,
    build_realistic_triangle,
    build_triangle,
    get_mujoco_spec,
)
from mujoco_truss_gen.mujoco_model.geometry import get_perimeter
from mujoco_truss_gen.mujoco_model.io_viewer import save_xml, view
from mujoco_truss_gen.mujoco_model.model import MujocoModel
from mujoco_truss_gen.mujoco_model.presets import get_octahedron_definition

__all__ = [
    "MujocoModel",
    "build_abstract_triangle",
    "build_realistic_triangle",
    "build_triangle",
    "build_world",
    "create_node_bodies",
    "create_triangle_bodies",
    "get_mujoco_spec",
    "get_octahedron_definition",
    "get_perimeter",
    "save_xml",
    "view",
]
