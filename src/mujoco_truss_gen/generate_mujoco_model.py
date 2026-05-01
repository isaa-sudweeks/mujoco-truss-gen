from __future__ import annotations

from mujoco_truss_gen.mujoco_model.bodies import (
    add_free_node_body,
    add_planar_node_body,
    add_slide_node_body,
    build_world,
    connect_node_to_ball,
    create_connector_balls,
    create_node_bodies,
    create_triangle_bodies,
    disable_geom_contacts,
    find_original_node,
)
from mujoco_truss_gen.mujoco_model.builders import (
    build_abstract_triangle,
    build_realistic_triangle,
    build_triangle,
    clone_shared_nodes,
    get_mujoco_spec,
)
from mujoco_truss_gen.mujoco_model.constants import (
    CONNECTOR_MASS,
    CONNECTOR_RADIUS,
    NODE_MASS,
    NODE_RADIUS,
    ROD_MASS,
    ROD_RADIUS,
    TENDON_RGBA,
    TRUSS_RGBA,
)
from mujoco_truss_gen.mujoco_model.constraints import (
    add_orientation_constraints,
    add_perimeter_constraint,
)
from mujoco_truss_gen.mujoco_model.geometry import as_mujoco_quat, get_perimeter, triangle_frame
from mujoco_truss_gen.mujoco_model.io_viewer import save_xml, view
from mujoco_truss_gen.mujoco_model.model_types import (
    EdgeKey,
    EdgeTendonMap,
    NodeDict,
    TriangleDict,
    Vector,
)
from mujoco_truss_gen.mujoco_model.presets import (
    PRESETS,
    get_icosahedron_definition,
    get_octahedron_definition,
    get_preset_definition,
)
from mujoco_truss_gen.mujoco_model.tendons import (
    add_actuator,
    add_edge_tendon,
    add_realistic_actuator,
    add_tendon,
    edge_key,
    initialize_actuator_lengths,
)


def main() -> None:
    node_dict, triangle_dict = get_octahedron_definition()
    spec = build_world()
    build_triangle(spec, node_dict, triangle_dict, realistic=False)

    for name, perimeter in get_perimeter(node_dict, triangle_dict).items():
        print(f"{name}: perimeter = {perimeter:.4f}")

    view(spec)


# Backwards-compatible aliases for existing local scripts.
_as_mujoco_quat = as_mujoco_quat
_triangle_frame = triangle_frame
_find_original_node = find_original_node
_disable_geom_contacts = disable_geom_contacts
_add_planar_node_body = add_planar_node_body
_add_free_node_body = add_free_node_body
_add_slide_node_body = add_slide_node_body
_create_connector_balls = create_connector_balls
_connect_node_to_ball = connect_node_to_ball
_edge_key = edge_key
connect_triangeles = create_triangle_bodies
add_perimeter_constrait = add_perimeter_constraint


__all__ = [
    "CONNECTOR_MASS",
    "CONNECTOR_RADIUS",
    "EdgeKey",
    "EdgeTendonMap",
    "NODE_MASS",
    "NODE_RADIUS",
    "NodeDict",
    "PRESETS",
    "ROD_MASS",
    "ROD_RADIUS",
    "TENDON_RGBA",
    "TRUSS_RGBA",
    "TriangleDict",
    "Vector",
    "add_actuator",
    "add_edge_tendon",
    "add_orientation_constraints",
    "add_perimeter_constraint",
    "add_perimeter_constrait",
    "add_realistic_actuator",
    "add_tendon",
    "build_abstract_triangle",
    "build_realistic_triangle",
    "build_triangle",
    "build_world",
    "clone_shared_nodes",
    "connect_triangeles",
    "create_node_bodies",
    "create_triangle_bodies",
    "get_icosahedron_definition",
    "get_mujoco_spec",
    "get_octahedron_definition",
    "get_perimeter",
    "get_preset_definition",
    "initialize_actuator_lengths",
    "main",
    "save_xml",
    "view",
]


if __name__ == "__main__":
    main()
