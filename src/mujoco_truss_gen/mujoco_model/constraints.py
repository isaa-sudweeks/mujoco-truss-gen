from __future__ import annotations

import mujoco

from mujoco_truss_gen.mujoco_model.constants import (
    PERIMETER_CONSTRAINT_TENDON_RGBA,
    ROUTE_CONSTRAINT_DATA,
    TENDON_CONSTRAINT_DATA,
    TrussPhysicalParameters,
)
from mujoco_truss_gen.mujoco_model.model_types import (
    EdgeTendonMap,
    NodeDict,
    ShapeDict,
    TriangleDict,
)
from mujoco_truss_gen.mujoco_model.tendons import edge_key


def add_orientation_constraints(
    spec: mujoco.MjSpec,
    node_dict: NodeDict | TriangleDict,
    maybe_node_dict: NodeDict | None = None,
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    """Add soft weld constraints between each node and its initial world pose."""
    params = physical_params or TrussPhysicalParameters()
    if maybe_node_dict is not None:
        node_dict = maybe_node_dict

    for node in node_dict:
        weld = spec.add_equality(
            name=f"weld_world_{node}",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
        )
        weld.name1 = node
        weld.solref = params.orientation_weld_solref
        weld.solimp = params.orientation_weld_solimp
        weld.data[10] = params.orientation_weld_torquescale


def add_perimeter_constraint(
    spec: mujoco.MjSpec,
    triangle_dict: TriangleDict,
    edge_tendons: EdgeTendonMap | None = None,
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    params = physical_params or TrussPhysicalParameters()
    for index, triangle_nodes in enumerate(triangle_dict.values()):
        nodes = triangle_nodes[:3]
        passive_node = triangle_nodes[3]
        passive_index = nodes.index(passive_node)
        active_nodes = [node for node in nodes if node != passive_node]

        edge_from = nodes[(passive_index + 1) % 3]
        edge_to = nodes[(passive_index + 2) % 3]
        if edge_tendons is None:
            passive_edge_tendon = f"tendon_{edge_from}_{edge_to}"
        else:
            passive_edge_tendon = edge_tendons[edge_key(edge_from, edge_to)]

        tendon_name = f"Perimeter_Constraint_{index}"
        tendon = spec.add_tendon(
            name=tendon_name,
            width=params.perimeter_constraint_tendon_width,
            rgba=PERIMETER_CONSTRAINT_TENDON_RGBA,
        )
        tendon.wrap_site(active_nodes[0])
        tendon.wrap_site(passive_node)
        tendon.wrap_site(active_nodes[1])

        constraint = spec.add_equality(name=tendon_name, type=mujoco.mjtEq.mjEQ_TENDON)
        constraint.name1 = passive_edge_tendon
        constraint.name2 = tendon_name
        constraint.data[:5] = TENDON_CONSTRAINT_DATA
        constraint.solref = params.tendon_constraint_solref
        constraint.solimp[:3] = params.tendon_constraint_solimp


def add_route_length_constraints(
    spec: mujoco.MjSpec,
    shape_dict: ShapeDict,
    route_tendons: dict[str, str],
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    params = physical_params or TrussPhysicalParameters()
    for shape_name, shape in shape_dict.items():
        if shape.get("disable_route_length_constraint", False):
            continue
        tendon_name = route_tendons[shape_name]
        constraint = spec.add_equality(
            name=f"Route_Length_Constraint_{shape_name}",
            type=mujoco.mjtEq.mjEQ_TENDON,
        )
        constraint.name1 = tendon_name
        constraint.data[:5] = ROUTE_CONSTRAINT_DATA
        constraint.solref = params.tendon_constraint_solref
        constraint.solimp[:3] = params.tendon_constraint_solimp
