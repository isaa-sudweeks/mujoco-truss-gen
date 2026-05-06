from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.bodies import (
    build_world,
    create_node_bodies,
    create_triangle_bodies,
)
from mujoco_truss_gen.mujoco_model.constraints import (
    add_perimeter_constraint,
    add_route_length_constraints,
)
from mujoco_truss_gen.mujoco_model.model_types import (
    EdgeTendonMap,
    NodeDict,
    ShapeDict,
    TriangleDict,
)
from mujoco_truss_gen.mujoco_model.presets import get_preset_definition
from mujoco_truss_gen.mujoco_model.tendons import (
    add_actuator,
    add_edge_tendon,
    add_realistic_actuator,
    add_route_tendon,
    add_tendon,
    edge_key,
)


def clone_shared_nodes(
    node_dict: NodeDict,
    triangle_dict: TriangleDict,
    clone_offset: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], np.ndarray, float]:
    """Clone shared nodes so each triangle has independent planar node bodies.

    The input dictionaries are mutated in place. The returned original-node mapping
    and instance map are used when realistic connector balls are enabled.
    """
    original_node_dict = {
        name: np.array(position, dtype=float) for name, position in node_dict.items()
    }
    positions = np.array(list(original_node_dict.values()))
    center = np.mean(positions, axis=0)
    scale = 1.0 + clone_offset

    node_instances = {name: [] for name in original_node_dict}
    node_owner = {}
    new_node_dict = {}
    min_z = float("inf")

    for triangle_name, triangle_nodes in list(triangle_dict.items()):
        new_nodes = list(triangle_nodes)
        triangle_positions = [original_node_dict[node] for node in triangle_nodes[:3]]
        triangle_centroid = np.mean(triangle_positions, axis=0)

        for index, node in enumerate(triangle_nodes[:3]):
            if node not in node_owner:
                node_owner[node] = triangle_name
                instance_name = node
            else:
                instance_name = f"{node}_tri_{triangle_name}"

            node_instances[node].append(instance_name)

            original_position = original_node_dict[node]
            new_position = (
                center
                + scale * (triangle_centroid - center)
                + (original_position - triangle_centroid)
            )
            new_node_dict[instance_name] = new_position.tolist()
            min_z = min(min_z, float(new_position[2]))

            new_nodes[index] = instance_name
            if triangle_nodes[3] == node:
                new_nodes[3] = instance_name

        triangle_dict[triangle_name] = new_nodes

    for original_position in original_node_dict.values():
        ball_z = center[2] + scale * (original_position[2] - center[2])
        min_z = min(min_z, float(ball_z))

    z_offset = max(0.0, 0.11 - min_z)
    if z_offset > 0.0:
        center[2] += z_offset
        for position in original_node_dict.values():
            position[2] += z_offset
        for position in new_node_dict.values():
            position[2] += z_offset

    node_dict.clear()
    node_dict.update(new_node_dict)
    return original_node_dict, node_instances, center, scale


def build_abstract_triangle(spec: mujoco.MjSpec, triangle_dict: TriangleDict) -> None:
    actuator_names: set[str] = set()
    for triangle_nodes in triangle_dict.values():
        nodes = triangle_nodes[:3]
        passive_node = triangle_nodes[3]

        add_tendon(spec, from_node_name=nodes[0], to_node_name=nodes[1])
        add_tendon(spec, from_node_name=nodes[1], to_node_name=nodes[2])
        add_tendon(spec, from_node_name=nodes[2], to_node_name=nodes[0])

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            if from_node == passive_node or to_node == passive_node:
                add_actuator(
                    spec,
                    tendon_name=f"tendon_{from_node}_{to_node}",
                    kp=5000.0,
                    dampratio=1.0,
                    used_names=actuator_names,
                )

    add_perimeter_constraint(spec, triangle_dict)


def build_realistic_triangle(spec: mujoco.MjSpec, triangle_dict: TriangleDict) -> None:
    edge_tendons: EdgeTendonMap = {}
    actuated_tendons: set[str] = set()
    actuator_names: set[str] = set()

    for triangle_nodes in triangle_dict.values():
        nodes = triangle_nodes[:3]
        passive_node = triangle_nodes[3]

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            add_edge_tendon(spec, edge_tendons, from_node, to_node)

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            if from_node == passive_node or to_node == passive_node:
                tendon_name = edge_tendons[edge_key(from_node, to_node)]
                if tendon_name in actuated_tendons:
                    continue
                add_realistic_actuator(
                    spec,
                    tendon_name=tendon_name,
                    kp=1000.0,
                    dampratio=1.0,
                    used_names=actuator_names,
                )
                actuated_tendons.add(tendon_name)

    add_perimeter_constraint(spec, triangle_dict, edge_tendons)


def build_abstract_shapes(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    shape_dict: ShapeDict,
) -> None:
    shape_dict = _copy_shape_dict(shape_dict)
    _validate_shape_dict(node_dict, shape_dict)

    create_node_bodies(spec, node_dict)

    edge_tendons: EdgeTendonMap = {}
    route_tendons = {}
    actuated_tendons: set[str] = set()
    actuator_names: set[str] = set()

    for shape_name, shape in shape_dict.items():
        route = shape["route"]
        for from_node, to_node in zip(route, route[1:], strict=False):
            add_edge_tendon(spec, edge_tendons, from_node, to_node)

        for from_node, to_node in shape["active_edges"]:
            tendon_name = edge_tendons[edge_key(from_node, to_node)]
            if tendon_name in actuated_tendons:
                continue
            add_actuator(
                spec,
                tendon_name=tendon_name,
                kp=5000.0,
                dampratio=1.0,
                used_names=actuator_names,
            )
            actuated_tendons.add(tendon_name)

        route_tendons[shape_name] = add_route_tendon(spec, shape_name, route)

    add_route_length_constraints(
        spec,
        shape_dict,
        route_tendons,
    )


def build_triangle(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    triangle_dict: TriangleDict,
    *,
    realistic: bool = False,
) -> None:
    _validate_node_dict(node_dict)
    _validate_triangle_dict(node_dict, triangle_dict)
    node_dict = _copy_node_dict(node_dict)
    triangle_dict = _copy_triangle_dict(triangle_dict)

    if realistic:
        original_node_dict, node_instances, center, scale = clone_shared_nodes(
            node_dict,
            triangle_dict,
        )
        create_triangle_bodies(
            spec,
            original_node_dict,
            node_instances,
            node_dict,
            triangle_dict,
            center,
            scale,
            realistic=True,
        )
    else:
        create_node_bodies(spec, node_dict)
        build_abstract_triangle(spec, triangle_dict)
        return

    build_realistic_triangle(spec, triangle_dict)


def build_shapes(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    *,
    realistic: bool = False,
) -> None:
    if realistic:
        raise NotImplementedError(
            "Routed shape dictionaries are only supported with realistic=False."
        )

    _validate_node_dict(node_dict)
    node_dict = _copy_node_dict(node_dict)
    build_abstract_shapes(spec, node_dict, shape_dict)


def _copy_node_dict(node_dict: NodeDict) -> NodeDict:
    return {name: list(position) for name, position in node_dict.items()}


def _copy_triangle_dict(triangle_dict: TriangleDict) -> TriangleDict:
    return {name: list(nodes) for name, nodes in triangle_dict.items()}


def _copy_shape_dict(shape_dict: ShapeDict) -> ShapeDict:
    copied: dict[str, Any] = {}
    for name, shape in shape_dict.items():
        if not isinstance(shape, dict):
            copied[name] = shape
            continue
        copied[name] = {}
        for key, value in shape.items():
            if isinstance(value, list | tuple):
                copied[name][key] = [
                    list(edge) if isinstance(edge, list | tuple) else edge for edge in value
                ]
            else:
                copied[name][key] = value
    return copied


def _validate_node_dict(node_dict: NodeDict) -> None:
    if not isinstance(node_dict, dict) or not node_dict:
        raise ValueError("node_dict must be a non-empty dictionary of node names to 3D positions.")

    for node_name, position in node_dict.items():
        if not isinstance(node_name, str) or not node_name:
            raise ValueError(f"Node name {node_name!r} must be a non-empty string.")
        if not isinstance(position, list | tuple | np.ndarray) or len(position) != 3:
            raise ValueError(f"Node '{node_name}' position must contain exactly three numbers.")
        try:
            np.array(position, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Node '{node_name}' position must contain only numbers.") from exc


def _validate_triangle_dict(node_dict: NodeDict, triangle_dict: TriangleDict) -> None:
    if not isinstance(triangle_dict, dict) or not triangle_dict:
        raise ValueError(
            "triangle_dict must be a non-empty dictionary of triangle names to node lists."
        )

    for triangle_name, triangle_nodes in triangle_dict.items():
        if not isinstance(triangle_name, str) or not triangle_name:
            raise ValueError(f"Triangle name {triangle_name!r} must be a non-empty string.")
        if not isinstance(triangle_nodes, list | tuple) or len(triangle_nodes) != 4:
            raise ValueError(
                f"Triangle '{triangle_name}' must contain exactly four node names: "
                "three vertices followed by one passive node."
            )

        nodes = list(triangle_nodes[:3])
        passive_node = triangle_nodes[3]
        if not all(isinstance(node, str) and node for node in triangle_nodes):
            raise ValueError(f"Triangle '{triangle_name}' entries must be non-empty node names.")
        if len(set(nodes)) != 3:
            raise ValueError(f"Triangle '{triangle_name}' vertices must be three unique nodes.")

        missing_nodes = [node for node in triangle_nodes if node not in node_dict]
        if missing_nodes:
            missing = ", ".join(dict.fromkeys(missing_nodes))
            raise ValueError(f"Triangle '{triangle_name}' references unknown node(s): {missing}.")
        if passive_node not in nodes:
            raise ValueError(
                f"Triangle '{triangle_name}' passive node '{passive_node}' must be one "
                "of its first three vertex nodes."
            )


def _validate_shape_dict(node_dict: NodeDict, shape_dict: ShapeDict) -> None:
    if not isinstance(shape_dict, dict) or not shape_dict:
        raise ValueError("shape_dict must be a non-empty dictionary of shape definitions.")

    for shape_name, shape in shape_dict.items():
        if not isinstance(shape_name, str) or not shape_name:
            raise ValueError(f"Shape name {shape_name!r} must be a non-empty string.")
        if not isinstance(shape, dict):
            raise ValueError(f"Shape '{shape_name}' must be a dictionary.")
        if "route" not in shape:
            raise ValueError(f"Shape '{shape_name}' is missing required 'route'.")
        if "active_edges" not in shape:
            raise ValueError(f"Shape '{shape_name}' is missing required 'active_edges'.")

        route = shape["route"]
        active_edges = shape["active_edges"]
        if not isinstance(route, list | tuple) or len(route) < 2:
            raise ValueError(f"Shape '{shape_name}' route must contain at least two node names.")
        if not isinstance(active_edges, list):
            raise ValueError(f"Shape '{shape_name}' active_edges must be a list of node pairs.")
        if not all(isinstance(node, str) and node for node in route):
            raise ValueError(f"Shape '{shape_name}' route entries must be non-empty node names.")

        missing_nodes = [node for node in route if node not in node_dict]
        if missing_nodes:
            missing = ", ".join(missing_nodes)
            raise ValueError(f"Shape '{shape_name}' route references unknown node(s): {missing}.")

        route_edges = {
            edge_key(from_node, to_node)
            for from_node, to_node in zip(route, route[1:], strict=False)
        }
        normalized_active_edges = []
        for edge in active_edges:
            if not isinstance(edge, list | tuple) or len(edge) != 2:
                raise ValueError(
                    f"Shape '{shape_name}' active edge {edge!r} must contain two node names."
                )
            from_node, to_node = edge
            if from_node not in node_dict or to_node not in node_dict:
                raise ValueError(
                    f"Shape '{shape_name}' active edge {edge!r} references an unknown node."
                )
            key = edge_key(from_node, to_node)
            if key not in route_edges:
                raise ValueError(
                    f"Shape '{shape_name}' active edge {edge!r} is not adjacent in the route."
                )
            normalized_active_edges.append([from_node, to_node])

        shape["active_edges"] = normalized_active_edges


def _looks_like_shape_dict(candidate: Any) -> bool:
    return bool(candidate) and all(
        isinstance(value, dict) and ("route" in value or "active_edges" in value)
        for value in candidate.values()
    )


def get_mujoco_spec(*args: Any, realistic: bool = False, **kwargs: Any) -> mujoco.MjSpec:
    if kwargs:
        unexpected = ", ".join(kwargs)
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    spec = build_world()
    if len(args) == 2:
        node_dict, structure_dict = args
    elif len(args) == 1 and isinstance(args[0], str):
        node_dict, structure_dict = get_preset_definition(args[0])
    else:
        raise ValueError(
            "get_mujoco_spec() takes node_dict and triangle_dict, node_dict and shape_dict, "
            "or one structure_type string."
        )

    if isinstance(structure_dict, dict) and _looks_like_shape_dict(structure_dict):
        build_shapes(spec, node_dict, structure_dict, realistic=realistic)
    else:
        build_triangle(spec, node_dict, structure_dict, realistic=realistic)
    return spec
