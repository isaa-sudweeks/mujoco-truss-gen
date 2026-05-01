from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.bodies import (
    build_world,
    create_node_bodies,
    create_triangle_bodies,
)
from mujoco_truss_gen.mujoco_model.constraints import add_perimeter_constraint
from mujoco_truss_gen.mujoco_model.model_types import EdgeTendonMap, NodeDict, TriangleDict
from mujoco_truss_gen.mujoco_model.presets import get_preset_definition
from mujoco_truss_gen.mujoco_model.tendons import (
    add_actuator,
    add_edge_tendon,
    add_realistic_actuator,
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
                )

    add_perimeter_constraint(spec, triangle_dict)


def build_realistic_triangle(spec: mujoco.MjSpec, triangle_dict: TriangleDict) -> None:
    edge_tendons: EdgeTendonMap = {}
    actuated_tendons: set[str] = set()

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
                )
                actuated_tendons.add(tendon_name)

    add_perimeter_constraint(spec, triangle_dict, edge_tendons)


def build_triangle(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    triangle_dict: TriangleDict,
    *,
    realistic: bool = False,
) -> None:
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


def _copy_node_dict(node_dict: NodeDict) -> NodeDict:
    return {name: list(position) for name, position in node_dict.items()}


def _copy_triangle_dict(triangle_dict: TriangleDict) -> TriangleDict:
    return {name: list(nodes) for name, nodes in triangle_dict.items()}


def get_mujoco_spec(*args: Any, realistic: bool = False, **kwargs: Any) -> mujoco.MjSpec:
    if kwargs:
        unexpected = ", ".join(kwargs)
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    spec = build_world()
    if len(args) == 2:
        node_dict, triangle_dict = args
    elif len(args) == 1 and isinstance(args[0], str):
        node_dict, triangle_dict = get_preset_definition(args[0])
    else:
        raise ValueError(
            "get_mujoco_spec() takes node_dict and triangle_dict, or one structure_type string."
        )

    build_triangle(spec, node_dict, triangle_dict, realistic=realistic)
    return spec
