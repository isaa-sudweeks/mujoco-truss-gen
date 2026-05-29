from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.bodies import (
    add_routed_node_body,
    build_world,
    connect_routed_node_to_ball,
    create_connector_balls,
    create_node_bodies,
    create_triangle_bodies,
    find_original_node,
)
from mujoco_truss_gen.mujoco_model.constants import (
    MIN_NODE_CENTER_Z,
    REALISTIC_NODE_CLONE_OFFSET,
    TrussPhysicalParameters,
)
from mujoco_truss_gen.mujoco_model.constraints import (
    add_perimeter_constraint,
)
from mujoco_truss_gen.mujoco_model.model_types import (
    EdgeTendonMap,
    NodeDict,
    ShapeDict,
    TriangleDict,
)
from mujoco_truss_gen.mujoco_model.presets import get_preset_definition
from mujoco_truss_gen.mujoco_model.sensors import (
    DEFAULT_ACCELEROMETER_CONFIG,
    AccelerometerConfig,
    add_node_accelerometers,
)
from mujoco_truss_gen.mujoco_model.tendons import (
    add_actuator,
    add_edge_tendon,
    add_realistic_actuator,
    add_route_tendon,
    add_tendon,
    edge_key,
)

STL_SOURCE_METADATA_KEY = "_mujoco_truss_gen_source"
STL_SOURCE_METADATA_VALUE = "stl"
REALISTIC_ROUTED_TARGET_EDGE_LENGTH = 1.0
_DEFAULT_ACCELEROMETER_CONFIG = object()


def clone_shared_nodes(
    node_dict: NodeDict,
    triangle_dict: TriangleDict,
    clone_offset: float = REALISTIC_NODE_CLONE_OFFSET,
    min_node_center_z: float = MIN_NODE_CENTER_Z,
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

    z_offset = max(0.0, min_node_center_z - min_z)
    if z_offset > 0.0:
        center[2] += z_offset
        for position in original_node_dict.values():
            position[2] += z_offset
        for position in new_node_dict.values():
            position[2] += z_offset

    node_dict.clear()
    node_dict.update(new_node_dict)
    return original_node_dict, node_instances, center, scale


def clone_routed_nodes(
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    clone_offset: float = REALISTIC_NODE_CLONE_OFFSET,
    min_node_center_z: float = MIN_NODE_CENTER_Z,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], np.ndarray, float, dict[str, np.ndarray]]:
    """Clone routed node occurrences and offset them along local route bisectors."""
    original_node_dict = {
        name: np.array(position, dtype=float) for name, position in node_dict.items()
    }
    center = np.mean(np.array(list(original_node_dict.values())), axis=0)
    scale = 1.0 + clone_offset
    mean_edge_length = _mean_route_edge_length(original_node_dict, shape_dict)

    node_instances = {name: [] for name in original_node_dict}
    node_owner: dict[str, str] = {}
    new_node_dict: NodeDict = {}
    hinge_axes: dict[str, np.ndarray] = {}
    ball_positions = {
        name: center + scale * (position - center)
        for name, position in original_node_dict.items()
    }

    for shape_name, shape in shape_dict.items():
        route = list(shape["route"])
        new_route = []
        for index, node in enumerate(route):
            if node not in node_owner:
                node_owner[node] = f"{shape_name}_{index}"
                instance_name = node
            else:
                instance_name = f"{node}_route_{shape_name}_{index}"

            node_instances[node].append(instance_name)
            normal = _route_occurrence_normal(original_node_dict, route, index)
            bisector = _route_occurrence_bisector(original_node_dict, route, index)
            new_node_dict[instance_name] = (ball_positions[node] + bisector).tolist()
            hinge_axes[instance_name] = normal
            new_route.append(instance_name)

        shape["route"] = new_route
        shape["active_edges"] = [
            [new_route[index], new_route[index + 1]]
            for index in range(len(new_route) - 1)
        ]

    offset_length = _choose_routed_offset_length(
        new_node_dict,
        shape_dict,
        node_instances,
        ball_positions,
        hinge_axes,
        mean_edge_length,
        REALISTIC_ROUTED_TARGET_EDGE_LENGTH,
    )
    new_node_dict = _routed_clone_positions_for_offset(
        shape_dict,
        node_instances,
        ball_positions,
        hinge_axes,
        offset_length,
    )
    _relax_routed_clone_positions(
        new_node_dict,
        shape_dict,
        node_instances,
        ball_positions,
        hinge_axes,
        offset_length,
    )

    min_z = min(float(position[2]) for position in new_node_dict.values())
    for ball_position in ball_positions.values():
        min_z = min(min_z, float(ball_position[2]))

    z_offset = max(0.0, min_node_center_z - min_z)
    if z_offset > 0.0:
        center[2] += z_offset
        for position in original_node_dict.values():
            position[2] += z_offset
        for position in ball_positions.values():
            position[2] += z_offset
        for position in new_node_dict.values():
            position[2] = float(position[2]) + z_offset

    node_dict.clear()
    node_dict.update(new_node_dict)
    return original_node_dict, node_instances, center, scale, hinge_axes


def build_abstract_triangle(
    spec: mujoco.MjSpec,
    node_dict_or_triangle_dict: NodeDict | TriangleDict,
    triangle_dict: TriangleDict | None = None,
    *,
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    params = physical_params or TrussPhysicalParameters()
    node_dict = node_dict_or_triangle_dict if triangle_dict is not None else None
    if triangle_dict is None:
        triangle_dict = node_dict_or_triangle_dict
    actuator_names: set[str] = set()
    for triangle_nodes in triangle_dict.values():
        nodes = triangle_nodes[:3]
        passive_node = triangle_nodes[3]

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            add_tendon(
                spec,
                from_node_name=from_node,
                to_node_name=to_node,
                tendon_range=(
                    _upper_scaled_tendon_range(
                        _distance_between_nodes(node_dict, from_node, to_node),
                        params,
                    )
                    if node_dict is not None
                    else None
                ),
                physical_params=params,
            )

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            if from_node == passive_node or to_node == passive_node:
                add_actuator(
                    spec,
                    tendon_name=f"tendon_{from_node}_{to_node}",
                    kp=params.abstract_actuator_kp,
                    dampratio=params.actuator_dampratio,
                    used_names=actuator_names,
                    physical_params=params,
                )

    add_perimeter_constraint(spec, triangle_dict, physical_params=params)


def build_realistic_triangle(
    spec: mujoco.MjSpec,
    triangle_dict: TriangleDict,
    *,
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    params = physical_params or TrussPhysicalParameters()
    edge_tendons: EdgeTendonMap = {}
    actuated_tendons: set[str] = set()
    actuator_names: set[str] = set()

    for triangle_nodes in triangle_dict.values():
        nodes = triangle_nodes[:3]
        passive_node = triangle_nodes[3]

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            add_edge_tendon(
                spec,
                edge_tendons,
                from_node,
                to_node,
                physical_params=params,
            )

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            if from_node == passive_node or to_node == passive_node:
                tendon_name = edge_tendons[edge_key(from_node, to_node)]
                if tendon_name in actuated_tendons:
                    continue
                add_realistic_actuator(
                    spec,
                    tendon_name=tendon_name,
                    kp=params.realistic_actuator_kp,
                    dampratio=params.actuator_dampratio,
                    used_names=actuator_names,
                    physical_params=params,
                )
                actuated_tendons.add(tendon_name)

    add_perimeter_constraint(spec, triangle_dict, edge_tendons, physical_params=params)


def build_abstract_shapes(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    *,
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    params = physical_params or TrussPhysicalParameters()
    shape_dict = _copy_shape_dict(shape_dict)
    _validate_shape_dict(node_dict, shape_dict)

    create_node_bodies(spec, node_dict, physical_params=params)

    edge_tendons: EdgeTendonMap = {}
    actuated_tendons: set[str] = set()
    actuator_names: set[str] = set()

    for shape_name, shape in shape_dict.items():
        route = shape["route"]
        scale_limits_to_geometry = _is_stl_imported_shape(shape)
        for from_node, to_node in zip(route, route[1:], strict=False):
            edge_length = _distance_between_nodes(node_dict, from_node, to_node)
            tendon_name = add_edge_tendon(
                spec,
                edge_tendons,
                from_node,
                to_node,
                tendon_range=(
                    _scaled_tendon_range(edge_length, params)
                    if scale_limits_to_geometry
                    else None
                ),
                physical_params=params,
            )
            if tendon_name in actuated_tendons:
                continue
            add_actuator(
                spec,
                tendon_name=tendon_name,
                kp=params.abstract_actuator_kp,
                dampratio=params.actuator_dampratio,
                used_names=actuator_names,
                actrange=(
                    _scaled_actuator_range(edge_length, params)
                    if scale_limits_to_geometry
                    else None
                ),
                physical_params=params,
            )
            actuated_tendons.add(tendon_name)

        add_route_tendon(
            spec,
            shape_name,
            route,
            tendon_range=(
                _scaled_tendon_range(_route_length(node_dict, route), params)
                if scale_limits_to_geometry
                else None
            ),
            physical_params=params,
        )


def build_realistic_shapes(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    node_instances: dict[str, list[str]],
    original_node_dict: dict[str, np.ndarray],
    center: np.ndarray,
    scale: float,
    hinge_axes: dict[str, np.ndarray],
    *,
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    params = physical_params or TrussPhysicalParameters()
    connector_balls = create_connector_balls(
        spec,
        original_node_dict,
        node_instances,
        center,
        scale,
        physical_params=params,
    )
    passive_nodes = _routed_passive_nodes(shape_dict)

    for instance_name, position in node_dict.items():
        original_name = find_original_node(node_instances, instance_name)
        connector_direction = None
        if original_name and original_name in connector_balls:
            connector_direction = (
                np.array(connector_balls[original_name].pos, dtype=float)
                - np.array(position, dtype=float)
            )

        is_passive = instance_name in passive_nodes
        node_body = add_routed_node_body(
            spec,
            node_name=instance_name,
            position=position,
            hinge_axis=hinge_axes[instance_name].tolist(),
            connector_direction=connector_direction.tolist()
            if connector_direction is not None
            else None,
            mass=params.passive_node_mass if is_passive else params.active_node_mass,
            box_size=params.box_size,
            passive=is_passive,
            edge_tendon_width=params.edge_tendon_width,
        )

        if not original_name or original_name not in connector_balls:
            continue

        connect_routed_node_to_ball(
            spec,
            node_body=node_body,
            instance_name=instance_name,
            ball=connector_balls[original_name],
            original_name=original_name,
            node_dict=node_dict,
            physical_params=params,
        )

    edge_tendons: EdgeTendonMap = {}
    actuated_tendons: set[str] = set()
    actuator_names: set[str] = set()

    for shape_name, shape in shape_dict.items():
        route = shape["route"]
        for from_node, to_node in zip(route, route[1:], strict=False):
            tendon_name = add_edge_tendon(
                spec,
                edge_tendons,
                from_node,
                to_node,
                physical_params=params,
            )
            if tendon_name in actuated_tendons:
                continue
            add_realistic_actuator(
                spec,
                tendon_name=tendon_name,
                kp=params.realistic_actuator_kp,
                dampratio=params.actuator_dampratio,
                used_names=actuator_names,
                physical_params=params,
            )
            actuated_tendons.add(tendon_name)

        add_route_tendon(spec, shape_name, route, physical_params=params)


def _routed_passive_nodes(shape_dict: ShapeDict) -> set[str]:
    passive_nodes = set()
    for shape in shape_dict.values():
        route = shape["route"]
        passive_nodes.update((route[0], route[-1]))
    return passive_nodes


def build_triangle(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    triangle_dict: TriangleDict,
    *,
    realistic: bool = False,
    physical_params: TrussPhysicalParameters | None = None,
    accelerometer_config: AccelerometerConfig | dict[str, Any] | None | object = (
        _DEFAULT_ACCELEROMETER_CONFIG
    ),
) -> None:
    params = physical_params or TrussPhysicalParameters()
    _validate_node_dict(node_dict)
    _validate_triangle_dict(node_dict, triangle_dict)
    node_dict = _copy_node_dict(node_dict)
    triangle_dict = _copy_triangle_dict(triangle_dict)

    if realistic:
        original_node_dict, node_instances, center, scale = clone_shared_nodes(
            node_dict,
            triangle_dict,
            clone_offset=params.realistic_node_clone_offset,
            min_node_center_z=params.min_node_center_z,
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
            physical_params=params,
        )
        add_node_accelerometers(
            spec,
            list(node_dict),
            (
                DEFAULT_ACCELEROMETER_CONFIG
                if accelerometer_config is _DEFAULT_ACCELEROMETER_CONFIG
                else accelerometer_config
            ),
        )
    else:
        _lift_nodes_above_ground(node_dict, params)
        create_node_bodies(
            spec,
            node_dict,
            _triangle_node_masses(triangle_dict, params),
            physical_params=params,
        )
        build_abstract_triangle(spec, node_dict, triangle_dict, physical_params=params)
        return

    build_realistic_triangle(spec, triangle_dict, physical_params=params)


def build_shapes(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    *,
    realistic: bool = False,
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    params = physical_params or TrussPhysicalParameters()
    _validate_node_dict(node_dict)
    node_dict = _copy_node_dict(node_dict)
    shape_dict = _copy_shape_dict(shape_dict)
    _validate_shape_dict(node_dict, shape_dict)

    if realistic:
        original_node_dict, node_instances, center, scale, hinge_axes = clone_routed_nodes(
            node_dict,
            shape_dict,
            clone_offset=params.realistic_node_clone_offset,
            min_node_center_z=params.min_node_center_z,
        )
        build_realistic_shapes(
            spec,
            node_dict,
            shape_dict,
            node_instances,
            original_node_dict,
            center,
            scale,
            hinge_axes,
            physical_params=params,
        )
        return

    _lift_nodes_above_ground(node_dict, params)
    build_abstract_shapes(spec, node_dict, shape_dict, physical_params=params)


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


def _lift_nodes_above_ground(
    node_dict: NodeDict,
    physical_params: TrussPhysicalParameters,
) -> None:
    min_z = min(float(position[2]) for position in node_dict.values())
    z_offset = max(0.0, physical_params.min_node_center_z - min_z)
    if z_offset == 0.0:
        return

    for position in node_dict.values():
        position[2] = float(position[2]) + z_offset


def _is_stl_imported_shape(shape: dict[str, Any]) -> bool:
    return shape.get(STL_SOURCE_METADATA_KEY) == STL_SOURCE_METADATA_VALUE


def _distance_between_nodes(node_dict: NodeDict, from_node: str, to_node: str) -> float:
    from_position = np.array(node_dict[from_node], dtype=float)
    to_position = np.array(node_dict[to_node], dtype=float)
    return float(np.linalg.norm(to_position - from_position))


def _route_length(node_dict: NodeDict, route: list[str]) -> float:
    return sum(
        _distance_between_nodes(node_dict, from_node, to_node)
        for from_node, to_node in zip(route, route[1:], strict=False)
    )


def _mean_route_edge_length(
    node_dict: dict[str, np.ndarray],
    shape_dict: ShapeDict,
) -> float:
    lengths = []
    for shape in shape_dict.values():
        route = shape["route"]
        for from_node, to_node in zip(route, route[1:], strict=False):
            lengths.append(float(np.linalg.norm(node_dict[to_node] - node_dict[from_node])))
    return float(np.mean(lengths)) if lengths else 1.0


def _route_occurrence_normal(
    node_dict: dict[str, np.ndarray],
    route: list[str],
    index: int,
) -> np.ndarray:
    if len(route) >= 3:
        if index == 0:
            indices = (0, 1, 2)
        elif index == len(route) - 1:
            indices = (len(route) - 3, len(route) - 2, len(route) - 1)
        else:
            indices = (index - 1, index, index + 1)
        a, b, c = (node_dict[route[route_index]] for route_index in indices)
        normal = np.cross(a - b, c - b)
        unit = _unit_vector(normal)
        if unit is not None:
            return unit
    return np.array([0.0, 0.0, 1.0])


def _route_occurrence_bisector(
    node_dict: dict[str, np.ndarray],
    route: list[str],
    index: int,
) -> np.ndarray:
    node_position = node_dict[route[index]]
    directions = []
    if index > 0:
        directions.append(_unit_vector(node_dict[route[index - 1]] - node_position))
    if index < len(route) - 1:
        directions.append(_unit_vector(node_dict[route[index + 1]] - node_position))

    valid_directions = [direction for direction in directions if direction is not None]
    if len(valid_directions) == 2:
        bisector = _unit_vector(valid_directions[0] + valid_directions[1])
        if bisector is not None:
            return bisector
    if len(valid_directions) == 1:
        return valid_directions[0]
    return np.array([1.0, 0.0, 0.0])


def _relax_routed_clone_positions(
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    node_instances: dict[str, list[str]],
    ball_positions: dict[str, np.ndarray],
    hinge_axes: dict[str, np.ndarray],
    offset_length: float,
    iterations: int = 200,
) -> None:
    instance_to_original = {
        instance_name: original_name
        for original_name, instances in node_instances.items()
        for instance_name in instances
    }
    positions = {name: np.array(position, dtype=float) for name, position in node_dict.items()}

    for _ in range(iterations):
        updated = {name: position.copy() for name, position in positions.items()}
        for shape in shape_dict.values():
            route = shape["route"]
            for index, instance_name in enumerate(route):
                original_name = instance_to_original[instance_name]
                node_position = positions[instance_name]
                directions = []
                if index > 0:
                    directions.append(_unit_vector(positions[route[index - 1]] - node_position))
                if index < len(route) - 1:
                    directions.append(_unit_vector(positions[route[index + 1]] - node_position))

                valid_directions = [direction for direction in directions if direction is not None]
                if len(valid_directions) == 2:
                    offset_direction = _unit_vector(valid_directions[0] + valid_directions[1])
                elif len(valid_directions) == 1:
                    offset_direction = valid_directions[0]
                else:
                    offset_direction = None
                if offset_direction is None:
                    continue

                hinge_axis = hinge_axes[instance_name]
                offset_direction = _unit_vector(
                    offset_direction - hinge_axis * float(np.dot(offset_direction, hinge_axis))
                )
                if offset_direction is None:
                    continue

                target_position = ball_positions[original_name] + offset_length * offset_direction
                updated[instance_name] = 0.5 * positions[instance_name] + 0.5 * target_position
        positions = updated

    for name, position in positions.items():
        node_dict[name] = position.tolist()


def _choose_routed_offset_length(
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    node_instances: dict[str, list[str]],
    ball_positions: dict[str, np.ndarray],
    hinge_axes: dict[str, np.ndarray],
    mean_edge_length: float,
    target_edge_length: float,
) -> float:
    max_offset = max(mean_edge_length, target_edge_length) * 2.0
    candidates = np.linspace(1e-6, max_offset, 81)
    best_offset = float(candidates[0])
    best_error = float("inf")

    for candidate in candidates:
        candidate_positions = _routed_clone_positions_for_offset(
            shape_dict,
            node_instances,
            ball_positions,
            hinge_axes,
            float(candidate),
        )
        _relax_routed_clone_positions(
            candidate_positions,
            shape_dict,
            node_instances,
            ball_positions,
            hinge_axes,
            float(candidate),
            iterations=80,
        )
        error = _route_edge_length_error(candidate_positions, shape_dict, target_edge_length)
        if error < best_error:
            best_error = error
            best_offset = float(candidate)

    return best_offset


def _routed_clone_positions_for_offset(
    shape_dict: ShapeDict,
    node_instances: dict[str, list[str]],
    ball_positions: dict[str, np.ndarray],
    hinge_axes: dict[str, np.ndarray],
    offset_length: float,
) -> NodeDict:
    instance_to_original = {
        instance_name: original_name
        for original_name, instances in node_instances.items()
        for instance_name in instances
    }
    positions: NodeDict = {}
    for shape in shape_dict.values():
        route = shape["route"]
        for index, instance_name in enumerate(route):
            original_name = instance_to_original[instance_name]
            ball_position = ball_positions[original_name]
            directions = []
            if index > 0:
                directions.append(ball_positions[instance_to_original[route[index - 1]]] - ball_position)
            if index < len(route) - 1:
                directions.append(ball_positions[instance_to_original[route[index + 1]]] - ball_position)

            valid_directions = [
                direction / norm
                for direction in directions
                if (norm := float(np.linalg.norm(direction))) >= 1e-10
            ]
            if len(valid_directions) == 2:
                offset_direction = _unit_vector(valid_directions[0] + valid_directions[1])
            elif len(valid_directions) == 1:
                offset_direction = valid_directions[0]
            else:
                offset_direction = None
            if offset_direction is None:
                offset_direction = np.array([1.0, 0.0, 0.0])

            hinge_axis = hinge_axes[instance_name]
            offset_direction = _unit_vector(
                offset_direction - hinge_axis * float(np.dot(offset_direction, hinge_axis))
            )
            if offset_direction is None:
                offset_direction = np.array([1.0, 0.0, 0.0])

            positions[instance_name] = (
                ball_position + offset_length * offset_direction
            ).tolist()
    return positions


def _route_edge_length_error(
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    target_edge_length: float,
) -> float:
    squared_errors = []
    for shape in shape_dict.values():
        route = shape["route"]
        for from_node, to_node in zip(route, route[1:], strict=False):
            edge_length = _distance_between_nodes(node_dict, from_node, to_node)
            squared_errors.append((edge_length - target_edge_length) ** 2)
    return float(np.mean(squared_errors)) if squared_errors else 0.0


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        return None
    return vector / norm


def _triangle_node_masses(
    triangle_dict: TriangleDict,
    physical_params: TrussPhysicalParameters,
) -> dict[str, float]:
    node_masses: dict[str, float] = {}
    for triangle_nodes in triangle_dict.values():
        for node in triangle_nodes[:3]:
            node_masses.setdefault(node, physical_params.active_node_mass)
        node_masses[triangle_nodes[3]] = physical_params.passive_node_mass
    return node_masses


def _scaled_tendon_range(
    length: float,
    physical_params: TrussPhysicalParameters,
) -> list[float]:
    if length <= 0.0:
        raise ValueError("Cannot scale tendon limits for a zero-length edge.")
    return [
        length * physical_params.tendon_range_min_factor,
        length * physical_params.tendon_range_max_factor,
    ]


def _upper_scaled_tendon_range(
    length: float,
    physical_params: TrussPhysicalParameters,
) -> list[float]:
    if length <= 0.0:
        raise ValueError("Cannot scale tendon limits for a zero-length edge.")
    return [
        0.5,
        length * physical_params.tendon_range_max_factor,
    ]


def _scaled_actuator_range(
    length: float,
    physical_params: TrussPhysicalParameters,
) -> list[float]:
    if length <= 0.0:
        raise ValueError("Cannot scale actuator limits for a zero-length STL route edge.")
    return [
        length * physical_params.actuator_range_min_factor,
        length * physical_params.actuator_range_max_factor,
    ]


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


def get_mujoco_spec(
    *args: Any,
    realistic: bool = False,
    scale: float = 1.0,
    physical_params: TrussPhysicalParameters | None = None,
    accelerometer_config: AccelerometerConfig | dict[str, Any] | None | object = (
        _DEFAULT_ACCELEROMETER_CONFIG
    ),
    **kwargs: Any,
) -> mujoco.MjSpec:
    if kwargs:
        unexpected = ", ".join(kwargs)
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    params = physical_params or TrussPhysicalParameters()
    spec = build_world()
    if len(args) == 2:
        if scale != 1.0:
            raise ValueError("scale is only supported when building a named preset.")
        node_dict, structure_dict = args
    elif len(args) == 1 and isinstance(args[0], str):
        node_dict, structure_dict = get_preset_definition(args[0], scale=scale)
    else:
        raise ValueError(
            "get_mujoco_spec() takes node_dict and triangle_dict, node_dict and shape_dict, "
            "or one structure_type string."
        )

    if isinstance(structure_dict, dict) and _looks_like_shape_dict(structure_dict):
        if accelerometer_config is not _DEFAULT_ACCELEROMETER_CONFIG:
            raise ValueError("accelerometer_config is only supported with triangle models.")
        build_shapes(
            spec,
            node_dict,
            structure_dict,
            realistic=realistic,
            physical_params=params,
        )
    else:
        build_triangle(
            spec,
            node_dict,
            structure_dict,
            realistic=realistic,
            physical_params=params,
            accelerometer_config=accelerometer_config,
        )
    return spec
