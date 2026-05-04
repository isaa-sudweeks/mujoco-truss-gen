from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from mujoco_truss_gen.mujoco_model.model_types import NodeDict, ShapeDict, TriangleDict, Vector


def as_mujoco_quat(rotation_matrix: np.ndarray) -> list[float]:
    """Convert a scipy xyzw quaternion to MuJoCo's wxyz convention."""
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]


def triangle_frame(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> tuple[np.ndarray, list[float], list[Vector]]:
    """Return a triangle-local frame, body quaternion, and local node positions."""
    x_axis = p2 - p1
    edge_length = np.linalg.norm(x_axis)
    if edge_length > 1e-6:
        x_axis = x_axis / edge_length
    else:
        x_axis = np.array([1.0, 0.0, 0.0])

    p1_to_p3 = p3 - p1
    z_axis = np.cross(x_axis, p1_to_p3)
    z_norm = np.linalg.norm(z_axis)
    if z_norm > 1e-6:
        z_axis = z_axis / z_norm
    else:
        z_axis = np.array([0.0, 0.0, 1.0])

    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    local_positions = [
        [0.0, 0.0, 0.0],
        [float(edge_length), 0.0, 0.0],
        np.dot(rotation_matrix.T, p1_to_p3).tolist(),
    ]
    return rotation_matrix, as_mujoco_quat(rotation_matrix), local_positions


def get_perimeter(node_dict: NodeDict, triangle_dict: TriangleDict) -> dict[str, float]:
    perimeters = {}
    for name, nodes in triangle_dict.items():
        triangle_nodes = nodes[:3]
        positions = [np.array(node_dict[node], dtype=float) for node in triangle_nodes]
        perimeters[name] = float(
            sum(np.linalg.norm(positions[(index + 1) % 3] - positions[index]) for index in range(3))
        )
    return perimeters


def get_route_lengths(node_dict: NodeDict, shape_dict: ShapeDict) -> dict[str, float]:
    route_lengths = {}
    for name, shape in shape_dict.items():
        route = shape["route"]
        route_lengths[name] = float(
            sum(
                np.linalg.norm(
                    np.array(node_dict[to_node], dtype=float)
                    - np.array(node_dict[from_node], dtype=float)
                )
                for from_node, to_node in zip(route, route[1:], strict=False)
            )
        )
    return route_lengths
