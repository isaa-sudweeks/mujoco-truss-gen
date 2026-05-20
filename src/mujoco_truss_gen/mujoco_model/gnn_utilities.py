from __future__ import annotations

from typing import Literal

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.model import ModelSource, MujocoModel

GraphView = Literal["physical", "logical"]
LogicalAggregation = Literal["mean", "connector_ball"]


def get_edge_index(source: ModelSource, *, graph_view: GraphView = "physical") -> np.ndarray:
    """
    Returns an edge index array of shape (2, num_directed_edges) for use in PyTorch Geometric.
    Each undirected structural edge is represented as two directed edges.

    Args:
        source: A MujocoModel, mujoco.MjSpec, or XML string/path.
        graph_view: ``"physical"`` preserves the model's compiled node bodies.
            ``"logical"`` collapses realistic cloned nodes back to their source nodes.

    Returns:
        np.ndarray: Integer array of shape (2, E) where E is the number of directed edges.
    """
    model = _coerce_model(source)
    if graph_view not in ("physical", "logical"):
        raise ValueError("graph_view must be 'physical' or 'logical'.")

    node_names = model.node_names
    edge_pairs = model.structural_edges
    if graph_view == "logical":
        node_names = _logical_node_names(model)
        edge_pairs = [
            (_logical_node_name(node_a), _logical_node_name(node_b))
            for node_a, node_b in model.structural_edges
        ]

    edges = []
    for node_a, node_b in edge_pairs:
        if node_a == node_b:
            continue
        try:
            # Map node names to their 0-based index in the model's node list
            idx_a = node_names.index(node_a)
            idx_b = node_names.index(node_b)

            # PyTorch Geometric expects undirected graphs to be represented as bidirectional edges
            edges.append([idx_a, idx_b])
            edges.append([idx_b, idx_a])
        except ValueError:
            continue

    if not edges:
        return np.empty((2, 0), dtype=np.int64)

    # PyTorch Geometric typically expects shape (2, num_edges)
    edge_index = np.array(edges, dtype=np.int64).T

    # Remove any duplicate edges
    edge_index = np.unique(edge_index, axis=1)

    return edge_index


def get_node_features(
    source: ModelSource,
    *,
    graph_view: GraphView = "physical",
    aggregation: LogicalAggregation = "mean",
) -> np.ndarray:
    """
    Returns a node feature array of shape (num_nodes, num_features) for use in PyTorch Geometric.
    The features for each node are its [x, y, z] position and [vx, vy, vz] linear velocity.

    Args:
        source: A MujocoModel, mujoco.MjSpec, or XML string/path.
        graph_view: ``"physical"`` preserves the model's compiled node bodies.
            ``"logical"`` collapses realistic cloned nodes back to their source nodes.
        aggregation: Logical-node aggregation used only when ``graph_view="logical"``.
            ``"mean"`` averages cloned node bodies. ``"connector_ball"`` uses the
            matching connector ball body when present and falls back to the physical
            node instance otherwise.

    Returns:
        np.ndarray: Float array of shape (N, 6) where N is the number of nodes.
    """
    model = _coerce_model(source)
    if graph_view not in ("physical", "logical"):
        raise ValueError("graph_view must be 'physical' or 'logical'.")
    if aggregation not in ("mean", "connector_ball"):
        raise ValueError("aggregation must be 'mean' or 'connector_ball'.")

    if graph_view == "logical":
        return _get_logical_node_features(model, aggregation)

    positions = model.get_node_position_matrix()
    velocities = model.get_node_linear_velocity_matrix()

    # Concatenate positions and velocities to create a (N, 6) feature matrix
    # Convert to np.float32 as PyTorch typically uses 32-bit floats for features
    if positions.size == 0 or velocities.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    node_features = np.concatenate([positions, velocities], axis=1)

    return node_features.astype(np.float32)


def _coerce_model(source: ModelSource) -> MujocoModel:
    if isinstance(source, MujocoModel):
        return source
    return MujocoModel(source)


def _logical_node_name(node_name: str) -> str:
    return node_name.split("_tri_", 1)[0]


def _logical_node_names(model: MujocoModel) -> list[str]:
    return sorted(
        {_logical_node_name(node_name) for node_name in model.node_names},
        key=_node_sort_key,
    )


def _node_sort_key(node_name: str) -> tuple[int, int | str]:
    suffix = node_name.removeprefix("node_")
    if suffix.isdigit():
        return (0, int(suffix))
    return (1, suffix)


def _get_logical_node_features(
    model: MujocoModel,
    aggregation: LogicalAggregation,
) -> np.ndarray:
    logical_node_names = _logical_node_names(model)
    if not logical_node_names:
        return np.empty((0, 6), dtype=np.float32)

    features = []
    node_positions = model.get_node_position_dict()
    node_velocities = model.get_node_velocity_linear_dict()
    for logical_name in logical_node_names:
        if aggregation == "connector_ball":
            ball_feature = _connector_ball_feature(model, logical_name)
            if ball_feature is not None:
                features.append(ball_feature)
                continue

        instance_names = [
            node_name
            for node_name in model.node_names
            if _logical_node_name(node_name) == logical_name
        ]
        if not instance_names:
            continue

        positions = np.array([node_positions[node_name] for node_name in instance_names])
        velocities = np.array([node_velocities[node_name] for node_name in instance_names])
        features.append(
            np.concatenate([np.mean(positions, axis=0), np.mean(velocities, axis=0)])
        )

    if not features:
        return np.empty((0, 6), dtype=np.float32)

    return np.array(features, dtype=np.float32)


def _connector_ball_feature(model: MujocoModel, logical_name: str) -> np.ndarray | None:
    body_id = mujoco.mj_name2id(
        model.model,
        mujoco.mjtObj.mjOBJ_BODY,
        f"connector_ball_{logical_name}",
    )
    if body_id < 0:
        return None

    return np.concatenate(
        [
            model.data.xpos[body_id],
            model.data.cvel[body_id][3:],
        ]
    )
