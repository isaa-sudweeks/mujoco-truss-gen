from __future__ import annotations

import numpy as np

from mujoco_truss_gen.mujoco_model.model import ModelSource, MujocoModel


def get_edge_index(source: ModelSource) -> np.ndarray:
    """
    Returns an edge index array of shape (2, num_directed_edges) for use in PyTorch Geometric.
    Each undirected structural edge is represented as two directed edges.

    Args:
        source: A MujocoModel, mujoco.MjSpec, or XML string/path.

    Returns:
        np.ndarray: Integer array of shape (2, E) where E is the number of directed edges.
    """
    if not isinstance(source, MujocoModel):
        model = MujocoModel(source)
    else:
        model = source

    edges = []
    for node_a, node_b in model.structural_edges:
        try:
            # Map node names to their 0-based index in the model's node list
            idx_a = model.node_names.index(node_a)
            idx_b = model.node_names.index(node_b)

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


def get_node_features(source: ModelSource) -> np.ndarray:
    """
    Returns a node feature array of shape (num_nodes, num_features) for use in PyTorch Geometric.
    The features for each node are its [x, y, z] position and [vx, vy, vz] linear velocity.

    Args:
        source: A MujocoModel, mujoco.MjSpec, or XML string/path.

    Returns:
        np.ndarray: Float array of shape (N, 6) where N is the number of nodes.
    """
    if not isinstance(source, MujocoModel):
        model = MujocoModel(source)
    else:
        model = source

    positions = model.get_node_position_matrix()
    velocities = model.get_node_linear_velocity_matrix()

    # Concatenate positions and velocities to create a (N, 6) feature matrix
    # Convert to np.float32 as PyTorch typically uses 32-bit floats for features
    if positions.size == 0 or velocities.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    node_features = np.concatenate([positions, velocities], axis=1)

    return node_features.astype(np.float32)
