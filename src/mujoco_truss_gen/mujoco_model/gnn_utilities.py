from __future__ import annotations

from typing import Any, Literal

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.model import ModelSource, MujocoModel

GraphView = Literal["physical", "logical", "control"]
LogicalAggregation = Literal["mean", "connector_ball"]


def get_edge_index(source: ModelSource, *, graph_view: GraphView = "physical") -> np.ndarray:
    """
    Returns an edge index array of shape (2, num_directed_edges) for use in PyTorch Geometric.
    Each undirected structural edge is represented as two directed edges.

    Args:
        source: A MujocoModel, mujoco.MjSpec, or XML string/path.
        graph_view: ``"physical"`` preserves the model's compiled node bodies.
            ``"logical"`` collapses realistic cloned nodes back to their source nodes.
            ``"control"`` returns the policy/control graph, including actuated and
            virtual connector edges, when the model provides control metadata.

    Returns:
        np.ndarray: Integer array of shape (2, E) where E is the number of directed edges.
    """
    model = _coerce_model(source)
    if graph_view not in ("physical", "logical", "control"):
        raise ValueError("graph_view must be 'physical', 'logical', or 'control'.")

    node_names = model.node_names
    edge_pairs = model.structural_edges
    if graph_view == "logical":
        node_names = _logical_node_names(model)
        edge_pairs = [
            (_logical_node_name(node_a), _logical_node_name(node_b))
            for node_a, node_b in model.structural_edges
        ]
    elif graph_view == "control":
        return _get_control_edge_index(model)

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
            ``"control"`` returns features for policy/control nodes. Abstract
            duplicate control nodes may alias the same physical body.
        aggregation: Logical-node aggregation used only when ``graph_view="logical"``.
            ``"mean"`` averages cloned node bodies. ``"connector_ball"`` uses the
            matching connector ball body when present and falls back to the physical
            node instance otherwise.

    Returns:
        np.ndarray: Float array of shape (N, 6) where N is the number of nodes.
    """
    model = _coerce_model(source)
    if graph_view not in ("physical", "logical", "control"):
        raise ValueError("graph_view must be 'physical', 'logical', or 'control'.")
    if aggregation not in ("mean", "connector_ball"):
        raise ValueError("aggregation must be 'mean' or 'connector_ball'.")

    if graph_view == "logical":
        return _get_logical_node_features(model, aggregation)
    if graph_view == "control":
        return _get_control_node_features(model)

    positions = model.get_node_position_matrix()
    velocities = model.get_node_linear_velocity_matrix()

    # Concatenate positions and velocities to create a (N, 6) feature matrix
    # Convert to np.float32 as PyTorch typically uses 32-bit floats for features
    if positions.size == 0 or velocities.size == 0:
        return np.empty((0, 6), dtype=np.float32)

    node_features = np.concatenate([positions, velocities], axis=1)

    return node_features.astype(np.float32)


def get_edge_types(source: ModelSource, *, graph_view: GraphView = "physical") -> np.ndarray:
    """
    Returns edge type labels aligned with ``get_edge_index`` columns.

    For ``graph_view="control"``, each undirected control edge contributes two
    directed edge-index columns and two matching type labels. Labels are either
    ``"actuated"`` for physical tendon/tube message edges or ``"connector"`` for
    virtual same-connector message edges. Other graph views return ``"structural"``
    labels aligned with their directed structural edges.
    """
    model = _coerce_model(source)
    if graph_view not in ("physical", "logical", "control"):
        raise ValueError("graph_view must be 'physical', 'logical', or 'control'.")

    if graph_view == "control":
        edge_types = []
        for edge in model.control_graph.edges:
            edge_types.extend((edge.type, edge.type))
        return np.array(edge_types, dtype=object)

    edge_index = get_edge_index(model, graph_view=graph_view)
    return np.full(edge_index.shape[1], "structural", dtype=object)


def get_networkx_graph(
    source: ModelSource,
    *,
    graph_view: GraphView = "control",
    aggregation: LogicalAggregation = "mean",
):
    """
    Returns a NetworkX graph for the requested GNN graph view.

    Node attributes include ``label`` and, when available, ``feature`` and ``pos``.
    Edge attributes include ``type``: ``"actuated"``, ``"connector"``, or
    ``"structural"``. The control graph is returned as a ``networkx.MultiGraph`` so
    parallel actuated/connector relationships between the same nodes can be preserved.
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "get_networkx_graph() requires networkx. Install networkx to visualize "
            "GNN graph structures."
        ) from exc

    model = _coerce_model(source)
    node_names = _graph_node_names(model, graph_view)
    features = get_node_features(model, graph_view=graph_view, aggregation=aggregation)
    graph = nx.MultiGraph()

    for index, node_name in enumerate(node_names):
        attributes: dict[str, Any] = {"label": node_name}
        if index < features.shape[0]:
            feature = features[index]
            attributes["feature"] = feature.copy()
            if feature.size >= 2:
                attributes["pos"] = (float(feature[0]), float(feature[1]))
        graph.add_node(node_name, **attributes)

    for from_node, to_node, edge_type in _named_graph_edges(model, graph_view):
        graph.add_edge(from_node, to_node, type=edge_type)

    return graph


def view_graph(
    source: ModelSource,
    *,
    graph_view: GraphView = "control",
    aggregation: LogicalAggregation = "mean",
    layout: Literal["spring", "kamada_kawai", "spectral", "physical"] = "spring",
    seed: int = 7,
    with_labels: bool = True,
    show_edge_types: bool = True,
    show: bool = True,
    ax: Any | None = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    node_size: int = 900,
    font_size: int = 8,
):
    """
    Plot a model graph view with NetworkX and Matplotlib.

    Args:
        source: A MujocoModel, mujoco.MjSpec, XML string/path, or mujoco.MjModel.
        graph_view: ``"physical"``, ``"logical"``, or ``"control"``.
        aggregation: Logical-node aggregation for ``graph_view="logical"``.
        layout: NetworkX layout. ``"physical"`` uses node x/y positions from the
            selected graph features when available.
        seed: Seed used by the spring layout.
        with_labels: Draw node labels.
        show_edge_types: Use different styles for actuated, connector, and
            structural edges and draw a legend.
        show: Call ``plt.show()`` before returning.
        ax: Optional Matplotlib axes to draw into.

    Returns:
        ``(fig, ax, graph)`` so callers can save or further customize the plot.
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ImportError(
            "view_graph() requires matplotlib and networkx. Install both packages "
            "to visualize GNN graph structures."
        ) from exc

    graph = get_networkx_graph(
        source,
        graph_view=graph_view,
        aggregation=aggregation,
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    positions = _graph_layout_positions(graph, layout=layout, seed=seed, nx=nx)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color="#f4f7fb",
        edgecolors="#1f2937",
        linewidths=1.2,
        node_size=node_size,
    )

    edge_styles = {
        "actuated": {"edge_color": "#2563eb", "style": "solid", "width": 2.0},
        "connector": {"edge_color": "#6b7280", "style": "dashed", "width": 1.6},
        "structural": {"edge_color": "#111827", "style": "solid", "width": 1.8},
    }
    if show_edge_types:
        for edge_type, style in edge_styles.items():
            edges = [
                (from_node, to_node)
                for from_node, to_node, edge_data in graph.edges(data=True)
                if edge_data.get("type", "structural") == edge_type
            ]
            if not edges:
                continue
            nx.draw_networkx_edges(graph, positions, edgelist=edges, ax=ax, **style)
    else:
        nx.draw_networkx_edges(graph, positions, ax=ax, edge_color="#111827", width=1.8)

    if with_labels:
        nx.draw_networkx_labels(
            graph,
            positions,
            labels={node_name: node_name for node_name in graph.nodes},
            ax=ax,
            font_size=font_size,
        )

    ax.set_title(f"{graph_view.capitalize()} graph")
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="datalim")

    if show_edge_types:
        present_edge_types = {
            edge_data.get("type", "structural") for _, _, edge_data in graph.edges(data=True)
        }
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=edge_styles[edge_type]["edge_color"],
                linestyle=edge_styles[edge_type]["style"],
                linewidth=edge_styles[edge_type]["width"],
                label=edge_type,
            )
            for edge_type in ("actuated", "connector", "structural")
            if edge_type in present_edge_types
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, loc="best")

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax, graph


def _coerce_model(source: ModelSource) -> MujocoModel:
    if isinstance(source, MujocoModel):
        return source
    return MujocoModel(source)


def _graph_node_names(model: MujocoModel, graph_view: GraphView) -> list[str]:
    if graph_view == "physical":
        return list(model.node_names)
    if graph_view == "logical":
        return _logical_node_names(model)
    if graph_view == "control":
        return list(model.control_graph.control_node_names)
    raise ValueError("graph_view must be 'physical', 'logical', or 'control'.")


def _named_graph_edges(
    model: MujocoModel,
    graph_view: GraphView,
) -> list[tuple[str, str, str]]:
    if graph_view == "physical":
        return [
            (from_node, to_node, "structural")
            for from_node, to_node in _unique_named_edges(model.structural_edges)
        ]
    if graph_view == "logical":
        logical_edges = [
            (_logical_node_name(from_node), _logical_node_name(to_node))
            for from_node, to_node in model.structural_edges
            if _logical_node_name(from_node) != _logical_node_name(to_node)
        ]
        return [
            (from_node, to_node, "structural")
            for from_node, to_node in _unique_named_edges(logical_edges)
        ]
    if graph_view == "control":
        return [
            (edge.from_node, edge.to_node, edge.type)
            for edge in model.control_graph.edges
        ]
    raise ValueError("graph_view must be 'physical', 'logical', or 'control'.")


def _unique_named_edges(edges: list[tuple[str, str]]) -> list[tuple[str, str]]:
    unique_edges = []
    seen = set()
    for from_node, to_node in edges:
        key = tuple(sorted((from_node, to_node)))
        if from_node == to_node or key in seen:
            continue
        seen.add(key)
        unique_edges.append((from_node, to_node))
    return unique_edges


def _graph_layout_positions(
    graph: Any,
    *,
    layout: str,
    seed: int,
    nx: Any,
) -> dict[str, tuple[float, float]]:
    if layout == "physical":
        physical_positions = {
            node_name: node_data["pos"]
            for node_name, node_data in graph.nodes(data=True)
            if "pos" in node_data
        }
        if len(physical_positions) == graph.number_of_nodes():
            return physical_positions
        layout = "spring"

    if layout == "spring":
        return nx.spring_layout(graph, seed=seed)
    if layout == "kamada_kawai":
        return nx.kamada_kawai_layout(graph)
    if layout == "spectral":
        return nx.spectral_layout(graph)
    raise ValueError("layout must be 'spring', 'kamada_kawai', 'spectral', or 'physical'.")


def _get_control_edge_index(model: MujocoModel) -> np.ndarray:
    node_names = model.control_graph.control_node_names
    node_index = {node_name: index for index, node_name in enumerate(node_names)}
    edges = []
    for edge in model.control_graph.edges:
        if edge.from_node not in node_index or edge.to_node not in node_index:
            continue
        from_index = node_index[edge.from_node]
        to_index = node_index[edge.to_node]
        if from_index == to_index:
            continue
        edges.append([from_index, to_index])
        edges.append([to_index, from_index])

    if not edges:
        return np.empty((2, 0), dtype=np.int64)
    return np.array(edges, dtype=np.int64).T


def _get_control_node_features(model: MujocoModel) -> np.ndarray:
    positions = model.get_control_node_position_matrix()
    velocities = model.get_control_node_linear_velocity_matrix()
    if positions.size == 0 or velocities.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    return np.concatenate([positions, velocities], axis=1).astype(np.float32)


def _logical_node_name(node_name: str) -> str:
    return node_name.split("_tri_", 1)[0].split("_route_", 1)[0]


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
