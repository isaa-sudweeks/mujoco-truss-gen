from __future__ import annotations

import importlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from mujoco_truss_gen.mujoco_model.model_types import NodeDict, ShapeDict

_STL_SOURCE_METADATA_KEY = "_mujoco_truss_gen_source"
_STL_SOURCE_METADATA_VALUE = "stl"


def stl_to_shape_dict(
    filename: str | Path,
    *,
    merge_tolerance: float = 1e-6,
    target_edge_length: float | None = None,
    target_node_count: int | None = None,
    scale: float = 1.0,
    offset: list[float] | tuple[float, float, float] | np.ndarray | None = None,
    preview: bool = False,
    preview_block: bool = True,
    verbose: bool = True,
    progress_stream: TextIO | None = None,
) -> tuple[NodeDict, ShapeDict]:
    """Convert an STL mesh into routed tube dictionaries.

    The returned ``shape_dict`` uses one edge-disjoint route per generated tube.
    All adjacent edges in each route are marked active.
    """
    _validate_options(
        merge_tolerance=merge_tolerance,
        target_edge_length=target_edge_length,
        target_node_count=target_node_count,
        scale=scale,
        offset=offset,
    )

    progress = _Progress(verbose, progress_stream)
    progress.log(f"Loading STL mesh from {filename!s}.")
    trimesh = _import_trimesh()
    mesh = trimesh.load_mesh(filename)
    vertices, faces = _mesh_arrays(mesh)
    progress.log(
        f"Loaded mesh with {len(vertices):,} vertices and {len(faces):,} faces "
        f"after {progress.elapsed():.2f}s."
    )

    progress.log("Transforming vertices and merging nearby duplicates.")
    transformed_vertices = _transform_vertices(vertices, scale=scale, offset=offset)
    merged_vertices, vertex_map = _merge_vertices(
        transformed_vertices,
        merge_tolerance,
        progress=progress,
    )
    progress.log(
        f"Merged to {len(merged_vertices):,} unique graph nodes "
        f"after {progress.elapsed():.2f}s."
    )

    progress.log("Extracting graph edges from mesh faces.")
    edges = _edges_from_faces(faces, vertex_map, progress=progress)

    if not edges:
        raise ValueError("STL mesh must contain at least one non-degenerate edge.")
    progress.log(f"Extracted {len(edges):,} unique graph edges after {progress.elapsed():.2f}s.")

    progress.log("Simplifying graph.")
    simplified_vertices, simplified_edges = _simplify_graph(
        merged_vertices,
        edges,
        target_edge_length=target_edge_length,
        target_node_count=target_node_count,
    )
    progress.log(
        f"Simplified graph to {len(simplified_vertices):,} nodes and "
        f"{len(simplified_edges):,} edges after {progress.elapsed():.2f}s."
    )

    progress.log("Decomposing graph into routed tube paths.")
    node_dict, index_to_name = _node_dict(simplified_vertices)
    shape_dict = _shape_dict_from_edges(
        simplified_edges,
        index_to_name,
        simplified_vertices,
    )
    if not shape_dict:
        raise ValueError("STL mesh did not produce any routed tube paths.")
    route_edges = sum(len(shape["route"]) - 1 for shape in shape_dict.values())
    progress.log(
        f"Generated {len(shape_dict):,} routed tube paths covering {route_edges:,} route edges "
        f"in {progress.elapsed():.2f}s."
    )

    if preview:
        progress.log("Opening STL routing preview.")
        _preview_routed_graph(
            merged_vertices,
            edges,
            simplified_vertices,
            simplified_edges,
            node_dict,
            shape_dict,
            block=preview_block,
        )

    return node_dict, shape_dict


class _Progress:
    def __init__(self, enabled: bool, stream: TextIO | None) -> None:
        self.enabled = enabled
        self.stream = stream if stream is not None else sys.stderr
        self.start = time.perf_counter()
        self.last_update = self.start

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def log(self, message: str) -> None:
        if not self.enabled:
            return
        print(f"[mujoco-truss-gen] {message}", file=self.stream, flush=True)

    def log_interval(self, message: str, *, interval: float = 2.0) -> None:
        now = time.perf_counter()
        if now - self.last_update < interval:
            return
        self.last_update = now
        self.log(message)


def _import_trimesh() -> Any:
    try:
        return importlib.import_module("trimesh")
    except ImportError as exc:
        raise ImportError(
            "STL mesh import requires the optional 'mesh' dependencies. "
            'Install them with: pip install "mujoco-truss-gen[mesh]"'
        ) from exc


def _validate_options(
    *,
    merge_tolerance: float,
    target_edge_length: float | None,
    target_node_count: int | None,
    scale: float,
    offset: list[float] | tuple[float, float, float] | np.ndarray | None,
) -> None:
    if merge_tolerance < 0:
        raise ValueError("merge_tolerance must be non-negative.")
    if scale <= 0:
        raise ValueError("scale must be greater than zero.")
    if target_edge_length is not None and target_node_count is not None:
        raise ValueError("Pass either target_edge_length or target_node_count, not both.")
    if target_edge_length is not None and target_edge_length <= 0:
        raise ValueError("target_edge_length must be greater than zero.")
    if target_node_count is not None and target_node_count < 2:
        raise ValueError("target_node_count must be at least 2.")
    if offset is not None:
        try:
            offset_array = np.array(offset, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("offset must contain exactly three numbers.") from exc
        if offset_array.shape != (3,):
            raise ValueError("offset must contain exactly three numbers.")


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(getattr(mesh, "vertices", []), dtype=float)
    faces = np.array(getattr(mesh, "faces", []), dtype=int)

    if vertices.size == 0:
        raise ValueError("STL mesh must contain vertices.")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("STL mesh vertices must be an Nx3 array.")
    if faces.size == 0:
        raise ValueError("STL mesh must contain triangular faces.")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("STL mesh faces must be an Nx3 array.")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("STL mesh faces reference vertices outside the mesh.")

    return vertices, faces


def _transform_vertices(
    vertices: np.ndarray,
    *,
    scale: float,
    offset: list[float] | tuple[float, float, float] | np.ndarray | None,
) -> np.ndarray:
    transformed = vertices * scale
    if offset is not None:
        transformed = transformed + np.array(offset, dtype=float)
    return transformed


def _merge_vertices(
    vertices: np.ndarray,
    merge_tolerance: float,
    *,
    progress: _Progress,
) -> tuple[list[np.ndarray], dict[int, int]]:
    merged: list[np.ndarray] = []
    vertex_map: dict[int, int] = {}

    if merge_tolerance == 0:
        exact_index: dict[tuple[float, float, float], int] = {}
        for index, vertex in enumerate(vertices):
            key = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
            target_index = exact_index.get(key)
            if target_index is None:
                target_index = len(merged)
                exact_index[key] = target_index
                merged.append(np.array(vertex, dtype=float))
            vertex_map[index] = target_index
            progress.log_interval(
                f"Merged {index + 1:,}/{len(vertices):,} vertices; "
                f"{len(merged):,} unique so far."
            )
        return merged, vertex_map

    grid: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, vertex in enumerate(vertices):
        target_index = None
        cell = _grid_cell(vertex, merge_tolerance)
        for neighbor_cell in _neighbor_cells(cell):
            for merged_index in grid.get(neighbor_cell, []):
                if np.linalg.norm(vertex - merged[merged_index]) <= merge_tolerance:
                    target_index = merged_index
                    break
            if target_index is not None:
                break

        if target_index is None:
            target_index = len(merged)
            merged.append(np.array(vertex, dtype=float))
            grid[cell].append(target_index)

        vertex_map[index] = target_index
        progress.log_interval(
            f"Merged {index + 1:,}/{len(vertices):,} vertices; "
            f"{len(merged):,} unique so far."
        )

    return merged, vertex_map


def _edges_from_faces(
    faces: np.ndarray,
    vertex_map: dict[int, int],
    *,
    progress: _Progress,
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for index, face in enumerate(faces):
        mapped_face = [vertex_map[int(vertex_index)] for vertex_index in face]
        for edge in (
            (mapped_face[0], mapped_face[1]),
            (mapped_face[1], mapped_face[2]),
            (mapped_face[2], mapped_face[0]),
        ):
            if edge[0] == edge[1]:
                continue
            edges.add(_edge_key(edge[0], edge[1]))
        progress.log_interval(
            f"Processed {index + 1:,}/{len(faces):,} faces; "
            f"{len(edges):,} unique edges so far."
        )
    return edges


def _simplify_graph(
    vertices: list[np.ndarray],
    edges: set[tuple[int, int]],
    *,
    target_edge_length: float | None,
    target_node_count: int | None,
) -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    if target_edge_length is None and target_node_count is None:
        return vertices, edges

    if target_node_count is not None:
        return _simplify_graph_to_node_count(vertices, edges, target_node_count)

    if target_edge_length is not None:
        return _simplify_graph_by_voxel_size(vertices, edges, target_edge_length)

    return vertices, edges


def _simplify_graph_to_node_count(
    vertices: list[np.ndarray],
    edges: set[tuple[int, int]],
    target_node_count: int,
) -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    if target_node_count >= len(vertices):
        return vertices, edges

    points = np.array(vertices, dtype=float)
    center_indices = _farthest_point_indices(points, target_node_count)
    labels = _assign_points_to_centers(points, points[center_indices])
    return _rebuild_graph_from_labels(
        vertices,
        edges,
        labels,
        target_node_count,
        representatives=points[center_indices],
    )


def _simplify_graph_by_voxel_size(
    vertices: list[np.ndarray],
    edges: set[tuple[int, int]],
    voxel_size: float,
) -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    points = np.array(vertices, dtype=float)
    origin = points.min(axis=0)
    cells = np.floor((points - origin) / voxel_size).astype(int)
    cell_to_label: dict[tuple[int, int, int], int] = {}
    labels = np.empty(len(vertices), dtype=int)
    for index, cell in enumerate(cells):
        key = (int(cell[0]), int(cell[1]), int(cell[2]))
        label = cell_to_label.get(key)
        if label is None:
            label = len(cell_to_label)
            cell_to_label[key] = label
        labels[index] = label

    return _rebuild_graph_from_labels(vertices, edges, labels, len(cell_to_label))


def _farthest_point_indices(points: np.ndarray, count: int) -> np.ndarray:
    center = points.mean(axis=0)
    first_index = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    selected = np.empty(count, dtype=int)
    selected[0] = first_index
    min_squared_distances = np.sum((points - points[first_index]) ** 2, axis=1)

    for selected_count in range(1, count):
        next_index = int(np.argmax(min_squared_distances))
        selected[selected_count] = next_index
        squared_distances = np.sum((points - points[next_index]) ** 2, axis=1)
        min_squared_distances = np.minimum(min_squared_distances, squared_distances)

    return selected


def _assign_points_to_centers(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    labels = np.empty(len(points), dtype=int)
    chunk_size = max(1, 2_000_000 // len(centers))
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        chunk = points[start:stop]
        squared_distances = np.sum(
            (chunk[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        labels[start:stop] = np.argmin(squared_distances, axis=1)
    return labels


def _rebuild_graph_from_labels(
    vertices: list[np.ndarray],
    edges: set[tuple[int, int]],
    labels: np.ndarray,
    label_count: int,
    representatives: np.ndarray | None = None,
) -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    sums = np.zeros((label_count, 3), dtype=float)
    counts = np.zeros(label_count, dtype=int)
    for index, vertex in enumerate(vertices):
        label = int(labels[index])
        sums[label] += vertex
        counts[label] += 1

    old_label_to_new: dict[int, int] = {}
    new_vertices = []
    for label in range(label_count):
        if counts[label] == 0:
            continue
        old_label_to_new[label] = len(new_vertices)
        if representatives is None:
            new_vertices.append(sums[label] / counts[label])
        else:
            new_vertices.append(np.array(representatives[label], dtype=float))

    new_edges: set[tuple[int, int]] = set()
    for first, second in edges:
        first_label = int(labels[first])
        second_label = int(labels[second])
        if first_label == second_label:
            continue
        new_edges.add(_edge_key(old_label_to_new[first_label], old_label_to_new[second_label]))

    if not new_edges:
        raise ValueError("Simplification removed all graph edges.")

    return new_vertices, new_edges


def _contract_graph_edges(
    vertices: list[np.ndarray],
    edges: set[tuple[int, int]],
    *,
    target_edge_length: float | None,
    target_node_count: int | None,
) -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    parent = list(range(len(vertices)))
    member_count = [1] * len(vertices)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> bool:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return False
        if root_second < root_first:
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        member_count[root_first] += member_count[root_second]
        return True

    sorted_edges = sorted(
        edges,
        key=lambda edge: (
            float(np.linalg.norm(vertices[edge[1]] - vertices[edge[0]])),
            edge[0],
            edge[1],
        ),
    )

    if target_edge_length is not None:
        for first, second in sorted_edges:
            if np.linalg.norm(vertices[second] - vertices[first]) < target_edge_length:
                union(first, second)

    if target_node_count is not None:
        component_count = len(vertices)
        for first, second in sorted_edges:
            if component_count <= target_node_count:
                break
            if union(first, second):
                component_count -= 1

    return _rebuild_graph(vertices, edges, find)


def _rebuild_graph(
    vertices: list[np.ndarray],
    edges: set[tuple[int, int]],
    find: Any,
) -> tuple[list[np.ndarray], set[tuple[int, int]]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(vertices)):
        groups[find(index)].append(index)

    sorted_groups = sorted(groups.values(), key=lambda group: min(group))
    old_to_new = {}
    new_vertices = []
    for new_index, group in enumerate(sorted_groups):
        for old_index in group:
            old_to_new[old_index] = new_index
        new_vertices.append(np.mean([vertices[old_index] for old_index in group], axis=0))

    new_edges: set[tuple[int, int]] = set()
    for first, second in edges:
        new_first = old_to_new[first]
        new_second = old_to_new[second]
        if new_first == new_second:
            continue
        new_edges.add(_edge_key(new_first, new_second))

    if not new_edges:
        raise ValueError("Simplification removed all graph edges.")

    return new_vertices, new_edges


def _node_dict(vertices: list[np.ndarray]) -> tuple[NodeDict, dict[int, str]]:
    node_dict: NodeDict = {}
    index_to_name = {}
    for index, vertex in enumerate(vertices, start=1):
        name = f"node_{index}"
        index_to_name[index - 1] = name
        node_dict[name] = [float(value) for value in vertex]
    return node_dict, index_to_name


def _shape_dict_from_edges(
    edges: set[tuple[int, int]],
    index_to_name: dict[int, str],
    vertices: list[np.ndarray],
) -> ShapeDict:
    shape_dict: ShapeDict = {}
    path_index = 1
    global_target_length = _target_route_length(edges, vertices)

    for component_edges in _connected_edge_components(edges):
        for trail in _decompose_component_edges(
            component_edges,
            vertices,
            target_route_length=global_target_length,
        ):
            if len(trail) < 2:
                continue
            route = [index_to_name[index] for index in trail]
            active_edges = [
                [from_node, to_node]
                for from_node, to_node in zip(route, route[1:], strict=False)
            ]
            shape_dict[f"path_{path_index}"] = {
                "route": route,
                "active_edges": active_edges,
                "disable_route_length_constraint": True,
                _STL_SOURCE_METADATA_KEY: _STL_SOURCE_METADATA_VALUE,
            }
            path_index += 1

    return shape_dict


def _preview_routed_graph(
    original_vertices: list[np.ndarray],
    original_edges: set[tuple[int, int]],
    simplified_vertices: list[np.ndarray],
    simplified_edges: set[tuple[int, int]],
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    *,
    block: bool,
) -> None:
    if _requires_external_preview_process():
        _preview_routed_graph_external(
            original_vertices,
            original_edges,
            simplified_vertices,
            simplified_edges,
            node_dict,
            shape_dict,
            block=block,
        )
        return

    _render_routed_graph_preview(
        original_vertices,
        original_edges,
        simplified_vertices,
        simplified_edges,
        node_dict,
        shape_dict,
        block=block,
    )


def _render_routed_graph_preview(
    original_vertices: list[np.ndarray],
    original_edges: set[tuple[int, int]],
    simplified_vertices: list[np.ndarray],
    simplified_edges: set[tuple[int, int]],
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    *,
    block: bool,
) -> None:
    try:
        pyplot = importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(
            "STL mesh preview requires matplotlib. Install it with: "
            'pip install "mujoco-truss-gen[mesh]" matplotlib'
        ) from exc

    figure = pyplot.figure(figsize=(12, 6))
    original_axis = figure.add_subplot(1, 2, 1, projection="3d")
    routed_axis = figure.add_subplot(1, 2, 2, projection="3d")

    _plot_edge_graph(
        original_axis,
        original_vertices,
        original_edges,
        color="0.55",
        linewidth=0.8,
        alpha=0.5,
    )
    original_axis.scatter(
        [float(vertex[0]) for vertex in original_vertices],
        [float(vertex[1]) for vertex in original_vertices],
        [float(vertex[2]) for vertex in original_vertices],
        s=8,
        c="black",
        alpha=0.55,
    )
    original_axis.set_title(
        f"Original graph\n{len(original_vertices):,} nodes, {len(original_edges):,} edges"
    )

    _plot_edge_graph(
        routed_axis,
        simplified_vertices,
        simplified_edges,
        color="0.82",
        linewidth=0.9,
        alpha=0.5,
    )
    _plot_routes(routed_axis, node_dict, shape_dict)
    routed_axis.set_title(
        "Simplified routed paths\n"
        f"{len(simplified_vertices):,} nodes, {len(simplified_edges):,} edges, "
        f"{len(shape_dict):,} paths"
    )

    for axis, vertices in (
        (original_axis, original_vertices),
        (routed_axis, simplified_vertices),
    ):
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        _set_equal_3d_limits(axis, vertices)

    figure.suptitle("STL to MuJoCo routed tube preview")
    figure.tight_layout()
    pyplot.show(block=block)


def _requires_external_preview_process() -> bool:
    return sys.platform == "darwin"


def _preview_routed_graph_external(
    original_vertices: list[np.ndarray],
    original_edges: set[tuple[int, int]],
    simplified_vertices: list[np.ndarray],
    simplified_edges: set[tuple[int, int]],
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    *,
    block: bool,
) -> None:
    payload_path = _write_preview_payload(
        original_vertices,
        original_edges,
        simplified_vertices,
        simplified_edges,
        node_dict,
        shape_dict,
        block=block,
    )
    python_executable = _preview_python_executable()
    command = [
        python_executable,
        "-c",
        (
            "from mujoco_truss_gen.mesh_import.stl import "
            "_render_preview_payload; "
            "import sys; "
            "_render_preview_payload(sys.argv[1])"
        ),
        str(payload_path),
    ]
    try:
        if block:
            subprocess.run(command, check=True)
        else:
            subprocess.Popen(command)
    except Exception:
        payload_path.unlink(missing_ok=True)
        raise


def _write_preview_payload(
    original_vertices: list[np.ndarray],
    original_edges: set[tuple[int, int]],
    simplified_vertices: list[np.ndarray],
    simplified_edges: set[tuple[int, int]],
    node_dict: NodeDict,
    shape_dict: ShapeDict,
    *,
    block: bool,
) -> Path:
    payload_file = tempfile.NamedTemporaryFile(
        prefix="mujoco_truss_preview_",
        suffix=".npz",
        delete=False,
    )
    payload_path = Path(payload_file.name)
    payload_file.close()

    np.savez_compressed(
        payload_path,
        original_vertices=np.array(original_vertices, dtype=float),
        original_edges=np.array(sorted(original_edges), dtype=int),
        simplified_vertices=np.array(simplified_vertices, dtype=float),
        simplified_edges=np.array(sorted(simplified_edges), dtype=int),
        node_dict=json.dumps(node_dict),
        shape_dict=json.dumps(shape_dict),
        block=block,
    )
    return payload_path


def _render_preview_payload(payload_path: str | Path) -> None:
    path = Path(payload_path)
    try:
        payload = np.load(path, allow_pickle=False)
        try:
            _render_routed_graph_preview(
                [vertex for vertex in payload["original_vertices"]],
                {tuple(edge) for edge in payload["original_edges"].tolist()},
                [vertex for vertex in payload["simplified_vertices"]],
                {tuple(edge) for edge in payload["simplified_edges"].tolist()},
                json.loads(str(payload["node_dict"])),
                json.loads(str(payload["shape_dict"])),
                block=bool(payload["block"]),
            )
        finally:
            payload.close()
    finally:
        path.unlink(missing_ok=True)


def _preview_python_executable() -> str:
    candidate = Path(sys.prefix) / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def _plot_edge_graph(
    axis: Any,
    vertices: list[np.ndarray],
    edges: set[tuple[int, int]],
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    for first, second in sorted(edges):
        start = vertices[first]
        end = vertices[second]
        axis.plot(
            [float(start[0]), float(end[0])],
            [float(start[1]), float(end[1])],
            [float(start[2]), float(end[2])],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )


def _plot_routes(axis: Any, node_dict: NodeDict, shape_dict: ShapeDict) -> None:
    colormap = importlib.import_module("matplotlib.pyplot").get_cmap("tab20")
    for index, shape in enumerate(shape_dict.values()):
        route = shape["route"]
        positions = np.array([node_dict[node_name] for node_name in route], dtype=float)
        if len(positions) < 2:
            continue
        axis.plot(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=colormap(index % colormap.N),
            linewidth=2.0,
            alpha=0.9,
        )
        axis.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=colormap(index % colormap.N),
            s=12,
            alpha=0.9,
        )


def _set_equal_3d_limits(axis: Any, vertices: list[np.ndarray]) -> None:
    points = np.array(vertices, dtype=float)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = float(np.max(maximum - minimum) / 2.0)
    if radius == 0:
        radius = 0.5

    axis.set_xlim(float(center[0] - radius), float(center[0] + radius))
    axis.set_ylim(float(center[1] - radius), float(center[1] + radius))
    axis.set_zlim(float(center[2] - radius), float(center[2] + radius))


def _connected_edge_components(edges: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)

    components = []
    visited_vertices: set[int] = set()
    for start in sorted(adjacency):
        if start in visited_vertices:
            continue

        stack = [start]
        vertices = set()
        while stack:
            vertex = stack.pop()
            if vertex in vertices:
                continue
            vertices.add(vertex)
            stack.extend(sorted(adjacency[vertex] - vertices, reverse=True))

        visited_vertices.update(vertices)
        components.append(
            {
                edge
                for edge in edges
                if edge[0] in vertices and edge[1] in vertices
            }
        )

    return components


def _decompose_component_edges(
    component_edges: set[tuple[int, int]],
    vertices: list[np.ndarray],
    *,
    target_route_length: float | None = None,
) -> list[list[int]]:
    unused_edges = set(component_edges)
    adjacency: dict[int, set[int]] = defaultdict(set)
    for first, second in component_edges:
        adjacency[first].add(second)
        adjacency[second].add(first)

    trails = []
    initial_degrees = _unused_degrees(unused_edges)
    odd_vertex_count = sum(1 for degree in initial_degrees.values() if degree % 2 == 1)
    remaining_length = _total_edge_length(unused_edges, vertices)
    minimum_path_count = max(1, odd_vertex_count // 2)
    length_path_count = (
        math.ceil(remaining_length / target_route_length)
        if target_route_length is not None and target_route_length > 0
        else 1
    )
    target_path_count = max(minimum_path_count, length_path_count)
    previous_start: int | None = None

    while unused_edges:
        remaining_path_slots = max(1, target_path_count - len(trails))
        target_length = remaining_length / remaining_path_slots
        stop_at_target = remaining_path_slots > 1
        start = _choose_start_vertex(unused_edges, vertices, previous_start)
        previous_start = start

        trail = [start]
        current = start
        trail_length = 0.0
        while True:
            candidates = _unused_neighbors(current, adjacency, unused_edges)
            if not candidates:
                break

            previous = trail[-2] if len(trail) > 1 else None
            next_vertex = _choose_next_vertex(
                previous,
                current,
                candidates,
                vertices,
                unused_edges,
                adjacency,
                target_length=target_length,
                current_length=trail_length,
            )
            edge = _edge_key(current, next_vertex)
            unused_edges.remove(edge)
            edge_length = _edge_length(edge, vertices)
            remaining_length -= edge_length
            trail_length += edge_length
            trail.append(next_vertex)
            current = next_vertex

            if stop_at_target and trail_length >= target_length:
                break

        trails.append(trail)

    return trails


def _choose_start_vertex(
    unused_edges: set[tuple[int, int]],
    vertices: list[np.ndarray],
    previous_start: int | None,
) -> int:
    degrees = _unused_degrees(unused_edges)
    odd_vertices = sorted(vertex for vertex, degree in degrees.items() if degree % 2 == 1)
    candidates = odd_vertices if odd_vertices else sorted(degrees)

    if previous_start is None:
        center = np.mean([vertices[index] for index in candidates], axis=0)
        return min(
            candidates,
            key=lambda index: (
                -float(np.linalg.norm(vertices[index] - center)),
                float(vertices[index][0]),
                float(vertices[index][1]),
                float(vertices[index][2]),
                index,
            ),
        )

    return max(
        candidates,
        key=lambda index: (
            float(np.linalg.norm(vertices[index] - vertices[previous_start])),
            -index,
        ),
    )


def _choose_next_vertex(
    previous: int | None,
    current: int,
    candidates: list[int],
    vertices: list[np.ndarray],
    unused_edges: set[tuple[int, int]],
    adjacency: dict[int, set[int]],
    *,
    target_length: float,
    current_length: float,
) -> int:
    return max(
        candidates,
        key=lambda candidate: _next_vertex_score(
            previous,
            current,
            candidate,
            vertices,
            unused_edges,
            adjacency,
            target_length=target_length,
            current_length=current_length,
        ),
    )


def _next_vertex_score(
    previous: int | None,
    current: int,
    candidate: int,
    vertices: list[np.ndarray],
    unused_edges: set[tuple[int, int]],
    adjacency: dict[int, set[int]],
    *,
    target_length: float,
    current_length: float,
) -> tuple[float, float, float, int]:
    edge = _edge_key(current, candidate)
    edge_length = _edge_length(edge, vertices)
    target_error = abs((current_length + edge_length) - target_length)
    target_score = -target_error / max(target_length, 1e-12)

    if previous is None:
        straightness = 0.0
    else:
        incoming = vertices[current] - vertices[previous]
        outgoing = vertices[candidate] - vertices[current]
        incoming_norm = float(np.linalg.norm(incoming))
        outgoing_norm = float(np.linalg.norm(outgoing))
        straightness = (
            float(np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))
            if incoming_norm > 0 and outgoing_norm > 0
            else 0.0
        )

    candidate_degree = len(_unused_neighbors(candidate, adjacency, unused_edges))
    endpoint_score = 0.0
    if candidate_degree <= 1:
        endpoint_score = 1.25 if current_length >= target_length * 0.75 else -0.5

    return (
        straightness + 0.5 * target_score + endpoint_score,
        edge_length,
        -float(candidate),
        -candidate,
    )


def _unused_neighbors(
    vertex: int,
    adjacency: dict[int, set[int]],
    unused_edges: set[tuple[int, int]],
) -> list[int]:
    return sorted(
        neighbor
        for neighbor in adjacency[vertex]
        if _edge_key(vertex, neighbor) in unused_edges
    )


def _total_edge_length(edges: set[tuple[int, int]], vertices: list[np.ndarray]) -> float:
    return sum(_edge_length(edge, vertices) for edge in edges)


def _target_route_length(edges: set[tuple[int, int]], vertices: list[np.ndarray]) -> float:
    degrees = _unused_degrees(edges)
    odd_vertex_count = sum(1 for degree in degrees.values() if degree % 2 == 1)
    target_path_count = max(1, odd_vertex_count // 2)
    return _total_edge_length(edges, vertices) / target_path_count


def _edge_length(edge: tuple[int, int], vertices: list[np.ndarray]) -> float:
    return float(np.linalg.norm(vertices[edge[1]] - vertices[edge[0]]))


def _unused_degrees(edges: set[tuple[int, int]]) -> dict[int, int]:
    degrees: dict[int, int] = defaultdict(int)
    for first, second in edges:
        degrees[first] += 1
        degrees[second] += 1
    return degrees


def _edge_key(first: int, second: int) -> tuple[int, int]:
    if first < second:
        return first, second
    return second, first


def _grid_cell(vertex: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    cell = np.floor(vertex / cell_size).astype(int)
    return int(cell[0]), int(cell[1]), int(cell[2])


def _neighbor_cells(cell: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    x, y, z = cell
    return [
        (x + dx, y + dy, z + dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
