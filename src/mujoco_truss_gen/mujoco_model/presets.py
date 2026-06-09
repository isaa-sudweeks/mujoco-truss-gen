from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from functools import cache
from itertools import combinations

import numpy as np

from mujoco_truss_gen.mujoco_model.model_types import NodeDict, ShapeDict, TriangleDict


def _validate_scale(scale: float) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be greater than zero.")
    return scale


def _scale_node_dict(node_dict: NodeDict, scale: float) -> NodeDict:
    scale = _validate_scale(scale)
    return {
        node_name: [scale * coordinate for coordinate in position]
        for node_name, position in node_dict.items()
    }


def get_octahedron_definition(scale: float = 1.0) -> tuple[NodeDict, TriangleDict]:
    node_dict = {
        "node_1": [0.0, 0.0, 0.1],
        "node_2": [1.0, 0.0, 0.1],
        "node_3": [0.5, 0.8660, 0.1],
        "node_4": [0.5, -0.2887, 0.9165],
        "node_5": [0.0, 0.5774, 0.9165],
        "node_6": [1.0, 0.5774, 0.9165],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_4", "node_1"],
        "triangle_2": ["node_1", "node_5", "node_3", "node_1"],
        "triangle_3": ["node_3", "node_6", "node_2", "node_6"],
        "triangle_4": ["node_4", "node_6", "node_5", "node_6"],
    }
    return _scale_node_dict(node_dict, scale), triangle_dict


def get_icosahedron_definition(scale: float = 1.0) -> tuple[NodeDict, TriangleDict]:
    phi = (1.0 + 5.0**0.5) / 2.0
    vertex_scale = 0.5
    z_offset = 0.95
    vertices = [
        (-1.0, phi, 0.0),
        (1.0, phi, 0.0),
        (-1.0, -phi, 0.0),
        (1.0, -phi, 0.0),
        (0.0, -1.0, phi),
        (0.0, 1.0, phi),
        (0.0, -1.0, -phi),
        (0.0, 1.0, -phi),
        (phi, 0.0, -1.0),
        (phi, 0.0, 1.0),
        (-phi, 0.0, -1.0),
        (-phi, 0.0, 1.0),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    node_dict = {
        f"node_{index}": [vertex_scale * x, vertex_scale * y, vertex_scale * z + z_offset]
        for index, (x, y, z) in enumerate(vertices, start=1)
    }
    triangle_dict = {
        f"triangle_{index}": [
            *(f"node_{vertex_index + 1}" for vertex_index in face),
            f"node_{face[0] + 1}",
        ]
        for index, face in enumerate(faces, start=1)
    }

    return _scale_node_dict(node_dict, scale), triangle_dict


def get_solar_array_definition(scale: float = 1.0) -> tuple[NodeDict, TriangleDict]:
    node_dict = {
        "node_1": [0.0, 0.0, 0.1],
        "node_2": [1.0, 0.0, 0.1],
        "node_3": [0.5, 0.8660, 0.1],
        "node_4": [0.5, -0.2887, 0.9165],
        "node_5": [0.0, 0.5774, 0.9165],
        "node_6": [1.0, 0.5774, 0.9165],
        "node_7": [0.0, 0.0, 1.8330],
        "node_8": [1.0, 0.0, 1.8330],
        "node_9": [0.5, 0.8660, 1.8330],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_4", "node_1"],
        "triangle_2": ["node_1", "node_5", "node_3", "node_1"],
        "triangle_3": ["node_3", "node_6", "node_2", "node_6"],
        "triangle_4": ["node_4", "node_6", "node_8", "node_8"],
        "triangle_5": ["node_5", "node_6", "node_9", "node_9"],
        "triangle_6": ["node_5", "node_4", "node_7", "node_7"],
        "triangle_7": ["node_7", "node_8", "node_9", "node_9"],
    }
    return _scale_node_dict(node_dict, scale), triangle_dict


def get_tetrahedron_definition(scale: float = 1.0) -> tuple[NodeDict, ShapeDict]:
    node_dict = {
        "node_1": [0.0, 0.0, 0.1],
        "node_2": [1.0, 0.0, 0.1],
        "node_3": [0.5, 0.8660, 0.1],
        "node_4": [0.5, np.sqrt(3) / 6, 0.1 + np.sqrt(2 / 3)],
    }
    shape_dict = {
        "path_1": {
            "route": ["node_1", "node_2", "node_4", "node_3"],
            "active_edges": [
                ["node_1", "node_2"],
                ["node_4", "node_3"],
            ],
        },
        "path_2": {
            "route": ["node_2", "node_3", "node_1", "node_4"],
            "active_edges": [
                ["node_2", "node_3"],
                ["node_1", "node_4"],
            ],
        },
    }

    return _scale_node_dict(node_dict, scale), shape_dict


USEVITCH_GRAPH_LABELS: tuple[int, ...] = (
    1514879,
    210272254,
    212365307,
    54501547959,
    64702095263,
    49530656767,
    53827448765,
    54364254015,
    44565393342,
    44968308287,
    44137822173,
    60202270686,
    60243677150,
)
USEVITCH_EMBEDDING_TRIALS = 64
USEVITCH_DISCONNECTED_DISTANCE = 10.0
USEVITCH_MDS_WCRI_FALLBACK_THRESHOLD = 1e-4
USEVITCH_MDS_DEFAULT_DISTANCE_PARAMETERS = (0.0, 1.0, USEVITCH_DISCONNECTED_DISTANCE)
USEVITCH_MDS_FALLBACK_DISTANCE_PARAMETER_SETS: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.0, 1.5),
    (0.0, 1.0, 2.0),
    (0.5, 1.5, 1.5),
    (0.5, 1.5, 2.0),
    (0.5, 1.5, 2.5),
    (0.5, 1.5, 3.0),
    (0.75, 1.25, 1.5),
    (0.75, 1.25, 2.0),
    (0.75, 1.25, 3.0),
)
USEVITCH_MDS_MAX_ITERATIONS = 300
USEVITCH_MDS_TOLERANCE = 1e-9
USEVITCH_WCRI_TIE_RELATIVE_TOLERANCE = 0.01


def get_usevitch_graph_definition(
    graph_label: int,
    partition_index: int = 1,
    scale: float = 1.0,
) -> tuple[NodeDict, TriangleDict]:
    """Return a Fig. 3 triangle-decomposable graph from Usevitch et al. (2025).

    ``graph_label`` is the paper's decimal encoding of the upper triangular
    adjacency matrix. ``partition_index`` is 1-based because several graphs in
    the paper have multiple valid edge-disjoint triangle partitions. Partitions
    are recomputed with the paper's exhaustive exact-cover criterion: enumerate
    graph triangles, then select edge-disjoint triangles that cover every graph
    edge exactly once. The integer programming formulations from the paper are
    not used for these small built-in preset graphs.
    """
    scale = _validate_scale(scale)
    graph_label = int(graph_label)
    partition_index = int(partition_index)
    if partition_index < 1:
        raise ValueError("partition_index must be greater than zero.")

    node_count = _node_count_from_usevitch_label(graph_label)
    edges = _usevitch_edges_from_label(graph_label, node_count)
    partitions = _triangle_partitions(edges, node_count)
    if not partitions:
        raise ValueError(f"Graph label {graph_label} has no triangle partition.")
    if partition_index > len(partitions):
        raise ValueError(
            f"Graph label {graph_label} has {len(partitions)} partition(s); "
            f"got partition_index={partition_index}."
        )

    node_dict = _usevitch_embedding(graph_label, node_count, tuple(sorted(edges)))
    triangle_dict = {
        f"triangle_{index}": [
            *(f"node_{node + 1}" for node in triangle),
            f"node_{triangle[0] + 1}",
        ]
        for index, triangle in enumerate(partitions[partition_index - 1], start=1)
    }
    return _scale_node_dict(node_dict, scale), triangle_dict


def _node_count_from_usevitch_label(graph_label: int) -> int:
    for node_count in (7, 8, 9):
        edge_slots = node_count * (node_count - 1) // 2
        if graph_label < 2**edge_slots:
            edges = _usevitch_edges_from_label(graph_label, node_count)
            if len(edges) == 3 * node_count - 6:
                return node_count
    raise ValueError(f"Unknown Usevitch graph label: {graph_label}")


def _usevitch_edges_from_label(graph_label: int, node_count: int) -> set[tuple[int, int]]:
    edge_slots = node_count * (node_count - 1) // 2
    bit_string = bin(graph_label)[2:].zfill(edge_slots)[::-1]
    return {
        (first, second)
        for bit, (first, second) in zip(bit_string, combinations(range(node_count), 2), strict=True)
        if bit == "1"
    }


def _triangle_partitions(
    edges: set[tuple[int, int]],
    node_count: int,
) -> list[list[tuple[int, int, int]]]:
    """Enumerate all exact covers of graph edges by 3-cycles."""
    candidate_triangles = [
        triangle
        for triangle in combinations(range(node_count), 3)
        if all(tuple(sorted(edge)) in edges for edge in combinations(triangle, 2))
    ]
    triangle_edges = [
        {tuple(sorted(edge)) for edge in combinations(triangle, 2)}
        for triangle in candidate_triangles
    ]
    edge_to_triangles: dict[tuple[int, int], list[int]] = {edge: [] for edge in edges}
    for triangle_index, triangle_edge_set in enumerate(triangle_edges):
        for edge in triangle_edge_set:
            edge_to_triangles[edge].append(triangle_index)

    partitions: list[list[tuple[int, int, int]]] = []

    def search(chosen: list[int], remaining_edges: set[tuple[int, int]]) -> None:
        if not remaining_edges:
            partitions.append([candidate_triangles[index] for index in chosen])
            return

        next_edge = min(
            remaining_edges,
            key=lambda edge: sum(
                triangle_edges[index] <= remaining_edges for index in edge_to_triangles[edge]
            ),
        )
        for triangle_index in edge_to_triangles[next_edge]:
            triangle_edge_set = triangle_edges[triangle_index]
            if triangle_edge_set <= remaining_edges:
                search(chosen + [triangle_index], remaining_edges - triangle_edge_set)

    search([], set(edges))
    return partitions


def _usevitch_embedding(
    graph_label: int,
    node_count: int,
    edges: tuple[tuple[int, int], ...],
) -> NodeDict:
    coordinates = _best_usevitch_embedding(graph_label, node_count, edges)
    node_dict = {}
    for index, position in enumerate(coordinates, start=1):
        node_dict[f"node_{index}"] = [float(coordinate) for coordinate in position]
    return node_dict


@cache
def _best_usevitch_embedding(
    graph_label: int,
    node_count: int,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    best_coordinates: np.ndarray | None = None
    best_wcri = -np.inf
    best_edge_rms_error = np.inf

    best_coordinates, best_wcri, best_edge_rms_error = _search_usevitch_mds_parameters(
        node_count,
        edges,
        np.random.default_rng(graph_label),
        USEVITCH_MDS_DEFAULT_DISTANCE_PARAMETERS,
        best_coordinates,
        best_wcri,
        best_edge_rms_error,
    )
    if best_coordinates is not None and best_wcri > USEVITCH_MDS_WCRI_FALLBACK_THRESHOLD:
        return _normalize_usevitch_embedding(best_coordinates)

    for parameter_index, distance_parameters in enumerate(
        USEVITCH_MDS_FALLBACK_DISTANCE_PARAMETER_SETS,
        start=1,
    ):
        best_coordinates, best_wcri, best_edge_rms_error = _search_usevitch_mds_parameters(
            node_count,
            edges,
            np.random.default_rng([graph_label, parameter_index]),
            distance_parameters,
            best_coordinates,
            best_wcri,
            best_edge_rms_error,
        )

    if best_coordinates is None:
        raise ValueError(f"Could not embed Usevitch graph label {graph_label}.")
    return _normalize_usevitch_embedding(best_coordinates)


def _search_usevitch_mds_parameters(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    distance_parameters: tuple[float, float, float],
    best_coordinates: np.ndarray | None,
    best_wcri: float,
    best_edge_rms_error: float,
) -> tuple[np.ndarray | None, float, float]:
    (
        connected_distance_min,
        connected_distance_max,
        disconnected_distance,
    ) = distance_parameters
    for _ in range(USEVITCH_EMBEDDING_TRIALS + 1):
        distances = _random_usevitch_distance_matrix(
            node_count,
            edges,
            rng,
            connected_distance_min=connected_distance_min,
            connected_distance_max=connected_distance_max,
            disconnected_distance=disconnected_distance,
        )
        coordinates = _metric_mds(
            distances,
            dimensions=3,
            max_iterations=USEVITCH_MDS_MAX_ITERATIONS,
            tolerance=USEVITCH_MDS_TOLERANCE,
        )
        coordinates = _normalize_usevitch_candidate_edge_lengths(coordinates, edges)
        wcri = _worst_case_rigidity_index(coordinates, edges)
        edge_rms_error = _edge_length_rms_error(coordinates, edges, target_length=1.0)
        wcri_tie_tolerance = USEVITCH_WCRI_TIE_RELATIVE_TOLERANCE * max(abs(best_wcri), 1e-12)
        if best_coordinates is None or wcri > best_wcri + wcri_tie_tolerance or (
            abs(wcri - best_wcri) <= wcri_tie_tolerance
            and edge_rms_error < best_edge_rms_error
        ):
            best_coordinates = coordinates
            best_wcri = wcri
            best_edge_rms_error = edge_rms_error

    return best_coordinates, best_wcri, best_edge_rms_error


def _random_usevitch_distance_matrix(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
    connected_distance_min: float = 0.0,
    connected_distance_max: float = 1.0,
    disconnected_distance: float = USEVITCH_DISCONNECTED_DISTANCE,
) -> np.ndarray:
    distances = np.full(
        (node_count, node_count),
        disconnected_distance,
        dtype=float,
    )
    np.fill_diagonal(distances, 0.0)
    for first, second in edges:
        edge_distance = float(rng.uniform(connected_distance_min, connected_distance_max))
        distances[first, second] = edge_distance
        distances[second, first] = edge_distance
    return distances


def _metric_mds(
    distances: np.ndarray,
    dimensions: int,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Metric MDS via SMACOF, matching mdscale-style stress minimization."""
    node_count = distances.shape[0]
    coordinates = _classical_mds(distances, dimensions=dimensions)
    coordinates -= np.mean(coordinates, axis=0)
    previous_stress = np.inf

    for _ in range(max_iterations):
        coordinate_deltas = coordinates[:, None, :] - coordinates[None, :, :]
        embedded_distances = np.linalg.norm(coordinate_deltas, axis=2)
        ratios = np.zeros_like(distances)
        np.divide(
            distances,
            embedded_distances,
            out=ratios,
            where=embedded_distances > 1e-12,
        )
        np.fill_diagonal(ratios, 0.0)

        transform = -ratios
        np.fill_diagonal(transform, -np.sum(transform, axis=1))
        updated = transform @ coordinates / node_count
        updated -= np.mean(updated, axis=0)

        updated_distances = np.linalg.norm(
            updated[:, None, :] - updated[None, :, :],
            axis=2,
        )
        residuals = updated_distances - distances
        stress = 0.5 * float(np.sum(np.square(residuals)))
        if abs(previous_stress - stress) <= tolerance * max(previous_stress, 1.0):
            coordinates = updated
            break
        coordinates = updated
        previous_stress = stress

    return coordinates


def _normalize_usevitch_candidate_edge_lengths(
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    """Scale an MDS candidate so its mean structural edge length is one."""
    edge_lengths = _edge_lengths(coordinates, edges)
    if edge_lengths.size == 0:
        return coordinates

    mean_edge_length = float(np.mean(edge_lengths))
    if mean_edge_length <= 1e-12:
        return coordinates
    return coordinates / mean_edge_length


def _edge_length_rms_error(
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    target_length: float,
) -> float:
    edge_lengths = _edge_lengths(coordinates, edges)
    if edge_lengths.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(edge_lengths - target_length))))


def _edge_lengths(
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    return np.array(
        [np.linalg.norm(coordinates[second] - coordinates[first]) for first, second in edges],
        dtype=float,
    )


def _classical_mds(distances: np.ndarray, dimensions: int) -> np.ndarray:
    node_count = distances.shape[0]
    centering = np.eye(node_count) - np.full((node_count, node_count), 1.0 / node_count)
    gram = -0.5 * centering @ np.square(distances) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    positive = np.maximum(eigenvalues[:dimensions], 0.0)
    coordinates = eigenvectors[:, :dimensions] * np.sqrt(positive)
    if coordinates.shape[1] < dimensions:
        coordinates = np.pad(coordinates, ((0, 0), (0, dimensions - coordinates.shape[1])))
    return coordinates


def _worst_case_rigidity_index(
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> float:
    rows = []
    for first, second in edges:
        delta = coordinates[second] - coordinates[first]
        row = np.zeros(coordinates.shape[0] * coordinates.shape[1], dtype=float)
        row[first * coordinates.shape[1] : (first + 1) * coordinates.shape[1]] = -delta
        row[second * coordinates.shape[1] : (second + 1) * coordinates.shape[1]] = delta
        rows.append(row)

    if not rows:
        return 0.0

    rigidity_matrix = np.vstack(rows)
    gram = rigidity_matrix.T @ rigidity_matrix
    norm = np.trace(gram)
    if norm <= 0.0:
        return 0.0

    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(gram)))
    rigid_body_modes = 6
    if eigenvalues.size <= rigid_body_modes:
        return 0.0
    return float(max(eigenvalues[rigid_body_modes] / norm, 0.0))


def _normalize_usevitch_embedding(coordinates: np.ndarray) -> np.ndarray:
    coordinates = coordinates - np.mean(coordinates, axis=0)
    coordinates[:, 2] -= np.min(coordinates[:, 2])
    coordinates[:, 2] += 0.1
    return coordinates


def _make_usevitch_preset(
    graph_label: int,
    partition_index: int,
) -> Callable[[float], tuple[NodeDict, TriangleDict]]:
    return lambda scale=1.0: get_usevitch_graph_definition(
        graph_label,
        partition_index=partition_index,
        scale=scale,
    )


def _usevitch_presets() -> dict[str, Callable[[float], tuple[NodeDict, TriangleDict]]]:
    presets: dict[str, Callable[[float], tuple[NodeDict, TriangleDict]]] = {}
    for graph_label in USEVITCH_GRAPH_LABELS:
        node_count = _node_count_from_usevitch_label(graph_label)
        partition_count = len(
            _triangle_partitions(_usevitch_edges_from_label(graph_label, node_count), node_count)
        )
        if partition_count == 1:
            presets[f"usevitch_{graph_label}"] = _make_usevitch_preset(graph_label, 1)
            continue
        for partition_index in range(1, partition_count + 1):
            presets[f"usevitch_{graph_label}_p{partition_index}"] = _make_usevitch_preset(
                graph_label,
                partition_index,
            )
    return presets


PRESETS: dict[str, Callable[[float], tuple[NodeDict, TriangleDict | ShapeDict]]] = {
    "octahedron": get_octahedron_definition,
    "icosahedron": get_icosahedron_definition,
    "solar_array": get_solar_array_definition,
    "tetrahedron": get_tetrahedron_definition,
    **_usevitch_presets(),
}


def get_preset_definition(
    structure_type: str,
    scale: float = 1.0,
) -> tuple[NodeDict, TriangleDict | ShapeDict]:
    try:
        preset = PRESETS[structure_type]
    except KeyError as exc:
        known_presets = ", ".join(sorted(PRESETS))
        message = f"Unknown structure type: {structure_type}. Known presets: {known_presets}"
        raise ValueError(message) from exc

    node_dict, structure_dict = preset(scale)
    return node_dict, deepcopy(structure_dict)
