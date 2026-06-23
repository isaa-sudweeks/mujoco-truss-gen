from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from functools import cache
from itertools import combinations

import networkx as nx
import numpy as np
from scipy.optimize import least_squares

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
EMBEDDING_GROUND_FACE_TOLERANCE = 1e-9

HENNEBERG_PRESET_SPECS: tuple[tuple[int, int], ...] = (
    (5, 1),
    (6, 1),
    (6, 2),
    (6, 3),
    (7, 1),
    (7, 3),
    (8, 1),
    (8, 2),
    (8, 3),
)
HENNEBERG_PRESET_VARIANT_COUNTS: dict[tuple[int, int], int] = {
    (5, 1): 1,
    (6, 1): 2,
    (6, 2): 1,
    (6, 3): 1,
    (7, 1): 10,
    (7, 3): 3,
    (8, 1): 85,
    (8, 2): 190,
    (8, 3): 89,
}
HENNEBERG_EMBEDDING_TRIALS = 24
HENNEBERG_NONEDGE_DISTANCE = 1.6
HENNEBERG_RIGIDITY_THRESHOLD = 1e-4
HENNEBERG_LAYOUT_ITERATIONS = 250
HENNEBERG_LAYOUT_REFINEMENT_EVALUATIONS = 400


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
        return _normalize_usevitch_embedding(best_coordinates, edges)

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
    return _normalize_usevitch_embedding(best_coordinates, edges)


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
    rigidity_matrix = _rigidity_matrix(coordinates, edges)
    if rigidity_matrix.size == 0:
        return 0.0

    gram = rigidity_matrix.T @ rigidity_matrix
    norm = np.trace(gram)
    if norm <= 0.0:
        return 0.0

    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(gram)))
    rigid_body_modes = 6
    if eigenvalues.size <= rigid_body_modes:
        return 0.0
    return float(max(eigenvalues[rigid_body_modes] / norm, 0.0))


def _rigidity_matrix(
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    rows = []
    dimensions = coordinates.shape[1]
    for first, second in edges:
        delta = coordinates[second] - coordinates[first]
        row = np.zeros(coordinates.shape[0] * dimensions, dtype=float)
        row[first * dimensions : (first + 1) * dimensions] = -delta
        row[second * dimensions : (second + 1) * dimensions] = delta
        rows.append(row)

    if not rows:
        return np.empty((0, coordinates.shape[0] * dimensions), dtype=float)
    return np.vstack(rows)


def _rigidity_matrix_rank(
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    tolerance: float = 1e-7,
) -> int:
    return int(np.linalg.matrix_rank(_rigidity_matrix(coordinates, edges), tol=tolerance))


def _normalize_usevitch_embedding(
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    coordinates = coordinates - np.mean(coordinates, axis=0)
    return _align_node_one_ground_face(
        coordinates,
        _triangular_faces_through_node_one(coordinates.shape[0], edges),
    )


def _triangular_faces_through_node_one(
    node_count: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    edge_set = frozenset(edges)
    faces = []
    for second, third in combinations(range(1, node_count), 2):
        face_edges = (
            _graph_edge_key(0, second),
            _graph_edge_key(0, third),
            _graph_edge_key(second, third),
        )
        if all(edge in edge_set for edge in face_edges):
            faces.append((0, second, third))
    return tuple(faces)


def _align_node_one_ground_face(
    coordinates: np.ndarray,
    candidate_faces: tuple[tuple[int, int, int], ...],
) -> np.ndarray:
    """Place a support face through node_1 on z=0 with node_1 at the origin."""
    coordinates = np.array(coordinates, dtype=float, copy=True)
    if coordinates.shape[0] == 0:
        return coordinates
    if coordinates.shape[0] < 3:
        return _snap_near_zero(coordinates - coordinates[0])

    ground_face = _select_node_one_ground_face(coordinates, candidate_faces)
    if ground_face is None:
        return _snap_near_zero(coordinates - coordinates[0])
    aligned, _ = _align_to_ground_face(coordinates, ground_face)
    return _snap_near_zero(aligned)


def _select_node_one_ground_face(
    coordinates: np.ndarray,
    candidate_faces: tuple[tuple[int, int, int], ...],
) -> tuple[int, int, int] | None:
    unique_candidate_faces = tuple(dict.fromkeys(_valid_node_one_faces(candidate_faces)))
    fallback_faces = tuple(
        face
        for face in combinations(range(coordinates.shape[0]), 3)
        if 0 in face and face not in unique_candidate_faces
    )

    best_fallback: tuple[float, tuple[int, int, int]] | None = None
    for faces in (unique_candidate_faces, fallback_faces):
        for face in faces:
            alignment = _try_align_to_ground_face(coordinates, face)
            if alignment is None:
                continue
            _, min_z = alignment
            if min_z >= -EMBEDDING_GROUND_FACE_TOLERANCE:
                return face
            fallback = (min_z, face)
            if best_fallback is None or fallback > best_fallback:
                best_fallback = fallback

    if best_fallback is None:
        return None
    return best_fallback[1]


def _valid_node_one_faces(
    faces: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(sorted(face)) for face in faces if len(set(face)) == 3 and 0 in face)


def _try_align_to_ground_face(
    coordinates: np.ndarray,
    face: tuple[int, int, int],
) -> tuple[np.ndarray, float] | None:
    try:
        return _align_to_ground_face(coordinates, face)
    except ValueError:
        return None


def _align_to_ground_face(
    coordinates: np.ndarray,
    face: tuple[int, int, int],
) -> tuple[np.ndarray, float]:
    anchor_node = 0
    x_axis_node = sorted(face)[-2]
    shifted = coordinates - coordinates[anchor_node]
    x_axis = shifted[x_axis_node]
    x_axis_norm = float(np.linalg.norm(x_axis))
    if x_axis_norm <= 1e-12:
        raise ValueError("Cannot align a ground face with coincident x-axis nodes.")
    x_axis = x_axis / x_axis_norm

    face_nodes = tuple(node for node in face if node != anchor_node)
    normal = np.cross(shifted[face_nodes[0]], shifted[face_nodes[1]])
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-12:
        raise ValueError("Cannot align a degenerate ground face.")
    normal = normal / normal_norm

    # Keep the x axis exactly in the ground-face plane before building the frame.
    x_axis = x_axis - float(np.dot(x_axis, normal)) * normal
    x_axis_norm = float(np.linalg.norm(x_axis))
    if x_axis_norm <= 1e-12:
        raise ValueError("Cannot align a ground face with an invalid x-axis node.")
    x_axis = x_axis / x_axis_norm

    alignments = []
    for z_axis in (normal, -normal):
        y_axis = np.cross(z_axis, x_axis)
        y_axis_norm = float(np.linalg.norm(y_axis))
        if y_axis_norm <= 1e-12:
            continue
        y_axis = y_axis / y_axis_norm
        basis = np.column_stack((x_axis, y_axis, z_axis))
        aligned = shifted @ basis
        min_z = float(np.min(aligned[:, 2]))
        total_z = float(np.sum(aligned[:, 2]))
        alignments.append((min_z, total_z, aligned))

    if not alignments:
        raise ValueError("Cannot construct a ground-face frame.")
    min_z, _, aligned = max(alignments, key=lambda option: (option[0], option[1]))
    return aligned, min_z


def _snap_near_zero(coordinates: np.ndarray) -> np.ndarray:
    coordinates[np.abs(coordinates) <= EMBEDDING_GROUND_FACE_TOLERANCE] = 0.0
    return coordinates


def get_henneberg_routed_graph_definition(
    node_count: int,
    tube_count: int,
    scale: float = 1.0,
    *,
    preset_index: int = 1,
) -> tuple[NodeDict, ShapeDict]:
    """Return a curated routed Henneberg graph preset.

    H1/H2 candidate graphs are generated deterministically from ``K4``. The
    selected graph must support the requested number of edge-disjoint routed
    trails, and its selected 3D embedding must pass the same rigidity-matrix
    style infinitesimal-rigidity gate used for Usevitch embeddings.
    """
    scale = _validate_scale(scale)
    node_count = int(node_count)
    tube_count = int(tube_count)
    preset_index = int(preset_index)
    if (node_count, tube_count) not in HENNEBERG_PRESET_SPECS:
        supported = ", ".join(
            f"n{nodes}_{tubes}tube" for nodes, tubes in HENNEBERG_PRESET_SPECS
        )
        raise ValueError(
            f"Unsupported Henneberg routed graph preset: "
            f"node_count={node_count}, tube_count={tube_count}. "
            f"Supported presets: {supported}"
        )
    if preset_index < 1:
        raise ValueError("preset_index must be greater than zero.")

    coordinates, routes = _selected_henneberg_routed_graph(
        node_count,
        tube_count,
        preset_index,
    )
    node_dict = {
        f"node_{index + 1}": [float(coordinate) for coordinate in position]
        for index, position in enumerate(coordinates)
    }
    shape_dict = {
        f"path_{index}": _henneberg_route_shape(route)
        for index, route in enumerate(routes, start=1)
    }
    return _scale_node_dict(node_dict, scale), shape_dict


def _henneberg_route_shape(route: tuple[int, ...]) -> dict[str, list[list[str]] | list[str]]:
    node_route = [f"node_{node + 1}" for node in route]
    return {
        "route": node_route,
        "active_edges": [
            [from_node, to_node]
            for from_node, to_node in zip(node_route, node_route[1:], strict=False)
        ],
    }


@cache
def _selected_henneberg_routed_graph(
    node_count: int,
    tube_count: int,
    preset_index: int,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    variant_count = HENNEBERG_PRESET_VARIANT_COUNTS[(node_count, tube_count)]
    if preset_index > variant_count:
        raise ValueError(
            f"Henneberg n{node_count} {tube_count}-tube has {variant_count} "
            f"preset variant(s); got preset_index={preset_index}."
        )

    candidate_index = 0
    for graph_index, graph in enumerate(_henneberg_graphs_by_node_count(node_count)[node_count]):
        if _minimum_trail_count(graph) != tube_count:
            continue
        if graph.number_of_edges() % tube_count != 0:
            continue

        candidate_index += 1
        if candidate_index != preset_index:
            continue

        routes = _decompose_henneberg_routes(graph, tube_count)
        if routes is None:
            raise ValueError(
                f"Henneberg n{node_count} {tube_count}-tube preset variant "
                f"{preset_index} does not have a route decomposition."
            )

        try:
            embedding = _best_henneberg_embedding(graph, graph_index, tube_count)
        except ValueError as exc:
            raise ValueError(
                f"Could not embed Henneberg n{node_count} {tube_count}-tube "
                f"preset variant {preset_index}."
            ) from exc

        edges = _sorted_graph_edges(graph)
        if _rigidity_matrix_rank(embedding, edges) != 3 * node_count - 6:
            raise ValueError(
                f"Henneberg n{node_count} {tube_count}-tube preset variant "
                f"{preset_index} did not pass the infinitesimal-rigidity rank gate."
            )
        return embedding, routes

    raise ValueError(
        f"Could not find Henneberg n{node_count} {tube_count}-tube "
        f"preset variant {preset_index}."
    )


@cache
def _henneberg_graphs_by_node_count(max_node_count: int) -> dict[int, list[nx.Graph]]:
    max_node_count = int(max_node_count)
    if max_node_count < 4:
        raise ValueError("max_node_count must be at least 4.")

    graphs_by_node_count: dict[int, list[nx.Graph]] = {4: [nx.complete_graph(4)]}
    for node_count in range(5, max_node_count + 1):
        candidates = []
        for graph in graphs_by_node_count[node_count - 1]:
            candidates.extend(_henneberg_h1_graphs(graph))
            candidates.extend(_henneberg_h2_graphs(graph))

        candidates = [
            graph
            for graph in candidates
            if graph.number_of_edges() == 3 * graph.number_of_nodes() - 6
        ]
        graphs_by_node_count[node_count] = _unique_henneberg_graphs(candidates)

    return graphs_by_node_count


def _henneberg_h1_graphs(graph: nx.Graph) -> list[nx.Graph]:
    new_node = graph.number_of_nodes()
    graphs = []
    for neighbors in combinations(graph.nodes(), 3):
        candidate = graph.copy()
        candidate.add_node(new_node)
        candidate.add_edges_from((new_node, neighbor) for neighbor in neighbors)
        graphs.append(candidate)
    return graphs


def _henneberg_h2_graphs(graph: nx.Graph) -> list[nx.Graph]:
    new_node = graph.number_of_nodes()
    graphs = []
    for first, second in list(graph.edges()):
        remaining_nodes = [node for node in graph.nodes() if node not in (first, second)]
        for extra_neighbors in combinations(remaining_nodes, 2):
            candidate = graph.copy()
            candidate.remove_edge(first, second)
            candidate.add_node(new_node)
            candidate.add_edges_from(
                (
                    (new_node, first),
                    (new_node, second),
                    (new_node, extra_neighbors[0]),
                    (new_node, extra_neighbors[1]),
                )
            )
            graphs.append(candidate)
    return graphs


def _unique_henneberg_graphs(graphs: list[nx.Graph]) -> list[nx.Graph]:
    buckets: dict[tuple[int, int, tuple[int, ...], int], list[nx.Graph]] = {}
    unique_graphs = []
    for graph in graphs:
        bucket_key = (
            graph.number_of_nodes(),
            graph.number_of_edges(),
            tuple(sorted(dict(graph.degree()).values())),
            sum(nx.triangles(graph).values()) // 3,
        )
        bucket = buckets.setdefault(bucket_key, [])
        if any(nx.is_isomorphic(graph, unique_graph) for unique_graph in bucket):
            continue
        bucket.append(graph)
        unique_graphs.append(graph)
    return unique_graphs


def _odd_degree_nodes(graph: nx.Graph) -> list[int]:
    return [int(node) for node, degree in graph.degree() if degree % 2 == 1]


def _minimum_trail_count(graph: nx.Graph) -> int:
    if graph.number_of_edges() == 0:
        return 0
    return max(1, len(_odd_degree_nodes(graph)) // 2)


def _decompose_henneberg_routes(
    graph: nx.Graph,
    tube_count: int,
) -> tuple[tuple[int, ...], ...] | None:
    edge_count = graph.number_of_edges()
    if edge_count % tube_count != 0:
        return None

    if tube_count == 1:
        return _single_henneberg_route(graph)

    paired_routes = _paired_euler_henneberg_routes(graph, tube_count)
    if paired_routes is not None:
        return paired_routes

    target_route_edges = edge_count // tube_count
    all_edges = frozenset(_sorted_graph_edges(graph))
    nodes = tuple(sorted(int(node) for node in graph.nodes()))

    def search(
        remaining_edges: frozenset[tuple[int, int]],
        routes: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...] | None:
        remaining_routes = tube_count - len(routes)
        if remaining_routes == 0:
            return routes if not remaining_edges else None
        if len(remaining_edges) != remaining_routes * target_route_edges:
            return None

        for route in _trails_with_edge_count(nodes, remaining_edges, target_route_edges):
            route_edges = frozenset(
                _graph_edge_key(from_node, to_node)
                for from_node, to_node in zip(route, route[1:], strict=False)
            )
            if len(route_edges) != target_route_edges or not route_edges <= remaining_edges:
                continue
            result = search(remaining_edges - route_edges, (*routes, route))
            if result is not None:
                return result
        return None

    return search(all_edges, ())


def _single_henneberg_route(graph: nx.Graph) -> tuple[tuple[int, ...], ...] | None:
    if not nx.has_eulerian_path(graph):
        return None

    odd_nodes = sorted(_odd_degree_nodes(graph))
    source = odd_nodes[0] if odd_nodes else min(graph.nodes())
    route_edges = list(nx.eulerian_path(graph, source=source))
    if len(route_edges) != graph.number_of_edges():
        return None

    route = [int(route_edges[0][0])]
    route.extend(int(to_node) for _, to_node in route_edges)
    return (tuple(route),)


def _paired_euler_henneberg_routes(
    graph: nx.Graph,
    tube_count: int,
) -> tuple[tuple[int, ...], ...] | None:
    odd_nodes = tuple(sorted(_odd_degree_nodes(graph)))
    if len(odd_nodes) != 2 * tube_count:
        return None

    target_route_edges = graph.number_of_edges() // tube_count
    for odd_pairing in _node_pairings(odd_nodes):
        multigraph = nx.MultiGraph(graph)
        for dummy_index, (first, second) in enumerate(odd_pairing):
            multigraph.add_edge(first, second, key=f"dummy_{dummy_index}", dummy=True)

        circuit = list(nx.eulerian_circuit(multigraph, source=odd_pairing[0][0], keys=True))
        routes = _split_euler_circuit_at_dummy_edges(multigraph, circuit)
        if routes is None:
            continue
        if len(routes) != tube_count:
            continue
        if {len(route) - 1 for route in routes} != {target_route_edges}:
            continue
        return routes
    return None


def _node_pairings(nodes: tuple[int, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not nodes:
        return ((),)

    first = nodes[0]
    pairings = []
    for index in range(1, len(nodes)):
        second = nodes[index]
        remaining = nodes[1:index] + nodes[index + 1 :]
        for rest in _node_pairings(remaining):
            pairings.append(((first, second), *rest))
    return tuple(pairings)


def _split_euler_circuit_at_dummy_edges(
    multigraph: nx.MultiGraph,
    circuit: list[tuple[int, int, str | int]],
) -> tuple[tuple[int, ...], ...] | None:
    if not circuit:
        return None

    dummy_indices = [
        index
        for index, (first, second, key) in enumerate(circuit)
        if multigraph.edges[first, second, key].get("dummy", False)
    ]
    if not dummy_indices:
        return None

    start_index = (dummy_indices[0] + 1) % len(circuit)
    rotated_circuit = circuit[start_index:] + circuit[:start_index]

    routes = []
    current_route = [int(rotated_circuit[0][0])]
    for first, second, key in rotated_circuit:
        if multigraph.edges[first, second, key].get("dummy", False):
            if len(current_route) > 1:
                routes.append(tuple(current_route))
            current_route = [int(second)]
            continue
        if current_route[-1] != int(first):
            return None
        current_route.append(int(second))

    if len(current_route) > 1:
        routes.append(tuple(current_route))
    return tuple(routes)


def _trails_with_edge_count(
    nodes: tuple[int, ...],
    edges: frozenset[tuple[int, int]],
    edge_count: int,
):
    adjacency: dict[int, list[tuple[int, tuple[int, int]]]] = {node: [] for node in nodes}
    for first, second in edges:
        adjacency[first].append((second, (first, second)))
        adjacency[second].append((first, (first, second)))
    for neighbors in adjacency.values():
        neighbors.sort()

    seen: set[tuple[int, ...]] = set()

    def canonical_route(route: tuple[int, ...]) -> tuple[int, ...]:
        reversed_route = tuple(reversed(route))
        return min(route, reversed_route)

    def search(route: tuple[int, ...], remaining_edges: frozenset[tuple[int, int]]):
        if len(route) == edge_count + 1:
            canonical = canonical_route(route)
            if canonical not in seen:
                seen.add(canonical)
                yield route
            return

        current = route[-1]
        for next_node, edge in adjacency[current]:
            if edge not in remaining_edges:
                continue
            yield from search((*route, next_node), remaining_edges - {edge})

    for node in nodes:
        yield from search((node,), edges)


def _best_henneberg_embedding(
    graph: nx.Graph,
    graph_index: int,
    tube_count: int,
) -> np.ndarray:
    node_count = graph.number_of_nodes()
    edges = _sorted_graph_edges(graph)
    best_coordinates: np.ndarray | None = None
    best_wcri = -np.inf
    best_edge_rms_error = np.inf

    for seed_index in range(HENNEBERG_EMBEDDING_TRIALS):
        seed = node_count * 10_000 + tube_count * 1_000 + graph_index * 100 + seed_index
        layout = nx.spring_layout(
            graph,
            dim=3,
            seed=seed,
            iterations=HENNEBERG_LAYOUT_ITERATIONS,
            scale=None,
        )
        coordinates = np.array(
            [layout[node] for node in sorted(graph.nodes())],
            dtype=float,
        )
        candidates = (
            _normalize_henneberg_embedding(coordinates, edges),
            _refine_henneberg_embedding(graph, coordinates, edges),
        )
        for candidate in candidates:
            wcri = _worst_case_rigidity_index(candidate, edges)
            edge_rms_error = _edge_length_rms_error(candidate, edges, target_length=1.0)
            if best_coordinates is None or wcri > best_wcri or (
                wcri == best_wcri and edge_rms_error < best_edge_rms_error
            ):
                best_coordinates = candidate
                best_wcri = wcri
                best_edge_rms_error = edge_rms_error

    if best_coordinates is None:
        raise ValueError("Could not embed Henneberg graph.")
    return best_coordinates


def _refine_henneberg_embedding(
    graph: nx.Graph,
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    node_names = tuple(sorted(graph.nodes()))
    node_index = {node: index for index, node in enumerate(node_names)}
    pairs = []
    for first_index, first in enumerate(node_names):
        for second in node_names[first_index + 1 :]:
            is_edge = graph.has_edge(first, second)
            pairs.append(
                (
                    node_index[first],
                    node_index[second],
                    1.0 if is_edge else HENNEBERG_NONEDGE_DISTANCE,
                    4.0 if is_edge else 0.5,
                )
            )

    initial = _normalize_henneberg_embedding(coordinates, edges)

    def residuals(flat_coordinates: np.ndarray) -> np.ndarray:
        candidate = flat_coordinates.reshape((len(node_names), 3))
        candidate = candidate - np.mean(candidate, axis=0)
        return np.array(
            [
                math.sqrt(weight)
                * (np.linalg.norm(candidate[second] - candidate[first]) - target_distance)
                for first, second, target_distance, weight in pairs
            ],
            dtype=float,
        )

    result = least_squares(
        residuals,
        initial.reshape(-1),
        max_nfev=HENNEBERG_LAYOUT_REFINEMENT_EVALUATIONS,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
    )
    return _normalize_henneberg_embedding(result.x.reshape((len(node_names), 3)), edges)


def _normalize_henneberg_embedding(
    coordinates: np.ndarray,
    edges: tuple[tuple[int, int], ...],
) -> np.ndarray:
    coordinates = coordinates - np.mean(coordinates, axis=0)
    coordinates = _normalize_usevitch_candidate_edge_lengths(coordinates, edges)
    coordinates = coordinates - np.mean(coordinates, axis=0)
    return _align_node_one_ground_face(
        coordinates,
        _triangular_faces_through_node_one(coordinates.shape[0], edges),
    )


def _sorted_graph_edges(graph: nx.Graph) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_graph_edge_key(first, second) for first, second in graph.edges()))


def _graph_edge_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


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


def _make_henneberg_preset(
    node_count: int,
    tube_count: int,
    preset_index: int = 1,
) -> Callable[[float], tuple[NodeDict, ShapeDict]]:
    return lambda scale=1.0: get_henneberg_routed_graph_definition(
        node_count,
        tube_count,
        preset_index=preset_index,
        scale=scale,
    )


def _henneberg_presets() -> dict[str, Callable[[float], tuple[NodeDict, ShapeDict]]]:
    presets: dict[str, Callable[[float], tuple[NodeDict, ShapeDict]]] = {}
    for node_count, tube_count in HENNEBERG_PRESET_SPECS:
        presets[f"henneberg_n{node_count}_{tube_count}tube"] = _make_henneberg_preset(
            node_count,
            tube_count,
        )
        for preset_index in range(1, HENNEBERG_PRESET_VARIANT_COUNTS[(node_count, tube_count)] + 1):
            presets[f"henneberg_n{node_count}_{tube_count}tube_{preset_index}"] = (
                _make_henneberg_preset(node_count, tube_count, preset_index)
            )
    return presets


PRESETS: dict[str, Callable[[float], tuple[NodeDict, TriangleDict | ShapeDict]]] = {
    "octahedron": get_octahedron_definition,
    "icosahedron": get_icosahedron_definition,
    "solar_array": get_solar_array_definition,
    "tetrahedron": get_tetrahedron_definition,
    **_usevitch_presets(),
    **_henneberg_presets(),
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
