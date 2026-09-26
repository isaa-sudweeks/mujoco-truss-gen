"""Build the data behind the training-distribution page of the preset catalog.

Computes per-preset graph, actuation, and geometry descriptors, two pairwise
distance matrices, their 2D classical-MDS layouts, and the Henneberg graphs that
have no preset. Run from the repository root with::

    PYTHONPATH=src .venv/bin/python tools/build_training_distribution.py
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from render_preset_catalog import (
    _initial_wcri,
    _is_henneberg_alias,
    _metadata,
    _node_sort_key,
    _structural_edges,
)
from scipy.spatial import ConvexHull

from mujoco_truss_gen import PRESETS, get_mujoco_spec, get_preset_definition
from mujoco_truss_gen.mujoco_model import presets as preset_module

# Copied from GNN-SAC config/cross_validation/node_count_loso.yaml.
CURRENT_SPLIT: dict[str, Any] = {
    "name": "node_count_loso",
    "groups": {
        "node_4": ["tetrahedron"],
        "node_5": ["henneberg_n5_1tube_1"],
        "node_6": ["henneberg_n6_1tube_2", "henneberg_n6_2tube_1", "octahedron"],
        "node_8": [
            "henneberg_n8_1tube_57",
            "henneberg_n8_2tube_187",
            "henneberg_n8_3tube_64",
            "usevitch_210272254_p1",
        ],
        "node_9": ["usevitch_60243677150_p1"],
    },
    "final_test": ["henneberg_n7_1tube_3", "henneberg_n7_3tube_1", "usevitch_1514879"],
}

# Descriptors used for the "descriptors" distance basis: one per independent idea.
DISTANCE_DESCRIPTORS = (
    "nodes",
    "members",
    "degree_std",
    "triangles_per_node",
    "symmetry",
    "wcri",
    "resting_faces",
    "aspect_ratio",
)
WL_ITERATIONS = 3
HULL_TOLERANCE = 1e-6


def _graph(nodes: dict[str, list[float]], structure: dict[str, Any]) -> tuple[nx.Graph, np.ndarray]:
    node_names = sorted(nodes, key=_node_sort_key)
    index = {name: position for position, name in enumerate(node_names)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(node_names)))
    graph.add_edges_from((index[a], index[b]) for a, b in _structural_edges(structure))
    coordinates = np.asarray([nodes[name] for name in node_names], dtype=float)
    return graph, coordinates


def _symmetry(graph: nx.Graph) -> float:
    matcher = nx.algorithms.isomorphism.GraphMatcher(graph, graph)
    return math.log2(sum(1 for _ in matcher.isomorphisms_iter()))


def _algebraic_connectivity(graph: nx.Graph) -> float:
    laplacian = nx.normalized_laplacian_matrix(graph, nodelist=sorted(graph)).toarray()
    return float(np.sort(np.linalg.eigvalsh(laplacian))[1])


def _resting_faces(coordinates: np.ndarray) -> int:
    """Count merged convex-hull faces the equal-mass center of mass projects inside."""
    hull = ConvexHull(coordinates)
    center = coordinates.mean(axis=0)
    faces: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        for plane, triangles in faces:
            if np.allclose(plane, equation, atol=HULL_TOLERANCE):
                triangles.append(coordinates[simplex])
                break
        else:
            faces.append((equation, [coordinates[simplex]]))

    stable = 0
    for plane, triangles in faces:
        normal = plane[:3]
        projected = center - (normal @ center + plane[3]) * normal
        if any(_inside_triangle(projected, triangle) for triangle in triangles):
            stable += 1
    return stable


def _inside_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, point - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    denominator = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return v >= -HULL_TOLERANCE and w >= -HULL_TOLERANCE and v + w <= 1 + HULL_TOLERANCE


def _aspect_ratio(coordinates: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(np.cov((coordinates - coordinates.mean(axis=0)).T))
    return float(math.sqrt(eigenvalues[-1] / max(eigenvalues[0], 1e-12)))


def _wl_histogram(graph: nx.Graph) -> Counter[str]:
    labelled = graph.copy()
    nx.set_node_attributes(
        labelled, {node: str(degree) for node, degree in graph.degree()}, "degree"
    )
    hashes = nx.weisfeiler_lehman_subgraph_hashes(
        labelled,
        node_attr="degree",
        iterations=WL_ITERATIONS,
        include_initial_labels=True,
    )
    return Counter(
        f"{iteration}:{label}"
        for labels in hashes.values()
        for iteration, label in enumerate(labels)
    )


def describe_preset(name: str) -> tuple[dict[str, Any], nx.Graph, Counter[str]]:
    """Return one preset's descriptor record, its graph, and its WL label histogram."""
    nodes, structure = get_preset_definition(name)
    graph, coordinates = _graph(nodes, structure)
    model = get_mujoco_spec(nodes, structure, realistic=False).compile()
    first_member = next(iter(structure.values()))
    member_type = "tube" if isinstance(first_member, dict) else "triangle"
    degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
    wcri, node_count, edge_count = _initial_wcri(nodes, structure)
    record: dict[str, Any] = {
        "name": name,
        "family": _metadata(name)["family"],
        "member_type": member_type,
        "members": len(structure),
        "nodes": node_count,
        "edges": edge_count,
        "actuators": int(model.nu),
        "constraints": int(model.neq),
        "shape_dof": int(model.nu - model.neq),
        "degree_max": int(degrees.max()),
        "degree_std": float(degrees.std()),
        "triangles_per_node": sum(nx.triangles(graph).values()) / 3 / node_count,
        "algebraic_connectivity": _algebraic_connectivity(graph),
        "diameter": nx.diameter(graph),
        "symmetry": _symmetry(graph),
        "wcri": wcri,
        "resting_faces": _resting_faces(coordinates),
        "aspect_ratio": _aspect_ratio(coordinates),
        "image": f"images/{name}.png",
    }
    if member_type == "tube":
        record["tube_count"] = len(structure)
    return record, graph, _wl_histogram(graph)


def _graph_groups(graphs: list[nx.Graph]) -> list[int]:
    """Assign each graph the index of the first isomorphic graph in the list."""
    buckets: dict[tuple[Any, ...], list[int]] = {}
    groups = []
    for position, graph in enumerate(graphs):
        key = (
            graph.number_of_nodes(),
            tuple(sorted(degree for _, degree in graph.degree())),
            sum(nx.triangles(graph).values()),
        )
        bucket = buckets.setdefault(key, [])
        match = next((other for other in bucket if nx.is_isomorphic(graph, graphs[other])), None)
        if match is None:
            bucket.append(position)
            match = position
        groups.append(match)
    return groups


def _wl_distances(histograms: list[Counter[str]]) -> np.ndarray:
    vocabulary = {label: index for index, label in enumerate(sorted(set().union(*histograms)))}
    features = np.zeros((len(histograms), len(vocabulary)))
    for row, histogram in enumerate(histograms):
        for label, count in histogram.items():
            features[row, vocabulary[label]] = count
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return np.clip(1.0 - features @ features.T, 0.0, None)


def _descriptor_distances(records: list[dict[str, Any]]) -> np.ndarray:
    features = np.asarray([[record[key] for key in DISTANCE_DESCRIPTORS] for record in records])
    spread = features.std(axis=0)
    features = (features - features.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    differences = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.square(differences).sum(axis=-1))


def _basis(distances: np.ndarray) -> dict[str, Any]:
    np.fill_diagonal(distances, 0.0)
    distances = (distances + distances.T) / 2
    scale = float(distances.max()) or 1.0
    normalized = distances / scale

    layout = preset_module._classical_mds(normalized, 2)
    count = len(normalized)
    centering = np.eye(count) - np.full((count, count), 1.0 / count)
    eigenvalues = np.linalg.eigvalsh(-0.5 * centering @ np.square(normalized) @ centering)
    positive = np.sort(eigenvalues[eigenvalues > 0])[::-1]
    explained = float(positive[:2].sum() / positive.sum()) if positive.size else 0.0

    upper = normalized[np.triu_indices(count, k=1)]
    quantized = np.round(upper * 65535).astype("<u2")
    return {
        "distances": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "layout": np.round(layout, 5).tolist(),
        "explained": round(explained, 4),
    }


def _gaps(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count enumerated Henneberg graphs per (nodes, required tubes) that have no preset."""
    in_library = Counter(
        (record["nodes"], record["tube_count"])
        for record in records
        if record["member_type"] == "tube"
    )
    enumerated = preset_module._henneberg_graphs_by_node_count(8)
    gaps = []
    for node_count, graphs in sorted(enumerated.items()):
        required = Counter(preset_module._minimum_trail_count(graph) for graph in graphs)
        for tubes, total in sorted(required.items()):
            missing = total - in_library[(node_count, tubes)]
            if missing <= 0:
                continue
            reason = "unequal tubes" if (3 * node_count - 6) % tubes else "failed to embed"
            gaps.append(
                {
                    "nodes": node_count,
                    "tubes": tubes,
                    "enumerated": total,
                    "missing": missing,
                    "reason": reason,
                }
            )
    return gaps


def _rounded(record: dict[str, Any]) -> dict[str, Any]:
    """Keep 5 significant digits so tiny values such as near-zero WCRI survive."""
    return {
        key: float(f"{value:.5g}") if isinstance(value, float) else value
        for key, value in record.items()
    }


def build(names: list[str]) -> dict[str, Any]:
    records, graphs, histograms, failures = [], [], [], []
    for position, name in enumerate(names, start=1):
        print(f"[{position}/{len(names)}] {name}", flush=True)
        try:
            record, graph, histogram = describe_preset(name)
        except Exception as exc:  # Keep going so one bad preset does not lose the page.
            failures.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        records.append(record)
        graphs.append(graph)
        histograms.append(histogram)

    for record, group in zip(records, _graph_groups(graphs), strict=True):
        record["graph_group"] = group

    wl = _wl_distances(histograms)
    groups = [record["graph_group"] for record in records]
    collided = {
        index
        for index, row in enumerate(wl)
        for other in np.flatnonzero(row < 1e-12)
        if groups[index] != groups[other]
    }
    return {
        "generated_by": "tools/build_training_distribution.py",
        "presets": [_rounded(record) for record in records],
        "failures": failures,
        "bases": {
            "graph": _basis(wl),
            "descriptors": _basis(_descriptor_distances(records)),
        },
        "distance_descriptors": list(DISTANCE_DESCRIPTORS),
        "wl_collisions": len(collided),
        "split": CURRENT_SPLIT,
        "gaps": _gaps(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("preset_catalog/distribution-data.json")
    )
    parser.add_argument("--presets", help="Comma-separated preset names (for smoke tests).")
    args = parser.parse_args()

    if args.presets:
        names = [name.strip() for name in args.presets.split(",") if name.strip()]
    else:
        names = [name for name in sorted(PRESETS) if not _is_henneberg_alias(name)]
    data = build(names)
    args.output.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(data['presets'])} presets, {len(data['failures'])} failed)")


if __name__ == "__main__":
    main()
