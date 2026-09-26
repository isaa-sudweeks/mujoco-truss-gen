from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "preset_catalog"
sys.path.insert(0, str(ROOT / "tools"))

import build_training_distribution as builder  # noqa: E402

SMALL_SET = [
    "tetrahedron",
    "henneberg_n5_1tube_1",
    "octahedron",
    "henneberg_n6_2tube_1",
    "henneberg_n7_1tube_3",
    "usevitch_1514879",
    "solar_array",
    "usevitch_60243677150_p1",
    "usevitch_60243677150_p2",
    "icosahedron",
]


@pytest.fixture(scope="module")
def small_data() -> dict[str, Any]:
    return builder.build(SMALL_SET)


def _by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {preset["name"]: preset for preset in data["presets"]}


def _decoded(basis: dict[str, Any], count: int) -> np.ndarray:
    return np.frombuffer(base64.b64decode(basis["distances"]), dtype="<u2")


def test_descriptors_for_known_presets(small_data: dict[str, Any]) -> None:
    presets = _by_name(small_data)
    assert small_data["failures"] == []
    for preset in presets.values():
        assert preset["edges"] == 3 * preset["nodes"] - 6
        assert preset["shape_dof"] == preset["actuators"] - preset["constraints"]

    tetrahedron = presets["tetrahedron"]
    assert tetrahedron["tube_count"] == 2
    assert tetrahedron["actuators"] == 6
    assert tetrahedron["resting_faces"] == 4
    assert tetrahedron["symmetry"] == pytest.approx(math.log2(24), abs=1e-4)

    octahedron = presets["octahedron"]
    assert octahedron["member_type"] == "triangle"
    assert (octahedron["members"], octahedron["actuators"], octahedron["constraints"]) == (4, 8, 4)
    assert octahedron["resting_faces"] == 8
    assert octahedron["symmetry"] == pytest.approx(math.log2(48), abs=1e-4)

    assert presets["icosahedron"]["actuators"] == 40
    assert presets["icosahedron"]["resting_faces"] == 20
    assert (
        presets["solar_array"]["graph_group"]
        == presets["usevitch_60243677150_p1"]["graph_group"]
        == presets["usevitch_60243677150_p2"]["graph_group"]
    )
    assert presets["tetrahedron"]["graph_group"] != presets["octahedron"]["graph_group"]


def test_distance_bases_decode(small_data: dict[str, Any]) -> None:
    count = len(small_data["presets"])
    for basis in small_data["bases"].values():
        assert _decoded(basis, count).size == count * (count - 1) // 2
        assert len(basis["layout"]) == count
        assert 0.0 < basis["explained"] <= 1.0


def test_browser_coverage_logic(small_data: dict[str, Any]) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser logic test cannot run.")

    payload = {
        "distances": small_data["bases"]["graph"]["distances"],
        "presets": [{"name": p["name"], "nodes": p["nodes"]} for p in small_data["presets"]],
        "finalTest": ["henneberg_n7_1tube_3", "usevitch_1514879"],
    }
    completed = subprocess.run(
        [node, str(ROOT / "tests" / "distribution_web_runner.mjs")],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    node_counts = [preset["nodes"] for preset in payload["presets"]]
    pool = sorted(nodes for nodes in node_counts if nodes != 7)

    assert result["diagonal"] == [0] * len(node_counts)
    assert result["symmetric"]
    assert result["deterministic"]
    fps_worst = result["fpsWorst"]
    assert all(later <= earlier for earlier, later in zip(fps_worst, fps_worst[1:], strict=False))
    assert result["fpsWorst"][-1] == pytest.approx(result["randomWorstMedian"][-1])
    for order in result["orders"]:
        assert sorted(order) == pool
    assert sorted(result["sfpsFirst"]) == sorted(set(pool))

    per_size: defaultdict[int, float] = defaultdict(float)
    for nodes, weight in zip(node_counts, result["perSizeWeights"], strict=True):
        per_size[nodes] += weight
    assert all(total == pytest.approx(1.0) for total in per_size.values())

    yaml = result["yaml"]
    for fragment in ("cross_validation:", "  groups:", "  final_test:", "    - usevitch_1514879"):
        assert fragment in yaml
    assert "node_7" not in yaml


def test_committed_data_matches_catalog() -> None:
    data = json.loads((CATALOG / "distribution-data.json").read_text(encoding="utf-8"))
    manifest = json.loads((CATALOG / "manifest.json").read_text(encoding="utf-8"))
    rendered = {preset["name"] for preset in manifest["presets"] if preset["status"] == "ok"}
    names = [preset["name"] for preset in data["presets"]]
    assert set(names) == rendered

    count = len(names)
    for basis in data["bases"].values():
        assert _decoded(basis, count).size == count * (count - 1) // 2
        assert len(basis["layout"]) == count
    split_names = [name for group in data["split"]["groups"].values() for name in group]
    assert set(split_names + data["split"]["final_test"]) <= set(names)
    for name in data["split"]["final_test"]:
        assert _by_name(data)[name]["nodes"] == 7


@pytest.mark.parametrize("page", ["index.html", "terrain.html", "distribution.html"])
def test_site_pages_share_navigation(page: str) -> None:
    document = (CATALOG / page).read_text(encoding="utf-8")
    for href in (
        'href="distribution.html"',
        'href="terrain.html"',
        'href="https://github.com/isaa-sudweeks/mujoco-truss-gen"',
    ):
        assert href in document
