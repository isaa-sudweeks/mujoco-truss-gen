from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mujoco_truss_gen import TerrainConfig, generate_terrain

ROOT = Path(__file__).resolve().parents[1]
TERRAIN_KINDS = ("flat", "slope", "stairs", "waves", "rough")


def _web_config(config: TerrainConfig) -> dict[str, object]:
    return {
        "kind": config.kind,
        "halfSizeX": config.half_size[0],
        "halfSizeY": config.half_size[1],
        "resolution": config.resolution[0],
        "amplitude": config.amplitude,
        "featureSize": config.feature_size,
        "maxSlopeDegrees": config.max_slope_degrees,
        "slopeAngleDegrees": config.slope_angle_degrees,
        "slopeDirectionDegrees": config.slope_direction_degrees,
        "stairHeight": config.stair_height,
        "stairRun": config.stair_run,
        "spawnFlatRadius": config.spawn_flat_radius,
        "spawnBlendWidth": config.spawn_blend_width,
        "seed": config.seed,
    }


def test_browser_generator_matches_python_for_every_family() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed; browser parity test cannot run.")

    configs = [
        TerrainConfig(
            kind=kind,
            half_size=(4.0, 3.0),
            resolution=(25, 25),
            amplitude=0.17,
            feature_size=0.9,
            max_slope_degrees=27.0,
            slope_angle_degrees=9.0,
            slope_direction_degrees=35.0,
            stair_height=0.08,
            stair_run=0.7,
            spawn_flat_radius=1.1,
            spawn_blend_width=0.45,
            seed=19,
        )
        for kind in TERRAIN_KINDS
    ]
    completed = subprocess.run(
        [node, str(ROOT / "tests" / "terrain_web_runner.mjs")],
        input=json.dumps([_web_config(config) for config in configs]),
        text=True,
        capture_output=True,
        check=True,
    )
    web_results = json.loads(completed.stdout)

    for config, web in zip(configs, web_results, strict=True):
        python = generate_terrain(config)
        np.testing.assert_allclose(np.asarray(web["heights"]), python.heights.ravel(), atol=2e-14)
        assert web["elevationMin"] == pytest.approx(python.elevation_min, abs=2e-14)
        assert web["elevationMax"] == pytest.approx(
            python.elevation_min + python.elevation_range,
            abs=2e-14,
        )
        assert web["maxGeneratedSlope"] == pytest.approx(
            python.max_generated_slope,
            abs=2e-14,
        )


def test_static_catalog_links_complete_terrain_explorer() -> None:
    catalog = (ROOT / "preset_catalog" / "index.html").read_text(encoding="utf-8")
    page = (ROOT / "preset_catalog" / "terrain.html").read_text(encoding="utf-8")

    assert 'href="terrain.html"' in catalog
    assert 'id="height-map"' in page
    assert 'id="profile"' in page
    assert 'id="copy-config"' in page
    assert 'src="terrain-app.mjs"' in page
