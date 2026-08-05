from __future__ import annotations

import pytest
from experiments.benchmark_mjx_env import (
    REPRESENTATIVE_MODELS,
    _aggregate_workloads,
    _capacity_comparisons,
    _required_model_finiteness,
)


def _result(
    model: str,
    implementation: str,
    graph_mode: str,
    *,
    vector_step_seconds: float,
    capacity_scale: int = 1,
) -> dict:
    return {
        "status": "ok",
        "case": {
            "model": model,
            "batch_size": 512,
            "implementation": implementation,
            "graph_mode": graph_mode,
            "capacity_scale": capacity_scale,
        },
        "timing": {"vector_step_seconds": vector_step_seconds},
        "capacity_high_water": {"overflow": False},
        "validation": {
            "observation_checksum": 10.0,
            "reward_checksum": 2.0,
            "done_count": 0,
            "finite": True,
        },
    }


def test_representative_workload_uses_sum_of_topology_bucket_times() -> None:
    results = []
    for model in REPRESENTATIVE_MODELS:
        results.append(_result(model, "jax", "warp", vector_step_seconds=3.0))
        results.append(_result(model, "warp", "warp", vector_step_seconds=2.0))

    workloads = _aggregate_workloads(results)

    assert len(workloads) == 1
    assert workloads[0]["graph_mode"] == "warp"
    assert workloads[0]["jax_transitions_per_second"] == pytest.approx(1536 / 9)
    assert workloads[0]["warp_transitions_per_second"] == pytest.approx(1536 / 6)
    assert workloads[0]["physics_speedup"] == pytest.approx(1.5)
    assert workloads[0]["estimated_total_training_speedup"] == pytest.approx(
        1 / ((1 - 0.4067) + 0.4067 / 1.5)
    )
    assert workloads[0]["performance_gate_passed"] is True


def test_capacity_gate_requires_matching_finite_base_and_doubled_runs() -> None:
    base = _result(
        "tetrahedron:abstract",
        "warp",
        "warp",
        vector_step_seconds=1.0,
        capacity_scale=1,
    )
    doubled = _result(
        "tetrahedron:abstract",
        "warp",
        "warp",
        vector_step_seconds=1.0,
        capacity_scale=2,
    )

    comparisons = _capacity_comparisons([base, doubled])

    assert comparisons == [
        {
            "model": "tetrahedron:abstract",
            "batch_size": 512,
            "graph_mode": "warp",
            "passed": True,
            "base_validation": base["validation"],
            "doubled_validation": doubled["validation"],
        }
    ]

    doubled["validation"]["reward_checksum"] = 3.0
    assert _capacity_comparisons([base, doubled])[0]["passed"] is False


def test_required_model_finiteness_fails_nonfinite_or_missing_cases() -> None:
    jax_result = _result("octahedron:realistic", "jax", "warp", vector_step_seconds=1.0)
    warp_result = _result("octahedron:realistic", "warp", "warp_staged", vector_step_seconds=1.0)
    warp_result["validation"]["finite"] = False
    warp_result["validation"]["done_count"] = 512

    analysis = _required_model_finiteness(
        [jax_result, warp_result],
        required_models=("octahedron:realistic",),
        batch_sizes=(512,),
        implementations=("jax", "warp"),
        graph_modes=("warp_staged", "warp_staged_ex"),
    )

    assert analysis["passed"] is False
    assert [
        (item["graph_mode"], item["present"], item["finite"]) for item in analysis["failures"]
    ] == [
        ("warp_staged", True, False),
        ("warp_staged_ex", False, False),
    ]
