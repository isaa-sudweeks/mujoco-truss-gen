from __future__ import annotations

import numpy as np
from experiments.diagnose_mjx_warp_realistic import (
    _classify_warp_run,
    _first_divergence,
    _first_nonfinite,
)


def _trace() -> dict[str, np.ndarray]:
    return {
        "step_index": np.arange(3),
        "qpos": np.zeros((3, 2, 2)),
        "qvel": np.zeros((3, 2, 2)),
        "observation": np.zeros((3, 2, 2)),
        "reward": np.zeros((3, 2)),
        "ctrl": np.zeros((3, 2, 2)),
        "contact_count": np.zeros((3, 2), dtype=int),
        "constraint_count": np.zeros((3, 2), dtype=int),
        "done": np.zeros((3, 2), dtype=bool),
    }


def test_diagnostic_finds_first_nonfinite_field_and_environment() -> None:
    trace = _trace()
    trace["qvel"][2, 1, 0] = np.inf
    trace["observation"][1, 0, 1] = np.nan

    result = _first_nonfinite(trace)
    assert result is not None
    value = result.pop("value")
    assert result == {
        "step": 1,
        "field": "observation",
        "environment_index": 0,
        "element_index": [1],
    }
    assert np.isnan(value)


def test_diagnostic_finds_first_tolerance_divergence() -> None:
    reference = _trace()
    candidate = _trace()
    candidate["qpos"][2, 1, 0] = 0.1
    candidate["reward"][1, 0] = 0.2

    divergence = _first_divergence(reference, candidate, rtol=1e-3, atol=1e-4)

    assert divergence is not None
    assert divergence["step"] == 1
    assert divergence["field"] == "reward"
    assert divergence["environment_index"] == 0


def test_diagnostic_classifies_nonfinite_without_capacity_overflow() -> None:
    summary = {
        "first_nonfinite": {"step": 4, "field": "qvel"},
        "capacity_high_water": {"overflow": False},
    }

    assert (
        _classify_warp_run(
            summary,
            capacity_divergence=None,
            jax_divergence={"step": 1, "field": "qpos"},
        )
        == "numerical_or_contact_instability_without_capacity_exhaustion"
    )
