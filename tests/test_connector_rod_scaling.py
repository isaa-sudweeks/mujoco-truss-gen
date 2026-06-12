from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from mujoco_truss_gen import MujocoModel, TrussPhysicalParameters, get_mujoco_spec


def _connector_rod_lengths(scale: float, configured_length: float | None) -> np.ndarray:
    root = ET.fromstring(
        get_mujoco_spec(
            "octahedron",
            realistic=True,
            scale=scale,
            physical_params=TrussPhysicalParameters(
                connector_rod_length=configured_length,
            ),
        ).to_xml()
    )
    return np.asarray(
        [
            np.linalg.norm(np.fromstring(site.get("pos", ""), sep=" "))
            for body in root.findall(".//body")
            if body.get("name", "").startswith("rod_")
            for site in [body.find("site")]
            if site is not None
        ],
        dtype=float,
    )


def _actuated_tendon_lengths(scale: float, connector_rod_length: float) -> np.ndarray:
    model = MujocoModel(
        get_mujoco_spec(
            "octahedron",
            realistic=True,
            scale=scale,
            physical_params=TrussPhysicalParameters(
                connector_rod_length=connector_rod_length,
            ),
        )
    )
    tendon_ids = model.model.actuator_trnid[model.external_actuator_ids, 0]
    return model.data.ten_length[tendon_ids].copy()


def test_absolute_connector_rod_length_is_independent_of_preset_scale() -> None:
    configured_length = 0.2
    small_rods = _connector_rod_lengths(0.5, configured_length)
    large_rods = _connector_rod_lengths(2.0, configured_length)

    assert small_rods.size == 12
    assert large_rods.size == 12
    np.testing.assert_allclose(small_rods, configured_length, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(large_rods, configured_length, rtol=2e-4, atol=2e-5)

    small_tendons = _actuated_tendon_lengths(0.5, configured_length)
    large_tendons = _actuated_tendon_lengths(2.0, configured_length)
    np.testing.assert_allclose(large_tendons, 4.0 * small_tendons, rtol=2e-4, atol=2e-5)


def test_default_clone_offset_keeps_legacy_scale_proportional_rods() -> None:
    unit_rods = _connector_rod_lengths(1.0, None)
    doubled_rods = _connector_rod_lengths(2.0, None)

    np.testing.assert_allclose(doubled_rods, 2.0 * unit_rods, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("value", [0.0, -0.1, float("inf"), float("nan")])
def test_connector_rod_length_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError, match="connector_rod_length must be greater than zero"):
        get_mujoco_spec(
            "octahedron",
            realistic=True,
            physical_params=TrussPhysicalParameters(connector_rod_length=value),
        )
