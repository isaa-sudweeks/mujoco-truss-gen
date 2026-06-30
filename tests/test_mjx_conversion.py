from __future__ import annotations

import mujoco
import pytest
from mujoco import mjx

from mujoco_truss_gen import PRESETS, get_mujoco_spec


def _is_indexed_henneberg_variant(preset_name: str) -> bool:
    if not preset_name.startswith("henneberg_"):
        return False
    return preset_name.rsplit("_", maxsplit=1)[-1].isdigit()


CANONICAL_PRESET_NAMES = tuple(
    preset_name for preset_name in PRESETS if not _is_indexed_henneberg_variant(preset_name)
)


@pytest.mark.parametrize("preset_name", CANONICAL_PRESET_NAMES)
def test_abstract_preset_model_converts_to_mjx(preset_name: str) -> None:
    mujoco_model = get_mujoco_spec(preset_name, realistic=False).compile()

    mjx_model = mjx.put_model(mujoco_model)

    for size_name in (
        "nq",
        "nv",
        "nu",
        "na",
        "nbody",
        "njnt",
        "ngeom",
        "nsite",
        "ntendon",
        "neq",
        "nsensor",
    ):
        assert getattr(mjx_model, size_name) == getattr(mujoco_model, size_name)

    for array_name in (
        "qpos0",
        "qpos_spring",
        "body_mass",
        "geom_type",
        "site_pos",
        "tendon_lengthspring",
        "eq_type",
        "sensor_type",
    ):
        assert getattr(mjx_model, array_name).shape == getattr(mujoco_model, array_name).shape


@pytest.mark.parametrize("preset_name", CANONICAL_PRESET_NAMES)
def test_realistic_preset_model_converts_to_mjx(preset_name: str) -> None:
    mujoco_model = get_mujoco_spec(preset_name, realistic=True).compile()

    mjx_model = mjx.put_model(mujoco_model)

    assert mjx_model.nq == mujoco_model.nq
    assert mjx_model.nu == mujoco_model.nu
    assert mjx_model.ngeom == mujoco_model.ngeom


def test_abstract_preset_state_converts_to_mjx() -> None:
    mujoco_model = get_mujoco_spec("tetrahedron", realistic=False).compile()
    mujoco_data = mujoco.MjData(mujoco_model)

    mjx_data = mjx.put_data(mujoco_model, mujoco_data)

    for array_name in ("qpos", "qvel", "act", "ctrl", "sensordata"):
        assert getattr(mjx_data, array_name).shape == getattr(mujoco_data, array_name).shape
