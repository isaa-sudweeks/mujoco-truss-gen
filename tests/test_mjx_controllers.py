from __future__ import annotations

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from mujoco import mjx

from mujoco_truss_gen import MjxAngleBisectorController, MujocoModel, get_mujoco_spec
from mujoco_truss_gen.mjx_controllers import _nearest_equivalent_angles
from mujoco_truss_gen.mujoco_model.controllers import _nearest_equivalent_angle


@pytest.mark.parametrize(
    "preset_name",
    ("tetrahedron", "octahedron", "icosahedron", "solar_array"),
)
def test_mjx_angle_bisector_initialization_matches_cpu(preset_name: str) -> None:
    model = MujocoModel(get_mujoco_spec(preset_name, realistic=True))
    controller = MjxAngleBisectorController(model.angle_bisector_controller.targets)
    expected = model.data.ctrl.copy()
    data = mjx.put_data(model.model, model.data)
    data = data.replace(ctrl=jnp.zeros_like(data.ctrl))

    actual = jax.jit(controller.initialize)(data)

    np.testing.assert_allclose(
        actual.ctrl[model.internal_actuator_ids],
        expected[model.internal_actuator_ids],
        rtol=1e-6,
        atol=1e-6,
    )


def test_mjx_angle_bisector_update_matches_cpu_for_live_routed_plane() -> None:
    model = MujocoModel(get_mujoco_spec("tetrahedron", realistic=True))
    cpu_controller = model.angle_bisector_controller
    target = next(
        target
        for target in cpu_controller.targets
        if target.roll_actuator_id is not None and target.neighbor_candidate_site_ids
    )
    moved_site_id = target.neighbor_candidate_site_ids[0]
    moved_node_name = model.model.site(moved_site_id).name
    joint_id = mujoco.mj_name2id(
        model.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{moved_node_name}_z",
    )
    model.data.qpos[int(model.model.jnt_qposadr[joint_id])] += 0.35
    mujoco.mj_forward(model.model, model.data)
    mjx_data = mjx.put_data(model.model, model.data)

    cpu_controller.update(model.model, model.data)
    jax_controller = MjxAngleBisectorController(cpu_controller.targets)
    actual = jax.jit(jax_controller.update)(mjx_data)

    np.testing.assert_allclose(
        actual.ctrl[model.internal_actuator_ids],
        model.data.ctrl[model.internal_actuator_ids],
        rtol=1e-5,
        atol=1e-6,
    )


def test_mjx_angle_bisector_degenerate_target_retains_previous_control() -> None:
    model = MujocoModel(get_mujoco_spec("octahedron", realistic=True))
    controller = MjxAngleBisectorController(model.angle_bisector_controller.targets)
    target = model.angle_bisector_controller.targets[0]
    data = mjx.put_data(model.model, model.data)
    site_xpos = data.site_xpos
    for site_id in target.neighbor_site_ids:
        site_xpos = site_xpos.at[site_id].set(site_xpos[target.node_site_id])
    data = data.replace(site_xpos=site_xpos)

    actual = jax.jit(controller.update)(data)

    assert float(actual.ctrl[target.actuator_id]) == pytest.approx(
        float(data.ctrl[target.actuator_id])
    )


@pytest.mark.parametrize(
    ("angle", "reference"),
    ((-3.0, 3.0), (3.0, -3.0), (0.2, 8.0), (-0.2, -8.0)),
)
def test_mjx_nearest_equivalent_angle_matches_cpu(angle: float, reference: float) -> None:
    actual = _nearest_equivalent_angles(jnp.asarray(angle), jnp.asarray(reference))

    assert float(actual) == pytest.approx(_nearest_equivalent_angle(angle, reference), abs=1e-6)
