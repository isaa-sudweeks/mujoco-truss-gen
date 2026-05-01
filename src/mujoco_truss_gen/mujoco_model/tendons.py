from __future__ import annotations

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.constants import TENDON_RGBA
from mujoco_truss_gen.mujoco_model.model_types import EdgeKey, EdgeTendonMap


def edge_key(from_node_name: str, to_node_name: str) -> EdgeKey:
    return tuple(sorted((from_node_name, to_node_name)))


def add_tendon(spec: mujoco.MjSpec, from_node_name: str, to_node_name: str) -> str:
    tendon_name = f"tendon_{from_node_name}_{to_node_name}"
    tendon = spec.add_tendon(
        name=tendon_name,
        range=[0.5, 2.0],
        width=0.05,
        rgba=TENDON_RGBA,
    )
    tendon.wrap_site(from_node_name)
    tendon.wrap_site(to_node_name)
    return tendon_name


def add_edge_tendon(
    spec: mujoco.MjSpec,
    edge_tendons: EdgeTendonMap,
    from_node_name: str,
    to_node_name: str,
) -> str:
    key = edge_key(from_node_name, to_node_name)
    if key not in edge_tendons:
        edge_tendons[key] = add_tendon(spec, from_node_name, to_node_name)
    return edge_tendons[key]


def add_actuator(spec: mujoco.MjSpec, tendon_name: str, kp: float, dampratio: float) -> None:
    actuator = spec.add_actuator(
        name=f"act_{tendon_name}",
        trntype=mujoco.mjtTrn.mjTRN_TENDON,
        target=tendon_name,
        ctrllimited=True,
        ctrlrange=[-0.05, 0.05],
        actlimited=True,
        actrange=[0.0, 3.0],
    )
    actuator.set_to_intvelocity(kp=kp, dampratio=dampratio)


def add_realistic_actuator(spec: mujoco.MjSpec, tendon_name: str, kp: float, dampratio: float) -> None:
    actuator = spec.add_actuator(
        name=f"act_{tendon_name}",
        trntype=mujoco.mjtTrn.mjTRN_TENDON,
        target=tendon_name,
        ctrllimited=True,
        ctrlrange=[-0.05, 0.05],
        actlimited=True,
        actrange=[0.0, 3.0],
    )
    actuator.dyntype = mujoco.mjtDyn.mjDYN_INTEGRATOR
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE

    nominal_mass = 1.0
    kv = 2.0 * dampratio * np.sqrt(kp * nominal_mass)
    actuator.gainprm[0] = kp
    actuator.biasprm[1] = -kp
    actuator.biasprm[2] = -kv


def initialize_actuator_lengths(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Initialize tendon-integrator actuator state to the current tendon lengths."""
    if model.na == 0:
        return

    mujoco.mj_forward(model, data)
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_TENDON:
            continue
        if model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_INTEGRATOR:
            continue

        act_adr = model.actuator_actadr[actuator_id]
        if act_adr < 0:
            continue

        tendon_id = model.actuator_trnid[actuator_id, 0]
        data.act[act_adr] = data.ten_length[tendon_id]

    mujoco.mj_forward(model, data)
