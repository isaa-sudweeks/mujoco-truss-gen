from __future__ import annotations

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.constants import TENDON_MATERIAL, TENDON_RGBA
from mujoco_truss_gen.mujoco_model.model_types import EdgeKey, EdgeTendonMap


def edge_key(from_node_name: str, to_node_name: str) -> EdgeKey:
    return tuple(sorted((from_node_name, to_node_name)))


def actuator_name_for_tendon(tendon_name: str) -> str:
    edge_name = tendon_name.removeprefix("tendon_")
    nodes = edge_name.split("_node_")
    if len(nodes) != 2:
        return f"act_{edge_name}"

    node_a = nodes[0].removeprefix("node_")
    node_b = nodes[1].removeprefix("node_")
    node_a, node_b = sorted((node_a, node_b), key=_node_suffix_sort_key)
    if node_a.isdigit() and node_b.isdigit() and len(node_a) == 1 and len(node_b) == 1:
        return f"act_{node_a}{node_b}"
    return f"act_{node_a}_{node_b}"


def _node_suffix_sort_key(node_suffix: str) -> tuple[int, int | str]:
    if node_suffix.isdigit():
        return (0, int(node_suffix))
    return (1, node_suffix)


def add_tendon(
    spec: mujoco.MjSpec,
    from_node_name: str,
    to_node_name: str,
    *,
    tendon_range: list[float] | tuple[float, float] | None = None,
) -> str:
    tendon_name = f"tendon_{from_node_name}_{to_node_name}"
    tendon = spec.add_tendon(
        name=tendon_name,
        range=tendon_range if tendon_range is not None else [0.5, 2.0],
        width=0.05,
        rgba=TENDON_RGBA,
        material=TENDON_MATERIAL,
    )
    tendon.wrap_site(from_node_name)
    tendon.wrap_site(to_node_name)
    return tendon_name


def add_route_tendon(
    spec: mujoco.MjSpec,
    route_name: str,
    route: list[str],
    *,
    tendon_range: list[float] | tuple[float, float] | None = None,
) -> str:
    tendon_name = f"route_{route_name}"
    tendon = spec.add_tendon(
        name=tendon_name,
        range=tendon_range if tendon_range is not None else [0.5, 10.0],
        width=0.02,
        rgba=TENDON_RGBA,
        material=TENDON_MATERIAL,
    )
    for node_name in route:
        tendon.wrap_site(node_name)
    return tendon_name


def add_edge_tendon(
    spec: mujoco.MjSpec,
    edge_tendons: EdgeTendonMap,
    from_node_name: str,
    to_node_name: str,
    *,
    tendon_range: list[float] | tuple[float, float] | None = None,
) -> str:
    key = edge_key(from_node_name, to_node_name)
    if key not in edge_tendons:
        edge_tendons[key] = add_tendon(
            spec,
            from_node_name,
            to_node_name,
            tendon_range=tendon_range,
        )
    return edge_tendons[key]


def unique_actuator_name(tendon_name: str, used_names: set[str] | None = None) -> str:
    base_name = actuator_name_for_tendon(tendon_name)
    if used_names is None or base_name not in used_names:
        return base_name

    suffix = 2
    while f"{base_name}_{suffix}" in used_names:
        suffix += 1
    return f"{base_name}_{suffix}"


def add_actuator(
    spec: mujoco.MjSpec,
    tendon_name: str,
    kp: float,
    dampratio: float,
    used_names: set[str] | None = None,
    *,
    actrange: list[float] | tuple[float, float] | None = None,
) -> None:
    actuator_name = unique_actuator_name(tendon_name, used_names)
    actuator = spec.add_actuator(
        name=actuator_name,
        trntype=mujoco.mjtTrn.mjTRN_TENDON,
        target=tendon_name,
        ctrllimited=True,
        ctrlrange=[-0.05, 0.05],
        actlimited=True,
        actrange=actrange if actrange is not None else [0.0, 3.0],
    )
    if used_names is not None:
        used_names.add(actuator_name)
    actuator.set_to_intvelocity(kp=kp, dampratio=dampratio)


def add_realistic_actuator(
    spec: mujoco.MjSpec,
    tendon_name: str,
    kp: float,
    dampratio: float,
    used_names: set[str] | None = None,
) -> None:
    actuator_name = unique_actuator_name(tendon_name, used_names)
    actuator = spec.add_actuator(
        name=actuator_name,
        trntype=mujoco.mjtTrn.mjTRN_TENDON,
        target=tendon_name,
        ctrllimited=True,
        ctrlrange=[-0.05, 0.05],
        actlimited=True,
        actrange=[0.0, 3.0],
    )
    if used_names is not None:
        used_names.add(actuator_name)
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
