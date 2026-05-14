from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

ANGLE_BISECTOR_ACTUATOR_PREFIX = "bisector_act_"


@dataclass(frozen=True, slots=True)
class AngleBisectorTarget:
    node_name: str
    neighbor_names: tuple[str, str]
    actuator_id: int
    node_body_id: int
    parent_body_id: int
    node_site_id: int
    neighbor_site_ids: tuple[int, int]
    initial_rod_vector: np.ndarray


class AngleBisectorController:
    """Drive realistic connector rods to bisect their triangle tendon angles."""

    def __init__(self, model: mujoco.MjModel, xml: str | None):
        self.targets = self._targets_from_xml(model, xml) if xml else []

    @property
    def enabled(self) -> bool:
        return bool(self.targets)

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for target in self.targets:
            node_pos = data.site_xpos[target.node_site_id]
            neighbor_a = data.site_xpos[target.neighbor_site_ids[0]]
            neighbor_b = data.site_xpos[target.neighbor_site_ids[1]]

            dir_a = _unit_vector(neighbor_a - node_pos)
            dir_b = _unit_vector(neighbor_b - node_pos)
            if dir_a is None or dir_b is None:
                continue

            bisector_world = _unit_vector(dir_a + dir_b)
            if bisector_world is None:
                continue

            parent_xmat = data.xmat[target.parent_body_id].reshape(3, 3)
            bisector_parent = parent_xmat.T @ bisector_world
            angle = _signed_planar_angle(target.initial_rod_vector, -bisector_parent)
            if angle is None:
                continue

            data.ctrl[target.actuator_id] = angle

    @classmethod
    def _targets_from_xml(
        cls,
        model: mujoco.MjModel,
        xml: str | None,
    ) -> list[AngleBisectorTarget]:
        if not xml:
            return []

        root = ET.fromstring(xml)
        worldbody = root.find("worldbody")
        if worldbody is None:
            return []

        targets = []
        for triangle_body in worldbody.findall("body"):
            triangle_name = triangle_body.get("name", "")
            if not triangle_name.startswith("tri_"):
                continue

            node_bodies = [
                child
                for child in triangle_body.findall("body")
                if _body_name(child).startswith("node_")
            ]
            node_names = [_body_name(node_body) for node_body in node_bodies]
            if len(node_names) != 3:
                continue

            for node_body in node_bodies:
                node_name = _body_name(node_body)
                initial_rod_vector = _rod_vector(node_body, node_name)
                if initial_rod_vector is None:
                    continue

                actuator_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    angle_bisector_actuator_name(node_name),
                )
                if actuator_id < 0:
                    continue

                node_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node_name)
                node_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, node_name)
                neighbor_names = tuple(name for name in node_names if name != node_name)
                neighbor_site_ids = tuple(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                    for name in neighbor_names
                )
                if (
                    node_body_id < 0
                    or node_site_id < 0
                    or any(site_id < 0 for site_id in neighbor_site_ids)
                ):
                    continue

                parent_body_id = int(model.body_parentid[node_body_id])
                targets.append(
                    AngleBisectorTarget(
                        node_name=node_name,
                        neighbor_names=(neighbor_names[0], neighbor_names[1]),
                        actuator_id=actuator_id,
                        node_body_id=node_body_id,
                        parent_body_id=parent_body_id,
                        node_site_id=node_site_id,
                        neighbor_site_ids=(neighbor_site_ids[0], neighbor_site_ids[1]),
                        initial_rod_vector=initial_rod_vector,
                    )
                )

        return targets


def angle_bisector_actuator_name(node_name: str) -> str:
    return f"{ANGLE_BISECTOR_ACTUATOR_PREFIX}{node_name}"


def _body_name(body_elem: ET.Element) -> str:
    return body_elem.get("name", "")


def _rod_vector(node_body: ET.Element, node_name: str) -> np.ndarray | None:
    rod_body = node_body.find(f"./body[@name='rod_{node_name}']")
    if rod_body is None:
        return None

    tip_site = rod_body.find(f"./site[@name='tip_site_{node_name}']")
    if tip_site is None:
        return None

    pos = tip_site.get("pos")
    if pos is None:
        return None

    vector = np.fromstring(pos, sep=" ", dtype=float)
    if vector.size != 3:
        return None
    return vector


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        return None
    return vector / norm


def _signed_planar_angle(from_vector: np.ndarray, to_vector: np.ndarray) -> float | None:
    from_xy = from_vector[:2]
    to_xy = to_vector[:2]
    from_norm = float(np.linalg.norm(from_xy))
    to_norm = float(np.linalg.norm(to_xy))
    if from_norm < 1e-10 or to_norm < 1e-10:
        return None

    from_xy = from_xy / from_norm
    to_xy = to_xy / to_norm
    cross_z = from_xy[0] * to_xy[1] - from_xy[1] * to_xy[0]
    dot = float(np.clip(np.dot(from_xy, to_xy), -1.0, 1.0))
    return math.atan2(cross_z, dot)
