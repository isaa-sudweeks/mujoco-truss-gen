from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass

import mujoco
import numpy as np

ANGLE_BISECTOR_ACTUATOR_PREFIX = "bisector_act_"


@dataclass(frozen=True, slots=True)
class AngleBisectorTarget:
    node_name: str
    neighbor_names: tuple[str, ...]
    actuator_id: int
    node_body_id: int
    parent_body_id: int
    node_site_id: int
    neighbor_site_ids: tuple[int, ...]
    initial_rod_vector: np.ndarray
    hinge_axis: np.ndarray


@dataclass(frozen=True, slots=True)
class NodeVelocityEdge:
    tendon_name: str
    actuator_id: int
    from_node: str
    to_node: str


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
            neighbor_positions = [
                data.site_xpos[site_id] for site_id in target.neighbor_site_ids
            ]

            if len(neighbor_positions) == 1:
                target_world = _unit_vector(node_pos - neighbor_positions[0])
                if target_world is None:
                    continue
            else:
                dir_a = _unit_vector(neighbor_positions[0] - node_pos)
                dir_b = _unit_vector(neighbor_positions[1] - node_pos)
                if dir_a is None or dir_b is None:
                    continue

                bisector_world = _unit_vector(dir_a + dir_b)
                if bisector_world is None:
                    continue
                target_world = -bisector_world

            parent_xmat = data.xmat[target.parent_body_id].reshape(3, 3)
            target_parent = parent_xmat.T @ target_world
            angle = _signed_angle_about_axis(
                target.initial_rod_vector,
                target_parent,
                target.hinge_axis,
            )
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
                hinge_axis = _hinge_axis(node_body, node_name)
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
                        hinge_axis=hinge_axis,
                    )
                )

        route_neighbors = _route_target_neighbors_from_xml(root)
        for node_name, neighbor_names in route_neighbors.items():
            if len(neighbor_names) not in {1, 2}:
                continue

            node_body = worldbody.find(f"./body[@name='{node_name}']")
            if node_body is None:
                continue

            initial_rod_vector = _rod_vector(node_body, node_name)
            if initial_rod_vector is None:
                continue

            actuator_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                angle_bisector_actuator_name(node_name),
            )
            node_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node_name)
            node_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, node_name)
            neighbor_site_ids = tuple(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                for name in neighbor_names
            )
            if (
                actuator_id < 0
                or node_body_id < 0
                or node_site_id < 0
                or any(site_id < 0 for site_id in neighbor_site_ids)
            ):
                continue

            parent_body_id = int(model.body_parentid[node_body_id])
            targets.append(
                AngleBisectorTarget(
                    node_name=node_name,
                    neighbor_names=neighbor_names,
                    actuator_id=actuator_id,
                    node_body_id=node_body_id,
                    parent_body_id=parent_body_id,
                    node_site_id=node_site_id,
                    neighbor_site_ids=neighbor_site_ids,
                    initial_rod_vector=initial_rod_vector,
                    hinge_axis=_hinge_axis(node_body, node_name),
                )
            )

        return targets


class NodeVelocityController:
    """Map node-level scalar velocity commands to routed-tube edge actuators."""

    def __init__(
        self,
        model: mujoco.MjModel,
        xml: str | None,
        node_names: Iterable[str],
        site_to_node: dict[str, str],
        external_actuator_ids: Iterable[int],
    ):
        self.node_names = list(node_names)
        self.node_index = {node_name: index for index, node_name in enumerate(self.node_names)}

        route_node_paths = _route_node_paths_from_xml(xml, site_to_node) if xml else []
        self.route_node_paths = route_node_paths
        passive_nodes = {
            node_name
            for route in route_node_paths
            for node_name in (route[0], route[-1])
            if node_name in self.node_index
        }
        self.passive_node_names = sorted(
            passive_nodes,
            key=lambda node_name: self.node_index[node_name],
        )
        self.passive_node_mask = np.array(
            [node_name in passive_nodes for node_name in self.node_names],
            dtype=bool,
        )

        oriented_edges = _first_route_edge_orientations(route_node_paths)
        tendon_site_nodes = _tendon_site_nodes_from_xml(xml, site_to_node) if xml else {}
        self.edges = _node_velocity_edges(
            model,
            external_actuator_ids,
            oriented_edges,
            tendon_site_nodes,
            self.node_index,
        )
        self.edge_names = [edge.tendon_name for edge in self.edges]
        self.actuator_ids = np.array([edge.actuator_id for edge in self.edges], dtype=int)
        self.incidence_matrix = self._build_incidence_matrix()
        self.latest_raw_node_commands = np.zeros(len(self.node_names), dtype=float)
        self.latest_node_commands = np.zeros(len(self.node_names), dtype=float)
        self.latest_edge_commands = np.zeros(len(self.edges), dtype=float)

    @property
    def enabled(self) -> bool:
        return bool(self.route_node_paths and self.edges)

    def transform(self, node_commands: np.ndarray) -> np.ndarray:
        node_commands = np.asarray(node_commands, dtype=float)
        if node_commands.shape != (len(self.node_names),):
            raise ValueError(
                f"Expected {len(self.node_names)} node command(s), got shape "
                f"{node_commands.shape}."
            )

        effective_node_commands = node_commands.copy()
        effective_node_commands[self.passive_node_mask] = 0.0
        edge_commands = self.incidence_matrix @ effective_node_commands

        self.latest_raw_node_commands = node_commands.copy()
        self.latest_node_commands = effective_node_commands.copy()
        self.latest_edge_commands = edge_commands.copy()
        return edge_commands

    def clipped_edge_commands(self, model: mujoco.MjModel, node_commands: np.ndarray) -> np.ndarray:
        edge_commands = self.transform(node_commands)
        if edge_commands.size == 0:
            return edge_commands

        ctrlrange = model.actuator_ctrlrange[self.actuator_ids]
        clipped = np.clip(edge_commands, ctrlrange[:, 0], ctrlrange[:, 1])
        self.latest_edge_commands = clipped.copy()
        return clipped

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        node_commands: np.ndarray,
    ) -> np.ndarray:
        edge_commands = self.clipped_edge_commands(model, node_commands)
        data.ctrl[self.actuator_ids] = edge_commands
        return edge_commands

    def _build_incidence_matrix(self) -> np.ndarray:
        incidence = np.zeros((len(self.edges), len(self.node_names)), dtype=float)
        for row, edge in enumerate(self.edges):
            incidence[row, self.node_index[edge.from_node]] = -1.0
            incidence[row, self.node_index[edge.to_node]] = 1.0
        return incidence


def angle_bisector_actuator_name(node_name: str) -> str:
    return f"{ANGLE_BISECTOR_ACTUATOR_PREFIX}{node_name}"


def _route_node_paths_from_xml(xml: str | None, site_to_node: dict[str, str]) -> list[list[str]]:
    if not xml:
        return []

    root = ET.fromstring(xml)
    tendon_root = root.find("tendon")
    if tendon_root is None:
        return []

    routes = []
    for spatial in tendon_root.findall("spatial"):
        if not spatial.get("name", "").startswith("route_"):
            continue
        route = [
            site_to_node[site_name]
            for site_ref in spatial.findall("site")
            if (site_name := site_ref.get("site")) in site_to_node
        ]
        if len(route) >= 2:
            routes.append(route)
    return routes


def _tendon_site_nodes_from_xml(
    xml: str | None,
    site_to_node: dict[str, str],
) -> dict[str, tuple[str, ...]]:
    if not xml:
        return {}

    root = ET.fromstring(xml)
    tendon_root = root.find("tendon")
    if tendon_root is None:
        return {}

    tendon_sites = {}
    for spatial in tendon_root.findall("spatial"):
        tendon_name = spatial.get("name")
        if not tendon_name:
            continue
        nodes = tuple(
            site_to_node[site_name]
            for site_ref in spatial.findall("site")
            if (site_name := site_ref.get("site")) in site_to_node
        )
        if nodes:
            tendon_sites[tendon_name] = nodes
    return tendon_sites


def _first_route_edge_orientations(
    route_node_paths: list[list[str]],
) -> dict[tuple[str, str], tuple[str, str]]:
    orientations = {}
    for route in route_node_paths:
        for from_node, to_node in zip(route, route[1:], strict=False):
            key = tuple(sorted((from_node, to_node)))
            orientations.setdefault(key, (from_node, to_node))
    return orientations


def _node_velocity_edges(
    model: mujoco.MjModel,
    external_actuator_ids: Iterable[int],
    oriented_edges: dict[tuple[str, str], tuple[str, str]],
    tendon_site_nodes: dict[str, tuple[str, ...]],
    node_index: dict[str, int],
) -> list[NodeVelocityEdge]:
    edges = []
    for actuator_id in external_actuator_ids:
        tendon_id = int(model.actuator_trnid[actuator_id, 0])
        if tendon_id < 0:
            continue

        tendon_name = model.tendon(tendon_id).name
        if not tendon_name.startswith("tendon_"):
            continue

        node_pair = _edge_nodes_for_tendon(tendon_name, tendon_site_nodes)
        if node_pair is None:
            continue

        key = tuple(sorted(node_pair))
        from_node, to_node = oriented_edges.get(key, node_pair)
        if from_node not in node_index or to_node not in node_index:
            continue
        edges.append(
            NodeVelocityEdge(
                tendon_name=tendon_name,
                actuator_id=int(actuator_id),
                from_node=from_node,
                to_node=to_node,
            )
        )
    return edges


def _edge_nodes_for_tendon(
    tendon_name: str,
    tendon_site_nodes: dict[str, tuple[str, ...]],
) -> tuple[str, str] | None:
    sites = tendon_site_nodes.get(tendon_name, ())
    if len(sites) == 2 and sites[0] != sites[1]:
        return sites[0], sites[1]

    edge = tendon_name.removeprefix("tendon_").split("_node_")
    if len(edge) != 2:
        return None
    node_a = edge[0] if edge[0].startswith("node_") else f"node_{edge[0]}"
    node_b = f"node_{edge[1]}"
    if node_a == node_b:
        return None
    return node_a, node_b


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


def _hinge_axis(node_body: ET.Element, node_name: str) -> np.ndarray:
    joint = node_body.find(f"./joint[@name='{node_name}_z_hinge']")
    if joint is None:
        return np.array([0.0, 0.0, 1.0])

    axis = np.fromstring(joint.get("axis", "0 0 1"), sep=" ", dtype=float)
    if axis.size != 3:
        return np.array([0.0, 0.0, 1.0])

    unit = _unit_vector(axis)
    if unit is None:
        return np.array([0.0, 0.0, 1.0])
    return unit


def _route_target_neighbors_from_xml(root: ET.Element) -> dict[str, tuple[str, ...]]:
    tendon_root = root.find("tendon")
    if tendon_root is None:
        return {}

    neighbors: dict[str, list[str]] = {}
    for spatial in tendon_root.findall("spatial"):
        if not spatial.get("name", "").startswith("route_"):
            continue
        route = [
            site_ref.get("site")
            for site_ref in spatial.findall("site")
            if site_ref.get("site")
        ]
        for index, node_name in enumerate(route):
            adjacent = []
            if index > 0:
                adjacent.append(route[index - 1])
            if index < len(route) - 1:
                adjacent.append(route[index + 1])
            if not adjacent:
                continue
            neighbors.setdefault(node_name, [])
            for neighbor_name in adjacent:
                if neighbor_name not in neighbors[node_name]:
                    neighbors[node_name].append(neighbor_name)

    return {node_name: tuple(node_neighbors) for node_name, node_neighbors in neighbors.items()}


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        return None
    return vector / norm


def _signed_angle_about_axis(
    from_vector: np.ndarray,
    to_vector: np.ndarray,
    axis: np.ndarray,
) -> float | None:
    axis_unit = _unit_vector(axis)
    if axis_unit is None:
        return None

    from_projected = from_vector - axis_unit * float(np.dot(from_vector, axis_unit))
    to_projected = to_vector - axis_unit * float(np.dot(to_vector, axis_unit))
    from_unit = _unit_vector(from_projected)
    to_unit = _unit_vector(to_projected)
    if from_unit is None or to_unit is None:
        return None

    cross = np.cross(from_unit, to_unit)
    signed_cross = float(np.dot(axis_unit, cross))
    dot = float(np.clip(np.dot(from_unit, to_unit), -1.0, 1.0))
    return math.atan2(signed_cross, dot)
