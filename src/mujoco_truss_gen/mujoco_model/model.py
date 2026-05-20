from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.controllers import (
    ANGLE_BISECTOR_ACTUATOR_PREFIX,
    AngleBisectorController,
)
from mujoco_truss_gen.mujoco_model.tendons import initialize_actuator_lengths

ModelSource = mujoco.MjSpec | mujoco.MjModel | str | Path


class MujocoModel:
    """Compiled MuJoCo model plus truss-specific metadata and query helpers."""

    def __init__(self, source: ModelSource):
        self.xml: str | None = None

        if isinstance(source, mujoco.MjSpec):
            self.xml = source.to_xml()
            self.model = source.compile()
        elif isinstance(source, mujoco.MjModel):
            self.model = source
        elif isinstance(source, str | Path):
            text_or_path = str(source)
            path = Path(text_or_path)
            if path.exists():
                self.xml = path.read_text(encoding="utf-8")
                self.model = mujoco.MjModel.from_xml_path(str(path))
            elif text_or_path.lstrip().startswith("<"):
                self.xml = text_or_path
                self.model = mujoco.MjModel.from_xml_string(text_or_path)
            else:
                raise FileNotFoundError(f"MuJoCo XML file does not exist: {source}")
        else:
            raise TypeError(
                "MujocoModel source must be a mujoco.MjSpec, mujoco.MjModel, XML string, "
                "or XML path."
            )

        self.data = mujoco.MjData(self.model)
        if self.xml is not None:
            self._load_model_metadata_from_xml(self.xml)
        else:
            self._load_model_metadata_from_model()

        self.angle_bisector_controller = AngleBisectorController(self.model, self.xml)
        self.internal_actuator_ids = np.array(
            [
                actuator_id
                for actuator_id in range(self.model.nu)
                if self.model.actuator(actuator_id).name.startswith(
                    ANGLE_BISECTOR_ACTUATOR_PREFIX
                )
            ],
            dtype=int,
        )
        internal_actuator_ids = set(self.internal_actuator_ids.tolist())
        self.external_actuator_ids = np.array(
            [
                actuator_id
                for actuator_id in range(self.model.nu)
                if actuator_id not in internal_actuator_ids
            ],
            dtype=int,
        )
        self.internal_actuator_names = [
            self.model.actuator(actuator_id).name
            for actuator_id in self.internal_actuator_ids
        ]
        self.external_actuator_names = [
            self.model.actuator(actuator_id).name
            for actuator_id in self.external_actuator_ids
        ]
        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.ctrl_home = np.zeros(self.model.nu, dtype=float)
        self.act_home = np.ones(self.model.na, dtype=float)
        mujoco.mj_forward(self.model, self.data)
        self.apply_angle_bisector_control()
        mujoco.mj_forward(self.model, self.data)
        initialize_actuator_lengths(self.model, self.data)
        self.init_act = self.data.act.copy()
        self.wcrm = False
        self.initial_critical_eig = max(self._critical_eig(), 1e-8)

    def _load_model_metadata_from_xml(self, xml: str) -> None:
        root = ET.fromstring(xml)

        self.node_names: list[str] = []
        self.node_axes: dict[str, tuple[str, ...]] = {}
        self.node_body_ids: dict[str, int] = {}
        self.site_to_node: dict[str, str] = {}

        def dominant_axis(axis_str: str) -> str:
            axis = np.fromstring(axis_str, sep=" ", dtype=float)
            if axis.size != 3:
                raise ValueError(f"Invalid joint axis '{axis_str}' in MuJoCo XML")
            return "xyz"[int(np.argmax(np.abs(axis)))]

        def visit_body(body_elem: ET.Element, inherited_node: str | None = None) -> None:
            body_name = body_elem.get("name")
            current_node = inherited_node

            if body_name and body_name.startswith("node_"):
                current_node = body_name
                self.node_names.append(body_name)
                joint_axes = []
                for joint in body_elem.findall("joint"):
                    if joint.get("type", "hinge") == "slide":
                        joint_axes.append(dominant_axis(joint.get("axis", "0 0 0")))
                self.node_axes[body_name] = tuple(sorted(joint_axes, key="xyz".index))
                self.node_body_ids[body_name] = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_name,
                )

            if current_node is not None:
                for site in body_elem.findall("site"):
                    site_name = site.get("name")
                    if site_name:
                        self.site_to_node[site_name] = current_node

            for child_body in body_elem.findall("body"):
                visit_body(child_body, current_node)

        worldbody = root.find("worldbody")
        if worldbody is not None:
            for body in worldbody.findall("body"):
                visit_body(body)

        self._finalize_node_metadata()
        self.structural_edges = self._structural_edges_from_xml(root)

    def _load_model_metadata_from_model(self) -> None:
        self.node_names = [
            self.model.body(body_id).name
            for body_id in range(self.model.nbody)
            if self.model.body(body_id).name.startswith("node_")
        ]
        self.node_body_ids = {
            node_name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, node_name)
            for node_name in self.node_names
        }
        self.node_axes = {
            node_name: self._body_slide_axes(body_id)
            for node_name, body_id in self.node_body_ids.items()
        }
        self.site_to_node = {}
        self._finalize_node_metadata()
        self.structural_edges = self._structural_edges_from_model_names()

    def _body_slide_axes(self, body_id: int) -> tuple[str, ...]:
        axes = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_bodyid[joint_id] != body_id:
                continue
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_SLIDE:
                continue
            axis = self.model.jnt_axis[joint_id]
            axes.append("xyz"[int(np.argmax(np.abs(axis)))])
        return tuple(sorted(axes, key="xyz".index))

    def _finalize_node_metadata(self) -> None:
        self.node_names = sorted(
            dict.fromkeys(self.node_names),
            key=lambda name: _node_sort_key(name),
        )
        self.active_axes = self.node_axes[self.node_names[0]] if self.node_names else ("x", "z")
        if not self.active_axes:
            self.active_axes = ("x", "z")
        self.axis_indices = tuple("xyz".index(axis) for axis in self.active_axes)

    def set_wcrm(self, wcrm: bool) -> None:
        self.wcrm = wcrm

    def _uses_realistic_triangle_bodies(self) -> bool:
        return any(
            self.model.body(body_id).name.startswith("tri_")
            for body_id in range(self.model.nbody)
        )

    def _structural_edges_from_xml(self, root: ET.Element) -> list[tuple[str, str]]:
        structural_tendon_names = set()
        actuator = root.find("actuator")
        if actuator is not None:
            for actuator_elem in actuator:
                tendon_name = actuator_elem.get("tendon")
                if tendon_name:
                    structural_tendon_names.add(tendon_name)

        equality = root.find("equality")
        if equality is not None:
            for constraint in equality.findall("tendon"):
                for attr_name in ("tendon1", "tendon2"):
                    tendon_name = constraint.get(attr_name)
                    if tendon_name:
                        structural_tendon_names.add(tendon_name)

        tendon_defs = {}
        tendon_root = root.find("tendon")
        if tendon_root is not None:
            for spatial in tendon_root.findall("spatial"):
                sites = [site_ref.get("site") for site_ref in spatial.findall("site")]
                tendon_defs[spatial.get("name")] = [site for site in sites if site]

        structural_edges = []
        for tendon_name in sorted(structural_tendon_names):
            sites = tendon_defs.get(tendon_name, [])
            if len(sites) != 2:
                continue
            node_pair = tuple(self.site_to_node.get(site_name) for site_name in sites)
            if None not in node_pair and node_pair[0] != node_pair[1]:
                structural_edges.append((node_pair[0], node_pair[1]))
        structural_edge_keys = {tuple(sorted(edge)) for edge in structural_edges}

        for tendon_name, sites in sorted(tendon_defs.items()):
            if not tendon_name.startswith("tendon_") or len(sites) != 2:
                continue
            node_pair = tuple(self.site_to_node.get(site_name) for site_name in sites)
            if None in node_pair or node_pair[0] == node_pair[1]:
                continue
            key = tuple(sorted(node_pair))
            if key in structural_edge_keys:
                continue
            structural_edges.append((node_pair[0], node_pair[1]))
            structural_edge_keys.add(key)

        return structural_edges

    def _structural_edges_from_model_names(self) -> list[tuple[str, str]]:
        structural_edges = []
        for tendon_id in range(self.model.ntendon):
            tendon_name = self.model.tendon(tendon_id).name
            if not tendon_name.startswith("tendon_"):
                continue
            edge = tendon_name.removeprefix("tendon_").split("_node_")
            if len(edge) != 2:
                continue
            node_a = edge[0] if edge[0].startswith("node_") else f"node_{edge[0]}"
            node_b = f"node_{edge[1]}"
            if node_a in self.node_body_ids and node_b in self.node_body_ids:
                structural_edges.append((node_a, node_b))
        return structural_edges

    def reset(self, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng()
        self.data.qpos[:] = self.init_qpos + rng.uniform(-0.005, 0.005, size=self.model.nq)
        mujoco.mj_normalizeQuat(self.model, self.data.qpos)
        self.data.qvel[:] = self.init_qvel + rng.uniform(-0.005, 0.005, size=self.model.nv)
        self.data.ctrl[:] = self.ctrl_home.copy()
        if self.model.na:
            self.data.act[:] = self.init_act.copy()
        mujoco.mj_forward(self.model, self.data)
        self.apply_angle_bisector_control()
        mujoco.mj_forward(self.model, self.data)

    def apply_angle_bisector_control(self) -> None:
        self.angle_bisector_controller.update(self.model, self.data)

    def get_external_ctrlrange(self) -> np.ndarray:
        return self.model.actuator_ctrlrange[self.external_actuator_ids]

    def get_external_ctrl(self) -> np.ndarray:
        return self.data.ctrl[self.external_actuator_ids].copy()

    def set_external_ctrl(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[self.external_actuator_ids] = ctrl

    def get_node_loc_dict(self) -> dict[str, np.ndarray]:
        return {self.model.body(i).name: self.data.xpos[i].copy() for i in range(self.model.nbody)}

    def get_node_velocity_dict(self) -> dict[str, np.ndarray]:
        return {self.model.body(i).name: self.data.cvel[i].copy() for i in range(self.model.nbody)}

    def get_edge_length_dict(self) -> dict[str, float]:
        return {
            self.model.tendon(ten).name: float(self.data.ten_length[ten])
            for ten in range(self.model.ntendon)
        }

    def get_edge_velocity_dict(self) -> dict[str, float]:
        return {
            self.model.tendon(ten).name: float(self.data.ten_velocity[ten])
            for ten in range(self.model.ntendon)
        }

    def get_node_position_dict(self) -> dict[str, np.ndarray]:
        return {
            node_name: self.data.xpos[self.node_body_ids[node_name]].copy()
            for node_name in self.node_names
        }

    def get_node_velocity_linear_dict(self) -> dict[str, np.ndarray]:
        return {
            node_name: self.data.cvel[self.node_body_ids[node_name]][3:].copy()
            for node_name in self.node_names
        }

    def get_node_position_matrix(self) -> np.ndarray:
        return np.array(
            [self.data.xpos[self.node_body_ids[node_name]] for node_name in self.node_names]
        )

    def get_node_linear_velocity_matrix(self) -> np.ndarray:
        return np.array(
            [self.data.cvel[self.node_body_ids[node_name]][3:] for node_name in self.node_names]
        )

    def _logical_node_name(self, node_name: str) -> str:
        return node_name.split("_tri_", 1)[0]

    def _logical_rigidity_graph(
        self,
    ) -> tuple[list[str], dict[str, np.ndarray], list[tuple[str, str]], tuple[int, ...]]:
        physical_positions = self.get_node_position_dict()
        logical_instances: dict[str, list[str]] = {}
        for node_name in self.node_names:
            logical_instances.setdefault(self._logical_node_name(node_name), []).append(node_name)

        node_names = sorted(logical_instances, key=_node_sort_key)
        node_positions = {}
        for node_name in node_names:
            connector_body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"connector_ball_{node_name}",
            )
            if connector_body_id >= 0:
                node_positions[node_name] = self.data.xpos[connector_body_id].copy()
                continue

            node_positions[node_name] = np.mean(
                [physical_positions[instance] for instance in logical_instances[node_name]],
                axis=0,
            )

        edges = []
        edge_keys = set()
        for node_a, node_b in self.structural_edges:
            logical_a = self._logical_node_name(node_a)
            logical_b = self._logical_node_name(node_b)
            if logical_a == logical_b:
                continue
            key = tuple(sorted((logical_a, logical_b)))
            if key in edge_keys:
                continue
            edges.append((logical_a, logical_b))
            edge_keys.add(key)

        return node_names, node_positions, edges, (0, 1, 2)

    def _rigidity_matrix_data(self) -> tuple[np.ndarray, int]:
        if self._uses_realistic_triangle_bodies():
            node_names, node_positions, structural_edges, axis_indices = (
                self._logical_rigidity_graph()
            )
        else:
            node_names = self.node_names
            node_positions = self.get_node_position_dict()
            structural_edges = self.structural_edges
            axis_indices = self.axis_indices

        dims = len(axis_indices)
        num_nodes = len(node_names)
        rows = []

        for node_a, node_b in structural_edges:
            pa = node_positions[node_a][list(axis_indices)]
            pb = node_positions[node_b][list(axis_indices)]
            delta = pb - pa
            length = np.linalg.norm(delta)
            if length < 1e-8:
                continue

            direction = delta / length
            row = np.zeros(num_nodes * dims, dtype=float)
            ia = node_names.index(node_a) * dims
            ib = node_names.index(node_b) * dims
            row[ia : ia + dims] = -direction
            row[ib : ib + dims] = direction
            rows.append(row)

        if not rows:
            return np.zeros((0, num_nodes * dims), dtype=float), dims
        return np.vstack(rows), dims

    def _rigidity_matrix(self) -> np.ndarray:
        return self._rigidity_matrix_data()[0]

    def _critical_eig(self) -> float:
        rigidity_matrix, dims = self._rigidity_matrix_data()
        if rigidity_matrix.size == 0:
            return 0.0

        if self.wcrm:
            norm = np.linalg.trace(rigidity_matrix.T @ rigidity_matrix)
        else:
            norm = 1.0
        eigvals = np.linalg.eigvalsh(rigidity_matrix.T @ rigidity_matrix)
        eigvals = np.sort(np.real(eigvals))
        rigid_body_modes = dims + (dims * (dims - 1)) // 2
        if eigvals.size <= rigid_body_modes:
            return 0.0
        return float(max(eigvals[rigid_body_modes] / norm, 0.0))

    def collapse_check(self) -> float:
        return self._critical_eig() / self.initial_critical_eig

    def get_forward_velocity(self) -> float:
        linear_velocities = self.get_node_linear_velocity_matrix()
        if linear_velocities.size == 0:
            return 0.0
        return float(np.mean(linear_velocities[:, 0]))

    def get_slip_penalty(self, height: float = 0.2) -> float:
        positions = self.get_node_position_matrix()
        linear_velocities = self.get_node_linear_velocity_matrix()
        if positions.size == 0:
            return 0.0
        contact_mask = positions[:, 2] < height
        return float(np.sum(np.abs(linear_velocities[contact_mask, 0])))


def _node_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_")[1]), name
    except (IndexError, ValueError):
        return 10**9, name
