from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R

Vector = list[float]
NodeDict = dict[str, Vector]
TriangleDict = dict[str, list[str]]
EdgeKey = tuple[str, str]
EdgeTendonMap = dict[EdgeKey, str]

NODE_RADIUS = 0.1
NODE_MASS = 0.1
CONNECTOR_RADIUS = 0.05
CONNECTOR_MASS = 0.05
ROD_RADIUS = 0.025
ROD_MASS = 0.05
TRUSS_RGBA = [0.9, 0.9, 0.9, 1.0]
TENDON_RGBA = [0.0, 0.1804, 0.3647, 1.0]


def _as_mujoco_quat(rotation_matrix: np.ndarray) -> list[float]:
    """Convert a scipy xyzw quaternion to MuJoCo's wxyz convention."""
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    return [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]


def _triangle_frame(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> tuple[np.ndarray, list[float], list[Vector]]:
    """Return a triangle-local frame, body quaternion, and local node positions."""
    x_axis = p2 - p1
    edge_length = np.linalg.norm(x_axis)
    if edge_length > 1e-6:
        x_axis = x_axis / edge_length
    else:
        x_axis = np.array([1.0, 0.0, 0.0])

    p1_to_p3 = p3 - p1
    z_axis = np.cross(x_axis, p1_to_p3)
    z_norm = np.linalg.norm(z_axis)
    if z_norm > 1e-6:
        z_axis = z_axis / z_norm
    else:
        z_axis = np.array([0.0, 0.0, 1.0])

    y_axis = np.cross(z_axis, x_axis)
    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
    local_positions = [
        [0.0, 0.0, 0.0],
        [float(edge_length), 0.0, 0.0],
        np.dot(rotation_matrix.T, p1_to_p3).tolist(),
    ]
    return rotation_matrix, _as_mujoco_quat(rotation_matrix), local_positions


def _find_original_node(node_instances: dict[str, list[str]], instance_name: str) -> str | None:
    for original_name, instances in node_instances.items():
        if instance_name in instances:
            return original_name
    return None


def _disable_geom_contacts(geom: Any) -> None:
    geom.contype = 0
    geom.conaffinity = 1


def _add_planar_node_body(parent_body: Any, node_name: str, local_position: Vector, index: int) -> Any:
    node_body = parent_body.add_body(name=node_name, pos=local_position)
    node_body.add_site(name=node_name)
    node_geom = node_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[NODE_RADIUS],
        rgba=TRUSS_RGBA,
        mass=NODE_MASS,
    )
    _disable_geom_contacts(node_geom)

    if index == 1:
        node_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            name=f"{node_name}_x",
            pos=[0.0, 0.0, 0.0],
            axis=[1.0, 0.0, 0.0],
        )
    elif index == 2:
        node_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            name=f"{node_name}_x",
            pos=[0.0, 0.0, 0.0],
            axis=[1.0, 0.0, 0.0],
        )
        node_body.add_joint(
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            name=f"{node_name}_y",
            pos=[0.0, 0.0, 0.0],
            axis=[0.0, 1.0, 0.0],
        )

    return node_body


def _add_free_node_body(spec: mujoco.MjSpec, node_name: str, position: Vector) -> Any:
    node_body = spec.worldbody.add_body(name=node_name, pos=position)
    node_body.gravcomp = 1.0
    node_body.add_freejoint()
    node_body.add_site(name=node_name)
    node_geom = node_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[NODE_RADIUS],
        rgba=TRUSS_RGBA,
        mass=NODE_MASS,
    )
    _disable_geom_contacts(node_geom)
    return node_body


def _add_slide_node_body(spec: mujoco.MjSpec, node_name: str, position: Vector) -> Any:
    node_body = spec.worldbody.add_body(name=node_name, pos=position)
    node_body.add_site(name=node_name)
    node_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[NODE_RADIUS],
        rgba=TRUSS_RGBA,
    )
    node_body.add_joint(
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        name=f"{node_name}_x",
        pos=[0.0, 0.0, 0.0],
        axis=[1.0, 0.0, 0.0],
    )
    node_body.add_joint(
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        name=f"{node_name}_y",
        pos=[0.0, 0.0, 0.0],
        axis=[0.0, 1.0, 0.0],
    )
    node_body.add_joint(
        type=mujoco.mjtJoint.mjJNT_SLIDE,
        name=f"{node_name}_z",
        pos=[0.0, 0.0, 0.0],
        axis=[0.0, 0.0, 1.0],
    )
    return node_body


def _create_connector_balls(
    spec: mujoco.MjSpec,
    original_node_dict: dict[str, np.ndarray],
    node_instances: dict[str, list[str]],
    center: np.ndarray,
    scale: float,
) -> dict[str, Any]:
    connector_balls = {}
    for original_name, instances in node_instances.items():
        if len(instances) <= 1:
            continue

        original_position = original_node_dict[original_name]
        ball_position = (center + scale * (original_position - center)).tolist()
        ball = spec.worldbody.add_body(name=f"connector_ball_{original_name}", pos=ball_position)
        ball.add_site(name=f"ball_site_{original_name}", pos=[0.0, 0.0, 0.0])

        ball_geom = ball.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[CONNECTOR_RADIUS],
            rgba=TRUSS_RGBA,
            mass=CONNECTOR_MASS,
        )
        _disable_geom_contacts(ball_geom)

        for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
            ball.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)

        connector_balls[original_name] = ball

    return connector_balls


def _connect_node_to_ball(
    spec: mujoco.MjSpec,
    node_body: Any,
    instance_name: str,
    ball: Any,
    original_name: str,
    node_dict: NodeDict,
    rotation_matrix: np.ndarray,
) -> None:
    ball_position = np.array(ball.pos, dtype=float)
    node_position = np.array(node_dict[instance_name], dtype=float)
    rod_vector = np.dot(rotation_matrix.T, ball_position - node_position)

    rod = node_body.add_body(name=f"rod_{instance_name}", pos=[0.0, 0.0, 0.0])
    rod.add_site(name=f"tip_site_{instance_name}", pos=rod_vector.tolist())
    rod_geom = rod.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0.0, 0.0, 0.0, *rod_vector.tolist()],
        size=[ROD_RADIUS],
        rgba=TRUSS_RGBA,
        mass=ROD_MASS,
    )
    _disable_geom_contacts(rod_geom)
    rod.add_joint(
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0.0, 0.0, 1.0],
        damping=1.0,
        stiffness=5.0,
    )

    constraint = spec.add_equality(
        name=f"connect_{instance_name}",
        type=mujoco.mjtEq.mjEQ_CONNECT,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
    )
    constraint.name1 = f"tip_site_{instance_name}"
    constraint.name2 = f"ball_site_{original_name}"
    constraint.solref = [0.01, 1.0]
    constraint.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]


def create_triangle_bodies(
    spec: mujoco.MjSpec,
    original_node_dict: dict[str, np.ndarray],
    node_instances: dict[str, list[str]],
    node_dict: NodeDict,
    triangle_dict: TriangleDict,
    center: np.ndarray,
    scale: float,
    *,
    realistic: bool = False,
) -> None:
    """Create triangle bodies and optionally connect shared vertices with connector balls."""
    connector_balls = (
        _create_connector_balls(spec, original_node_dict, node_instances, center, scale)
        if realistic
        else {}
    )

    for triangle_name, triangle_nodes in triangle_dict.items():
        node_names = triangle_nodes[:3]
        positions = [np.array(node_dict[node_name], dtype=float) for node_name in node_names]
        rotation_matrix, quaternion, local_positions = _triangle_frame(*positions)

        triangle_body = spec.worldbody.add_body(name=f"tri_{triangle_name}", pos=positions[0].tolist())
        triangle_body.quat = quaternion
        triangle_body.add_freejoint()

        for index, instance_name in enumerate(node_names):
            node_body = _add_planar_node_body(
                triangle_body,
                node_name=instance_name,
                local_position=local_positions[index],
                index=index,
            )

            original_name = _find_original_node(node_instances, instance_name)
            if not original_name or original_name not in connector_balls:
                continue

            _connect_node_to_ball(
                spec,
                node_body=node_body,
                instance_name=instance_name,
                ball=connector_balls[original_name],
                original_name=original_name,
                node_dict=node_dict,
                rotation_matrix=rotation_matrix,
            )


def create_node_bodies(spec: mujoco.MjSpec, node_dict: NodeDict) -> None:
    """Create the abstract per-node slide-joint model used by realistic=False."""
    for node_name, position in node_dict.items():
        _add_slide_node_body(spec, node_name=node_name, position=position)


def _edge_key(from_node_name: str, to_node_name: str) -> EdgeKey:
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
    key = _edge_key(from_node_name, to_node_name)
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


def add_orientation_constraints(
    spec: mujoco.MjSpec,
    node_dict: NodeDict | TriangleDict,
    maybe_node_dict: NodeDict | None = None,
) -> None:
    """Add soft weld constraints between each node and its initial world pose."""
    if maybe_node_dict is not None:
        node_dict = maybe_node_dict

    for node in node_dict:
        weld = spec.add_equality(
            name=f"weld_world_{node}",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
        )
        weld.name1 = node
        weld.solref = [0.2, 5.0]
        weld.solimp = [0.2, 0.3, 0.001, 0.5, 2.0]
        weld.data[10] = 6000.0


def add_perimeter_constraint(
    spec: mujoco.MjSpec,
    triangle_dict: TriangleDict,
    edge_tendons: EdgeTendonMap | None = None,
) -> None:
    for index, triangle_nodes in enumerate(triangle_dict.values()):
        nodes = triangle_nodes[:3]
        passive_node = triangle_nodes[3]
        passive_index = nodes.index(passive_node)
        active_nodes = [node for node in nodes if node != passive_node]

        edge_from = nodes[(passive_index + 1) % 3]
        edge_to = nodes[(passive_index + 2) % 3]
        if edge_tendons is None:
            passive_edge_tendon = f"tendon_{edge_from}_{edge_to}"
        else:
            passive_edge_tendon = edge_tendons[_edge_key(edge_from, edge_to)]

        tendon_name = f"Perimeter_Constraint_{index}"
        tendon = spec.add_tendon(name=tendon_name)
        tendon.wrap_site(active_nodes[0])
        tendon.wrap_site(passive_node)
        tendon.wrap_site(active_nodes[1])

        constraint = spec.add_equality(name=tendon_name, type=mujoco.mjtEq.mjEQ_TENDON)
        constraint.name1 = passive_edge_tendon
        constraint.name2 = tendon_name
        constraint.data[:5] = [0.0, -1.0, 0.0, 0.0, 0.0]
        constraint.solref = [0.02, 1.0]
        constraint.solimp[:3] = [0.9, 0.95, 0.001]


def clone_shared_nodes(
    node_dict: NodeDict,
    triangle_dict: TriangleDict,
    clone_offset: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], np.ndarray, float]:
    """Clone shared nodes so each triangle has independent planar node bodies.

    The input dictionaries are mutated in place. The returned original-node mapping
    and instance map are used when realistic connector balls are enabled.
    """
    original_node_dict = {name: np.array(position, dtype=float) for name, position in node_dict.items()}
    positions = np.array(list(original_node_dict.values()))
    center = np.mean(positions, axis=0)
    scale = 1.0 + clone_offset

    node_instances = {name: [] for name in original_node_dict}
    node_owner = {}
    new_node_dict = {}
    min_z = float("inf")

    for triangle_name, triangle_nodes in list(triangle_dict.items()):
        new_nodes = list(triangle_nodes)
        triangle_positions = [original_node_dict[node] for node in triangle_nodes[:3]]
        triangle_centroid = np.mean(triangle_positions, axis=0)

        for index, node in enumerate(triangle_nodes[:3]):
            if node not in node_owner:
                node_owner[node] = triangle_name
                instance_name = node
            else:
                instance_name = f"{node}_tri_{triangle_name}"

            node_instances[node].append(instance_name)

            original_position = original_node_dict[node]
            new_position = center + scale * (triangle_centroid - center) + (
                original_position - triangle_centroid
            )
            new_node_dict[instance_name] = new_position.tolist()
            min_z = min(min_z, float(new_position[2]))

            new_nodes[index] = instance_name
            if triangle_nodes[3] == node:
                new_nodes[3] = instance_name

        triangle_dict[triangle_name] = new_nodes

    for original_position in original_node_dict.values():
        ball_z = center[2] + scale * (original_position[2] - center[2])
        min_z = min(min_z, float(ball_z))

    z_offset = max(0.0, 0.11 - min_z)
    if z_offset > 0.0:
        center[2] += z_offset
        for position in original_node_dict.values():
            position[2] += z_offset
        for position in new_node_dict.values():
            position[2] += z_offset

    node_dict.clear()
    node_dict.update(new_node_dict)
    return original_node_dict, node_instances, center, scale


def build_abstract_triangle(spec: mujoco.MjSpec, triangle_dict: TriangleDict) -> None:
    for triangle_nodes in triangle_dict.values():
        nodes = triangle_nodes[:3]
        passive_node = triangle_nodes[3]

        add_tendon(spec, from_node_name=nodes[0], to_node_name=nodes[1])
        add_tendon(spec, from_node_name=nodes[1], to_node_name=nodes[2])
        add_tendon(spec, from_node_name=nodes[2], to_node_name=nodes[0])

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            if from_node == passive_node or to_node == passive_node:
                add_actuator(
                    spec,
                    tendon_name=f"tendon_{from_node}_{to_node}",
                    kp=5000.0,
                    dampratio=1.0,
                )

    add_perimeter_constraint(spec, triangle_dict)


def build_realistic_triangle(spec: mujoco.MjSpec, triangle_dict: TriangleDict) -> None:
    edge_tendons: EdgeTendonMap = {}
    actuated_tendons: set[str] = set()

    for triangle_nodes in triangle_dict.values():
        nodes = triangle_nodes[:3]
        passive_node = triangle_nodes[3]

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            add_edge_tendon(spec, edge_tendons, from_node, to_node)

        for index, from_node in enumerate(nodes):
            to_node = nodes[(index + 1) % 3]
            if from_node == passive_node or to_node == passive_node:
                tendon_name = edge_tendons[_edge_key(from_node, to_node)]
                if tendon_name in actuated_tendons:
                    continue
                add_realistic_actuator(
                    spec,
                    tendon_name=tendon_name,
                    kp=1000.0,
                    dampratio=1.0,
                )
                actuated_tendons.add(tendon_name)

    add_perimeter_constraint(spec, triangle_dict, edge_tendons)


def build_triangle(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    triangle_dict: TriangleDict,
    *,
    realistic: bool = False,
) -> None:
    if realistic:
        original_node_dict, node_instances, center, scale = clone_shared_nodes(node_dict, triangle_dict)
        create_triangle_bodies(
            spec,
            original_node_dict,
            node_instances,
            node_dict,
            triangle_dict,
            center,
            scale,
            realistic=True,
        )
    else:
        create_node_bodies(spec, node_dict)
        build_abstract_triangle(spec, triangle_dict)
        return

    build_realistic_triangle(spec, triangle_dict)


def get_perimeter(node_dict: NodeDict, triangle_dict: TriangleDict) -> dict[str, float]:
    perimeters = {}
    for name, nodes in triangle_dict.items():
        triangle_nodes = nodes[:3]
        positions = [np.array(node_dict[node], dtype=float) for node in triangle_nodes]
        perimeters[name] = float(
            sum(np.linalg.norm(positions[(index + 1) % 3] - positions[index]) for index in range(3))
        )
    return perimeters


def view(spec: mujoco.MjSpec) -> None:
    """Compile and view the MuJoCo spec."""
    model = spec.compile()
    if hasattr(model, "model") and hasattr(model, "data"):
        mj_model = model.model
        data = model.data
    elif isinstance(model, mujoco.MjModel):
        mj_model = model
        data = mujoco.MjData(mj_model)
    else:
        raise TypeError(
            "view() expects a mujoco.MjModel or an object with 'model' and 'data' attributes."
        )

    initialize_actuator_lengths(mj_model, data)
    with mujoco.viewer.launch_passive(mj_model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            if data.time == 0.0:
                initialize_actuator_lengths(mj_model, data)
            mujoco.mj_step(mj_model, data)
            viewer.sync()
            time.sleep(max(mj_model.opt.timestep, 0.001))


def build_world() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(name="top", pos=[0.0, 0.0, 1.0])
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10.0, 10.0, 0.1],
        rgba=TRUSS_RGBA,
    )
    return spec


def save_xml(spec: mujoco.MjSpec, filename: str | Path) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path


def get_octahedron_definition() -> tuple[NodeDict, TriangleDict]:
    node_dict = {
        "node_1": [0.0, 0.0, 0.1],
        "node_2": [1.0, 0.0, 0.1],
        "node_3": [0.5, 0.8660, 0.1],
        "node_4": [0.5, -0.2887, 0.9165],
        "node_5": [0.0, 0.5774, 0.9165],
        "node_6": [1.0, 0.5774, 0.9165],
    }
    triangle_dict = {
        "triangle_1": ["node_1", "node_2", "node_4", "node_1"],
        "triangle_2": ["node_1", "node_5", "node_3", "node_1"],
        "triangle_3": ["node_3", "node_6", "node_2", "node_6"],
        "triangle_4": ["node_4", "node_6", "node_5", "node_6"],
    }
    return node_dict, triangle_dict


def get_mujoco_spec(*args: Any, realistic: bool = False, **kwargs: Any) -> mujoco.MjSpec:
    if kwargs:
        unexpected = ", ".join(kwargs)
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    spec = build_world()
    if len(args) == 2:
        node_dict, triangle_dict = args
    elif len(args) == 1 and isinstance(args[0], str):
        if args[0] != "octahedron":
            raise ValueError(f"Unknown structure type: {args[0]}")
        node_dict, triangle_dict = get_octahedron_definition()
    else:
        raise ValueError(
            "get_mujoco_spec() takes node_dict and triangle_dict, or one structure_type string."
        )

    build_triangle(spec, node_dict, triangle_dict, realistic=realistic)
    return spec


def main() -> None:
    node_dict, triangle_dict = get_octahedron_definition()
    spec = build_world()
    build_triangle(spec, node_dict, triangle_dict, realistic=False)

    for name, perimeter in get_perimeter(node_dict, triangle_dict).items():
        print(f"{name}: perimeter = {perimeter:.4f}")

    view(spec)


# Backwards-compatible aliases for existing local scripts.
connect_triangeles = create_triangle_bodies
add_perimeter_constrait = add_perimeter_constraint


if __name__ == "__main__":
    main()
