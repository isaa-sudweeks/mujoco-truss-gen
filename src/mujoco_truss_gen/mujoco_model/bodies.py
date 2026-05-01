from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.constants import (
    CONNECTOR_MASS,
    CONNECTOR_RADIUS,
    NODE_MASS,
    NODE_RADIUS,
    ROD_MASS,
    ROD_RADIUS,
    TRUSS_RGBA,
)
from mujoco_truss_gen.mujoco_model.geometry import triangle_frame
from mujoco_truss_gen.mujoco_model.model_types import NodeDict, TriangleDict, Vector


def find_original_node(node_instances: dict[str, list[str]], instance_name: str) -> str | None:
    for original_name, instances in node_instances.items():
        if instance_name in instances:
            return original_name
    return None


def disable_geom_contacts(geom: Any) -> None:
    geom.contype = 0
    geom.conaffinity = 1


def add_planar_node_body(
    parent_body: Any, node_name: str, local_position: Vector, index: int
) -> Any:
    node_body = parent_body.add_body(name=node_name, pos=local_position)
    node_body.add_site(name=node_name)
    node_geom = node_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[NODE_RADIUS],
        rgba=TRUSS_RGBA,
        mass=NODE_MASS,
    )
    disable_geom_contacts(node_geom)

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


def add_free_node_body(spec: mujoco.MjSpec, node_name: str, position: Vector) -> Any:
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
    disable_geom_contacts(node_geom)
    return node_body


def add_slide_node_body(spec: mujoco.MjSpec, node_name: str, position: Vector) -> Any:
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


def create_connector_balls(
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
        disable_geom_contacts(ball_geom)

        for axis in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]):
            ball.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)

        connector_balls[original_name] = ball

    return connector_balls


def connect_node_to_ball(
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
    disable_geom_contacts(rod_geom)
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
        create_connector_balls(spec, original_node_dict, node_instances, center, scale)
        if realistic
        else {}
    )

    for triangle_name, triangle_nodes in triangle_dict.items():
        node_names = triangle_nodes[:3]
        positions = [np.array(node_dict[node_name], dtype=float) for node_name in node_names]
        rotation_matrix, quaternion, local_positions = triangle_frame(*positions)

        triangle_body = spec.worldbody.add_body(
            name=f"tri_{triangle_name}", pos=positions[0].tolist()
        )
        triangle_body.quat = quaternion
        triangle_body.add_freejoint()

        for index, instance_name in enumerate(node_names):
            node_body = add_planar_node_body(
                triangle_body,
                node_name=instance_name,
                local_position=local_positions[index],
                index=index,
            )

            original_name = find_original_node(node_instances, instance_name)
            if not original_name or original_name not in connector_balls:
                continue

            connect_node_to_ball(
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
        add_slide_node_body(spec, node_name=node_name, position=position)


def build_world() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(name="top", pos=[0.0, 0.0, 1.0])
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10.0, 10.0, 0.1],
        rgba=TRUSS_RGBA,
    )
    return spec
