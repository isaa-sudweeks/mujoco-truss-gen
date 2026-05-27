from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.constants import (
    BOX_SIZE,
    GEOM_CONTACT_AFFINITY,
    GEOM_CONTACT_TYPE,
    MODEL_INTEGRATOR,
    NODE_MASS,
    NODE_MATERIAL,
    NODE_RADIUS,
    NODE_RGBA,
    ROD_MATERIAL,
    ROD_RGBA,
    TrussPhysicalParameters,
)
from mujoco_truss_gen.mujoco_model.controllers import angle_bisector_actuator_name
from mujoco_truss_gen.mujoco_model.geometry import triangle_frame
from mujoco_truss_gen.mujoco_model.model_types import NodeDict, TriangleDict, Vector


def find_original_node(node_instances: dict[str, list[str]], instance_name: str) -> str | None:
    for original_name, instances in node_instances.items():
        if instance_name in instances:
            return original_name
    return None


def disable_geom_contacts(geom: Any) -> None:
    geom.contype = GEOM_CONTACT_TYPE
    geom.conaffinity = GEOM_CONTACT_AFFINITY


def add_planar_node_body(
    parent_body: Any,
    node_name: str,
    local_position: Vector,
    index: int,
    connector_direction: Vector | None = None,
    mass: float = NODE_MASS,
    box_size: list[float] | tuple[float, float, float] = BOX_SIZE,
) -> Any:
    node_body = parent_body.add_body(name=node_name, pos=local_position)
    node_body.add_site(name=node_name)
    node_geom = node_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=box_size,
        rgba=NODE_RGBA,
        material=NODE_MATERIAL,
        mass=mass,
    )
    if connector_direction is not None:
        node_geom.quat = _face_normal_quat(connector_direction)
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

    node_body.add_joint(
        type=mujoco.mjtJoint.mjJNT_HINGE,
        name=f"{node_name}_z_hinge",
        axis=[0.0, 0.0, 1.0],
        pos=[0.0, 0.0, 0.0],
    )

    return node_body


def _face_normal_quat(connector_direction: Vector) -> list[float]:
    direction = np.array(connector_direction, dtype=float)
    planar_direction = direction[:2]
    norm = float(np.linalg.norm(planar_direction))
    if norm < 1e-10:
        return [1.0, 0.0, 0.0, 0.0]

    x_axis = planar_direction / norm
    angle = float(np.arctan2(x_axis[1], x_axis[0]))
    half_angle = 0.5 * angle
    return [float(np.cos(half_angle)), 0.0, 0.0, float(np.sin(half_angle))]


def add_free_node_body(
    spec: mujoco.MjSpec,
    node_name: str,
    position: Vector,
    physical_params: TrussPhysicalParameters | None = None,
) -> Any:
    params = physical_params or TrussPhysicalParameters()
    node_body = spec.worldbody.add_body(name=node_name, pos=position)
    node_body.add_freejoint()
    node_body.add_site(name=node_name)
    node_geom = node_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[params.node_radius],
        rgba=NODE_RGBA,
        material=NODE_MATERIAL,
        mass=params.node_mass,
    )
    disable_geom_contacts(node_geom)
    return node_body


def add_slide_node_body(
    spec: mujoco.MjSpec,
    node_name: str,
    position: Vector,
    mass: float = NODE_MASS,
    node_radius: float = NODE_RADIUS,
) -> Any:
    node_body = spec.worldbody.add_body(name=node_name, pos=position)
    node_body.add_site(name=node_name)
    node_body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[node_radius],
        rgba=NODE_RGBA,
        material=NODE_MATERIAL,
        mass=mass,
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


def _node_mass(
    node_name: str,
    triangle_nodes: list[str],
    physical_params: TrussPhysicalParameters,
) -> float:
    if node_name == triangle_nodes[3]:
        return physical_params.passive_node_mass
    return physical_params.active_node_mass


def create_connector_balls(
    spec: mujoco.MjSpec,
    original_node_dict: dict[str, np.ndarray],
    node_instances: dict[str, list[str]],
    center: np.ndarray,
    scale: float,
    physical_params: TrussPhysicalParameters | None = None,
) -> dict[str, Any]:
    params = physical_params or TrussPhysicalParameters()
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
            size=[params.connector_radius],
            rgba=NODE_RGBA,
            material=NODE_MATERIAL,
            mass=params.connector_mass,
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
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    params = physical_params or TrussPhysicalParameters()
    ball_position = np.array(ball.pos, dtype=float)
    node_position = np.array(node_dict[instance_name], dtype=float)
    rod_vector = np.dot(rotation_matrix.T, ball_position - node_position)

    rod = node_body.add_body(name=f"rod_{instance_name}", pos=[0.0, 0.0, 0.0])
    rod.add_site(name=f"tip_site_{instance_name}", pos=rod_vector.tolist())
    rod_geom = rod.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        fromto=[0.0, 0.0, 0.0, *rod_vector.tolist()],
        size=[params.rod_radius],
        rgba=ROD_RGBA,
        material=ROD_MATERIAL,
        mass=params.rod_mass,
    )
    disable_geom_contacts(rod_geom)

    actuator = spec.add_actuator(
        name=angle_bisector_actuator_name(instance_name),
        trntype=mujoco.mjtTrn.mjTRN_JOINT,
        target=f"{instance_name}_z_hinge",
        ctrllimited=True,
        ctrlrange=params.hinge_ctrl_range,
        forcelimited=True,
        forcerange=params.hinge_force_range,
    )
    actuator.set_to_position(kp=params.hinge_position_kp)

    constraint = spec.add_equality(
        name=f"connect_{instance_name}",
        type=mujoco.mjtEq.mjEQ_CONNECT,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
    )
    constraint.name1 = f"tip_site_{instance_name}"
    constraint.name2 = f"ball_site_{original_name}"
    constraint.solref = params.connect_constraint_solref
    constraint.solimp = params.connect_constraint_solimp


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
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    """Create triangle bodies and optionally connect shared vertices with connector balls."""
    params = physical_params or TrussPhysicalParameters()
    connector_balls = (
        create_connector_balls(
            spec,
            original_node_dict,
            node_instances,
            center,
            scale,
            physical_params=params,
        )
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
        triangle_body.gravcomp = params.triangle_body_gravcomp
        triangle_body.explicitinertial = True
        triangle_body.mass = params.triangle_body_mass
        triangle_body.inertia = [params.triangle_body_mass] * 3
        triangle_body.add_freejoint()

        for index, instance_name in enumerate(node_names):
            original_name = find_original_node(node_instances, instance_name)
            connector_direction = None
            if original_name and original_name in connector_balls:
                ball_position = np.array(connector_balls[original_name].pos, dtype=float)
                node_position = np.array(node_dict[instance_name], dtype=float)
                connector_direction = np.dot(rotation_matrix.T, ball_position - node_position)

            node_body = add_planar_node_body(
                triangle_body,
                node_name=instance_name,
                local_position=local_positions[index],
                index=index,
                connector_direction=connector_direction,
                mass=_node_mass(instance_name, triangle_nodes, params),
                box_size=params.box_size,
            )

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
                physical_params=params,
            )


def create_node_bodies(
    spec: mujoco.MjSpec,
    node_dict: NodeDict,
    node_masses: dict[str, float] | None = None,
    physical_params: TrussPhysicalParameters | None = None,
) -> None:
    """Create the abstract per-node slide-joint model used by realistic=False."""
    params = physical_params or TrussPhysicalParameters()
    for node_name, position in node_dict.items():
        add_slide_node_body(
            spec,
            node_name=node_name,
            position=position,
            mass=node_masses.get(node_name, params.active_node_mass)
            if node_masses
            else params.node_mass,
            node_radius=params.node_radius,
        )


def build_world() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_string(
        f"""
<mujoco>
  <option integrator="{MODEL_INTEGRATOR}"/>
  <visual>
    <global azimuth="120" elevation="-25"/>
    <headlight ambient="0.24 0.24 0.24"
               diffuse="0.56 0.56 0.56"
               specular="0.16 0.16 0.16"/>
    <rgba haze="0.76 0.82 0.88 1"/>
  </visual>
  <asset>
    <texture name="skybox"
             type="skybox"
             builtin="gradient"
             rgb1="0.54 0.64 0.76"
             rgb2="0.84 0.88 0.93"
             width="512"
             height="3072"/>
    <texture name="ground_checker"
             type="2d"
             builtin="checker"
             rgb1="0.68 0.70 0.72"
             rgb2="0.44 0.47 0.51"
             mark="edge"
             markrgb="0.56 0.58 0.60"
             width="512"
             height="512"/>
    <material name="ground_grid"
              texture="ground_checker"
              texrepeat="12 12"
              texuniform="true"
              reflectance="0.12"/>
    <material name="blue_firehose"
              rgba="0.0 0.1804 0.3647 1"
              specular="0.08"
              shininess="0.12"
              reflectance="0.01"/>
    <material name="connector_steel"
              rgba="0.62 0.64 0.66 1"
              specular="0.75"
              shininess="0.65"
              reflectance="0.22"/>
    <material name="node_black"
              rgba="0.18 0.18 0.18 1"
              specular="0.25"
              shininess="0.35"
              reflectance="0.08"/>
  </asset>
  <worldbody>
    <light name="key"
           pos="0 -3.5 5"
           dir="0 0.7 -1"
           directional="true"
           diffuse="0.56 0.56 0.56"
           ambient="0.13 0.13 0.13"
           specular="0.16 0.16 0.16"/>
    <light name="fill"
           pos="3.5 3.5 4"
           dir="-0.6 -0.6 -1"
           directional="true"
           diffuse="0.18 0.20 0.23"
           ambient="0.05 0.05 0.05"
           specular="0.05 0.05 0.05"/>
    <geom name="ground"
          type="plane"
          size="12 12 0.1"
          material="ground_grid"/>
  </worldbody>
</mujoco>
"""
    )
    return spec
