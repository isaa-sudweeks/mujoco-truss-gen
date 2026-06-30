from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import jax.numpy as jnp
import numpy as np
from mujoco import mjx  # type: ignore[import-untyped]

from mujoco_truss_gen.mujoco_model.controllers import AngleBisectorTarget

_VECTOR_EPSILON = 1e-10


class MjxAngleBisectorController:
    """Vectorized MJX implementation of the angle-bisector controller."""

    def __init__(self, targets: Sequence[AngleBisectorTarget]):
        self.target_count = len(targets)
        self.enabled = bool(targets)
        self.actuator_ids: frozenset[int] = frozenset()
        if not targets:
            return

        self._node_site_ids = jnp.asarray(
            [target.node_site_id for target in targets], dtype=jnp.int32
        )
        self._parent_body_ids = jnp.asarray(
            [target.parent_body_id for target in targets], dtype=jnp.int32
        )
        self._actuator_ids = jnp.asarray(
            [target.actuator_id for target in targets], dtype=jnp.int32
        )
        self._initial_rod_vectors = jnp.asarray(
            np.stack([target.initial_rod_vector for target in targets])
        )
        self._hinge_axes = jnp.asarray(np.stack([target.hinge_axis for target in targets]))

        neighbor_ids = np.empty((self.target_count, 2), dtype=np.int32)
        neighbor_counts = np.empty(self.target_count, dtype=np.int32)
        for index, target in enumerate(targets):
            neighbor_counts[index] = len(target.neighbor_site_ids)
            neighbor_ids[index] = target.node_site_id
            neighbor_ids[index, : len(target.neighbor_site_ids)] = target.neighbor_site_ids
        self._neighbor_site_ids = jnp.asarray(neighbor_ids)
        self._neighbor_counts = jnp.asarray(neighbor_counts)

        max_candidates = max(1, *(len(target.neighbor_candidate_site_ids) for target in targets))
        candidate_ids = np.empty((self.target_count, max_candidates), dtype=np.int32)
        candidate_mask = np.zeros((self.target_count, max_candidates), dtype=bool)
        for index, target in enumerate(targets):
            candidate_ids[index] = target.node_site_id
            count = len(target.neighbor_candidate_site_ids)
            candidate_ids[index, :count] = target.neighbor_candidate_site_ids
            candidate_mask[index, :count] = True
        self._candidate_site_ids = jnp.asarray(candidate_ids)
        self._candidate_mask = jnp.asarray(candidate_mask)
        self._use_nearest_candidates = jnp.asarray(
            [
                len(target.neighbor_site_ids) == 2 and len(target.neighbor_candidate_site_ids) >= 2
                for target in targets
            ]
        )

        angular_target_indices = [
            index
            for index, target in enumerate(targets)
            if target.angular_actuator_id is not None and target.angular_hinge_axis is not None
        ]
        self._angular_target_indices = jnp.asarray(angular_target_indices, dtype=jnp.int32)
        self._angular_actuator_ids = jnp.asarray(
            [targets[index].angular_actuator_id for index in angular_target_indices],
            dtype=jnp.int32,
        )
        self._angular_hinge_axes = jnp.asarray(
            np.stack(
                [
                    cast(np.ndarray, targets[index].angular_hinge_axis)
                    for index in angular_target_indices
                ]
            )
            if angular_target_indices
            else np.empty((0, 3))
        )

        roll_target_indices = [
            index
            for index, target in enumerate(targets)
            if target.roll_actuator_id is not None and target.roll_hinge_axis is not None
        ]
        self._roll_target_indices = jnp.asarray(roll_target_indices, dtype=jnp.int32)
        self._roll_actuator_ids = jnp.asarray(
            [targets[index].roll_actuator_id for index in roll_target_indices],
            dtype=jnp.int32,
        )
        self._roll_hinge_axes = jnp.asarray(
            np.stack(
                [cast(np.ndarray, targets[index].roll_hinge_axis) for index in roll_target_indices]
            )
            if roll_target_indices
            else np.empty((0, 3))
        )

        owned_ids = {
            actuator_id
            for target in targets
            for actuator_id in (
                target.actuator_id,
                target.angular_actuator_id,
                target.roll_actuator_id,
            )
            if actuator_id is not None
        }
        self.actuator_ids = frozenset(owned_ids)

    def initialize(self, data: mjx.Data) -> mjx.Data:
        """Set controls without using pre-reset controls as continuity references."""

        return self._apply(data, use_previous=False)

    def update(self, data: mjx.Data) -> mjx.Data:
        """Update controls while selecting angles nearest to the previous controls."""

        return self._apply(data, use_previous=True)

    def _apply(self, data: mjx.Data, *, use_previous: bool) -> mjx.Data:
        if not self.enabled:
            return data

        ctrl = data.ctrl
        node_positions = data.site_xpos[self._node_site_ids]
        fixed_neighbor_positions = data.site_xpos[self._neighbor_site_ids]
        candidate_positions = data.site_xpos[self._candidate_site_ids]
        candidate_distances = jnp.linalg.norm(
            candidate_positions - node_positions[:, None, :], axis=-1
        )
        candidate_distances = jnp.where(self._candidate_mask, candidate_distances, jnp.inf)
        nearest_indices = jnp.argsort(candidate_distances, axis=1)[:, :2]
        nearest_positions = jnp.take_along_axis(
            candidate_positions, nearest_indices[:, :, None], axis=1
        )
        neighbor_positions = jnp.where(
            self._use_nearest_candidates[:, None, None],
            nearest_positions,
            fixed_neighbor_positions,
        )

        one_target, one_valid = _unit_vector(node_positions - neighbor_positions[:, 0])
        direction_a, direction_a_valid = _unit_vector(neighbor_positions[:, 0] - node_positions)
        direction_b, direction_b_valid = _unit_vector(neighbor_positions[:, 1] - node_positions)
        bisector, bisector_valid = _unit_vector(direction_a + direction_b)
        plane_normal, plane_normal_valid = _unit_vector(jnp.cross(direction_a, direction_b))
        has_one_neighbor = self._neighbor_counts == 1
        target_world = jnp.where(has_one_neighbor[:, None], one_target, -bisector)
        target_valid = jnp.where(
            has_one_neighbor,
            one_valid,
            direction_a_valid & direction_b_valid & bisector_valid,
        )
        plane_normal_valid = (
            ~has_one_neighbor & direction_a_valid & direction_b_valid & plane_normal_valid
        )

        parent_xmat = data.xmat[self._parent_body_ids].reshape((-1, 3, 3))
        target_parent = jnp.einsum("tji,tj->ti", parent_xmat, target_world)
        angles, angle_valid = _signed_angle_about_axis(
            self._initial_rod_vectors,
            target_parent,
            self._hinge_axes,
        )
        angle_valid &= target_valid
        if use_previous:
            angles = _nearest_equivalent_angles(angles, ctrl[self._actuator_ids])
        angles = jnp.where(angle_valid, angles, ctrl[self._actuator_ids])
        ctrl = ctrl.at[self._actuator_ids].set(angles)

        if self._angular_target_indices.size == 0:
            return data.replace(ctrl=ctrl)

        angular_indices = self._angular_target_indices
        angular_hinge_axes = self._angular_hinge_axes
        angular_yaw_angles = angles[angular_indices]
        angular_hinge_axes_yawed, yawed_axis_valid = _rotate_about_axis(
            angular_hinge_axes,
            self._hinge_axes[angular_indices],
            angular_yaw_angles,
        )
        yawed_rods, yawed_rod_valid = _rotate_about_axis(
            self._initial_rod_vectors[angular_indices],
            self._hinge_axes[angular_indices],
            angular_yaw_angles,
        )
        angular_angles, angular_valid = _signed_angle_about_axis(
            yawed_rods,
            target_parent[angular_indices],
            angular_hinge_axes_yawed,
        )
        angular_valid &= angle_valid[angular_indices] & yawed_axis_valid & yawed_rod_valid
        if use_previous:
            angular_angles = _nearest_equivalent_angles(
                angular_angles, ctrl[self._angular_actuator_ids]
            )
        angular_angles = jnp.where(
            angular_valid,
            angular_angles,
            ctrl[self._angular_actuator_ids],
        )
        ctrl = ctrl.at[self._angular_actuator_ids].set(angular_angles)

        if self._roll_target_indices.size == 0:
            return data.replace(ctrl=ctrl)

        angular_angle_by_target = jnp.zeros(self.target_count, dtype=angular_angles.dtype)
        angular_angle_by_target = angular_angle_by_target.at[angular_indices].set(angular_angles)
        angular_axis_by_target = jnp.zeros((self.target_count, 3), dtype=target_parent.dtype)
        angular_axis_by_target = angular_axis_by_target.at[angular_indices].set(
            angular_hinge_axes_yawed
        )
        angular_valid_by_target = jnp.zeros(self.target_count, dtype=jnp.bool_)
        angular_valid_by_target = angular_valid_by_target.at[angular_indices].set(angular_valid)

        roll_indices = self._roll_target_indices
        roll_yaw_angles = angles[roll_indices]
        rolled_axes, rolled_axis_yaw_valid = _rotate_about_axis(
            self._roll_hinge_axes,
            self._hinge_axes[roll_indices],
            roll_yaw_angles,
        )
        rolled_axes, rolled_axis_angular_valid = _rotate_about_axis(
            rolled_axes,
            angular_axis_by_target[roll_indices],
            angular_angle_by_target[roll_indices],
        )
        rolled_normals, rolled_normal_yaw_valid = _rotate_about_axis(
            self._hinge_axes[roll_indices],
            self._hinge_axes[roll_indices],
            roll_yaw_angles,
        )
        rolled_normals, rolled_normal_angular_valid = _rotate_about_axis(
            rolled_normals,
            angular_axis_by_target[roll_indices],
            angular_angle_by_target[roll_indices],
        )
        target_normals_parent = jnp.einsum(
            "tji,tj->ti",
            parent_xmat[roll_indices],
            plane_normal[roll_indices],
        )
        base_roll_valid = (
            angle_valid[roll_indices]
            & angular_valid_by_target[roll_indices]
            & plane_normal_valid[roll_indices]
            & rolled_axis_yaw_valid
            & rolled_axis_angular_valid
            & rolled_normal_yaw_valid
            & rolled_normal_angular_valid
        )

        if use_previous:
            positive_angles, positive_valid = _signed_angle_about_axis(
                rolled_normals, target_normals_parent, rolled_axes
            )
            negative_angles, negative_valid = _signed_angle_about_axis(
                rolled_normals, -target_normals_parent, rolled_axes
            )
            previous_roll = ctrl[self._roll_actuator_ids]
            positive_angles = _nearest_equivalent_angles(positive_angles, previous_roll)
            negative_angles = _nearest_equivalent_angles(negative_angles, previous_roll)
            use_positive = jnp.abs(positive_angles - previous_roll) <= jnp.abs(
                negative_angles - previous_roll
            )
            roll_angles = jnp.where(use_positive, positive_angles, negative_angles)
            roll_valid = base_roll_valid & jnp.where(use_positive, positive_valid, negative_valid)
        else:
            target_normals_parent = jnp.where(
                (jnp.sum(rolled_normals * target_normals_parent, axis=-1) < 0.0)[:, None],
                -target_normals_parent,
                target_normals_parent,
            )
            roll_angles, roll_angle_valid = _signed_angle_about_axis(
                rolled_normals, target_normals_parent, rolled_axes
            )
            roll_valid = base_roll_valid & roll_angle_valid

        roll_angles = jnp.where(
            roll_valid,
            roll_angles,
            ctrl[self._roll_actuator_ids],
        )
        ctrl = ctrl.at[self._roll_actuator_ids].set(roll_angles)
        return data.replace(ctrl=ctrl)


def _unit_vector(vector: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    norm = jnp.linalg.norm(vector, axis=-1)
    valid = norm >= _VECTOR_EPSILON
    safe_norm = jnp.where(valid, norm, 1.0)
    return vector / safe_norm[..., None], valid


def _signed_angle_about_axis(
    from_vector: jnp.ndarray,
    to_vector: jnp.ndarray,
    axis: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    axis_unit, axis_valid = _unit_vector(axis)
    from_projected = from_vector - axis_unit * jnp.sum(from_vector * axis_unit, axis=-1)[..., None]
    to_projected = to_vector - axis_unit * jnp.sum(to_vector * axis_unit, axis=-1)[..., None]
    from_unit, from_valid = _unit_vector(from_projected)
    to_unit, to_valid = _unit_vector(to_projected)
    signed_cross = jnp.sum(axis_unit * jnp.cross(from_unit, to_unit), axis=-1)
    dot = jnp.clip(jnp.sum(from_unit * to_unit, axis=-1), -1.0, 1.0)
    return jnp.arctan2(signed_cross, dot), axis_valid & from_valid & to_valid


def _nearest_equivalent_angles(angle: jnp.ndarray, reference: jnp.ndarray) -> jnp.ndarray:
    delta = angle - reference
    return reference + jnp.arctan2(jnp.sin(delta), jnp.cos(delta))


def _rotate_about_axis(
    vector: jnp.ndarray,
    axis: jnp.ndarray,
    angle: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    axis_unit, axis_valid = _unit_vector(axis)
    cosine = jnp.cos(angle)[..., None]
    sine = jnp.sin(angle)[..., None]
    rotated = (
        vector * cosine
        + jnp.cross(axis_unit, vector) * sine
        + axis_unit * jnp.sum(axis_unit * vector, axis=-1)[..., None] * (1.0 - cosine)
    )
    return jnp.where(axis_valid[..., None], rotated, vector), axis_valid
