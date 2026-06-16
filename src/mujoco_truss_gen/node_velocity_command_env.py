from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from mujoco_truss_gen.base_env import TrussEnvConfig
from mujoco_truss_gen.mujoco_model.controllers import NodeVelocityController
from mujoco_truss_gen.mujoco_model.model import ModelSource
from mujoco_truss_gen.relative_observation_env import MujocoRelativeObsEnv


class MujocoNodeVelocityCommandEnv(MujocoRelativeObsEnv):
    """Relative-observation environment with routed-tube node velocity commands."""

    def __init__(
        self,
        model_source: TrussEnvConfig | ModelSource,
        render_mode: str | None = None,
        rank: int = 0,
        **config_overrides: Any,
    ):
        super().__init__(model_source, render_mode=render_mode, rank=rank, **config_overrides)

    def _on_model_changed(self) -> None:
        self.node_velocity_controller = NodeVelocityController(
            self.mj_model.model,
            self.mj_model.xml,
            self.mj_model.node_names,
            self.mj_model.site_to_node,
            self.mj_model.external_actuator_ids,
        )
        if not self.node_velocity_controller.enabled:
            raise ValueError(
                "MujocoNodeVelocityCommandEnv requires a model with control graph metadata "
                "or a routed continuous-tube model with route tendons and edge actuators."
            )
        super()._on_model_changed()

    def _define_action_space(self) -> None:
        self.action_space = spaces.Box(
            low=-self.config.speed,
            high=self.config.speed,
            shape=(len(self.node_velocity_controller.node_names),),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        node_positions = self.mj_model.get_control_node_position_matrix()
        node_velocities = self.mj_model.get_control_node_linear_velocity_matrix()
        if node_positions.size == 0 or node_velocities.size == 0:
            node_positions = self.mj_model.get_node_position_matrix()
            node_velocities = self.mj_model.get_node_linear_velocity_matrix()

        com = np.mean(node_positions, axis=0) if node_positions.size else np.zeros(3)
        normalize_observations = bool(self.config.normalize_observations)
        bbox_dimensions = self.mj_model.initial_bounding_box_dimensions

        relative_positions = []
        absolute_velocities = []
        for pos, vel in zip(node_positions, node_velocities, strict=False):
            for axis_idx in range(3):
                divisor = bbox_dimensions[axis_idx] if normalize_observations else 1.0
                if axis_idx == 2:
                    relative_positions.append(pos[axis_idx] / divisor)
                else:
                    relative_positions.append((pos[axis_idx] - com[axis_idx]) / divisor)
                absolute_velocities.append(vel[axis_idx] / divisor)

        return np.concatenate(
            [
                np.array(relative_positions, dtype=np.float32),
                np.array(absolute_velocities, dtype=np.float32),
                self.node_velocity_controller.latest_node_commands.astype(np.float32),
            ]
        ).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        edge_ctrl = self.node_velocity_controller.clipped_edge_commands(
            self.mj_model.model,
            action,
        )

        previous_com = self._center_of_mass()
        self._advance(edge_ctrl)
        reward, info, terminated = self._compute_reward(action, previous_com)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info
