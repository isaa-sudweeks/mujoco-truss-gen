from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from mujoco_truss_gen.base_env import MujocoTrussEnv, TrussEnvConfig
from mujoco_truss_gen.mujoco_model.model import ModelSource


class MujocoRelativeObsEnv(MujocoTrussEnv):
    """Truss environment with translationally invariant node-position observations.

    Actions are normalized actuator command deltas. Each component in ``[-1, 1]``
    changes the previous actuator control by ``config.speed`` before clipping to
    the model's actuator control range.
    """

    def __init__(
        self,
        model_source: TrussEnvConfig | ModelSource,
        render_mode: str | None = None,
        rank: int = 0,
        **config_overrides: Any,
    ):
        super().__init__(model_source, render_mode=render_mode, rank=rank, **config_overrides)

    def _define_action_space(self) -> None:
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.mj_model.external_actuator_ids),),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        node_positions = self.mj_model.get_node_position_dict()
        node_velocities = self.mj_model.get_node_velocity_linear_dict()
        position_matrix = self.mj_model.get_node_position_matrix()
        com = np.mean(position_matrix, axis=0) if position_matrix.size else np.zeros(3)
        active_axes = self.mj_model.active_axes

        relative_positions = []
        absolute_velocities = []

        for node_name in self.mj_model.node_names:
            pos = node_positions[node_name]
            vel = node_velocities[node_name]

            for axis in active_axes:
                axis_idx = "xyz".index(axis)
                if axis == "z":
                    relative_positions.append(pos[axis_idx])
                else:
                    relative_positions.append(pos[axis_idx] - com[axis_idx])
                absolute_velocities.append(vel[axis_idx])

        return np.concatenate(
            [
                np.array(relative_positions, dtype=np.float32),
                np.array(absolute_velocities, dtype=np.float32),
                self.mj_model.get_external_ctrl().astype(np.float32),
            ]
        ).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        ctrl = self.mj_model.get_external_ctrl() + action * self.config.speed
        ctrlrange = self.mj_model.get_external_ctrlrange()
        ctrl_low = ctrlrange[:, 0]
        ctrl_high = ctrlrange[:, 1]
        ctrl = np.clip(ctrl, ctrl_low, ctrl_high)

        previous_com = self._center_of_mass()
        self._advance(ctrl)
        reward, info, terminated = self._compute_reward(action, previous_com)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info
