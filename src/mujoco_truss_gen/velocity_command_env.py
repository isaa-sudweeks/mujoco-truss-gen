from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from mujoco_truss_gen.base_env import TrussEnvConfig
from mujoco_truss_gen.mujoco_model.model import ModelSource
from mujoco_truss_gen.relative_observation_env import MujocoRelativeObsEnv


class MujocoVelocityCommandEnv(MujocoRelativeObsEnv):
    """Relative-observation environment with direct actuator velocity commands."""

    def __init__(
        self,
        model_source: TrussEnvConfig | ModelSource,
        render_mode: str | None = None,
        rank: int = 0,
        **config_overrides: Any,
    ):
        super().__init__(model_source, render_mode=render_mode, rank=rank, **config_overrides)
        self.action_space = spaces.Box(
            low=-self.config.speed,
            high=self.config.speed,
            shape=(self.mj_model.model.nu,),
            dtype=np.float32,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._advance(action)
        reward, info, terminated = self._compute_reward(action)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info
