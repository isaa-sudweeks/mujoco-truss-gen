from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from mujoco_truss_gen.mujoco_model.model import ModelSource, MujocoModel

try:
    import mujoco.viewer as mujoco_viewer
except ImportError:
    mujoco_viewer = None


@dataclass(slots=True)
class TrussEnvConfig:
    """Configuration shared by the provided Gymnasium truss environments."""

    model_source: ModelSource
    max_steps: int = 10_000
    nsubsteps: int = 1
    speed: float = 0.01
    forward_weight: float = 5.0
    energy_weight: float = 0.005
    alive_bonus: float = 0.1
    rigidity_weight: float = 0.5
    slip_weight: float = 0.1
    critical_eig_threshold: float = 0.03
    slip_height: float = 0.2
    control_noise_std: float = 0.0
    control_noise_relative: bool = True
    runtime_apply_control_noise: bool = False


class MujocoTrussEnv(gym.Env):
    """Base Gymnasium environment for generated MuJoCo truss models."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        model_source: TrussEnvConfig | ModelSource,
        render_mode: str | None = None,
        rank: int = 0,
        **config_overrides: Any,
    ):
        super().__init__()
        self.config = _coerce_config(model_source, config_overrides)
        self.render_mode = render_mode
        self.rank = rank
        self.mj_model = MujocoModel(self.config.model_source)

        self.viewer = None
        self.renderer = None
        self.steps = 0
        self.max_steps = int(self.config.max_steps)
        self.nsubsteps = int(self.config.nsubsteps)
        self.control_noise_std = float(self.config.control_noise_std)
        self.control_noise_relative = bool(self.config.control_noise_relative)
        self.apply_control_noise = (
            bool(self.config.runtime_apply_control_noise) and self.control_noise_std > 0.0
        )

        self._set_render_fps()
        self._define_action_space()
        self._define_observation_space()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.mj_model.reset(self.np_random)
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        node_positions = self.mj_model.get_node_position_matrix()
        node_velocities = self.mj_model.get_node_linear_velocity_matrix()
        com = np.mean(node_positions, axis=0) if node_positions.size else np.zeros(3)
        com_vel = np.mean(node_velocities, axis=0) if node_velocities.size else np.zeros(3)

        return np.concatenate(
            [
                self.mj_model.data.ten_length,
                self.mj_model.data.ten_velocity,
                [com[0], com[2]],
                [com_vel[0], com_vel[2]],
            ]
        ).astype(np.float32)

    def _define_action_space(self) -> None:
        ctrlrange = self.mj_model.model.actuator_ctrlrange
        self.action_space = spaces.Box(
            low=ctrlrange[:, 0].astype(np.float32),
            high=ctrlrange[:, 1].astype(np.float32),
            dtype=np.float32,
        )

    def _define_observation_space(self) -> None:
        dummy_obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._advance(action)
        reward, info, terminated = self._compute_reward(action)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _advance(self, ctrl: np.ndarray) -> None:
        self.mj_model.data.ctrl[:] = self._apply_control_noise(ctrl)

        for _ in range(self.nsubsteps):
            mujoco.mj_step(self.mj_model.model, self.mj_model.data)
            if self.viewer is not None:
                self.viewer.sync()

        self.steps += 1

    def _apply_control_noise(self, ctrl: np.ndarray) -> np.ndarray:
        ctrl = np.asarray(ctrl, dtype=np.float32)
        ctrl_low = self.mj_model.model.actuator_ctrlrange[:, 0]
        ctrl_high = self.mj_model.model.actuator_ctrlrange[:, 1]
        clipped_ctrl = np.clip(ctrl, ctrl_low, ctrl_high)

        if not self.apply_control_noise:
            return clipped_ctrl

        if self.control_noise_relative:
            noise_scale = (ctrl_high - ctrl_low) * self.control_noise_std
        else:
            noise_scale = np.full_like(clipped_ctrl, self.control_noise_std, dtype=np.float32)

        noisy_ctrl = clipped_ctrl + self.np_random.normal(loc=0.0, scale=noise_scale).astype(
            np.float32
        )
        return np.clip(noisy_ctrl, ctrl_low, ctrl_high)

    def render(self):
        if self.render_mode == "rgb_array":
            if self.rank != 0:
                return np.zeros((480, 640, 3), dtype=np.uint8)
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.mj_model.model, 480, 640)
                self.cam = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(self.mj_model.model, self.cam)
                self.cam.distance = self.mj_model.model.stat.extent * 1.5

            self._update_camera_lookat(self.cam)
            self.renderer.update_scene(self.mj_model.data, camera=self.cam)
            return self.renderer.render()

        if self.render_mode == "human":
            if self.viewer is None:
                if mujoco_viewer is None:
                    raise RuntimeError(
                        "MuJoCo human viewer is unavailable in this Python environment. "
                        "Install a MuJoCo build that includes the viewer module."
                    )
                self.viewer = mujoco_viewer.launch_passive(self.mj_model.model, self.mj_model.data)

            self._update_camera_lookat(self.viewer.cam)
            self.viewer.sync()
            return None

        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict[str, float], bool]:
        critical_eig = float(self.mj_model.collapse_check())
        terminated = critical_eig < self.config.critical_eig_threshold

        forward_vel = self.mj_model.get_forward_velocity()
        energy_penalty = float(np.sum(np.square(action)))
        slip_penalty = float(self.mj_model.get_slip_penalty(height=self.config.slip_height))

        forward_reward = self.config.forward_weight * forward_vel
        energy_reward = -self.config.energy_weight * energy_penalty
        rigidity_reward = self.config.rigidity_weight * critical_eig
        slip_reward = -self.config.slip_weight * slip_penalty
        total_reward = (
            forward_reward + self.config.alive_bonus + energy_reward + rigidity_reward + slip_reward
        )
        info = {
            "forward": forward_reward,
            "alive": self.config.alive_bonus,
            "energy": energy_reward,
            "rigidity": rigidity_reward,
            "slip": slip_reward,
            "critical_eig": critical_eig,
        }

        return float(total_reward), info, terminated

    def _set_render_fps(self) -> None:
        dt = self.nsubsteps * self.mj_model.model.opt.timestep
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = int(np.round(1.0 / dt)) if dt > 0 else 20

    def _update_camera_lookat(self, camera: Any) -> None:
        positions = self.mj_model.get_node_position_matrix()
        if positions.size:
            camera.lookat[:] = np.mean(positions, axis=0)


def _coerce_config(
    model_source: TrussEnvConfig | ModelSource,
    overrides: dict[str, Any],
) -> TrussEnvConfig:
    if isinstance(model_source, TrussEnvConfig):
        if overrides:
            values = {field: getattr(model_source, field) for field in TrussEnvConfig.__slots__}
            values.update(overrides)
            return TrussEnvConfig(**values)
        return model_source

    if hasattr(model_source, "model_source") or hasattr(model_source, "xml_path"):
        values = {
            field: getattr(model_source, field)
            for field in TrussEnvConfig.__slots__
            if hasattr(model_source, field)
        }
        values["model_source"] = getattr(
            model_source,
            "model_source",
            getattr(model_source, "xml_path", None),
        )
        values.update(overrides)
        return TrussEnvConfig(**values)

    if isinstance(model_source, dict):
        values = dict(model_source)
        if "xml_path" in values and "model_source" not in values:
            values["model_source"] = values.pop("xml_path")
        values.update(overrides)
        return TrussEnvConfig(**values)

    return TrussEnvConfig(model_source=model_source, **overrides)
