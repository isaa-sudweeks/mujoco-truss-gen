from __future__ import annotations

from collections.abc import Callable
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


Range = tuple[float, float]
ModelFactory = Callable[[np.random.Generator], ModelSource]


@dataclass(slots=True)
class DomainRandomizationConfig:
    """Per-episode domain randomization for Gymnasium truss environments.

    ``model_factory`` is called on every reset and should return a fresh model
    source. Use it for scale, topology, geometry, or physical parameters that
    are baked into the compiled MuJoCo model.

    The remaining fields mutate the compiled ``mujoco.MjModel`` at reset time.
    They are sampled independently and restored from nominal model values before
    each new sample is applied.
    """

    model_factory: ModelFactory | None = None
    body_mass_multiplier_range: Range | None = None
    body_inertia_multiplier_range: Range | None = None
    dof_damping_multiplier_range: Range | None = None
    actuator_gain_multiplier_range: Range | None = None
    actuator_bias_multiplier_range: Range | None = None
    geom_friction_slide_range: Range | None = None
    gravity_z_range: Range | None = None


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
    domain_randomization: DomainRandomizationConfig | None = None
    max_forward_velocity: float | None = 1.0
    zero_positive_forward_reward_on_termination: bool = True
    collapse_penalty: float = 0.0
    zero_alive_bonus_on_termination: bool = True
    zero_rigidity_reward_on_termination: bool = True
    zero_velocity_shaping_on_termination: bool = True


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
        self._runtime_nominals: dict[str, np.ndarray] = {}

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

        self._on_model_changed()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        randomization_info = self._randomize_domain()
        self.mj_model.reset(self.np_random)
        self.steps = 0
        return self._get_obs(), {"domain_randomization": randomization_info}

    def _on_model_changed(self) -> None:
        self._capture_runtime_nominals()
        self._set_render_fps()
        self._define_action_space()
        self._define_observation_space()

    def _randomize_domain(self) -> dict[str, float]:
        randomization = self.config.domain_randomization
        if randomization is None:
            return {}

        if randomization.model_factory is not None:
            self._replace_model(randomization.model_factory(self.np_random))
        else:
            self._restore_runtime_nominals()

        return self._apply_runtime_domain_randomization(randomization)

    def _replace_model(self, model_source: ModelSource) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

        self.mj_model = MujocoModel(model_source)
        self._on_model_changed()

    def _capture_runtime_nominals(self) -> None:
        model = self.mj_model.model
        self._runtime_nominals = {
            "body_mass": model.body_mass.copy(),
            "body_inertia": model.body_inertia.copy(),
            "dof_damping": model.dof_damping.copy(),
            "actuator_gainprm": model.actuator_gainprm.copy(),
            "actuator_biasprm": model.actuator_biasprm.copy(),
            "geom_friction": model.geom_friction.copy(),
            "gravity": model.opt.gravity.copy(),
        }

    def _restore_runtime_nominals(self) -> None:
        if not self._runtime_nominals:
            return

        model = self.mj_model.model
        model.body_mass[:] = self._runtime_nominals["body_mass"]
        model.body_inertia[:] = self._runtime_nominals["body_inertia"]
        model.dof_damping[:] = self._runtime_nominals["dof_damping"]
        model.actuator_gainprm[:] = self._runtime_nominals["actuator_gainprm"]
        model.actuator_biasprm[:] = self._runtime_nominals["actuator_biasprm"]
        model.geom_friction[:] = self._runtime_nominals["geom_friction"]
        model.opt.gravity[:] = self._runtime_nominals["gravity"]

    def _apply_runtime_domain_randomization(
        self,
        randomization: DomainRandomizationConfig,
    ) -> dict[str, float]:
        self._restore_runtime_nominals()
        model = self.mj_model.model
        samples: dict[str, float] = {}

        body_mass_multiplier = _sample_range(
            self.np_random,
            randomization.body_mass_multiplier_range,
            "body_mass_multiplier_range",
        )
        if body_mass_multiplier is not None:
            model.body_mass[:] = self._runtime_nominals["body_mass"] * body_mass_multiplier
            samples["body_mass_multiplier"] = body_mass_multiplier

        body_inertia_multiplier = _sample_range(
            self.np_random,
            randomization.body_inertia_multiplier_range,
            "body_inertia_multiplier_range",
        )
        if body_inertia_multiplier is not None:
            model.body_inertia[:] = (
                self._runtime_nominals["body_inertia"] * body_inertia_multiplier
            )
            samples["body_inertia_multiplier"] = body_inertia_multiplier

        dof_damping_multiplier = _sample_range(
            self.np_random,
            randomization.dof_damping_multiplier_range,
            "dof_damping_multiplier_range",
        )
        if dof_damping_multiplier is not None:
            model.dof_damping[:] = (
                self._runtime_nominals["dof_damping"] * dof_damping_multiplier
            )
            samples["dof_damping_multiplier"] = dof_damping_multiplier

        actuator_gain_multiplier = _sample_range(
            self.np_random,
            randomization.actuator_gain_multiplier_range,
            "actuator_gain_multiplier_range",
        )
        if actuator_gain_multiplier is not None:
            model.actuator_gainprm[:] = (
                self._runtime_nominals["actuator_gainprm"] * actuator_gain_multiplier
            )
            samples["actuator_gain_multiplier"] = actuator_gain_multiplier

        actuator_bias_multiplier = _sample_range(
            self.np_random,
            randomization.actuator_bias_multiplier_range,
            "actuator_bias_multiplier_range",
        )
        if actuator_bias_multiplier is not None:
            model.actuator_biasprm[:] = (
                self._runtime_nominals["actuator_biasprm"] * actuator_bias_multiplier
            )
            samples["actuator_bias_multiplier"] = actuator_bias_multiplier

        geom_friction_slide = _sample_range(
            self.np_random,
            randomization.geom_friction_slide_range,
            "geom_friction_slide_range",
        )
        if geom_friction_slide is not None:
            model.geom_friction[:, 0] = geom_friction_slide
            samples["geom_friction_slide"] = geom_friction_slide

        gravity_z = _sample_range(
            self.np_random,
            randomization.gravity_z_range,
            "gravity_z_range",
        )
        if gravity_z is not None:
            model.opt.gravity[2] = gravity_z
            samples["gravity_z"] = gravity_z

        if samples:
            mujoco.mj_setConst(model, self.mj_model.data)

        return samples

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
        ctrlrange = self.mj_model.get_external_ctrlrange()
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
        previous_com = self._center_of_mass()
        self._advance(action)
        reward, info, terminated = self._compute_reward(action, previous_com)
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, info

    def _advance(self, ctrl: np.ndarray) -> None:
        self.mj_model.set_external_ctrl(self._apply_control_noise(ctrl))

        for _ in range(self.nsubsteps):
            self.mj_model.apply_angle_bisector_control()
            mujoco.mj_step(self.mj_model.model, self.mj_model.data)
            if self.viewer is not None:
                self.viewer.sync()

        self.steps += 1

    def _apply_control_noise(self, ctrl: np.ndarray) -> np.ndarray:
        ctrl = np.asarray(ctrl, dtype=np.float32)
        ctrlrange = self.mj_model.get_external_ctrlrange()
        ctrl_low = ctrlrange[:, 0]
        ctrl_high = ctrlrange[:, 1]
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

    def _center_of_mass(self) -> np.ndarray:
        positions = self.mj_model.get_node_position_matrix()
        if positions.size == 0:
            return np.zeros(3, dtype=float)
        return np.mean(positions, axis=0)

    def _compute_reward(
        self,
        action: np.ndarray,
        previous_com: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float | bool], bool]:
        critical_eig_raw = float(self.mj_model.collapse_check())
        terminated = (
            not np.isfinite(critical_eig_raw)
            or critical_eig_raw < self.config.critical_eig_threshold
        )
        critical_eig = critical_eig_raw if np.isfinite(critical_eig_raw) else 0.0

        com_delta_x = 0.0
        if previous_com is None:
            raw_forward_vel = self.mj_model.get_forward_velocity()
        else:
            current_com = self._center_of_mass()
            com_delta_x = float(current_com[0] - previous_com[0])
            dt = float(self.nsubsteps) * float(self.mj_model.model.opt.timestep)
            raw_forward_vel = 0.0 if dt <= 0.0 else com_delta_x / dt

        reward_forward_vel = float(raw_forward_vel) if np.isfinite(raw_forward_vel) else 0.0
        if self.config.max_forward_velocity is None:
            forward_vel = reward_forward_vel
        else:
            velocity_limit = abs(float(self.config.max_forward_velocity))
            forward_vel = float(np.clip(reward_forward_vel, -velocity_limit, velocity_limit))

        if terminated and self.config.zero_positive_forward_reward_on_termination:
            forward_vel = min(forward_vel, 0.0)

        energy_penalty = float(np.sum(np.square(action)))
        if terminated and self.config.zero_velocity_shaping_on_termination:
            slip_penalty = 0.0
        else:
            slip_penalty = float(self.mj_model.get_slip_penalty(height=self.config.slip_height))
            if not np.isfinite(slip_penalty):
                slip_penalty = 0.0

        forward_reward = (
            self.config.forward_weight
            * forward_vel
            / max(float(self.mj_model.initial_bounding_box_diagonal), 1e-8)
        )
        energy_reward = -self.config.energy_weight * energy_penalty
        rigidity_reward = self.config.rigidity_weight * critical_eig
        if terminated and self.config.zero_rigidity_reward_on_termination:
            rigidity_reward = 0.0
        slip_reward = -self.config.slip_weight * slip_penalty
        alive_reward = float(self.config.alive_bonus)
        if terminated and self.config.zero_alive_bonus_on_termination:
            alive_reward = 0.0
        collapse_penalty = -abs(float(self.config.collapse_penalty)) if terminated else 0.0
        total_reward = (
            forward_reward
            + alive_reward
            + energy_reward
            + rigidity_reward
            + slip_reward
            + collapse_penalty
        )
        info = {
            "forward": forward_reward,
            "forward_velocity": forward_vel,
            "forward_velocity_raw": float(raw_forward_vel),
            "com_delta_x": com_delta_x,
            "alive": alive_reward,
            "energy": energy_reward,
            "rigidity": rigidity_reward,
            "slip": slip_reward,
            "critical_eig": critical_eig,
            "critical_eig_raw": critical_eig_raw,
            "collapse_penalty": collapse_penalty,
            "terminated_by_collapse": terminated,
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


def _sample_range(
    rng: np.random.Generator,
    value_range: Range | None,
    name: str,
) -> float | None:
    if value_range is None:
        return None

    low, high = (float(value_range[0]), float(value_range[1]))
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        raise ValueError(f"{name} must contain finite values with low <= high.")
    return float(rng.uniform(low, high))
