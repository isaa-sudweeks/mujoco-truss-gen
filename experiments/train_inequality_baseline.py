"""Train a small PPO baseline with the inequality action parameterization.

The policy emits the same four parameters used by ``inequality_params.py``:
``[s, w1, w2, w3]``. They are mapped to logical node commands with one
component parallel to the rigidity gradient and three components in its null
space.

Stable-Baselines3 is the default because environment stepping and the MJX
gradient dominate runtime. SBX can be selected explicitly with ``--backend
sbx`` when running in an environment with a compatible JAX version.

Examples:
    uv run --with stable-baselines3 --with wandb --with 'imageio[ffmpeg]' \
        python experiments/train_inequality_baseline.py
    uv run python experiments/train_inequality_baseline.py --check-only
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from inequality_params import (
    DEFAULT_SPEED,
    RIGIDITY_THRESHOLD,
    TetrahedronSimulation,
)
from scipy.linalg import null_space

DEFAULT_OUTPUT = Path(__file__).with_name("runs") / "inequality_ppo"
DEFAULT_WANDB_PROJECT = "mujoco-truss-gen"


def parameterized_commands(
    action: np.ndarray,
    gradient: np.ndarray,
    critical_eigenvalue: float,
    *,
    speed: float,
    kappa: float,
) -> np.ndarray:
    """Apply the action mapping from ``TetrahedronControlGUI``."""
    action = np.asarray(action, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if action.shape != (4,) or gradient.shape != (4,):
        raise ValueError("action and gradient must both have shape (4,)")

    gradient_norm = float(np.linalg.norm(gradient))
    if not np.isfinite(gradient_norm) or gradient_norm <= 1e-12:
        return np.zeros(4, dtype=float)

    perpendicular_basis = null_space(gradient.reshape(1, -1))
    perpendicular_command = perpendicular_basis @ action[1:]
    parallel_command = (action[0] - kappa * critical_eigenvalue) * gradient / gradient_norm
    return np.clip(parallel_command + perpendicular_command, -speed, speed)


class InequalityTetrahedronEnv(gym.Env[np.ndarray, np.ndarray]):
    """Small locomotion task using the inequality parameterized action space."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        *,
        speed: float = DEFAULT_SPEED,
        scale: float = 1.0,
        frame_skip: int = 5,
        max_steps: int = 5_000,
        kappa: float = 0.0,
        forward_weight: float = 1.0,
        energy_weight: float = 0.002,
        rigidity_weight: float = 0.01,
        alive_bonus: float = 0.01,
        collapse_penalty: float = 1.0,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
    ) -> None:
        super().__init__()
        if frame_skip <= 0 or max_steps <= 0:
            raise ValueError("frame_skip and max_steps must be greater than zero")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        if render_width <= 0 or render_height <= 0:
            raise ValueError("render_width and render_height must be greater than zero")

        self.simulation = TetrahedronSimulation(speed=speed, scale=scale)
        if len(self.simulation.logical_node_names) != 4:
            raise ValueError("This baseline expects exactly four logical tetrahedron nodes")

        self.speed = float(speed)
        self.frame_skip = int(frame_skip)
        self.max_steps = int(max_steps)
        self.kappa = float(kappa)
        self.forward_weight = float(forward_weight)
        self.energy_weight = float(energy_weight)
        self.rigidity_weight = float(rigidity_weight)
        self.alive_bonus = float(alive_bonus)
        self.collapse_penalty = abs(float(collapse_penalty))
        self.render_mode = render_mode
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.renderer: mujoco.Renderer | None = None
        self.camera: mujoco.MjvCamera | None = None
        self.position_scale = max(
            float(self.simulation.truss.initial_bounding_box_diagonal),
            1e-8,
        )
        self.steps = 0
        self.gradient = np.zeros(4, dtype=float)
        self.critical_eigenvalue = 0.0
        self.rigidity_ratio = 0.0

        # These bounds exactly match the four bipolar sliders in inequality_params.py.
        self.action_space = spaces.Box(
            low=-self.speed,
            high=self.speed,
            shape=(4,),
            dtype=np.float32,
        )
        observation_size = 4 * 3 + 4 * 3 + 4 + 4 + 1
        self.observation_space = spaces.Box(
            low=-1e6,
            high=1e6,
            shape=(observation_size,),
            dtype=np.float32,
        )
        dt = self.frame_skip * float(self.simulation.model.opt.timestep)
        self.metadata = dict(self.metadata)
        self.metadata["render_fps"] = max(1, round(1.0 / dt))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        del options
        self.simulation.reset()
        self.steps = 0
        self._update_rigidity_state(compute_gradient=True)
        return self._observation(), self._info()

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        action = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        commands = parameterized_commands(
            action,
            self.gradient,
            self.critical_eigenvalue,
            speed=self.speed,
            kappa=self.kappa,
        )

        previous_com = self.simulation.center_of_mass.copy()
        self.simulation.set_logical_commands(commands)
        self.simulation.step(self.frame_skip)
        self.steps += 1

        self._update_rigidity_state(compute_gradient=False)
        state_is_finite = bool(
            np.all(np.isfinite(self.simulation.data.qpos))
            and np.all(np.isfinite(self.simulation.data.qvel))
        )
        terminated = bool(
            not state_is_finite
            or not np.isfinite(self.rigidity_ratio)
            or self.rigidity_ratio < RIGIDITY_THRESHOLD
        )
        truncated = self.steps >= self.max_steps

        dt = self.frame_skip * float(self.simulation.model.opt.timestep)
        com_delta_x = float(self.simulation.center_of_mass[0] - previous_com[0])
        forward_velocity = com_delta_x / max(dt, 1e-8)
        normalized_commands = commands / self.speed
        forward_reward = self.forward_weight * forward_velocity / self.position_scale
        energy_reward = -self.energy_weight * float(
            np.dot(normalized_commands, normalized_commands)
        )
        rigidity_reward = self.rigidity_weight * float(np.clip(self.rigidity_ratio, 0.0, 1.0))
        alive_reward = self.alive_bonus
        collapse_reward = 0.0
        if terminated:
            forward_reward = min(forward_reward, 0.0)
            rigidity_reward = 0.0
            alive_reward = 0.0
            collapse_reward = -self.collapse_penalty

        reward = float(
            forward_reward + energy_reward + rigidity_reward + alive_reward + collapse_reward
        )
        if not terminated:
            self._update_rigidity_state(compute_gradient=True)
        else:
            self.gradient.fill(0.0)

        info = self._info()
        info.update(
            {
                "forward_velocity": forward_velocity,
                "forward_reward": forward_reward,
                "energy_reward": energy_reward,
                "rigidity_reward": rigidity_reward,
                "alive_reward": alive_reward,
                "collapse_reward": collapse_reward,
            }
        )
        return self._observation(), reward, terminated, truncated, info

    def _update_rigidity_state(self, *, compute_gradient: bool) -> None:
        self.critical_eigenvalue, self.rigidity_ratio = self.simulation.rigidity_state()
        if compute_gradient:
            self.gradient = self.simulation.eigenvalue_gradient(
                self.simulation.logical_commands,
            )

    def _logical_velocities(self) -> np.ndarray:
        velocities = np.zeros((4, 3), dtype=float)
        object_velocity = np.zeros(6, dtype=float)
        for index, node_name in enumerate(self.simulation.logical_node_names):
            mujoco.mj_objectVelocity(
                self.simulation.model,
                self.simulation.data,
                mujoco.mjtObj.mjOBJ_BODY,
                self.simulation.logical_node_body_ids[node_name],
                object_velocity,
                0,
            )
            velocities[index] = object_velocity[3:]
        return velocities

    def _observation(self) -> np.ndarray:
        positions = np.stack(
            [
                self.simulation.data.xpos[self.simulation.logical_node_body_ids[node_name]]
                for node_name in self.simulation.logical_node_names
            ]
        )
        center = np.mean(positions, axis=0)
        relative_positions = positions - center
        relative_positions[:, 2] = positions[:, 2]

        gradient_norm = float(np.linalg.norm(self.gradient))
        normalized_gradient = (
            self.gradient / gradient_norm if gradient_norm > 1e-12 else np.zeros(4)
        )
        observation = np.concatenate(
            [
                relative_positions.ravel() / self.position_scale,
                self._logical_velocities().ravel() / self.position_scale,
                self.simulation.logical_commands / self.speed,
                normalized_gradient,
                [self.rigidity_ratio],
            ]
        )
        return np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    def _info(self) -> dict[str, float]:
        return {
            "critical_eigenvalue": float(self.critical_eigenvalue),
            "rigidity_ratio": float(self.rigidity_ratio),
            "com_x": float(self.simulation.center_of_mass[0]),
        }

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            visual = self.simulation.model.vis.global_
            visual.offwidth = max(int(visual.offwidth), self.render_width)
            visual.offheight = max(int(visual.offheight), self.render_height)
            self.renderer = mujoco.Renderer(
                self.simulation.model,
                height=self.render_height,
                width=self.render_width,
            )
            self.camera = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.simulation.model, self.camera)
            self.camera.distance = self.simulation.model.stat.extent * 1.5

        assert self.camera is not None
        self.camera.lookat[:] = self.simulation.center_of_mass
        self.renderer.update_scene(self.simulation.data, camera=self.camera)
        return np.asarray(self.renderer.render()).copy()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            self.camera = None


def load_ppo(backend: str) -> tuple[type[Any], str]:
    """Load PPO without making a training library mandatory for env checks."""
    selected = backend
    if backend == "auto":
        if importlib.util.find_spec("stable_baselines3") is not None:
            selected = "sb3"
        elif importlib.util.find_spec("sbx") is not None:
            selected = "sbx"
        else:
            selected = "sb3"

    try:
        if selected == "sbx":
            from sbx import PPO
        else:
            from stable_baselines3 import PPO
    except ImportError as error:
        package = "sbx-rl" if selected == "sbx" else "stable-baselines3"
        raise SystemExit(
            f"Missing training dependency. Install it with `uv pip install {package}` "
            f"or run with `uv run --with {package} ...`."
        ) from error
    return PPO, selected


def make_env(
    args: argparse.Namespace,
    *,
    render_mode: str | None = None,
) -> InequalityTetrahedronEnv:
    return InequalityTetrahedronEnv(
        speed=args.speed,
        scale=args.scale,
        frame_skip=args.frame_skip,
        max_steps=args.episode_steps,
        kappa=args.kappa,
        render_mode=render_mode,
        render_width=args.video_width,
        render_height=args.video_height,
    )


def make_wandb_callback(run: Any, log_interval: int) -> Any:
    """Create an SB3-compatible callback without requiring wandb integration extras."""
    from stable_baselines3.common.callbacks import BaseCallback

    class WandbStatisticsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.last_log_step = 0
            self.rewards: list[float] = []
            self.info_values: dict[str, list[float]] = {}

        def _flush(self) -> None:
            metrics: dict[str, float] = {
                "train/timesteps": float(self.num_timesteps),
            }
            if self.rewards:
                metrics["train/reward_mean"] = float(np.mean(self.rewards))
            for key, values in self.info_values.items():
                if values:
                    metrics[f"env/{key}_mean"] = float(np.mean(values))
            for key, value in self.model.logger.name_to_value.items():
                if isinstance(value, (bool, int, float, np.number)) and np.isfinite(value):
                    metrics[f"sb3/{key}"] = float(value)

            run.log(metrics, step=self.num_timesteps)
            self.rewards.clear()
            self.info_values.clear()
            self.last_log_step = self.num_timesteps

        def _on_step(self) -> bool:
            rewards = np.asarray(self.locals.get("rewards", []), dtype=float).ravel()
            self.rewards.extend(float(value) for value in rewards if np.isfinite(value))
            for info in self.locals.get("infos", []):
                for key, value in info.items():
                    if isinstance(value, (bool, int, float, np.number)) and np.isfinite(value):
                        self.info_values.setdefault(key, []).append(float(value))

            if self.num_timesteps - self.last_log_step >= log_interval:
                self._flush()
            return True

        def _on_training_end(self) -> None:
            self._flush()

    return WandbStatisticsCallback()


def evaluate(
    model: Any,
    args: argparse.Namespace,
    run: Any,
    wandb: Any,
    training_step: int,
) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as error:
        raise SystemExit(
            "Evaluation video recording requires `imageio[ffmpeg]`. Install it with "
            "`uv pip install 'imageio[ffmpeg]'`."
        ) from error

    env = make_env(args, render_mode="rgb_array")
    returns: list[float] = []
    distances: list[float] = []
    minimum_rigidity: list[float] = []
    video_count = (
        args.eval_episodes
        if args.eval_video_episodes < 0
        else min(args.eval_episodes, args.eval_video_episodes)
    )
    video_dir = args.video_dir or args.output.parent / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    try:
        for episode in range(args.eval_episodes):
            observation, _ = env.reset(seed=args.seed + episode)
            start_x = env.simulation.center_of_mass[0]
            episode_return = 0.0
            episode_minimum_rigidity = env.rigidity_ratio
            terminated = truncated = False
            video_path = video_dir / f"eval_episode_{episode + 1:03d}.mp4"
            writer = None
            if episode < video_count:
                writer = imageio.get_writer(
                    video_path,
                    fps=args.video_fps or env.metadata["render_fps"],
                    codec="libx264",
                    quality=8,
                )
                writer.append_data(env.render())
            try:
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    episode_return += reward
                    episode_minimum_rigidity = min(
                        episode_minimum_rigidity,
                        info["rigidity_ratio"],
                    )
                    if writer is not None:
                        writer.append_data(env.render())
            finally:
                if writer is not None:
                    writer.close()

            returns.append(episode_return)
            distances.append(float(env.simulation.center_of_mass[0] - start_x))
            minimum_rigidity.append(float(episode_minimum_rigidity))
            episode_metrics: dict[str, Any] = {
                "eval/episode_return": episode_return,
                "eval/delta_x_m": distances[-1],
                "eval/minimum_rigidity_ratio": minimum_rigidity[-1],
                "eval/episode": episode + 1,
            }
            if episode < video_count:
                episode_metrics[f"eval/video_episode_{episode + 1}"] = wandb.Video(
                    str(video_path),
                    fps=args.video_fps or env.metadata["render_fps"],
                    format="mp4",
                )
            run.log(episode_metrics, step=training_step + episode + 1)
    finally:
        env.close()

    summary = {
        "eval/return_mean": float(np.mean(returns)),
        "eval/return_std": float(np.std(returns)),
        "eval/delta_x_mean_m": float(np.mean(distances)),
        "eval/delta_x_std_m": float(np.std(distances)),
        "eval/minimum_rigidity_mean": float(np.mean(minimum_rigidity)),
    }
    run.log(summary, step=training_step + args.eval_episodes + 1)
    print(
        "evaluation: "
        f"return={np.mean(returns):.3f} +/- {np.std(returns):.3f}, "
        f"delta_x={np.mean(distances):.4f} +/- {np.std(distances):.4f} m"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "sbx", "sb3"), default="auto")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--episode-steps", type=int, default=5_000)
    parser.add_argument("--frame-skip", type=int, default=5)
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-log-interval", type=int, default=100)
    parser.add_argument(
        "--eval-video-episodes",
        type=int,
        default=-1,
        help="Number of eval episodes to record; -1 records every eval episode.",
    )
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument(
        "--video-fps",
        type=int,
        help="Video playback FPS; defaults to the environment control frequency.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Compile the gradient and run two random environment steps without PPO.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.speed <= 0.0:
        raise SystemExit("--speed must be greater than zero")
    if args.n_steps <= 1 or args.batch_size <= 1:
        raise SystemExit("--n-steps and --batch-size must be greater than one")
    if args.n_steps % args.batch_size != 0:
        raise SystemExit("--batch-size must divide --n-steps for this single-env baseline")
    if args.wandb_log_interval <= 0:
        raise SystemExit("--wandb-log-interval must be greater than zero")
    if args.video_fps is not None and args.video_fps <= 0:
        raise SystemExit("--video-fps must be greater than zero")

    env = make_env(args)
    if args.check_only:
        try:
            observation, info = env.reset(seed=args.seed)
            print(
                f"reset: observation={observation.shape}, "
                f"rigidity_ratio={info['rigidity_ratio']:.4f}"
            )
            for step in range(2):
                observation, reward, terminated, truncated, info = env.step(
                    env.action_space.sample()
                )
                print(
                    f"step={step + 1} reward={reward:+.4f} "
                    f"rigidity_ratio={info['rigidity_ratio']:.4f} "
                    f"terminated={terminated} truncated={truncated}"
                )
                if terminated or truncated:
                    break
        finally:
            env.close()
        return

    try:
        import wandb
    except ImportError as error:
        env.close()
        raise SystemExit(
            "W&B logging requires `wandb`. Install it with `uv pip install wandb`."
        ) from error

    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=config,
        save_code=True,
    )
    try:
        PPO, backend = load_ppo(args.backend)
        run.config.update({"resolved_backend": backend})
        print(f"training PPO with backend={backend}, timesteps={args.timesteps}")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [64, 64]},
            seed=args.seed,
            verbose=1,
        )
        callback = make_wandb_callback(run, args.wandb_log_interval)
        try:
            model.learn(
                total_timesteps=args.timesteps,
                callback=callback,
                progress_bar=args.progress,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            model.save(args.output)
            model_path = args.output if args.output.suffix == ".zip" else Path(f"{args.output}.zip")
            print(f"saved model to {model_path}")
        finally:
            env.close()

        model_artifact = wandb.Artifact(
            name=f"{run.id}-model",
            type="model",
            metadata={"backend": backend, "timesteps": args.timesteps},
        )
        model_artifact.add_file(str(model_path))
        run.log_artifact(model_artifact)

        if args.eval_episodes > 0:
            evaluate(model, args, run, wandb, int(model.num_timesteps))
    finally:
        env.close()
        run.finish()


if __name__ == "__main__":
    main()
