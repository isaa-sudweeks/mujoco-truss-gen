from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import null_space

from mujoco_truss_gen import MujocoModel, NodeVelocityController, get_mujoco_spec
from mujoco_truss_gen.mujoco_model.tendons import initialize_actuator_lengths
from mujoco_truss_gen.numerical import numerical_gradient

DEFAULT_SPEED = 0.05
DEFAULT_FPS = 30
DEFAULT_WIDTH = 980
DEFAULT_HEIGHT = 720
RIGIDITY_THRESHOLD = 0.03
GRADIENT_EPSILON_FRACTION = 0.01
GRADIENT_UPDATE_INTERVAL_FRAMES = 3

COLORS = {
    "app": "#080C11",
    "topbar": "#0C1219",
    "surface": "#0F161E",
    "surface_alt": "#121B24",
    "surface_hover": "#17232E",
    "border": "#24313D",
    "border_bright": "#344553",
    "text": "#E7EDF2",
    "muted": "#81909E",
    "faint": "#52616E",
    "cyan": "#49C7D4",
    "cyan_dim": "#205D66",
    "amber": "#E9A84A",
    "green": "#57D18C",
    "track": "#303C47",
    "danger": "#E27272",
}


def logical_node_name(node_name: str) -> str:
    """Collapse realistic route clones back to their tetrahedron node name."""
    return node_name.split("_route_", 1)[0].split("_tri_", 1)[0]


class TetrahedronSimulation:
    """Realistic routed-tube tetrahedron driven by logical node commands."""

    def __init__(self, speed: float = DEFAULT_SPEED, scale: float = 1.0):
        if speed <= 0.0:
            raise ValueError("speed must be greater than zero")

        self.speed = float(speed)
        spec = get_mujoco_spec("tetrahedron", realistic=True, scale=scale)
        self.truss = MujocoModel(spec)
        self.model = self.truss.model
        self.data = self.truss.data
        self.controller = NodeVelocityController(
            self.model,
            self.truss.xml,
            self.truss.node_names,
            self.truss.site_to_node,
            self.truss.external_actuator_ids,
        )
        if not self.controller.enabled:
            raise RuntimeError("The realistic tetrahedron has no routed node controller.")

        instances: dict[str, list[int]] = defaultdict(list)
        for index, node_name in enumerate(self.controller.node_names):
            instances[logical_node_name(node_name)].append(index)
        self.logical_node_names = sorted(instances, key=self._node_sort_key)
        self.logical_node_instances = dict(instances)
        self.logical_node_body_ids = {
            node_name: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"connector_ball_{node_name}",
            )
            for node_name in self.logical_node_names
        }
        self.logical_commands = np.zeros(len(self.logical_node_names), dtype=float)
        self.physical_commands = np.zeros(len(self.controller.node_names), dtype=float)
        self.edge_commands = np.zeros(len(self.controller.edge_names), dtype=float)
        self.reset()

    @staticmethod
    def _node_sort_key(node_name: str) -> tuple[int, str]:
        try:
            return int(node_name.split("_")[1]), node_name
        except (IndexError, ValueError):
            return 10**9, node_name

    def set_logical_commands(self, commands: np.ndarray) -> None:
        self.logical_commands[:] = self._validated_logical_commands(commands)
        self.physical_commands[:] = self._physical_commands(self.logical_commands)

    def _validated_logical_commands(self, commands: np.ndarray) -> np.ndarray:
        commands = np.asarray(commands, dtype=float)
        expected_shape = (len(self.logical_node_names),)
        if commands.shape != expected_shape:
            raise ValueError(
                f"Expected node commands with shape {expected_shape}, got {commands.shape}."
            )
        return np.clip(commands, -self.speed, self.speed)

    def _physical_commands(self, logical_commands: np.ndarray) -> np.ndarray:
        physical_commands = np.zeros(len(self.controller.node_names), dtype=float)
        for logical_index, node_name in enumerate(self.logical_node_names):
            physical_commands[self.logical_node_instances[node_name]] = logical_commands[
                logical_index
            ]
        return physical_commands

    def eigenvalue_gradient(
        self,
        commands: np.ndarray,
        *,
        rollout_steps: int = 1,
        epsilon: float | None = None,
    ) -> np.ndarray:
        """Differentiate the post-rollout seventh eigenvalue by control command."""
        commands = self._validated_logical_commands(commands)
        if rollout_steps <= 0:
            raise ValueError("rollout_steps must be greater than zero")

        if epsilon is None:
            epsilon = self.speed * GRADIENT_EPSILON_FRACTION
        state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, state_spec), dtype=float)
        mujoco.mj_getState(self.model, self.data, state, state_spec)
        probe_data = mujoco.MjData(self.model)

        def post_rollout_eigenvalue(probe_commands: np.ndarray) -> float:
            mujoco.mj_setState(self.model, probe_data, state, state_spec)
            mujoco.mj_forward(self.model, probe_data)
            physical_commands = self._physical_commands(probe_commands)
            physical_commands[self.controller.passive_node_mask] = 0.0
            edge_commands = self.controller.incidence_matrix @ physical_commands
            ctrlrange = self.model.actuator_ctrlrange[self.controller.actuator_ids]
            edge_commands = np.clip(edge_commands, ctrlrange[:, 0], ctrlrange[:, 1])

            live_data = self.truss.data
            self.truss.data = probe_data
            try:
                for _ in range(rollout_steps):
                    probe_data.ctrl[self.controller.actuator_ids] = edge_commands
                    self.truss.apply_angle_bisector_control()
                    mujoco.mj_step(self.model, probe_data)
                return float(self.truss._critical_eig())
            finally:
                self.truss.data = live_data

        return numerical_gradient(
            post_rollout_eigenvalue,
            commands,
            epsilon=epsilon,
            lower_bound=-self.speed,
            upper_bound=self.speed,
        )

    def step(self, steps: int = 1) -> None:
        for _ in range(steps):
            self.edge_commands = self.controller.apply(
                self.model,
                self.data,
                self.physical_commands,
            )
            self.truss.apply_angle_bisector_control()
            mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        self.logical_commands.fill(0.0)
        self.physical_commands.fill(0.0)
        self.edge_commands.fill(0.0)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.truss.apply_angle_bisector_control()
        initialize_actuator_lengths(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    @property
    def center_of_mass(self) -> np.ndarray:
        positions = self.truss.get_node_position_matrix()
        return np.mean(positions, axis=0) if positions.size else np.zeros(3)

    def logical_node_positions(self) -> dict[str, np.ndarray]:
        return {
            node_name: self.data.xpos[body_id].copy()
            for node_name, body_id in self.logical_node_body_ids.items()
            if body_id >= 0
        }

    def rigidity_state(self) -> tuple[float, float]:
        critical_eigenvalue = float(self.truss._critical_eig())
        normalized = critical_eigenvalue / self.truss.initial_critical_eig
        return critical_eigenvalue, normalized


class BipolarSlider:
    """A compact, custom-drawn velocity slider with a fixed zero point."""

    def __init__(
        self,
        parent: Any,
        tk: Any,
        node_name: str,
        limit: float,
        font_family: str,
        command: Any,
    ):
        self.tk = tk
        self.limit = float(limit)
        self.command = command
        self.value = 0.0

        self.frame = tk.Frame(parent, bg=COLORS["surface_alt"], padx=12, pady=4)
        header = tk.Frame(self.frame, bg=COLORS["surface_alt"])
        header.pack(fill="x")
        tk.Label(
            header,
            text=node_name,
            bg=COLORS["surface_alt"],
            fg=COLORS["text"],
            font=(font_family, 11, "bold"),
        ).pack(side="left")
        self.value_label = tk.Label(
            header,
            text="+0.0000 m/s",
            bg=COLORS["surface_alt"],
            fg=COLORS["cyan"],
            font=("Menlo", 10),
        )
        self.value_label.pack(side="right")

        self.canvas = tk.Canvas(
            self.frame,
            height=22,
            bg=COLORS["surface_alt"],
            highlightthickness=0,
            cursor="hand2",
        )
        self.canvas.pack(fill="x")
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<Button-1>", self._set_from_event)
        self.canvas.bind("<B1-Motion>", self._set_from_event)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.set(0.0))

    def pack(self, **kwargs: Any) -> None:
        self.frame.pack(**kwargs)

    def get(self) -> float:
        return self.value

    def set(self, value: float, notify: bool = True) -> None:
        self.value = float(np.clip(value, -self.limit, self.limit))
        self.value_label.configure(
            text=f"{self.value:+.4f} m/s",
            fg=COLORS["amber"] if self.value > 0.0 else COLORS["cyan"],
        )
        self.draw()
        if notify:
            self.command()

    def draw(self) -> None:
        canvas = self.canvas
        width = max(canvas.winfo_width(), 40)
        x0, x1 = 10.0, width - 10.0
        center = (x0 + x1) / 2.0
        y = 11.0
        value_x = center + (self.value / self.limit) * (x1 - x0) / 2.0

        canvas.delete("all")
        canvas.create_line(x0, y, x1, y, fill=COLORS["track"], width=5)
        if abs(self.value) > 1e-10:
            canvas.create_line(
                center,
                y,
                value_x,
                y,
                fill=COLORS["amber"] if self.value > 0.0 else COLORS["cyan"],
                width=5,
            )
        for tick_x in (x0, center, x1):
            canvas.create_line(tick_x, y - 5, tick_x, y + 5, fill=COLORS["border_bright"])
        canvas.create_oval(
            value_x - 6,
            y - 6,
            value_x + 6,
            y + 6,
            fill=COLORS["text"],
            outline=COLORS["surface_alt"],
            width=2,
        )

    def _set_from_event(self, event: Any) -> None:
        width = max(self.canvas.winfo_width(), 40)
        x0, x1 = 10.0, width - 10.0
        fraction = np.clip((event.x - x0) / (x1 - x0), 0.0, 1.0)
        raw_value = (2.0 * fraction - 1.0) * self.limit
        resolution = self.limit / 200.0
        self.set(round(raw_value / resolution) * resolution)


class CanvasButton:
    """Theme-stable button that avoids platform-native text color overrides."""

    def __init__(
        self,
        parent: Any,
        tk: Any,
        text: str,
        command: Any,
        font_family: str,
        primary: bool = False,
    ):
        self.tk = tk
        self.text = text
        self.command = command
        self.primary = primary
        self.hovered = False
        self.pressed = False
        self.canvas = tk.Canvas(
            parent,
            height=36,
            width=100,
            bg=COLORS["surface"],
            highlightthickness=0,
            cursor="hand2",
            takefocus=True,
        )
        self.font = (font_family, 9, "bold")
        self.canvas.bind("<Configure>", lambda _event: self.draw())
        self.canvas.bind("<Enter>", self._enter)
        self.canvas.bind("<Leave>", self._leave)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Key-space>", self._keyboard_activate)
        self.canvas.bind("<Key-Return>", self._keyboard_activate)

    def pack(self, **kwargs: Any) -> None:
        self.canvas.pack(**kwargs)

    def configure(self, **kwargs: Any) -> None:
        if "text" in kwargs:
            self.text = str(kwargs["text"])
        self.draw()

    def draw(self) -> None:
        width = max(self.canvas.winfo_width(), 40)
        height = max(self.canvas.winfo_height(), 30)
        if self.primary:
            fill = COLORS["cyan_dim"]
            outline = COLORS["cyan"]
        else:
            fill = COLORS["surface_alt"]
            outline = COLORS["border_bright"]
        if self.hovered:
            fill = COLORS["cyan_dim"] if self.primary else COLORS["surface_hover"]
        if self.pressed:
            fill = COLORS["cyan"] if self.primary else COLORS["border_bright"]

        self.canvas.delete("all")
        self.canvas.create_rectangle(
            1,
            1,
            width - 1,
            height - 1,
            fill=fill,
            outline=outline,
            width=1,
        )
        self.canvas.create_text(
            width / 2,
            height / 2,
            text=self.text,
            fill=COLORS["app"] if self.primary and self.pressed else COLORS["text"],
            font=self.font,
        )

    def _enter(self, _event: Any) -> None:
        self.hovered = True
        self.draw()

    def _leave(self, _event: Any) -> None:
        self.hovered = False
        self.pressed = False
        self.draw()

    def _press(self, _event: Any) -> None:
        self.pressed = True
        self.draw()

    def _release(self, event: Any) -> None:
        was_pressed = self.pressed
        self.pressed = False
        inside = (
            0 <= event.x < self.canvas.winfo_width() and 0 <= event.y < self.canvas.winfo_height()
        )
        self.draw()
        if was_pressed and inside:
            self.command()

    def _keyboard_activate(self, _event: Any) -> str:
        self.command()
        return "break"


class TetrahedronControlGUI:
    """Professional dark control console with an embedded MuJoCo renderer."""

    def __init__(
        self,
        simulation: TetrahedronSimulation,
        fps: int = DEFAULT_FPS,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ):
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError("This experiment requires Python with tkinter support.") from exc

        if fps <= 0:
            raise ValueError("fps must be greater than zero")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than zero")

        self.tk = tk
        self.ttk = ttk
        self.simulation = simulation
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.frame_period_ms = max(1, round(1000 / self.fps))
        self.steps_per_frame = max(
            1,
            round(1.0 / (self.fps * self.simulation.model.opt.timestep)),
        )
        self.running = True
        self.closed = False
        self.last_wall_time = time.perf_counter()
        self.drag_position: tuple[int, int] | None = None
        self.current_critical_eigenvalue = self.simulation.rigidity_state()[0]
        self.eigenvalue_gradient = np.zeros(len(self.simulation.logical_node_names), dtype=float)
        self.control_parameters = np.zeros(len(self.simulation.logical_node_names), dtype=float)
        self.simulation_frame = 0

        self.root = tk.Tk()
        self.font_family = self._choose_font()
        self.root.title("Tetra Lab - Realistic Tetrahedron")
        self.root.configure(bg=COLORS["app"])
        self.root.geometry(f"{self.width + 430}x{self.height + 92}")
        self.root.minsize(1180, 720)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        visual = self.simulation.model.vis.global_
        visual.offwidth = max(int(visual.offwidth), self.width)
        visual.offheight = max(int(visual.offheight), self.height)
        self.renderer = mujoco.Renderer(
            self.simulation.model,
            height=self.height,
            width=self.width,
        )
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.simulation.model, self.camera)
        self.camera.lookat[:] = self.simulation.center_of_mass

        self._build_layout()
        self._bind_camera_controls()
        self._render_frame()

    def _choose_font(self) -> str:
        available = set(self.root.tk.call("font", "families"))
        for candidate in ("SF Pro Display", "Helvetica Neue", "Arial"):
            if candidate in available:
                return candidate
        return "TkDefaultFont"

    def _build_layout(self) -> None:
        tk = self.tk
        self._build_topbar()

        body = tk.Frame(self.root, bg=COLORS["app"], padx=16, pady=16)
        body.pack(fill="both", expand=True)
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, minsize=394)

        viewport = tk.Frame(
            body,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        viewport.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        viewport.grid_rowconfigure(0, weight=1)
        viewport.grid_columnconfigure(0, weight=1)

        self.image_label = tk.Label(viewport, bg="#05090D", anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        viewport_footer = tk.Frame(viewport, bg=COLORS["surface"], height=40)
        viewport_footer.grid(row=1, column=0, sticky="ew")
        viewport_footer.grid_propagate(False)
        tk.Label(
            viewport_footer,
            text="3D VIEWPORT",
            bg=COLORS["surface"],
            fg=COLORS["faint"],
            font=(self.font_family, 9, "bold"),
        ).pack(side="left", padx=14)
        tk.Label(
            viewport_footer,
            text="Drag to orbit   |   Scroll to zoom",
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            font=(self.font_family, 9),
        ).pack(side="right", padx=14)

        panel = tk.Frame(
            body,
            width=394,
            bg=COLORS["surface"],
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        self._build_control_panel(panel)

    def _build_topbar(self) -> None:
        tk = self.tk
        topbar = tk.Frame(
            self.root,
            bg=COLORS["topbar"],
            height=60,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        mark = tk.Canvas(topbar, width=34, height=34, bg=COLORS["topbar"], highlightthickness=0)
        mark.pack(side="left", padx=(18, 10))
        mark.create_polygon(17, 4, 4, 28, 30, 28, outline=COLORS["cyan"], fill="", width=2)
        mark.create_line(17, 4, 17, 21, fill=COLORS["cyan"], width=2)
        mark.create_line(4, 28, 17, 21, 30, 28, fill=COLORS["cyan"], width=2)

        brand = tk.Frame(topbar, bg=COLORS["topbar"])
        brand.pack(side="left")
        tk.Label(
            brand,
            text="TETRA LAB",
            bg=COLORS["topbar"],
            fg=COLORS["text"],
            font=(self.font_family, 13, "bold"),
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="REALISTIC TETRAHEDRON CONTROL",
            bg=COLORS["topbar"],
            fg=COLORS["muted"],
            font=(self.font_family, 8),
        ).pack(anchor="w")

        status = tk.Frame(topbar, bg=COLORS["topbar"])
        status.pack(side="right", padx=20)
        self.status_dot = tk.Canvas(
            status, width=14, height=14, bg=COLORS["topbar"], highlightthickness=0
        )
        self.status_dot.pack(side="left", padx=(0, 5))
        self.status_dot.create_oval(3, 3, 11, 11, fill=COLORS["green"], outline="")
        self.status_label = tk.Label(
            status,
            text="LIVE",
            bg=COLORS["topbar"],
            fg=COLORS["green"],
            font=(self.font_family, 9, "bold"),
        )
        self.status_label.pack(side="left")

    def _build_control_panel(self, panel: Any) -> None:
        tk = self.tk
        content = tk.Frame(panel, bg=COLORS["surface"], padx=16, pady=15)
        content.pack(fill="both", expand=True)

        self._section_heading(
            content,
            "NODE COMMANDS",
            f"RANGE +/- {self.simulation.speed:.3f} m/s",
        )
        self.node_sliders: list[BipolarSlider] = []
        for index, _node_name in enumerate(self.simulation.logical_node_names, start=1):
            slider = BipolarSlider(
                content,
                tk,
                f"Node {index:02d}",
                self.simulation.speed,
                self.font_family,
                self._on_slider_changed,
            )
            slider.pack(fill="x", pady=(0, 7))
            self.node_sliders.append(slider)

        buttons = tk.Frame(content, bg=COLORS["surface"])
        buttons.pack(fill="x", pady=(3, 14))
        self.pause_button = self._button(buttons, "Pause", self.toggle_running, primary=True)
        self.pause_button.pack(side="left", fill="x", expand=True)
        self._button(buttons, "Zero all", self.zero_commands).pack(
            side="left", fill="x", expand=True, padx=7
        )
        self._button(buttons, "Reset", self.reset).pack(side="left", fill="x", expand=True)

        self._section_heading(content, "TELEMETRY", "LIVE STATE")
        metrics = tk.Frame(content, bg=COLORS["surface"])
        metrics.pack(fill="x", pady=(0, 14))
        self.time_var = tk.StringVar(value="0.000 s")
        self.com_var = tk.StringVar(value="+0.000  +0.000  +0.000")
        self.wall_rate_var = tk.StringVar(value="0.00x")
        self._metric(metrics, "SIM TIME", self.time_var).pack(side="left", fill="x", expand=True)
        self._metric(metrics, "CENTER OF MASS", self.com_var, wide=True).pack(
            side="left", fill="x", expand=True, padx=7
        )
        self._metric(metrics, "REAL TIME", self.wall_rate_var).pack(
            side="left", fill="x", expand=True
        )
        self._build_rigidity_readout(content)

        self._section_heading(content, "EDGE COMMANDS", "ROUTED ACTUATORS")
        self.edge_rows: list[tuple[Any, Any]] = []
        edge_box = tk.Frame(content, bg=COLORS["surface_alt"], padx=10, pady=7)
        edge_box.pack(fill="both", expand=True)
        for index, edge in enumerate(self.simulation.controller.edges, start=1):
            row = tk.Frame(edge_box, bg=COLORS["surface_alt"], height=22)
            row.pack(fill="x")
            row.pack_propagate(False)
            tk.Label(
                row,
                text=f"E{index:02d}",
                width=4,
                anchor="w",
                bg=COLORS["surface_alt"],
                fg=COLORS["muted"],
                font=(self.font_family, 9, "bold"),
            ).pack(side="left")
            tk.Label(
                row,
                text=(
                    f"{logical_node_name(edge.from_node).replace('node_', 'N')}  ->  "
                    f"{logical_node_name(edge.to_node).replace('node_', 'N')}"
                ),
                width=11,
                anchor="w",
                bg=COLORS["surface_alt"],
                fg=COLORS["text"],
                font=("Menlo", 9),
            ).pack(side="left")
            bar = tk.Canvas(
                row,
                height=16,
                bg=COLORS["surface_alt"],
                highlightthickness=0,
            )
            bar.pack(side="left", fill="x", expand=True, padx=5)
            value = tk.Label(
                row,
                text="+0.0000",
                width=8,
                anchor="e",
                bg=COLORS["surface_alt"],
                fg=COLORS["cyan"],
                font=("Menlo", 9),
            )
            value.pack(side="right")
            self.edge_rows.append((bar, value))

    def _build_rigidity_readout(self, parent: Any) -> None:
        tk = self.tk
        frame = tk.Frame(parent, bg=COLORS["surface_alt"], padx=10, pady=5)
        frame.pack(fill="x", pady=(0, 8))

        header = tk.Frame(frame, bg=COLORS["surface_alt"])
        header.pack(fill="x")
        tk.Label(
            header,
            text="RIGIDITY CONSTRAINT",
            bg=COLORS["surface_alt"],
            fg=COLORS["faint"],
            font=(self.font_family, 7, "bold"),
        ).pack(side="left")
        self.rigidity_var = tk.StringVar(value="100.0%")
        self.rigidity_value_label = tk.Label(
            header,
            textvariable=self.rigidity_var,
            bg=COLORS["surface_alt"],
            fg=COLORS["green"],
            font=("Menlo", 9, "bold"),
        )
        self.rigidity_value_label.pack(side="right")

        self.rigidity_canvas = tk.Canvas(
            frame,
            height=12,
            bg=COLORS["surface_alt"],
            highlightthickness=0,
        )
        self.rigidity_canvas.pack(fill="x", pady=(5, 0))

    def _section_heading(self, parent: Any, title: str, detail: str) -> None:
        tk = self.tk
        row = tk.Frame(parent, bg=COLORS["surface"])
        row.pack(fill="x", pady=(0, 8))
        tk.Label(
            row,
            text=title,
            bg=COLORS["surface"],
            fg=COLORS["text"],
            font=(self.font_family, 10, "bold"),
        ).pack(side="left")
        tk.Label(
            row,
            text=detail,
            bg=COLORS["surface"],
            fg=COLORS["faint"],
            font=(self.font_family, 8),
        ).pack(side="right")

    def _button(
        self,
        parent: Any,
        text: str,
        command: Any,
        primary: bool = False,
    ) -> CanvasButton:
        return CanvasButton(
            parent,
            self.tk,
            text=text,
            command=command,
            font_family=self.font_family,
            primary=primary,
        )

    def _metric(self, parent: Any, title: str, variable: Any, wide: bool = False) -> Any:
        frame = self.tk.Frame(parent, bg=COLORS["surface_alt"], padx=9, pady=8)
        self.tk.Label(
            frame,
            text=title,
            bg=COLORS["surface_alt"],
            fg=COLORS["faint"],
            font=(self.font_family, 7, "bold"),
        ).pack(anchor="w")
        self.tk.Label(
            frame,
            textvariable=variable,
            bg=COLORS["surface_alt"],
            fg=COLORS["text"],
            font=("Menlo", 8 if wide else 10),
        ).pack(anchor="w", pady=(3, 0))
        return frame

    def _bind_camera_controls(self) -> None:
        self.image_label.bind("<ButtonPress-1>", self._start_drag)
        self.image_label.bind("<B1-Motion>", self._drag_camera)
        self.image_label.bind("<ButtonRelease-1>", self._stop_drag)
        self.image_label.bind("<MouseWheel>", self._zoom_camera)
        self.image_label.bind("<Button-4>", lambda _event: self._zoom(-1.0))
        self.image_label.bind("<Button-5>", lambda _event: self._zoom(1.0))

    def _start_drag(self, event: Any) -> None:
        self.drag_position = (event.x, event.y)

    def _drag_camera(self, event: Any) -> None:
        if self.drag_position is None:
            return
        previous_x, previous_y = self.drag_position
        self.camera.azimuth -= (event.x - previous_x) * 0.35
        self.camera.elevation = float(
            np.clip(self.camera.elevation - (event.y - previous_y) * 0.25, -89.0, 20.0)
        )
        self.drag_position = (event.x, event.y)

    def _stop_drag(self, _event: Any) -> None:
        self.drag_position = None

    def _zoom_camera(self, event: Any) -> None:
        self._zoom(-float(event.delta) / 120.0)

    def _zoom(self, direction: float) -> None:
        self.camera.distance = float(
            np.clip(self.camera.distance * (1.0 + 0.08 * direction), 0.7, 12.0)
        )

    def _on_slider_changed(self) -> None:
        self.control_parameters[:] = [slider.get() for slider in self.node_sliders]

    def _update_parameterized_control(self) -> None:
        self.current_critical_eigenvalue = self.simulation.rigidity_state()[0]
        self.eigenvalue_gradient = self.simulation.eigenvalue_gradient(
            self.simulation.logical_commands,
            rollout_steps=self.steps_per_frame,
        )

        gradient_norm = np.linalg.norm(self.eigenvalue_gradient)
        if gradient_norm <= 1e-12:
            return

        s = self.control_parameters[0]
        w = self.control_parameters[1:]
        perpendicular_basis = null_space(self.eigenvalue_gradient.reshape(1, -1))
        u_perp = perpendicular_basis @ w
        kappa = 0.0
        u_parallel = (
            (s - kappa * self.current_critical_eigenvalue)
            * self.eigenvalue_gradient
            / gradient_norm
        )
        self.simulation.set_logical_commands(u_parallel + u_perp)

    def softplus(self, x):
        return np.log1p(np.exp(x))

    def zero_commands(self) -> None:
        for slider in self.node_sliders:
            slider.set(0.0, notify=False)
        self._on_slider_changed()

    def reset(self) -> None:
        self.zero_commands()
        self.simulation.reset()
        self.simulation_frame = 0
        self.camera.lookat[:] = self.simulation.center_of_mass

    def toggle_running(self) -> None:
        self.running = not self.running
        self.pause_button.configure(text="Pause" if self.running else "Resume")
        color = COLORS["green"] if self.running else COLORS["amber"]
        self.status_dot.delete("all")
        self.status_dot.create_oval(3, 3, 11, 11, fill=color, outline="")
        self.status_label.configure(
            text="LIVE" if self.running else "PAUSED",
            fg=color,
        )
        self.last_wall_time = time.perf_counter()

    def _render_frame(self) -> None:
        if self.closed:
            return

        frame_start = time.perf_counter()
        if self.running:
            if self.simulation_frame % GRADIENT_UPDATE_INTERVAL_FRAMES == 0:
                self._update_parameterized_control()
            self.simulation.step(self.steps_per_frame)
            self.simulation_frame += 1

        self.renderer.update_scene(self.simulation.data, camera=self.camera)
        self._add_node_labels()
        pixels = np.ascontiguousarray(self.renderer.render())
        ppm = f"P6 {self.width} {self.height} 255\n".encode("ascii") + pixels.tobytes()
        self.photo = self.tk.PhotoImage(data=ppm, format="PPM")
        self.image_label.configure(image=self.photo)
        self._update_readouts(frame_start)
        elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
        self.root.after(max(1, round(self.frame_period_ms - elapsed_ms)), self._render_frame)

    def _add_node_labels(self) -> None:
        scene = self.renderer.scene
        positions = self.simulation.logical_node_positions()
        label_offset = np.array([0.0, 0.0, 0.14], dtype=float)
        identity = np.eye(3, dtype=float).ravel()
        size = np.array([0.025, 0.025, 0.025], dtype=float)
        color = np.array([0.08, 0.08, 0.08, 1.0], dtype=np.float32)

        for index, node_name in enumerate(self.simulation.logical_node_names, start=1):
            if node_name not in positions or scene.ngeom >= scene.maxgeom:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_LABEL,
                size,
                positions[node_name] + label_offset,
                identity,
                color,
            )
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.label = f"Node {index:02d}"
            scene.ngeom += 1

    def _update_readouts(self, frame_start: float) -> None:
        com = self.simulation.center_of_mass
        wall_dt = max(frame_start - self.last_wall_time, 1e-9)
        simulated_dt = (
            self.steps_per_frame * self.simulation.model.opt.timestep if self.running else 0.0
        )
        self.last_wall_time = frame_start
        self.time_var.set(f"{self.simulation.data.time:.3f} s")
        self.com_var.set(f"{com[0]:+.2f} {com[1]:+.2f} {com[2]:+.2f}")
        self.wall_rate_var.set(f"{simulated_dt / wall_dt:.2f}x")
        critical_eigenvalue, rigidity_ratio = self.simulation.rigidity_state()
        self._update_rigidity_readout(critical_eigenvalue, rigidity_ratio)
        for (bar, label), value in zip(
            self.edge_rows,
            self.simulation.edge_commands,
            strict=True,
        ):
            self._draw_edge_bar(bar, float(value))
            label.configure(
                text=f"{value:+.4f}",
                fg=COLORS["amber"] if value > 0.0 else COLORS["cyan"],
            )

    def _update_rigidity_readout(
        self,
        critical_eigenvalue: float,
        rigidity_ratio: float,
    ) -> None:
        if not np.isfinite(rigidity_ratio):
            rigidity_ratio = 0.0
        if rigidity_ratio < RIGIDITY_THRESHOLD:
            color = COLORS["danger"]
            state = "CRITICAL"
        elif rigidity_ratio < 0.15:
            color = COLORS["amber"]
            state = "LOW"
        else:
            color = COLORS["green"]
            state = "NOMINAL"

        self.rigidity_var.set(
            f"{state}  {100.0 * rigidity_ratio:6.1f}%  lambda={critical_eigenvalue:.4f}"
        )
        self.rigidity_value_label.configure(fg=color)

        canvas = self.rigidity_canvas
        width = max(canvas.winfo_width(), 30)
        fill_width = np.clip(rigidity_ratio, 0.0, 1.0) * (width - 2.0)
        threshold_x = RIGIDITY_THRESHOLD * (width - 2.0) + 1.0
        canvas.delete("all")
        canvas.create_rectangle(1, 3, width - 1, 9, fill=COLORS["track"], outline="")
        canvas.create_rectangle(1, 3, fill_width + 1, 9, fill=color, outline="")
        canvas.create_line(
            threshold_x,
            1,
            threshold_x,
            11,
            fill=COLORS["text"],
            width=1,
        )

    def _draw_edge_bar(self, canvas: Any, value: float) -> None:
        width = max(canvas.winfo_width(), 30)
        center = width / 2.0
        y = 8.0
        max_command = self.simulation.speed * 2.0
        value_x = center + np.clip(value / max_command, -1.0, 1.0) * (center - 4.0)
        canvas.delete("all")
        canvas.create_line(3, y, width - 3, y, fill=COLORS["track"], width=3)
        canvas.create_line(center, y - 5, center, y + 5, fill=COLORS["border_bright"])
        if abs(value) > 1e-10:
            canvas.create_line(
                center,
                y,
                value_x,
                y,
                fill=COLORS["amber"] if value > 0.0 else COLORS["cyan"],
                width=3,
            )

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.renderer.close()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control and render a realistic routed-tube tetrahedron.",
    )
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Slider limit in m/s.")
    parser.add_argument("--scale", type=float, default=1.0, help="Tetrahedron model scale.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="GUI render rate.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Render width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Render height.")
    parser.add_argument(
        "--headless-steps",
        type=int,
        default=0,
        help="Run this many physics steps without opening the GUI (useful for validation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation = TetrahedronSimulation(speed=args.speed, scale=args.scale)
    if args.headless_steps:
        simulation.step(args.headless_steps)
        print(
            f"Simulated {args.headless_steps} steps; "
            f"time={simulation.data.time:.4f}s; nodes={simulation.logical_node_names}"
        )
        return

    gui = TetrahedronControlGUI(
        simulation,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    gui.run()


if __name__ == "__main__":
    main()
