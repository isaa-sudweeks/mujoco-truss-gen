from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from typing import Any

import mujoco
import numpy as np

from mujoco_truss_gen.mujoco_model.controllers import (
    AngleBisectorController,
    NodeVelocityController,
)
from mujoco_truss_gen.mujoco_model.model import ModelSource, MujocoModel
from mujoco_truss_gen.mujoco_model.tendons import initialize_actuator_lengths


@dataclass(slots=True)
class NodeVelocityViewerState:
    """Node slider state and tendon command readouts for routed-tube viewing."""

    node_names: list[str]
    edge_names: list[str]
    passive_node_names: list[str] = field(default_factory=list)
    speed: float = 0.01
    node_commands: np.ndarray = field(init=False)
    edge_commands: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.node_commands = np.zeros(len(self.node_names), dtype=float)
        self.edge_commands = np.zeros(len(self.edge_names), dtype=float)

    def set_node_command(self, node_name: str, value: float) -> None:
        index = self.node_names.index(node_name)
        if node_name in self.passive_node_names:
            value = 0.0
        self.node_commands[index] = float(np.clip(value, -self.speed, self.speed))

    def set_edge_commands(self, edge_commands: np.ndarray) -> None:
        edge_commands = np.asarray(edge_commands, dtype=float)
        if edge_commands.shape != (len(self.edge_names),):
            raise ValueError(
                f"Expected {len(self.edge_names)} edge command(s), got shape "
                f"{edge_commands.shape}."
        )
        self.edge_commands = edge_commands.copy()


class NodeVelocitySliderPanel:
    """Small Tk panel with node command sliders and tendon command labels."""

    def __init__(self, state: NodeVelocityViewerState):
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:
            raise RuntimeError(
                "The routed-tube node velocity viewer requires tkinter for its "
                "slider panel."
            ) from exc

        self._tk = tk
        self._closed = False
        self.state = state
        self.root = tk.Tk()
        self.root.title("Routed Tube Node Control")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(container, text="Node actions").grid(row=0, column=0, sticky="w")
        ttk.Label(container, text="Reported tendon actions").grid(row=0, column=1, sticky="w")

        self.node_vars: dict[str, Any] = {}
        node_frame = ttk.Frame(container)
        node_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        for row, node_name in enumerate(state.node_names):
            var = tk.DoubleVar(value=0.0)
            self.node_vars[node_name] = var
            ttk.Label(node_frame, text=node_name).grid(row=row, column=0, sticky="w")
            scale = ttk.Scale(
                node_frame,
                from_=-state.speed,
                to=state.speed,
                variable=var,
                orient="horizontal",
                length=260,
            )
            scale.grid(row=row, column=1, sticky="ew", padx=8)
            if node_name in state.passive_node_names:
                scale.state(["disabled"])
            node_frame.columnconfigure(1, weight=1)

        self.edge_vars: dict[str, Any] = {}
        edge_frame = ttk.Frame(container)
        edge_frame.grid(row=1, column=1, sticky="nsew")
        for row, edge_name in enumerate(state.edge_names):
            value = tk.StringVar(value="0.0000")
            self.edge_vars[edge_name] = value
            ttk.Label(edge_frame, text=edge_name).grid(row=row, column=0, sticky="w")
            ttk.Label(edge_frame, textvariable=value, width=12, anchor="e").grid(
                row=row,
                column=1,
                sticky="e",
                padx=(8, 0),
            )

        reset = ttk.Button(container, text="Zero all", command=self.zero_all)
        reset.grid(row=2, column=0, sticky="w", pady=(12, 0))

    @property
    def is_running(self) -> bool:
        return not self._closed

    def read_node_commands(self) -> np.ndarray:
        for node_name, var in self.node_vars.items():
            self.state.set_node_command(node_name, float(var.get()))
        return self.state.node_commands.copy()

    def set_edge_commands(self, edge_commands: np.ndarray) -> None:
        self.state.set_edge_commands(edge_commands)
        for edge_name, value in zip(
            self.state.edge_names,
            self.state.edge_commands,
            strict=False,
        ):
            self.edge_vars[edge_name].set(f"{value:.4f}")

    def update(self) -> None:
        if self._closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except self._tk.TclError:
            self._closed = True

    def zero_all(self) -> None:
        for var in self.node_vars.values():
            var.set(0.0)

    def close(self) -> None:
        self._closed = True
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass


def view(spec: mujoco.MjSpec, *, node_controls: bool = False) -> None:
    """Compile and view the MuJoCo spec."""
    if node_controls:
        view_node_velocity(spec)
        return

    try:
        import mujoco.viewer as mujoco_viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo passive viewer is unavailable in this Python environment. "
            "Install a MuJoCo build that includes the viewer module, and on macOS "
            "run viewer scripts with mjpython."
        ) from exc

    model = spec.compile()
    if hasattr(model, "model") and hasattr(model, "data"):
        mj_model = model.model
        data = model.data
    elif isinstance(model, mujoco.MjModel):
        mj_model = model
        data = mujoco.MjData(mj_model)
    else:
        raise TypeError(
            "view() expects a mujoco.MjModel or an object with 'model' and 'data' attributes."
        )

    controller = AngleBisectorController(mj_model, spec.to_xml())
    initialize_actuator_lengths(mj_model, data)
    controller.update(mj_model, data)
    with mujoco_viewer.launch_passive(mj_model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            if data.time == 0.0:
                initialize_actuator_lengths(mj_model, data)
            controller.update(mj_model, data)
            mujoco.mj_step(mj_model, data)
            viewer.sync()
            time.sleep(max(mj_model.opt.timestep, 0.001))


def view_node_velocity(source: ModelSource, *, speed: float = 0.01) -> None:
    """View a routed continuous-tube model with node sliders and tendon readouts."""
    if _running_on_macos():
        raise RuntimeError(
            "The routed-tube node velocity slider panel uses tkinter, which is not "
            "compatible with this macOS/Tk build. Use view(spec) for the standard "
            "MuJoCo viewer."
        )

    mj_model = MujocoModel(source)
    controller = NodeVelocityController(
        mj_model.model,
        mj_model.xml,
        mj_model.node_names,
        mj_model.site_to_node,
        mj_model.external_actuator_ids,
    )
    if not controller.enabled:
        raise ValueError(
            "view_node_velocity() requires a routed continuous-tube model with route "
            "tendons and edge actuators."
        )

    state = NodeVelocityViewerState(
        controller.node_names,
        controller.edge_names,
        controller.passive_node_names,
        speed=speed,
    )
    panel = NodeVelocitySliderPanel(state)

    try:
        import mujoco.viewer as mujoco_viewer
    except ImportError as exc:
        panel.close()
        raise RuntimeError(
            "MuJoCo passive viewer is unavailable in this Python environment. "
            "Install a MuJoCo build that includes the viewer module, and on macOS "
            "run viewer scripts with mjpython."
        ) from exc

    initialize_actuator_lengths(mj_model.model, mj_model.data)
    try:
        with mujoco_viewer.launch_passive(
            mj_model.model,
            mj_model.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.sync()
            while viewer.is_running() and panel.is_running:
                if mj_model.data.time == 0.0:
                    initialize_actuator_lengths(mj_model.model, mj_model.data)
                node_commands = panel.read_node_commands()
                edge_commands = controller.apply(mj_model.model, mj_model.data, node_commands)
                panel.set_edge_commands(edge_commands)
                panel.update()
                mj_model.apply_angle_bisector_control()
                mujoco.mj_step(mj_model.model, mj_model.data)
                viewer.sync()
                time.sleep(max(mj_model.model.opt.timestep, 0.001))
    finally:
        panel.close()


def view_node_velocity_terminal(source: ModelSource, *, speed: float = 0.01) -> None:
    """View a routed continuous-tube model and accept node commands from stdin."""
    try:
        import mujoco.viewer as mujoco_viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo passive viewer is unavailable in this Python environment. "
            "Install a MuJoCo build that includes the viewer module, and on macOS "
            "run viewer scripts with mjpython."
        ) from exc

    mj_model = MujocoModel(source)
    controller = NodeVelocityController(
        mj_model.model,
        mj_model.xml,
        mj_model.node_names,
        mj_model.site_to_node,
        mj_model.external_actuator_ids,
    )
    if not controller.enabled:
        raise ValueError(
            "view_node_velocity_terminal() requires a routed continuous-tube model "
            "with route tendons and edge actuators."
        )

    node_commands = np.zeros(len(controller.node_names), dtype=float)
    command_queue: SimpleQueue[str] = SimpleQueue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=_read_terminal_commands,
        args=(command_queue, stop_event),
        daemon=True,
    )

    _print_terminal_help(controller, speed)
    input_thread.start()
    initialize_actuator_lengths(mj_model.model, mj_model.data)

    try:
        with mujoco_viewer.launch_passive(mj_model.model, mj_model.data) as viewer:
            viewer.sync()
            while viewer.is_running() and not stop_event.is_set():
                while not command_queue.empty():
                    should_quit = _apply_terminal_command(
                        command_queue.get(),
                        controller,
                        node_commands,
                        speed,
                    )
                    if should_quit:
                        stop_event.set()
                        break

                if mj_model.data.time == 0.0:
                    initialize_actuator_lengths(mj_model.model, mj_model.data)
                controller.apply(mj_model.model, mj_model.data, node_commands)
                mj_model.apply_angle_bisector_control()
                mujoco.mj_step(mj_model.model, mj_model.data)
                viewer.sync()
                time.sleep(max(mj_model.model.opt.timestep, 0.001))
    finally:
        stop_event.set()


def save_xml(spec: mujoco.MjSpec, filename: str | Path) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path


def _read_terminal_commands(command_queue: SimpleQueue[str], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            command_queue.put(input("node> "))
        except EOFError:
            command_queue.put("quit")
            return


def _apply_terminal_command(
    raw_command: str,
    controller: NodeVelocityController,
    node_commands: np.ndarray,
    speed: float,
) -> bool:
    tokens = raw_command.strip().split()
    if not tokens:
        return False

    command = tokens[0].lower()
    if command in {"q", "quit", "exit"}:
        return True

    if command in {"h", "help", "?"}:
        _print_terminal_help(controller, speed)
        return False

    if command in {"nodes", "list"}:
        _print_terminal_nodes(controller)
        return False

    if command == "show":
        _print_terminal_state(controller, node_commands)
        return False

    if command in {"z", "zero", "reset"}:
        node_commands.fill(0.0)
        print("All node commands set to 0.")
        return False

    if command == "set":
        if len(tokens) != 3:
            print("Usage: set <node-name-or-index> <value>")
            return False
        _set_terminal_node_command(tokens[1], tokens[2], controller, node_commands, speed)
        return False

    if command == "add":
        if len(tokens) != 3:
            print("Usage: add <node-name-or-index> <delta>")
            return False
        _add_terminal_node_command(tokens[1], tokens[2], controller, node_commands, speed)
        return False

    if len(tokens) == 2:
        _set_terminal_node_command(tokens[0], tokens[1], controller, node_commands, speed)
        return False

    print(f"Unknown command: {raw_command!r}. Type 'help' for commands.")
    return False


def _set_terminal_node_command(
    node_token: str,
    value_token: str,
    controller: NodeVelocityController,
    node_commands: np.ndarray,
    speed: float,
) -> None:
    node_index = _terminal_node_index(node_token, controller)
    if node_index is None:
        return
    value = _terminal_float(value_token)
    if value is None:
        return
    node_commands[node_index] = _clip_terminal_node_command(
        controller.node_names[node_index],
        value,
        controller,
        speed,
    )
    _print_terminal_state(controller, node_commands)


def _add_terminal_node_command(
    node_token: str,
    value_token: str,
    controller: NodeVelocityController,
    node_commands: np.ndarray,
    speed: float,
) -> None:
    node_index = _terminal_node_index(node_token, controller)
    if node_index is None:
        return
    value = _terminal_float(value_token)
    if value is None:
        return
    node_name = controller.node_names[node_index]
    node_commands[node_index] = _clip_terminal_node_command(
        node_name,
        node_commands[node_index] + value,
        controller,
        speed,
    )
    _print_terminal_state(controller, node_commands)


def _terminal_node_index(
    node_token: str,
    controller: NodeVelocityController,
) -> int | None:
    if node_token in controller.node_index:
        return controller.node_index[node_token]

    try:
        node_index = int(node_token)
    except ValueError:
        print(f"Unknown node: {node_token!r}. Type 'nodes' to list valid nodes.")
        return None

    if not 0 <= node_index < len(controller.node_names):
        print(f"Node index {node_index} is out of range.")
        return None
    return node_index


def _terminal_float(value_token: str) -> float | None:
    try:
        return float(value_token)
    except ValueError:
        print(f"Expected a numeric value, got {value_token!r}.")
        return None


def _clip_terminal_node_command(
    node_name: str,
    value: float,
    controller: NodeVelocityController,
    speed: float,
) -> float:
    if node_name in controller.passive_node_names:
        print(f"{node_name} is a passive route endpoint; keeping command at 0.")
        return 0.0
    return float(np.clip(value, -speed, speed))


def _print_terminal_help(controller: NodeVelocityController, speed: float) -> None:
    print(
        "\nTerminal node control commands:\n"
        "  set <node> <value>   set a node command, e.g. set node_2 0.01\n"
        "  <node> <value>       shorthand for set\n"
        "  add <node> <delta>   increment a node command\n"
        "  zero                 reset all node commands to 0\n"
        "  show                 print node and tendon commands\n"
        "  nodes                list controllable nodes\n"
        "  quit                 close the control loop\n"
        f"Values are clipped to +/-{speed:g}.\n"
    )
    _print_terminal_nodes(controller)


def _print_terminal_nodes(controller: NodeVelocityController) -> None:
    print("Nodes:")
    for index, node_name in enumerate(controller.node_names):
        suffix = " passive" if node_name in controller.passive_node_names else ""
        print(f"  {index}: {node_name}{suffix}")


def _print_terminal_state(
    controller: NodeVelocityController,
    node_commands: np.ndarray,
) -> None:
    edge_commands = controller.transform(node_commands)
    print("Node commands:")
    for node_name, value in zip(controller.node_names, node_commands, strict=False):
        print(f"  {node_name}: {value:.4f}")
    print("Tendon commands:")
    for edge_name, value in zip(controller.edge_names, edge_commands, strict=False):
        print(f"  {edge_name}: {value:.4f}")


def _running_on_macos() -> bool:
    return sys.platform == "darwin"
