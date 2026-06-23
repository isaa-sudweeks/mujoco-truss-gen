"""Interactively position-command a realistic truss preset in MuJoCo.

Run on macOS with:

    .venv/bin/mjpython experiments/system_id/interactive_octahedron_position_control.py
    .venv/bin/mjpython experiments/system_id/interactive_octahedron_position_control.py octahedron
    .venv/bin/mjpython experiments/system_id/interactive_octahedron_position_control.py tetrahedron

Enter a Python/JSON array at the ``position>`` prompt. Each entry controls one
non-passive control node; the array-to-node mapping is printed at startup and
shown as labels in the viewer. Values are relative scalar node-position offsets
in meters. Passive control nodes remain at zero, and node commands are mapped to
tendon commands with the same oriented incidence rule used by the routed-tube
node controller.
"""

from __future__ import annotations

import argparse
import ast
import threading
import time
from dataclasses import dataclass
from queue import Empty, SimpleQueue

import mujoco
import numpy as np

from mujoco_truss_gen import (
    PRESETS,
    MujocoVelocityCommandEnv,
    NodeVelocityController,
    TrussEnvConfig,
    TrussPhysicalParameters,
    get_mujoco_spec,
)
from mujoco_truss_gen.mujoco_model.tendons import initialize_actuator_lengths

# -----------------------------------------------------------------------------
# Editable experiment parameters
# -----------------------------------------------------------------------------

# Here is what I got to work -15000,0,-4000,-8000,0,15000,8000,4000

DEFAULT_ROBOT = "tetrahedron"
ROBOT_SCALE = 1.2
CONNECTOR_ROD_LENGTH = 0.22289

PHYSICAL_PARAMS = TrussPhysicalParameters(
    active_node_mass=1.98,
    passive_node_mass=1.0,
    connector_radius=0.05,
    connector_mass=0.1,
    rod_radius=0.025,
    rod_mass=0.426655,
    triangle_body_mass=0.01,
    triangle_body_gravcomp=1.0,
    hinge_position_kp=100.0,
    hinge_damping=10.0,
    realistic_actuator_kp=10000.0,
    actuator_dampratio=1.0,
    realistic_actuator_nominal_mass=1.98,
    actuator_ctrl_range=[-0.05, 0.05],
    default_actuator_range=[0.0, 3.0],
    edge_tendon_width=0.05,
    realistic_node_clone_offset=0.5,
    connector_rod_length=CONNECTOR_ROD_LENGTH,
)

TIMESTEP = 0.002
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)
GEOM_FRICTION = np.array([1.0, 0.005, 0.0001], dtype=float)

# Position error [m] -> intvelocity input [m/s].
POSITION_GAIN = 4.0
# Multiplies every value entered in a terminal position array.
POSITION_COMMAND_SCALE = 0.0254 / 1125
MAX_NODE_POSITION_OFFSET = 1.0
STEPS_PER_CONTROL_UPDATE = 5
RESET_SEED = 0


@dataclass(frozen=True, slots=True)
class ActuatorEdge:
    """One intvelocity actuator and its oriented control-graph endpoints."""

    actuator_id: int
    activation_id: int
    tendon_name: str
    from_node: str
    to_node: str


@dataclass(frozen=True, slots=True)
class ControlTopology:
    """Control nodes and actuator edges for the selected robot."""

    node_names: list[str]
    command_node_names: list[str]
    passive_node_names: list[str]
    node_to_physical_node: dict[str, str]
    node_to_logical_node: dict[str, str]
    edges: list[ActuatorEdge]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively position-command a realistic truss preset."
    )
    parser.add_argument(
        "robot",
        nargs="?",
        default=DEFAULT_ROBOT,
        choices=sorted(PRESETS),
        help=f"Preset robot configuration to load. Defaults to {DEFAULT_ROBOT!r}.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=ROBOT_SCALE,
        help=f"Preset scale passed to get_mujoco_spec(). Defaults to {ROBOT_SCALE:g}.",
    )
    return parser.parse_args()


def build_environment(robot: str, scale: float) -> MujocoVelocityCommandEnv:
    spec = get_mujoco_spec(
        robot,
        realistic=True,
        scale=scale,
        physical_params=PHYSICAL_PARAMS,
    )
    env = MujocoVelocityCommandEnv(
        TrussEnvConfig(
            model_source=spec,
            max_steps=10**9,
            nsubsteps=STEPS_PER_CONTROL_UPDATE,
            speed=max(abs(value) for value in PHYSICAL_PARAMS.actuator_ctrl_range),
            critical_eig_threshold=0.0,
        ),
        render_mode="human",
    )
    env.mj_model.model.opt.timestep = TIMESTEP
    env.mj_model.model.opt.gravity[:] = GRAVITY
    env.mj_model.model.geom_friction[:] = GEOM_FRICTION
    return env


def control_topology(env: MujocoVelocityCommandEnv) -> ControlTopology:
    """Read control-node and actuator orientation metadata from the generated model."""
    model = env.mj_model.model
    controller = NodeVelocityController(
        model,
        env.mj_model.xml,
        env.mj_model.node_names,
        env.mj_model.site_to_node,
        env.mj_model.external_actuator_ids,
    )
    if not controller.enabled:
        raise RuntimeError("The selected model does not expose a node velocity control graph.")

    edges = []
    for edge in controller.edges:
        actuator_id = int(edge.actuator_id)
        if model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_INTEGRATOR:
            raise RuntimeError(
                f"External actuator {model.actuator(actuator_id).name!r} is not intvelocity."
            )

        tendon_id = int(model.actuator_trnid[actuator_id, 0])
        tendon_name = model.tendon(tendon_id).name
        activation_id = int(model.actuator_actadr[actuator_id])
        if activation_id < 0:
            raise RuntimeError(f"Actuator {model.actuator(actuator_id).name!r} has no state.")

        edges.append(
            ActuatorEdge(
                actuator_id,
                activation_id,
                tendon_name,
                edge.from_node,
                edge.to_node,
            )
        )

    external_actuator_count = len(env.mj_model.external_actuator_ids)
    if len(edges) != external_actuator_count:
        raise RuntimeError(
            f"Expected one control edge per external actuator, got {len(edges)} edge(s) "
            f"for {external_actuator_count} external actuator(s)."
        )

    passive_node_names = list(controller.passive_node_names)
    passive_nodes = set(passive_node_names)
    command_node_names = [
        node_name for node_name in controller.node_names if node_name not in passive_nodes
    ]
    if not command_node_names:
        raise RuntimeError("The selected model has no non-passive control nodes to command.")

    graph = env.mj_model.control_graph
    return ControlTopology(
        node_names=list(controller.node_names),
        command_node_names=command_node_names,
        passive_node_names=passive_node_names,
        node_to_physical_node={
            node_name: graph.control_node_to_physical_node.get(node_name, node_name)
            for node_name in controller.node_names
        },
        node_to_logical_node={
            node_name: graph.control_node_to_logical_node.get(node_name, node_name)
            for node_name in controller.node_names
        },
        edges=edges,
    )


def incidence_matrix(
    node_names: list[str],
    edges: list[ActuatorEdge],
) -> np.ndarray:
    node_index = {node_name: index for index, node_name in enumerate(node_names)}
    matrix = np.zeros((len(edges), len(node_names)), dtype=float)
    for row, edge in enumerate(edges):
        if edge.from_node in node_index:
            matrix[row, node_index[edge.from_node]] = -1.0
        if edge.to_node in node_index:
            matrix[row, node_index[edge.to_node]] = 1.0
    return matrix


def parse_position_command(raw_command: str, node_count: int) -> np.ndarray | str:
    command = raw_command.strip()
    keyword = command.lower()
    if keyword in {"help", "show", "zero", "reset", "quit", "exit", "q"}:
        return keyword

    try:
        values = np.asarray(ast.literal_eval(command), dtype=float)
    except (SyntaxError, ValueError, TypeError):
        example = [0.0] * node_count
        raise ValueError(f"Enter a numeric array such as {example}.") from None

    if values.shape != (node_count,):
        raise ValueError(f"Expected an array with {node_count} values, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("All position values must be finite.")
    return values


def print_help(topology: ControlTopology) -> None:
    print("\nCommands:")
    print(
        "  [p0, p1, ...]  set the "
        f"{len(topology.command_node_names)} non-passive control-node positions"
    )
    print("  show            show the current position target")
    print("  zero            set every position target to zero")
    print("  reset           reset the simulation and zero the target")
    print("  help            show this message")
    print("  quit            close the viewer")
    print("Array mapping:")
    for index, node_name in enumerate(topology.command_node_names):
        logical_node = topology.node_to_logical_node[node_name]
        physical_node = topology.node_to_physical_node[node_name]
        logical_suffix = "" if logical_node == node_name else f", logical={logical_node}"
        physical_suffix = "" if physical_node == node_name else f", physical={physical_node}"
        print(f"  index {index}: {node_name}{logical_suffix}{physical_suffix}")
    if topology.passive_node_names:
        print("Passive control nodes:")
        for node_name in topology.passive_node_names:
            print(f"  {node_name}")
    print("Actuator edge mapping:")
    for edge in topology.edges:
        print(f"  {edge.tendon_name}: {edge.to_node} - {edge.from_node}")
    print(
        "Passive control nodes are fixed at command zero. Tendon offsets use "
        "node[to] - node[from], matching the routed-tube node controller."
    )
    print(f"Typed values are multiplied by POSITION_COMMAND_SCALE={POSITION_COMMAND_SCALE:g}.")
    print(f"Effective targets are clipped to +/-{MAX_NODE_POSITION_OFFSET:g} m.\n")


def read_terminal(command_queue: SimpleQueue[str], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            command_queue.put(input("position> "))
        except EOFError:
            command_queue.put("quit")
            return


def active_node_positions(
    env: MujocoVelocityCommandEnv,
    node_names: list[str],
    node_to_physical_node: dict[str, str],
) -> dict[str, np.ndarray]:
    model = env.mj_model.model
    data = env.mj_model.data
    positions = {}
    for node_name in node_names:
        physical_node = node_to_physical_node[node_name]
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            physical_node,
        )
        if site_id >= 0:
            positions[node_name] = data.site_xpos[site_id].copy()
    return positions


def add_node_labels(
    env: MujocoVelocityCommandEnv,
    node_names: list[str],
    node_to_physical_node: dict[str, str],
    robot_scale: float,
) -> None:
    viewer = env.viewer
    if viewer is None:
        return

    scene = viewer.user_scn
    scene.ngeom = 0
    positions = active_node_positions(env, node_names, node_to_physical_node)
    offset = np.array([0.0, 0.0, 0.12 * robot_scale], dtype=float)
    identity = np.eye(3, dtype=float).ravel()
    size = np.full(3, 0.025 * robot_scale, dtype=float)
    color = np.array([0.05, 0.05, 0.05, 1.0], dtype=np.float32)

    for index, node_name in enumerate(node_names):
        if node_name not in positions or scene.ngeom >= scene.maxgeom:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LABEL,
            size,
            positions[node_name] + offset,
            identity,
            color,
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.label = f"Active {index}"
        scene.ngeom += 1


def reset_environment(
    env: MujocoVelocityCommandEnv,
    topology: ControlTopology,
) -> np.ndarray:
    env.reset(seed=RESET_SEED)
    initialize_actuator_lengths(env.mj_model.model, env.mj_model.data)
    return env.mj_model.data.act[[edge.activation_id for edge in topology.edges]].copy()


def main() -> None:
    if POSITION_GAIN <= 0.0:
        raise ValueError("POSITION_GAIN must be greater than zero.")
    if not np.isfinite(POSITION_COMMAND_SCALE):
        raise ValueError("POSITION_COMMAND_SCALE must be finite.")
    if TIMESTEP <= 0.0:
        raise ValueError("TIMESTEP must be greater than zero.")

    args = parse_args()
    env = build_environment(args.robot, args.scale)
    topology = control_topology(env)
    node_names = topology.command_node_names
    edges = topology.edges
    command_matrix = incidence_matrix(node_names, edges)
    target_positions = np.zeros(len(node_names), dtype=float)
    home_activations = reset_environment(env, topology)
    activation_ids = np.array([edge.activation_id for edge in edges], dtype=int)

    command_queue: SimpleQueue[str] = SimpleQueue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=read_terminal,
        args=(command_queue, stop_event),
        daemon=True,
    )

    print(f"Realistic {args.robot} intvelocity position control")
    print_help(topology)
    input_thread.start()

    try:
        env.render()
        while env.viewer is not None and env.viewer.is_running() and not stop_event.is_set():
            while True:
                try:
                    raw_command = command_queue.get_nowait()
                except Empty:
                    break

                try:
                    command = parse_position_command(raw_command, len(node_names))
                except ValueError as exc:
                    print(f"Invalid command: {exc}")
                    continue

                if isinstance(command, np.ndarray):
                    target_positions[:] = np.clip(
                        command * POSITION_COMMAND_SCALE,
                        -MAX_NODE_POSITION_OFFSET,
                        MAX_NODE_POSITION_OFFSET,
                    )
                    print(f"Input: {command.tolist()}")
                    print(f"Scaled target: {target_positions.tolist()}")
                elif command in {"quit", "exit", "q"}:
                    stop_event.set()
                elif command == "help":
                    print_help(topology)
                elif command == "show":
                    print(f"Target: {target_positions.tolist()}")
                elif command == "zero":
                    target_positions.fill(0.0)
                    print("Target zeroed.")
                elif command == "reset":
                    target_positions.fill(0.0)
                    home_activations = reset_environment(env, topology)
                    print("Simulation reset and target zeroed.")

            desired_activations = home_activations + command_matrix @ target_positions
            current_activations = env.mj_model.data.act[activation_ids]
            edge_velocities = POSITION_GAIN * (desired_activations - current_activations)
            edge_velocities = np.clip(
                edge_velocities,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)

            _, _, terminated, truncated, info = env.step(edge_velocities)
            if terminated:
                print(
                    "Warning: rigidity collapse detected "
                    f"(critical_eig={info['critical_eig_raw']:.5g})."
                )
            if truncated:
                print("Maximum environment step count reached.")
                stop_event.set()

            add_node_labels(env, node_names, topology.node_to_physical_node, args.scale)
            env.render()
            time.sleep(max(TIMESTEP * STEPS_PER_CONTROL_UPDATE, 0.001))
    finally:
        stop_event.set()
        env.close()


if __name__ == "__main__":
    main()
