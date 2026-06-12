"""Interactively position-command the realistic octahedron in MuJoCo.

Run on macOS with:

    .venv/bin/mjpython experiments/system_id/interactive_octahedron_position_control.py

Enter an eight-element Python/JSON array at the ``position>`` prompt. Each
entry controls one active physical triangle node; the array-to-node mapping is
printed at startup and shown as labels in the viewer. Values are relative
scalar node-position offsets in meters. Passive triangle nodes remain at zero,
and active node commands are mapped to tendon commands with the same oriented
incidence rule used by the routed-tube node controller.
"""

from __future__ import annotations

import ast
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from queue import Empty, SimpleQueue

import mujoco
import numpy as np

from mujoco_truss_gen import (
    MujocoVelocityCommandEnv,
    TrussEnvConfig,
    TrussPhysicalParameters,
    get_mujoco_spec,
)
from mujoco_truss_gen.mujoco_model.tendons import initialize_actuator_lengths

# -----------------------------------------------------------------------------
# Editable experiment parameters
# -----------------------------------------------------------------------------

ROBOT_SCALE = 1.27
CONNECTOR_ROD_LENGTH = 0.288675

PHYSICAL_PARAMS = TrussPhysicalParameters(
    active_node_mass=0.1,
    passive_node_mass=0.1,
    connector_radius=0.05,
    connector_mass=0.05,
    rod_radius=0.025,
    rod_mass=0.05,
    triangle_body_mass=0.01,
    triangle_body_gravcomp=1.0,
    hinge_position_kp=10.0,
    hinge_damping=1.0,
    realistic_actuator_kp=1000.0,
    actuator_dampratio=1.0,
    realistic_actuator_nominal_mass=0.1,
    actuator_ctrl_range=[-0.1, 0.1],
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
POSITION_COMMAND_SCALE = 0.01 * 0.0254
MAX_NODE_POSITION_OFFSET = 1.0
STEPS_PER_CONTROL_UPDATE = 5
RESET_SEED = 0


@dataclass(frozen=True, slots=True)
class ActuatorEdge:
    """One intvelocity actuator and its physical active/passive endpoints."""

    actuator_id: int
    activation_id: int
    tendon_name: str
    triangle_name: str
    from_node: str
    to_node: str
    active_node: str
    passive_node: str


def build_environment() -> MujocoVelocityCommandEnv:
    spec = get_mujoco_spec(
        "octahedron",
        realistic=True,
        scale=ROBOT_SCALE,
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


def actuator_edges(env: MujocoVelocityCommandEnv) -> list[ActuatorEdge]:
    """Read actuator orientation and active/passive nodes from generated XML."""
    xml = env.mj_model.xml
    if xml is None:
        raise RuntimeError("The generated model did not retain its MuJoCo XML.")

    root = ET.fromstring(xml)
    tendon_sites = {
        spatial.get("name", ""): [site.get("site", "") for site in spatial.findall("site")]
        for spatial in root.findall("./tendon/spatial")
    }
    node_triangles = {
        node_body.get("name", ""): triangle_body.get("name", "")
        for triangle_body in root.findall("./worldbody/body")
        if triangle_body.get("name", "").startswith("tri_")
        for node_body in triangle_body.findall("body")
        if node_body.get("name", "").startswith("node_")
    }
    raw_edges = []
    model = env.mj_model.model

    for actuator_id in env.mj_model.external_actuator_ids:
        actuator_id = int(actuator_id)
        if model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_INTEGRATOR:
            raise RuntimeError(
                f"External actuator {model.actuator(actuator_id).name!r} is not intvelocity."
            )

        tendon_id = int(model.actuator_trnid[actuator_id, 0])
        tendon_name = model.tendon(tendon_id).name
        sites = tendon_sites.get(tendon_name, [])
        physical_nodes = [
            env.mj_model.site_to_node[site_name]
            for site_name in sites
            if site_name in env.mj_model.site_to_node
        ]
        if len(physical_nodes) != 2:
            raise RuntimeError(
                f"Expected tendon {tendon_name!r} to connect two nodes, got {physical_nodes}."
            )
        triangle_names = {node_triangles.get(node_name) for node_name in physical_nodes}
        if None in triangle_names or len(triangle_names) != 1:
            raise RuntimeError(
                f"Expected tendon {tendon_name!r} to stay within one triangle, "
                f"got endpoint triangles {triangle_names}."
            )

        activation_id = int(model.actuator_actadr[actuator_id])
        if activation_id < 0:
            raise RuntimeError(f"Actuator {model.actuator(actuator_id).name!r} has no state.")
        raw_edges.append(
            (
                actuator_id,
                activation_id,
                tendon_name,
                triangle_names.pop(),
                physical_nodes[0],
                physical_nodes[1],
            )
        )

    passive_by_triangle = {}
    triangle_names = {edge[3] for edge in raw_edges}
    for triangle_name in triangle_names:
        triangle_edges = [edge for edge in raw_edges if edge[3] == triangle_name]
        if len(triangle_edges) != 2:
            raise RuntimeError(
                f"Expected two active tendons in {triangle_name!r}, got {len(triangle_edges)}."
            )
        shared_nodes = set(triangle_edges[0][4:6]) & set(triangle_edges[1][4:6])
        if len(shared_nodes) != 1:
            raise RuntimeError(
                f"Could not identify one passive node in {triangle_name!r}: {shared_nodes}."
            )
        passive_by_triangle[triangle_name] = shared_nodes.pop()

    edges = []
    for actuator_id, activation_id, tendon_name, triangle_name, from_node, to_node in raw_edges:
        passive_node = passive_by_triangle[triangle_name]
        active_nodes = {from_node, to_node} - {passive_node}
        if len(active_nodes) != 1:
            raise RuntimeError(f"Could not identify the active endpoint of {tendon_name!r}.")
        edges.append(
            ActuatorEdge(
                actuator_id=actuator_id,
                activation_id=activation_id,
                tendon_name=tendon_name,
                triangle_name=triangle_name,
                from_node=from_node,
                to_node=to_node,
                active_node=active_nodes.pop(),
                passive_node=passive_node,
            )
        )

    active_nodes = [edge.active_node for edge in edges]
    if len(active_nodes) != 8 or len(set(active_nodes)) != 8:
        raise RuntimeError(f"Expected eight unique active nodes, got {active_nodes}.")
    return edges


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
        raise ValueError("Enter a numeric array such as [0, 0.02, 0, 0, 0, 0, 0, 0].") from None

    if values.shape != (node_count,):
        raise ValueError(f"Expected an array with {node_count} values, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("All position values must be finite.")
    return values


def print_help(edges: list[ActuatorEdge]) -> None:
    print("\nCommands:")
    print("  [p0, p1, ...]  set the eight active physical node positions")
    print("  show            show the current position target")
    print("  zero            set every position target to zero")
    print("  reset           reset the simulation and zero the target")
    print("  help            show this message")
    print("  quit            close the viewer")
    print("Array mapping:")
    for index, edge in enumerate(edges):
        sign = "+" if edge.to_node == edge.active_node else "-"
        print(
            f"  index {index}: {edge.active_node} in {edge.triangle_name} "
            f"({edge.tendon_name} offset = {sign}command[{index}])"
        )
    print(
        "Each passive triangle node is fixed at command zero. Tendon offsets use "
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
) -> dict[str, np.ndarray]:
    model = env.mj_model.model
    data = env.mj_model.data
    positions = {}
    for node_name in node_names:
        site_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            node_name,
        )
        if site_id >= 0:
            positions[node_name] = data.site_xpos[site_id].copy()
    return positions


def add_node_labels(
    env: MujocoVelocityCommandEnv,
    node_names: list[str],
) -> None:
    viewer = env.viewer
    if viewer is None:
        return

    scene = viewer.user_scn
    scene.ngeom = 0
    positions = active_node_positions(env, node_names)
    offset = np.array([0.0, 0.0, 0.12 * ROBOT_SCALE], dtype=float)
    identity = np.eye(3, dtype=float).ravel()
    size = np.full(3, 0.025 * ROBOT_SCALE, dtype=float)
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
    edges: list[ActuatorEdge],
) -> np.ndarray:
    env.reset(seed=RESET_SEED)
    initialize_actuator_lengths(env.mj_model.model, env.mj_model.data)
    return env.mj_model.data.act[[edge.activation_id for edge in edges]].copy()


def main() -> None:
    if POSITION_GAIN <= 0.0:
        raise ValueError("POSITION_GAIN must be greater than zero.")
    if not np.isfinite(POSITION_COMMAND_SCALE):
        raise ValueError("POSITION_COMMAND_SCALE must be finite.")
    if TIMESTEP <= 0.0:
        raise ValueError("TIMESTEP must be greater than zero.")

    env = build_environment()
    edges = actuator_edges(env)
    node_names = [edge.active_node for edge in edges]
    command_matrix = incidence_matrix(node_names, edges)
    target_positions = np.zeros(len(node_names), dtype=float)
    home_activations = reset_environment(env, edges)
    activation_ids = np.array([edge.activation_id for edge in edges], dtype=int)

    command_queue: SimpleQueue[str] = SimpleQueue()
    stop_event = threading.Event()
    input_thread = threading.Thread(
        target=read_terminal,
        args=(command_queue, stop_event),
        daemon=True,
    )

    print("Realistic octahedron intvelocity position control")
    print_help(edges)
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
                    print_help(edges)
                elif command == "show":
                    print(f"Target: {target_positions.tolist()}")
                elif command == "zero":
                    target_positions.fill(0.0)
                    print("Target zeroed.")
                elif command == "reset":
                    target_positions.fill(0.0)
                    home_activations = reset_environment(env, edges)
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

            add_node_labels(env, node_names)
            env.render()
            time.sleep(max(TIMESTEP * STEPS_PER_CONTROL_UPDATE, 0.001))
    finally:
        stop_event.set()
        env.close()


if __name__ == "__main__":
    main()
