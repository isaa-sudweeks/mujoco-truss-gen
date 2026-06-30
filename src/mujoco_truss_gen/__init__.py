from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mujoco_truss_gen.base_env import (
    DomainRandomizationConfig,
    MujocoTrussEnv,
    TrussEnvConfig,
)
from mujoco_truss_gen.mesh_import import stl_to_shape_dict
from mujoco_truss_gen.mujoco_model import (
    DEFAULT_ACCELEROMETER_CONFIG,
    HENNEBERG_PRESET_SPECS,
    HENNEBERG_PRESET_VARIANT_COUNTS,
    HENNEBERG_RIGIDITY_THRESHOLD,
    PRESETS,
    USEVITCH_GRAPH_LABELS,
    AccelerometerConfig,
    NodeVelocityController,
    TrussPhysicalParameters,
    add_node_accelerometers,
    build_abstract_shapes,
    build_shapes,
    build_triangle,
    build_world,
    get_edge_index,
    get_edge_types,
    get_henneberg_routed_graph_definition,
    get_icosahedron_definition,
    get_mujoco_spec,
    get_networkx_graph,
    get_node_features,
    get_octahedron_definition,
    get_perimeter,
    get_preset_definition,
    get_route_lengths,
    get_usevitch_graph_definition,
    save_xml,
    view,
    view_graph,
    view_node_velocity,
    view_node_velocity_terminal,
)
from mujoco_truss_gen.mujoco_model.model import MujocoModel
from mujoco_truss_gen.node_velocity_command_env import MujocoNodeVelocityCommandEnv
from mujoco_truss_gen.relative_observation_env import MujocoRelativeObsEnv
from mujoco_truss_gen.velocity_command_env import MujocoVelocityCommandEnv

if TYPE_CHECKING:
    from mujoco_truss_gen.mjx_controllers import MjxAngleBisectorController
    from mujoco_truss_gen.mjx_env import MjxEnvState, MjxNodeVelocityEnv


def __getattr__(name: str) -> Any:
    if name in {"MjxAngleBisectorController", "MjxEnvState", "MjxNodeVelocityEnv"}:
        from mujoco_truss_gen.mjx_controllers import MjxAngleBisectorController
        from mujoco_truss_gen.mjx_env import MjxEnvState, MjxNodeVelocityEnv

        exports = {
            "MjxAngleBisectorController": MjxAngleBisectorController,
            "MjxEnvState": MjxEnvState,
            "MjxNodeVelocityEnv": MjxNodeVelocityEnv,
        }
        globals().update(exports)
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AccelerometerConfig",
    "DEFAULT_ACCELEROMETER_CONFIG",
    "DomainRandomizationConfig",
    "MujocoModel",
    "MjxEnvState",
    "MjxNodeVelocityEnv",
    "MujocoNodeVelocityCommandEnv",
    "MujocoRelativeObsEnv",
    "MujocoTrussEnv",
    "MujocoVelocityCommandEnv",
    "MjxAngleBisectorController",
    "NodeVelocityController",
    "PRESETS",
    "HENNEBERG_PRESET_SPECS",
    "HENNEBERG_PRESET_VARIANT_COUNTS",
    "HENNEBERG_RIGIDITY_THRESHOLD",
    "TrussEnvConfig",
    "TrussPhysicalParameters",
    "USEVITCH_GRAPH_LABELS",
    "add_node_accelerometers",
    "build_abstract_shapes",
    "build_shapes",
    "build_triangle",
    "build_world",
    "get_edge_index",
    "get_edge_types",
    "get_networkx_graph",
    "get_henneberg_routed_graph_definition",
    "get_icosahedron_definition",
    "get_mujoco_spec",
    "get_node_features",
    "get_octahedron_definition",
    "get_perimeter",
    "get_preset_definition",
    "get_usevitch_graph_definition",
    "get_route_lengths",
    "save_xml",
    "stl_to_shape_dict",
    "view",
    "view_graph",
    "view_node_velocity",
    "view_node_velocity_terminal",
]
