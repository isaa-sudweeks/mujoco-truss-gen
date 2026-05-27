from __future__ import annotations

from dataclasses import dataclass, field

MODEL_INTEGRATOR = "implicitfast"
NODE_RADIUS = 0.1
MIN_NODE_CENTER_Z = NODE_RADIUS + 0.05
NODE_MASS = 1.98
ACTIVE_NODE_MASS = NODE_MASS
PASSIVE_NODE_MASS = NODE_MASS
CONNECTOR_RADIUS = 0.05
CONNECTOR_MASS = 0.05
ROD_RADIUS = 0.025
ROD_MASS = 0.05
TRIANGLE_BODY_MASS = 0.01
TRIANGLE_BODY_GRAVCOMP = 1.0
HINGE_POSITION_KP = 1000.0
HINGE_FORCE_RANGE = [-100.0, 100.0]
HINGE_CTRL_RANGE = [-3.141592653589793, 3.141592653589793]
CONNECT_CONSTRAINT_SOLREF = [0.01, 1.0]
CONNECT_CONSTRAINT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
GEOM_CONTACT_TYPE = 0
GEOM_CONTACT_AFFINITY = 1
DEFAULT_EDGE_TENDON_RANGE = [0.5, 2.0]
DEFAULT_ROUTE_TENDON_RANGE = [0.5, 10.0]
EDGE_TENDON_WIDTH = 0.05
ROUTE_TENDON_WIDTH = 0.02
PERIMETER_CONSTRAINT_TENDON_WIDTH = 0.0001
PERIMETER_CONSTRAINT_TENDON_RGBA = [0.0, 0.0, 0.0, 0.0]
ACTUATOR_CTRL_RANGE = [-0.05, 0.05]
DEFAULT_ACTUATOR_RANGE = [0.0, 3.0]
ABSTRACT_ACTUATOR_KP = 5000.0
REALISTIC_ACTUATOR_KP = 1000.0
ACTUATOR_DAMPRATIO = 1.0
REALISTIC_ACTUATOR_NOMINAL_MASS = 1.0
TENDON_RANGE_MIN_FACTOR = 0.5
TENDON_RANGE_MAX_FACTOR = 2.0
ACTUATOR_RANGE_MIN_FACTOR = 0.0
ACTUATOR_RANGE_MAX_FACTOR = 3.0
REALISTIC_NODE_CLONE_OFFSET = 0.5
ORIENTATION_WELD_SOLREF = [0.2, 5.0]
ORIENTATION_WELD_SOLIMP = [0.2, 0.3, 0.001, 0.5, 2.0]
ORIENTATION_WELD_TORQUESCALE = 6000.0
TENDON_CONSTRAINT_DATA = [0.0, -1.0, 0.0, 0.0, 0.0]
ROUTE_CONSTRAINT_DATA = [0.0, 0.0, 0.0, 0.0, 0.0]
TENDON_CONSTRAINT_SOLREF = [0.02, 1.0]
TENDON_CONSTRAINT_SOLIMP = [0.9, 0.95, 0.001]
NODE_RGBA = [0.18, 0.18, 0.18, 1.0]
ROD_RGBA = [0.62, 0.64, 0.66, 1.0]
TENDON_RGBA = [0.0, 0.1804, 0.3647, 1.0]
NODE_MATERIAL = "node_black"
ROD_MATERIAL = "connector_steel"
TENDON_MATERIAL = "blue_firehose"
TRUSS_RGBA = NODE_RGBA
BOX_SIZE = [0.05, 0.05, 0.1]


@dataclass(slots=True)
class TrussPhysicalParameters:
    node_radius: float = NODE_RADIUS
    box_size: list[float] = field(default_factory=lambda: list(BOX_SIZE))
    min_node_center_z: float = MIN_NODE_CENTER_Z
    active_node_mass: float = ACTIVE_NODE_MASS
    passive_node_mass: float = PASSIVE_NODE_MASS
    node_mass: float = NODE_MASS
    connector_radius: float = CONNECTOR_RADIUS
    connector_mass: float = CONNECTOR_MASS
    rod_radius: float = ROD_RADIUS
    rod_mass: float = ROD_MASS
    triangle_body_mass: float = TRIANGLE_BODY_MASS
    triangle_body_gravcomp: float = TRIANGLE_BODY_GRAVCOMP
    hinge_position_kp: float = HINGE_POSITION_KP
    hinge_force_range: list[float] = field(default_factory=lambda: list(HINGE_FORCE_RANGE))
    hinge_ctrl_range: list[float] = field(default_factory=lambda: list(HINGE_CTRL_RANGE))
    connect_constraint_solref: list[float] = field(
        default_factory=lambda: list(CONNECT_CONSTRAINT_SOLREF)
    )
    connect_constraint_solimp: list[float] = field(
        default_factory=lambda: list(CONNECT_CONSTRAINT_SOLIMP)
    )
    default_edge_tendon_range: list[float] = field(
        default_factory=lambda: list(DEFAULT_EDGE_TENDON_RANGE)
    )
    default_route_tendon_range: list[float] = field(
        default_factory=lambda: list(DEFAULT_ROUTE_TENDON_RANGE)
    )
    edge_tendon_width: float = EDGE_TENDON_WIDTH
    route_tendon_width: float = ROUTE_TENDON_WIDTH
    perimeter_constraint_tendon_width: float = PERIMETER_CONSTRAINT_TENDON_WIDTH
    actuator_ctrl_range: list[float] = field(default_factory=lambda: list(ACTUATOR_CTRL_RANGE))
    default_actuator_range: list[float] = field(
        default_factory=lambda: list(DEFAULT_ACTUATOR_RANGE)
    )
    abstract_actuator_kp: float = ABSTRACT_ACTUATOR_KP
    realistic_actuator_kp: float = REALISTIC_ACTUATOR_KP
    actuator_dampratio: float = ACTUATOR_DAMPRATIO
    realistic_actuator_nominal_mass: float = REALISTIC_ACTUATOR_NOMINAL_MASS
    tendon_range_min_factor: float = TENDON_RANGE_MIN_FACTOR
    tendon_range_max_factor: float = TENDON_RANGE_MAX_FACTOR
    actuator_range_min_factor: float = ACTUATOR_RANGE_MIN_FACTOR
    actuator_range_max_factor: float = ACTUATOR_RANGE_MAX_FACTOR
    realistic_node_clone_offset: float = REALISTIC_NODE_CLONE_OFFSET
    orientation_weld_solref: list[float] = field(
        default_factory=lambda: list(ORIENTATION_WELD_SOLREF)
    )
    orientation_weld_solimp: list[float] = field(
        default_factory=lambda: list(ORIENTATION_WELD_SOLIMP)
    )
    orientation_weld_torquescale: float = ORIENTATION_WELD_TORQUESCALE
    tendon_constraint_solref: list[float] = field(
        default_factory=lambda: list(TENDON_CONSTRAINT_SOLREF)
    )
    tendon_constraint_solimp: list[float] = field(
        default_factory=lambda: list(TENDON_CONSTRAINT_SOLIMP)
    )
