from __future__ import annotations

from dataclasses import dataclass, field

# MuJoCo time integrator used by generated models.
MODEL_INTEGRATOR = "implicitfast"
# Radius of spherical abstract nodes.
NODE_RADIUS = 0.1
# Minimum allowed world-space node center height after geometry lifting.
MIN_NODE_CENTER_Z = NODE_RADIUS + 0.05
# Default node mass used when active/passive roles are not distinguished.
NODE_MASS = 0.1
# Mass for actuated route or triangle nodes.
ACTIVE_NODE_MASS = NODE_MASS
# Mass for passive route endpoints or triangle passive nodes.
PASSIVE_NODE_MASS = NODE_MASS
# Radius of connector balls that tie cloned realistic node instances together.
CONNECTOR_RADIUS = 0.05
# Mass of each realistic connector ball.
CONNECTOR_MASS = 0.05
# Radius of connector rods between cloned nodes and connector balls.
ROD_RADIUS = 0.025
# Mass of each connector rod.
ROD_MASS = 0.05
# Default absolute nominal connector rod length.
CONNECTOR_ROD_LENGTH = 0.2032
# Inertial mass assigned to each realistic triangle body shell.
TRIANGLE_BODY_MASS = 0.01
# Gravity compensation applied to realistic triangle body shells.
TRIANGLE_BODY_GRAVCOMP = 1.0
# Proportional gain for hinge position actuators that orient connector rods.
HINGE_POSITION_KP = 45.0
# Passive damping on hinge joints that orient connector rods.
HINGE_DAMPING = 0.75
# Force limits for hinge position actuators.
HINGE_FORCE_RANGE = [-50.0, 50.0]
# Control angle limits for hinge position actuators, in radians.
HINGE_CTRL_RANGE = [-3.141592653589793, 3.141592653589793]
# MuJoCo solref values for connector rod-to-ball equality constraints.
CONNECT_CONSTRAINT_SOLREF = [0.01, 1.0]
# MuJoCo solimp values for connector rod-to-ball equality constraints.
CONNECT_CONSTRAINT_SOLIMP = [0.95, 0.99, 0.001, 0.5, 2.0]
# Collision type mask assigned to generated truss geoms.
GEOM_CONTACT_TYPE = 1
# Collision affinity mask assigned to generated truss geoms.
GEOM_CONTACT_AFFINITY = 1
# Default min/max length range for structural edge tendons.
DEFAULT_EDGE_TENDON_RANGE = [0.2, 2.0]
# Default min/max length range for non-actuated route tendons.
DEFAULT_ROUTE_TENDON_RANGE = [0.5, 10.0]
# Visual width of structural edge tendons.
EDGE_TENDON_WIDTH = 0.05
# Visual width of route metadata tendons.
ROUTE_TENDON_WIDTH = 0.02
# Visual width of hidden perimeter-constraint tendons.
PERIMETER_CONSTRAINT_TENDON_WIDTH = 0.0001
# RGBA color used to hide perimeter-constraint tendons.
PERIMETER_CONSTRAINT_TENDON_RGBA = [0.0, 0.0, 0.0, 0.0]
# Velocity-control limits for tendon actuators.
ACTUATOR_CTRL_RANGE = [-0.1, 0.1]
# Default integrator activation range for tendon actuators.
DEFAULT_ACTUATOR_RANGE = [0.0, 3.0]
# Proportional gain for abstract tendon actuators.
ABSTRACT_ACTUATOR_KP = 5000.0
# Proportional gain for realistic tendon actuators.
REALISTIC_ACTUATOR_KP = 1000.0
# Critical damping ratio used when deriving tendon actuator damping.
ACTUATOR_DAMPRATIO = 1.0
# Nominal mass used to convert realistic tendon kp/dampratio into kv.
REALISTIC_ACTUATOR_NOMINAL_MASS = 0.1
# Geometry-scaled tendon range lower multiplier.
TENDON_RANGE_MIN_FACTOR = 0.5
# Geometry-scaled tendon range upper multiplier.
TENDON_RANGE_MAX_FACTOR = 2.0
# Geometry-scaled actuator activation range lower multiplier.
ACTUATOR_RANGE_MIN_FACTOR = 0.0
# Geometry-scaled actuator activation range upper multiplier.
ACTUATOR_RANGE_MAX_FACTOR = 3.0
# Radial offset factor used when cloning shared realistic nodes.
REALISTIC_NODE_CLONE_OFFSET = 0.5
# MuJoCo solref values for optional orientation weld constraints.
ORIENTATION_WELD_SOLREF = [0.2, 5.0]
# MuJoCo solimp values for optional orientation weld constraints.
ORIENTATION_WELD_SOLIMP = [0.2, 0.3, 0.001, 0.5, 2.0]
# Torque scale for optional orientation weld constraints.
ORIENTATION_WELD_TORQUESCALE = 6000.0
# Equality polynomial data for tendon perimeter constraints.
TENDON_CONSTRAINT_DATA = [0.0, -1.0, 0.0, 0.0, 0.0]
# Equality polynomial data for route length constraints.
ROUTE_CONSTRAINT_DATA = [0.0, 0.0, 0.0, 0.0, 0.0]
# MuJoCo solref values for tendon equality constraints.
TENDON_CONSTRAINT_SOLREF = [0.02, 1.0]
# MuJoCo solimp values for tendon equality constraints.
TENDON_CONSTRAINT_SOLIMP = [0.9, 0.95, 0.001]
# Default node color.
NODE_RGBA = [0.18, 0.18, 0.18, 1.0]
# Default connector rod color.
ROD_RGBA = [0.62, 0.64, 0.66, 1.0]
# Default tendon color.
TENDON_RGBA = [0.0, 0.1804, 0.3647, 1.0]
# Material name used by node geoms.
NODE_MATERIAL = "node_black"
# Material name used by connector rod geoms.
ROD_MATERIAL = "connector_steel"
# Material name used by tendon visuals.
TENDON_MATERIAL = "blue_firehose"
# Backward-compatible truss color alias.
TRUSS_RGBA = NODE_RGBA
# Half-extents for box-shaped node geoms.
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
    hinge_damping: float = HINGE_DAMPING
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
    # Absolute nominal connector rod length. Set to None to use the legacy
    # scale-proportional realistic_node_clone_offset behavior.
    connector_rod_length: float | None = CONNECTOR_ROD_LENGTH
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
