# TrussPhysicalParameters

Pass a `TrussPhysicalParameters` instance to `get_mujoco_spec()` to configure
the geometry, masses, actuators, and constraints of a generated robot. These
are fixed settings for the generated model. Build a new spec and compile it
again when you change them.

```python
from mujoco_truss_gen import MujocoModel, TrussPhysicalParameters, get_mujoco_spec

params = TrussPhysicalParameters(connector_rod_length=0.18, hinge_position_kp=30.0)
spec = get_mujoco_spec("octahedron", realistic=True, physical_params=params)
robot = MujocoModel(spec)

# Check the gain on the compiled connector hinge actuators.
for index in robot.internal_actuator_ids:
    actuator = robot.model.actuator(index)
    print(actuator.name, actuator.gainprm[0])
```

Use the [first robot tutorial](first-robot.md) to simulate or view the resulting
spec. Preset `scale=` controls the starting structural geometry, while
`terrain=` configures the ground. [Domain randomization](environments.md#domain-randomization)
covers parameters that vary across environment resets.

## Nodes and initial placement

| Field | Default | Effect |
| --- | --- | --- |
| `node_radius` | `0.1` | Radius in meters of spherical abstract nodes. |
| `box_size` | `[0.05, 0.05, 0.1]` | Half-extents in meters of realistic box-shaped node geoms. Its first entry also sets the half-length of passive cylinders in realistic routed models. |
| `min_node_center_z` | `0.15` | Minimum initial world-space node center height in meters. The builder lifts the geometry together if its lowest center is below this threshold. It measures center height in the world frame. |
| `active_node_mass` | `0.1` | Mass in kilograms for nodes assigned the active role in triangle models and realistic routed models. |
| `passive_node_mass` | `0.1` | Mass in kilograms for passive nodes in triangle models and passive endpoints in realistic routed models. In abstract triangle models, a shared node takes this mass if any triangle designates it as passive. |
| `node_mass` | `0.1` | Uniform node mass in kilograms when roles are unspecified, including abstract routed models. |

## Realistic connectors and triangle bodies

| Field | Default | Effect |
| --- | --- | --- |
| `connector_radius` | `0.05` | Radius in meters of each ball that connects cloned instances of a shared node. |
| `connector_mass` | `0.05` | Mass in kilograms of each connector ball. |
| `rod_radius` | `0.025` | Radius in meters of the connector rods. |
| `rod_mass` | `0.05` | Mass in kilograms of each connector rod. |
| `triangle_body_mass` | `0.01` | Explicit mass in kilograms of each realistic triangle parent body. The generator also uses this number for each diagonal inertia entry. Child node and rod masses are assigned separately. |
| `triangle_body_gravcomp` | `1.0` | Fraction of gravity compensated on the realistic triangle parent body. `0.0` leaves its weight fully active, and `1.0` compensates its full weight. |
| `connector_rod_length` | `0.2032` | Absolute nominal rod length in meters, independent of preset scale. For irregular triangles it specifies the mean rod length. Pass `None` to use the legacy offset factor. |
| `realistic_node_clone_offset` | `0.5` | Dimensionless factor controlling the separation of cloned node geometry when `connector_rod_length=None`. The resulting distance scales with the input geometry. |
| `hinge_position_kp` | `45.0` | Proportional gain of the internal connector hinge position servos that follow the bisector controller's angle targets. |
| `hinge_damping` | `0.75` | Passive damping applied to realistic node hinge joints. |
| `hinge_force_range` | `[-50.0, 50.0]` | Lower and upper output limits for the connector hinge servos, expressed as torque for hinge actuation. |
| `hinge_ctrl_range` | `[-π, π]` | Lower and upper connector hinge position commands in radians. |

## Tendon geometry and limits

| Field | Default | Effect |
| --- | --- | --- |
| `default_edge_tendon_range` | `[0.2, 2.0]` | Fallback edge-tendon length range in meters. The normal model builders use the first entry for the lower limit and calculate the upper limit from edge length. |
| `default_route_tendon_range` | `[0.5, 10.0]` | Fallback length range in meters for a whole-route tendon. The routed model builders use the first entry for the lower limit and calculate the upper limit from total route length. |
| `tendon_range_min_factor` | `0.5` | Present in the dataclass but currently unused. The normal builders take the minimum from the corresponding default range. |
| `tendon_range_max_factor` | `2.0` | Multiplier applied to the initial edge length or total route length to compute its tendon upper limit. |
| `edge_tendon_width` | `0.05` | Display width of structural tendons in meters. Also sets the radius of passive endpoint cylinders in realistic routed models, where it affects contact geometry. |
| `route_tendon_width` | `0.02` | Display width in meters of the whole-route tendon. |
| `perimeter_constraint_tendon_width` | `0.0001` | Width in meters of the transparent auxiliary tendon used by the triangle perimeter constraint. |

## External tendon actuators

A tendon actuator integrates its velocity command into a target length. The
command range bounds how fast the target moves. The activation range bounds
the target length itself. The tendon range limits the physical tendon length.

| Field | Default | Effect |
| --- | --- | --- |
| `actuator_ctrl_range` | `[-0.1, 0.1]` | Lower and upper external tendon actuator velocity commands in meters per second. |
| `default_actuator_range` | `[0.0, 3.0]` | Default activation range in meters for integrated tendon length targets. Abstract STL-imported routes use geometry-scaled activation limits. |
| `abstract_actuator_kp` | `1000.0` | Proportional gain of external tendon servos in abstract models. |
| `realistic_actuator_kp` | `1000.0` | Proportional gain of external tendon servos in realistic models. |
| `actuator_dampratio` | `1.0` | Dimensionless ratio used to calculate the tendon servo's velocity damping coefficient. |
| `realistic_actuator_nominal_mass` | `0.1` | Nominal mass in kilograms used in the damping calculation for both model modes. The generator computes `kv = 2 * dampratio * sqrt(kp * nominal_mass)`. This field only enters that calculation. |
| `actuator_range_min_factor` | `0.0` | Lower activation limit as a fraction of initial edge length for abstract STL-imported routes. |
| `actuator_range_max_factor` | `3.0` | Upper activation limit as a multiple of initial edge length for abstract STL-imported routes. |

## Constraint solver settings

Each pair of `solref` and `solimp` fields applies to a particular equality
constraint. With the positive defaults below, `solref` holds the time constant
in seconds and the damping ratio. `solimp` controls how strongly the solver
enforces the constraint as its error changes. In the three-entry fields, the
generator sets the first three coefficients and keeps MuJoCo's defaults for
the last two. [MuJoCo solver parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters)

| Field | Default | Effect |
| --- | --- | --- |
| `connect_constraint_solref` | `[0.01, 1.0]` | Reference response for realistic rod-to-ball connect constraints. |
| `connect_constraint_solimp` | `[0.95, 0.99, 0.001, 0.5, 2.0]` | Impedance curve for realistic rod-to-ball connect constraints. |
| `tendon_constraint_solref` | `[0.02, 1.0]` | Reference response for the triangle perimeter equality. |
| `tendon_constraint_solimp` | `[0.9, 0.95, 0.001]` | Impedance coefficients for the triangle perimeter equality. |
| `route_length_constraint_solref` | `[0.01, 1.0]` | Reference response for the routed tendon total-length equality, when enabled. |
| `route_length_constraint_solimp` | `[0.95, 0.99, 0.001]` | Impedance coefficients for the routed tendon total-length equality, when enabled. |
| `orientation_weld_solref` | `[0.2, 5.0]` | Reference response for optional node-to-world welds added by the low-level `add_orientation_constraints()` helper. |
| `orientation_weld_solimp` | `[0.2, 0.3, 0.001, 0.5, 2.0]` | Impedance curve for those optional welds. |
| `orientation_weld_torquescale` | `6000.0` | MuJoCo weld length scale that weights rotation error relative to translation error in those optional welds. |

`get_mujoco_spec()` leaves the optional node-to-world welds out of the generated
model. Routed shapes get the total-length equality unless
`disable_route_length_constraint=True`, as in STL-imported shapes.
