# Model Generation

This package generates MuJoCo `MjSpec` models from either triangle dictionaries
or routed shape dictionaries.

## Custom Triangle Trusses

Custom trusses are represented with two dictionaries.

`node_dict` maps node names to 3D positions:

```python
node_dict = {
    "node_1": [0.0, 0.0, 0.2],
    "node_2": [0.8, 0.0, 0.2],
    "node_3": [0.4, 0.7, 0.2],
}
```

`triangle_dict` maps triangle names to four node names:

```python
triangle_dict = {
    "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
}
```

The first three names are the triangle vertices. The fourth name is the passive
node for that triangle's perimeter constraint and must be one of the first three
vertex names.

Build a model from those dictionaries:

```python
from mujoco_truss_gen import get_mujoco_spec

spec = get_mujoco_spec(node_dict, triangle_dict, realistic=False)
model = spec.compile()
```

## Routed Shape Dictionaries

Continuous tube shapes can be represented with a shape dictionary. Each shape
defines a routed path:

```python
import numpy as np

node_dict = {
    "node_1": [0.0, 0.0, 0.1],
    "node_2": [1.0, 0.0, 0.1],
    "node_3": [0.5, 0.8660, 0.1],
    "node_4": [0.5, np.sqrt(3) / 6, 0.1 + np.sqrt(2 / 3)],
}
shape_dict = {
    "path_1": {
        "route": ["node_1", "node_2", "node_4", "node_3"],
        "active_edges": [
            ["node_1", "node_2"],
            ["node_4", "node_3"],
        ],
    },
    "path_2": {
        "route": ["node_2", "node_3", "node_1", "node_4"],
        "active_edges": [
            ["node_2", "node_3"],
            ["node_1", "node_4"],
        ],
    },
}

spec = get_mujoco_spec(node_dict, shape_dict, realistic=False)
model = spec.compile()
```

The route creates one edge tendon for each adjacent node pair in the path.
Every unique route edge receives an actuator. Routed continuous-tube models do
not use route-length equality constraints, so they can run without MuJoCo tendon
constraints. Route tendons are still emitted as non-actuated route metadata and
visual tendons.

`get_mujoco_spec()`, `build_triangle()`, and `build_shapes()` treat caller-provided
dictionaries as read-only inputs. Realistic builders clone shared nodes
internally, but they do not mutate the original dictionaries passed by the
caller.

## Public Generation Helpers

- `build_world()` creates a base `mujoco.MjSpec` containing a checker-textured
  ground plane, bright skybox, and key/fill lighting.
- `build_triangle(spec, node_dict, triangle_dict, realistic=False)` adds truss
  bodies, sites, tendons, actuators, and perimeter constraints to an existing
  spec.
- `build_shapes(spec, node_dict, shape_dict, realistic=False)` adds routed
  continuous-tube shapes with unconstrained per-edge tendon actuators. With
  `realistic=True`, shared routed node occurrences are cloned and connected to
  connector balls by in-plane bisector rods.
- `get_mujoco_spec("octahedron", realistic=False, scale=1.0)` and the other
  names in `PRESETS` build built-in presets. Increase or decrease `scale` to
  generate the same preset in a different unit scale.
- `get_mujoco_spec(node_dict, triangle_dict, realistic=False)` builds a custom
  dictionary-defined truss.
- `get_mujoco_spec(node_dict, shape_dict, realistic=False)` builds a routed
  continuous-tube shape model.
- `get_octahedron_definition(scale=1.0)` returns fresh node and triangle
  dictionaries for the built-in preset.
- `get_icosahedron_definition(scale=1.0)` returns fresh node and triangle
  dictionaries for the built-in preset.
- `get_usevitch_graph_definition(graph_label, partition_index=1, scale=1.0)`
  returns one of the non-octahedron triangle-decomposable graph definitions
  enumerated in Usevitch et al. (2025), Fig. 3. Named presets use the paper's
  graph labels, for example `usevitch_1514879` or
  `usevitch_60243677150_p3` when multiple partitions exist. Triangle
  partitions are recomputed from the decoded graph with the paper's exhaustive
  exact-cover criterion: enumerate every graph 3-cycle, then select
  edge-disjoint triangles that cover every graph edge exactly once. The integer
  programming formulations described in the paper are not used for these small
  built-in preset graphs. Node positions are generated with the paper's
  multidimensional-scaling embedding search: each trial assigns random
  distances to connected node pairs, assigns distance `10` to disjoint node
  pairs, and computes a 3D MDS embedding. Each MDS candidate is then rescaled
  so the mean structural edge length is `1.0`, which keeps model sizes close to
  the other built-in presets without changing the candidate's shape. The
  selected embedding is the candidate with the largest worst case rigidity
  index; when candidates have nearly identical rigidity index values, the
  tie-breaker is the smaller RMS error from unit structural edge lengths.
- `get_perimeter(node_dict, triangle_dict)` computes each triangle perimeter
  from the first three vertices.
- `save_xml(spec, filename)` writes `spec.to_xml()` to disk and returns the
  resolved path.
- `view(spec)` compiles and opens the generated model in MuJoCo's passive
  viewer.
- `view_node_velocity_terminal(spec, speed=0.01)` opens a routed
  continuous-tube model in MuJoCo's passive viewer and accepts terminal commands
  that set node-level scalar velocity commands. Those node commands are mapped
  to routed tendon actuator commands each simulation step.

```python
from mujoco_truss_gen import get_mujoco_spec, view_node_velocity_terminal

spec = get_mujoco_spec("tetrahedron", realistic=False)
view_node_velocity_terminal(spec)
```

Example terminal commands:

```text
nodes
set node_2 0.01
add node_2 -0.002
show
zero
quit
```

## Input Expectations

- Node names should be unique strings. Names beginning with `node_` are required
  for the built-in metadata and environment helpers.
- Node positions must be 3D numeric sequences.
- Triangle entries must contain exactly the three vertex nodes plus one passive
  node.
- The passive node must appear in that triangle's first three vertices.
- Custom triangle definitions are validated before MuJoCo objects are created,
  and validation errors name the node, triangle, or shape entry that needs to be
  fixed.
- Shape entries must contain `route` and `active_edges` keys. `active_edges` is
  accepted for compatibility, but all adjacent route edges are actuated.
- Shape routes must contain at least two node names. Each active edge must be an
  adjacent pair in the route.
- The builder helpers should be used when environment rigidity and slip helpers
  are needed, because those helpers infer structure from generated body, site,
  tendon, and actuator names.

## Model Modes

- `realistic=False` creates one world-body per node with slide joints on `x`,
  `y`, and `z`. This is the simpler abstract model and is useful for fast
  algorithm development.
- `realistic=True` creates one free triangle body per triangle for triangle
  dictionaries. For routed shape dictionaries, it creates direct world-level
  cloned route node bodies rather than a shared plane parent. In both cases,
  shared logical nodes are connected through connector balls. Connector rods are
  initialized in the original local route or triangle plane, and the internal
  bisector controller keeps each rod aligned with the projected angle bisector
  of the adjacent edges. In routed shape dictionaries, passive route endpoints
  are rendered as cylinders with their flat-face normal aligned to the connector
  rod and diameter matched to the edge tendon width. The routed-tube
  `realistic=True` path is still incomplete and should be treated as a known
  modeling limitation until the routed connector/body geometry is fixed.

Realistic triangle models include one accelerometer sensor per generated node
site by default. Pass `accelerometer_config=AccelerometerConfig(...)` to
configure sensor fields such as `noise`, `cutoff`, `nsample`, `delay`, or
`name_prefix`; pass `accelerometer_config=None` to omit them.

## References

- Nathan Usevitch, Isaac Weaver, and James Usevitch. "Triangle-Decomposable
  Graphs for Isoperimetric Robots." arXiv:2505.01624, 2025.
  <https://arxiv.org/abs/2505.01624>
