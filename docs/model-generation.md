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
  generate the same preset with different structural tendon lengths. For a
  realistic model, set `TrussPhysicalParameters(connector_rod_length=...)` to
  keep the nominal connector rod length fixed independently of preset scale.
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
  built-in preset graphs. Node positions are generated with a
  multidimensional-scaling embedding search based on the paper's graph-derived
  distance matrices: each trial assigns random distances to connected node
  pairs, assigns a larger fixed distance to disjoint node pairs, and computes a
  3D MDS embedding. The default search uses the paper-style `[0, 1)` connected
  range with disjoint-pair distance `10`; if that search does not find a
  candidate with worst case rigidity index above `1e-4`, a fallback search
  evaluates several connected-distance ranges and disjoint-pair distances.
  Each MDS candidate is then rescaled so the mean structural edge length is
  `1.0`, which keeps model sizes close to the other built-in presets without
  changing the candidate's shape. The selected embedding is the candidate with
  the largest worst case rigidity index; when candidates have nearly identical
  rigidity index values, the tie-breaker is the smaller RMS error from unit
  structural edge lengths. Generated embeddings are finally transformed into a
  reproducible frame: a support face through `node_1` lies on `z = 0`,
  `node_1` is at the origin, and the second-highest numbered node on that face
  lies on the positive x axis.
- `get_henneberg_routed_graph_definition(node_count, tube_count, scale=1.0, preset_index=1)`
  returns a curated routed continuous-tendon graph generated from Henneberg H1
  and H2 moves starting at `K4`. Named presets are
  indexed as `henneberg_n{nodes}_{tubes}tube_{index}`, for example
  `henneberg_n6_1tube_2`. The unsuffixed names such as `henneberg_n6_1tube`
  are aliases for variant `_1`. Available variant counts are:
  `henneberg_n5_1tube_1`; `henneberg_n6_1tube_1` through `_2`;
  `henneberg_n6_2tube_1`; `henneberg_n6_3tube_1`;
  `henneberg_n7_1tube_1` through `_10`; `henneberg_n7_3tube_1` through `_3`;
  `henneberg_n8_1tube_1` through `_85`; `henneberg_n8_2tube_1` through `_190`;
  and `henneberg_n8_3tube_1` through `_89`.
  Henneberg candidates are filtered to minimally rigid edge counts
  (`3 * nodes - 6`), deduplicated with NetworkX graph isomorphism checks, and
  classified by the minimum number of trails required to cover every edge.
  `2tube` and `3tube` presets use equal edge counts per routed tendon. Each
  candidate embedding starts from `networkx.spring_layout(..., dim=3)`, is
  refined against unit edge-length and separated non-edge targets, and is
  accepted only if the selected coordinates pass an infinitesimal-rigidity rank
  gate using the 3D rigidity matrix. Worst case rigidity index is still used to
  choose the best embedding candidate, but it is a quality score rather than the
  hard indexed-preset acceptance criterion. The final coordinates use the same
  reproducible `node_1` ground-face frame as the Usevitch graph presets.
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
  of the adjacent edges. `connector_rod_length` is an absolute nominal length;
  for regular triangles it is the length of every rod, while irregular custom
  triangles use it as the mean rod length so each triangle remains rigid and
  its tendon geometry is not distorted. The default is absolute and independent
  of preset scale. Set `connector_rod_length=None` to use the legacy
  scale-proportional `realistic_node_clone_offset` behavior. In routed shape
  dictionaries, passive route endpoints
  are rendered as cylinders with their flat-face normal aligned to the connector
  rod and diameter matched to the edge tendon width. The routed-tube
  `realistic=True` path is still incomplete and should be treated as a known
  modeling limitation until the routed connector/body geometry is fixed.

Generated robot geoms collide with the ground but not with one another. This
ground-only profile avoids redundant internal contacts and is compatible with
MJX's supported collision pairs.

Realistic triangle models include one accelerometer sensor per generated node
site by default. Pass `accelerometer_config=AccelerometerConfig(...)` to
configure sensor fields such as `noise`, `cutoff`, `nsample`, `delay`, or
`name_prefix`; pass `accelerometer_config=None` to omit them.

## References

- Nathan Usevitch, Isaac Weaver, and James Usevitch. "Triangle-Decomposable
  Graphs for Isoperimetric Robots." arXiv:2505.01624, 2025.
  <https://arxiv.org/abs/2505.01624>
