# mujoco-truss-gen

`mujoco-truss-gen` is a Python package for generating MuJoCo models and
Gymnasium-style environments for triangle-based isoperimetric truss robots.

The package is intended for members of the isoperimetric robot research workflow
who need a shared, installable source of MuJoCo robot models instead of copying
model-generation code between reinforcement learning, planning, simulation, and
optimization projects.

## Project Status

This repository is an internal lab prototype. It has a working installable
package, a small public API, built-in polyhedron presets, and tests that verify
basic model generation and environment stepping. The API may still change before
the package is treated as stable research infrastructure.

Current scope:

- Generate MuJoCo `MjSpec` models for triangle-based truss structures.
- Generate built-in octahedron and icosahedron robot presets.
- Build either an abstract per-node slide-joint model or a more realistic
  triangle-body model with connector balls for shared nodes.
- Add tendon actuators and perimeter constraints.
- Save generated MuJoCo XML.
- Wrap generated models in Gymnasium-compatible environments.
- Provide base, relative-observation, and velocity-command environment variants.

Known limitations:

- The named preset registry is intentionally small: `"octahedron"` and
  `"icosahedron"` are included.
- Custom robot definitions are supported through dictionaries.
- The default rewards are research defaults, not task-independent objectives.
- The environment classes are starting points. Most RL, planning, or
  optimization tasks should subclass or wrap them for task-specific observations,
  rewards, resets, and termination logic.
- The human viewer requires a Python environment where `mujoco.viewer` is
  available.
- It is only possible to create custom trusses that are made up of independant
  triangles. Right now it is not possible to create a truss structure that
  includes triangles that share edges.


Future Work:
- Make it so for any continuous tube structure we can generate a valid MuJoCo model without having to manually define all of the active edges. This seems like it can be achomplished by simply alternating between active and inactive edges as we traverse the tube. 
- Maybe it would be cool to make it possible to add rigid elements between sets of nodes to allow the simulation of structures like the Treg Rover. 
- I think it would be cool to create some process which allows us to import any step file and it can automatically generate some feasable truss structure to approximate that shape.
  

## Installation

After a release has been published to PyPI:

```bash
python -m pip install mujoco-truss-gen
```

To upgrade to the newest published version:

```bash
python -m pip install --upgrade mujoco-truss-gen
```

For local development from a clone:

```bash
git clone https://github.com/isaa-sudweeks/mujoco-truss-gen.git
cd mujoco-truss-gen
python -m pip install -e ".[dev]"
```

The package requires Python 3.10 or newer and installs these runtime
dependencies:

- `gymnasium`
- `mujoco`
- `numpy`
- `scipy`

## Publishing Releases

PyPI publishing is automated through the GitHub Actions workflow in
`.github/workflows/publish.yml`. The workflow runs when a GitHub Release is
published, and it can also be started manually from the GitHub Actions tab.

The workflow:
For small test releases, you can use a pre-release tag (e.g., `0.1.0a1`).

For bug fixes or backwards-compatible changes, you can use a patch release tag (e.g., `0.1.1`).

For new features or breaking changes, you can use a minor or major release tag (e.g., `0.2.0` or `1.0.0`).

- Installs the package test dependencies.
- Runs `python -m pytest`.
- Builds the source distribution and wheel with `python -m build`.
- Checks the built distributions with `python -m twine check dist/*`.
- Publishes the distributions to PyPI.

The publish job uses PyPI Trusted Publishing, so the repository does not need a
long-lived PyPI API token in GitHub Secrets. PyPI must be configured to trust
this GitHub Actions workflow before the first automated release. For the
existing `mujoco-truss-gen` PyPI project, add a GitHub Actions trusted publisher
with these settings:

- PyPI project name: `mujoco-truss-gen`
- GitHub repository owner: `isaa-sudweeks`
- GitHub repository name: `mujoco-truss-gen`
- Workflow filename: `publish.yml`
- GitHub environment name: `pypi`

To publish a new version:

1. Update `version` in `pyproject.toml`.
2. Commit and push the change.
3. Create and publish a GitHub Release for that commit.
4. Confirm the `Publish to PyPI` workflow passes.

PyPI versions are immutable. If a release workflow fails after uploading a
version, fix the issue, bump `version` again, and publish a new release.

## Publishing Releases Manually

Release checklist:

1. Update `version` in `pyproject.toml`.
2. Run `python -m pytest`.
3. Run `python -m ruff check .`.
4. Run `python -m ruff format --check .`.
5. Build distributions with `python -m build`.
6. Upload with `python -m twine upload dist/*`.
7. Verify installation in a clean environment with
   `python -m pip install mujoco-truss-gen`.

Users update to the newest published package with:

```bash
python -m pip install --upgrade mujoco-truss-gen
```

## Quick Start

Generate the built-in octahedron model:

```python
from mujoco_truss_gen import get_mujoco_spec

spec = get_mujoco_spec("octahedron", realistic=False)
model = spec.compile()
```

Save the generated XML:

```python
from mujoco_truss_gen import get_mujoco_spec, save_xml

spec = get_mujoco_spec("octahedron", realistic=False)
xml_path = save_xml(spec, "octahedron.xml")
```

Run one Gymnasium step:

```python
import numpy as np

from mujoco_truss_gen import MujocoRelativeObsEnv, TrussEnvConfig, get_mujoco_spec

spec = get_mujoco_spec("octahedron", realistic=False)
env = MujocoRelativeObsEnv(
    TrussEnvConfig(
        model_source=spec,
        max_steps=1_000,
        nsubsteps=4,
        speed=0.01,
    )
)

obs, info = env.reset(seed=0)
action = np.zeros(env.action_space.shape, dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

View a generated model programmatically using the passive viewer:

```python
from mujoco_truss_gen import get_mujoco_spec, view

spec = get_mujoco_spec("octahedron", realistic=False)
view(spec)
```

> [!NOTE]
> On macOS, `mujoco.viewer.launch_passive` requires the script to be run with `mjpython` instead of the standard `python` executable:
> ```bash
> mjpython your_script.py
> ```

Alternatively, open the built-in octahedron model from the command line:

```bash
python -m mujoco_truss_gen.generate_mujoco_model
```

## Defining a Custom Truss

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

Continuous tube shapes can also be represented with a shape dictionary. Each
shape defines a routed path and the route edges that should be actuated:

```python
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
Active edges receive actuators. A route-length equality constraint keeps the
total routed length constant, so non-actuated passive edges absorb length
changes caused by the active edges. Routed shape dictionaries currently support
`realistic=False`.

`get_mujoco_spec()` and `build_triangle()` treat caller-provided dictionaries as
read-only inputs. The realistic builder clones shared nodes internally, but it
does not mutate the original `node_dict` or `triangle_dict` passed by the caller.

## Graph Neural Network (GNN) Utilities

You can extract node features and edge indices directly from a spec or model, formatted specifically for use with PyTorch Geometric (`torch_geometric.data.Data`):

```python
import torch
from mujoco_truss_gen import get_mujoco_spec, get_edge_index, get_node_features

spec = get_mujoco_spec("octahedron", realistic=False)

# Get edge indices in PyG COO format: shape (2, num_directed_edges)
edge_index = get_edge_index(spec)

# Get node features (position and velocity): shape (num_nodes, 6)
node_features = get_node_features(spec)

# Convert to PyTorch tensors
edge_index_tensor = torch.from_numpy(edge_index)
x_tensor = torch.from_numpy(node_features)
```

## Model Generation Contract

Public generation helpers:

- `build_world()` creates a base `mujoco.MjSpec` containing a ground plane and
  top light.
- `build_triangle(spec, node_dict, triangle_dict, realistic=False)` adds truss
  bodies, sites, tendons, actuators, and perimeter constraints to an existing
  spec.
- `build_shapes(spec, node_dict, shape_dict, realistic=False)` adds routed
  continuous-tube shapes with per-edge tendons and route-length constraints.
- `get_mujoco_spec("octahedron", realistic=False)` and
  `get_mujoco_spec("icosahedron", realistic=False)` build built-in presets.
- `get_mujoco_spec(node_dict, triangle_dict, realistic=False)` builds a custom
  dictionary-defined truss.
- `get_mujoco_spec(node_dict, shape_dict, realistic=False)` builds a routed
  continuous-tube shape model.
- `get_octahedron_definition()` returns fresh node and triangle dictionaries for
  the built-in preset.
- `get_icosahedron_definition()` returns fresh node and triangle dictionaries
  for the built-in preset.
- `get_perimeter(node_dict, triangle_dict)` computes each triangle perimeter
  from the first three vertices.
- `get_edge_index(source)` extracts structural connectivity into a PyTorch Geometric compatible `(2, E)` numpy array.
- `get_node_features(source)` extracts node positions and velocities into a PyTorch Geometric compatible `(N, 6)` numpy array.
- `save_xml(spec, filename)` writes `spec.to_xml()` to disk and returns the
  resolved path.
- `view(spec)` compiles and opens the generated model in MuJoCo's passive
  viewer.

Input expectations:

- Node names should be unique strings. Names beginning with `node_` are required
  for the built-in metadata and environment helpers.
- Node positions must be 3D numeric sequences.
- Triangle entries must contain exactly the three vertex nodes plus one passive
  node.
- The passive node must appear in that triangle's first three vertices.
- Shape entries must contain `route` and `active_edges` keys.
- Shape routes must contain at least two node names. Each active edge must be an
  adjacent pair in the route.
- The builder helpers should be used when environment rigidity and slip helpers
  are needed, because those helpers infer structure from generated body, site,
  tendon, and actuator names.

Model modes:

- `realistic=False` creates one world-body per node with slide joints on `x`,
  `y`, and `z`. This is the simpler abstract model and is useful for fast
  algorithm development.
- `realistic=True` creates triangle bodies, clones shared triangle nodes inside
  the generated model, and connects shared vertices through connector balls.
  This is intended to better represent the triangle-module structure.

## Environment Contract

The environment constructors accept any of these model sources:

- `mujoco.MjSpec`
- `mujoco.MjModel`
- XML string
- path to an XML file
- `TrussEnvConfig`

Available environments:

- `MujocoTrussEnv`: base environment with tendon lengths, tendon velocities,
  center-of-mass position, and center-of-mass velocity in the observation.
- `MujocoRelativeObsEnv`: relative node-position observations and normalized
  actuator delta actions.
- `MujocoVelocityCommandEnv`: relative observations with direct velocity command
  actions.

Shared configuration is provided by `TrussEnvConfig`:

```python
from mujoco_truss_gen import TrussEnvConfig

config = TrussEnvConfig(
    model_source=spec,
    max_steps=10_000,
    nsubsteps=1,
    speed=0.01,
    forward_weight=5.0,
    energy_weight=0.005,
    alive_bonus=0.1,
    rigidity_weight=0.5,
    slip_weight=0.1,
    critical_eig_threshold=0.03,
    slip_height=0.2,
    control_noise_std=0.0,
    control_noise_relative=True,
    runtime_apply_control_noise=False,
)
```

Step/reset behavior:

- `reset(seed=...)` follows the Gymnasium API and returns `(obs, info)`.
- `step(action)` returns `(obs, reward, terminated, truncated, info)`.
- `truncated` becomes true when `max_steps` is reached.
- `terminated` becomes true when the normalized rigidity metric falls below
  `critical_eig_threshold`.
- `info` includes reward components and `critical_eig`.

Action behavior:

- `MujocoTrussEnv` sends clipped actuator controls directly in the MuJoCo
  actuator control range.
- `MujocoRelativeObsEnv` expects actions in `[-1, 1]`; each action component
  changes the previous control by `action * config.speed`.
- `MujocoVelocityCommandEnv` expects actions in `[-config.speed, config.speed]`
  and sends those values directly.

Reward behavior:

- The default reward combines forward velocity, alive bonus, energy penalty,
  rigidity reward, and slip penalty.
- These defaults are provided for experimentation, not as a canonical objective
  for every isoperimetric robot task.
- Custom tasks should subclass an environment and override `_get_obs()`,
  `_compute_reward()`, `reset()`, or `step()` as needed.

Rendering:

- `render_mode="rgb_array"` returns a rendered NumPy RGB image.
- `render_mode="human"` opens a passive MuJoCo viewer when the local MuJoCo
  viewer module is available.

## Development

Set up a development environment:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

Run linting and formatting checks:

```bash
python -m ruff check .
python -m ruff format --check .
```

Build a local distribution:

```bash
python -m build
```


## Repository Layout

```text
mujoco-truss-gen/
├── LICENSE
├── README.md
├── pyproject.toml
├── tests/
│   └── test_envs.py
└── src/
    ├── generate_mujoco_model.py
    └── mujoco_truss_gen/
        ├── __init__.py
        ├── base_env.py
        ├── generate_mujoco_model.py
        ├── relative_observation_env.py
        ├── velocity_command_env.py
        └── mujoco_model/
            ├── bodies.py
            ├── builders.py
            ├── constants.py
            ├── constraints.py
            ├── geometry.py
            ├── io_viewer.py
            ├── model.py
            ├── model_types.py
            ├── presets.py
            └── tendons.py
```

## Citation

There is no formal citation for this package yet.

## License

This project is distributed under the BSD-3-Clause license. See `LICENSE` for
the full license text.
