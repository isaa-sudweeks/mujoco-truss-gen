# mujoco-truss-gen

A Python library for generating MuJoCo/Gymnasium environments for arbitrary configurations of isoperimetric robots.

The main implemented code builds MuJoCo model specifications for triangle-based truss structures, including an octahedron example, and provides Gymnasium-compatible environment wrappers around generated specs.

## Project Status

TBD: Add the current development status, research goals, and any known limitations.

Known today:

- Source code lives under `src/mujoco_truss_gen/`.
- `pyproject.toml` defines packaging metadata and the console script entry point.
- MuJoCo model generation code lives under `src/mujoco_truss_gen/mujoco_model/`.
- `src/generate_mujoco_model.py` remains as a temporary compatibility shim.
- The generator and environments use Gymnasium, MuJoCo, NumPy, and SciPy.

## Features

Implemented or partially implemented:

- Build a MuJoCo `MjSpec` world with a plane and top light.
- Define node and triangle dictionaries for truss structures.
- Generate an octahedron truss definition.
- Build abstract triangle-based truss models.
- Build a more realistic triangle model with cloned shared nodes and connector balls.
- Add tendons, tendon actuators, and perimeter constraints.
- Compute triangle perimeters.
- Save generated MuJoCo XML.
- Launch the MuJoCo passive viewer for inspection.
- Wrap generated specs in Gymnasium environments.
- Provide relative-position and velocity-command environment variants.

Planned / TBD:

- More tests and validation examples.
- Documentation for custom robot definitions.
- Examples for training or controlling generated environments.

## Repository Layout

```text
mujoco-truss-gen/
├── README.md
├── pyproject.toml
└── src/
    ├── generate_mujoco_model.py
    └── mujoco_truss_gen/
        ├── __init__.py
        ├── base_env.py
        ├── generate_mujoco_model.py
        ├── relative_observation_env.py
        ├── velocity_command_env.py
        └── mujoco_model/
            ├── __init__.py
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

## Requirements

The package imports:

- Python 3.10+ expected, based on current type-hint syntax.
- `gymnasium`
- `mujoco`
- `numpy`
- `scipy`

## Installation

For local development, install the package in editable mode:

```bash
pip install -e .
```

## Quick Start

The current script can build and view the built-in octahedron example when run directly:

```bash
PYTHONPATH=src mjpython -m mujoco_truss_gen.generate_mujoco_model
```

The current high-level API is available from `mujoco_truss_gen`:

```python
from mujoco_truss_gen import build_triangle, build_world, get_octahedron_definition

node_dict, triangle_dict = get_octahedron_definition()
spec = build_world()
build_triangle(spec, node_dict, triangle_dict, realistic=False)
```

To create a complete MuJoCo spec for the built-in structure:

```python
from mujoco_truss_gen import get_mujoco_spec

spec = get_mujoco_spec("octahedron", realistic=False)
```

To save XML:

```python
from mujoco_truss_gen import save_xml

save_xml(spec, "model.xml")
```

To run the generated model as a Gymnasium environment:

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

The environment constructors accept a generated `mujoco.MjSpec`, a compiled `mujoco.MjModel`, an XML string, or a path to an XML file.

## Defining a Truss

Trusses are currently represented with two dictionaries:

- `node_dict`: maps node names to 3D positions.
- `triangle_dict`: maps triangle names to a list of four node names.

The first three names in each triangle define the triangle vertices. The fourth name marks the passive node for perimeter constraints.

Example from the built-in octahedron:

```python
node_dict = {
    "node_1": [0.0, 0.0, 0.1],
    "node_2": [1.0, 0.0, 0.1],
    "node_3": [0.5, 0.8660, 0.1],
}

triangle_dict = {
    "triangle_1": ["node_1", "node_2", "node_3", "node_1"],
}
```

TBD: Add a complete custom robot example and explain the passive-node convention in more detail.

## Main API Surface

Current public helpers exported from `mujoco_truss_gen` include:

- `TrussEnvConfig`: shared configuration for the provided environments.
- `MujocoModel`: compiled MuJoCo model wrapper with truss metadata and query helpers.
- `MujocoTrussEnv`: base Gymnasium environment with absolute tendon and center-of-mass observations.
- `MujocoRelativeObsEnv`: Gymnasium environment with COM-relative node positions and actuator delta actions.
- `MujocoVelocityCommandEnv`: relative-observation environment with direct actuator velocity commands.
- `build_world()`: create the base MuJoCo world.
- `build_triangle(spec, node_dict, triangle_dict, realistic=False)`: add truss bodies, tendons, actuators, and constraints.
- `get_octahedron_definition()`: return the built-in octahedron node and triangle dictionaries.
- `get_mujoco_spec(...)`: build a complete spec from dictionaries or the `"octahedron"` preset.
- `get_perimeter(node_dict, triangle_dict)`: compute perimeters for each triangle.
- `save_xml(spec, filename)`: write the generated MuJoCo XML to disk.
- `view(spec)`: compile and inspect the model in the MuJoCo passive viewer.

The environment classes are intended as starting points. For custom tasks, subclass `MujocoTrussEnv` or one of its variants and override `_get_obs()`, `_compute_reward()`, or `step()` as needed.

## Environment Design Notes

The generator/environment split is the intended boundary:

- Use `get_mujoco_spec(...)`, `build_world()`, and `build_triangle(...)` to produce the robot model.
- Pass the resulting spec, compiled model, XML string, or XML path into an environment.
- Use `TrussEnvConfig` for shared training knobs such as episode length, substeps, action speed, reward weights, and optional control noise.
- Subclass the environment layer for task-specific observations, actions, rewards, and termination conditions.

Important assumptions:

- Generated bodies representing truss nodes should keep names beginning with `node_`.
- Generated structural tendons and sites should come from this package's builder helpers if you want rigidity and slip helper methods to work automatically.
- `MujocoRelativeObsEnv` uses normalized delta actions; `MujocoVelocityCommandEnv` sends direct actuator controls.
- The base reward is a research default, not a universal objective. Most training runs should tune or override `_compute_reward()`.

## Development

TBD: Add development setup, formatting, linting, testing, and contribution instructions.

Suggested sections to fill in later:

- Environment setup
- Running tests
- Generating example XML
- Running a viewer demo
- Adding a new robot preset
- Release process

## Roadmap

TBD: Replace this checklist with current research and engineering priorities.

- Add automated tests for geometry generation and MuJoCo spec creation.
- Add examples for abstract and realistic model generation.
- Document actuator, tendon, and constraint assumptions.

## Citation

TBD: Add citation information if this repository supports a paper, thesis, or research artifact.

## License

TBD: Add license information.
