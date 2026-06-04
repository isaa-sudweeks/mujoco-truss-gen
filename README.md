# mujoco-truss-gen

`mujoco-truss-gen` is a Python package for generating MuJoCo models and
Gymnasium-style environments for triangle-based isoperimetric truss robots.

The package is intended for members of the isoperimetric robot research workflow
who need a shared, installable source of MuJoCo robot models instead of copying
model-generation code between reinforcement learning, planning, simulation, and
optimization projects.

## Status

This repository is an internal lab prototype. It has a working installable
package, built-in structure presets, a small public API, and tests that verify
basic model generation and environment stepping. The API may still change before
the package is treated as stable research infrastructure.

Supported workflows include:

- Generate MuJoCo `MjSpec` models for triangle-based truss structures.
- Use built-in `"octahedron"`, `"icosahedron"`, `"solar_array"`,
  `"tetrahedron"`, and Usevitch et al. triangle-decomposable graph presets.
- Build abstract slide-joint models or realistic triangle-body models.
- Save generated MuJoCo XML.
- Wrap generated models in Gymnasium-compatible environments.
- Convert STL meshes into experimental routed-tube shape dictionaries.

## Installation

After a release has been published to PyPI:

```bash
python -m pip install mujoco-truss-gen
```

For local development from a clone:

```bash
git clone https://github.com/isaa-sudweeks/mujoco-truss-gen.git
cd mujoco-truss-gen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The package requires Python 3.10 or newer and installs these runtime
dependencies:

- `gymnasium`
- `mujoco`
- `numpy`
- `scipy`

## Quick Start

Generate the built-in octahedron model:

```python
from mujoco_truss_gen import get_mujoco_spec

spec = get_mujoco_spec("octahedron", realistic=False)
model = spec.compile()
```

Save generated XML:

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

Open the built-in octahedron model from the command line:

```bash
python -m mujoco_truss_gen.generate_mujoco_model
```

On macOS, MuJoCo's passive viewer may require running viewer scripts with
`mjpython` instead of the standard `python` executable.

Routed continuous-tube presets such as `tetrahedron` are unconstrained
all-edge-actuated models. They also support `realistic=True`, which clones
shared routed node occurrences and connects them through in-plane bisector rods.
Use `MujocoNodeVelocityCommandEnv` for node-level scalar velocity commands that
are mapped through the route incidence matrix to edge actuator commands. For
manual testing, `view_node_velocity_terminal(spec)` opens the MuJoCo viewer and
accepts terminal commands such as `set node_2 0.01`, `show`, `zero`, and `quit`.

## Documentation

- [Model generation](docs/model-generation.md): custom trusses, routed shape
  dictionaries, model modes, and generation helper contracts.
- [Environments](docs/environments.md): environment constructors, actions,
  observations, rewards, rendering, and `TrussEnvConfig`.
- [STL import](docs/stl-import.md): optional STL-to-routed-tube conversion and
  preview behavior.
- [GNN utilities](docs/gnn-utilities.md): extracting graph features and edge
  indices for PyTorch Geometric workflows.
- [Development](docs/development.md): local setup, tests, linting, formatting,
  and package builds.
- [Releasing](docs/releasing.md): automated and manual PyPI release steps.
- [Roadmap](docs/roadmap.md): known limitations and planned work.

## Example

Start from the included custom-truss example:

```bash
python examples/custom_truss.py
```

## Citation

There is no formal citation for this package yet.

The Usevitch graph presets are based on:

Nathan Usevitch, Isaac Weaver, and James Usevitch. "Triangle-Decomposable
Graphs for Isoperimetric Robots." arXiv:2505.01624, 2025.
<https://arxiv.org/abs/2505.01624>

## License

This project is distributed under the BSD-3-Clause license. See `LICENSE` for
the full license text.
