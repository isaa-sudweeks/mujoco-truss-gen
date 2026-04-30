# mujoco-truss-gen

A Python library for generating MuJoCo/Gym environments for arbitrary configurations of isoperimetric robots.

This repository is currently in an early skeleton stage. The main implemented code builds MuJoCo model specifications for triangle-based truss structures, including an octahedron example.

## Project Status

TBD: Add the current development status, research goals, and any known limitations.

Known today:

- Source code lives under `src/mujoco-truss-gen/`.
- `pyproject.toml` exists but has not been populated yet.
- The current generator script is `src/mujoco-truss-gen/generate_mujoco_model.py`.
- The generator uses MuJoCo, NumPy, and SciPy.

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

Planned / TBD:

- Gymnasium or Gym environment wrapper.
- Package metadata and install instructions.
- Tests and validation examples.
- Documentation for custom robot definitions.
- Examples for training or controlling generated environments.

## Repository Layout

```text
mujoco-truss-gen/
├── README.md
├── pyproject.toml
└── src/
    └── mujoco-truss-gen/
        ├── __init__.py
        └── generate_mujoco_model.py
```

## Requirements

The current generator imports:

- Python 3.10+ expected, based on current type-hint syntax.
- `mujoco`
- `numpy`
- `scipy`

TBD: Add the exact supported Python versions and pinned dependency ranges once `pyproject.toml` is configured.

## Installation

TBD: Add package installation instructions after `pyproject.toml` is populated.

For local development, the project will likely use a standard editable install:

```bash
pip install -e .
```

## Quick Start

TBD: Confirm the final import path once the package layout is finalized.

The current script can build and view the built-in octahedron example when run directly:

```bash
python src/mujoco-truss-gen/generate_mujoco_model.py
```

The current high-level API in `generate_mujoco_model.py` is:

```python
node_dict, triangle_dict = get_octahedron_definition()
spec = build_world()
build_triangle(spec, node_dict, triangle_dict, realistic=False)
```

To create a complete MuJoCo spec for the built-in structure:

```python
spec = get_mujoco_spec("octahedron", realistic=False)
```

To save XML:

```python
save_xml(spec, "model.xml")
```

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

Current public helpers in `generate_mujoco_model.py` include:

- `build_world()`: create the base MuJoCo world.
- `build_triangle(spec, node_dict, triangle_dict, realistic=False)`: add truss bodies, tendons, actuators, and constraints.
- `get_octahedron_definition()`: return the built-in octahedron node and triangle dictionaries.
- `get_mujoco_spec(...)`: build a complete spec from dictionaries or the `"octahedron"` preset.
- `get_perimeter(node_dict, triangle_dict)`: compute perimeters for each triangle.
- `save_xml(spec, filename)`: write the generated MuJoCo XML to disk.
- `view(spec)`: compile and inspect the model in the MuJoCo passive viewer.

TBD: Decide which helpers should be considered stable public API before publishing.

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

- Configure package metadata in `pyproject.toml`.
- Rename or expose the package with a Python-importable module name if needed.
- Add automated tests for geometry generation and MuJoCo spec creation.
- Add examples for abstract and realistic model generation.
- Add Gymnasium-compatible environment classes.
- Document actuator, tendon, and constraint assumptions.

## Citation

TBD: Add citation information if this repository supports a paper, thesis, or research artifact.

## License

TBD: Add license information.
