# Development

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
├── docs/
├── examples/
├── src/
│   ├── generate_mujoco_model.py
│   └── mujoco_truss_gen/
│       ├── mesh_import/
│       └── mujoco_model/
├── tests/
├── LICENSE
├── README.md
├── pyproject.toml
└── uv.lock
```

The package code lives under `src/mujoco_truss_gen`. Tests live under `tests`.
User-facing examples live under `examples`.
