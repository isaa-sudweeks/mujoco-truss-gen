# Roadmap

This repository is an internal lab prototype. It has a working installable
package, a small public API, built-in polyhedron presets, and tests that verify
basic model generation and environment stepping. The API may still change before
the package is treated as stable research infrastructure.

## Current Scope

- Generate MuJoCo `MjSpec` models for triangle-based truss structures.
- Generate built-in octahedron, icosahedron, solar-array, tetrahedron, and
  Usevitch et al. triangle-decomposable graph presets.
- Build either an abstract per-node slide-joint model or a more realistic
  triangle-body model with connector balls, rods, and face-aligned shared-node
  connectors.
- Add tendon actuators and perimeter constraints.
- Save generated MuJoCo XML.
- Wrap generated models in Gymnasium-compatible environments.
- Provide base, relative-observation, and velocity-command environment variants.
- Convert STL meshes into experimental routed-tube shape dictionaries.

## Known Limitations

- The named preset registry includes hand-authored base structures plus
  generated Usevitch et al. graph presets named by paper graph label.
- Custom robot definitions are supported through dictionaries.
- The default rewards are research defaults, not task-independent objectives.
- The environment classes are starting points. Most reinforcement learning,
  planning, or optimization tasks should subclass or wrap them for task-specific
  observations, rewards, resets, and termination logic.
- The human viewer requires a Python environment where `mujoco.viewer` is
  available.
- Custom trusses are currently represented as independent triangles or routed
  shape paths.
- Routed shape paths are currently practical only for short paths. Longer paths
  likely need a higher-level node-command abstraction that maps node commands to
  edge commands internally.
- The routed-tube `realistic=True` generation path is not fully correct yet.
  It should be treated as experimental until the routed connector/body geometry
  is fixed.
- The STL-to-MuJoCo path is experimental and can produce unstable generated
  models.

## Future Work

- Add a higher-level command abstraction for continuous tube structures so users
  do not need to define every active edge manually.
- Add stability checks or repair passes for STL-derived routed graphs.
- Add rigid elements between sets of nodes to support structures such as the
  Treg Rover.
