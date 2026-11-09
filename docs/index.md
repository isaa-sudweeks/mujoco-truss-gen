# Start Here

`mujoco-truss-gen` builds MuJoCo models of triangle-based isoperimetric truss
robots. Use the compiled models directly with MuJoCo or wrap them in a Gymnasium
environment.

- [Simulate your first robot](first-robot.md) to build an octahedron, change
  its physical parameters, and watch it move in the MuJoCo viewer.
- [Model generation](model-generation.md) covers custom geometry, routed shapes,
  and abstract and realistic models.
- [Environments](environments.md) covers the Gymnasium and batched MJX interfaces.

## Choose a simulation workflow

| Goal | Start with |
| --- | --- |
| Inspect a generated robot and watch it settle | [Built-in viewer](first-robot.md#2-build-and-view-the-robot) |
| Apply your own commands or run on a machine without a display | [Scripted CPU simulation](first-robot.md#3-run-a-scripted-simulation) |
| Read robot state, send commands, or read accelerometers | [State, actions, and sensors](state-actions-sensors.md) |
| Change mass, geometry, or actuator settings for a run | [Physical parameters](physical-parameters.md) |
| Use observations, actions, rewards, or batched simulation | [Environments](environments.md) |

## Browser tools

- [Preset catalog](https://isaa-sudweeks.github.io/mujoco-truss-gen/)
- [Terrain and environment calculator](https://isaa-sudweeks.github.io/mujoco-truss-gen/terrain.html)
- [Training distribution](https://isaa-sudweeks.github.io/mujoco-truss-gen/distribution.html)
