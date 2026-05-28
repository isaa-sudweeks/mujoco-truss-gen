# Environments

The package provides Gymnasium-compatible environments around generated MuJoCo
truss models.

## Model Sources

Environment constructors accept any of these model sources:

- `mujoco.MjSpec`
- `mujoco.MjModel`
- XML string
- path to an XML file
- `TrussEnvConfig`

## Environment Classes

- `MujocoTrussEnv`: base environment with tendon lengths, tendon velocities,
  center-of-mass position, and center-of-mass velocity in the observation.
- `MujocoRelativeObsEnv`: relative node-position observations and normalized
  actuator delta actions.
- `MujocoVelocityCommandEnv`: relative observations with direct velocity command
  actions.
- `MujocoNodeVelocityCommandEnv`: relative observations with routed-tube
  node-level scalar velocity commands mapped to edge actuators.

## Shared Configuration

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

## Step and Reset Behavior

- `reset(seed=...)` follows the Gymnasium API and returns `(obs, info)`.
- `step(action)` returns `(obs, reward, terminated, truncated, info)`.
- `truncated` becomes true when `max_steps` is reached.
- `terminated` becomes true when the normalized rigidity metric falls below
  `critical_eig_threshold`.
- `info` includes reward components and `critical_eig`.

## Actions

- `MujocoTrussEnv` sends clipped actuator controls directly in the MuJoCo
  actuator control range.
- `MujocoRelativeObsEnv` expects actions in `[-1, 1]`; each action component
  changes the previous control by `action * config.speed`.
- `MujocoVelocityCommandEnv` expects actions in `[-config.speed, config.speed]`
  and sends those values directly.
- `MujocoNodeVelocityCommandEnv` expects one scalar per model node in
  `[-config.speed, config.speed]`. Nodes that are the start or end of any route
  are passive and are zeroed before control is applied. The environment
  multiplies the effective node command vector by the routed-tube oriented
  incidence matrix, where each edge command is
  `node_action[to_node] - node_action[from_node]`, then clips the result to the
  MuJoCo actuator control range.

```python
import numpy as np

from mujoco_truss_gen import (
    MujocoNodeVelocityCommandEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)

spec = get_mujoco_spec("tetrahedron", realistic=False)
env = MujocoNodeVelocityCommandEnv(TrussEnvConfig(spec, speed=0.01))
obs, _ = env.reset(seed=1)

action = np.zeros(env.action_space.shape, dtype=np.float32)
action[1] = 0.01
obs, reward, terminated, truncated, info = env.step(action)
```

## Rewards

The default reward combines forward velocity, alive bonus, energy penalty,
rigidity reward, and slip penalty. These defaults are provided for
experimentation, not as a canonical objective for every isoperimetric robot
task.

Custom tasks should subclass an environment and override `_get_obs()`,
`_compute_reward()`, `reset()`, or `step()` as needed.

## Rendering

- `render_mode="rgb_array"` returns a rendered NumPy RGB image.
- `render_mode="human"` opens a passive MuJoCo viewer when the local MuJoCo
  viewer module is available.
- `view(spec)` compiles a generated spec and opens the standard MuJoCo passive
  viewer.
- `view_node_velocity_terminal(spec)` opens the MuJoCo viewer for a routed
  continuous-tube model and reads node-level scalar velocity commands from the
  terminal. Each node command is mapped through `NodeVelocityController` into
  routed tendon actuator commands every simulation step.

```python
from mujoco_truss_gen import get_mujoco_spec, view_node_velocity_terminal

spec = get_mujoco_spec("tetrahedron", realistic=False)
view_node_velocity_terminal(spec, speed=0.01)
```

While the viewer is running, enter commands at the `node>` prompt:

```text
nodes                 # list node indices and names
set node_2 0.01       # set a node command by name
1 -0.005              # shorthand: set node index 1 to -0.005
add node_2 0.002      # increment a node command
show                  # print current node and tendon commands
zero                  # reset all node commands to 0
quit                  # close the control loop
```

Node command values are clipped to `[-speed, speed]`. Route endpoints are
passive and remain zero. `view_node_velocity(spec)` is the Tk slider-panel
variant, but it is not supported on macOS builds where Tk and `mjpython`
conflict; use `view_node_velocity_terminal(spec)` for local testing there.
