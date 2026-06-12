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
from mujoco_truss_gen import DomainRandomizationConfig, TrussEnvConfig

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
    normalize_observations=False,
    domain_randomization=None,
)
```

Set `normalize_observations=True` to divide observed coordinate components by
the matching dimension of the robot's initial node-position bounding box. For
example, observed x positions are divided by the initial x span. In the
relative-observation envs, node velocity components are normalized the same
way. Zero-width axes use a divisor of `1.0`.

## Domain Randomization

Use `DomainRandomizationConfig` to sample a new domain on each environment
reset. The sampled values remain fixed for that episode.

Runtime randomization mutates fields on the compiled MuJoCo model and is the
cheapest option:

```python
from mujoco_truss_gen import (
    DomainRandomizationConfig,
    MujocoTrussEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)

spec = get_mujoco_spec("octahedron", realistic=False)
env = MujocoTrussEnv(
    TrussEnvConfig(
        spec,
        domain_randomization=DomainRandomizationConfig(
            body_mass_multiplier_range=(0.8, 1.2),
            body_inertia_multiplier_range=(0.8, 1.2),
            dof_damping_multiplier_range=(0.7, 1.3),
            actuator_gain_multiplier_range=(0.75, 1.25),
            actuator_bias_multiplier_range=(0.75, 1.25),
            geom_friction_slide_range=(0.4, 1.2),
            gravity_z_range=(-10.5, -8.8),
        ),
    )
)

obs, info = env.reset(seed=1)
print(info["domain_randomization"])
```

Use `model_factory` for changes that are baked into the compiled model, such as
scale, node locations, topology, or `TrussPhysicalParameters` used while
building the XML:

```python
import numpy as np

from mujoco_truss_gen import (
    DomainRandomizationConfig,
    MujocoTrussEnv,
    TrussEnvConfig,
    TrussPhysicalParameters,
    get_mujoco_spec,
)


def randomized_model(rng: np.random.Generator):
    scale = rng.uniform(0.75, 1.25)
    params = TrussPhysicalParameters(
        active_node_mass=rng.uniform(0.005, 0.02),
        passive_node_mass=rng.uniform(0.005, 0.02),
        realistic_actuator_kp=rng.uniform(700.0, 1300.0),
    )
    return get_mujoco_spec(
        "octahedron",
        realistic=True,
        scale=scale,
        physical_params=params,
    )


env = MujocoTrussEnv(
    TrussEnvConfig(
        get_mujoco_spec("octahedron", realistic=True),
        domain_randomization=DomainRandomizationConfig(
            model_factory=randomized_model,
            geom_friction_slide_range=(0.4, 1.2),
        ),
    )
)
```

When using vectorized Gymnasium environments, give each worker the same
randomization config. Each worker samples independently at reset, while the
vectorized setup provides parallel training throughput.

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

The default reward combines center-of-mass forward velocity, alive bonus,
energy penalty, rigidity reward, and slip penalty. The forward reward is
normalized by the diagonal length of the robot's initial node-position bounding
box, making the term more comparable across differently sized topologies.
`max_forward_velocity` clips this normalized velocity, in robot bounding-box
diagonals per second, unless set to `None`. Clipping after normalization gives
uniformly scaled robots the same forward-reward range.

When the truss crosses the collapse threshold, the default reward policy avoids
paying positive forward progress or alive bonus from the unstable terminal
state, zeroes terminal rigidity reward, and skips velocity-derived shaping such
as slip. Non-finite rigidity metrics are treated as collapse-terminal states.
`collapse_penalty` is interpreted as a penalty magnitude, so positive and
negative configured values both contribute a non-positive terminal reward. The
raw COM velocity, physical clipped velocity, raw and clipped normalized
velocities, COM displacement, raw and reward-safe rigidity metrics, collapse
flag, and reward components remain available in `info`. These defaults are
provided for experimentation, not as a canonical objective for every
isoperimetric robot task.

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
