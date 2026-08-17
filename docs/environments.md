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

## Terrain Model Sources

Terrain is part of the MuJoCo model source. Build a static terrain for either the
native or MJX environment by passing `terrain=TerrainConfig(...)` to
`get_mujoco_spec()`; see [Height-field terrain](terrain.md). Height-field resets
preserve initial local ground clearance after X/Y/yaw transforms, and slip shaping
uses local terrain height.

Native environments can use `DomainRandomizationConfig.model_factory` to rebuild
terrain on each reset. `MjxNodeVelocityEnv` owns one fixed compiled terrain and does
not support `model_factory`; use separate MJX environment instances for different
terrain configurations.

## Domain Randomization

Use `DomainRandomizationConfig` to sample a new domain on each environment
reset. The sampled values remain fixed for that episode.

Runtime randomization mutates fields on the compiled MuJoCo model and is the
cheapest option:

```python
import numpy as np

from mujoco_truss_gen import (
    DomainRandomizationConfig,
    MujocoTrussEnv,
    TrussEnvConfig,
    get_mujoco_spec,
)

spec = get_mujoco_spec("octahedron", realistic=True)
env = MujocoTrussEnv(
    TrussEnvConfig(
        spec,
        domain_randomization=DomainRandomizationConfig(
            body_mass_multiplier_range=(0.8, 1.2),
            body_inertia_multiplier_range=(0.8, 1.2),
            dof_damping_multiplier_range=(0.7, 1.3),
            dof_armature_range=(0.0, 0.02),
            dof_frictionloss_range=(0.0, 0.05),
            actuator_gain_multiplier_range=(0.75, 1.25),
            actuator_bias_multiplier_range=(0.75, 1.25),
            actuator_dynprm_multiplier_range=(0.75, 1.25),
            geom_friction_slide_range=(0.4, 1.2),
            geom_friction_torsional_range=(0.0001, 0.01),
            geom_friction_rolling_range=(0.0001, 0.01),
            tendon_stiffness_range=(0.0, 1.0),
            tendon_damping_range=(0.0, 1.0),
            tendon_armature_range=(0.0, 0.02),
            tendon_frictionloss_range=(0.0, 0.05),
            gravity_z_range=(-10.5, -8.8),
            initial_translation_x_range=(-0.5, 0.5),
            initial_translation_y_range=(-0.5, 0.5),
            initial_yaw_range=(-np.pi, np.pi),
        ),
    )
)

obs, info = env.reset(seed=1)
print(info["domain_randomization"])
```

Mass, inertia, and DOF damping ranges are global multipliers: they scale the
existing model arrays and preserve their nominal ratios. A DOF damping
multiplier has no effect when every nominal damping value is zero; constructing
an environment with that combination emits a warning. The simplified built-in
tetrahedron, octahedron, and icosahedron presets (`realistic=False`) have zero
nominal DOF damping, while their realistic variants have nonzero hinge damping.
Configure nominal joint damping or use a realistic preset when randomizing this
multiplier. Other zero-default DOF and tendon fields use absolute sampled
values. Runtime randomization intentionally
does not change actuator control ranges, force ranges, action-space bounds, or
the reset `qpos`/`qvel` perturbation.

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

Initial-pose ranges apply one rigid planar transform at every reset. Translation
uses MuJoCo world-length units and yaw uses radians. Yaw is about world Z through
the nominal node-position centroid, so relative geometry, node heights, and the
selected ground-contact face are preserved. Omit any range to disable that
component. MJX samples the three values independently for each batch element;
`reset_where` changes them only for masked elements.

## Step and Reset Behavior

- `reset(seed=...)` follows the Gymnasium API and returns `(obs, info)`.
- `step(action)` returns `(obs, reward, terminated, truncated, info)`.
- `truncated` becomes true when `max_steps` is reached.
- `terminated` becomes true when the normalized rigidity metric falls below
  `critical_eig_threshold`.
- `info` includes reward components and `critical_eig`.
- Native MuJoCo and MJX environments evaluate rigidity after every physics
  substep and stop advancing the current action at the first collapsed or
  nonfinite state. This applies to both the JAX and Warp MJX implementations.
  Their `info` also includes `minimum_substep_critical_eig_raw`,
  `substeps_executed`, and `terminated_during_substeps`.

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

## Batched MJX Environment

`MjxNodeVelocityEnv` provides a pure, batch-native accelerator path for abstract
truss models. It preserves the node-velocity action, observation, reward, and
episode semantics described above, but uses explicit JAX state and random keys
instead of the mutable Gymnasium interface. Every input and output has a leading
batch dimension.

```python
import jax
import jax.numpy as jnp

from mujoco_truss_gen import MjxNodeVelocityEnv, TrussEnvConfig, get_mujoco_spec

env = MjxNodeVelocityEnv(
    TrussEnvConfig(
        get_mujoco_spec("tetrahedron", realistic=False),
        max_steps=1_000,
        nsubsteps=2,
        speed=0.01,
    )
)

batch_size = 256
reset = jax.jit(env.reset)
step = jax.jit(env.step)
reset_where = jax.jit(env.reset_where)

keys = jax.random.split(jax.random.key(0), batch_size)
obs, state = reset(keys)

step_keys = jax.random.split(jax.random.key(1), batch_size)
actions = jnp.zeros((batch_size, env.action_size), dtype=jnp.float32)
obs, state, reward, done, info = step(step_keys, state, actions)

reset_keys = jax.random.split(jax.random.key(2), batch_size)
obs, state = reset_where(reset_keys, state, done)
```

`done` is the elementwise union of task termination and time-limit truncation;
the separate boolean arrays are available as `info["terminated"]` and
`info["truncated"]`. Completed environments are not reset automatically.

The accelerator environment supports one fixed abstract or generated realistic
model per instance. Realistic angle-bisector controls are evaluated as batched
JAX operations before every MJX physics substep. Runtime
`DomainRandomizationConfig` ranges are sampled independently per batched
environment on reset and remain fixed for that episode; the sampled values are
available on `state.domain_randomization`. `model_factory`, other internal
actuator types, rendering, and batches containing different model shapes are not
supported. A different batch size can be used, but it causes JAX to compile a
separate executable.

MJX-JAX remains the default implementation. An MJX-Warp candidate is available
for CUDA benchmarking without changing the batch-first API:

```python
env = MjxNodeVelocityEnv(
    TrussEnvConfig(get_mujoco_spec("tetrahedron", realistic=False)),
    mjx_impl="warp",
    warp_graph_mode="warp_staged",
    warp_naconmax=32_768,  # total contact capacity across the batch
    warp_njmax=256,        # constraint capacity per environment
)
```

When `warp_naconmax` is omitted, the adapter allocates a conservative shared
contact capacity of 32,768. This avoids MuJoCo-Warp's single-world default,
which does not scale when MJX later batches the data across many environments
and can truncate broadphase contacts while continuing execution. Explicit
values still override the default; use `buffer_diagnostics(state)` to validate
workload-specific capacity.

Install it with `python -m pip install "mujoco-truss-gen[warp]"`. Warp requires
the active JAX default device to be an NVIDIA CUDA GPU; construction fails with
an actionable error when CUDA or the optional dependency is unavailable.
Supported graph modes are `warp`, `warp_staged`, and `warp_staged_ex`.
`buffer_diagnostics(state)` synchronizes and reports Warp contact/constraint
capacity use, so it is intended for validation and benchmarks rather than the
compiled training hot path.

Generated truss geoms use ground-only collision masks: robot-ground contact is
preserved, while internal robot-robot contacts are disabled. Warm JIT throughput
and reset costs can be measured with the reproducible benchmark matrix:

```bash
python experiments/benchmark_mjx_env.py
python experiments/benchmark_mjx_env.py \
  --models tetrahedron:abstract,octahedron:abstract \
  --batch-sizes 128,256,512
```

Each case runs in a fresh process. The command writes raw results to
`benchmark_results/mjx_warp_orc.json` and the adoption analysis to
`docs/benchmarks/mjx_warp.md`. Warp is not eligible for adoption unless the
representative 1,536-environment workload is at least 1.5x faster than MJX-JAX,
the CUDA parity tests pass, and no contact/constraint capacity gate fails.

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
