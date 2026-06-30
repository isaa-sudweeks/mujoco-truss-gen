# GPU Optimization Assessment

The primary limitation is architectural: `mujoco-truss-gen` currently has no
batched, accelerator-native environment path. The Gymnasium environments execute
scalar NumPy operations and CPU `mujoco.mj_step`; installing `mujoco-mjx` alone
does not use GPUs.

The measurements below were collected from the current code on Apple Silicon.
Absolute rates will differ on a research cluster, but the relative bottlenecks
show where optimization work should be concentrated.

## Ranked performance issues

### 1. No vectorized MJX or MuJoCo Warp environment implementation

The library contains no JAX/MJX stepping implementation despite depending on
`mujoco-mjx`. `MujocoTrussEnv.step()` performs one Python environment step around
CPU MuJoCo.

Consequences:

- A100 and H200 GPUs currently provide no simulation acceleration.
- Gymnasium process vectorization still provides CPU parallelism rather than GPU
  batching.
- Observations, rewards, resets, controllers, and termination checks are not
  JIT-compatible.
- Running one environment or model state per GPU would heavily underutilize each
  GPU.

The target architecture should use one compiled model replicated per GPU, with
hundreds or thousands of environment states batched on each GPU.

### 2. Realistic models currently fail MJX conversion

Every tested `realistic=True` preset failed `mjx.put_model()` with:

```text
NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_BOX) collisions not implemented.
```

This affects the tetrahedron, octahedron, icosahedron, and solar-array models.
The corresponding abstract models converted successfully.

The immediate cause is the combination of realistic box/cylinder geometry and
unrestricted collision masks. After restricting robot collisions to ground-only,
all four realistic models converted successfully, and a JIT-compiled realistic
tetrahedron MJX step ran successfully.

### 3. The Python angle-bisector controller dominates realistic stepping

`AngleBisectorController.update()` loops over every target and performs many
small NumPy operations, including cross products, norms, rotations, angle
calculations, list construction, and dictionary access. It runs once per physics
substep.

Measured costs:

| Realistic model | Raw `mj_step` | Controller only |
|---|---:|---:|
| Tetrahedron | 101 us | 1,129 us |
| Octahedron | 173 us | 461 us |
| Icosahedron | 3,715 us | 2,272 us |
| Solar array | 377 us | 837 us |

For the tetrahedron, the controller is roughly 11 times more expensive than
MuJoCo itself. It also cannot be batched or JIT-compiled in its current form.

It should be replaced with either:

- Native MuJoCo constraints or actuation, if the same behavior can be modeled
  physically; or
- A pure JAX batched controller included inside the accelerator step.

### 4. Collision filtering enables excessive robot-to-robot contacts

The misleadingly named `disable_geom_contacts()` assigns both `contype` and
`conaffinity` to `1`. As a result, all generated truss geoms can collide with one
another.

This produced approximately 241 active contacts per step for the realistic
icosahedron. Restricting collisions to robot-ground contact reduced that to about
four and changed raw physics time from 3.72 ms to 0.204 ms: approximately an
18-times improvement in `mj_step`, or 2.5 times for controller-plus-physics.

The library needs deliberate collision groups and exclusions:

- Preserve ground contact.
- Disable collisions between directly connected components.
- Disable internal rod/node collisions unless physically required.
- Enable only selected self-collision pairs.
- Provide an MJX-compatible collision profile.

### 5. Python environment bookkeeping is 20 times slower than abstract physics

For the abstract tetrahedron:

- Raw `mj_step`: approximately 5.2 us.
- Full `MujocoNodeVelocityCommandEnv.step`: approximately 105 us.

The library spends most of its time allocating observations, rebuilding rigidity
topology, scanning names, calculating reward terms, and creating dictionaries.
The simulator accounts for only about five percent of the environment step.

Important examples include:

- `_uses_realistic_connector_balls()` scans every body name every step.
- `_logical_rigidity_graph()` rebuilds static mappings and performs repeated
  `mj_name2id` calls.
- `MujocoRelativeObsEnv._get_obs()` separately builds position dictionaries,
  velocity dictionaries, and a position matrix.
- Node observations are constructed through nested Python loops and temporary
  lists.

All topology, body IDs, logical-node aggregation maps, axis indices, and edge
indices should be precomputed once.

### 6. Rigidity termination performs unnecessary per-step reconstruction and eigendecomposition

`_critical_eig()` rebuilds the rigidity matrix and runs a full symmetric
eigendecomposition every environment step. `_compute_reward()` also computes
rigidity and slip even when their reward weights are zero.

Recommended changes:

- Cache all static topology and scatter indices.
- Construct the matrix with vectorized indexed assignments.
- Allow rigidity checks every `N` control steps.
- Skip rigidity and slip calculations when disabled.
- Implement the calculation as a batched device operation for GPU environments.
- Consider a cheaper collapse proxy if exact eigenvalues are unnecessary every
  step.

### 7. Realistic generation causes major state and constraint growth

The realistic icosahedron expands from:

- 36 to 276 velocity degrees of freedom.
- 13 to 133 geoms.
- 0 to 60 controller targets.
- 20 to 80 equality constraints.

This comes from cloned node bodies, connector balls, rods, hinges, internal
actuators, and equality constraints. Even after fixing collision filtering, its
raw physics step remained roughly nine times slower than the abstract model.

A performance-oriented realistic representation should reduce cloned bodies and
replace chains of constrained dynamic bodies with simpler kinematics or
composite constraints where physically acceptable.

### 8. Structural domain randomization rebuilds compiled models

A `model_factory` causes a new `MujocoModel` to be constructed during reset. On
an accelerator, model shape changes can invalidate batching and trigger
recompilation.

GPU rollouts need fixed model shapes. Randomized parameters should normally be
batched runtime arrays; topology variants should be grouped into separate
compiled workloads.

### 9. No performance regression coverage

Tests validate model behavior but do not check:

- MJX conversion of every preset.
- Batched stepping.
- Contact counts.
- Controller cost.
- Environment steps per second.
- Model size or equality-constraint growth.

This allowed the collision configuration and realistic MJX incompatibility to
remain undetected.

## Recommended implementation order

1. Add collision groups and exclusions, plus MJX conversion tests.
2. Define a functional, batched accelerator environment API.
3. Port or eliminate the Python angle-bisector controller.
4. Cache all static model metadata and vectorize observations and rewards.
5. Reduce realistic model degrees of freedom and constraints.
6. Make rigidity and slip computation optional and batched.
7. Add benchmark and scaling tests.

## Recommended cluster architecture

Start with one GPU and many environment states per model. Scale to two through
eight GPUs only after one GPU reaches adequate utilization. Each GPU should hold
a replica of the model plus a large shard of batched environment states, not a
single environment.

Keep simulation, policy inference, observation construction, rewards, and
rollout storage on the same device. Synchronize only aggregated training data or
gradients between GPUs. Batch size should be selected empirically for each model
and accelerator because these truss models are too small to utilize an A100 or
H200 efficiently with only a few simultaneous states.
