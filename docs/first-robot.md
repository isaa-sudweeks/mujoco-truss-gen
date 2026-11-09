# Simulate your first truss robot

Build an octahedron and watch it simulate with the package's built-in
`view()` helper. Then apply a scripted tendon command with or without a live
window.

## 1. Install the package

In a terminal on macOS or Linux with Python 3.10 or newer, run:

```bash
git clone https://github.com/isaa-sudweeks/mujoco-truss-gen.git
cd mujoco-truss-gen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 2. Build and view the robot

Save this as `view_robot.py` in the repository root:

```python
from mujoco_truss_gen import get_mujoco_spec, view

spec = get_mujoco_spec("octahedron", realistic=False)
view(spec)
```

Run the script from the activated virtual environment:

```bash
# macOS
mjpython view_robot.py

# Linux with a graphical desktop
python view_robot.py
```

`get_mujoco_spec()` builds an editable description of the robot and its world.
`view()` compiles the spec, initializes the tendon actuators at their starting
lengths, advances the physics, and refreshes the display. It uses MuJoCo's
window and handles the simulation loop for you.

Drag to rotate the camera, scroll to zoom, and close the window to end the
simulation. The robot starts with zero external tendon commands, so you may
mainly see it settle under gravity. For interactive node commands on a routed
model, see [Rendering](environments.md#rendering).

## 3. Run a scripted simulation

Save this as `first_robot.py`. The same control loop runs headlessly or
displays the robot live with `--view`.

```python
import argparse
import math
import time

import mujoco

from mujoco_truss_gen import MujocoModel, TrussPhysicalParameters, get_mujoco_spec

parser = argparse.ArgumentParser()
parser.add_argument("--view", action="store_true", help="Watch the commanded robot live")
args = parser.parse_args()

params = TrussPhysicalParameters(active_node_mass=0.15, passive_node_mass=0.15)
spec = get_mujoco_spec("octahedron", realistic=False, physical_params=params)
robot = MujocoModel(spec)
model, data = robot.model, robot.data
actuator_id = robot.external_actuator_ids[0]

print(f"Commanding {model.actuator(actuator_id).name}")
print(f"Node mass: {model.body('node_1').mass[0]:.3f} kg")

def step():
    # Extend and retract one tendon target over a four-second cycle.
    data.ctrl[actuator_id] = 0.08 * math.sin(2 * math.pi * data.time / 4)
    robot.apply_angle_bisector_control()
    mujoco.mj_step(model, data)


if args.view:
    from mujoco import viewer

    with viewer.launch_passive(model, data) as window:
        window.sync()
        while window.is_running():
            start = time.monotonic()
            step()
            window.sync()
            time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - start)))
else:
    for _ in range(500):
        step()

print(f"Simulated {data.time:.3f} seconds")
print(f"Node 1 position: {data.site('node_1').xpos}")
```

`MujocoModel(spec)` creates the compiled `MjModel` and mutable `MjData`
state and initializes the tendon actuators. `data.ctrl` contains the actuator
commands. The loop commands one external tendon and leaves the others at zero.
`apply_angle_bisector_control()` updates the internal hinge targets for models
with connector hinges. `mujoco.mj_step()` advances the physics by
`model.opt.timestep`.

### Watch the robot respond to control

From the activated virtual environment, run:

```bash
# macOS
mjpython first_robot.py --view

# Linux with a graphical desktop
python first_robot.py --view
```

Watch the commanded edge extend and retract. The terminal prints its actuator
name. Close the window to end the run.

The built-in `view(spec)` helper in step 2 owns its simulation loop. Here the
passive viewer shares the same `model` and `data` as your scripted loop, so it
displays the state produced by your command. `window.sync()` refreshes the
display after each physics step, and the sleep paces the loop near real time.

For a headless run, use `python first_robot.py`. It runs 500 steps, or one
simulated second at the default 2 ms timestep, then prints the final position.

## 4. Change fixed physical parameters

Set both `active_node_mass` and `passive_node_mass` to `0.20` and rerun
the script. The printed mass should change to `0.200 kg`. The abstract
triangle builder uses the passive mass for any node designated as passive
in at least one triangle. The defaults come from
`TrussPhysicalParameters()`, and you only need to specify the fields you want
to override. Pass the resulting object as `physical_params=` when generating
the spec. Each compiled model uses those settings for the entire run. To
compare settings, generate and compile a fresh spec for each parameter set.

| What to change | Example fields |
| --- | --- |
| Triangle node masses in either model mode | `active_node_mass`, `passive_node_mass` |
| Abstract tendon actuator gain | `abstract_actuator_kp` |
| Realistic connector properties | `connector_mass`, `connector_rod_length`, `hinge_position_kp` |

For a realistic octahedron, replace the parameter and spec lines with:

```python
params = TrussPhysicalParameters(active_node_mass=0.20, hinge_position_kp=30.0)
spec = get_mujoco_spec("octahedron", realistic=True, physical_params=params)
```

Use the same parameter and spec lines in `view_robot.py` to watch the
configured model. In the scripted run, omit the mass print for the realistic
variant. Its mass is distributed across triangle bodies, node geoms, and
connectors. The node site still provides a position, and the scripted command
still targets an external tendon.

See the [physical parameter reference](physical-parameters.md) for every field,
its default, and which models use it.

See [model generation](model-generation.md) for custom trusses, [terrain](terrain.md)
for changing the ground, and [environments](environments.md) for using the
same spec in Gymnasium.
