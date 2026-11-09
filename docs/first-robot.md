# Simulate your first truss robot

Build an octahedron, step it with native MuJoCo, and open a live viewer.
The example uses a scripted tendon command and runs on the CPU.

## 1. Install the package

In a terminal on macOS or Linux with Python 3.10 or newer, run:

```bash
git clone https://github.com/isaa-sudweeks/mujoco-truss-gen.git
cd mujoco-truss-gen
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 2. Build and step the robot

Save this script as `first_robot.py` in the repository root.

```python
import argparse
import math
import time

import mujoco

from mujoco_truss_gen import MujocoModel, TrussPhysicalParameters, get_mujoco_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", action="store_true", help="Open the MuJoCo viewer")
    args = parser.parse_args()

    params = TrussPhysicalParameters(active_node_mass=0.15, passive_node_mass=0.15)
    spec = get_mujoco_spec("octahedron", realistic=False, physical_params=params)
    robot = MujocoModel(spec)
    model, data = robot.model, robot.data
    actuator_id = robot.external_actuator_ids[0]
    print(f"Commanding {model.actuator(actuator_id).name}")
    print(f"Node mass: {model.body('node_1').mass[0]:.3f} kg")

    def step():
        # Oscillate one tendon velocity command at 1 Hz.
        data.ctrl[actuator_id] = 0.02 * math.sin(2 * math.pi * data.time)
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
                remaining = model.opt.timestep - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
    else:
        for _ in range(500):
            step()

    print(f"Simulated {data.time:.3f} seconds")
    print(f"Node 1 position: {data.body('node_1').xpos}")


if __name__ == "__main__":
    main()
```

Run the headless simulation with `python first_robot.py`. It prints the
actuator name, the configured node mass, the elapsed simulation time, and
the final position of node 1.

`get_mujoco_spec()` describes the robot and its world. `MujocoModel(spec)`
compiles the spec, creates the mutable `MjData` state, and initializes the
integrated-velocity actuators at their starting tendon lengths.
`mujoco.mj_step()` advances the physics by one timestep. The example sends
a velocity command to one external actuator while leaving the others at zero
command. The bisector control call updates the internal hinge targets for
models that have them.

## 3. Watch the same simulation

Run one of these commands from the activated virtual environment:

```bash
# macOS
mjpython first_robot.py --view

# Linux with a graphical desktop
python first_robot.py --view
```

The passive viewer displays the same state advanced by the script.
`window.sync()` updates the display after each step, and the sleep paces the
loop toward real time. Drag to rotate the camera, scroll to zoom, and close
the window to end the simulation. On a machine without a graphical session,
use the headless command in step 2. See the [MuJoCo viewer guide](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).

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

For this variant, remove the abstract-model `node_1` mass and position print
lines. Realistic models attach node sites and geoms to triangle bodies, so
you can inspect site positions with `data.site(site_name).xpos` using names
from `model.site(i).name`. The hinge gain applies to the internal connector
position servos. The scripted command still targets an external tendon.

See [model generation](model-generation.md) for custom trusses, [terrain](terrain.md)
for changing the ground, and [environments](environments.md) for using the
same spec in Gymnasium.
