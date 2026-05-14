from __future__ import annotations

import time
from pathlib import Path

import mujoco

from mujoco_truss_gen.mujoco_model.controllers import AngleBisectorController
from mujoco_truss_gen.mujoco_model.tendons import initialize_actuator_lengths


def view(spec: mujoco.MjSpec) -> None:
    """Compile and view the MuJoCo spec."""
    try:
        import mujoco.viewer as mujoco_viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo passive viewer is unavailable in this Python environment. "
            "Install a MuJoCo build that includes the viewer module, and on macOS "
            "run viewer scripts with mjpython."
        ) from exc

    model = spec.compile()
    if hasattr(model, "model") and hasattr(model, "data"):
        mj_model = model.model
        data = model.data
    elif isinstance(model, mujoco.MjModel):
        mj_model = model
        data = mujoco.MjData(mj_model)
    else:
        raise TypeError(
            "view() expects a mujoco.MjModel or an object with 'model' and 'data' attributes."
        )

    controller = AngleBisectorController(mj_model, spec.to_xml())
    initialize_actuator_lengths(mj_model, data)
    controller.update(mj_model, data)
    with mujoco_viewer.launch_passive(mj_model, data) as viewer:
        viewer.sync()
        while viewer.is_running():
            if data.time == 0.0:
                initialize_actuator_lengths(mj_model, data)
            controller.update(mj_model, data)
            mujoco.mj_step(mj_model, data)
            viewer.sync()
            time.sleep(max(mj_model.opt.timestep, 0.001))


def save_xml(spec: mujoco.MjSpec, filename: str | Path) -> Path:
    path = Path(filename)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml(), encoding="utf-8")
    return path
