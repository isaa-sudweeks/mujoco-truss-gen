from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mujoco


@dataclass(frozen=True)
class AccelerometerConfig:
    """Configuration for node-mounted MuJoCo accelerometers."""

    noise: float = 0.5
    cutoff: float | None = None
    nsample: int | None = None
    delay: float | None = None
    name_prefix: str = "accel"
    pos: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None


DEFAULT_ACCELEROMETER_CONFIG = AccelerometerConfig()


def add_node_accelerometers(
    spec: mujoco.MjSpec,
    node_names: list[str],
    config: AccelerometerConfig | Mapping[str, Any] | None = None,
) -> None:
    """Attach one accelerometer per node, optionally at a custom body-local pose."""
    if config is None:
        return

    config = _coerce_accelerometer_config(config)
    for node_name in node_names:
        sensor_name = f"{config.name_prefix}_{node_name}"
        site_name = node_name
        if config.pos is not None or config.quat is not None:
            site_name = f"{sensor_name}_site"
            spec.site(node_name).parent.add_site(
                name=site_name,
                pos=config.pos,
                quat=config.quat,
            )
        kwargs = {
            "name": sensor_name,
            "type": mujoco.mjtSensor.mjSENS_ACCELEROMETER,
            "objtype": mujoco.mjtObj.mjOBJ_SITE,
            "objname": site_name,
            "noise": config.noise,
        }
        if config.cutoff is not None:
            kwargs["cutoff"] = config.cutoff
        if config.nsample is not None:
            kwargs["nsample"] = config.nsample
        if config.delay is not None:
            kwargs["delay"] = config.delay

        spec.add_sensor(**kwargs)


def _coerce_accelerometer_config(
    config: AccelerometerConfig | Mapping[str, Any],
) -> AccelerometerConfig:
    if isinstance(config, AccelerometerConfig):
        return config
    return AccelerometerConfig(**dict(config))
