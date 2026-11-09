# Sensors

## Node accelerometers

Realistic triangle models have one accelerometer per generated node by default.
Pass `accelerometer_config=None` to disable them. Use `AccelerometerConfig` to
set `noise`, `cutoff`, `nsample`, `delay`, or `name_prefix`.

### Mounting position and orientation

```python
from mujoco_truss_gen import AccelerometerConfig, get_mujoco_spec

spec = get_mujoco_spec(
    "octahedron",
    realistic=True,
    accelerometer_config=AccelerometerConfig(
        pos=(0.01, 0.0, 0.0),
        quat=(0.70710678, 0.0, 0.0, 0.70710678),
    ),
)
model = spec.compile()
```

This places each accelerometer 10 mm along the node's local X axis and rotates
it 90° about local Z. The same pose applies to every generated node. Specify
`pos` in meters from the node center and `quat` in MuJoCo's `(w, x, y, z)`
order. Both are relative to the node body frame, which can differ from the
housing box's geometry frame.

Omitting position or orientation gives zero offset or identity rotation,
respectively. With a custom pose, the accelerometer uses a dedicated site
rigidly fixed to the node body, while the original node site stays available
for control.

See [Model Generation](model-generation.md) for how realistic triangle models
are assembled.
