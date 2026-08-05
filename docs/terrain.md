# Height-Field Terrain

Generated models use the existing checker-textured plane when terrain is omitted.
Pass a `TerrainConfig` to `build_world()` or `get_mujoco_spec()` to replace that
plane with a configurable MuJoCo height field:

```python
from mujoco_truss_gen import TerrainConfig, get_mujoco_spec

terrain = TerrainConfig(
    kind="rough",
    half_size=(12.0, 12.0),
    resolution=(129, 129),
    amplitude=0.15,
    feature_size=0.8,
    max_slope_degrees=35.0,
    seed=7,
)
spec = get_mujoco_spec(
    "tetrahedron",
    realistic=True,
    terrain=terrain,
)
```

All distances and elevations use MuJoCo world-length units. Angles use degrees.

## Terrain Families

Every explicit terrain uses the same rectangular height-field representation.
Holding `half_size` and `resolution` fixed therefore preserves the compiled model
shape across these terrain families:

- `flat`: an explicit zero-height height field. Use this instead of omitting
  terrain when a collection of precompiled models must all contain a height field.
- `slope`: a planar grade controlled by `slope_angle_degrees` and
  `slope_direction_degrees`; direction `0` rises toward world `+X`.
- `stairs`: steps controlled by `stair_height`, `stair_run`, and
  `slope_direction_degrees`.
- `waves`: crossed sinusoidal features controlled by `amplitude` and
  `feature_size`.
- `rough`: deterministic multi-scale smoothed noise controlled by `amplitude`,
  `feature_size`, and `seed`.

`max_slope_degrees` caps the sampled gradient of `waves` and `rough` terrain.
It does not alter the requested slope angle or the intentionally discontinuous
stair profile. Slope limiting can reduce the realized amplitude when the requested
height and feature size would otherwise exceed the cap.

MuJoCo normalizes height-field samples internally. `generate_terrain()` returns a
`TerrainData` object containing both physical elevations and the normalized values
used to build the model:

```python
from mujoco_truss_gen import TerrainConfig, generate_terrain

data = generate_terrain(TerrainConfig(kind="rough", seed=11))
print(data.heights.shape)
print(data.grid_spacing)
print(data.max_generated_slope)
```

Generation is deterministic: identical configurations produce identical grids,
and changing `seed` changes rough terrain without changing its shape or extent.

## Safe Spawn Region

`spawn_flat_radius` keeps a circular region around the origin at exactly world
`z = 0`. `spawn_blend_width` smoothly transitions from that patch into the selected
terrain. The default patch contains the built-in presets at their default scale.

When an environment reset translates or rotates a robot onto an elevated part of
an explicit height field, the native and MJX environments lift the full reset pose
enough to preserve the model's initial node-to-ground clearances. Slip shaping also
measures node height relative to the local triangulated terrain surface rather than
using a fixed world-Z cutoff. The default plane reset behavior is unchanged.

## Terrain Figures and Explorer

Install the optional plotting dependency:

```bash
python -m pip install "mujoco-truss-gen[terrain]"
```

`view_terrain()` follows the same pattern as `view_graph()`: it returns the figure,
axes, and generated data so a caller can save or customize the result.

```python
from mujoco_truss_gen import TerrainConfig, view_terrain

fig, axes, data = view_terrain(
    TerrainConfig(kind="waves", amplitude=0.12, feature_size=1.0),
    show=False,
)
fig.savefig("terrain.png", dpi=160)
```

The interactive explorer exposes terrain family, amplitude, feature size, slope,
stairs, resolution, and seed controls while updating the height map and `+X`
centerline together:

```python
from mujoco_truss_gen import view_terrain_explorer

explorer = view_terrain_explorer()
```

The returned `TerrainExplorer` keeps the Matplotlib widgets alive and exposes the
current `explorer.config` and `explorer.terrain` values.

The public [browser terrain explorer](https://isaa-sudweeks.github.io/mujoco-truss-gen/terrain.html)
provides the same terrain families and deterministic generation without requiring a
local Python installation. Its shareable settings can be copied directly as a
`TerrainConfig`.

## Native and MJX Use

An explicit terrain is embedded in the model source and works with native MuJoCo
and `MjxNodeVelocityEnv`. One `MjxNodeVelocityEnv` still owns one fixed terrain;
changing the terrain requires constructing a separate environment. Keep terrain
extent and resolution fixed when grouping precompiled terrain variants for training.

Native environments can use `DomainRandomizationConfig.model_factory` to generate
a new terrain/model on reset. `MjxNodeVelocityEnv` intentionally rejects
`model_factory`, so per-element or per-reset MJX terrain randomization is not yet
part of this API.
