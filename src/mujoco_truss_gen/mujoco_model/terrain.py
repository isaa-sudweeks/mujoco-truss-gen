from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import mujoco
import numpy as np

TerrainKind = Literal["flat", "slope", "stairs", "waves", "rough"]

TERRAIN_HFIELD_NAME = "terrain_heightfield"
TERRAIN_GEOM_NAME = "ground"


@dataclass(frozen=True, slots=True)
class TerrainConfig:
    """Physical configuration for a generated MuJoCo height-field terrain.

    Horizontal distances and elevations use MuJoCo world-length units. Angles
    use degrees. Every terrain kind uses the same rectangular height-field
    representation so changing ``kind`` does not change the compiled model
    structure when the extent and resolution are held fixed.
    """

    kind: TerrainKind = "flat"
    half_size: tuple[float, float] = (12.0, 12.0)
    resolution: tuple[int, int] = (129, 129)
    amplitude: float = 0.15
    feature_size: float = 0.8
    max_slope_degrees: float = 35.0
    slope_angle_degrees: float = 12.0
    slope_direction_degrees: float = 0.0
    stair_height: float = 0.1
    stair_run: float = 0.6
    spawn_flat_radius: float = 1.5
    spawn_blend_width: float = 0.75
    seed: int = 7
    base_depth: float = 0.1
    friction: tuple[float, float, float] = (1.0, 0.005, 0.0001)

    def __post_init__(self) -> None:
        if self.kind not in ("flat", "slope", "stairs", "waves", "rough"):
            raise ValueError("kind must be 'flat', 'slope', 'stairs', 'waves', or 'rough'.")
        if len(self.half_size) != 2 or any(
            not np.isfinite(value) or value <= 0.0 for value in self.half_size
        ):
            raise ValueError("half_size must contain two finite positive values.")
        if len(self.resolution) != 2 or any(
            isinstance(value, bool) or int(value) != value or int(value) < 3
            for value in self.resolution
        ):
            raise ValueError("resolution must contain two integers greater than or equal to 3.")
        if not np.isfinite(self.amplitude) or self.amplitude < 0.0:
            raise ValueError("amplitude must be finite and non-negative.")
        if not np.isfinite(self.feature_size) or self.feature_size <= 0.0:
            raise ValueError("feature_size must be finite and positive.")
        if (
            not np.isfinite(self.max_slope_degrees)
            or self.max_slope_degrees <= 0.0
            or self.max_slope_degrees >= 90.0
        ):
            raise ValueError("max_slope_degrees must be finite and between 0 and 90.")
        if not np.isfinite(self.slope_angle_degrees) or abs(self.slope_angle_degrees) >= 90.0:
            raise ValueError("slope_angle_degrees must be finite and between -90 and 90.")
        if not np.isfinite(self.slope_direction_degrees):
            raise ValueError("slope_direction_degrees must be finite.")
        if not np.isfinite(self.stair_height) or self.stair_height <= 0.0:
            raise ValueError("stair_height must be finite and positive.")
        if not np.isfinite(self.stair_run) or self.stair_run <= 0.0:
            raise ValueError("stair_run must be finite and positive.")
        if not np.isfinite(self.spawn_flat_radius) or self.spawn_flat_radius < 0.0:
            raise ValueError("spawn_flat_radius must be finite and non-negative.")
        if not np.isfinite(self.spawn_blend_width) or self.spawn_blend_width < 0.0:
            raise ValueError("spawn_blend_width must be finite and non-negative.")
        if self.spawn_flat_radius + self.spawn_blend_width >= min(self.half_size):
            raise ValueError("The spawn flat radius and blend width must fit inside the terrain.")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer.")
        if not np.isfinite(self.base_depth) or self.base_depth <= 0.0:
            raise ValueError("base_depth must be finite and positive.")
        if len(self.friction) != 3 or any(
            not np.isfinite(value) or value < 0.0 for value in self.friction
        ):
            raise ValueError("friction must contain three finite non-negative values.")


@dataclass(frozen=True, slots=True)
class TerrainData:
    """Generated physical terrain samples and MuJoCo normalization metadata."""

    config: TerrainConfig
    x: np.ndarray
    y: np.ndarray
    heights: np.ndarray
    normalized_heights: np.ndarray
    elevation_min: float
    elevation_range: float
    max_generated_slope: float

    @property
    def grid_spacing(self) -> tuple[float, float]:
        return float(self.x[1] - self.x[0]), float(self.y[1] - self.y[0])

    def height_at(
        self,
        x: np.ndarray | float,
        y: np.ndarray | float,
        *,
        outside: float = -np.inf,
    ) -> np.ndarray:
        """Sample MuJoCo's piecewise-planar terrain surface at world ``x``/``y``."""

        return _sample_height_grid(
            self.heights,
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            x_min=float(self.x[0]),
            x_max=float(self.x[-1]),
            y_min=float(self.y[0]),
            y_max=float(self.y[-1]),
            outside=outside,
        )


def generate_terrain(config: TerrainConfig | None = None) -> TerrainData:
    """Generate a deterministic physical height grid from ``config``."""

    config = config or TerrainConfig()
    nrow, ncol = (int(value) for value in config.resolution)
    radius_x, radius_y = (float(value) for value in config.half_size)
    x = np.linspace(-radius_x, radius_x, ncol, dtype=float)
    y = np.linspace(-radius_y, radius_y, nrow, dtype=float)
    xx: np.ndarray
    yy: np.ndarray
    xx, yy = np.meshgrid(x, y)

    if config.kind == "flat":
        heights = np.zeros_like(xx)
    elif config.kind == "slope":
        direction = np.deg2rad(float(config.slope_direction_degrees))
        projected = xx * np.cos(direction) + yy * np.sin(direction)
        heights = np.tan(np.deg2rad(float(config.slope_angle_degrees))) * projected
    elif config.kind == "stairs":
        direction = np.deg2rad(float(config.slope_direction_degrees))
        projected = xx * np.cos(direction) + yy * np.sin(direction)
        heights = float(config.stair_height) * np.floor(projected / float(config.stair_run))
    elif config.kind == "waves":
        feature_size = float(config.feature_size)
        heights = float(config.amplitude) * np.sin(2.0 * np.pi * xx / feature_size)
        heights *= np.cos(2.0 * np.pi * yy / (1.35 * feature_size))
    else:
        heights = _rough_height_field(config, shape=(nrow, ncol), spacing=(x, y))

    heights = heights * _spawn_blend_mask(xx, yy, config)
    if config.kind in ("rough", "waves"):
        heights = _limit_slope(heights, x=x, y=y, max_degrees=config.max_slope_degrees)

    heights = np.asarray(heights, dtype=float)
    elevation_min = float(np.min(heights))
    elevation_max = float(np.max(heights))
    elevation_range = elevation_max - elevation_min
    if elevation_range > 0.0:
        normalized = (heights - elevation_min) / elevation_range
    else:
        normalized = np.zeros_like(heights)

    return TerrainData(
        config=config,
        x=x,
        y=y,
        heights=heights,
        normalized_heights=np.asarray(normalized, dtype=np.float32),
        elevation_min=elevation_min,
        elevation_range=elevation_range,
        max_generated_slope=_maximum_slope(heights, x=x, y=y),
    )


def add_terrain(spec: mujoco.MjSpec, config: TerrainConfig) -> TerrainData:
    """Replace the base world's plane with a generated height-field terrain."""

    terrain = generate_terrain(config)
    existing_ground = spec.geom(TERRAIN_GEOM_NAME)
    if existing_ground is not None:
        spec.delete(existing_ground)
    existing_hfield = spec.hfield(TERRAIN_HFIELD_NAME)
    if existing_hfield is not None:
        spec.delete(existing_hfield)

    elevation_scale = max(terrain.elevation_range, 1e-6)
    hfield = spec.add_hfield(
        name=TERRAIN_HFIELD_NAME,
        size=[
            float(config.half_size[0]),
            float(config.half_size[1]),
            elevation_scale,
            float(config.base_depth),
        ],
        nrow=int(config.resolution[0]),
        ncol=int(config.resolution[1]),
    )
    hfield.userdata = terrain.normalized_heights.ravel().tolist()
    spec.worldbody.add_geom(
        name=TERRAIN_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_HFIELD,
        hfieldname=TERRAIN_HFIELD_NAME,
        pos=[0.0, 0.0, terrain.elevation_min],
        material="ground_grid",
        friction=list(config.friction),
    )
    return terrain


def sample_model_terrain_height(
    model: mujoco.MjModel,
    xy: np.ndarray,
    *,
    outside: float = -np.inf,
) -> np.ndarray:
    """Sample the named ground plane or height field in a compiled model."""

    points = np.asarray(xy, dtype=float)
    if points.shape[-1] != 2:
        raise ValueError("xy must have a final dimension of size 2.")
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TERRAIN_GEOM_NAME)
    if ground_id < 0:
        return cast(np.ndarray, np.full(points.shape[:-1], outside, dtype=float))

    geom_type = int(model.geom_type[ground_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return cast(
            np.ndarray,
            np.full(points.shape[:-1], float(model.geom_pos[ground_id, 2]), dtype=float),
        )
    if geom_type != int(mujoco.mjtGeom.mjGEOM_HFIELD):
        return cast(np.ndarray, np.full(points.shape[:-1], outside, dtype=float))

    hfield_id = int(model.geom_dataid[ground_id])
    nrow = int(model.hfield_nrow[hfield_id])
    ncol = int(model.hfield_ncol[hfield_id])
    address = int(model.hfield_adr[hfield_id])
    values = np.asarray(model.hfield_data[address : address + nrow * ncol]).reshape(nrow, ncol)
    radius_x, radius_y, elevation_scale, _base_depth = model.hfield_size[hfield_id]
    geom_position = model.geom_pos[ground_id]
    normalized = _sample_height_grid(
        values,
        points[..., 0],
        points[..., 1],
        x_min=float(geom_position[0] - radius_x),
        x_max=float(geom_position[0] + radius_x),
        y_min=float(geom_position[1] - radius_y),
        y_max=float(geom_position[1] + radius_y),
        outside=np.nan,
    )
    physical = float(geom_position[2]) + normalized * float(elevation_scale)
    return cast(np.ndarray, np.where(np.isnan(normalized), outside, physical))


def _rough_height_field(
    config: TerrainConfig,
    *,
    shape: tuple[int, int],
    spacing: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    x, y = spacing
    xx, yy = np.meshgrid(x, y)
    result: np.ndarray = np.zeros(shape, dtype=float)
    total_weight = 0.0
    for octave in range(4):
        feature_size = float(config.feature_size) / (2**octave)
        weight = 0.5**octave
        result += weight * _smooth_value_noise(
            xx / feature_size,
            yy / feature_size,
            seed=int(config.seed),
            octave=octave,
        )
        total_weight += weight
    result /= total_weight
    peak = float(np.max(np.abs(result)))
    if peak > 0.0:
        result *= float(config.amplitude) / peak
    return result


def _smooth_value_noise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    octave: int,
) -> np.ndarray:
    """Return portable smooth value noise shared with the website implementation."""

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fraction_x = x - x0
    fraction_y = y - y0
    blend_x = fraction_x * fraction_x * (3.0 - 2.0 * fraction_x)
    blend_y = fraction_y * fraction_y * (3.0 - 2.0 * fraction_y)
    lower_left = _hash_noise(x0, y0, seed=seed, octave=octave)
    lower_right = _hash_noise(x0 + 1, y0, seed=seed, octave=octave)
    upper_left = _hash_noise(x0, y0 + 1, seed=seed, octave=octave)
    upper_right = _hash_noise(x0 + 1, y0 + 1, seed=seed, octave=octave)
    lower = lower_left + blend_x * (lower_right - lower_left)
    upper = upper_left + blend_x * (upper_right - upper_left)
    return cast(np.ndarray, lower + blend_y * (upper - lower))


def _hash_noise(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    octave: int,
) -> np.ndarray:
    mask = 0xFFFFFFFF
    hashed = (
        x * 374761393
        + y * 668265263
        + (seed & mask) * 1442695041
        + octave * 1013904223
    ) & mask
    hashed = ((hashed ^ (hashed >> 13)) * 1274126177) & mask
    hashed = (hashed ^ (hashed >> 16)) & mask
    return cast(np.ndarray, hashed.astype(float) / 2147483647.5 - 1.0)


def _spawn_blend_mask(xx: np.ndarray, yy: np.ndarray, config: TerrainConfig) -> np.ndarray:
    radius = np.hypot(xx, yy)
    flat_radius = float(config.spawn_flat_radius)
    blend_width = float(config.spawn_blend_width)
    if blend_width == 0.0:
        return cast(np.ndarray, np.asarray((radius > flat_radius).astype(float)))
    blend = np.clip((radius - flat_radius) / blend_width, 0.0, 1.0)
    return cast(np.ndarray, np.asarray(blend * blend * (3.0 - 2.0 * blend)))


def _maximum_slope(heights: np.ndarray, *, x: np.ndarray, y: np.ndarray) -> float:
    spacing_x = np.diff(x)[None, :]
    spacing_y = np.diff(y)[:, None]
    lower_left = heights[:-1, :-1]
    lower_right = heights[:-1, 1:]
    upper_left = heights[1:, :-1]
    upper_right = heights[1:, 1:]

    lower_gradient_x = (lower_right - lower_left) / spacing_x
    lower_gradient_y = (upper_right - lower_right) / spacing_y
    upper_gradient_x = (upper_right - upper_left) / spacing_x
    upper_gradient_y = (upper_left - lower_left) / spacing_y
    return float(
        max(
            np.max(np.hypot(lower_gradient_x, lower_gradient_y)),
            np.max(np.hypot(upper_gradient_x, upper_gradient_y)),
        )
    )


def _limit_slope(
    heights: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
    max_degrees: float,
) -> np.ndarray:
    measured = _maximum_slope(heights, x=x, y=y)
    maximum = float(np.tan(np.deg2rad(max_degrees)))
    if measured > maximum and measured > 0.0:
        return cast(np.ndarray, heights * (maximum / measured))
    return heights


def _sample_height_grid(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    outside: float,
) -> np.ndarray:
    x, y = np.broadcast_arrays(x, y)
    inside = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    nrow, ncol = values.shape
    col = np.clip((x - x_min) / (x_max - x_min) * (ncol - 1), 0.0, ncol - 1)
    row = np.clip((y - y_min) / (y_max - y_min) * (nrow - 1), 0.0, nrow - 1)
    col0 = np.floor(col).astype(int)
    row0 = np.floor(row).astype(int)
    col1 = np.minimum(col0 + 1, ncol - 1)
    row1 = np.minimum(row0 + 1, nrow - 1)
    col_fraction = col - col0
    row_fraction = row - row0
    lower_left = values[row0, col0]
    lower_right = values[row0, col1]
    upper_left = values[row1, col0]
    upper_right = values[row1, col1]
    lower_triangle = (
        (1.0 - col_fraction) * lower_left
        + (col_fraction - row_fraction) * lower_right
        + row_fraction * upper_right
    )
    upper_triangle = (
        (1.0 - row_fraction) * lower_left
        + (row_fraction - col_fraction) * upper_left
        + col_fraction * upper_right
    )
    sampled = np.where(col_fraction >= row_fraction, lower_triangle, upper_triangle)
    return cast(np.ndarray, np.where(inside, sampled, outside))
