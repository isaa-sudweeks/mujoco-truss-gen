export const TERRAIN_KINDS = ["flat", "slope", "stairs", "waves", "rough"];

export const DEFAULT_CONFIG = Object.freeze({
  kind: "rough",
  halfSizeX: 12,
  halfSizeY: 12,
  resolution: 129,
  amplitude: 0.15,
  featureSize: 0.8,
  maxSlopeDegrees: 35,
  slopeAngleDegrees: 12,
  slopeDirectionDegrees: 0,
  stairHeight: 0.1,
  stairRun: 0.6,
  spawnFlatRadius: 1.5,
  spawnBlendWidth: 0.75,
  seed: 7,
});

export function normalizeConfig(values = {}) {
  const config = { ...DEFAULT_CONFIG, ...values };
  if (!TERRAIN_KINDS.includes(config.kind)) throw new Error(`Unknown terrain: ${config.kind}`);
  for (const name of ["halfSizeX", "halfSizeY", "featureSize", "stairHeight", "stairRun"]) {
    if (!Number.isFinite(config[name]) || config[name] <= 0) throw new Error(`${name} must be positive`);
  }
  for (const name of ["amplitude", "spawnFlatRadius", "spawnBlendWidth"]) {
    if (!Number.isFinite(config[name]) || config[name] < 0) throw new Error(`${name} cannot be negative`);
  }
  if (!Number.isInteger(config.resolution) || config.resolution < 3) {
    throw new Error("resolution must be an integer of at least 3");
  }
  if (!(config.maxSlopeDegrees > 0 && config.maxSlopeDegrees < 90)) {
    throw new Error("maxSlopeDegrees must be between 0 and 90");
  }
  if (!(Math.abs(config.slopeAngleDegrees) < 90)) {
    throw new Error("slopeAngleDegrees must be between -90 and 90");
  }
  if (config.spawnFlatRadius + config.spawnBlendWidth >= Math.min(config.halfSizeX, config.halfSizeY)) {
    throw new Error("The spawn region must fit inside the terrain");
  }
  config.seed = Math.max(0, Math.trunc(config.seed));
  return config;
}

export function generateTerrain(values = {}) {
  const config = normalizeConfig(values);
  const rows = config.resolution;
  const cols = config.resolution;
  const dx = (2 * config.halfSizeX) / (cols - 1);
  const dy = (2 * config.halfSizeY) / (rows - 1);
  const heights = new Float64Array(rows * cols);
  const direction = radians(config.slopeDirectionDegrees);
  const directionX = Math.cos(direction);
  const directionY = Math.sin(direction);

  for (let row = 0; row < rows; row += 1) {
    const y = -config.halfSizeY + row * dy;
    for (let col = 0; col < cols; col += 1) {
      const x = -config.halfSizeX + col * dx;
      const projected = x * directionX + y * directionY;
      let height = 0;
      if (config.kind === "slope") {
        height = Math.tan(radians(config.slopeAngleDegrees)) * projected;
      } else if (config.kind === "stairs") {
        height = config.stairHeight * Math.floor(projected / config.stairRun);
      } else if (config.kind === "waves") {
        height = config.amplitude * Math.sin((2 * Math.PI * x) / config.featureSize);
        height *= Math.cos((2 * Math.PI * y) / (1.35 * config.featureSize));
      } else if (config.kind === "rough") {
        height = roughHeight(x, y, config);
      }
      heights[row * cols + col] = height;
    }
  }

  if (config.kind === "rough") {
    let peak = 0;
    for (const height of heights) peak = Math.max(peak, Math.abs(height));
    if (peak > 0) {
      const amplitudeScale = config.amplitude / peak;
      for (let index = 0; index < heights.length; index += 1) {
        heights[index] *= amplitudeScale;
      }
    }
  }

  for (let row = 0; row < rows; row += 1) {
    const y = -config.halfSizeY + row * dy;
    for (let col = 0; col < cols; col += 1) {
      const x = -config.halfSizeX + col * dx;
      heights[row * cols + col] *= spawnMask(Math.hypot(x, y), config);
    }
  }

  if (config.kind === "rough" || config.kind === "waves") {
    const measured = maximumSlope(heights, rows, cols, dx, dy);
    const limit = Math.tan(radians(config.maxSlopeDegrees));
    if (measured > limit && measured > 0) {
      const scale = limit / measured;
      for (let index = 0; index < heights.length; index += 1) heights[index] *= scale;
    }
  }

  let elevationMin = Infinity;
  let elevationMax = -Infinity;
  for (const height of heights) {
    elevationMin = Math.min(elevationMin, height);
    elevationMax = Math.max(elevationMax, height);
  }
  return {
    config,
    rows,
    cols,
    dx,
    dy,
    heights,
    elevationMin,
    elevationMax,
    elevationRange: elevationMax - elevationMin,
    maxGeneratedSlope: maximumSlope(heights, rows, cols, dx, dy),
  };
}

function roughHeight(x, y, config) {
  let value = 0;
  let totalWeight = 0;
  for (let octave = 0; octave < 4; octave += 1) {
    const featureSize = config.featureSize / 2 ** octave;
    const weight = 0.5 ** octave;
    value += weight * smoothValueNoise(x / featureSize, y / featureSize, config.seed, octave);
    totalWeight += weight;
  }
  return value / totalWeight;
}

function smoothValueNoise(x, y, seed, octave) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const tx = smoothstep(x - x0);
  const ty = smoothstep(y - y0);
  const lowerLeft = hashNoise(x0, y0, seed, octave);
  const lowerRight = hashNoise(x0 + 1, y0, seed, octave);
  const upperLeft = hashNoise(x0, y0 + 1, seed, octave);
  const upperRight = hashNoise(x0 + 1, y0 + 1, seed, octave);
  const lower = lowerLeft + tx * (lowerRight - lowerLeft);
  const upper = upperLeft + tx * (upperRight - upperLeft);
  return lower + ty * (upper - lower);
}

function hashNoise(x, y, seed, octave) {
  let hashed = (
    Math.imul(x, 374761393)
    + Math.imul(y, 668265263)
    + Math.imul(seed, 1442695041)
    + Math.imul(octave, 1013904223)
  ) >>> 0;
  hashed = Math.imul(hashed ^ (hashed >>> 13), 1274126177) >>> 0;
  hashed = (hashed ^ (hashed >>> 16)) >>> 0;
  return hashed / 2147483647.5 - 1;
}

function spawnMask(radius, config) {
  if (config.spawnBlendWidth === 0) return radius > config.spawnFlatRadius ? 1 : 0;
  const blend = clamp(
    (radius - config.spawnFlatRadius) / config.spawnBlendWidth,
    0,
    1,
  );
  return smoothstep(blend);
}

function maximumSlope(heights, rows, cols, dx, dy) {
  let maximum = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      const left = heights[row * cols + Math.max(col - 1, 0)];
      const right = heights[row * cols + Math.min(col + 1, cols - 1)];
      const lower = heights[Math.max(row - 1, 0) * cols + col];
      const upper = heights[Math.min(row + 1, rows - 1) * cols + col];
      const gx = (right - left) / (dx * (col === 0 || col === cols - 1 ? 1 : 2));
      const gy = (upper - lower) / (dy * (row === 0 || row === rows - 1 ? 1 : 2));
      maximum = Math.max(maximum, Math.hypot(gx, gy));
    }
  }
  return maximum;
}

export function sampleHeight(terrain, x, y, outside = Number.NaN) {
  const { config, rows, cols, heights } = terrain;
  if (x < -config.halfSizeX || x > config.halfSizeX || y < -config.halfSizeY || y > config.halfSizeY) {
    return outside;
  }
  const column = clamp(((x + config.halfSizeX) / (2 * config.halfSizeX)) * (cols - 1), 0, cols - 1);
  const row = clamp(((y + config.halfSizeY) / (2 * config.halfSizeY)) * (rows - 1), 0, rows - 1);
  const col0 = Math.floor(column);
  const row0 = Math.floor(row);
  const col1 = Math.min(col0 + 1, cols - 1);
  const row1 = Math.min(row0 + 1, rows - 1);
  const u = column - col0;
  const v = row - row0;
  const h00 = heights[row0 * cols + col0];
  const h01 = heights[row0 * cols + col1];
  const h10 = heights[row1 * cols + col0];
  const h11 = heights[row1 * cols + col1];
  if (u >= v) return (1 - u) * h00 + (u - v) * h01 + v * h11;
  return (1 - v) * h00 + (v - u) * h10 + u * h11;
}

export function pythonConfig(config) {
  return `TerrainConfig(
    kind="${config.kind}",
    half_size=(${formatNumber(config.halfSizeX)}, ${formatNumber(config.halfSizeY)}),
    resolution=(${config.resolution}, ${config.resolution}),
    amplitude=${formatNumber(config.amplitude)},
    feature_size=${formatNumber(config.featureSize)},
    max_slope_degrees=${formatNumber(config.maxSlopeDegrees)},
    slope_angle_degrees=${formatNumber(config.slopeAngleDegrees)},
    slope_direction_degrees=${formatNumber(config.slopeDirectionDegrees)},
    stair_height=${formatNumber(config.stairHeight)},
    stair_run=${formatNumber(config.stairRun)},
    spawn_flat_radius=${formatNumber(config.spawnFlatRadius)},
    spawn_blend_width=${formatNumber(config.spawnBlendWidth)},
    seed=${config.seed},
)`;
}

function formatNumber(value) {
  return Number(value).toFixed(6).replace(/\.?0+$/, "") || "0";
}

function smoothstep(value) {
  return value * value * (3 - 2 * value);
}

function radians(degrees) {
  return (degrees * Math.PI) / 180;
}

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}
