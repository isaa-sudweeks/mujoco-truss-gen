import {
  DEFAULT_CONFIG,
  generateTerrain,
  normalizeConfig,
  pythonConfig,
  sampleHeight,
} from "./terrain.mjs";

const CONTROL_NAMES = [
  "kind",
  "resolution",
  "amplitude",
  "featureSize",
  "maxSlopeDegrees",
  "slopeAngleDegrees",
  "slopeDirectionDegrees",
  "stairHeight",
  "stairRun",
  "spawnFlatRadius",
  "spawnBlendWidth",
  "halfSizeX",
  "seed",
];

const EXAMPLES = {
  "gentle-waves": { kind: "waves", amplitude: 0.08, featureSize: 1.8, maxSlopeDegrees: 18, seed: 7 },
  "shallow-stairs": { kind: "stairs", stairHeight: 0.06, stairRun: 1.1, slopeDirectionDegrees: 0, seed: 7 },
  "coarse-rough": { kind: "rough", amplitude: 0.16, featureSize: 1.7, maxSlopeDegrees: 28, seed: 14 },
  "steep-rough": { kind: "rough", amplitude: 0.28, featureSize: 0.55, maxSlopeDegrees: 45, seed: 31 },
};

const controls = Object.fromEntries(CONTROL_NAMES.map((name) => [name, document.getElementById(name)]));
const exampleControl = document.getElementById("example");
const errorElement = document.getElementById("error");
const metricsElement = document.getElementById("metrics");
const codeElement = document.getElementById("python-config");
const mapCanvas = document.getElementById("height-map");
const profileCanvas = document.getElementById("profile");
const tooltip = document.getElementById("tooltip");
const mapWrap = document.getElementById("map-wrap");
let terrain;
let scheduled = false;

function configFromControls() {
  return normalizeConfig({
    kind: controls.kind.value,
    resolution: Number(controls.resolution.value),
    amplitude: Number(controls.amplitude.value),
    featureSize: Number(controls.featureSize.value),
    maxSlopeDegrees: Number(controls.maxSlopeDegrees.value),
    slopeAngleDegrees: Number(controls.slopeAngleDegrees.value),
    slopeDirectionDegrees: Number(controls.slopeDirectionDegrees.value),
    stairHeight: Number(controls.stairHeight.value),
    stairRun: Number(controls.stairRun.value),
    spawnFlatRadius: Number(controls.spawnFlatRadius.value),
    spawnBlendWidth: Number(controls.spawnBlendWidth.value),
    halfSizeX: Number(controls.halfSizeX.value),
    halfSizeY: Number(controls.halfSizeX.value),
    seed: Number(controls.seed.value),
  });
}

function applyConfig(config) {
  for (const name of CONTROL_NAMES) {
    if (name in config) controls[name].value = config[name];
  }
  controls.halfSizeX.value = config.halfSizeX;
  update();
}

function scheduleUpdate(markCustom = true) {
  if (markCustom) exampleControl.value = "custom";
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(() => {
    scheduled = false;
    update();
  });
}

function update() {
  try {
    const config = configFromControls();
    terrain = generateTerrain(config);
    errorElement.hidden = true;
    updateVisibleControls(config.kind);
    updateOutputs(config);
    updateMetrics(terrain);
    codeElement.textContent = `from mujoco_truss_gen import TerrainConfig\n\n${pythonConfig(config)}`;
    document.getElementById("map-title").textContent = `${titleCase(config.kind)} terrain`;
    drawHeightMap(terrain);
    drawProfile(terrain);
    updateUrl(config);
  } catch (error) {
    errorElement.textContent = error.message;
    errorElement.hidden = false;
  }
}

function updateVisibleControls(kind) {
  for (const element of document.querySelectorAll("[data-kinds]")) {
    element.hidden = !element.dataset.kinds.split(" ").includes(kind);
  }
}

function updateOutputs(config) {
  const values = {
    amplitude: format(config.amplitude),
    featureSize: format(config.featureSize),
    maxSlopeDegrees: `${format(config.maxSlopeDegrees)}°`,
    slopeAngleDegrees: `${format(config.slopeAngleDegrees)}°`,
    slopeDirectionDegrees: `${format(config.slopeDirectionDegrees)}°`,
    stairHeight: format(config.stairHeight),
    stairRun: format(config.stairRun),
    spawnFlatRadius: format(config.spawnFlatRadius),
    spawnBlendWidth: format(config.spawnBlendWidth),
    halfSizeX: `${format(config.halfSizeX)} × ${format(config.halfSizeY)}`,
    seed: String(config.seed),
  };
  for (const [name, value] of Object.entries(values)) {
    const output = document.querySelector(`output[data-for="${name}"]`);
    if (output) output.value = value;
  }
}

function updateMetrics(data) {
  const slopeDegrees = degrees(Math.atan(data.maxGeneratedSlope));
  const ratio = data.config.featureSize / 0.1;
  const cards = [
    ["Height range", `${format(data.elevationMin, 3)} to ${format(data.elevationMax, 3)}`],
    ["Realized max slope", `${format(slopeDegrees, 1)}°`],
    ["Grid spacing", `${format(data.dx, 3)} × ${format(data.dy, 3)}`],
    ["Feature / node radius", `${format(ratio, 1)}×`],
  ];
  metricsElement.replaceChildren(...cards.map(([label, value]) => {
    const card = document.createElement("div");
    card.className = "metric";
    const strong = document.createElement("strong");
    strong.textContent = value;
    const span = document.createElement("span");
    span.textContent = label;
    card.append(strong, span);
    return card;
  }));
}

function drawHeightMap(data) {
  const context = mapCanvas.getContext("2d");
  const width = mapCanvas.width;
  const height = mapCanvas.height;
  const plot = { left: 64, top: 28, right: width - 28, bottom: height - 55 };
  const plotWidth = plot.right - plot.left;
  const plotHeight = plot.bottom - plot.top;
  const theme = canvasTheme();
  context.clearRect(0, 0, width, height);
  context.fillStyle = theme.surface;
  context.fillRect(0, 0, width, height);

  const imageCanvas = document.createElement("canvas");
  imageCanvas.width = data.cols;
  imageCanvas.height = data.rows;
  const imageContext = imageCanvas.getContext("2d");
  const image = imageContext.createImageData(data.cols, data.rows);
  const magnitude = Math.max(Math.abs(data.elevationMin), Math.abs(data.elevationMax), 1e-9);
  for (let displayRow = 0; displayRow < data.rows; displayRow += 1) {
    const sourceRow = data.rows - 1 - displayRow;
    for (let col = 0; col < data.cols; col += 1) {
      const heightValue = data.heights[sourceRow * data.cols + col];
      const color = terrainColor(heightValue / magnitude);
      const offset = (displayRow * data.cols + col) * 4;
      image.data.set([...color, 255], offset);
    }
  }
  imageContext.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = true;
  context.drawImage(imageCanvas, plot.left, plot.top, plotWidth, plotHeight);
  drawContours(context, data, plot, theme);
  drawMapOverlays(context, data, plot, theme);
  drawAxes(context, data, plot, theme);
  mapCanvas._plot = plot;
}

function drawContours(context, data, plot, theme) {
  if (data.elevationRange < 1e-12) return;
  context.save();
  context.strokeStyle = theme.contour;
  context.lineWidth = 0.75;
  for (let levelIndex = 1; levelIndex <= 7; levelIndex += 1) {
    const level = data.elevationMin + (data.elevationRange * levelIndex) / 8;
    context.beginPath();
    for (let row = 0; row < data.rows - 1; row += 1) {
      for (let col = 0; col < data.cols - 1; col += 1) {
        contourCell(context, data, plot, row, col, level);
      }
    }
    context.stroke();
  }
  context.restore();
}

function contourCell(context, data, plot, row, col, level) {
  const values = [
    data.heights[row * data.cols + col],
    data.heights[row * data.cols + col + 1],
    data.heights[(row + 1) * data.cols + col + 1],
    data.heights[(row + 1) * data.cols + col],
  ];
  const points = [
    worldToMap(data, plot, -data.config.halfSizeX + col * data.dx, -data.config.halfSizeY + row * data.dy),
    worldToMap(data, plot, -data.config.halfSizeX + (col + 1) * data.dx, -data.config.halfSizeY + row * data.dy),
    worldToMap(data, plot, -data.config.halfSizeX + (col + 1) * data.dx, -data.config.halfSizeY + (row + 1) * data.dy),
    worldToMap(data, plot, -data.config.halfSizeX + col * data.dx, -data.config.halfSizeY + (row + 1) * data.dy),
  ];
  const crossings = [];
  for (const [first, second] of [[0, 1], [1, 2], [2, 3], [3, 0]]) {
    if ((values[first] < level) === (values[second] < level) || values[first] === values[second]) continue;
    const amount = (level - values[first]) / (values[second] - values[first]);
    crossings.push([
      points[first][0] + amount * (points[second][0] - points[first][0]),
      points[first][1] + amount * (points[second][1] - points[first][1]),
    ]);
  }
  for (let index = 0; index + 1 < crossings.length; index += 2) {
    context.moveTo(...crossings[index]);
    context.lineTo(...crossings[index + 1]);
  }
}

function drawMapOverlays(context, data, plot, theme) {
  const center = worldToMap(data, plot, 0, 0);
  const radiusScale = (plot.right - plot.left) / (2 * data.config.halfSizeX);
  context.save();
  context.setLineDash([8, 6]);
  context.strokeStyle = theme.blend;
  context.lineWidth = 2;
  context.beginPath();
  context.arc(...center, (data.config.spawnFlatRadius + data.config.spawnBlendWidth) * radiusScale, 0, 2 * Math.PI);
  context.stroke();
  context.setLineDash([]);
  context.strokeStyle = theme.spawn;
  context.lineWidth = 2.5;
  context.beginPath();
  context.arc(...center, data.config.spawnFlatRadius * radiusScale, 0, 2 * Math.PI);
  context.stroke();
  context.fillStyle = theme.robot;
  context.beginPath();
  context.arc(...center, Math.min(1, data.config.spawnFlatRadius * 0.7) * radiusScale, 0, 2 * Math.PI);
  context.fill();
  const arrowEnd = worldToMap(data, plot, Math.min(data.config.halfSizeX * 0.55, data.config.spawnFlatRadius + 2), 0);
  drawArrow(context, center[0] + 8, center[1], arrowEnd[0], arrowEnd[1], theme.ink);
  context.font = "700 15px system-ui";
  context.fillStyle = theme.ink;
  context.fillText("+X", arrowEnd[0] + 7, arrowEnd[1] + 5);
  context.restore();
}

function drawAxes(context, data, plot, theme) {
  context.save();
  context.strokeStyle = theme.axis;
  context.fillStyle = theme.muted;
  context.lineWidth = 1;
  context.font = "12px system-ui";
  context.textAlign = "center";
  context.beginPath();
  context.rect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
  context.stroke();
  for (let index = 0; index <= 4; index += 1) {
    const x = plot.left + ((plot.right - plot.left) * index) / 4;
    const value = -data.config.halfSizeX + (2 * data.config.halfSizeX * index) / 4;
    context.fillText(format(value, 1), x, plot.bottom + 20);
    const y = plot.bottom - ((plot.bottom - plot.top) * index) / 4;
    context.textAlign = "right";
    context.fillText(format(value, 1), plot.left - 10, y + 4);
    context.textAlign = "center";
  }
  context.font = "600 13px system-ui";
  context.fillText("X position (world units)", (plot.left + plot.right) / 2, plot.bottom + 43);
  context.translate(18, (plot.top + plot.bottom) / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("Y position (world units)", 0, 0);
  context.restore();
}

function drawProfile(data) {
  const context = profileCanvas.getContext("2d");
  const width = profileCanvas.width;
  const height = profileCanvas.height;
  const plot = { left: 62, top: 22, right: width - 25, bottom: height - 48 };
  const theme = canvasTheme();
  context.clearRect(0, 0, width, height);
  context.fillStyle = theme.surface;
  context.fillRect(0, 0, width, height);
  const margin = Math.max(data.elevationRange * 0.12, 0.02);
  const minimum = Math.min(data.elevationMin - margin, -margin);
  const maximum = Math.max(data.elevationMax + margin, margin);
  const xPixel = (x) => plot.left + ((x + data.config.halfSizeX) / (2 * data.config.halfSizeX)) * (plot.right - plot.left);
  const yPixel = (value) => plot.bottom - ((value - minimum) / (maximum - minimum)) * (plot.bottom - plot.top);
  const spawnLeft = xPixel(-data.config.spawnFlatRadius);
  const spawnRight = xPixel(data.config.spawnFlatRadius);
  context.fillStyle = theme.spawnFill;
  context.fillRect(spawnLeft, plot.top, spawnRight - spawnLeft, plot.bottom - plot.top);
  context.strokeStyle = theme.grid;
  context.lineWidth = 1;
  for (let index = 0; index <= 4; index += 1) {
    const y = plot.top + ((plot.bottom - plot.top) * index) / 4;
    context.beginPath(); context.moveTo(plot.left, y); context.lineTo(plot.right, y); context.stroke();
  }
  context.strokeStyle = theme.zero;
  context.setLineDash([5, 4]);
  context.beginPath(); context.moveTo(plot.left, yPixel(0)); context.lineTo(plot.right, yPixel(0)); context.stroke();
  context.setLineDash([]);
  context.strokeStyle = theme.profile;
  context.lineWidth = 3;
  context.beginPath();
  const row = Math.floor(data.rows / 2);
  for (let col = 0; col < data.cols; col += 1) {
    const x = -data.config.halfSizeX + col * data.dx;
    const heightValue = data.heights[row * data.cols + col];
    if (col === 0) context.moveTo(xPixel(x), yPixel(heightValue));
    else context.lineTo(xPixel(x), yPixel(heightValue));
  }
  context.stroke();
  context.strokeStyle = theme.axis;
  context.strokeRect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
  context.fillStyle = theme.muted;
  context.font = "12px system-ui";
  context.textAlign = "center";
  for (let index = 0; index <= 4; index += 1) {
    const x = plot.left + ((plot.right - plot.left) * index) / 4;
    const value = -data.config.halfSizeX + (2 * data.config.halfSizeX * index) / 4;
    context.fillText(format(value, 1), x, plot.bottom + 19);
  }
  context.textAlign = "right";
  context.fillText(format(maximum, 2), plot.left - 8, plot.top + 4);
  context.fillText(format(minimum, 2), plot.left - 8, plot.bottom + 4);
  context.restore();
}

function worldToMap(data, plot, x, y) {
  return [
    plot.left + ((x + data.config.halfSizeX) / (2 * data.config.halfSizeX)) * (plot.right - plot.left),
    plot.bottom - ((y + data.config.halfSizeY) / (2 * data.config.halfSizeY)) * (plot.bottom - plot.top),
  ];
}

function mapPointer(event) {
  if (!terrain || !mapCanvas._plot) return;
  const rectangle = mapCanvas.getBoundingClientRect();
  const canvasX = ((event.clientX - rectangle.left) / rectangle.width) * mapCanvas.width;
  const canvasY = ((event.clientY - rectangle.top) / rectangle.height) * mapCanvas.height;
  const plot = mapCanvas._plot;
  if (canvasX < plot.left || canvasX > plot.right || canvasY < plot.top || canvasY > plot.bottom) {
    tooltip.hidden = true;
    return;
  }
  const x = ((canvasX - plot.left) / (plot.right - plot.left)) * 2 * terrain.config.halfSizeX - terrain.config.halfSizeX;
  const y = ((plot.bottom - canvasY) / (plot.bottom - plot.top)) * 2 * terrain.config.halfSizeY - terrain.config.halfSizeY;
  tooltip.textContent = `x ${format(x, 2)} · y ${format(y, 2)} · z ${format(sampleHeight(terrain, x, y), 4)}`;
  tooltip.style.left = `${event.clientX - rectangle.left}px`;
  tooltip.style.top = `${event.clientY - rectangle.top}px`;
  tooltip.hidden = false;
}

function updateUrl(config) {
  const parameters = new URLSearchParams();
  const shortNames = {
    kind: "kind", resolution: "res", amplitude: "amp", featureSize: "feature",
    maxSlopeDegrees: "maxSlope", slopeAngleDegrees: "slope", slopeDirectionDegrees: "direction",
    stairHeight: "stepHeight", stairRun: "stepRun", spawnFlatRadius: "spawn",
    spawnBlendWidth: "blend", halfSizeX: "size", seed: "seed",
  };
  for (const [name, parameter] of Object.entries(shortNames)) {
    if (config[name] !== DEFAULT_CONFIG[name]) parameters.set(parameter, config[name]);
  }
  history.replaceState(null, "", `${location.pathname}${parameters.size ? `?${parameters}` : ""}`);
}

function configFromUrl() {
  const parameters = new URLSearchParams(location.search);
  const mapping = {
    kind: ["kind", String], resolution: ["res", Number], amplitude: ["amp", Number],
    featureSize: ["feature", Number], maxSlopeDegrees: ["maxSlope", Number],
    slopeAngleDegrees: ["slope", Number], slopeDirectionDegrees: ["direction", Number],
    stairHeight: ["stepHeight", Number], stairRun: ["stepRun", Number],
    spawnFlatRadius: ["spawn", Number], spawnBlendWidth: ["blend", Number],
    halfSizeX: ["size", Number], seed: ["seed", Number],
  };
  const config = { ...DEFAULT_CONFIG };
  for (const [name, [parameter, convert]] of Object.entries(mapping)) {
    if (parameters.has(parameter)) config[name] = convert(parameters.get(parameter));
  }
  config.halfSizeY = config.halfSizeX;
  return config;
}

function drawArrow(context, startX, startY, endX, endY, color) {
  const angle = Math.atan2(endY - startY, endX - startX);
  context.strokeStyle = color;
  context.fillStyle = color;
  context.lineWidth = 2;
  context.beginPath(); context.moveTo(startX, startY); context.lineTo(endX, endY); context.stroke();
  context.beginPath();
  context.moveTo(endX, endY);
  context.lineTo(endX - 10 * Math.cos(angle - Math.PI / 6), endY - 10 * Math.sin(angle - Math.PI / 6));
  context.lineTo(endX - 10 * Math.cos(angle + Math.PI / 6), endY - 10 * Math.sin(angle + Math.PI / 6));
  context.closePath(); context.fill();
}

function terrainColor(value) {
  const stops = [
    [-1, [29, 61, 112]], [-0.45, [34, 146, 158]], [0, [235, 229, 133]],
    [0.5, [190, 137, 77]], [1, [120, 62, 48]],
  ];
  const clamped = Math.max(-1, Math.min(1, value));
  for (let index = 0; index < stops.length - 1; index += 1) {
    const [startValue, startColor] = stops[index];
    const [endValue, endColor] = stops[index + 1];
    if (clamped <= endValue) {
      const amount = (clamped - startValue) / (endValue - startValue);
      return startColor.map((channel, channelIndex) => Math.round(channel + amount * (endColor[channelIndex] - channel)));
    }
  }
  return stops.at(-1)[1];
}

function canvasTheme() {
  const dark = matchMedia("(prefers-color-scheme: dark)").matches;
  return dark ? {
    surface: "#18231e", ink: "#edf5f0", muted: "#a3b2aa", axis: "rgba(237,245,240,.42)",
    contour: "rgba(8,15,12,.5)", blend: "#f0a04a", spawn: "#ff7b64", robot: "rgba(117,212,173,.3)",
    spawnFill: "rgba(255,123,100,.12)", grid: "rgba(237,245,240,.1)", zero: "rgba(237,245,240,.55)", profile: "#75d4ad",
  } : {
    surface: "#fffefa", ink: "#18221f", muted: "#62706b", axis: "rgba(24,34,31,.38)",
    contour: "rgba(10,18,15,.42)", blend: "#d56e25", spawn: "#cf3e33", robot: "rgba(20,107,84,.25)",
    spawnFill: "rgba(207,62,51,.1)", grid: "rgba(24,34,31,.1)", zero: "rgba(24,34,31,.52)", profile: "#146b54",
  };
}

function format(value, digits = 2) {
  return Number(value).toFixed(digits).replace(/\.?0+$/, "") || "0";
}

function degrees(radians) { return (radians * 180) / Math.PI; }
function titleCase(value) { return value.charAt(0).toUpperCase() + value.slice(1); }

for (const control of Object.values(controls)) {
  control.addEventListener("input", () => scheduleUpdate(true));
  control.addEventListener("change", () => scheduleUpdate(true));
}

exampleControl.addEventListener("change", () => {
  if (exampleControl.value === "custom") return;
  applyConfig({ ...DEFAULT_CONFIG, ...EXAMPLES[exampleControl.value] });
});

document.getElementById("reset").addEventListener("click", () => {
  exampleControl.value = "custom";
  applyConfig(DEFAULT_CONFIG);
});

document.getElementById("copy-config").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(codeElement.textContent);
  temporaryLabel(event.currentTarget, "Copied config");
});

document.getElementById("copy-link").addEventListener("click", async (event) => {
  await navigator.clipboard.writeText(location.href);
  temporaryLabel(event.currentTarget, "Copied link");
});

function temporaryLabel(button, label) {
  const original = button.textContent;
  button.textContent = label;
  setTimeout(() => { button.textContent = original; }, 1400);
}

mapCanvas.addEventListener("pointermove", mapPointer);
mapCanvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
mapCanvas.addEventListener("touchstart", (event) => mapPointer(event.touches[0]), { passive: true });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => update());

applyConfig(configFromUrl());
