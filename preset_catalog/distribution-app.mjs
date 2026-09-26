import {
  HELD_OUT_NODE_COUNT,
  MAX_TRAINING_SIZE,
  RANDOM_DRAWS,
  coverageCurve,
  coverageOf,
  decodeDistances,
  farthestPointOrder,
  randomBand,
  randomOrder,
  searchPresets,
  spearman,
  splitYaml,
  stratifiedFarthestPointOrder,
  stratifiedRandomOrder,
  targetWeights,
} from "./distribution.mjs";

const SVG_NS = "http://www.w3.org/2000/svg";
// Stops in the --n0 … --n6 sequential ramp defined in distribution.css.
const RAMP_STOPS = 7;
const OCCUPANCY_COLUMNS = [
  { label: "1 tube", short: "1 tube", tubes: 1 },
  { label: "2 tubes", short: "2", tubes: 2 },
  { label: "3 tubes", short: "3", tubes: 3 },
  { label: "4 tubes", short: "4", tubes: 4 },
  { label: "Triangles", short: "Tri.", tubes: null },
];
const MATRIX_DESCRIPTORS = [
  { key: "degree_std", label: "Degree spread" },
  { key: "triangles_per_node", label: "Triangles / node" },
  { key: "symmetry", label: "Symmetry (log₂ |Aut|)" },
  { key: "wcri", label: "WCRI (log)", log: true },
  { key: "resting_faces", label: "Resting faces" },
];
const FAMILY_COLORS = { Henneberg: "var(--series-1)", Usevitch: "var(--series-2)", "Built-in": "var(--series-3)" };
const DESCRIPTOR_NAMES = {
  nodes: "nodes",
  members: "members",
  degree_std: "degree spread",
  triangles_per_node: "triangles / node",
  symmetry: "symmetry",
  wcri: "WCRI",
  resting_faces: "resting faces",
  aspect_ratio: "aspect ratio",
};
const STORAGE_KEY = "mujoco-truss-gen:training-picks";
// Above this many search matches, plots only dim the rest: per-robot rings would be clutter.
const RING_LIMIT = 40;

const response = await fetch("distribution-data.json");
const data = await response.json();
const presets = data.presets;
const NODE_COUNTS = [...new Set(presets.map((preset) => preset.nodes))].sort((a, b) => a - b);
const count = presets.length;
const indexByName = new Map(presets.map((preset, index) => [preset.name, index]));
const trainIndices = Object.values(data.split.groups).flat().map((name) => indexByName.get(name)).filter((index) => index !== undefined);
const trainSet = new Set(trainIndices);
const finalTestSet = new Set(data.split.final_test.map((name) => indexByName.get(name)));
const pool = presets.map((_, index) => index).filter((index) => presets[index].nodes !== HELD_OUT_NODE_COUNT);
const heldOut = presets.map((_, index) => index).filter((index) => presets[index].nodes === HELD_OUT_NODE_COUNT);
const matrices = Object.fromEntries(Object.entries(data.bases).map(([key, basis]) => [key, decodeDistances(basis.distances, count)]));
const orderCache = new Map();
const curveCache = new Map();

const state = {
  basis: "graph",
  colorBy: "nodes",
  hovered: null,
  hoverCell: null,
  cellFilter: null,
  brushed: null,
  brushBox: null,
  search: null,
  picks: loadPicks(),
  metric: "worst",
  target: "pool",
  weighting: "robot",
  k: 12,
};

// ---------- helpers ----------

function svg(tag, attributes = {}, parent = null) {
  const element = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value !== undefined && value !== null) element.setAttribute(name, value);
  }
  if (parent) parent.append(element);
  return element;
}

function text(parent, x, y, content, attributes = {}) {
  const element = svg("text", { x, y, ...attributes }, parent);
  element.textContent = content;
  return element;
}

function node(tag, className, content) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (content !== undefined) element.textContent = content;
  return element;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16));
}

function rampColor(fraction) {
  const stops = Array.from({ length: RAMP_STOPS }, (_, index) => hexToRgb(cssVar(`--n${index}`)));
  const position = Math.max(0, Math.min(1, fraction)) * (stops.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(stops.length - 1, lower + 1);
  const mix = stops[lower].map((channel, index) => Math.round(channel + (stops[upper][index] - channel) * (position - lower)));
  return `rgb(${mix.join(",")})`;
}

function isDark(color) {
  const match = color.match(/\d+/g);
  const [r, g, b] = color.startsWith("#") ? hexToRgb(color) : match.map(Number);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b < 140;
}

function format(value) {
  if (!Number.isFinite(value)) return "–";
  if (value === 0) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 1) return value.toFixed(2).replace(/\.?0+$/, "");
  return value.toPrecision(2);
}

function tickLabel(value) {
  return String(Number(value.toPrecision(3)));
}

const SUPERSCRIPTS = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

function powerLabel(exponent) {
  const rounded = Math.round(exponent);
  if (Math.abs(exponent - rounded) > 1e-9) return tickLabel(10 ** exponent);
  return `10${[...String(rounded)].map((character) => SUPERSCRIPTS[character]).join("")}`;
}

function niceTicks(minimum, maximum, target = 4) {
  const span = maximum - minimum || 1;
  const raw = span / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((factor) => factor * power).find((candidate) => span / candidate <= target) ?? power * 10;
  const ticks = [];
  for (let tick = Math.ceil(minimum / step) * step; tick <= maximum + step * 1e-9; tick += step) ticks.push(Number(tick.toFixed(10)));
  return ticks;
}

function memberLabel(preset) {
  if (preset.member_type === "tube") return `${preset.tube_count} tube${preset.tube_count === 1 ? "" : "s"}`;
  return `${preset.members} triangles`;
}

function roleLabel(index) {
  if (trainSet.has(index)) return "Current split · train";
  if (finalTestSet.has(index)) return "Current split · final test";
  if (presets[index].nodes === HELD_OUT_NODE_COUNT) return "Held out (n = 7)";
  return "";
}

const wcriLogs = presets.map((preset) => Math.log10(Math.max(preset.wcri, 1e-6)));
const wcriRange = [Math.min(...wcriLogs), Math.max(...wcriLogs)];

function colorOf(index) {
  const preset = presets[index];
  if (state.colorBy === "member_type") return preset.member_type === "tube" ? "var(--series-1)" : "var(--series-2)";
  if (state.colorBy === "family") return FAMILY_COLORS[preset.family];
  if (state.colorBy === "wcri") return rampColor((wcriLogs[index] - wcriRange[0]) / (wcriRange[1] - wcriRange[0]));
  return nodeColor(preset.nodes);
}

function nodeColor(nodes) {
  const index = NODE_COUNTS.indexOf(nodes);
  return rampColor(NODE_COUNTS.length > 1 ? index / (NODE_COUNTS.length - 1) : 0);
}

function markerPath(x, y, radius, memberType) {
  if (memberType === "triangle") {
    const r = radius * 1.3;
    return `M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`;
  }
  return `M${x - radius},${y}a${radius},${radius} 0 1,0 ${radius * 2},0a${radius},${radius} 0 1,0 ${-radius * 2},0`;
}

let intersection = { search: null, filter: null, value: null };

/** Robots passing both the search and the cell filter or brush; either alone if only one is set. */
function filteredSet() {
  const filter = state.cellFilter ?? state.brushed;
  if (!state.search || !filter) return state.search ?? filter;
  if (intersection.search !== state.search || intersection.filter !== filter) {
    intersection = { search: state.search, filter, value: new Set([...filter].filter((index) => state.search.has(index))) };
  }
  return intersection.value;
}

function activeSet() {
  return state.hoverCell ?? filteredSet();
}

/** Brushed robots that are still visible under the search. */
function brushedVisible() {
  if (!state.brushed) return [];
  return [...state.brushed].filter((index) => !state.search || state.search.has(index));
}

/**
 * Redraw a few search matches on top of a scatter plot with a ring, so they are not buried under
 * other marks. Skipped above RING_LIMIT, and when nothing it depends on changed since the last draw.
 */
function drawSearchLayer(layer, positions, drawMark, ringRadius) {
  const active = activeSet();
  const key = [state.search, active, state.colorBy];
  if (layer.searchKey?.every((value, position) => value === key[position])) return;
  layer.searchKey = key;
  layer.replaceChildren();
  if (!state.search || state.search.size > RING_LIMIT) return;
  for (const index of state.search) {
    const position = positions[index];
    if (!position) continue;
    const group = svg("g", {}, layer);
    if (active && !active.has(index)) group.classList.add("dim");
    svg("circle", { cx: position[0], cy: position[1], r: ringRadius, class: "search-halo" }, group);
    drawMark(group, index, position);
    svg("circle", { cx: position[0], cy: position[1], r: ringRadius, class: "search-ring" }, group);
  }
}

function pickable(index) {
  return presets[index].nodes !== HELD_OUT_NODE_COUNT;
}

function togglePick(index) {
  if (!pickable(index)) return;
  if (state.picks.has(index)) state.picks.delete(index);
  else state.picks.add(index);
  picksChanged();
}

function loadPicks() {
  try {
    const names = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
    return new Set(names.map((name) => indexByName.get(name)).filter((index) => index !== undefined && pickable(index)));
  } catch {
    return new Set();
  }
}

function savePicks() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...state.picks].map((index) => presets[index].name)));
  } catch {
    // Storage is a convenience; the page works without it.
  }
}

function svgPoint(svgElement, event) {
  const box = svgElement.getBoundingClientRect();
  const viewBox = svgElement.viewBox.baseVal;
  return [((event.clientX - box.left) / box.width) * viewBox.width, ((event.clientY - box.top) / box.height) * viewBox.height];
}

// ---------- tooltip ----------

const tooltip = document.getElementById("tooltip");

function showTooltip(event, build) {
  tooltip.replaceChildren();
  build(tooltip);
  tooltip.hidden = false;
  const box = tooltip.getBoundingClientRect();
  let left = event.clientX + 16;
  let top = event.clientY + 16;
  if (left + box.width > window.innerWidth - 8) left = event.clientX - box.width - 16;
  if (top + box.height > window.innerHeight - 8) top = event.clientY - box.height - 16;
  tooltip.style.left = `${Math.max(8, left)}px`;
  tooltip.style.top = `${Math.max(8, top)}px`;
}

function hideTooltip() {
  tooltip.hidden = true;
}

function row(parent, label, value) {
  const line = node("div", "row");
  line.append(node("span", "muted", label), node("span", "value", String(value)));
  parent.append(line);
}

function robotTooltip(index) {
  return (element) => {
    const preset = presets[index];
    const image = node("img");
    image.src = preset.image;
    image.alt = "";
    element.append(image, node("strong", "", preset.name));
    row(element, "Nodes", preset.nodes);
    row(element, "Members", memberLabel(preset));
    row(element, "Family", preset.family);
    row(element, "WCRI", format(preset.wcri));
    row(element, "Automorphisms", Math.round(2 ** preset.symmetry));
    row(element, "Resting faces", preset.resting_faces);
    const role = roleLabel(index);
    if (role) element.append(node("div", "muted", role));
    const hint = !pickable(index) ? "Held out, can’t be picked" : state.picks.has(index) ? "Click to remove from picks" : "Click to add to picks";
    element.append(node("div", "muted", hint));
  };
}

function setHovered(index, event) {
  if (index === null) {
    if (state.hovered !== null) {
      state.hovered = null;
      emphasize();
    }
    hideTooltip();
    return;
  }
  if (state.hovered !== index) {
    state.hovered = index;
    emphasize();
  }
  showTooltip(event, robotTooltip(index));
}

// ---------- legend & toolbar ----------

function legendKey(parent, drawMark, label) {
  const key = node("span", "key");
  const mark = svg("svg", { width: 14, height: 14, viewBox: "-7 -7 14 14", "aria-hidden": "true" });
  drawMark(mark);
  key.append(mark, document.createTextNode(label));
  parent.append(key);
}

function renderLegend() {
  const legend = document.getElementById("legend");
  legend.replaceChildren();
  const dot = (color) => (mark) => svg("circle", { r: 4.5, style: `fill:${color}` }, mark);
  if (state.colorBy === "nodes") {
    NODE_COUNTS.forEach((nodes) => legendKey(legend, dot(nodeColor(nodes)), `n = ${nodes}`));
  } else if (state.colorBy === "member_type") {
    legendKey(legend, dot("var(--series-1)"), "Tube");
    legendKey(legend, dot("var(--series-2)"), "Triangle");
  } else if (state.colorBy === "family") {
    for (const [family, color] of Object.entries(FAMILY_COLORS)) legendKey(legend, dot(color), family);
  } else {
    const ramp = node("span", "ramp");
    const bar = node("span", "ramp-bar");
    bar.style.background = `linear-gradient(90deg, ${Array.from({ length: RAMP_STOPS }, (_, index) => `var(--n${index})`).join(",")})`;
    ramp.append(node("span", "", powerLabel(wcriRange[0])), bar, node("span", "", `${powerLabel(wcriRange[1])} WCRI`));
    legend.append(ramp);
  }
  if (state.colorBy !== "member_type") {
    const shape = (memberType) => (mark) => svg("path", { d: markerPath(0, 0, 4, memberType), style: "fill:var(--muted)" }, mark);
    legendKey(legend, shape("tube"), "Tube");
    legendKey(legend, shape("triangle"), "Triangle");
  }
  legendKey(legend, (mark) => svg("circle", { r: 5, style: "fill:none;stroke:var(--ink);stroke-width:1.6" }, mark), "Current train");
  legendKey(legend, (mark) => svg("circle", { r: 5, style: "fill:none;stroke:var(--ink);stroke-width:1.4;stroke-dasharray:2.2 1.8" }, mark), "Final test");
  legendKey(legend, (mark) => svg("circle", { r: 5, style: "fill:none;stroke:var(--accent);stroke-width:2.4" }, mark), "Your pick");
}

function renderPickControls() {
  const size = state.picks.size;
  document.getElementById("pick-count").textContent = `${size} pick${size === 1 ? "" : "s"}`;
  document.getElementById("clear-picks").disabled = size === 0;
  document.getElementById("copy-yaml").disabled = size === 0;
  const brushButton = document.getElementById("add-brushed");
  const brushable = brushedVisible().filter(pickable).length;
  brushButton.hidden = brushable === 0;
  brushButton.textContent = `Add ${brushable} brushed`;
}

// ---------- plot 1: occupancy ----------

const occupancyCells = NODE_COUNTS.map((nodes) =>
  OCCUPANCY_COLUMNS.map((column) => {
    const members = presets
      .map((preset, index) => [preset, index])
      .filter(([preset]) => preset.nodes === nodes && (column.tubes === null ? preset.member_type === "triangle" : preset.member_type === "tube" && preset.tube_count === column.tubes))
      .map(([, index]) => index);
    const gap = column.tubes === null ? null : data.gaps.find((entry) => entry.nodes === nodes && entry.tubes === column.tubes) ?? null;
    return {
      nodes,
      column,
      members: new Set(members),
      inSplit: members.filter((index) => trainSet.has(index) || finalTestSet.has(index)).length,
      gap,
      enumerated: column.tubes !== null && nodes <= 8,
    };
  }),
);
const occupancyMax = Math.max(...occupancyCells.flat().map((cell) => cell.members.size));

const occupancyChart = {
  element: document.getElementById("occupancy"),
  render() {
    const width = Math.max(320, this.element.clientWidth);
    const labelWidth = 70;
    const headerHeight = 30;
    const cellWidth = (width - labelWidth - 8) / OCCUPANCY_COLUMNS.length;
    const cellHeight = 50;
    const height = headerHeight + NODE_COUNTS.length * cellHeight + 8;
    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Robot counts by node count and member type" });
    const defs = svg("defs", {}, root);
    const pattern = svg("pattern", { id: "hatch", width: 7, height: 7, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
    svg("line", { x1: 0, y1: 0, x2: 0, y2: 7, style: "stroke:var(--hatch);stroke-width:2.5" }, pattern);

    OCCUPANCY_COLUMNS.forEach((column, col) => {
      text(root, labelWidth + (col + 0.5) * cellWidth, 19, cellWidth < 72 ? column.short : column.label, { "text-anchor": "middle", class: "strong" });
    });
    this.cells = [];
    // Cells are rebuilt below, so the one under the pointer will never see its pointerleave.
    state.hoverCell = null;
    NODE_COUNTS.forEach((nodes, rowIndex) => {
      const y = headerHeight + rowIndex * cellHeight;
      text(root, 8, y + cellHeight / 2 + (nodes === HELD_OUT_NODE_COUNT ? -2 : 4), `n = ${nodes}`, { class: "strong" });
      if (nodes === HELD_OUT_NODE_COUNT) text(root, 8, y + cellHeight / 2 + 12, "held out", { class: "faint", "font-size": 10 });
      occupancyCells[rowIndex].forEach((cell, col) => {
        const x = labelWidth + col * cellWidth;
        const available = cell.members.size;
        const group = svg("g", { class: "cell" }, root);
        const box = { x: x + 1, y: y + 1, width: cellWidth - 2, height: cellHeight - 2, rx: 6 };
        let fill = "transparent";
        if (available > 0) {
          const fraction = Math.log(available + 1) / Math.log(occupancyMax + 1);
          fill = cssVar(`--n${Math.round(fraction * (RAMP_STOPS - 1))}`);
          svg("rect", { ...box, style: `fill:${fill}` }, group);
        } else {
          svg("rect", { ...box, style: "fill:var(--grid)" }, group);
        }
        if (cell.gap) {
          const hatchHeight = available > 0 ? 9 : box.height;
          svg("rect", { x: box.x, y: box.y + box.height - hatchHeight, width: box.width, height: hatchHeight, rx: available > 0 ? 3 : 6, style: "fill:url(#hatch)" }, group);
        }
        const labelClass = available > 0 ? "" : "faint";
        const labelStyle = available > 0 ? `fill:${isDark(fill) ? "#ffffff" : "#0b0b0b"};font-weight:700` : "";
        const centerX = x + cellWidth / 2;
        let label = "0";
        if (available > 0) label = `${cell.inSplit} / ${available}`;
        else if (cell.gap) label = `${cell.gap.missing} missing`;
        else if (!cell.enumerated && cell.column.tubes !== null) label = "–";
        text(group, centerX, y + cellHeight / 2 + 4, label, { "text-anchor": "middle", class: labelClass, style: labelStyle, "font-size": 12 });
        if (available > 0 || cell.gap) {
          group.style.cursor = available > 0 ? "pointer" : "default";
          group.addEventListener("pointermove", (event) => {
            if (available > 0 && state.hoverCell !== cell.members) {
              state.hoverCell = cell.members;
              emphasize();
            }
            showTooltip(event, (element) => occupancyTooltip(element, cell));
          });
          group.addEventListener("pointerleave", () => {
            state.hoverCell = null;
            emphasize();
            hideTooltip();
          });
          if (available > 0) {
            group.addEventListener("click", () => {
              state.cellFilter = state.cellFilter === cell.members ? null : cell.members;
              state.brushed = null;
              render(["occupancy", "matrix"]);
              emphasize();
              renderPickControls();
            });
          }
        }
        if (state.search) {
          const hits = [...cell.members].filter((index) => state.search.has(index)).length;
          if (hits) {
            svg("rect", { x: box.x + 2.5, y: box.y + 2.5, width: box.width - 5, height: box.height - 5, rx: 4, class: "search-ring", style: "stroke-width:2.5" }, group);
            const badgeWidth = 7 + 6 * String(hits).length;
            svg("rect", { x: box.x + box.width - badgeWidth - 3, y: box.y + 3, width: badgeWidth, height: 13, rx: 6.5, class: "search-badge" }, group);
            text(group, box.x + box.width - badgeWidth / 2 - 3, box.y + 13, String(hits), { "text-anchor": "middle", class: "search-badge-text" });
          } else {
            group.style.opacity = ".35";
          }
        }
        if (state.cellFilter === cell.members) {
          svg("rect", { ...box, style: "fill:none;stroke:var(--ink);stroke-width:2" }, group);
        }
      });
    });
    this.element.replaceChildren(root);
  },
};

function occupancyTooltip(element, cell) {
  const heading = cell.column.tubes === null ? "triangle robots" : cell.column.label;
  element.append(node("strong", "", `n = ${cell.nodes}, ${heading}`));
  row(element, "In library", cell.members.size);
  row(element, "In current split", cell.inSplit);
  if (state.search) row(element, "Search matches", [...cell.members].filter((index) => state.search.has(index)).length);
  if (cell.gap) row(element, "Missing graphs", `${cell.gap.missing} (${cell.gap.reason})`);
  if (cell.members.size) element.append(node("div", "muted", state.cellFilter === cell.members ? "Click to clear the filter" : "Click to filter the other plots"));
}

// ---------- plot 2: topology map ----------

const mapChart = {
  element: document.getElementById("map"),
  render() {
    const width = Math.max(300, this.element.clientWidth);
    const height = Math.round(Math.max(300, Math.min(width * 0.78, 580)));
    const pad = 22;
    const layout = data.bases[state.basis].layout;
    const xs = layout.map(([x]) => x);
    const ys = layout.map(([, y]) => y);
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const scale = Math.min((width - 2 * pad) / (maxX - minX || 1), (height - 2 * pad) / (maxY - minY || 1));
    const offsetX = (width - (maxX - minX) * scale) / 2;
    const offsetY = (height - (maxY - minY) * scale) / 2;
    const positions = layout.map(([x, y]) => [offsetX + (x - minX) * scale, height - offsetY - (y - minY) * scale]);
    fanOut(positions, 5.5);
    this.positions = positions;

    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Topology map of every preset" });
    const order = presets.map((_, index) => index).sort((a, b) => drawRank(a) - drawRank(b));
    this.marks = new Map();
    for (const index of order) {
      const [x, y] = positions[index];
      const preset = presets[index];
      const group = svg("g", {}, root);
      svg("path", { d: markerPath(x, y, 4.3, preset.member_type), style: `fill:${colorOf(index)};stroke:var(--surface-strong);stroke-width:1.5` }, group);
      if (trainSet.has(index)) svg("circle", { cx: x, cy: y, r: 8, style: "fill:none;stroke:var(--ink);stroke-width:1.6" }, group);
      if (finalTestSet.has(index)) svg("circle", { cx: x, cy: y, r: 8, style: "fill:none;stroke:var(--ink);stroke-width:1.4;stroke-dasharray:2.4 2" }, group);
      if (state.picks.has(index)) svg("circle", { cx: x, cy: y, r: trainSet.has(index) ? 11 : 8, style: "fill:none;stroke:var(--accent);stroke-width:2.4" }, group);
      this.marks.set(index, group);
    }
    this.width = width;
    this.searchLayer = svg("g", { style: "pointer-events:none" }, root);
    this.hover = svg("circle", { r: 10, style: "fill:none;stroke:var(--ink);stroke-width:2;pointer-events:none", visibility: "hidden" }, root);
    root.addEventListener("pointermove", (event) => setHovered(nearest(positions, svgPoint(root, event), 16), event));
    root.addEventListener("pointerleave", () => setHovered(null));
    root.addEventListener("click", (event) => {
      const index = nearest(positions, svgPoint(root, event), 16);
      if (index !== null) togglePick(index);
    });
    this.element.replaceChildren(root);

    const basis = data.bases[state.basis];
    const notes = [`The 2D map keeps ${Math.round(basis.explained * 100)}% of the distance variance, so treat small gaps loosely.`];
    if (state.basis === "graph" && data.wl_collisions) notes.push(`${data.wl_collisions} robots share a WL signature with a different graph.`);
    else if (state.basis === "descriptors") notes.push(`Uses z-scored ${data.distance_descriptors.map((key) => DESCRIPTOR_NAMES[key] ?? key).join(", ")}.`);
    document.getElementById("map-note").textContent = notes.join(" ");
  },
  emphasize(active) {
    if (!this.marks) return;
    for (const [index, group] of this.marks) group.classList.toggle("dim", Boolean(active) && !active.has(index));
    drawSearchLayer(this.searchLayer, this.positions, (group, index, [x, y]) => {
      svg("path", { d: markerPath(x, y, 4.3, presets[index].member_type), style: `fill:${colorOf(index)};stroke:var(--surface-strong);stroke-width:1.5` }, group);
      if (state.search.size !== 1) return;
      const right = x < this.width * 0.62;
      text(group, x + (right ? 20 : -20), y + 4, presets[index].name, { "text-anchor": right ? "start" : "end", class: "search-label" });
    }, 15);
    placeHover(this.hover, state.hovered === null ? null : this.positions[state.hovered]);
  },
};

function drawRank(index) {
  let rank = presets[index].nodes === 8 ? 0 : 1;
  if (trainSet.has(index) || finalTestSet.has(index)) rank += 2;
  if (state.picks.has(index)) rank += 4;
  return rank;
}

function fanOut(positions, radius) {
  const buckets = new Map();
  positions.forEach(([x, y], index) => {
    const key = `${Math.round(x)}:${Math.round(y)}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(index);
  });
  for (const members of buckets.values()) {
    if (members.length < 2) continue;
    members.forEach((index, position) => {
      const angle = (2 * Math.PI * position) / members.length;
      positions[index] = [positions[index][0] + radius * Math.cos(angle), positions[index][1] + radius * Math.sin(angle)];
    });
  }
}

function nearest(positions, [x, y], limit, candidates = null) {
  let best = null;
  let bestDistance = limit * limit;
  const indices = candidates ?? positions.keys();
  for (const index of indices) {
    const position = positions[index];
    if (!position) continue;
    const distance = (position[0] - x) ** 2 + (position[1] - y) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  return best;
}

function placeHover(element, position) {
  if (!element) return;
  if (!position) {
    element.setAttribute("visibility", "hidden");
    return;
  }
  element.setAttribute("cx", position[0]);
  element.setAttribute("cy", position[1]);
  element.setAttribute("visibility", "visible");
}

// ---------- plot 3: descriptor matrix ----------

function jitter(index, salt) {
  const value = Math.sin(index * 12.9898 + salt * 78.233) * 43758.5453;
  return value - Math.floor(value) - 0.5;
}

const matrixValues = MATRIX_DESCRIPTORS.map((descriptor, salt) => {
  const raw = presets.map((preset) => (descriptor.log ? Math.log10(Math.max(preset[descriptor.key], 1e-6)) : preset[descriptor.key]));
  const unique = [...new Set(raw.map((value) => value.toFixed(6)))].map(Number).sort((a, b) => a - b);
  const discrete = unique.length <= 25;
  let spread = 0;
  if (discrete && unique.length > 1) spread = 0.32 * Math.min(...unique.slice(1).map((value, position) => value - unique[position]));
  const plotted = raw.map((value, index) => value + (discrete ? spread * 2 * jitter(index, salt) : 0));
  const minimum = Math.min(...plotted);
  const maximum = Math.max(...plotted);
  const margin = (maximum - minimum || 1) * 0.06;
  return { raw, plotted, domain: [minimum - margin, maximum + margin], discrete, unique };
});
const nodeEight = presets.map((_, index) => index).filter((index) => presets[index].nodes === 8);
const correlations = MATRIX_DESCRIPTORS.map((_, a) =>
  MATRIX_DESCRIPTORS.map((__, b) => ({
    all: spearman(matrixValues[a].raw, matrixValues[b].raw),
    within: spearman(nodeEight.map((index) => matrixValues[a].raw[index]), nodeEight.map((index) => matrixValues[b].raw[index])),
  })),
);

const matrixChart = {
  element: document.getElementById("matrix"),
  render() {
    const available = this.element.clientWidth;
    const count_ = MATRIX_DESCRIPTORS.length;
    const left = 58;
    const bottom = 42;
    const gap = 10;
    const width = Math.max(available, 700);
    const panelWidth = (width - left - 10 - gap * (count_ - 1)) / count_;
    const panelHeight = Math.max(92, Math.min(150, panelWidth * 0.72));
    const height = 12 + count_ * panelHeight + (count_ - 1) * gap + bottom;
    this.element.style.overflowX = width > available ? "auto" : "";
    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, width: width > available ? width : null, role: "img", "aria-label": "Scatter matrix of robot descriptors" });
    const active = activeSet();
    this.panels = [];
    this.hovers = [];
    this.pointMarks = [];
    this.searchLayers = [];

    for (let rowIndex = 0; rowIndex < count_; rowIndex += 1) {
      for (let col = 0; col < count_; col += 1) {
        const x0 = left + col * (panelWidth + gap);
        const y0 = 12 + rowIndex * (panelHeight + gap);
        const panel = svg("g", {}, root);
        svg("rect", { x: x0, y: y0, width: panelWidth, height: panelHeight, rx: 6, style: "fill:none;stroke:var(--grid)" }, panel);
        const xDomain = matrixValues[col].domain;
        const yDomain = matrixValues[rowIndex].domain;
        const sx = (value) => x0 + ((value - xDomain[0]) / (xDomain[1] - xDomain[0])) * panelWidth;
        const sy = (value) => y0 + panelHeight - ((value - yDomain[0]) / (yDomain[1] - yDomain[0])) * panelHeight;

        if (rowIndex === col) {
          drawHistogram(panel, matrixValues[col], x0, y0, panelWidth, panelHeight, sx);
          text(panel, x0 + 7, y0 + 15, MATRIX_DESCRIPTORS[col].label, { class: "title-text", "font-size": 11 });
        } else if (rowIndex > col) {
          const positions = [];
          for (let index = 0; index < count; index += 1) {
            positions[index] = [sx(matrixValues[col].plotted[index]), sy(matrixValues[rowIndex].plotted[index])];
          }
          const order = presets.map((_, index) => index).sort((a, b) => drawRank(a) - drawRank(b));
          for (const index of order) {
            const [x, y] = positions[index];
            const mark = svg("circle", { cx: x, cy: y, r: 2.6, style: `fill:${colorOf(index)};fill-opacity:.85` }, panel);
            if (active && !active.has(index)) mark.classList.add("dim");
            this.pointMarks.push([index, mark]);
          }
          this.panels.push({ x0, y0, width: panelWidth, height: panelHeight, positions });
          this.searchLayers.push([svg("g", { style: "pointer-events:none" }, panel), positions]);
          this.hovers.push([svg("circle", { r: 6, style: "fill:none;stroke:var(--ink);stroke-width:1.8;pointer-events:none", visibility: "hidden" }, panel), positions]);
        } else {
          drawCorrelation(panel, correlations[rowIndex][col], x0, y0, panelWidth, panelHeight);
        }

        if (col === 0 && rowIndex > 0) drawAxisTicks(root, "y", MATRIX_DESCRIPTORS[rowIndex], matrixValues[rowIndex], x0, sy);
        if (rowIndex === count_ - 1) drawAxisTicks(root, "x", MATRIX_DESCRIPTORS[col], matrixValues[col], y0 + panelHeight, sx);
      }
    }
    MATRIX_DESCRIPTORS.forEach((descriptor, col) => {
      text(root, left + col * (panelWidth + gap) + panelWidth / 2, height - 6, descriptor.label, { "text-anchor": "middle", class: "strong", "font-size": 11 });
    });

    this.brushRect = svg("rect", { style: "fill:var(--accent);fill-opacity:.1;stroke:var(--accent);stroke-width:1", visibility: "hidden" }, root);
    this.attachPointer(root);
    this.drawBrushBox();
    this.element.replaceChildren(root);
  },
  panelAt([x, y]) {
    return this.panels.find((panel) => x >= panel.x0 && x <= panel.x0 + panel.width && y >= panel.y0 && y <= panel.y0 + panel.height) ?? null;
  },
  attachPointer(root) {
    let brush = null;
    root.addEventListener("pointerdown", (event) => {
      const point = svgPoint(root, event);
      const panel = this.panelAt(point);
      if (!panel) return;
      brush = { panel, start: point, end: point, moved: false };
      root.setPointerCapture(event.pointerId);
    });
    root.addEventListener("pointermove", (event) => {
      const point = svgPoint(root, event);
      if (brush) {
        const { panel } = brush;
        brush.end = [Math.max(panel.x0, Math.min(panel.x0 + panel.width, point[0])), Math.max(panel.y0, Math.min(panel.y0 + panel.height, point[1]))];
        brush.moved ||= Math.hypot(brush.end[0] - brush.start[0], brush.end[1] - brush.start[1]) > 4;
        if (brush.moved) {
          const [x, y, w, h] = rectFrom(brush.start, brush.end);
          Object.entries({ x, y, width: w, height: h }).forEach(([name, value]) => this.brushRect.setAttribute(name, value));
          this.brushRect.setAttribute("visibility", "visible");
          hideTooltip();
        }
        return;
      }
      const panel = this.panelAt(point);
      setHovered(panel ? nearest(panel.positions, point, 9) : null, event);
    });
    root.addEventListener("pointerup", (event) => {
      if (!brush) return;
      const { panel, start, end, moved } = brush;
      brush = null;
      if (moved) {
        const [x, y, w, h] = rectFrom(start, end);
        state.brushBox = {
          panel: this.panels.indexOf(panel),
          box: [(x - panel.x0) / panel.width, (y - panel.y0) / panel.height, w / panel.width, h / panel.height],
        };
        const selected = new Set();
        panel.positions.forEach(([px, py], index) => {
          if (px >= x && px <= x + w && py >= y && py <= y + h) selected.add(index);
        });
        state.brushed = selected.size ? selected : null;
        state.cellFilter = null;
      } else {
        const index = nearest(panel.positions, svgPoint(root, event), 9);
        if (index !== null) togglePick(index);
        else state.brushed = null;
      }
      render(["occupancy"]);
      emphasize();
      renderPickControls();
    });
    root.addEventListener("pointerleave", () => {
      if (!brush) setHovered(null);
    });
  },
  drawBrushBox() {
    const panel = state.brushed && state.brushBox ? this.panels[state.brushBox.panel] : null;
    if (!panel) {
      this.brushRect.setAttribute("visibility", "hidden");
      return;
    }
    const [fx, fy, fw, fh] = state.brushBox.box;
    const box = { x: panel.x0 + fx * panel.width, y: panel.y0 + fy * panel.height, width: fw * panel.width, height: fh * panel.height };
    Object.entries(box).forEach(([name, value]) => this.brushRect.setAttribute(name, value));
    this.brushRect.setAttribute("visibility", "visible");
  },
  emphasize(active) {
    if (!this.pointMarks) return;
    this.drawBrushBox();
    for (const [index, mark] of this.pointMarks) mark.classList.toggle("dim", Boolean(active) && !active.has(index));
    for (const [layer, positions] of this.searchLayers) {
      drawSearchLayer(layer, positions, (group, index, [x, y]) => svg("circle", { cx: x, cy: y, r: 2.6, style: `fill:${colorOf(index)}` }, group), 5.5);
    }
    for (const [hover, positions] of this.hovers) placeHover(hover, state.hovered === null ? null : positions[state.hovered]);
  },
};

function rectFrom([x1, y1], [x2, y2]) {
  return [Math.min(x1, x2), Math.min(y1, y2), Math.abs(x2 - x1), Math.abs(y2 - y1)];
}

function drawHistogram(parent, values, x0, y0, width, height, sx) {
  const [low, high] = values.domain;
  const bins = values.discrete ? values.unique.map((value) => [value, value]) : Array.from({ length: 14 }, (_, index) => [low + ((high - low) * index) / 14, low + ((high - low) * (index + 1)) / 14]);
  const counts = bins.map(([start, end], position) =>
    values.raw.filter((value) => (values.discrete ? Math.abs(value - start) < 1e-6 : value >= start && (value < end || position === bins.length - 1))).length,
  );
  const maximum = Math.max(...counts);
  const top = y0 + 24;
  const usable = height - 28;
  const barWidth = values.discrete ? Math.max(3, Math.min(16, (width / Math.max(bins.length, 1)) * 0.6)) : width / bins.length - 2;
  bins.forEach(([start, end], position) => {
    if (!counts[position]) return;
    const barHeight = Math.max(2, (counts[position] / maximum) * usable);
    const center = values.discrete ? sx(start) : (sx(start) + sx(end)) / 2;
    const y = top + usable - barHeight;
    const r = Math.min(3, barWidth / 2, barHeight);
    const x = center - barWidth / 2;
    svg("path", { d: `M${x},${y + barHeight}V${y + r}q0,${-r} ${r},${-r}h${barWidth - 2 * r}q${r},0 ${r},${r}V${y + barHeight}Z`, style: "fill:var(--n3)" }, parent);
  });
}

function drawCorrelation(parent, correlation, x0, y0, width, height) {
  const lines = [
    ["All", correlation.all],
    ["n = 8", correlation.within],
  ];
  const barMax = width - 86;
  lines.forEach(([label, value], position) => {
    const y = y0 + height / 2 - 12 + position * 26;
    text(parent, x0 + 10, y + 4, label, { class: "faint", "font-size": 10 });
    text(parent, x0 + width - 10, y + 4, Number.isFinite(value) ? `ρ ${value >= 0 ? "" : "−"}${Math.abs(value).toFixed(2)}` : "ρ –", { "text-anchor": "end", class: position === 1 ? "strong" : "", "font-size": 12 });
    svg("rect", { x: x0 + 44, y: y - 3, width: Math.max(0, barMax - 44), height: 6, rx: 3, style: "fill:var(--grid)" }, parent);
    if (Number.isFinite(value)) svg("rect", { x: x0 + 44, y: y - 3, width: Math.max(1.5, Math.abs(value) * (barMax - 44)), height: 6, rx: 3, style: `fill:${position === 1 ? "var(--ink)" : "var(--faint)"}` }, parent);
  });
}

function drawAxisTicks(parent, axis, descriptor, values, anchor, scale) {
  const [low, high] = values.domain;
  let ticks = niceTicks(low, high, 3).filter((tick) => tick >= low && tick <= high);
  if (descriptor.log) {
    const exponents = [];
    const step = Math.max(1, Math.ceil((Math.floor(high) - Math.ceil(low)) / 3));
    for (let exponent = Math.ceil(low); exponent <= high; exponent += step) exponents.push(exponent);
    ticks = exponents;
  } else if (values.discrete && values.unique.length <= 4) {
    ticks = values.unique;
  }
  for (const tick of ticks) {
    const label = descriptor.log ? powerLabel(tick) : tickLabel(tick);
    if (axis === "y") text(parent, anchor - 6, scale(tick) + 3.5, label, { "text-anchor": "end", "font-size": 10 });
    else text(parent, scale(tick), anchor + 13, label, { "text-anchor": "middle", "font-size": 10 });
  }
}

// ---------- plot 4: coverage ----------

function cachedOrder(kind) {
  const key = `${state.basis}:${kind}`;
  if (!orderCache.has(key)) {
    const matrix = matrices[state.basis];
    const strataOf = (index) => presets[index].nodes;
    const limit = Math.min(MAX_TRAINING_SIZE, pool.length);
    orderCache.set(key, kind === "fps" ? farthestPointOrder(matrix, count, pool, limit) : stratifiedFarthestPointOrder(matrix, count, pool, strataOf, limit));
  }
  return orderCache.get(key);
}

function coverageSeries() {
  const matrix = matrices[state.basis];
  const targets = state.target === "held" ? heldOut : pool;
  const weights = targetWeights(presets, targets, state.weighting === "size");
  const limit = Math.min(MAX_TRAINING_SIZE, pool.length);
  const strataOf = (index) => presets[index].nodes;
  const metric = state.metric;
  const key = [state.basis, metric, state.target, state.weighting].join(":");
  if (!curveCache.has(key)) {
    const random = randomBand(matrix, count, (rng) => randomOrder(pool, limit, rng), targets, weights, metric, RANDOM_DRAWS, 1);
    const stratified = randomBand(matrix, count, (rng) => stratifiedRandomOrder(pool, strataOf, limit, rng), targets, weights, metric, RANDOM_DRAWS, 2);
    curveCache.set(key, {
      band: random,
      lines: [
        { label: "Farthest-point", values: coverageCurve(matrix, count, cachedOrder("fps"), targets, weights)[metric], color: "var(--series-1)" },
        { label: "Stratified farthest-point", values: coverageCurve(matrix, count, cachedOrder("sfps"), targets, weights)[metric], color: "var(--series-1)", dash: "6 4" },
        { label: "Random (median)", values: random.median, color: "var(--series-2)" },
        { label: "Stratified random", values: stratified.median, color: "var(--series-2)", dash: "6 4" },
      ],
    });
  }
  const current = coverageOf(matrix, count, trainIndices, targets, weights);
  const picks = coverageOf(matrix, count, [...state.picks], targets, weights);
  return {
    limit,
    ...curveCache.get(key),
    current: current && { k: trainIndices.length, value: current[metric] },
    picks: picks && { k: state.picks.size, value: picks[metric] },
  };
}

const coverageChart = {
  element: document.getElementById("coverage"),
  render() {
    const series = coverageSeries();
    const available = this.element.clientWidth;
    const width = Math.max(available, 560);
    const height = Math.round(Math.max(280, Math.min(400, width * 0.36)));
    const margin = { left: 56, right: 200, top: 18, bottom: 40 };
    this.element.style.overflowX = width > available ? "auto" : "";
    const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, width: width > available ? width : null, role: "img", "aria-label": "Coverage versus training-set size" });
    const plotRight = width - margin.right;
    const plotBottom = height - margin.bottom;
    const maxY = Math.max(...series.band.high, ...series.lines.flatMap((line) => line.values), series.current?.value ?? 0, series.picks?.value ?? 0) * 1.05 || 1;
    const sx = (k) => margin.left + ((k - 1) / (series.limit - 1)) * (plotRight - margin.left);
    const sy = (value) => plotBottom - (value / maxY) * (plotBottom - margin.top);

    for (const tick of niceTicks(0, maxY, 4)) {
      svg("line", { x1: margin.left, x2: plotRight, y1: sy(tick), y2: sy(tick), class: tick === 0 ? "axis" : "grid" }, root);
      text(root, margin.left - 8, sy(tick) + 3.5, tickLabel(tick), { "text-anchor": "end" });
    }
    for (const tick of [1, 8, 16, 24, 32, 40, 48, 56, 64].filter((k) => k <= series.limit)) {
      text(root, sx(tick), plotBottom + 16, String(tick), { "text-anchor": "middle" });
    }
    text(root, (margin.left + plotRight) / 2, height - 6, "Training-set size k", { "text-anchor": "middle", class: "strong" });
    const yLabel = text(root, 14, (margin.top + plotBottom) / 2, `${state.metric === "worst" ? "Worst-case" : "Mean"} relative distance`, { "text-anchor": "middle", class: "strong", transform: `rotate(-90 14 ${(margin.top + plotBottom) / 2})` });
    yLabel.setAttribute("font-size", 11);

    svg("line", { x1: sx(state.k), x2: sx(state.k), y1: margin.top, y2: plotBottom, style: "stroke:var(--accent);stroke-width:1;stroke-opacity:.45" }, root);

    const area = series.band.high.map((value, k) => `${k ? "L" : "M"}${sx(k + 1)},${sy(value)}`).join("") +
      series.band.low.map((_, k, low) => `L${sx(low.length - k)},${sy(low[low.length - 1 - k])}`).join("") + "Z";
    svg("path", { d: area, style: "fill:var(--series-2);fill-opacity:.1" }, root);
    for (const line of series.lines) {
      const d = line.values.map((value, k) => `${k ? "L" : "M"}${sx(k + 1)},${sy(value)}`).join("");
      svg("path", { d, style: `fill:none;stroke:${line.color};stroke-width:2;stroke-linejoin:round;stroke-linecap:round${line.dash ? `;stroke-dasharray:${line.dash}` : ""}` }, root);
    }
    drawEndLabels(root, series.lines, sx(series.limit), sy, margin.top, plotBottom);
    drawSearchJoins(root, series, sx, sy);

    const markers = [
      series.current && { ...series.current, label: `Current split (${series.current.k})`, style: "fill:var(--surface-strong);stroke:var(--ink);stroke-width:2" },
      series.picks && series.picks.k <= series.limit && { ...series.picks, label: `Your picks (${series.picks.k})`, style: "fill:var(--accent);stroke:var(--surface-strong);stroke-width:2" },
    ].filter(Boolean);
    markers.forEach((marker, position) => {
      const x = sx(marker.k);
      const y = sy(marker.value);
      svg("circle", { cx: x, cy: y, r: 5.5, style: marker.style }, root);
      const above = position === 0 || Math.abs(sy(markers[0].value) - y) > 18 || Math.abs(sx(markers[0].k) - x) > 90;
      text(root, x + 9, y + (above ? -9 : 16), marker.label, { class: "strong", "font-size": 11 });
    });

    const crosshair = svg("line", { y1: margin.top, y2: plotBottom, style: "stroke:var(--muted);stroke-width:1", visibility: "hidden" }, root);
    root.addEventListener("pointermove", (event) => {
      const [x] = svgPoint(root, event);
      if (x < margin.left - 10 || x > plotRight + 10) {
        crosshair.setAttribute("visibility", "hidden");
        hideTooltip();
        return;
      }
      const k = Math.max(1, Math.min(series.limit, Math.round(1 + ((x - margin.left) / (plotRight - margin.left)) * (series.limit - 1))));
      crosshair.setAttribute("x1", sx(k));
      crosshair.setAttribute("x2", sx(k));
      crosshair.setAttribute("visibility", "visible");
      showTooltip(event, (element) => {
        element.append(node("strong", "", `k = ${k}`));
        for (const line of series.lines) row(element, line.label, format(line.values[k - 1]));
        row(element, "Random 10–90%", `${format(series.band.low[k - 1])}–${format(series.band.high[k - 1])}`);
      });
    });
    root.addEventListener("pointerleave", () => {
      crosshair.setAttribute("visibility", "hidden");
      hideTooltip();
    });
    this.element.replaceChildren(root);
  },
};

/** Mark the k at which each search match joins the two farthest-point orders, and say so below the chart. */
function drawSearchJoins(parent, series, sx, sy) {
  const note = document.getElementById("coverage-search");
  note.hidden = !state.search;
  if (!state.search) return;
  const orders = [
    { order: cachedOrder("fps"), line: series.lines[0], label: "farthest-point" },
    { order: cachedOrder("sfps"), line: series.lines[1], label: "stratified farthest-point" },
  ];
  note.replaceChildren();
  if (!state.search.size) {
    note.append("No robots match the search.");
    return;
  }
  const joins = orders.map(({ order }) => [...state.search].map((index) => order.indexOf(index) + 1).filter((k) => k > 0 && k <= series.limit));
  if (state.search.size <= RING_LIMIT) {
    orders.forEach(({ line }, position) => {
      for (const k of joins[position]) {
        const [x, y] = [sx(k), sy(line.values[k - 1])];
        svg("circle", { cx: x, cy: y, r: 6.5, class: "search-halo" }, parent);
        svg("circle", { cx: x, cy: y, r: 4.5, style: "fill:var(--highlight);stroke:var(--surface-strong);stroke-width:1.5" }, parent);
      }
    });
  }

  const strong = (content) => node("strong", "", content);
  if (state.search.size === 1) {
    const [index] = state.search;
    note.append(strong(presets[index].name), ": ");
    if (!pickable(index)) {
      note.append("held out (n = 7), so no strategy ever picks it.");
      return;
    }
    const phrases = orders.map(({ label }, position) => (joins[position].length ? `${label} adds it at k = ${joins[position][0]}` : `${label} doesn’t add it by k = ${series.limit}`));
    note.append(`${phrases.join("; ")}.`);
    if (joins.some((ks) => ks.length)) note.append(" The pink dots mark those points.");
    return;
  }
  const inPool = [...state.search].filter(pickable).length;
  const summary = orders.map(({ label }, position) => `${joins[position].length} join${joins[position].length === 1 ? "s" : ""} ${label} by k = ${series.limit}`).join(", ");
  note.append(strong(`${state.search.size} matches`), ` (${inPool} pickable): ${summary}.`);
  note.append(state.search.size <= RING_LIMIT ? " The pink dots mark where each one is added." : ` Narrow the search to ${RING_LIMIT} or fewer to mark them.`);
}

function drawEndLabels(parent, lines, x, sy, top, bottom) {
  const labels = lines.map((line) => ({ line, anchor: sy(line.values.at(-1)) })).sort((a, b) => a.anchor - b.anchor);
  const spacing = 15;
  labels.forEach((label, position) => {
    label.y = Math.max(label.anchor, position ? labels[position - 1].y + spacing : top + 4);
  });
  const overflow = labels.at(-1).y - (bottom - 2);
  if (overflow > 0) labels.forEach((label) => (label.y -= overflow));
  for (const { line, anchor, y } of labels) {
    svg("path", { d: `M${x + 3},${anchor}C${x + 12},${anchor} ${x + 8},${y} ${x + 16},${y}`, style: "fill:none;stroke:var(--faint);stroke-width:1" }, parent);
    svg("line", { x1: x + 19, x2: x + 33, y1: y, y2: y, style: `stroke:${line.color};stroke-width:2${line.dash ? ";stroke-dasharray:4 3" : ""}` }, parent);
    text(parent, x + 38, y + 4, line.label, { class: "strong", "font-size": 11 });
  }
}

// ---------- table ----------

const TABLE_COLUMNS = [
  ["name", "Robot"], ["family", "Family"], ["nodes", "Nodes"], ["members", "Members"], ["actuators", "Actuators"],
  ["shape_dof", "Shape DOF"], ["degree_std", "Degree spread"], ["triangles_per_node", "Triangles / node"],
  ["symmetry", "Symmetry"], ["wcri", "WCRI"], ["resting_faces", "Resting faces"], ["aspect_ratio", "Aspect"],
];
let tableSort = { key: "nodes", ascending: true };

function renderTable() {
  const table = document.getElementById("table");
  const head = node("thead");
  const headRow = node("tr");
  for (const [key, label] of TABLE_COLUMNS) {
    const cell = node("th", "", label + (tableSort.key === key ? (tableSort.ascending ? " ↑" : " ↓") : ""));
    cell.addEventListener("click", () => {
      tableSort = { key, ascending: tableSort.key === key ? !tableSort.ascending : true };
      renderTable();
    });
    headRow.append(cell);
  }
  head.append(headRow);
  const body = node("tbody");
  const shown = presets.map((_, index) => index).filter((index) => !state.search || state.search.has(index));
  const sorted = shown.sort((a, b) => {
    const [x, y] = [presets[a][tableSort.key], presets[b][tableSort.key]];
    const order = typeof x === "string" ? x.localeCompare(y) : x - y;
    return (tableSort.ascending ? order : -order) || presets[a].name.localeCompare(presets[b].name);
  });
  for (const index of sorted) {
    const tr = node("tr");
    for (const [key] of TABLE_COLUMNS) {
      const value = key === "members" ? memberLabel(presets[index]) : presets[index][key];
      tr.append(node("td", "", typeof value === "number" ? format(value) : value));
    }
    body.append(tr);
  }
  table.replaceChildren(head, body);
  const summary = document.getElementById("table-summary");
  if (!state.search) summary.textContent = "All robots as a table";
  else summary.textContent = `${sorted.length} matching robot${sorted.length === 1 ? "" : "s"} as a table`;
}

// ---------- orchestration ----------

const charts = { occupancy: occupancyChart, map: mapChart, matrix: matrixChart, coverage: coverageChart };

function render(names = Object.keys(charts)) {
  for (const name of names) charts[name].render();
}

function emphasize() {
  const active = activeSet();
  mapChart.emphasize(active);
  matrixChart.emphasize(active);
}

function picksChanged() {
  savePicks();
  renderPickControls();
  render(["map", "coverage"]);
  emphasize();
}

function bindSegmented(id, key, after) {
  const group = document.getElementById(id);
  group.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-value]");
    if (!button) return;
    state[key] = button.dataset.value;
    for (const option of group.querySelectorAll("button")) option.setAttribute("aria-pressed", String(option === button));
    after();
  });
}

document.getElementById("color-by").addEventListener("change", (event) => {
  state.colorBy = event.target.value;
  renderLegend();
  render(["map", "matrix"]);
  emphasize();
});
bindSegmented("basis", "basis", () => {
  render(["map", "coverage"]);
  emphasize();
});
bindSegmented("metric", "metric", () => render(["coverage"]));
bindSegmented("target", "target", () => render(["coverage"]));
bindSegmented("weighting", "weighting", () => render(["coverage"]));

const kInput = document.getElementById("k");
kInput.max = String(Math.min(MAX_TRAINING_SIZE, pool.length));
kInput.addEventListener("input", () => {
  state.k = Number(kInput.value);
  document.getElementById("k-value").textContent = kInput.value;
  render(["coverage"]);
});
document.getElementById("load-picks").addEventListener("click", () => {
  state.picks = new Set(cachedOrder("sfps").slice(0, state.k));
  picksChanged();
});
document.getElementById("clear-picks").addEventListener("click", () => {
  state.picks = new Set();
  picksChanged();
});
document.getElementById("add-brushed").addEventListener("click", () => {
  for (const index of brushedVisible()) if (pickable(index)) state.picks.add(index);
  picksChanged();
});
document.getElementById("copy-yaml").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const yaml = splitYaml(presets, state.picks, data.split.final_test);
  try {
    await navigator.clipboard.writeText(yaml);
    button.textContent = "Copied";
  } catch {
    const area = document.createElement("textarea");
    area.value = yaml;
    document.body.append(area);
    area.select();
    button.textContent = document.execCommand("copy") ? "Copied" : "Copy failed";
    area.remove();
  }
  setTimeout(() => (button.textContent = "Copy YAML"), 1600);
});
const searchInput = document.getElementById("search");
const searchStatus = document.getElementById("search-status");
document.getElementById("robot-names").append(
  ...presets.map((preset) => preset.name).sort().map((name) => Object.assign(document.createElement("option"), { value: name })),
);

function searchChanged() {
  const query = searchInput.value;
  const matches = searchPresets(presets, query);
  // An empty set (no match) dims everything, rather than silently showing every robot.
  state.search = query.trim() ? new Set(matches) : null;
  searchStatus.textContent = !query.trim() ? "" : matches.length ? `${matches.length} match${matches.length === 1 ? "" : "es"}` : "No match";
  searchStatus.classList.toggle("empty", Boolean(query.trim()) && !matches.length);
  render(["occupancy", "coverage"]);
  renderTable();
  renderPickControls();
  emphasize();
}

searchInput.addEventListener("input", searchChanged);
searchInput.addEventListener("keydown", (event) => {
  if (event.key !== "Escape" || !searchInput.value) return;
  // Esc in a non-empty search clears just the search, not the brush or cell filter.
  event.preventDefault();
  event.stopPropagation();
  searchInput.value = "";
  searchChanged();
});
document.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  state.brushed = null;
  state.cellFilter = null;
  render(["occupancy"]);
  emphasize();
  renderPickControls();
});

const widths = new Map();
const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const name = entry.target.id;
    const width = Math.round(entry.contentRect.width);
    if (widths.get(name) === width) continue;
    const first = !widths.has(name);
    widths.set(name, width);
    if (!first) {
      charts[name].render();
      emphasize();
    }
  }
});
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
  renderLegend();
  render();
  emphasize();
});

renderLegend();
renderPickControls();
render();
renderTable();
for (const chart of Object.values(charts)) resizeObserver.observe(chart.element);
if (searchInput.value) searchChanged();
