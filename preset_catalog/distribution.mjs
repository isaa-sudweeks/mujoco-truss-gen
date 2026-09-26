// Pure logic for the training-distribution page: no DOM, so Node tests can import it.

export const MAX_TRAINING_SIZE = 64;
export const RANDOM_DRAWS = 200;
export const HELD_OUT_NODE_COUNT = 7;

/** Decode a base64 little-endian uint16 upper triangle into a full n x n distance matrix. */
export function decodeDistances(base64, count) {
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  const view = new DataView(bytes.buffer);
  const expected = (count * (count - 1)) / 2;
  if (view.byteLength !== expected * 2) {
    throw new Error(`Expected ${expected} distances for ${count} robots, got ${view.byteLength / 2}.`);
  }
  const matrix = new Float32Array(count * count);
  let offset = 0;
  for (let row = 0; row < count; row += 1) {
    for (let col = row + 1; col < count; col += 1) {
      const value = view.getUint16(offset, true) / 65535;
      offset += 2;
      matrix[row * count + col] = value;
      matrix[col * count + row] = value;
    }
  }
  return matrix;
}

/** Deterministic PRNG (mulberry32) returning floats in [0, 1). */
export function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffled(items, random) {
  const copy = items.slice();
  for (let index = copy.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1));
    [copy[index], copy[other]] = [copy[other], copy[index]];
  }
  return copy;
}

function groupBy(items, keyOf) {
  const groups = new Map();
  for (const item of items) {
    const key = keyOf(item);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  return [...groups.entries()].sort(([a], [b]) => a - b).map(([, members]) => members);
}

/** Take one item from each group in turn until `limit` items are taken or all run out. */
function roundRobin(groups, limit, take) {
  const order = [];
  const queues = groups.map((group) => group.slice());
  while (order.length < limit && queues.some((queue) => queue.length)) {
    for (const queue of queues) {
      if (!queue.length || order.length >= limit) continue;
      order.push(take(queue, order));
    }
  }
  return order;
}

export function randomOrder(pool, limit, random) {
  return shuffled(pool, random).slice(0, limit);
}

export function stratifiedRandomOrder(pool, strataOf, limit, random) {
  // Shuffle the strata too, so a draw that stops partway through a round is not biased to small n.
  const groups = shuffled(groupBy(pool, strataOf), random).map((group) => shuffled(group, random));
  return roundRobin(groups, limit, (queue) => queue.shift());
}

function medoid(matrix, count, candidates) {
  let best = candidates[0];
  let bestTotal = Infinity;
  for (const candidate of candidates) {
    let total = 0;
    for (const other of candidates) total += matrix[candidate * count + other];
    if (total < bestTotal) {
      bestTotal = total;
      best = candidate;
    }
  }
  return best;
}

function farthestFrom(matrix, count, candidates, selected) {
  let best = candidates[0];
  let bestDistance = -1;
  for (const candidate of candidates) {
    let nearest = Infinity;
    for (const chosen of selected) nearest = Math.min(nearest, matrix[candidate * count + chosen]);
    if (nearest > bestDistance) {
      bestDistance = nearest;
      best = candidate;
    }
  }
  return best;
}

/** Greedy farthest-point (k-center) order, starting at the pool medoid. */
export function farthestPointOrder(matrix, count, pool, limit) {
  const order = [];
  const remaining = pool.slice();
  while (order.length < limit && remaining.length) {
    const next = order.length ? farthestFrom(matrix, count, remaining, order) : medoid(matrix, count, remaining);
    order.push(next);
    remaining.splice(remaining.indexOf(next), 1);
  }
  return order;
}

/** Farthest-point order that visits node-count strata in turn (smallest size first). */
export function stratifiedFarthestPointOrder(matrix, count, pool, strataOf, limit) {
  return roundRobin(groupBy(pool, strataOf), limit, (queue, order) => {
    const next = order.length ? farthestFrom(matrix, count, queue, order) : medoid(matrix, count, queue);
    queue.splice(queue.indexOf(next), 1);
    return next;
  });
}

/**
 * Coverage after each prefix of `order`: worst-case and weighted-mean distance from every
 * target to its nearest selected robot. Entry k - 1 describes the first k picks.
 */
export function coverageCurve(matrix, count, order, targets, weights) {
  const nearest = new Float64Array(targets.length).fill(Infinity);
  const totalWeight = targets.reduce((sum, target) => sum + weights[target], 0);
  const worst = [];
  const mean = [];
  for (const pick of order) {
    let maximum = 0;
    let weighted = 0;
    targets.forEach((target, position) => {
      nearest[position] = Math.min(nearest[position], matrix[target * count + pick]);
      maximum = Math.max(maximum, nearest[position]);
      weighted += weights[target] * nearest[position];
    });
    worst.push(maximum);
    mean.push(weighted / totalWeight);
  }
  return { worst, mean };
}

export function coverageOf(matrix, count, selected, targets, weights) {
  if (!selected.length) return null;
  const curve = coverageCurve(matrix, count, selected, targets, weights);
  return { worst: curve.worst.at(-1), mean: curve.mean.at(-1) };
}

/** Per-target weights: 1 each, or 1 / (targets sharing its node count). */
export function targetWeights(presets, targets, perNodeCount) {
  const weights = new Float64Array(presets.length).fill(0);
  const sizes = new Map();
  for (const target of targets) sizes.set(presets[target].nodes, (sizes.get(presets[target].nodes) ?? 0) + 1);
  for (const target of targets) weights[target] = perNodeCount ? 1 / sizes.get(presets[target].nodes) : 1;
  return weights;
}

export function quantile(sortedValues, fraction) {
  const position = (sortedValues.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  return sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * (position - lower);
}

/** Median and 10-90% band across seeded random draws of one strategy. */
export function randomBand(matrix, count, makeOrder, targets, weights, metric, draws, seed) {
  const random = seededRandom(seed);
  const curves = [];
  for (let draw = 0; draw < draws; draw += 1) {
    curves.push(coverageCurve(matrix, count, makeOrder(random), targets, weights)[metric]);
  }
  const length = Math.min(...curves.map((curve) => curve.length));
  const low = [];
  const median = [];
  const high = [];
  for (let k = 0; k < length; k += 1) {
    const values = curves.map((curve) => curve[k]).sort((a, b) => a - b);
    low.push(quantile(values, 0.1));
    median.push(quantile(values, 0.5));
    high.push(quantile(values, 0.9));
  }
  return { low, median, high };
}

/** Spearman rank correlation with average ranks for ties. */
export function spearman(xs, ys) {
  if (xs.length < 3) return NaN;
  const rx = ranks(xs);
  const ry = ranks(ys);
  const meanX = rx.reduce((a, b) => a + b, 0) / rx.length;
  const meanY = ry.reduce((a, b) => a + b, 0) / ry.length;
  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  for (let index = 0; index < rx.length; index += 1) {
    covariance += (rx[index] - meanX) * (ry[index] - meanY);
    varianceX += (rx[index] - meanX) ** 2;
    varianceY += (ry[index] - meanY) ** 2;
  }
  return varianceX && varianceY ? covariance / Math.sqrt(varianceX * varianceY) : NaN;
}

function ranks(values) {
  const order = values.map((value, index) => [value, index]).sort((a, b) => a[0] - b[0]);
  const result = new Array(values.length);
  for (let start = 0; start < order.length; ) {
    let end = start;
    while (end + 1 < order.length && order[end + 1][0] === order[start][0]) end += 1;
    const rank = (start + end) / 2;
    for (let position = start; position <= end; position += 1) result[order[position][1]] = rank;
    start = end + 1;
  }
  return result;
}

/** GNN-SAC cross_validation YAML: picks grouped by node count, held-out robots as final_test. */
export function splitYaml(presets, picks, finalTest, name = "custom_picks") {
  const groups = new Map();
  for (const index of [...picks].sort((a, b) => presets[a].nodes - presets[b].nodes || presets[a].name.localeCompare(presets[b].name))) {
    const key = `node_${presets[index].nodes}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(presets[index].name);
  }
  const lines = ["# @package _global_", "", "cross_validation:", "  enabled: true", `  name: ${name}`, "  groups:"];
  for (const [key, names] of groups) {
    lines.push(`    ${key}:`);
    for (const presetName of names) lines.push(`      - ${presetName}`);
  }
  lines.push("  final_test:");
  for (const presetName of finalTest) lines.push(`    - ${presetName}`);
  lines.push("  held_out_group: null", "");
  return lines.join("\n");
}

/** Words a preset can be found by: its name, family, and member type. */
function searchText(preset) {
  return [preset.name, preset.family, preset.member_type].filter(Boolean).join(" ").toLowerCase();
}

/**
 * Indices of presets matching every whitespace-separated token of `query`. A token like "n8"
 * matches that exact node count; any other token matches a substring of the name, family, or
 * member type. A query that is exactly one preset's name matches only that preset.
 */
export function searchPresets(presets, query) {
  const normalized = query.trim().toLowerCase();
  if (!normalized) return [];
  const exact = presets.findIndex((preset) => preset.name.toLowerCase() === normalized);
  if (exact !== -1) return [exact];
  const tests = normalized.split(/\s+/).map((token) => {
    const nodes = token.match(/^n(\d+)$/);
    return nodes ? (preset) => preset.nodes === Number(nodes[1]) : (preset, haystack) => haystack.includes(token);
  });
  return presets.flatMap((preset, index) => {
    const haystack = searchText(preset);
    return tests.every((test) => test(preset, haystack)) ? [index] : [];
  });
}
