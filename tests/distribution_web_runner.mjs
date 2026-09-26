import {
  coverageCurve,
  decodeDistances,
  farthestPointOrder,
  randomBand,
  randomOrder,
  seededRandom,
  splitYaml,
  stratifiedFarthestPointOrder,
  stratifiedRandomOrder,
  targetWeights,
} from "../preset_catalog/distribution.mjs";

let input = "";
for await (const chunk of process.stdin) input += chunk;
const { distances, presets, finalTest } = JSON.parse(input);
const count = presets.length;
const matrix = decodeDistances(distances, count);
const everyone = presets.map((_, index) => index);
const pool = everyone.filter((index) => presets[index].nodes !== 7);
const strataOf = (index) => presets[index].nodes;
const limit = pool.length;
const weights = targetWeights(presets, everyone, false);

const first = seededRandom(3);
const second = seededRandom(3);
const fps = farthestPointOrder(matrix, count, pool, limit);
const sfps = stratifiedFarthestPointOrder(matrix, count, pool, strataOf, limit);
const draws = [];
const random = seededRandom(5);
for (let draw = 0; draw < 20; draw += 1) {
  draws.push(randomOrder(pool, limit, random), stratifiedRandomOrder(pool, strataOf, limit, random));
}

process.stdout.write(
  JSON.stringify({
    diagonal: everyone.map((index) => matrix[index * count + index]),
    symmetric: everyone.every((row) => everyone.every((col) => matrix[row * count + col] === matrix[col * count + row])),
    deterministic: Array.from({ length: 5 }, () => first() === second()).every(Boolean),
    fpsWorst: coverageCurve(matrix, count, fps, everyone, weights).worst,
    randomWorstMedian: randomBand(matrix, count, (rng) => randomOrder(pool, limit, rng), everyone, weights, "worst", 50, 1).median,
    orders: [fps, sfps, ...draws].map((order) => order.map((index) => presets[index].nodes)),
    sfpsFirst: sfps.slice(0, new Set(pool.map(strataOf)).size).map(strataOf),
    perSizeWeights: Array.from(targetWeights(presets, everyone, true)),
    yaml: splitYaml(presets, new Set(pool.slice(0, 3)), finalTest),
  }),
);
