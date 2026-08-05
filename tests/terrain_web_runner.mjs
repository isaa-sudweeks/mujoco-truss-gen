import { generateTerrain } from "../preset_catalog/terrain.mjs";

let input = "";
for await (const chunk of process.stdin) input += chunk;
const configurations = JSON.parse(input);
const results = configurations.map((config) => {
  const terrain = generateTerrain(config);
  return {
    heights: Array.from(terrain.heights),
    elevationMin: terrain.elevationMin,
    elevationMax: terrain.elevationMax,
    maxGeneratedSlope: terrain.maxGeneratedSlope,
  };
});
process.stdout.write(JSON.stringify(results));
