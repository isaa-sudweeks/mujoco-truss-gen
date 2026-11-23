"""Export the catalog's rendered presets as standalone MJCF for MuJoCo Live."""

import argparse
import json
from pathlib import Path

from mujoco_truss_gen import get_mujoco_spec, get_preset_definition, save_xml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog_dir", type=Path)
    catalog_dir = parser.parse_args().catalog_dir
    manifest = json.loads((catalog_dir / "manifest.json").read_text(encoding="utf-8"))
    arguments = manifest["arguments"]
    count = 0
    for record in manifest["presets"]:
        if record["status"] != "ok":
            continue
        name = record["name"]
        nodes, structure = get_preset_definition(name, scale=arguments["scale"])
        spec = get_mujoco_spec(nodes, structure, realistic=arguments["realistic"])
        if spec.assets:
            raise ValueError(f"{name} requires additional assets for MuJoCo Live")
        save_xml(spec, catalog_dir / "models" / f"{name}.xml")
        count += 1
        if count % 25 == 0:
            print(f"Exported {count} catalog models", flush=True)
    print(f"Exported {count} catalog models")


if __name__ == "__main__":
    main()
