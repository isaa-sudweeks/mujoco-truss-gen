"""Render every registered truss preset and build a rigidity-ranked HTML catalog.

Run from the repository root with::

    UV_CACHE_DIR=/tmp/mujoco-truss-gen-uv-cache uv run python tools/render_preset_catalog.py
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import html
import json
import re
import struct
import zlib
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mujoco_truss_gen import PRESETS, get_mujoco_spec, get_preset_definition
from mujoco_truss_gen.mujoco_model.model_types import NodeDict, ShapeDict, TriangleDict
from mujoco_truss_gen.mujoco_model.presets import _worst_case_rigidity_index

HENNEBERG_NAME = re.compile(r"^henneberg_n(?P<nodes>\d+)_(?P<tubes>\d+)tube(?:_(?P<index>\d+))?$")
USEVITCH_NAME = re.compile(r"^usevitch_(?P<label>\d+)(?:_p(?P<partition>\d+))?$")


def _is_henneberg_alias(name: str) -> bool:
    match = HENNEBERG_NAME.fullmatch(name)
    return match is not None and match.group("index") is None


def _node_sort_key(name: str) -> tuple[int, str]:
    suffix = name.removeprefix("node_")
    return (int(suffix), name) if suffix.isdigit() else (10**9, name)


def _structural_edges(structure: dict[str, Any]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for item in structure.values():
        if isinstance(item, dict) and "route" in item:
            nodes = item["route"]
            pairs = zip(nodes, nodes[1:], strict=False)
        else:
            nodes = item[:3]
            pairs = ((nodes[index], nodes[(index + 1) % 3]) for index in range(3))
        edges.update(tuple(sorted(pair)) for pair in pairs if pair[0] != pair[1])
    return edges


def _initial_wcri(nodes: NodeDict, structure: TriangleDict | ShapeDict) -> tuple[float, int, int]:
    node_names = sorted(nodes, key=_node_sort_key)
    node_index = {node_name: index for index, node_name in enumerate(node_names)}
    edges_by_name = _structural_edges(structure)
    edges = tuple(
        sorted((node_index[first], node_index[second]) for first, second in edges_by_name)
    )
    coordinates = np.asarray([nodes[node_name] for node_name in node_names], dtype=float)
    return _worst_case_rigidity_index(coordinates, edges), len(node_names), len(edges)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 image using only the Python standard library."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}.")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in image)

    def chunk(kind: bytes, data: bytes) -> bytes:
        payload = kind + data
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload))

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, level=6))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def _metadata(name: str) -> dict[str, Any]:
    henneberg = HENNEBERG_NAME.fullmatch(name)
    if henneberg:
        return {
            "family": "Henneberg",
            "tube_count": int(henneberg.group("tubes")),
            "variant": int(henneberg.group("index") or 1),
        }
    usevitch = USEVITCH_NAME.fullmatch(name)
    if usevitch:
        return {
            "family": "Usevitch",
            "partition": int(usevitch.group("partition") or 1),
        }
    return {"family": "Built-in"}


def _render(
    nodes: NodeDict,
    structure: TriangleDict | ShapeDict,
    image_path: Path,
    width: int,
    height: int,
    realistic: bool,
) -> None:
    model = get_mujoco_spec(nodes, structure, realistic=realistic).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = 135.0
    camera.elevation = -22.5
    camera.distance = max(float(model.stat.extent) * 1.65, 0.5)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        _write_png(image_path, renderer.render())
    finally:
        renderer.close()


def _write_catalog(
    output_dir: Path, records: list[dict[str, Any]], arguments: dict[str, Any]
) -> None:
    (output_dir / ".nojekyll").touch()
    successful = [record for record in records if record["status"] == "ok"]
    successful.sort(key=lambda record: (-record["wcri"], record["name"]))
    for rank, record in enumerate(successful, start=1):
        record["rank"] = rank
    failures = [record for record in records if record["status"] != "ok"]
    failures.sort(key=lambda record: record["name"])
    records[:] = successful + failures

    manifest = {"arguments": arguments, "presets": records}
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    cards = []
    for record in records:
        name = html.escape(record["name"])
        if record["status"] == "ok":
            details = [
                f'<span class="rank">#{record["rank"]}</span>',
                f"WCRI <code>{record['wcri']:.8g}</code>",
                f"{record['node_count']} nodes",
                f"{record['edge_count']} edges",
            ]
            if "tube_count" in record:
                details.append(f"{record['tube_count']} tube(s)")
            cards.append(
                f'<article class="card" data-name="{name.lower()}" '
                f'data-family="{record["family"].lower()}" data-rank="{record["rank"]}">'
                f'<a href="{html.escape(record["image"])}"><img loading="lazy" '
                f'src="{html.escape(record["image"])}" alt="Rendered {name}"></a>'
                f'<div class="body"><h2>{name}</h2><p>{" · ".join(details)}</p>'
                f'<p class="family">{record["family"]}</p></div></article>'
            )
        else:
            cards.append(
                f'<article class="card failed" data-name="{name.lower()}" data-family="failed">'
                f'<div class="body"><h2>{name}</h2><p>Render failed</p>'
                f"<pre>{html.escape(record['error'])}</pre></div></article>"
            )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MuJoCo truss preset catalog</title>
<style>
:root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; background:#111318; color:#eef1f6 }}
body {{ margin:0; padding:2rem; }} header {{ max-width:1100px; margin:0 auto 1.5rem; }}
h1 {{ margin-bottom:.35rem }} .summary {{ color:#aeb7c6 }} .controls {{ display:flex; gap:.75rem; flex-wrap:wrap; margin-top:1rem }}
input,select {{ padding:.7rem .85rem; background:#20242d; color:inherit; border:1px solid #3b4250; border-radius:.5rem }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:1rem; max-width:1600px; margin:auto }}
.card {{ overflow:hidden; background:#1a1e26; border:1px solid #303744; border-radius:.75rem }}
.card img {{ display:block; width:100%; aspect-ratio:{arguments["width"]}/{arguments["height"]}; object-fit:cover; background:#000 }}
.body {{ padding:.85rem 1rem }} h2 {{ font-size:.95rem; overflow-wrap:anywhere; margin:0 0 .55rem }}
p {{ margin:.35rem 0; color:#c1c9d6; font-size:.88rem }} .rank {{ color:#72d6a1; font-weight:700 }}
.family {{ color:#8da2c4 }} code {{ color:#f4c36b }} .failed {{ border-color:#8f3e48 }} pre {{ white-space:pre-wrap; color:#ff9da8 }}
</style></head><body><header><h1>MuJoCo truss preset catalog</h1>
<div class="summary">{len(successful)} presets ranked by initial worst-case rigidity index; {len(failures)} failures.</div>
<div class="controls"><input id="search" type="search" placeholder="Search preset name">
<select id="family"><option value="">All families</option><option>Built-in</option><option>Usevitch</option><option>Henneberg</option><option>Failed</option></select></div>
</header><main class="grid">{"".join(cards)}</main>
<script>
const search=document.querySelector('#search'), family=document.querySelector('#family');
function filter(){{const q=search.value.toLowerCase(),f=family.value.toLowerCase();document.querySelectorAll('.card').forEach(c=>c.hidden=!(c.dataset.name.includes(q)&&(!f||c.dataset.family===f)));}}
search.addEventListener('input',filter);family.addEventListener('change',filter);
</script></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def _write_progress(
    output_dir: Path, records: list[dict[str, Any]], arguments: dict[str, Any]
) -> None:
    progress = {"arguments": arguments, "presets": records}
    (output_dir / "progress.json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_progress(output_dir: Path, arguments: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path = output_dir / "progress.json"
    if not path.exists():
        return {}
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("arguments") != arguments:
        return {}
    return {record["name"]: record for record in progress.get("presets", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("preset_catalog"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--realistic", action="store_true")
    parser.add_argument("--include-aliases", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Render only the first N names (for smoke tests)."
    )
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.scale <= 0.0:
        parser.error("width, height, and scale must be positive")

    output_dir = args.output.resolve()
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    arguments = {
        "width": args.width,
        "height": args.height,
        "scale": args.scale,
        "realistic": args.realistic,
        "include_aliases": args.include_aliases,
    }
    names = sorted(PRESETS)
    if not args.include_aliases:
        names = [name for name in names if not _is_henneberg_alias(name)]
    if args.limit is not None:
        names = names[: args.limit]

    cached_records = {} if args.overwrite else _load_progress(output_dir, arguments)
    records: list[dict[str, Any]] = []
    interrupted = False
    try:
        for position, name in enumerate(names, start=1):
            print(f"[{position}/{len(names)}] {name}", flush=True)
            image_path = image_dir / f"{name}.png"
            cached = cached_records.get(name)
            if cached is not None and cached.get("status") == "ok" and image_path.exists():
                records.append(cached)
                continue

            record = {"name": name, **_metadata(name)}
            try:
                nodes, structure = get_preset_definition(name, scale=args.scale)
                wcri, node_count, edge_count = _initial_wcri(nodes, structure)
                record.update(wcri=wcri, node_count=node_count, edge_count=edge_count)
                if args.overwrite or not image_path.exists():
                    _render(
                        nodes,
                        structure,
                        image_path,
                        args.width,
                        args.height,
                        args.realistic,
                    )
                record.update(
                    status="ok",
                    image=image_path.relative_to(output_dir).as_posix(),
                )
            except Exception as exc:  # Continue so one bad preset does not lose the catalog.
                record.update(status="error", error=f"{type(exc).__name__}: {exc}")
            records.append(record)
            _write_progress(output_dir, records, arguments)
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted; writing a partial ranked catalog before exiting.", flush=True)
    finally:
        _write_progress(output_dir, records, arguments)
        _write_catalog(output_dir, records.copy(), arguments)

    failures = sum(record["status"] != "ok" for record in records)
    print(
        f"Wrote {output_dir / 'index.html'} ({len(records) - failures} rendered, {failures} failed)"
    )
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
