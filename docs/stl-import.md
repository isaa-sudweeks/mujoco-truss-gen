# STL Import

STL mesh import is available through the optional `mesh` dependency group:

```bash
python -m pip install "mujoco-truss-gen[mesh]"
```

The importer treats the STL as a routing graph. Mesh faces are used only to
derive deduplicated graph edges; each generated route becomes one continuous
tube in the existing `shape_dict` format.

```python
from mujoco_truss_gen import get_mujoco_spec, stl_to_shape_dict

node_dict, shape_dict = stl_to_shape_dict(
    "part.stl",
    merge_tolerance=1e-6,
    target_edge_length=0.05,
    preview=True,
)
spec = get_mujoco_spec(node_dict, shape_dict, realistic=False)
model = spec.compile()
```

Coordinates are interpreted as MuJoCo units by default. Use `scale` and
`offset` to convert STL units or move the imported graph. Pass either
`target_edge_length` or `target_node_count` to simplify the graph after nearby
vertices are merged.

The importer prints progress by default, including mesh size, graph size, route
count, and elapsed time for each stage. Pass `verbose=False` to silence that
output.

Pass `preview=True` to open a matplotlib preview before converting the routed
graph to MuJoCo. The preview shows the merged STL graph next to the simplified
routed paths and reports node, edge, and path counts so you can check how much
simplification was applied. Preview windows are disabled by default; pass
`preview=False` explicitly in scripts where you want to guarantee headless
operation. For non-blocking previews, pass `preview_block=False`.

On macOS, the preview is launched in a separate Python process so matplotlib can
own its GUI window on the main thread, including when the caller is running under
`mjpython`.

## Stability

The STL-to-MuJoCo path is experimental. Imported routed graphs may produce
unstable MuJoCo configurations, so generated models should be checked before
they are used for training, planning, or optimization runs.
