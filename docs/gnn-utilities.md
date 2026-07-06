# GNN Utilities

The package can extract node features and edge indices directly from a generated
spec or model. The returned arrays are formatted for use with PyTorch Geometric
`torch_geometric.data.Data` objects.

Install the optional graph dependencies before using the NetworkX and Matplotlib
helpers:

```bash
python -m pip install "mujoco-truss-gen[graph]"
```

```python
import torch

from mujoco_truss_gen import get_edge_index, get_mujoco_spec, get_node_features

spec = get_mujoco_spec("octahedron", realistic=False)

# PyG COO format: shape (2, num_directed_edges)
edge_index = get_edge_index(spec)

# Node positions and velocities: shape (num_nodes, 6)
node_features = get_node_features(spec)

edge_index_tensor = torch.from_numpy(edge_index)
x_tensor = torch.from_numpy(node_features)
```

For realistic models, pass `graph_view="logical"` to collapse cloned triangle
nodes back to the abstract logical graph:

```python
spec = get_mujoco_spec("octahedron", realistic=True)

edge_index = get_edge_index(spec, graph_view="logical")
node_features = get_node_features(
    spec,
    graph_view="logical",
    aggregation="connector_ball",  # or "mean"
)
```

Use `view_graph` to inspect the graph structure passed to a GNN:

```python
from mujoco_truss_gen import get_mujoco_spec, view_graph

spec = get_mujoco_spec("octahedron", realistic=True)

# Shows action/control nodes, actuated edges, and virtual connector edges.
fig, ax, graph = view_graph(spec, graph_view="control")

# Uses the current MuJoCo x/y/z node positions for a rotatable 3D view.
fig, ax, graph = view_graph(
    spec,
    graph_view="control",
    layout="physical",
    dim=3,
)
```

The 3D physical layout starts from the model's actual current pose, not a
separately computed graph embedding. Nodes that overlap in the default camera
view are spread slightly around their shared area so duplicate control nodes
remain visible. Set `overlap_offset=0` to disable this display-only separation;
the graph's `pos3d` data always retains the true coordinates. Force-directed
layouts also support 3D by setting `dim=3` with `layout="spring"` or another
supported layout.

For custom plotting, build the NetworkX graph directly:

```python
import networkx as nx
import matplotlib.pyplot as plt

from mujoco_truss_gen import get_mujoco_spec, get_networkx_graph

spec = get_mujoco_spec("tetrahedron", realistic=True)
graph = get_networkx_graph(spec, graph_view="control")

plt.figure(figsize=(8, 8))
nx.draw(graph, with_labels=True)
plt.show()
```

`graph_view="control"` preserves edge types in the NetworkX edge attribute
`type`: `actuated` edges map to tendon commands, while `connector` edges are
message-passing-only virtual edges between control nodes that share a connector.
