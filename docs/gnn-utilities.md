# GNN Utilities

The package can extract node features and edge indices directly from a generated
spec or model. The returned arrays are formatted for use with PyTorch Geometric
`torch_geometric.data.Data` objects.

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
