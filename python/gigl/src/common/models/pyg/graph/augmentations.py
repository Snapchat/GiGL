import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, dropout_edge, remove_self_loops


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """GRACE feature dropping function with probability drop_prob.
    From: https://github.com/CRIPAC-DIG/GRACE/blob/51b44961b68b2f38c60f85cf83db13bed8fd0780/model.py#L120
    """
    if drop_prob == 0:
        return x
    elif drop_prob < 0 or drop_prob > 1:
        raise ValueError(f"Invalid probability provided for Feat Drop, got {drop_prob}")
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def get_augmented_graph(
    graph: Data,
    edge_drop_ratio: float = 0.3,
    feat_drop_ratio: float = 0.3,
    graph_perm: bool = False,
) -> Data:
    """
    PyG implementation of DGL transformations. Supports augmentations such as dropping random edges (edge_drop_ratio), dropping random feature components (feat_drop_ratio),
    and graph permutation (shuffling the nodes and edges of the graph randomly)
    https://docs.dgl.ai/en/0.9.x/api/python/transforms.html
    """
    if edge_drop_ratio < 0 or edge_drop_ratio > 1:
        raise ValueError(
            f"Invalid probability provided for Edge Drop, got {edge_drop_ratio}"
        )
    if feat_drop_ratio < 0 or feat_drop_ratio > 1:
        raise ValueError(
            f"Invalid probability provided for Feat Drop, got {feat_drop_ratio}"
        )
    data = graph.clone()
    if graph_perm:
        row_perm = torch.randperm(data.x.size(0))
        data.x = data.x[row_perm, :]
        data.edge_index = torch.randint_like(data.edge_index, data.num_nodes - 1)
    _, edge_mask = dropout_edge(data.edge_index, p=edge_drop_ratio, training=True)

    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[edge_mask]

    data.edge_index = data.edge_index[:, edge_mask]
    data.edge_index, _ = remove_self_loops(data.edge_index)
    data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
    data.x = drop_feature(data.x, feat_drop_ratio)
    return data
