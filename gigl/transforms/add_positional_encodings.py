from typing import Dict, Optional

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_torch_sparse_tensor


def add_node_attr(
    data: HeteroData,
    values: Tensor,
    attr_name: Optional[str] = None,
    node_type_to_idx: Optional[Dict[str, tuple]] = None,
) -> HeteroData:
    """Helper function to add node attributes to a HeteroData object.

    Args:
        data: The HeteroData object to modify.
        values: The tensor of values (in homogeneous node order).
        attr_name: The name of the attribute to add. If None, concatenates to
            existing `x` attribute for each node type (or creates it).
        node_type_to_idx: Optional mapping from node type to (start, end) indices.
            If None, it will be computed from data.node_types.

    Returns:
        The modified HeteroData object.
    """
    if node_type_to_idx is None:
        node_type_to_idx = {}
        start_idx = 0
        for node_type in data.node_types:
            num_type_nodes = data[node_type].num_nodes
            node_type_to_idx[node_type] = (start_idx, start_idx + num_type_nodes)
            start_idx += num_type_nodes

    for node_type in data.node_types:
        start, end = node_type_to_idx[node_type]
        value = values[start:end]

        if attr_name is None:
            # Concatenate to existing x or create new x
            x = data[node_type].x
            if x is not None:
                x = x.view(-1, 1) if x.dim() == 1 else x
                data[node_type].x = torch.cat(
                    [x, value.to(x.device, x.dtype)], dim=-1
                )
            else:
                data[node_type].x = value
        else:
            data[node_type][attr_name] = value

    return data


@functional_transform('add_hetero_hop_distance_pe')
class AddHeteroHopDistancePE(BaseTransform):
    r"""Adds the hop distance positional encoding from the
    `"Graph Neural Networks with Learnable Structural and Positional
    Representations" <https://arxiv.org/abs/2110.07875>`_ paper to the given
    heterogeneous graph (functional name: :obj:`add_hetero_hop_distance_pe`).

    Args:
        k (int): The number of hops to consider.
        attr_name (str, optional): The attribute name of the positional
            encoding. (default: :obj:`"hop_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, the graph is
            assumed to be undirected, and multi-hop connectivity will be
            computed accordingly. (default: :obj:`False`)
    """
    def __init__(
        self,
        k: int,
        attr_name: Optional[str] = 'hop_pe',
        is_undirected: bool = False,
    ) -> None:
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected

    def forward(self, data: HeteroData) -> HeteroData:
        assert isinstance(data, HeteroData), (
            f"'{self.__class__.__name__}' only supports 'HeteroData' "
            f"(got '{type(data)}')"
        )

        # Convert to homogeneous to calculate subgraph hop distances
        homo_data = data.to_homogeneous()
        edge_index = homo_data.edge_index
        num_nodes = homo_data.num_nodes

        if num_nodes == 0:
            # Handle empty graph case
            for node_type in data.node_types:
                data[node_type][self.attr_name] = torch.zeros(
                    (data[node_type].num_nodes, self.k),
                    dtype=torch.float,
                )
            return data

        # Compute Adjacency Matrix (sparse for efficiency)
        adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes))

        if self.is_undirected:
            # Make symmetric for undirected graphs
            adj = adj + adj.t()
            adj = adj.coalesce()

        # Convert to dense for matrix power computation
        adj_dense = adj.to_dense()
        adj_dense = (adj_dense > 0).float()  # Binary adjacency

        # Iteratively compute k-hop reachability
        hop_pe = torch.zeros((num_nodes, self.k), dtype=torch.float)
        current_power = torch.eye(num_nodes, dtype=torch.float)

        for i in range(self.k):
            current_power = current_power @ adj_dense
            # Count number of paths at exactly i+1 hops (normalized)
            reachable = (current_power > 0).float()
            hop_pe[:, i] = reachable.sum(dim=-1)

        # Normalize hop distances (optional, for numerical stability)
        hop_pe = hop_pe / (hop_pe.max(dim=0, keepdim=True).values + 1e-8)

        # Map back to HeteroData node types
        add_node_attr(data, hop_pe, self.attr_name)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'


@functional_transform('add_hetero_random_walk_pe')
class AddHeteroRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding from the
    `"Graph Neural Networks with Learnable Structural and Positional
    Representations" <https://arxiv.org/abs/2110.07875>`_ paper to the given
    heterogeneous graph (functional name: :obj:`add_hetero_random_walk_pe`).

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the positional
            encoding. (default: :obj:`"random_walk_pe"`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_pe',
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: HeteroData) -> HeteroData:
        assert isinstance(data, HeteroData), (
            f"'{self.__class__.__name__}' only supports 'HeteroData' "
            f"(got '{type(data)}')"
        )

        # Convert to homogeneous
        homo_data = data.to_homogeneous()
        edge_index = homo_data.edge_index
        num_nodes = homo_data.num_nodes

        if num_nodes == 0:
            for node_type in data.node_types:
                data[node_type][self.attr_name] = torch.zeros(
                    (data[node_type].num_nodes, self.walk_length),
                    dtype=torch.float,
                )
            return data

        # Compute transition matrix (row-stochastic)
        adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes))
        adj_dense = adj.to_dense()

        # Compute degree and create transition matrix
        deg = adj_dense.sum(dim=1, keepdim=True)
        deg = torch.clamp(deg, min=1)  # Avoid division by zero
        transition = adj_dense / deg

        # Compute random walk probabilities
        pe = torch.zeros((num_nodes, self.walk_length), dtype=torch.float)
        current = torch.eye(num_nodes, dtype=torch.float)

        for i in range(self.walk_length):
            current = current @ transition
            # Diagonal gives probability of returning to the same node
            pe[:, i] = current.diag()

        # Map back to HeteroData node types
        add_node_attr(data, pe, self.attr_name)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(walk_length={self.walk_length})'
