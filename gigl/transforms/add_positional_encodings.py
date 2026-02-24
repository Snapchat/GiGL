from typing import Optional

import torch

from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_torch_sparse_tensor

from gigl.transforms.utils import add_node_attr, add_edge_attr

r"""
Positional and Structural Encodings for Heterogeneous Graphs.

This module provides PyG-compatible transforms for adding positional and structural
encodings to HeteroData objects. All transforms follow the PyG BaseTransform interface
and can be composed using `torch_geometric.transforms.Compose`.

Available Transforms:
    - AddHeteroRandomWalkPE: Random walk positional encoding (column sum of non-diagonal)
    - AddHeteroRandomWalkSE: Random walk structural encoding (diagonal elements)
    - AddHeteroHopDistanceEncoding: Shortest path distance encoding

Example Usage:
    >>> from torch_geometric.data import HeteroData
    >>> from torch_geometric.transforms import Compose
    >>> from gigl.transforms.add_positional_encodings import (
    ...     AddHeteroRandomWalkPE,
    ...     AddHeteroRandomWalkSE,
    ...     AddHeteroHopDistanceEncoding,
    ... )
    >>>
    >>> # Create a heterogeneous graph
    >>> data = HeteroData()
    >>> data['user'].x = torch.randn(5, 16)
    >>> data['item'].x = torch.randn(3, 16)
    >>> data['user', 'buys', 'item'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    >>> data['item', 'bought_by', 'user'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]])
    >>>
    >>> # Apply single transform
    >>> transform = AddHeteroRandomWalkPE(walk_length=8)
    >>> data = transform(data)
    >>> print(data['user'].random_walk_pe.shape)  # (5, 8)
    >>>
    >>> # Compose multiple transforms
    >>> transform = Compose([
    ...     AddHeteroRandomWalkPE(walk_length=8),
    ...     AddHeteroRandomWalkSE(walk_length=8),
    ...     AddHeteroHopDistanceEncoding(h_max=3, full_matrix=True),
    ... ])
    >>> data = transform(data)
    >>>
    >>> # For Graph Transformers, use full_matrix=True to get full pairwise distances
    >>> transform = AddHeteroHopDistanceEncoding(h_max=5, full_matrix=True)
    >>> data = transform(data)
    >>> print(data.hop_distance.shape)  # (num_total_nodes, num_total_nodes)
    >>>
    >>> # For heterogeneous Graph Transformers, use node_type_aware=True to preserve
    >>> # node type information for type-aware attention bias
    >>> transform = AddHeteroHopDistanceEncoding(h_max=5, full_matrix=True, node_type_aware=True)
    >>> data = transform(data)
    >>> print(data.hop_distance.shape)      # (8, 8) - pairwise distances
    >>> print(data.node_type_ids.shape)     # (8,) - type ID for each node
    >>> print(data.node_type_pair.shape)    # (8, 8) - encodes (src_type, dst_type) pairs
    >>> print(data.node_type_names)         # ['item', 'user'] - sorted alphabetically
    >>>
    >>> # In a Graph Transformer, combine hop distance and node type for attention bias:
    >>> # bias = hop_embedding[data.hop_distance.long()] + type_pair_embedding[data.node_type_pair]
"""


@functional_transform('add_hetero_random_walk_pe')
class AddHeteroRandomWalkPE(BaseTransform):
    r"""Adds the random walk positional encoding to the given heterogeneous graph
    (functional name: :obj:`add_hetero_random_walk_pe`).

    For each node j, computes the sum of transition probabilities from all other
    nodes to j after k steps of a random walk, for k = 1, 2, ..., walk_length.
    This captures how "reachable" or "central" a node is from the rest of the graph.

    The encoding is the column sum of non-diagonal elements of the k-step
    random walk matrix:
        PE[j, k] = Σ_{i≠j} (P^k)[i, j]

    where P is the transition matrix. This measures the probability mass flowing
    into node j from all other nodes at step k.

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the positional
            encoding. (default: :obj:`"random_walk_pe"`)
        is_undirected (bool, optional): If set to :obj:`True`, the graph is
            assumed to be undirected, and the adjacency matrix will be made
            symmetric. (default: :obj:`False`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_pe',
        is_undirected: bool = False,
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name
        self.is_undirected = is_undirected

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

        if self.is_undirected:
            # Make symmetric for undirected graphs
            adj_dense = adj_dense + adj_dense.t()

        # Compute degree and create transition matrix
        deg = adj_dense.sum(dim=1, keepdim=True)
        deg = torch.clamp(deg, min=1)  # Avoid division by zero
        transition = adj_dense / deg

        # Compute random walk positional encoding
        # PE[j, k] = sum of column j excluding diagonal = Σ_{i≠j} (P^k)[i, j]
        pe = torch.zeros((num_nodes, self.walk_length), dtype=torch.float)
        current = torch.eye(num_nodes, dtype=torch.float)

        for k in range(self.walk_length):
            current = current @ transition
            # Sum each column, excluding diagonal elements
            # column_sum[j] = Σ_i current[i, j]
            # diagonal[j] = current[j, j]
            # non_diagonal_column_sum[j] = column_sum[j] - diagonal[j]
            column_sum = current.sum(dim=0)  # Sum along rows for each column
            diagonal = current.diag()
            pe[:, k] = column_sum - diagonal

        # Map back to HeteroData node types
        add_node_attr(data, pe, self.attr_name)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(walk_length={self.walk_length})'


@functional_transform('add_hetero_random_walk_se')
class AddHeteroRandomWalkSE(BaseTransform):
    r"""Adds the random walk structural encoding from the
    `"Graph Neural Networks with Learnable Structural and Positional
    Representations" <https://arxiv.org/abs/2110.07875>`_ paper to the given
    heterogeneous graph (functional name: :obj:`add_hetero_random_walk_se`).

    For each node, computes the probability of returning to itself after k steps
    of a random walk, for k = 1, 2, ..., walk_length. This captures the local
    structural role of each node (e.g., cycles, clustering coefficient).

    The encoding is the diagonal of the k-step random walk matrix:
        SE[i, k] = (P^k)[i, i]

    where P is the transition matrix.

    Args:
        walk_length (int): The number of random walk steps.
        attr_name (str, optional): The attribute name of the structural
            encoding. (default: :obj:`"random_walk_se"`)
        is_undirected (bool, optional): If set to :obj:`True`, the graph is
            assumed to be undirected, and the adjacency matrix will be made
            symmetric. (default: :obj:`False`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_se',
        is_undirected: bool = False,
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name
        self.is_undirected = is_undirected

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

        if self.is_undirected:
            # Make symmetric for undirected graphs
            adj_dense = adj_dense + adj_dense.t()

        # Compute degree and create transition matrix
        deg = adj_dense.sum(dim=1, keepdim=True)
        deg = torch.clamp(deg, min=1)  # Avoid division by zero
        transition = adj_dense / deg

        # Compute random walk return probabilities (diagonal elements)
        se = torch.zeros((num_nodes, self.walk_length), dtype=torch.float)
        current = torch.eye(num_nodes, dtype=torch.float)

        for i in range(self.walk_length):
            current = current @ transition
            # Diagonal gives probability of returning to the same node
            se[:, i] = current.diag()

        # Map back to HeteroData node types
        add_node_attr(data, se, self.attr_name)

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(walk_length={self.walk_length})'



@functional_transform('add_hetero_hop_distance_encoding')
class AddHeteroHopDistanceEncoding(BaseTransform):
    r"""Adds hop distance positional encoding as relative encoding.

    For each pair of nodes (vi, vj), computes the shortest path distance p(vi, vj).
    This captures structural proximity and can be used with a learnable embedding
    matrix:

        h_hop(vi, vj) = W_hop · onehot(p(vi, vj))

    Based on the approach from `"Do Transformers Really Perform Bad for Graph
    Representation?" <https://arxiv.org/abs/2106.05234>`_ (Graphormer).

    For heterogeneous graphs, when `full_matrix=True`, additional node type
    information can be preserved by setting `node_type_aware=True`. This stores:
        - `data.hop_distance`: (num_nodes, num_nodes) distance matrix
        - `data.node_type_ids`: (num_nodes,) node type ID for each node
        - `data.node_type_pair`: (num_nodes, num_nodes) encodes (src_type, dst_type) pairs

    This allows Graph Transformers to use both structural (hop distance) and
    semantic (node type) information in attention bias computation.

    Args:
        h_max (int): Maximum hop distance to consider. Distances > h_max
            are clipped to h_max (representing "far" or "unreachable" nodes).
            Set to 2 - 3 for 2hop sampled subgraphs.
            Set to min(walk_length // 2, 10) for random walk sampled subgraphs.
        attr_name (str, optional): The attribute name of the positional
            encoding. (default: :obj:`"hop_distance"`)
        is_undirected (bool, optional): If set to :obj:`True`, the graph is
            assumed to be undirected for distance computation.
            (default: :obj:`False`)
        full_matrix (bool, optional): If set to :obj:`True`, stores the full
            pairwise distance matrix as a graph-level attribute (for use in
            Graph Transformers with fully-connected attention). If :obj:`False`,
            stores hop distances only for existing edges. Note that when
            :obj:`full_matrix=False`, the hop distance for existing edges is
            always 1 (direct connection), which may be redundant. Use
            :obj:`full_matrix=True` for attention bias in Graph Transformers.
            (default: :obj:`True`)
        node_type_aware (bool, optional): If set to :obj:`True` (only effective
            when `full_matrix=True`), also stores node type information:
            - `node_type_ids`: (num_nodes,) tensor mapping each node to its type ID
            - `node_type_pair`: (num_nodes, num_nodes) tensor encoding the
              (src_type, dst_type) pair as `src_type * num_node_types + dst_type`
            This enables type-aware attention bias in heterogeneous Graph Transformers.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        h_max: int,
        attr_name: Optional[str] = 'hop_distance',
        is_undirected: bool = False,
        full_matrix: bool = True,
        node_type_aware: bool = False,
    ) -> None:
        self.h_max = h_max
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.full_matrix = full_matrix
        self.node_type_aware = node_type_aware

    def forward(self, data: HeteroData) -> HeteroData:
        assert isinstance(data, HeteroData), (
            f"'{self.__class__.__name__}' only supports 'HeteroData' "
            f"(got '{type(data)}')"
        )

        # Convert to homogeneous to compute shortest paths
        homo_data = data.to_homogeneous()
        edge_index = homo_data.edge_index
        num_nodes = homo_data.num_nodes
        num_edges = edge_index.size(1)

        if num_nodes == 0 or num_edges == 0:
            # Handle empty graph case
            if self.full_matrix:
                data[self.attr_name] = torch.zeros(
                    (num_nodes, num_nodes), dtype=torch.long
                )
                if self.node_type_aware:
                    data['node_type_ids'] = torch.zeros(num_nodes, dtype=torch.long)
                    data['node_type_pair'] = torch.zeros(
                        (num_nodes, num_nodes), dtype=torch.long
                    )
            else:
                for edge_type in data.edge_types:
                    num_type_edges = data[edge_type].num_edges
                    data[edge_type][self.attr_name] = torch.zeros(
                        num_type_edges, dtype=torch.long
                    )
            return data

        # Build adjacency matrix for shortest path computation
        adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes))
        adj_dense = adj.to_dense()

        if self.is_undirected:
            # Make symmetric for undirected graphs
            adj_dense = adj_dense + adj_dense.t()

        adj_dense = (adj_dense > 0).float()  # Binary adjacency

        # Compute shortest path distances using BFS via matrix powers
        # dist_matrix[i, j] = shortest path distance from node i to node j
        dist_matrix = torch.full(
            (num_nodes, num_nodes), self.h_max, dtype=torch.long
        )
        dist_matrix.fill_diagonal_(0)  # Distance to self is 0

        # BFS: track which nodes are reachable and at what distance
        reachable = torch.eye(num_nodes, dtype=torch.bool)
        current_frontier = adj_dense.bool()

        for hop in range(1, self.h_max + 1):
            # Nodes newly reachable at this hop (not previously seen)
            newly_reachable = current_frontier & ~reachable
            # Set distance for newly reachable nodes
            dist_matrix[newly_reachable] = hop
            # Update reachable set
            reachable = reachable | current_frontier

            # Early exit if all nodes are reachable (no need to continue BFS)
            if reachable.all():
                break

            # Expand frontier to next hop neighbors
            current_frontier = (current_frontier.float() @ adj_dense) > 0

        if self.full_matrix:
            # Store full pairwise distance matrix as graph-level attribute on HeteroData
            # Shape: (num_nodes, num_nodes) - for use in Graph Transformers
            # Access via: data.hop_distance or data['hop_distance']
            # Can be used as attention bias: bias = learnable_embedding[data.hop_distance.long()]
            # Note: Node ordering follows data.to_homogeneous() order (by node_type alphabetically)
            data[self.attr_name] = dist_matrix.float()

            if self.node_type_aware:
                # Store node type information for heterogeneous-aware attention
                # homo_data.node_type contains the type ID for each node after to_homogeneous()
                node_type_ids = homo_data.node_type  # Shape: (num_nodes,)
                data['node_type_ids'] = node_type_ids

                # Compute pairwise node type encoding: (src_type, dst_type) -> single ID
                # node_type_pair[i, j] = node_type_ids[i] * num_node_types + node_type_ids[j]
                # This allows looking up type-specific attention biases
                num_node_types = len(data.node_types)
                # Outer product style: src_types[:, None] * num_types + dst_types[None, :]
                node_type_pair = (
                    node_type_ids.unsqueeze(1) * num_node_types +
                    node_type_ids.unsqueeze(0)
                )
                data['node_type_pair'] = node_type_pair

                # Also store the mapping from type ID to type name for reference
                # Node types are sorted alphabetically in to_homogeneous()
                data['node_type_names'] = sorted(data.node_types)
        else:
            # Extract hop distances for each edge in edge_index
            src_nodes = edge_index[0]  # Source nodes
            dst_nodes = edge_index[1]  # Destination nodes
            edge_hop_distances = dist_matrix[src_nodes, dst_nodes]
            # Store hop distances only for existing edges
            # Map back to HeteroData edge types
            add_edge_attr(data, edge_hop_distances.unsqueeze(-1).float(), self.attr_name)

        return data

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(h_max={self.h_max}, '
            f'full_matrix={self.full_matrix}, node_type_aware={self.node_type_aware})'
        )
