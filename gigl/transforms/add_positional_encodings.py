from typing import Optional

import torch

from torch_geometric.data import HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_torch_sparse_tensor

from gigl.transforms.utils import add_node_attr

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
    ...     AddHeteroHopDistanceEncoding(h_max=3),
    ... ])
    >>> data = transform(data)
    >>>
    >>> # For Graph Transformers, use hop distance encoding for attention bias
    >>> # Returns sparse matrix (0 for unreachable, 1-h_max for reachable pairs)
    >>> transform = AddHeteroHopDistanceEncoding(h_max=5)
    >>> data = transform(data)
    >>> print(data.hop_distance.shape)       # (num_total_nodes, num_total_nodes) sparse
    >>> print(data.hop_distance.is_sparse)   # True
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
        attach_to_x (bool, optional): If set to :obj:`True`, the encoding is
            concatenated directly to :obj:`data[node_type].x` for each node type
            instead of being stored as a separate attribute. (default: :obj:`False`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_pe',
        is_undirected: bool = False,
        attach_to_x: bool = False,
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.attach_to_x = attach_to_x

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
                empty_pe = torch.zeros(
                    (data[node_type].num_nodes, self.walk_length),
                    dtype=torch.float,
                )
                effective_attr_name = None if self.attach_to_x else self.attr_name
                add_node_attr(data, {node_type: empty_pe}, effective_attr_name)
            return data

        # Compute transition matrix (row-stochastic) using sparse operations
        adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes))

        if self.is_undirected:
            # Make symmetric for undirected graphs
            adj = (adj + adj.t()).coalesce()

        # Compute degree for row normalization
        adj_coalesced = adj.coalesce()
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, adj_coalesced.indices()[0], adj_coalesced.values().float())
        deg = torch.clamp(deg, min=1)  # Avoid division by zero

        # Create row-normalized transition matrix (sparse)
        # P[i,j] = A[i,j] / deg[i]
        row_indices = adj_coalesced.indices()[0]
        normalized_values = adj_coalesced.values().float() / deg[row_indices]
        transition = torch.sparse_coo_tensor(
            adj_coalesced.indices(),
            normalized_values,
            size=(num_nodes, num_nodes),
        ).coalesce()

        # Compute random walk positional encoding using sparse operations
        # PE[j, k] = sum of column j excluding diagonal = Σ_{i≠j} (P^k)[i, j]
        pe = torch.zeros((num_nodes, self.walk_length), dtype=torch.float, device=edge_index.device)

        # Start with identity matrix (sparse)
        identity_indices = torch.arange(num_nodes, device=edge_index.device)
        current = torch.sparse_coo_tensor(
            torch.stack([identity_indices, identity_indices]),
            torch.ones(num_nodes, device=edge_index.device),
            size=(num_nodes, num_nodes),
        ).coalesce()

        for k in range(self.walk_length):
            current = torch.sparse.mm(current, transition).coalesce()
            # Column sum = sum over rows for each column
            col_sum = torch.zeros(num_nodes, device=edge_index.device)
            col_sum.scatter_add_(0, current.indices()[1], current.values())
            # Extract diagonal elements
            diag = torch.zeros(num_nodes, device=edge_index.device)
            diag_mask = current.indices()[0] == current.indices()[1]
            if diag_mask.any():
                diag.scatter_add_(0, current.indices()[0][diag_mask], current.values()[diag_mask])
            pe[:, k] = col_sum - diag

        # Map back to HeteroData node types
        # If attach_to_x is True, pass None as attr_name to concatenate to x directly
        effective_attr_name = None if self.attach_to_x else self.attr_name
        add_node_attr(data, pe, effective_attr_name)

        return data

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(walk_length={self.walk_length}, '
            f'attach_to_x={self.attach_to_x})'
        )


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
        attach_to_x (bool, optional): If set to :obj:`True`, the encoding is
            concatenated directly to :obj:`data[node_type].x` for each node type
            instead of being stored as a separate attribute. (default: :obj:`False`)
    """
    def __init__(
        self,
        walk_length: int,
        attr_name: Optional[str] = 'random_walk_se',
        is_undirected: bool = False,
        attach_to_x: bool = False,
    ) -> None:
        self.walk_length = walk_length
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        self.attach_to_x = attach_to_x

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
                empty_se = torch.zeros(
                    (data[node_type].num_nodes, self.walk_length),
                    dtype=torch.float,
                )
                effective_attr_name = None if self.attach_to_x else self.attr_name
                add_node_attr(data, {node_type: empty_se}, effective_attr_name)
            return data

        # Compute transition matrix (row-stochastic) using sparse operations
        adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes))

        if self.is_undirected:
            # Make symmetric for undirected graphs
            adj = (adj + adj.t()).coalesce()

        # Compute degree for row normalization
        adj_coalesced = adj.coalesce()
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, adj_coalesced.indices()[0], adj_coalesced.values().float())
        deg = torch.clamp(deg, min=1)  # Avoid division by zero

        # Create row-normalized transition matrix (sparse)
        # P[i,j] = A[i,j] / deg[i]
        row_indices = adj_coalesced.indices()[0]
        normalized_values = adj_coalesced.values().float() / deg[row_indices]
        transition = torch.sparse_coo_tensor(
            adj_coalesced.indices(),
            normalized_values,
            size=(num_nodes, num_nodes),
        ).coalesce()

        # Compute random walk return probabilities (diagonal elements) using sparse operations
        se = torch.zeros((num_nodes, self.walk_length), dtype=torch.float, device=edge_index.device)

        # Start with identity matrix (sparse)
        identity_indices = torch.arange(num_nodes, device=edge_index.device)
        current = torch.sparse_coo_tensor(
            torch.stack([identity_indices, identity_indices]),
            torch.ones(num_nodes, device=edge_index.device),
            size=(num_nodes, num_nodes),
        ).coalesce()

        for k in range(self.walk_length):
            current = torch.sparse.mm(current, transition).coalesce()
            # Extract diagonal elements: probability of returning to the same node
            diag_mask = current.indices()[0] == current.indices()[1]
            if diag_mask.any():
                se[:, k].scatter_add_(0, current.indices()[0][diag_mask], current.values()[diag_mask])

        # Map back to HeteroData node types
        # If attach_to_x is True, pass None as attr_name to concatenate to x directly
        effective_attr_name = None if self.attach_to_x else self.attr_name
        add_node_attr(data, se, effective_attr_name)

        return data

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(walk_length={self.walk_length}, '
            f'attach_to_x={self.attach_to_x})'
        )



@functional_transform('add_hetero_hop_distance_encoding')
class AddHeteroHopDistanceEncoding(BaseTransform):
    r"""Adds hop distance positional encoding as relative encoding (sparse).

    For each pair of nodes (vi, vj), computes the shortest path distance p(vi, vj).
    This captures structural proximity and can be used with a learnable embedding
    matrix:

        h_hop(vi, vj) = W_hop · onehot(p(vi, vj))

    Based on the approach from `"Do Transformers Really Perform Bad for Graph
    Representation?" <https://arxiv.org/abs/2106.05234>`_ (Graphormer).

    The output is a **sparse matrix** where:
        - Reachable pairs (i, j) within h_max hops have value = hop distance (1 to h_max)
        - Unreachable pairs have value = 0 (not stored in sparse tensor)
        - Self-loops (diagonal) are not stored (distance to self is implicitly 0)

    This sparse representation avoids GPU memory blowup for large graphs.

    Args:
        h_max (int): Maximum hop distance to consider. Distances > h_max
            are treated as unreachable (value 0 in sparse matrix).
            Set to 2-3 for 2-hop sampled subgraphs.
            Set to min(walk_length // 2, 10) for random walk sampled subgraphs.
        attr_name (str, optional): The attribute name of the positional
            encoding. (default: :obj:`"hop_distance"`)
        is_undirected (bool, optional): If set to :obj:`True`, the graph is
            assumed to be undirected for distance computation.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        h_max: int,
        attr_name: Optional[str] = 'hop_distance',
        is_undirected: bool = False,
    ) -> None:
        self.h_max = h_max
        self.attr_name = attr_name
        self.is_undirected = is_undirected

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
            # Handle empty graph case - return empty sparse tensor
            empty_sparse = torch.sparse_coo_tensor(
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float),
                size=(num_nodes, num_nodes),
            ).coalesce()
            data[self.attr_name] = empty_sparse
            return data

        device = edge_index.device

        # Build sparse adjacency matrix for shortest path computation
        adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes))

        if self.is_undirected:
            # Make symmetric for undirected graphs
            adj = (adj + adj.t()).coalesce()

        # Binarize adjacency (sparse)
        adj_coalesced = adj.coalesce()
        adj = torch.sparse_coo_tensor(
            adj_coalesced.indices(),
            torch.ones(adj_coalesced.indices().size(1), device=device),
            size=(num_nodes, num_nodes),
        ).coalesce()

        # Compute sparse BFS to find all reachable pairs within h_max hops
        # Store (row, col, distance) for all reachable pairs
        all_rows = []
        all_cols = []
        all_dists = []

        # Track which pairs have been visited (using set of linear indices)
        # Start with diagonal (self-loops) as visited but don't include in output
        identity_indices = torch.arange(num_nodes, device=device)
        visited_linear = set((identity_indices * num_nodes + identity_indices).tolist())

        # Current frontier (sparse): edges reachable at current hop
        frontier = adj.coalesce()

        for hop in range(1, self.h_max + 1):
            frontier_indices = frontier.indices()

            if frontier_indices.size(1) == 0:
                break

            frontier_i = frontier_indices[0]
            frontier_j = frontier_indices[1]
            frontier_linear = (frontier_i * num_nodes + frontier_j).tolist()

            # Find newly reachable pairs (not in visited set)
            new_mask = []
            new_pairs_linear = []
            for idx, lin_idx in enumerate(frontier_linear):
                if lin_idx not in visited_linear:
                    new_mask.append(idx)
                    new_pairs_linear.append(lin_idx)
                    visited_linear.add(lin_idx)

            if new_mask:
                new_mask = torch.tensor(new_mask, device=device, dtype=torch.long)
                new_i = frontier_i[new_mask]
                new_j = frontier_j[new_mask]

                all_rows.append(new_i)
                all_cols.append(new_j)
                all_dists.append(torch.full((new_i.size(0),), hop, device=device, dtype=torch.float))

            # Expand frontier: frontier = frontier @ adj (sparse matmul)
            frontier = torch.sparse.mm(frontier, adj).coalesce()

        # Build sparse distance matrix
        if all_rows:
            dist_rows = torch.cat(all_rows)
            dist_cols = torch.cat(all_cols)
            dist_vals = torch.cat(all_dists)
        else:
            dist_rows = torch.zeros(0, dtype=torch.long, device=device)
            dist_cols = torch.zeros(0, dtype=torch.long, device=device)
            dist_vals = torch.zeros(0, dtype=torch.float, device=device)

        # Create sparse distance matrix
        # Unreachable pairs have value 0 (not stored)
        # Reachable pairs have value = hop distance (1 to h_max)
        dist_sparse = torch.sparse_coo_tensor(
            torch.stack([dist_rows, dist_cols]),
            dist_vals,
            size=(num_nodes, num_nodes),
        ).coalesce()

        # Store sparse pairwise distance matrix as graph-level attribute
        # Access via: data.hop_distance or data['hop_distance']
        # Usage in attention: dist = data.hop_distance.to_dense() for small graphs,
        #   or use sparse indexing for memory efficiency
        # Note: Node ordering follows data.to_homogeneous() order (by node_type alphabetically)
        data[self.attr_name] = dist_sparse

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(h_max={self.h_max})'
