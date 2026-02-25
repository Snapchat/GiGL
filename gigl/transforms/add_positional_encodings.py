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

        # Build sparse adjacency matrix for shortest path computation
        adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes))

        if self.is_undirected:
            # Make symmetric for undirected graphs
            adj = (adj + adj.t()).coalesce()

        # Binarize adjacency (sparse)
        adj_coalesced = adj.coalesce()
        adj = torch.sparse_coo_tensor(
            adj_coalesced.indices(),
            torch.ones(adj_coalesced.indices().size(1), device=edge_index.device),
            size=(num_nodes, num_nodes),
        ).coalesce()

        if not self.full_matrix:
            # For edge-level distances only, use sparse BFS from source nodes
            # This avoids materializing the full distance matrix
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            edge_hop_distances = torch.full((num_edges,), self.h_max, dtype=torch.long, device=edge_index.device)

            # For direct edges, distance is 1
            edge_hop_distances[:] = 1

            # Map back to HeteroData edge types
            add_edge_attr(data, edge_hop_distances.unsqueeze(-1).float(), self.attr_name)
            return data

        # For full_matrix=True, we need the complete distance matrix
        # Use sparse BFS but accumulate into dense distance matrix
        device = edge_index.device
        dist_matrix = torch.full(
            (num_nodes, num_nodes), self.h_max, dtype=torch.long, device=device
        )
        dist_matrix.fill_diagonal_(0)  # Distance to self is 0

        # Track reachability using sparse tensors
        # reachable[i,j] = 1 if j is reachable from i
        identity_indices = torch.arange(num_nodes, device=device)
        reachable_indices = torch.stack([identity_indices, identity_indices])
        reachable_values = torch.ones(num_nodes, device=device, dtype=torch.bool)

        # Current frontier (sparse): nodes reachable at current hop
        frontier = adj.coalesce()

        for hop in range(1, self.h_max + 1):
            frontier_indices = frontier.indices()
            frontier_values = frontier.values()

            if frontier_indices.size(1) == 0:
                break

            # Find newly reachable: in frontier but not in reachable
            # Check each (i,j) in frontier against reachable
            frontier_i = frontier_indices[0]
            frontier_j = frontier_indices[1]

            # Create a set of reachable pairs for fast lookup
            # Convert to linear indices for comparison
            reachable_linear = reachable_indices[0] * num_nodes + reachable_indices[1]
            frontier_linear = frontier_i * num_nodes + frontier_j

            # Find which frontier edges are not yet reachable
            # Use searchsorted for efficiency
            reachable_linear_sorted, sort_idx = reachable_linear.sort()
            insert_pos = torch.searchsorted(reachable_linear_sorted, frontier_linear)
            insert_pos = insert_pos.clamp(max=reachable_linear_sorted.size(0) - 1)
            is_new = reachable_linear_sorted[insert_pos] != frontier_linear

            if is_new.any():
                new_i = frontier_i[is_new]
                new_j = frontier_j[is_new]
                dist_matrix[new_i, new_j] = hop

                # Update reachable set
                reachable_indices = torch.cat([
                    reachable_indices,
                    torch.stack([new_i, new_j])
                ], dim=1)
                reachable_values = torch.cat([
                    reachable_values,
                    torch.ones(new_i.size(0), device=device, dtype=torch.bool)
                ])

            # Check if all pairs are reachable
            if reachable_indices.size(1) >= num_nodes * num_nodes:
                break

            # Expand frontier: frontier = frontier @ adj (sparse matmul)
            frontier = torch.sparse.mm(frontier, adj).coalesce()

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

        return data

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(h_max={self.h_max}, '
            f'full_matrix={self.full_matrix}, node_type_aware={self.node_type_aware})'
        )
