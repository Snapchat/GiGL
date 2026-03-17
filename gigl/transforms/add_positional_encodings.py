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


@functional_transform("add_hetero_random_walk_pe")
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
        attr_name: Optional[str] = "random_walk_pe",
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
        pe = torch.zeros(
            (num_nodes, self.walk_length), dtype=torch.float, device=edge_index.device
        )

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
                diag.scatter_add_(
                    0, current.indices()[0][diag_mask], current.values()[diag_mask]
                )
            pe[:, k] = col_sum - diag

        # Map back to HeteroData node types
        # If attach_to_x is True, pass None as attr_name to concatenate to x directly
        effective_attr_name = None if self.attach_to_x else self.attr_name
        add_node_attr(data, pe, effective_attr_name)

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(walk_length={self.walk_length}, "
            f"attach_to_x={self.attach_to_x})"
        )


@functional_transform("add_hetero_random_walk_se")
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
        attr_name: Optional[str] = "random_walk_se",
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
        se = torch.zeros(
            (num_nodes, self.walk_length), dtype=torch.float, device=edge_index.device
        )

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
                se[:, k].scatter_add_(
                    0, current.indices()[0][diag_mask], current.values()[diag_mask]
                )

        # Map back to HeteroData node types
        # If attach_to_x is True, pass None as attr_name to concatenate to x directly
        effective_attr_name = None if self.attach_to_x else self.attr_name
        add_node_attr(data, se, effective_attr_name)

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(walk_length={self.walk_length}, "
            f"attach_to_x={self.attach_to_x})"
        )


@functional_transform("add_hetero_hop_distance_encoding")
class AddHeteroHopDistanceEncoding(BaseTransform):
    r"""Adds hop distance positional encoding as relative encoding (sparse CSR).

    For each pair of nodes (vi, vj), computes the shortest path distance p(vi, vj).
    This captures structural proximity and can be used with a learnable embedding
    matrix:

        h_hop(vi, vj) = W_hop · onehot(p(vi, vj))

    Based on the approach from `"Do Transformers Really Perform Bad for Graph
    Representation?" <https://arxiv.org/abs/2106.05234>`_ (Graphormer).

    The output is a **sparse CSR matrix** where:
        - Reachable pairs (i, j) within h_max hops have value = hop distance (1 to h_max)
        - Unreachable pairs have value = 0 (not stored in sparse tensor)
        - Self-loops (diagonal) are not stored (distance to self is implicitly 0)

    CSR format is used for efficient row-based lookups during sequence building.

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
        attr_name: Optional[str] = "hop_distance",
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
            # Handle empty graph case - return empty sparse CSR tensor
            empty_sparse = torch.sparse_csr_tensor(
                torch.zeros(num_nodes + 1, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.float),
                size=(num_nodes, num_nodes),
            )
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

        # Memory-optimized BFS for computing shortest path distances
        #
        # Key memory optimizations:
        # 1. Use sorted linear indices with searchsorted for O(n log n) membership test
        #    (more memory efficient than torch.isin which may create hash tables)
        # 2. Store distances as int8 (h_max typically < 127)
        # 3. Avoid tensor concatenation in hot loop - use pre-sorted merge instead
        # 4. Explicit del statements to trigger garbage collection
        # 5. CSR format for sparse matmul (more memory efficient than COO)
        #
        # Memory complexity: O(nnz_frontier + nnz_visited) per iteration
        # where nnz_frontier can grow up to O(n^2) for dense graphs at large hop

        dist_matrix_rows = []
        dist_matrix_cols = []
        dist_matrix_vals = []

        # Choose tracking strategy based on graph size
        # For small graphs (n < 10000), bitmap is faster with O(1) lookup
        # For large graphs, sorted indices use less memory O(visited) vs O(n^2/8)
        USE_BITMAP = num_nodes < 10000

        if USE_BITMAP:
            # Dense bitmap: O(n^2 / 8) bytes, O(1) lookup
            # For n=10000, this is ~12.5 MB
            visited_bitmap = torch.zeros(
                num_nodes, num_nodes, dtype=torch.bool, device=device
            )
            visited_bitmap.fill_diagonal_(True)  # Mark diagonal as visited
        else:
            # Sorted linear indices: O(visited pairs) memory, O(log n) lookup
            identity_indices = torch.arange(num_nodes, device=device, dtype=torch.long)
            visited_linear = identity_indices * num_nodes + identity_indices  # Diagonal
            visited_linear = visited_linear.sort()[0]

        # Adjacency matrix in CSR format (more memory efficient for matmul)
        adj_csr = adj.to_sparse_csr()
        del adj  # Free COO adjacency

        # Current frontier (reachable pairs at current hop distance)
        frontier = adj_csr.to_sparse_coo().coalesce()

        for hop in range(1, self.h_max + 1):
            if hop > 1:
                # frontier = frontier @ adj (sparse matmul)
                # CSR @ CSR is most efficient
                frontier_csr = frontier.to_sparse_csr()
                del frontier
                frontier = (
                    torch.sparse.mm(frontier_csr, adj_csr).to_sparse_coo().coalesce()
                )
                del frontier_csr

            frontier_indices = frontier.indices()
            num_frontier = frontier_indices.size(1)
            if num_frontier == 0:
                break

            reach_i, reach_j = frontier_indices[0], frontier_indices[1]

            if USE_BITMAP:
                # O(1) lookup using dense bitmap
                is_visited = visited_bitmap[reach_i, reach_j]
                is_new = ~is_visited
                del is_visited
            else:
                # O(log n) lookup using sorted searchsorted
                frontier_linear = reach_i.long() * num_nodes + reach_j.long()
                insert_pos = torch.searchsorted(visited_linear, frontier_linear)
                insert_pos_clamped = insert_pos.clamp(max=visited_linear.size(0) - 1)
                is_visited = visited_linear[insert_pos_clamped] == frontier_linear
                is_new = ~is_visited
                del frontier_linear, insert_pos, insert_pos_clamped, is_visited

            num_new = is_new.sum().item()
            if num_new > 0:
                new_i = reach_i[is_new]
                new_j = reach_j[is_new]

                dist_matrix_rows.append(new_i)
                dist_matrix_cols.append(new_j)
                # Use int8 for hop distance (saves 4x memory vs float32)
                dist_matrix_vals.append(
                    torch.full((num_new,), hop, device=device, dtype=torch.int8)
                )

                # Update visited
                if USE_BITMAP:
                    visited_bitmap[new_i, new_j] = True
                else:
                    new_linear = new_i.long() * num_nodes + new_j.long()
                    visited_linear = torch.cat([visited_linear, new_linear]).sort()[0]
                    del new_linear

            del is_new, reach_i, reach_j

        # Clean up
        if USE_BITMAP:
            del visited_bitmap
        else:
            del visited_linear
        del adj_csr, frontier

        # Build sparse distance matrix
        if dist_matrix_rows:
            dist_rows = torch.cat(dist_matrix_rows)
            dist_cols = torch.cat(dist_matrix_cols)
            # Convert int8 to float for downstream compatibility
            dist_vals = torch.cat(dist_matrix_vals).float()
            # Free intermediate lists
            del dist_matrix_rows, dist_matrix_cols, dist_matrix_vals
        else:
            dist_rows = torch.zeros(0, dtype=torch.long, device=device)
            dist_cols = torch.zeros(0, dtype=torch.long, device=device)
            dist_vals = torch.zeros(0, dtype=torch.float, device=device)

        # Create sparse distance matrix in CSR format directly
        # CSR is more efficient for row-based lookups in _lookup_csr_values
        # Unreachable pairs have value 0 (not stored)
        # Reachable pairs have value = hop distance (1 to h_max)
        dist_coo = torch.sparse_coo_tensor(
            torch.stack([dist_rows, dist_cols]),
            dist_vals,
            size=(num_nodes, num_nodes),
        ).coalesce()
        dist_sparse = dist_coo.to_sparse_csr()
        del dist_coo

        # Store sparse pairwise distance matrix as graph-level attribute
        # Access via: data.hop_distance or data['hop_distance']
        # Usage in attention: use sparse indexing for memory efficiency
        # Note: Node ordering follows data.to_homogeneous() order (by node_type alphabetically)
        data[self.attr_name] = dist_sparse

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(h_max={self.h_max})"
