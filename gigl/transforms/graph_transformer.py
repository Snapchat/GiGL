# TODO: support RW sampling, edges from data.ppr_edge, data.ppr_weights
# TODO: output attention bias as well
"""
Transform HeteroData to Graph Transformer sequence input.

This module provides functionality to convert PyG HeteroData objects (typically
batched 2-hop subgraphs) into sequence format suitable for Graph Transformers.

For each anchor node in the batch, the transform extracts its k-hop neighborhood
and creates a fixed-length sequence of node features with padding.

Example Usage:
    >>> from torch_geometric.data import HeteroData
    >>> from gigl.transforms.graph_transformer import heterodata_to_graph_transformer_input
    >>>
    >>> # Create batched HeteroData (e.g., from NeighborLoader)
    >>> # First batch_size nodes in each node type are anchor nodes
    >>> data = HeteroData()
    >>> data['user'].x = torch.randn(100, 64)  # 100 users, first N are anchors
    >>> data['item'].x = torch.randn(50, 32)
    >>> data['user', 'buys', 'item'].edge_index = ...
    >>>
    >>> # Transform to Graph Transformer input
    >>> sequences = heterodata_to_graph_transformer_input(
    ...     data=data,
    ...     batch_size=32,
    ...     max_seq_len=128,
    ...     anchor_node_type='user',
    ... )
    >>> # sequences: (batch_size, max_seq_len, feature_dim)

    With Positional Encodings:
    You can attach node-level positional encodings (computed by transforms in
    gigl/transforms/add_positional_encodings.py) to node features:

    >>> from torch_geometric.transforms import Compose
    >>> from gigl.transforms.add_positional_encodings import (
    ...     AddHeteroRandomWalkPE,
    ...     AddHeteroRandomWalkSE,
    ...     AddHeteroHopDistanceEncoding,
    ... )
    >>>
    >>> # First apply PE transforms to the data
    >>> pe_transform = Compose([
    ...     AddHeteroRandomWalkPE(walk_length=8),
    ...     AddHeteroRandomWalkSE(walk_length=8),
    ...     AddHeteroHopDistanceEncoding(h_max=5),
    ... ])
    >>> data = pe_transform(data)
    >>>
    >>> # Transform to sequences with PE concatenated
    >>> sequences = heterodata_to_graph_transformer_input(
    ...     data=data,
    ...     batch_size=32,
    ...     max_seq_len=128,
    ...     anchor_node_type='user',
    ...     pe_attr_names=['random_walk_pe', 'random_walk_se'],  # Node-level PE
    ...     anchor_based_pe_attr_names=['hop_distance'],  # Anchor-based PE (N×N sparse CSR)
    ... )
    >>> # sequences: (batch_size, max_seq_len, feature_dim + 8 + 8 + 1)
"""

from typing import Optional

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import to_torch_sparse_tensor


def _concatenate_positional_encodings(
    data: HeteroData,
    homo_x: Tensor,
    pe_attr_names: list[str],
    device: torch.device,
) -> Tensor:
    """
    Concatenate positional encodings from HeteroData to homogeneous node features.

    Positional encodings are stored as per-node-type attributes (e.g.,
    data['user'].random_walk_pe). This function retrieves them for each node type,
    orders them according to the homogeneous graph ordering (alphabetically by
    node type), and concatenates them to the node features.

    Args:
        data: HeteroData with positional encodings as node attributes.
        homo_x: (num_nodes, feature_dim) homogeneous node features.
        pe_attr_names: List of attribute names to concatenate (e.g., ['random_walk_pe']).
        device: Device for tensors.

    Returns:
        (num_nodes, feature_dim + sum(pe_dims)) node features with PE concatenated.
    """
    # Node types are sorted alphabetically in to_homogeneous()
    sorted_node_types = sorted(data.node_types)

    pe_tensors = []
    for attr_name in pe_attr_names:
        # Collect PE for each node type in sorted order
        pe_parts = []
        for node_type in sorted_node_types:
            node_store = data[node_type]
            if hasattr(node_store, attr_name):
                pe = getattr(node_store, attr_name)
                pe_parts.append(pe.to(device))
            else:
                # If PE not available for this node type, create zeros
                # Infer PE dimension from another node type that has it
                pe_dim = None
                for nt in sorted_node_types:
                    if hasattr(data[nt], attr_name):
                        pe_dim = getattr(data[nt], attr_name).size(-1)
                        break
                if pe_dim is None:
                    raise ValueError(
                        f"Positional encoding '{attr_name}' not found in any node type. "
                        f"Make sure to apply the corresponding transform (e.g., "
                        f"AddHeteroRandomWalkPE) before calling this function."
                    )
                pe_parts.append(
                    torch.zeros(node_store.num_nodes, pe_dim, device=device)
                )

        # Concatenate PE for all node types (in homogeneous order)
        pe_tensors.append(torch.cat(pe_parts, dim=0))

    # Concatenate all PEs to node features
    return torch.cat([homo_x] + pe_tensors, dim=-1)


def heterodata_to_graph_transformer_input(
    data: HeteroData,
    batch_size: int,
    max_seq_len: int,
    anchor_node_type: NodeType,
    feature_dim: Optional[int] = None,
    hop_distance: int = 2,
    include_anchor_first: bool = True,
    padding_value: float = 0.0,
    pe_attr_names: Optional[list[str]] = None,
    anchor_based_pe_attr_names: Optional[list[str]] = None,
) -> Tensor:
    """
    Transform a HeteroData object to Graph Transformer sequence input.

    Given a batched HeteroData object where the first `batch_size` nodes of
    `anchor_node_type` are anchor nodes, this function extracts the k-hop
    neighborhood for each anchor and creates padded sequences.

    Uses sparse matrix operations for efficient batched k-hop neighbor extraction.

    Args:
        data: HeteroData object containing node features and edge indices.
            Expected to have node features as `data[node_type].x`.
        batch_size: Number of anchor nodes (first batch_size nodes of anchor_node_type).
        max_seq_len: Maximum sequence length (neighbors beyond this are truncated).
        anchor_node_type: The node type of anchor nodes.
        feature_dim: Output feature dimension. If None, inferred from data.
            If provided and different from input, features are projected.
        hop_distance: Number of hops to consider for neighborhood (default: 2).
        include_anchor_first: If True, anchor node is always first in sequence.
        padding_value: Value to use for padding (default: 0.0).
        pe_attr_names: List of positional encoding attribute names to concatenate
            to node features. These should be node-level attributes stored by
            transforms like AddHeteroRandomWalkPE or AddHeteroRandomWalkSE.
            Example: ['random_walk_pe', 'random_walk_se']
            If None, no positional encodings are attached. (default: None)
        anchor_based_pe_attr_names: List of graph-level attribute names containing
            sparse (N x N) matrices for anchor-based positional encodings. For each
            node in the sequence, the value PE[anchor_idx, node_idx] is looked up
            and concatenated to node features.
            Examples: ['hop_distance'] (from AddHeteroHopDistanceEncoding)
            If None, no anchor-based PEs are attached. (default: None)

    Returns:
        sequences: (batch_size, max_seq_len, feature_dim) padded node features
            where feature_dim includes concatenated positional encodings if specified.
    """
    device = data[anchor_node_type].x.device

    # Convert to homogeneous for easier neighborhood extraction
    homo_data = data.to_homogeneous()
    homo_x = homo_data.x  # (total_nodes, feature_dim)
    homo_edge_index = homo_data.edge_index  # (2, num_edges)

    num_nodes = homo_data.num_nodes

    # Concatenate positional encodings to node features if specified
    if pe_attr_names:
        homo_x = _concatenate_positional_encodings(
            data=data,
            homo_x=homo_x,
            pe_attr_names=pe_attr_names,
            device=device,
        )

    # Get node type to index mapping (sorted alphabetically)
    sorted_node_types = sorted(data.node_types)

    # Find anchor node indices in homogeneous graph
    # Nodes are ordered by node_type (alphabetically), then by original index
    offset = 0
    for nt in sorted_node_types:
        if nt == anchor_node_type:
            break
        offset += data[nt].num_nodes

    # Anchor nodes are at positions [offset, offset + batch_size)
    anchor_indices = torch.arange(offset, offset + batch_size, device=device)

    # Infer feature dimension
    if feature_dim is None:
        feature_dim = homo_x.size(1)

    # Use sparse matrix operations for efficient k-hop neighbor extraction
    # Returns: (batch_size, num_nodes) sparse matrix where non-zero entries are reachable
    reachable = _get_k_hop_neighbors_sparse(
        anchor_indices=anchor_indices,
        edge_index=homo_edge_index,
        num_nodes=num_nodes,
        k=hop_distance,
        device=device,
    )

    # Get anchor-based PE matrices if specified
    anchor_based_pe_matrices = []
    if anchor_based_pe_attr_names:
        for attr_name in anchor_based_pe_attr_names:
            if hasattr(data, attr_name):
                anchor_based_pe_matrices.append(getattr(data, attr_name))
            else:
                raise ValueError(
                    f"Anchor-based PE attribute '{attr_name}' not found in data. "
                    f"Make sure to apply the corresponding transform first."
                )

    # Build sequences for each anchor using the sparse reachable mask
    sequences = _build_sequences_from_sparse_neighbors(
        reachable=reachable,
        anchor_indices=anchor_indices,
        node_features=homo_x,
        max_seq_len=max_seq_len,
        include_anchor_first=include_anchor_first,
        padding_value=padding_value,
        device=device,
        anchor_based_pe_matrices=anchor_based_pe_matrices if anchor_based_pe_matrices else None,
    )

    return sequences


def _get_k_hop_neighbors_sparse(
    anchor_indices: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    k: int,
    device: torch.device,
) -> Tensor:
    """
    Get k-hop reachable nodes for all anchors using sparse matrix multiplication.

    Follows the same efficient pattern as AddHeteroRandomWalkPE: simple sparse
    matrix powers to accumulate reachability without expensive membership checks.

    Args:
        anchor_indices: (batch_size,) anchor node indices
        edge_index: (2, num_edges) edge index
        num_nodes: Total number of nodes
        k: Number of hops
        device: Device for tensors

    Returns:
        reachable: (batch_size, num_nodes) sparse tensor, non-zero if node is reachable within k hops
    """
    batch_size = anchor_indices.size(0)

    # Build sparse adjacency matrix (binarized) - coalesce once
    adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes)).coalesce()
    adj = torch.sparse_coo_tensor(
        adj.indices(),
        torch.ones(adj.indices().size(1), device=device, dtype=torch.float),
        size=(num_nodes, num_nodes),
    )  # No coalesce needed - indices already unique from previous coalesce

    # Initialize: sparse matrix where row i has a 1 at column anchor_indices[i]
    reachable = torch.sparse_coo_tensor(
        torch.stack([
            torch.arange(batch_size, device=device),
            anchor_indices,
        ]),
        torch.ones(batch_size, device=device, dtype=torch.float),
        size=(batch_size, num_nodes),
    )  # No coalesce needed - indices are unique

    current = reachable

    for _ in range(k):
        # Expand: current @ adj gives nodes reachable in one more hop
        current = torch.sparse.mm(current, adj)

        if current._nnz() == 0:
            break

        # Accumulate into reachable
        reachable = reachable + current

    # Coalesce and binarize final result
    reachable = reachable.coalesce()
    reachable = torch.sparse_coo_tensor(
        reachable.indices(),
        torch.ones(reachable._nnz(), device=device, dtype=torch.float),
        size=(batch_size, num_nodes),
    ).coalesce()

    return reachable


def _build_sequences_from_sparse_neighbors(
    reachable: Tensor,
    anchor_indices: Tensor,
    node_features: Tensor,
    max_seq_len: int,
    include_anchor_first: bool,
    padding_value: float,
    device: torch.device,
    anchor_based_pe_matrices: Optional[list[Tensor]] = None,
) -> Tensor:
    """
    Build padded sequences from sparse reachability information.

    Args:
        reachable: (batch_size, num_nodes) sparse tensor, non-zero if node is reachable
        anchor_indices: (batch_size,) anchor node indices
        node_features: (num_nodes, feature_dim) node features
        max_seq_len: Maximum sequence length
        include_anchor_first: Whether to put anchor first
        padding_value: Padding value for sequences
        device: Device for tensors
        anchor_based_pe_matrices: Optional list of (num_nodes, num_nodes) sparse tensors
            containing pairwise values. For each matrix, the value at [anchor, node]
            is looked up and concatenated to node features.

    Returns:
        sequences: (batch_size, max_seq_len, feature_dim + num_pe_matrices)
    """
    batch_size = anchor_indices.size(0)
    base_feature_dim = node_features.size(1)

    # Output feature dim includes one value per anchor-based PE matrix
    num_anchor_pe = len(anchor_based_pe_matrices) if anchor_based_pe_matrices else 0
    output_feature_dim = base_feature_dim + num_anchor_pe

    # Initialize output
    sequences = torch.full(
        (batch_size, max_seq_len, output_feature_dim),
        padding_value,
        dtype=node_features.dtype,
        device=device,
    )

    # Extract sparse indices - sorted by (batch, node) after coalesce
    indices = reachable.indices()
    batch_idx = indices[0]
    node_idx = indices[1]

    if batch_idx.numel() == 0:
        if include_anchor_first:
            sequences[:, 0, :base_feature_dim] = node_features[anchor_indices]
            # Anchor-based PE for self is 0 (distance to self)
            if num_anchor_pe > 0:
                sequences[:, 0, base_feature_dim:] = 0.0
        return sequences

    if include_anchor_first:
        # Place anchors at position 0
        sequences[:, 0, :base_feature_dim] = node_features[anchor_indices]
        # Anchor-based PE for self is 0
        if num_anchor_pe > 0:
            sequences[:, 0, base_feature_dim:] = 0.0

        # Remove anchors from node list
        keep = node_idx != anchor_indices[batch_idx]
        batch_idx = batch_idx[keep]
        node_idx = node_idx[keep]
        start_pos = 1
    else:
        start_pos = 0

    if batch_idx.numel() == 0:
        return sequences

    # Compute positions within each batch
    # Indices are sorted by batch, so detect boundaries
    n = batch_idx.size(0)
    is_new_batch = torch.zeros(n, dtype=torch.long, device=device)
    is_new_batch[0] = 1
    if n > 1:
        is_new_batch[1:] = (batch_idx[1:] != batch_idx[:-1]).long()

    group_id = is_new_batch.cumsum(0) - 1
    group_starts = torch.nonzero(is_new_batch, as_tuple=True)[0]
    positions = torch.arange(n, device=device) - group_starts[group_id] + start_pos

    # Filter to max_seq_len
    valid = positions < max_seq_len
    valid_batch_idx = batch_idx[valid]
    valid_positions = positions[valid]
    valid_node_idx = node_idx[valid]

    # Scatter node features
    sequences[valid_batch_idx, valid_positions, :base_feature_dim] = node_features[valid_node_idx]

    # Look up and scatter anchor-based PE values
    if anchor_based_pe_matrices:
        anchor_for_entry = anchor_indices[valid_batch_idx]

        for pe_idx, pe_matrix in enumerate(anchor_based_pe_matrices):
            pe_values = _lookup_csr_values(
                csr_matrix=pe_matrix,
                row_indices=anchor_for_entry,
                col_indices=valid_node_idx,
            )
            sequences[valid_batch_idx, valid_positions, base_feature_dim + pe_idx] = pe_values

    return sequences


def _lookup_csr_values(
    csr_matrix: Tensor,
    row_indices: Tensor,
    col_indices: Tensor,
    default_value: float = 0.0,
) -> Tensor:
    """
    Look up values in a CSR sparse matrix for given (row, col) pairs.

    Vectorized CSR lookup: for each query, slice the row and search for the column.
    Time complexity: O(n * avg_nnz_per_row), typically O(n) for sparse matrices.

    Args:
        csr_matrix: (num_rows, num_cols) sparse CSR tensor
        row_indices: (n,) row indices to look up
        col_indices: (n,) column indices to look up
        default_value: Value for missing entries (default: 0.0)

    Returns:
        (n,) values from csr_matrix[row, col], or default_value if not present
    """
    n = row_indices.size(0)
    device = row_indices.device

    if n == 0:
        return torch.zeros(0, device=device, dtype=torch.float)

    crow_indices = csr_matrix.crow_indices()
    col_indices_csr = csr_matrix.col_indices()
    values_csr = csr_matrix.values()

    # Get row start/end pointers
    row_starts = crow_indices[row_indices]
    row_ends = crow_indices[row_indices + 1]
    row_lengths = row_ends - row_starts
    max_row_len = row_lengths.max().item()

    if max_row_len == 0:
        return torch.full((n,), default_value, device=device, dtype=torch.float)

    # Build offset matrix: (n, max_row_len)
    offsets = row_starts.unsqueeze(1) + torch.arange(max_row_len, device=device)
    valid_mask = offsets < row_ends.unsqueeze(1)

    # Safe indexing with clamping
    nnz = col_indices_csr.size(0)
    offsets_clamped = offsets.clamp(max=max(nnz - 1, 0))

    # Get columns at offsets and find matches
    cols_at_offsets = col_indices_csr[offsets_clamped]
    col_matches = (cols_at_offsets == col_indices.unsqueeze(1)) & valid_mask

    # Find which queries have matches
    found = col_matches.any(dim=1)

    # Initialize output
    result = torch.full((n,), default_value, device=device, dtype=torch.float)

    if found.any():
        # Get match positions and retrieve values
        match_offsets = col_matches.float().argmax(dim=1)
        value_indices = row_starts[found] + match_offsets[found]
        result[found] = values_csr[value_indices].float()

    return result
