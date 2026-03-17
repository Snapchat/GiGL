# TODO: support RW sampling, edges from data.ppr_edge, data.ppr_weights
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
    >>> sequences, valid_mask, attention_bias_data = heterodata_to_graph_transformer_input(
    ...     data=data,
    ...     batch_size=32,
    ...     max_seq_len=128,
    ...     anchor_node_type='user',
    ... )
    >>> # sequences: (batch_size, max_seq_len, feature_dim)
    >>> # valid_mask: (batch_size, max_seq_len)

    With Relative Encodings:
    Relative encodings stored as sparse graph-level attributes can be returned as
    raw attention-bias features:

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
    >>> # Transform to sequences with relative bias features
    >>> sequences, valid_mask, attention_bias_data = heterodata_to_graph_transformer_input(
    ...     data=data,
    ...     batch_size=32,
    ...     max_seq_len=128,
    ...     anchor_node_type='user',
    ...     anchor_based_pe_attr_names=['hop_distance'],  # Anchor-based PE (N×N sparse CSR)
    ... )
    >>> # sequences: (batch_size, max_seq_len, feature_dim)
    >>> # attention_bias_data['anchor_bias']: (batch_size, max_seq_len, 1)
"""

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import to_torch_sparse_tensor

AttentionBiasData = dict[str, Optional[Tensor]]


def heterodata_to_graph_transformer_input(
    data: HeteroData,
    batch_size: int,
    max_seq_len: int,
    anchor_node_type: NodeType,
    anchor_node_ids: Optional[Tensor] = None,
    feature_dim: Optional[int] = None,
    hop_distance: int = 2,
    include_anchor_first: bool = True,
    padding_value: float = 0.0,
    anchor_based_pe_attr_names: Optional[list[str]] = None,
    pairwise_pe_attr_names: Optional[list[str]] = None,
) -> tuple[Tensor, Tensor, AttentionBiasData]:
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
            Ignored if anchor_node_ids is provided.
        max_seq_len: Maximum sequence length (neighbors beyond this are truncated).
        anchor_node_type: The node type of anchor nodes.
        anchor_node_ids: Optional tensor of local node indices within anchor_node_type
            to use as anchors. If None, uses first batch_size nodes. (default: None)
        feature_dim: Output feature dimension. If None, inferred from data.
            If provided and different from input, features are projected.
        hop_distance: Number of hops to consider for neighborhood (default: 2).
        include_anchor_first: If True, anchor node is always first in sequence.
        padding_value: Value to use for padding (default: 0.0).
        anchor_based_pe_attr_names: List of relative-encoding attribute names
            containing sparse (N x N) matrices for anchor-relative positional
            encodings.
            For each node in the sequence, the value PE[anchor_idx, node_idx] is
            looked up and returned as attention-bias features.
            Examples: ['hop_distance'] (from AddHeteroHopDistanceEncoding)
            If None, no anchor-based PEs are attached. (default: None)
        pairwise_pe_attr_names: List of relative-encoding attribute names
            containing sparse (N x N) matrices for pairwise relative encodings.
            For each pair of sequence nodes (i, j), the value PE[node_i, node_j]
            is looked up and returned as attention-bias features. (default: None)

    Returns:
        (sequences, valid_mask, attention_bias_data), where:
            sequences: (batch_size, max_seq_len, feature_dim) padded node features
                taken directly from ``data[node_type].x`` in homogeneous order.
            valid_mask: (batch_size, max_seq_len) bool tensor indicating which
                sequence positions correspond to real nodes.
            attention_bias_data: dictionary of raw attention-bias features with:
                ``"anchor_bias"`` shaped ``(batch, seq, num_anchor_attrs)`` or None
                ``"pairwise_bias"`` shaped
                ``(batch, seq, seq, num_pairwise_attrs)`` or None
    """
    device = data[anchor_node_type].x.device

    # Convert to homogeneous for easier neighborhood extraction
    homo_data = data.to_homogeneous()
    homo_x = homo_data.x  # (total_nodes, feature_dim)
    homo_edge_index = homo_data.edge_index  # (2, num_edges)

    num_nodes = homo_data.num_nodes

    # Get node type to index mapping (sorted alphabetically)
    sorted_node_types = sorted(data.node_types)

    # Find offset for anchor_node_type in homogeneous graph
    # Nodes are ordered by node_type (alphabetically), then by original index
    offset = 0
    for nt in sorted_node_types:
        if nt == anchor_node_type:
            break
        offset += data[nt].num_nodes

    # Determine anchor indices in homogeneous graph
    if anchor_node_ids is not None:
        # Use provided local indices, convert to homogeneous indices
        anchor_indices = offset + anchor_node_ids.to(device)
    else:
        # Default: first batch_size nodes of anchor_node_type
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

    pairwise_pe_matrices = []
    if pairwise_pe_attr_names:
        for attr_name in pairwise_pe_attr_names:
            if hasattr(data, attr_name):
                pairwise_pe_matrices.append(getattr(data, attr_name))
            else:
                raise ValueError(
                    f"Pairwise PE attribute '{attr_name}' not found in data. "
                    f"Make sure to apply the corresponding transform first."
                )

    node_index_sequences, valid_mask = _build_sequence_layout_from_sparse_neighbors(
        reachable=reachable,
        anchor_indices=anchor_indices,
        max_seq_len=max_seq_len,
        include_anchor_first=include_anchor_first,
        device=device,
    )

    node_feature_sequences = _gather_sequences_from_node_indices(
        node_index_sequences=node_index_sequences,
        node_features=homo_x,
        valid_mask=valid_mask,
        padding_value=padding_value,
    )

    anchor_feature_sequences = _lookup_anchor_relative_features(
        anchor_indices=anchor_indices,
        node_index_sequences=node_index_sequences,
        valid_mask=valid_mask,
        csr_matrices=anchor_based_pe_matrices if anchor_based_pe_matrices else None,
        device=device,
    )

    pairwise_feature_sequences: Optional[Tensor] = None
    if pairwise_pe_matrices:
        pairwise_feature_sequences = _lookup_pairwise_relative_features(
            node_index_sequences=node_index_sequences,
            valid_mask=valid_mask,
            csr_matrices=pairwise_pe_matrices,
            device=device,
        )

    return (
        node_feature_sequences,
        valid_mask,
        {
            "anchor_bias": anchor_feature_sequences,
            "pairwise_bias": pairwise_feature_sequences,
        },
    )


def _build_sequence_layout_from_sparse_neighbors(
    reachable: Tensor,
    anchor_indices: Tensor,
    max_seq_len: int,
    include_anchor_first: bool,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Build padded node-index sequences and a validity mask from reachability data.

    Returns:
        node_index_sequences: (batch_size, max_seq_len) long tensor with -1 padding
        valid_mask: (batch_size, max_seq_len) bool tensor
    """
    batch_size = anchor_indices.size(0)
    node_index_sequences = torch.full(
        (batch_size, max_seq_len),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )
    valid_mask = torch.zeros(
        (batch_size, max_seq_len),
        dtype=torch.bool,
        device=device,
    )

    indices = reachable.indices()
    batch_idx = indices[0]
    node_idx = indices[1]

    if include_anchor_first and max_seq_len > 0:
        node_index_sequences[:, 0] = anchor_indices
        valid_mask[:, 0] = True

        if batch_idx.numel() > 0:
            keep = node_idx != anchor_indices[batch_idx]
            batch_idx = batch_idx[keep]
            node_idx = node_idx[keep]
        start_pos = 1
    else:
        start_pos = 0

    if batch_idx.numel() == 0 or start_pos >= max_seq_len:
        return node_index_sequences, valid_mask

    n = batch_idx.size(0)
    is_new_batch = torch.zeros(n, dtype=torch.long, device=device)
    is_new_batch[0] = 1
    if n > 1:
        is_new_batch[1:] = (batch_idx[1:] != batch_idx[:-1]).long()

    group_id = is_new_batch.cumsum(0) - 1
    group_starts = torch.nonzero(is_new_batch, as_tuple=True)[0]
    positions = torch.arange(n, device=device) - group_starts[group_id] + start_pos

    valid = positions < max_seq_len
    valid_batch_idx = batch_idx[valid]
    valid_positions = positions[valid]
    valid_node_idx = node_idx[valid]

    node_index_sequences[valid_batch_idx, valid_positions] = valid_node_idx
    valid_mask[valid_batch_idx, valid_positions] = True

    return node_index_sequences, valid_mask


def _gather_sequences_from_node_indices(
    node_index_sequences: Tensor,
    node_features: Tensor,
    valid_mask: Tensor,
    padding_value: float,
) -> Tensor:
    """Gather node features into padded sequences using precomputed node indices."""
    batch_size, max_seq_len = node_index_sequences.shape
    feature_dim = node_features.size(-1)

    sequences = torch.full(
        (batch_size, max_seq_len, feature_dim),
        padding_value,
        dtype=node_features.dtype,
        device=node_features.device,
    )

    if feature_dim == 0 or not valid_mask.any():
        return sequences

    sequences[valid_mask] = node_features[node_index_sequences[valid_mask]]
    return sequences


def _lookup_anchor_relative_features(
    anchor_indices: Tensor,
    node_index_sequences: Tensor,
    valid_mask: Tensor,
    csr_matrices: Optional[list[Tensor]],
    device: torch.device,
) -> Optional[Tensor]:
    """Look up anchor-relative sparse values for each valid token in the sequence."""
    if not csr_matrices:
        return None

    batch_size, max_seq_len = node_index_sequences.shape
    num_attrs = len(csr_matrices)
    features = torch.zeros(
        (batch_size, max_seq_len, num_attrs),
        dtype=torch.float,
        device=device,
    )

    if not valid_mask.any():
        return features

    valid_batch_idx, valid_pos_idx = torch.nonzero(valid_mask, as_tuple=True)
    valid_node_idx = node_index_sequences[valid_mask]
    anchor_for_entry = anchor_indices[valid_batch_idx]

    for attr_idx, pe_matrix in enumerate(csr_matrices):
        pe_values = _lookup_csr_values(
            csr_matrix=pe_matrix,
            row_indices=anchor_for_entry,
            col_indices=valid_node_idx,
        )
        features[valid_batch_idx, valid_pos_idx, attr_idx] = pe_values

    return features


def _lookup_pairwise_relative_features(
    node_index_sequences: Tensor,
    valid_mask: Tensor,
    csr_matrices: list[Tensor],
    device: torch.device,
) -> Optional[Tensor]:
    """Look up pairwise sparse values for each valid token pair in the sequence."""
    if not csr_matrices:
        return None

    batch_size, max_seq_len = node_index_sequences.shape
    num_attrs = len(csr_matrices)
    features = torch.zeros(
        (batch_size, max_seq_len, max_seq_len, num_attrs),
        dtype=torch.float,
        device=device,
    )

    pair_valid_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    if not pair_valid_mask.any():
        return features

    row_indices = node_index_sequences.unsqueeze(2).expand(-1, -1, max_seq_len)
    col_indices = node_index_sequences.unsqueeze(1).expand(-1, max_seq_len, -1)

    valid_row_indices = row_indices[pair_valid_mask]
    valid_col_indices = col_indices[pair_valid_mask]

    for attr_idx, pe_matrix in enumerate(csr_matrices):
        pe_values = _lookup_csr_values(
            csr_matrix=pe_matrix,
            row_indices=valid_row_indices,
            col_indices=valid_col_indices,
        )
        features[..., attr_idx][pair_valid_mask] = pe_values

    return features


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
        torch.stack(
            [
                torch.arange(batch_size, device=device),
                anchor_indices,
            ]
        ),
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


class HeteroToGraphTransformerInput:
    """Callable wrapper around ``heterodata_to_graph_transformer_input``."""

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        anchor_node_type: NodeType,
        feature_dim: Optional[int] = None,
        hop_distance: int = 2,
        include_anchor_first: bool = True,
        padding_value: float = 0.0,
        anchor_based_pe_attr_names: Optional[list[str]] = None,
        pairwise_pe_attr_names: Optional[list[str]] = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.anchor_node_type = anchor_node_type
        self.feature_dim = feature_dim
        self.hop_distance = hop_distance
        self.include_anchor_first = include_anchor_first
        self.padding_value = padding_value
        self.anchor_based_pe_attr_names = anchor_based_pe_attr_names
        self.pairwise_pe_attr_names = pairwise_pe_attr_names

    def __call__(self, data: HeteroData):
        return heterodata_to_graph_transformer_input(
            data=data,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            anchor_node_type=self.anchor_node_type,
            feature_dim=self.feature_dim,
            hop_distance=self.hop_distance,
            include_anchor_first=self.include_anchor_first,
            padding_value=self.padding_value,
            anchor_based_pe_attr_names=self.anchor_based_pe_attr_names,
            pairwise_pe_attr_names=self.pairwise_pe_attr_names,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"batch_size={self.batch_size}, "
            f"max_seq_len={self.max_seq_len}, "
            f"anchor_node_type={self.anchor_node_type!r}, "
            f"hop_distance={self.hop_distance})"
        )
