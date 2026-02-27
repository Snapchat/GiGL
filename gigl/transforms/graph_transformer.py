"""
Transform HeteroData to Graph Transformer sequence input.

TODO: Doesn't support multiple node types with different feature dim yet.
TODO: Doesn't support relative encoding used for attention bias yet.

This module provides functionality to convert PyG HeteroData objects (typically
batched 2-hop subgraphs) into sequence format suitable for Graph Transformers.

For each anchor node in the batch, the transform extracts its k-hop neighborhood
and creates a fixed-length sequence of node features with padding.

Example Usage:
    >>> from torch_geometric.data import HeteroData
    >>> from gigl.transforms.graph_transformer import HeteroToGraphTransformerInput
    >>>
    >>> # Create batched HeteroData (e.g., from NeighborLoader)
    >>> # First batch_size nodes in each node type are anchor nodes
    >>> data = HeteroData()
    >>> data['user'].x = torch.randn(100, 64)  # 100 users, first N are anchors
    >>> data['item'].x = torch.randn(50, 32)
    >>> data['user', 'buys', 'item'].edge_index = ...
    >>>
    >>> # Transform to Graph Transformer input
    >>> transform = HeteroToGraphTransformerInput(
    ...     batch_size=32,
    ...     max_seq_len=128,
    ...     anchor_node_type='user',
    ... )
    >>> sequences, attention_mask = transform(data)
    >>> # sequences: (batch_size, max_seq_len, feature_dim)
    >>> # attention_mask: (batch_size, max_seq_len) - 1 for valid, 0 for padding

    Using with PyTorch TransformerEncoderLayer:
    >>> import torch.nn as nn
    >>>
    >>> feature_dim = 64
    >>> encoder_layer = nn.TransformerEncoderLayer(
    ...     d_model=feature_dim,
    ...     nhead=8,
    ...     dim_feedforward=256,
    ...     batch_first=True,
    ... )
    >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    >>>
    >>> # Transform data and pass to transformer
    >>> sequences, attention_mask = transform(data)
    >>> # Convert attention_mask to key_padding_mask (True = ignore)
    >>> key_padding_mask = (attention_mask == 0)
    >>> output = transformer_encoder(sequences, src_key_padding_mask=key_padding_mask)
    >>> # output: (batch_size, max_seq_len, feature_dim)

    Handling Multiple Node Types with Different Feature Dimensions:
    When node types have different feature dimensions, project them to a common
    dimension BEFORE calling to_homogeneous(). This ensures all nodes have the
    same feature dimension when converted to sequences.

    >>> import torch.nn as nn
    >>> from torch_geometric.data import HeteroData
    >>>
    >>> # Define per-node-type linear projections
    >>> class NodeTypeProjector(nn.Module):
    ...     def __init__(self, input_dims: dict, output_dim: int):
    ...         super().__init__()
    ...         self.projectors = nn.ModuleDict({
    ...             node_type: nn.Linear(in_dim, output_dim)
    ...             for node_type, in_dim in input_dims.items()
    ...         })
    ...
    ...     def forward(self, data: HeteroData) -> HeteroData:
    ...         # Project each node type's features to common dimension
    ...         for node_type, projector in self.projectors.items():
    ...             if node_type in data.node_types:
    ...                 data[node_type].x = projector(data[node_type].x)
    ...         return data
    >>>
    >>> # Example usage
    >>> input_dims = {'user': 128, 'item': 64, 'category': 32}
    >>> common_dim = 256
    >>> projector = NodeTypeProjector(input_dims, common_dim)
    >>>
    >>> # In your forward pass:
    >>> # 1. Project node features to common dimension
    >>> data = projector(data)
    >>> # 2. Now all node types have same feature dim, safe to transform
    >>> sequences, attention_mask = transform(data)
    >>> # 3. Pass to transformer
    >>> output = transformer_encoder(sequences, src_key_padding_mask=(attention_mask == 0))
"""

from typing import Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import to_torch_sparse_tensor


def hetero_to_graph_transformer_input(
    data: HeteroData,
    batch_size: int,
    max_seq_len: int,
    anchor_node_type: NodeType,
    feature_dim: Optional[int] = None,
    hop_distance: int = 2,
    include_anchor_first: bool = True,
    padding_value: float = 0.0,
) -> Tuple[Tensor, Tensor]:
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

    Returns:
        Tuple of:
            - sequences: (batch_size, max_seq_len, feature_dim) padded node features
            - attention_mask: (batch_size, max_seq_len) binary mask (1=valid, 0=padding)
    """
    device = data[anchor_node_type].x.device

    # Convert to homogeneous for easier neighborhood extraction
    homo_data = data.to_homogeneous()
    homo_x = homo_data.x  # (total_nodes, feature_dim)
    homo_edge_index = homo_data.edge_index  # (2, num_edges)

    num_nodes = homo_data.num_nodes

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

    # Build sequences for each anchor using the sparse reachable mask
    sequences, attention_mask = _build_sequences_from_sparse_neighbors(
        reachable=reachable,
        anchor_indices=anchor_indices,
        node_features=homo_x,
        max_seq_len=max_seq_len,
        include_anchor_first=include_anchor_first,
        padding_value=padding_value,
        device=device,
    )

    return sequences, attention_mask


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
) -> Tuple[Tensor, Tensor]:
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

    Returns:
        Tuple of:
            - sequences: (batch_size, max_seq_len, feature_dim)
            - attention_mask: (batch_size, max_seq_len)
    """
    batch_size = anchor_indices.size(0)
    feature_dim = node_features.size(1)

    # Initialize outputs
    sequences = torch.full(
        (batch_size, max_seq_len, feature_dim),
        padding_value,
        dtype=node_features.dtype,
        device=device,
    )
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float, device=device)

    # Extract sparse indices
    indices = reachable.indices()  # (2, nnz)
    batch_idx = indices[0]
    node_idx = indices[1]

    # For each batch, grab reachable node features
    for b in range(batch_size):
        # Get reachable nodes for this batch
        mask = batch_idx == b
        nodes = node_idx[mask]

        if include_anchor_first:
            # Put anchor first
            anchor = anchor_indices[b]
            other_nodes = nodes[nodes != anchor]
            nodes = torch.cat([anchor.unsqueeze(0), other_nodes])

        # Truncate to max_seq_len and place features
        n = min(nodes.size(0), max_seq_len)
        sequences[b, :n] = node_features[nodes[:n]]
        attention_mask[b, :n] = 1.0

    return sequences, attention_mask


class HeteroToGraphTransformerInput:
    """
    Transform class for converting HeteroData to Graph Transformer input.

    This class wraps the `hetero_to_graph_transformer_input` function and can
    be used as a callable transform in data pipelines.

    Args:
        batch_size: Number of anchor nodes per batch.
        max_seq_len: Maximum sequence length for transformer input.
        anchor_node_type: The node type of anchor nodes.
        feature_dim: Output feature dimension (optional).
        hop_distance: Number of hops to consider (default: 2).
        include_anchor_first: Put anchor node first in sequence (default: True).
        padding_value: Value for padding (default: 0.0).

    Example:
        >>> transform = HeteroToGraphTransformerInput(
        ...     batch_size=32,
        ...     max_seq_len=128,
        ...     anchor_node_type='user',
        ... )
        >>> sequences, attention_mask = transform(data)
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        anchor_node_type: NodeType,
        feature_dim: Optional[int] = None,
        hop_distance: int = 2,
        include_anchor_first: bool = True,
        padding_value: float = 0.0,
    ) -> None:
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.anchor_node_type = anchor_node_type
        self.feature_dim = feature_dim
        self.hop_distance = hop_distance
        self.include_anchor_first = include_anchor_first
        self.padding_value = padding_value

    def __call__(
        self, data: HeteroData
    ) -> Tuple[Tensor, Tensor]:
        """Transform HeteroData to Graph Transformer input."""
        return hetero_to_graph_transformer_input(
            data=data,
            batch_size=self.batch_size,
            max_seq_len=self.max_seq_len,
            anchor_node_type=self.anchor_node_type,
            feature_dim=self.feature_dim,
            hop_distance=self.hop_distance,
            include_anchor_first=self.include_anchor_first,
            padding_value=self.padding_value,
        )

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'batch_size={self.batch_size}, '
            f'max_seq_len={self.max_seq_len}, '
            f'anchor_node_type={self.anchor_node_type!r}, '
            f'hop_distance={self.hop_distance})'
        )
