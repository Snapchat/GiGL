"""
Transform HeteroData to Graph Transformer sequence input.

This module provides functionality to convert PyG HeteroData objects (typically
batched 2-hop subgraphs) into sequence format suitable for Graph Transformers.

For each anchor node in the batch, the transform extracts its k-hop neighborhood
and creates a fixed-length sequence of node features with padding.

Example Usage:
    >>> from torch_geometric.data import HeteroData
    >>> from gigl.transforms.hetero_to_graph_transformer import HeteroToGraphTransformerInput
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
    >>> sequences, attention_mask, anchor_positions = transform(data)
    >>> # sequences: (batch_size, max_seq_len, feature_dim)
    >>> # attention_mask: (batch_size, max_seq_len) - 1 for valid, 0 for padding
    >>> # anchor_positions: (batch_size,) - position of anchor in each sequence
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
) -> Tuple[Tensor, Tensor, Tensor]:
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
            - neighbor_counts: (batch_size,) number of neighbors per anchor (before padding)
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
    # Returns: (batch_size, num_nodes) sparse matrix where non-zero entries are neighbors
    neighbor_mask, hop_distances = _get_k_hop_neighbors_sparse(
        anchor_indices=anchor_indices,
        edge_index=homo_edge_index,
        num_nodes=num_nodes,
        k=hop_distance,
        device=device,
    )

    # Build sequences for each anchor using the sparse neighbor mask
    sequences, attention_mask, neighbor_counts = _build_sequences_from_sparse_neighbors(
        neighbor_mask=neighbor_mask,
        hop_distances=hop_distances,
        anchor_indices=anchor_indices,
        node_features=homo_x,
        max_seq_len=max_seq_len,
        include_anchor_first=include_anchor_first,
        padding_value=padding_value,
        device=device,
    )

    return sequences, attention_mask, neighbor_counts


def _get_k_hop_neighbors_sparse(
    anchor_indices: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    k: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Get k-hop neighbors for all anchors using sparse matrix multiplication.

    This is much more efficient than per-node BFS for batched operations.
    Uses fully sparse operations to avoid memory blowup.

    Args:
        anchor_indices: (batch_size,) anchor node indices
        edge_index: (2, num_edges) edge index
        num_nodes: Total number of nodes
        k: Number of hops
        device: Device for tensors

    Returns:
        Tuple of:
            - neighbor_mask: (batch_size, num_nodes) sparse bool tensor, True if node is neighbor
            - hop_distances: (batch_size, num_nodes) sparse tensor with hop distance (0 = not reachable)
    """
    batch_size = anchor_indices.size(0)

    # Build sparse adjacency matrix (binarized)
    adj = to_torch_sparse_tensor(edge_index, size=(num_nodes, num_nodes))
    adj = adj.coalesce()
    # Binarize adjacency
    adj = torch.sparse_coo_tensor(
        adj.indices(),
        torch.ones(adj.indices().size(1), device=device, dtype=torch.float),
        size=(num_nodes, num_nodes),
    ).coalesce()

    # Initialize frontier as sparse: each anchor can reach itself
    # frontier[i, j] = 1.0 if node j is in frontier for anchor i
    frontier_indices = torch.stack([
        torch.arange(batch_size, device=device),
        anchor_indices,
    ])
    frontier = torch.sparse_coo_tensor(
        frontier_indices,
        torch.ones(batch_size, device=device, dtype=torch.float),
        size=(batch_size, num_nodes),
    ).coalesce()

    # Track all reachable nodes and their hop distances using lists
    # We'll build the final sparse tensors at the end
    all_batch_indices = [torch.arange(batch_size, device=device)]  # hop 0: anchors
    all_node_indices = [anchor_indices.clone()]
    all_hop_values = [torch.zeros(batch_size, device=device, dtype=torch.long)]

    # Previous reachable set (sparse)
    reachable = frontier.clone()

    for hop in range(1, k + 1):
        # Expand frontier: frontier @ adj (sparse matmul)
        # frontier: (batch_size, num_nodes), adj: (num_nodes, num_nodes)
        next_frontier = torch.sparse.mm(frontier, adj).coalesce()

        if next_frontier._nnz() == 0:
            break

        # Get indices of nodes in next frontier
        next_indices = next_frontier.indices()  # (2, nnz)
        next_batch = next_indices[0]  # which anchor
        next_nodes = next_indices[1]  # which node

        # Find newly reachable: in next_frontier but not in reachable
        # Convert reachable to set of (batch, node) tuples for fast lookup
        reachable_indices = reachable.indices()
        reachable_set = set(zip(
            reachable_indices[0].tolist(),
            reachable_indices[1].tolist()
        ))

        # Filter to only newly reachable
        new_mask = torch.tensor([
            (b.item(), n.item()) not in reachable_set
            for b, n in zip(next_batch, next_nodes)
        ], device=device, dtype=torch.bool)

        if not new_mask.any():
            break

        new_batch = next_batch[new_mask]
        new_nodes = next_nodes[new_mask]

        # Store newly reachable nodes and their hop distances
        all_batch_indices.append(new_batch)
        all_node_indices.append(new_nodes)
        all_hop_values.append(torch.full((new_batch.size(0),), hop, device=device, dtype=torch.long))

        # Update reachable set
        new_frontier_indices = torch.stack([new_batch, new_nodes])
        new_entries = torch.sparse_coo_tensor(
            new_frontier_indices,
            torch.ones(new_batch.size(0), device=device, dtype=torch.float),
            size=(batch_size, num_nodes),
        )
        reachable = (reachable + new_entries).coalesce()
        # Binarize (in case of duplicates)
        reachable = torch.sparse_coo_tensor(
            reachable.indices(),
            torch.ones(reachable._nnz(), device=device, dtype=torch.float),
            size=(batch_size, num_nodes),
        ).coalesce()

        # Update frontier to only newly reachable for next iteration
        frontier = torch.sparse_coo_tensor(
            new_frontier_indices,
            torch.ones(new_batch.size(0), device=device, dtype=torch.float),
            size=(batch_size, num_nodes),
        ).coalesce()

    # Build final sparse tensors
    all_batch = torch.cat(all_batch_indices)
    all_nodes = torch.cat(all_node_indices)
    all_hops = torch.cat(all_hop_values)

    # neighbor_mask: sparse bool-like tensor (values are 1.0 for reachable)
    neighbor_indices = torch.stack([all_batch, all_nodes])
    neighbor_mask = torch.sparse_coo_tensor(
        neighbor_indices,
        torch.ones(all_batch.size(0), device=device, dtype=torch.bool),
        size=(batch_size, num_nodes),
    ).coalesce()

    # hop_distances: sparse tensor with hop values
    hop_distances = torch.sparse_coo_tensor(
        neighbor_indices,
        all_hops,
        size=(batch_size, num_nodes),
    ).coalesce()

    return neighbor_mask, hop_distances


def _build_sequences_from_sparse_neighbors(
    neighbor_mask: Tensor,
    hop_distances: Tensor,
    anchor_indices: Tensor,
    node_features: Tensor,
    max_seq_len: int,
    include_anchor_first: bool,
    padding_value: float,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Build padded sequences from sparse neighbor information.

    Args:
        neighbor_mask: (batch_size, num_nodes) sparse bool tensor
        hop_distances: (batch_size, num_nodes) sparse hop distance tensor
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
            - neighbor_counts: (batch_size,)
    """
    batch_size = anchor_indices.size(0)
    feature_dim = node_features.size(1)

    # Initialize output tensors
    sequences = torch.full(
        (batch_size, max_seq_len, feature_dim),
        padding_value,
        dtype=node_features.dtype,
        device=device,
    )
    attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float, device=device)
    neighbor_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

    # Extract indices from sparse tensors
    mask_indices = neighbor_mask.indices()  # (2, nnz)
    mask_batch = mask_indices[0]
    mask_nodes = mask_indices[1]

    dist_indices = hop_distances.indices()  # (2, nnz)
    dist_batch = dist_indices[0]
    dist_nodes = dist_indices[1]
    dist_values = hop_distances.values()

    for i in range(batch_size):
        anchor_idx = int(anchor_indices[i].item())

        # Get neighbor indices for this anchor from sparse tensor
        batch_mask = mask_batch == i
        neighbor_idx = mask_nodes[batch_mask]

        if neighbor_idx.numel() == 0:
            # No neighbors, just anchor
            if include_anchor_first:
                sequences[i, 0] = node_features[anchor_idx]
                attention_mask[i, 0] = 1.0
                neighbor_counts[i] = 1
            continue

        # Get hop distances for these neighbors
        dist_batch_mask = dist_batch == i
        dist_nodes_for_batch = dist_nodes[dist_batch_mask]
        dist_vals_for_batch = dist_values[dist_batch_mask]

        # Create mapping from node to distance
        node_to_dist = {n.item(): d.item() for n, d in zip(dist_nodes_for_batch, dist_vals_for_batch)}

        # Sort neighbors by hop distance (closer nodes first)
        neighbor_distances = torch.tensor(
            [node_to_dist.get(n.item(), 0) for n in neighbor_idx],
            device=device,
            dtype=torch.long
        )
        sorted_order = torch.argsort(neighbor_distances)
        sorted_neighbors = neighbor_idx[sorted_order]

        if include_anchor_first:
            # Remove anchor from sorted list if present (it has distance 0)
            is_anchor = sorted_neighbors == anchor_idx
            if is_anchor.any():
                sorted_neighbors = sorted_neighbors[~is_anchor]
            # Prepend anchor
            sorted_neighbors = torch.cat([
                torch.tensor([anchor_idx], device=device, dtype=torch.long),
                sorted_neighbors
            ])

        # Truncate to max_seq_len
        seq_len = min(sorted_neighbors.size(0), max_seq_len)
        sorted_neighbors = sorted_neighbors[:seq_len]

        # Fill sequence
        sequences[i, :seq_len] = node_features[sorted_neighbors]
        attention_mask[i, :seq_len] = 1.0
        neighbor_counts[i] = seq_len

    return sequences, attention_mask, neighbor_counts


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
        >>> sequences, mask, counts = transform(data)
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
    ) -> Tuple[Tensor, Tensor, Tensor]:
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
