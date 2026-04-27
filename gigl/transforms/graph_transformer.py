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
    ...     AddHeteroRandomWalkEncodings,
    ...     AddHeteroHopDistanceEncoding,
    ... )
    >>>
    >>> # First apply PE transforms to the data
    >>> pe_transform = Compose([
    ...     AddHeteroRandomWalkEncodings(walk_length=8),
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
    ...     anchor_based_attention_bias_attr_names=['hop_distance'],
    ... )
    >>> # sequences: (batch_size, max_seq_len, feature_dim)
    >>> # attention_bias_data['anchor_bias']: (batch_size, max_seq_len, 1)
"""

from typing import Literal, Optional, TypedDict

import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import NodeType
from torch_geometric.utils import to_torch_sparse_tensor

from gigl.src.common.types.graph_data import EdgeType

TokenInputData = dict[str, Tensor]


class SequenceAuxiliaryData(TypedDict):
    anchor_bias: Optional[Tensor]
    pairwise_bias: Optional[Tensor]
    pairwise_relation_mask: Optional[Tensor]
    token_input: Optional[TokenInputData]


PPR_WEIGHT_FEATURE_NAME = "ppr_weight"


def heterodata_to_graph_transformer_input(
    data: HeteroData,
    batch_size: int,
    max_seq_len: int,
    anchor_node_type: NodeType,
    anchor_node_ids: Optional[Tensor] = None,
    hop_distance: int = 2,
    sequence_construction_method: Literal["khop", "ppr"] = "khop",
    include_anchor_first: bool = True,
    padding_value: float = 0.0,
    anchor_based_attention_bias_attr_names: Optional[list[str]] = None,
    anchor_based_input_attr_names: Optional[list[str]] = None,
    pairwise_attention_bias_attr_names: Optional[list[str]] = None,
    relation_edge_types: Optional[list[EdgeType]] = None,
) -> tuple[Tensor, Tensor, SequenceAuxiliaryData]:
    """
    Transform a HeteroData object to Graph Transformer sequence input.

    Given a batched HeteroData object where the first `batch_size` nodes of
    `anchor_node_type` are anchor nodes, this function extracts the k-hop
    neighborhood for each anchor and creates padded sequences.

    Uses sparse matrix operations for efficient batched k-hop neighbor extraction.

    Args:
        data: HeteroData object containing node features and edge indices.
            Expected to have node features as `data[node_type].x`.
            All node types must have the same feature dimension.
        batch_size: Number of anchor nodes (first batch_size nodes of anchor_node_type).
            Ignored if anchor_node_ids is provided.
        max_seq_len: Maximum sequence length (neighbors beyond this are truncated).
        anchor_node_type: The node type of anchor nodes.
        anchor_node_ids: Optional tensor of local node indices within anchor_node_type
            to use as anchors. If None, uses first batch_size nodes. (default: None)
        hop_distance: Number of hops to consider for neighborhood when
            ``sequence_construction_method="khop"``. (default: 2)
        sequence_construction_method: Strategy used to build per-anchor sequences.
            ``"khop"`` performs the existing k-hop expansion over the sampled graph.
            ``"ppr"`` uses outgoing ``(anchor_type, "ppr", neighbor_type)`` edges,
            sorted by descending PPR weight from ``edge_attr``. (default: ``"khop"``)
        include_anchor_first: If True, anchor node is always first in sequence.
        padding_value: Value to use for padding (default: 0.0).
        anchor_based_attention_bias_attr_names: List of anchor-relative feature
            names used as attention bias. Sparse graph-level attributes are
            looked up from ``data`` and the reserved name ``"ppr_weight"``
            resolves to PPR edge weights in PPR sequence mode.
            Example: ['hop_distance', 'ppr_weight'].
        anchor_based_input_attr_names: List of anchor-relative attribute names
            returned as token-aligned model-input features. Sparse graph-level
            attributes are looked up from ``data`` and ``"ppr_weight"`` resolves
            to PPR edge weights in PPR sequence mode.
            Example: ['hop_distance', 'ppr_weight'].
        pairwise_attention_bias_attr_names: List of pairwise feature names used
            as attention bias. These must correspond to sparse graph-level
            attributes on ``data``. Example: ['pairwise_distance'].
        relation_edge_types: Optional ordered edge types used to materialize a
            dense per-token-pair relation mask. Each output channel corresponds
            to one edge type in this list. Directed edges are placed at
            ``[query_pos, key_pos] = [dst_token, src_token]``.

    Returns:
        (sequences, valid_mask, attention_bias_data), where:
            sequences: (batch_size, max_seq_len, feature_dim) padded node features
                taken directly from ``data[node_type].x`` in homogeneous order.
            valid_mask: (batch_size, max_seq_len) bool tensor indicating which
                sequence positions correspond to real nodes.
            sequence_auxiliary_data: dictionary of raw token-aligned and
                attention-bias features with:
                ``"anchor_bias"`` shaped ``(batch, seq, num_anchor_attrs)`` or None
                ``"pairwise_bias"`` shaped
                ``(batch, seq, seq, num_pairwise_attrs)`` or None
                ``"pairwise_relation_mask"`` shaped
                ``(batch, seq, seq, num_relations)`` or None
                ``"token_input"`` as a dict mapping attribute name to a
                ``(batch, seq, 1)`` tensor, or None

    Raises:
        ValueError: If node types have different feature dimensions.
        ValueError: If no node features exist in the data.
    """
    # Check each node type for valid features
    feature_dims: dict[str, int] = {}
    for nt in data.node_types:
        if not hasattr(data[nt], "x") or data[nt].x is None:
            raise ValueError(
                f"Node type '{nt}' has no features (x is None or missing). "
                "Graph Transformer input requires node features for all node types."
            )
        if data[nt].x.dim() < 2:
            raise ValueError(
                f"Node type '{nt}' has invalid feature shape {data[nt].x.shape}. "
                "Expected 2D tensor (num_nodes, feature_dim)."
            )
        feat_dim = data[nt].x.size(1)
        if feat_dim == 0:
            raise ValueError(
                f"Node type '{nt}' has zero feature dimension (shape {data[nt].x.shape}). "
                "Graph Transformer input requires non-zero feature dimension."
            )
        feature_dims[nt] = feat_dim

    # Check all node types have the same feature dimension
    unique_dims = set(feature_dims.values())
    if len(unique_dims) > 1:
        raise ValueError(
            f"All node types must have the same feature dimension for Graph Transformer input. "
            f"Please project features to equal dimensions. "
            f"Found different dimensions: {feature_dims}"
        )

    anchor_bias_attr_names = anchor_based_attention_bias_attr_names or []
    anchor_input_attr_names = anchor_based_input_attr_names or []
    pairwise_bias_attr_names = pairwise_attention_bias_attr_names or []

    if PPR_WEIGHT_FEATURE_NAME in pairwise_bias_attr_names:
        raise ValueError(
            f"'{PPR_WEIGHT_FEATURE_NAME}' is an anchor-relative feature and cannot "
            "be used as pairwise attention bias."
        )

    if (
        PPR_WEIGHT_FEATURE_NAME in anchor_bias_attr_names + anchor_input_attr_names
        and sequence_construction_method != "ppr"
    ):
        raise ValueError(
            "The reserved anchor-relative feature 'ppr_weight' requires "
            "sequence_construction_method='ppr'."
        )

    if sequence_construction_method == "ppr":
        _validate_ppr_sequence_input(data)

    device = data[anchor_node_type].x.device

    # Convert to homogeneous for easier neighborhood extraction
    homo_data = data.to_homogeneous()
    homo_x = homo_data.x  # (total_nodes, feature_dim)

    num_nodes = homo_data.num_nodes

    # Match the node-type ordering used by to_homogeneous() so homogeneous
    # indices line up with homo_x / homo_edge_index.
    node_type_order = list(getattr(homo_data, "_node_type_names", data.node_types))
    node_type_offsets = _get_node_type_offsets(
        data=data, node_type_order=node_type_order
    )

    # Find offset for anchor_node_type in homogeneous graph
    # Nodes are ordered by the homogeneous node-type order, then by original index.
    offset = node_type_offsets[anchor_node_type]

    # Determine anchor indices in homogeneous graph
    if anchor_node_ids is not None:
        # Use provided local indices, convert to homogeneous indices
        anchor_local_indices = anchor_node_ids.to(device)
    else:
        # Default: first batch_size nodes of anchor_node_type
        anchor_local_indices = torch.arange(batch_size, device=device)
    anchor_indices = offset + anchor_local_indices

    ppr_weight_sequences: Optional[Tensor] = None
    if sequence_construction_method == "khop":
        homo_edge_index = homo_data.edge_index  # (2, num_edges)
        # Use sparse matrix operations for efficient k-hop neighbor extraction
        # Returns: (batch_size, num_nodes) sparse matrix where non-zero entries are reachable
        reachable = _get_k_hop_neighbors_sparse(
            anchor_indices=anchor_indices,
            edge_index=homo_edge_index,
            num_nodes=num_nodes,
            k=hop_distance,
            device=device,
        )
        node_index_sequences, valid_mask = _build_sequence_layout_from_sparse_neighbors(
            reachable=reachable,
            anchor_indices=anchor_indices,
            max_seq_len=max_seq_len,
            include_anchor_first=include_anchor_first,
            device=device,
        )
    elif sequence_construction_method == "ppr":
        (
            node_index_sequences,
            valid_mask,
            ppr_weight_sequences,
        ) = _build_sequence_layout_from_ppr_edges(
            homo_data=homo_data,
            anchor_indices=anchor_indices,
            max_seq_len=max_seq_len,
            include_anchor_first=include_anchor_first,
            num_nodes=num_nodes,
            device=device,
            return_edge_weights=(
                PPR_WEIGHT_FEATURE_NAME
                in anchor_bias_attr_names + anchor_input_attr_names
            ),
        )
    else:
        raise ValueError(
            "sequence_construction_method must be one of ['khop', 'ppr'], "
            f"got '{sequence_construction_method}'."
        )

    anchor_matrix_attr_names = list(
        {
            attr_name
            for attr_name in (anchor_bias_attr_names + anchor_input_attr_names)
            if attr_name != PPR_WEIGHT_FEATURE_NAME
        }
    )
    anchor_based_matrices = _get_sparse_feature_matrices(
        data=data,
        attr_names=anchor_matrix_attr_names,
        missing_attr_error_prefix="Anchor-based attribute",
    )

    pairwise_pe_matrices = _get_sparse_feature_matrices(
        data=data,
        attr_names=pairwise_bias_attr_names,
        missing_attr_error_prefix="Pairwise PE attribute",
    )

    node_feature_sequences = _gather_sequences_from_node_indices(
        node_index_sequences=node_index_sequences,
        node_features=homo_x,
        valid_mask=valid_mask,
        padding_value=padding_value,
    )

    anchor_relative_feature_sequences = _lookup_anchor_relative_features(
        anchor_indices=anchor_indices,
        node_index_sequences=node_index_sequences,
        valid_mask=valid_mask,
        csr_matrices=anchor_based_matrices if anchor_based_matrices else None,
        device=device,
    )

    pairwise_feature_sequences = _lookup_pairwise_relative_features(
        node_index_sequences=node_index_sequences,
        valid_mask=valid_mask,
        csr_matrices=pairwise_pe_matrices if pairwise_pe_matrices else None,
        device=device,
    )
    pairwise_relation_mask = _lookup_pairwise_relation_masks(
        data=data,
        node_index_sequences=node_index_sequences,
        valid_mask=valid_mask,
        relation_edge_types=relation_edge_types,
        node_type_offsets=node_type_offsets,
        num_nodes=num_nodes,
        device=device,
    )

    anchor_bias_features = _compose_anchor_feature_tensor(
        anchor_relative_feature_sequences=anchor_relative_feature_sequences,
        available_anchor_attr_names=anchor_matrix_attr_names,
        requested_anchor_attr_names=anchor_bias_attr_names,
        ppr_weight_sequences=ppr_weight_sequences,
    )
    token_input_features = _compose_anchor_feature_dict(
        anchor_relative_feature_sequences=anchor_relative_feature_sequences,
        available_anchor_attr_names=anchor_matrix_attr_names,
        requested_anchor_attr_names=anchor_input_attr_names,
        ppr_weight_sequences=ppr_weight_sequences,
    )

    return (
        node_feature_sequences,
        valid_mask,
        {
            "anchor_bias": anchor_bias_features,
            "pairwise_bias": pairwise_feature_sequences,
            "pairwise_relation_mask": pairwise_relation_mask,
            "token_input": token_input_features,
        },
    )


def _get_node_type_offsets(
    data: HeteroData,
    node_type_order: list[NodeType],
) -> dict[NodeType, int]:
    offsets: dict[NodeType, int] = {}
    offset = 0
    for node_type in node_type_order:
        offsets[node_type] = offset
        offset += data[node_type].num_nodes
    return offsets


def _validate_ppr_sequence_input(data: HeteroData) -> None:
    if not data.edge_types:
        raise ValueError(
            "sequence_construction_method='ppr' requires at least one PPR edge type."
        )

    if any(edge_type[1] != "ppr" for edge_type in data.edge_types):
        raise ValueError(
            "sequence_construction_method='ppr' expects the hetero batch to contain "
            f"only PPR edges, got edge types: {data.edge_types}."
        )

    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        if not hasattr(edge_store, "edge_attr") or edge_store.edge_attr is None:
            raise ValueError(
                "sequence_construction_method='ppr' requires every PPR edge type to "
                f"have edge_attr weights, but {edge_type} is missing them."
            )


def _get_sparse_feature_matrices(
    data: HeteroData,
    attr_names: Optional[list[str]],
    missing_attr_error_prefix: str,
) -> list[Tensor]:
    matrices: list[Tensor] = []
    for attr_name in attr_names or []:
        if not hasattr(data, attr_name):
            raise ValueError(
                f"{missing_attr_error_prefix} '{attr_name}' not found in data. "
                "Make sure to apply the corresponding transform first."
            )
        matrices.append(getattr(data, attr_name))
    return matrices


def _compose_anchor_feature_tensor(
    anchor_relative_feature_sequences: Optional[Tensor],
    available_anchor_attr_names: list[str],
    requested_anchor_attr_names: list[str],
    ppr_weight_sequences: Optional[Tensor],
) -> Optional[Tensor]:
    if not requested_anchor_attr_names:
        return None

    feature_parts: list[Tensor] = []
    feature_index_by_name = {
        attr_name: idx for idx, attr_name in enumerate(available_anchor_attr_names)
    }

    for attr_name in requested_anchor_attr_names:
        if attr_name == PPR_WEIGHT_FEATURE_NAME:
            if ppr_weight_sequences is None:
                raise ValueError(
                    f"Requested '{PPR_WEIGHT_FEATURE_NAME}' but it was not computed."
                )
            feature_parts.append(ppr_weight_sequences)
            continue

        if anchor_relative_feature_sequences is None:
            raise ValueError(
                "Anchor-relative features were requested but not computed."
            )
        if attr_name not in feature_index_by_name:
            raise ValueError(
                f"Anchor-relative feature '{attr_name}' was requested but not found."
            )
        feature_idx = feature_index_by_name[attr_name]
        feature_parts.append(
            anchor_relative_feature_sequences[..., feature_idx : feature_idx + 1]
        )

    return torch.cat(feature_parts, dim=-1)


def _compose_anchor_feature_dict(
    anchor_relative_feature_sequences: Optional[Tensor],
    available_anchor_attr_names: list[str],
    requested_anchor_attr_names: list[str],
    ppr_weight_sequences: Optional[Tensor],
) -> Optional[TokenInputData]:
    if not requested_anchor_attr_names:
        return None

    feature_dict: TokenInputData = {}
    feature_index_by_name = {
        attr_name: idx for idx, attr_name in enumerate(available_anchor_attr_names)
    }

    for attr_name in requested_anchor_attr_names:
        if attr_name == PPR_WEIGHT_FEATURE_NAME:
            if ppr_weight_sequences is None:
                raise ValueError(
                    f"Requested '{PPR_WEIGHT_FEATURE_NAME}' but it was not computed."
                )
            feature_dict[attr_name] = ppr_weight_sequences
            continue

        if anchor_relative_feature_sequences is None:
            raise ValueError(
                "Anchor-relative features were requested but not computed."
            )
        if attr_name not in feature_index_by_name:
            raise ValueError(
                f"Anchor-relative feature '{attr_name}' was requested but not found."
            )
        feature_idx = feature_index_by_name[attr_name]
        feature_dict[attr_name] = anchor_relative_feature_sequences[
            ..., feature_idx : feature_idx + 1
        ]

    return feature_dict


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

    # Compute within-anchor sequence positions for each neighbor node.
    #
    # After extracting from the sparse tensor, we have flattened arrays:
    #   batch_idx = [0, 0, 0, 1, 1, 2, 2, 2, 2]  <- which anchor each node belongs to
    #   node_idx  = [5, 7, 9, 3, 8, 1, 4, 6, 10] <- the reachable neighbor node IDs
    #
    # We need to compute sequence positions (1, 2, 3, ...) for each anchor's neighbors:
    #   positions = [1, 2, 3, 1, 2, 1, 2, 3, 4]
    #                ^anchor0^  ^a1^  ^--anchor2--^
    #
    # This allows scattering into the 2D output:
    #   node_index_sequences[anchor_idx, position] = node_idx

    n = batch_idx.size(0)

    # Step 1: Mark where each anchor's neighbor group starts
    # is_group_start = [1, 0, 0, 1, 0, 1, 0, 0, 0]
    #                   ^        ^     ^
    #                anchor0  anchor1  anchor2 starts here
    is_group_start = torch.zeros(n, dtype=torch.long, device=device)
    is_group_start[0] = 1
    if n > 1:
        is_group_start[1:] = (batch_idx[1:] != batch_idx[:-1]).long()

    # Step 2: Assign each node to its anchor group (0-indexed)
    # group_id = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    group_id = is_group_start.cumsum(0) - 1

    # Step 3: Find the starting index of each group in the flattened array
    # group_starts = [0, 3, 5]  (indices where each anchor's neighbors begin)
    group_starts = torch.nonzero(is_group_start, as_tuple=True)[0]

    # Step 4: Compute position = (global_index - group_start) + start_pos
    # This gives within-group position offset by start_pos (usually 1 for anchor at pos 0)
    # positions = [1, 2, 3, 1, 2, 1, 2, 3, 4]
    positions = torch.arange(n, device=device) - group_starts[group_id] + start_pos

    # Step 5: Filter out positions that exceed max_seq_len (truncation)
    valid = positions < max_seq_len
    valid_batch_idx = batch_idx[valid]
    valid_positions = positions[valid]
    valid_node_idx = node_idx[valid]

    # Step 6: Scatter valid nodes into the output tensors
    node_index_sequences[valid_batch_idx, valid_positions] = valid_node_idx
    valid_mask[valid_batch_idx, valid_positions] = True

    return node_index_sequences, valid_mask


def _build_sequence_layout_from_ppr_edges(
    homo_data: Data,
    anchor_indices: Tensor,
    max_seq_len: int,
    include_anchor_first: bool,
    num_nodes: int,
    device: torch.device,
    return_edge_weights: bool = False,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """Build sequences directly from outgoing PPR edges for each anchor.

    The sequence order is:
    1. Anchor node first, when ``include_anchor_first`` is True.
    2. Destination nodes reachable by outgoing ``"ppr"`` edges from that anchor,
       sorted by descending PPR weight.
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
    ppr_weight_sequences = None
    if return_edge_weights:
        ppr_weight_sequences = torch.zeros(
            (batch_size, max_seq_len, 1),
            dtype=torch.float,
            device=device,
        )

    if include_anchor_first and max_seq_len > 0:
        node_index_sequences[:, 0] = anchor_indices
        valid_mask[:, 0] = True
        start_pos = 1
    else:
        start_pos = 0

    if start_pos >= max_seq_len:
        return node_index_sequences, valid_mask, ppr_weight_sequences

    if not hasattr(homo_data, "edge_attr") or homo_data.edge_attr is None:
        raise ValueError(
            "sequence_construction_method='ppr' requires homogeneous edge_attr weights."
        )

    edge_weights = homo_data.edge_attr
    if edge_weights.dim() == 2:
        if edge_weights.size(1) != 1:
            raise ValueError(
                "PPR edge weights must be 1D or shape [N, 1], "
                f"got {tuple(edge_weights.shape)}."
            )
        edge_weights = edge_weights.squeeze(1)
    elif edge_weights.dim() != 1:
        raise ValueError(
            "PPR edge weights must be 1D or shape [N, 1], "
            f"got {tuple(edge_weights.shape)}."
        )

    anchor_batch_index_by_homo_idx = torch.full(
        (num_nodes,),
        fill_value=-1,
        dtype=torch.long,
        device=device,
    )
    anchor_batch_index_by_homo_idx[anchor_indices] = torch.arange(
        batch_size, device=device
    )

    src_idx = homo_data.edge_index[0]
    dst_idx = homo_data.edge_index[1]
    anchor_batch_idx = anchor_batch_index_by_homo_idx[src_idx]
    keep = anchor_batch_idx >= 0
    if not keep.any():
        return node_index_sequences, valid_mask, ppr_weight_sequences

    all_anchor_batch_idx = anchor_batch_idx[keep]
    all_dst_idx = dst_idx[keep]
    all_weights = edge_weights[keep]

    if include_anchor_first:
        keep = all_dst_idx != anchor_indices[all_anchor_batch_idx]
        if not keep.any():
            return node_index_sequences, valid_mask, ppr_weight_sequences
        all_anchor_batch_idx = all_anchor_batch_idx[keep]
        all_dst_idx = all_dst_idx[keep]
        all_weights = all_weights[keep]

    # Flattened COO edges can be laid out in one pass by sorting first on weight
    # and then stably on anchor batch id, which preserves descending-weight order
    # within each anchor group without a Python loop.
    weight_order = torch.argsort(all_weights, descending=True, stable=True)
    all_anchor_batch_idx = all_anchor_batch_idx[weight_order]
    all_dst_idx = all_dst_idx[weight_order]
    all_weights = all_weights[weight_order]

    batch_order = torch.argsort(all_anchor_batch_idx, stable=True)
    sorted_batch_idx = all_anchor_batch_idx[batch_order]
    sorted_dst_idx = all_dst_idx[batch_order]
    sorted_weights = all_weights[batch_order]

    n = sorted_batch_idx.size(0)
    is_group_start = torch.zeros(n, dtype=torch.long, device=device)
    is_group_start[0] = 1
    if n > 1:
        is_group_start[1:] = (sorted_batch_idx[1:] != sorted_batch_idx[:-1]).long()

    group_id = is_group_start.cumsum(0) - 1
    group_starts = torch.nonzero(is_group_start, as_tuple=True)[0]
    positions = torch.arange(n, device=device) - group_starts[group_id] + start_pos

    valid = positions < max_seq_len
    valid_batch_idx = sorted_batch_idx[valid]
    valid_positions = positions[valid]
    valid_dst_idx = sorted_dst_idx[valid]
    valid_weights = sorted_weights[valid]

    node_index_sequences[valid_batch_idx, valid_positions] = valid_dst_idx
    valid_mask[valid_batch_idx, valid_positions] = True
    if ppr_weight_sequences is not None:
        ppr_weight_sequences[valid_batch_idx, valid_positions, 0] = (
            valid_weights.float()
        )

    return node_index_sequences, valid_mask, ppr_weight_sequences


def _gather_sequences_from_node_indices(
    node_index_sequences: Tensor,
    node_features: Tensor,
    valid_mask: Tensor,
    padding_value: float,
) -> Tensor:
    """Gather node features into padded sequences using precomputed node indices.

    Args:
        node_index_sequences: (batch_size, max_seq_len) node indices
        node_features: (num_nodes, feature_dim) node features
        valid_mask: (batch_size, max_seq_len) bool tensor indicating valid positions
        padding_value: Value to use for padding

    Returns:
        (batch_size, max_seq_len, feature_dim) padded sequences.
    """
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
    """
    Look up anchor-relative sparse values for each valid token in the sequence.

    For each node in the sequence, this looks up the value PE[anchor_idx, node_idx]
    from each provided sparse CSR matrix. This captures the relationship between
    each sequence token and its anchor node (e.g., hop distance from anchor).

    Args:
        anchor_indices: (batch_size,) anchor node indices in homogeneous graph
        node_index_sequences: (batch_size, max_seq_len) node indices for each sequence position
        valid_mask: (batch_size, max_seq_len) bool tensor indicating valid positions
        csr_matrices: List of sparse CSR matrices, each (num_nodes, num_nodes)
        device: Device for output tensor

    Returns:
        features: (batch_size, max_seq_len, num_attrs) tensor where
            features[b, i, k] = csr_matrices[k][anchor_indices[b], node_index_sequences[b, i]]
            for valid positions, 0.0 for padding positions.
        Returns None if csr_matrices is empty or None.

    Example:
        # batch_size=2, max_seq_len=4, num_attrs=1 (e.g., hop_distance)
        # anchor_indices = [10, 20]  (anchor nodes)
        # node_index_sequences = [[10, 5, 7, -1],   # anchor 10's sequence
        #                         [20, 3, 8, 9]]    # anchor 20's sequence
        # valid_mask = [[T, T, T, F], [T, T, T, T]]
        #
        # Output shape: (2, 4, 1)
        # features[0, :, 0] = [hop_dist[10,10], hop_dist[10,5], hop_dist[10,7], 0.0]
        # features[1, :, 0] = [hop_dist[20,20], hop_dist[20,3], hop_dist[20,8], hop_dist[20,9]]
    """
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
    csr_matrices: Optional[list[Tensor]],
    device: torch.device,
) -> Optional[Tensor]:
    """
    Look up pairwise sparse values for each valid token pair in the sequence.

    For each pair of nodes (i, j) in the sequence, this looks up the value
    PE[node_i, node_j] from each provided sparse CSR matrix. This captures
    pairwise relationships between all sequence tokens (e.g., random walk
    structural encoding between any two nodes).

    The output is typically used as attention bias in Graph Transformers,
    added to attention scores before softmax.

    Args:
        node_index_sequences: (batch_size, max_seq_len) node indices for each sequence position
        valid_mask: (batch_size, max_seq_len) bool tensor indicating valid positions
        csr_matrices: List of sparse CSR matrices, each (num_nodes, num_nodes)
        device: Device for output tensor

    Returns:
        features: (batch_size, max_seq_len, max_seq_len, num_attrs) tensor where
            features[b, i, j, k] = csr_matrices[k][node_index_sequences[b, i], node_index_sequences[b, j]]
            for valid (i, j) pairs, 0.0 for padding positions.
        Returns None if csr_matrices is empty.

    Example:
        # batch_size=2, max_seq_len=3, num_attrs=1 (e.g., random_walk_se)
        # node_index_sequences = [[10, 5, 7],   # anchor 0's sequence
        #                         [20, 3, -1]]  # anchor 1's sequence (padded)
        # valid_mask = [[T, T, T], [T, T, F]]
        #
        # Output shape: (2, 3, 3, 1)
        #
        # For batch 0 (all valid), features[0, :, :, 0] is a 3x3 matrix:
        #        node 10    node 5    node 7
        # node 10 [PE[10,10], PE[10,5], PE[10,7]]
        # node 5  [PE[5,10],  PE[5,5],  PE[5,7]]
        # node 7  [PE[7,10],  PE[7,5],  PE[7,7]]
        #
        # For batch 1 (position 2 is padding), features[1, :, :, 0]:
        #        node 20    node 3    (pad)
        # node 20 [PE[20,20], PE[20,3], 0.0]
        # node 3  [PE[3,20],  PE[3,3],  0.0]
        # (pad)   [0.0,       0.0,      0.0]
    """
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


def _lookup_pairwise_relation_masks(
    data: HeteroData,
    node_index_sequences: Tensor,
    valid_mask: Tensor,
    relation_edge_types: Optional[list[EdgeType]],
    node_type_offsets: dict[NodeType, int],
    num_nodes: int,
    device: torch.device,
) -> Optional[Tensor]:
    """Build a dense per-token-pair boolean multi-hot relation mask.

    For each ordered token pair ``(query_pos, key_pos)``, this returns a
    multi-hot vector indicating which directed sampled graph edges connect the
    underlying nodes as ``key -> query``. The relation channel order is defined
    by ``relation_edge_types``.
    """
    if not relation_edge_types:
        return None

    relation_adjacency_matrices = _build_relation_adjacency_matrices(
        data=data,
        relation_edge_types=relation_edge_types,
        node_type_offsets=node_type_offsets,
        num_nodes=num_nodes,
        device=device,
    )
    if not relation_adjacency_matrices:
        return None

    batch_size, max_seq_len = node_index_sequences.shape
    num_relations = len(relation_adjacency_matrices)
    relation_mask = torch.zeros(
        (batch_size, max_seq_len, max_seq_len, num_relations),
        dtype=torch.bool,
        device=device,
    )
    pair_valid_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)
    if not pair_valid_mask.any():
        return relation_mask

    row_indices = node_index_sequences.unsqueeze(2).expand(-1, -1, max_seq_len)
    col_indices = node_index_sequences.unsqueeze(1).expand(-1, max_seq_len, -1)
    valid_row_indices = row_indices[pair_valid_mask]
    valid_col_indices = col_indices[pair_valid_mask]

    for relation_idx, adjacency_matrix in enumerate(relation_adjacency_matrices):
        relation_values = _lookup_csr_values(
            csr_matrix=adjacency_matrix,
            row_indices=valid_row_indices,
            col_indices=valid_col_indices,
        )
        relation_mask[..., relation_idx][pair_valid_mask] = relation_values.ne(0.0)

    return relation_mask


def _build_relation_adjacency_matrices(
    data: HeteroData,
    relation_edge_types: list[EdgeType],
    node_type_offsets: dict[NodeType, int],
    num_nodes: int,
    device: torch.device,
) -> list[Tensor]:
    """Create one binary CSR adjacency matrix per requested edge type."""
    adjacency_matrices: list[Tensor] = []
    empty_row_ptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=device)
    empty_col_idx = torch.zeros(0, dtype=torch.int64, device=device)
    empty_values = torch.zeros(0, dtype=torch.float, device=device)

    for edge_type in relation_edge_types:
        if edge_type not in data.edge_types:
            adjacency_matrices.append(
                torch.sparse_csr_tensor(
                    empty_row_ptr,
                    empty_col_idx,
                    empty_values,
                    size=(num_nodes, num_nodes),
                )
            )
            continue

        edge_index = data[edge_type].edge_index.to(device)
        src_indices = edge_index[0].long() + int(node_type_offsets[edge_type[0]])
        dst_indices = edge_index[1].long() + int(node_type_offsets[edge_type[2]])
        if src_indices.numel() == 0:
            adjacency_matrices.append(
                torch.sparse_csr_tensor(
                    empty_row_ptr,
                    empty_col_idx,
                    empty_values,
                    size=(num_nodes, num_nodes),
                )
            )
            continue

        coalesced_adjacency = torch.sparse_coo_tensor(
            torch.stack([dst_indices, src_indices]),
            torch.ones(src_indices.numel(), dtype=torch.float, device=device),
            size=(num_nodes, num_nodes),
        ).coalesce()
        adjacency_matrices.append(
            torch.sparse_coo_tensor(
                coalesced_adjacency.indices(),
                torch.ones(
                    coalesced_adjacency.indices().size(1),
                    dtype=torch.float,
                    device=device,
                ),
                size=(num_nodes, num_nodes),
            )
            .coalesce()
            .to_sparse_csr()
        )

    return adjacency_matrices


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
