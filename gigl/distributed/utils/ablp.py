"""Utilities for remapping ABLP labels to sampled-subgraph indices."""

import torch
from torch_geometric.typing import EdgeType

from gigl.src.common.types.graph_data import NodeType
from gigl.types.graph import label_edge_type_to_message_passing_edge_type
from gigl.utils.data_splitters import PADDING_NODE


def label_edge_index_to_dict(
    label_edge_index: torch.Tensor, num_anchors: int
) -> dict[int, torch.Tensor]:
    """Convert a label edge index to the deprecated per-anchor dictionary.

    The edge index must be grouped by anchor, as guaranteed by
    :func:`remap_labels_to_local_edge_indices`.

    Args:
        label_edge_index: A ``[2, E]`` tensor. Row 0 contains local anchor
            indices and row 1 contains local label-node indices.
        num_anchors: Number of anchors in the sampled batch, including anchors
            with no labels.

    Returns:
        A mapping from every local anchor index to its local label-node indices.
    """
    counts = torch.bincount(label_edge_index[0], minlength=num_anchors)
    labels_by_anchor = torch.split(label_edge_index[1], counts.tolist())
    return {anchor: labels_by_anchor[anchor] for anchor in range(num_anchors)}


def _remap_label_tensor_to_local_edge_index(
    label_tensor: torch.Tensor,
    sorted_global_node_ids: torch.Tensor,
    local_ids_by_sorted_global_id: torch.Tensor,
) -> torch.Tensor:
    """Remap one padded global-label tensor to a local label edge index.

    Args:
        label_tensor: A ``[N_anchors, M]`` tensor of global label-node ids,
            padded with ``PADDING_NODE``.
        sorted_global_node_ids: Sorted global ids for the sampled supervision
            nodes.
        local_ids_by_sorted_global_id: Local node indices in the order given by
            ``sorted_global_node_ids``.

    Returns:
        A ``[2, E]`` long tensor on the input device. Row 0 contains local
        anchor indices and row 1 contains local label-node indices. Labels not
        present in the sampled subgraph are omitted.

    Raises:
        ValueError: If the sampled node map contains duplicate global ids.
    """
    num_anchors = int(label_tensor.size(0))
    num_nodes = int(sorted_global_node_ids.size(0))
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=label_tensor.device)
    if num_anchors == 0:
        return empty_edge_index

    num_labels = int(label_tensor.size(1))
    candidate_global_label_ids = label_tensor.reshape(-1)
    candidate_anchor_indices = torch.arange(
        num_anchors, device=label_tensor.device
    ).repeat_interleave(num_labels)

    is_not_padding = candidate_global_label_ids != PADDING_NODE
    candidate_global_label_ids = candidate_global_label_ids[is_not_padding]
    candidate_anchor_indices = candidate_anchor_indices[is_not_padding]
    if num_nodes == 0 or candidate_global_label_ids.numel() == 0:
        return empty_edge_index

    if not bool((sorted_global_node_ids[1:] > sorted_global_node_ids[:-1]).all()):
        raise ValueError(
            "Vectorized label remapping requires unique global ids in the "
            "sampled node map."
        )

    insertion_indices = torch.searchsorted(
        sorted_global_node_ids, candidate_global_label_ids
    )
    # An absent label larger than every sampled id inserts at num_nodes. Clamp
    # before gathering; the exact-match mask below still discards that label.
    insertion_indices = insertion_indices.clamp_(max=num_nodes - 1)
    is_in_sampled_subgraph = (
        sorted_global_node_ids[insertion_indices] == candidate_global_label_ids
    )

    local_label_indices = local_ids_by_sorted_global_id[insertion_indices][
        is_in_sampled_subgraph
    ]
    anchor_indices = candidate_anchor_indices[is_in_sampled_subgraph]
    return torch.stack((anchor_indices, local_label_indices))


def _remap_labels_by_edge_type(
    labels_by_edge_type: dict[EdgeType, torch.Tensor],
    local_id_to_global_id_by_node_type: dict[NodeType, torch.Tensor],
    sorted_node_lookup_by_type: dict[NodeType, tuple[torch.Tensor, torch.Tensor]],
) -> dict[EdgeType, torch.Tensor]:
    """Remap positive or negative labels for each supervision edge type."""
    output: dict[EdgeType, torch.Tensor] = {}
    for label_edge_type, label_tensor in labels_by_edge_type.items():
        if label_tensor.size(0) == 0:
            continue

        supervision_node_type = label_edge_type[2]
        if supervision_node_type not in sorted_node_lookup_by_type:
            sorted_node_lookup_by_type[supervision_node_type] = torch.sort(
                local_id_to_global_id_by_node_type[supervision_node_type]
            )
        (
            sorted_global_node_ids,
            local_ids_by_sorted_global_id,
        ) = sorted_node_lookup_by_type[supervision_node_type]
        output[label_edge_type_to_message_passing_edge_type(label_edge_type)] = (
            _remap_label_tensor_to_local_edge_index(
                label_tensor=label_tensor,
                sorted_global_node_ids=sorted_global_node_ids,
                local_ids_by_sorted_global_id=local_ids_by_sorted_global_id,
            )
        )
    return output


def remap_labels_to_local_edge_indices(
    local_id_to_global_id_by_node_type: dict[NodeType, torch.Tensor],
    positive_labels_by_edge_type: dict[EdgeType, torch.Tensor],
    negative_labels_by_edge_type: dict[EdgeType, torch.Tensor],
) -> tuple[dict[EdgeType, torch.Tensor], dict[EdgeType, torch.Tensor]]:
    """Remap padded global ABLP labels to local label edge indices.

    Positive and negative labels share one sorted node lookup per supervision
    node type. Pair order within an anchor is unspecified.

    Args:
        local_id_to_global_id_by_node_type: Per node type, a tensor whose
            ``i``-th entry is the global id for local node ``i``.
        positive_labels_by_edge_type: Per positive-label edge type, a padded
            ``[N_anchors, M]`` tensor of global label-node ids.
        negative_labels_by_edge_type: Equivalent negative-label tensors. May
            be empty.

    Returns:
        Positive and negative mappings keyed by message-passing edge type. Each
        value is a ``[2, E]`` local label edge index.
    """
    sorted_node_lookup_by_type: dict[NodeType, tuple[torch.Tensor, torch.Tensor]] = {}
    positive_label_edge_indices = _remap_labels_by_edge_type(
        labels_by_edge_type=positive_labels_by_edge_type,
        local_id_to_global_id_by_node_type=local_id_to_global_id_by_node_type,
        sorted_node_lookup_by_type=sorted_node_lookup_by_type,
    )
    negative_label_edge_indices = _remap_labels_by_edge_type(
        labels_by_edge_type=negative_labels_by_edge_type,
        local_id_to_global_id_by_node_type=local_id_to_global_id_by_node_type,
        sorted_node_lookup_by_type=sorted_node_lookup_by_type,
    )
    return positive_label_edge_indices, negative_label_edge_indices
