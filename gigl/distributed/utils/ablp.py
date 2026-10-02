"""Utilities for remapping ABLP labels to sampled-subgraph indices.

Tensor dimensions used in this module:

- ``A``: anchors in one sampled batch.
- ``M``: padded label slots per anchor.
- ``S``: sampled nodes for one supervision node type.
- ``K``: non-padding candidate labels.
- ``E``: output label pairs; ``E == K`` because sampling always includes labels.

A label edge index has shape ``[2, E]``. Row 0 indexes anchors in ``[0, A)``;
row 1 indexes local nodes in the supervision node store in ``[0, S)``.

Example::

    # Padded global labels: [A=3, M=2]
    label_tensor = [[30, -1], [40, 10], [-1, -1]]

    # Sampled supervision nodes: local id -> global id, [S=3]
    local_id_to_global_id = [40, 10, 30]

    # Local label edge index: [2, E=3]
    label_edge_index = [[0, 1, 1], [2, 0, 1]]

    # Deprecated dictionary view of the same output
    labels_by_anchor = {0: [2], 1: [0, 1], 2: []}

The corresponding label relationships are shown below. Anchor 2 has no label
edge because both of its input slots are padding.

.. code-block:: dot

   digraph ablp_labels {
     rankdir=LR;
     node [fontname="Helvetica"];
     a0 [label="anchor 0", shape=box];
     a1 [label="anchor 1", shape=box];
     a2 [label="anchor 2", shape=box];
     n0 [label="local 0\\nglobal 40"];
     n1 [label="local 1\\nglobal 10"];
     n2 [label="local 2\\nglobal 30"];
     a0 -> n2 [label="global label 30", color="forestgreen", style=dashed];
     a1 -> n0 [label="global label 40", color="forestgreen", style=dashed];
     a1 -> n1 [label="global label 10", color="forestgreen", style=dashed];
   }
"""

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
        label_edge_index: ``[2, E]`` local label edge index. Row 0 contains
            anchor indices in ``[0, A)``; row 1 contains local label-node
            indices. Pairs must be grouped by row-0 anchor index.
        num_anchors: ``A``, including anchors with no labels.

    Returns:
        Every local anchor index in ``[0, A)`` mapped to a ``[E_a]`` tensor of
        local label-node indices, where ``sum(E_a) == E``.

    Example:
        >>> label_edge_index_to_dict(
        ...     torch.tensor([[0, 1, 1], [2, 0, 1]]), num_anchors=3
        ... )
        {0: tensor([2]), 1: tensor([0, 1]), 2: tensor([], dtype=torch.int64)}
    """
    # counts and labels_by_anchor have shape [A] and A x [E_a], respectively.
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
        label_tensor: ``[A, M]`` global label-node ids, padded with
            ``PADDING_NODE``.
        sorted_global_node_ids: ``[S]`` sorted global ids for sampled
            supervision nodes.
        local_ids_by_sorted_global_id: ``[S]`` local node indices aligned with
            ``sorted_global_node_ids``.

    Returns:
        ``[2, E]`` long tensor on ``label_tensor.device``. Row 0 contains
        local anchor indices; row 1 contains local label-node indices. Every
        non-padding label must be present in the sampled subgraph, so
        ``E == K <= A * M``.

    Raises:
        ValueError: If the sampled node map contains duplicate global ids.
        ValueError: If a non-padding label was not sampled.

    Example:
        >>> _remap_label_tensor_to_local_edge_index(
        ...     label_tensor=torch.tensor([[30, -1], [40, 10], [-1, -1]]),
        ...     sorted_global_node_ids=torch.tensor([10, 30, 40]),
        ...     local_ids_by_sorted_global_id=torch.tensor([1, 2, 0]),
        ... )
        tensor([[0, 1, 1],
                [2, 0, 1]])
    """
    num_anchors = int(label_tensor.size(0))
    num_nodes = int(sorted_global_node_ids.size(0))
    # Empty label edge indices always retain the [2, E] contract.
    empty_edge_index = torch.empty((2, 0), dtype=torch.long, device=label_tensor.device)
    if num_anchors == 0:
        return empty_edge_index

    num_labels = int(label_tensor.size(1))
    candidate_global_label_ids = label_tensor.reshape(-1)  # [A * M]
    candidate_anchor_indices = torch.arange(
        num_anchors, device=label_tensor.device
    ).repeat_interleave(num_labels)  # [A * M]

    is_not_padding = candidate_global_label_ids != PADDING_NODE  # [A * M]
    candidate_global_label_ids = candidate_global_label_ids[is_not_padding]  # [K]
    candidate_anchor_indices = candidate_anchor_indices[is_not_padding]  # [K]
    if candidate_global_label_ids.numel() == 0:
        return empty_edge_index
    if num_nodes == 0:
        raise ValueError(
            "Every non-padding ABLP label must be present in the sampled subgraph."
        )

    if not bool((sorted_global_node_ids[1:] > sorted_global_node_ids[:-1]).all()):
        raise ValueError(
            "Vectorized label remapping requires unique global ids in the "
            "sampled node map."
        )

    insertion_indices = torch.searchsorted(
        sorted_global_node_ids, candidate_global_label_ids
    )  # [K]
    if bool((insertion_indices == num_nodes).any()) or not bool(
        (sorted_global_node_ids[insertion_indices] == candidate_global_label_ids).all()
    ):
        raise ValueError(
            "Every non-padding ABLP label must be present in the sampled subgraph."
        )

    local_label_indices = local_ids_by_sorted_global_id[insertion_indices]  # [E == K]
    anchor_indices = candidate_anchor_indices  # [E == K]
    return torch.stack((anchor_indices, local_label_indices))  # [2, E]


def _remap_labels_by_edge_type(
    labels_by_edge_type: dict[EdgeType, torch.Tensor],
    local_id_to_global_id_by_node_type: dict[NodeType, torch.Tensor],
    sorted_node_lookup_by_type: dict[NodeType, tuple[torch.Tensor, torch.Tensor]],
) -> dict[EdgeType, torch.Tensor]:
    """Remap positive or negative labels for each supervision edge type.

    Args:
        labels_by_edge_type: Per label edge type, ``[A_et, M_et]`` padded
            global label-node ids.
        local_id_to_global_id_by_node_type: Per supervision node type, ``[S_t]``
            local-to-global node ids.
        sorted_node_lookup_by_type: Cached ``[S_t]`` sorted global ids and
            aligned local node ids for each supervision node type.

    Returns:
        Per message-passing edge type, a ``[2, E_et]`` local label edge index.
        Edge types with ``A_et == 0`` are omitted.

    Example:
        A ``{label_edge_type: [[30, -1]]}`` input becomes
        ``{message_passing_edge_type: [[0], [2]]}`` when the destination node
        map is ``[40, 10, 30]``.
    """
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
    node type. Pair order within an anchor is unspecified; row-0 anchor indices
    remain grouped in ascending order.

    Args:
        local_id_to_global_id_by_node_type: Per node type ``t``, ``[S_t]``
            global ids where index ``i`` is the local node id.
        positive_labels_by_edge_type: Per positive-label edge type ``et``,
            ``[A_et, M_et]`` padded global label-node ids.
        negative_labels_by_edge_type: Equivalent ``[A_et, M_et]`` negative
            label tensors. May be empty.

    Returns:
        Positive and negative mappings keyed by message-passing edge type. Each
        value is a ``[2, E_et]`` local label edge index whose rows index the
        corresponding anchor batch and supervision node store.

    Example:
        With destination local-to-global ids ``[40, 10, 30]``, positive labels
        ``[[30, -1]]`` map to ``[[0], [2]]`` and negative labels ``[[40, 10]]``
        map to ``[[0, 0], [0, 1]]``. The returned positive and negative
        dictionaries retain these edge indices under their message-passing edge
        types.
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
