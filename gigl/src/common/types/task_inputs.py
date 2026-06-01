from dataclasses import dataclass
from typing import Optional

import torch

from gigl.src.common.types.graph_data import CondensedEdgeType, CondensedNodeType
from gigl.src.training.v1.lib.data_loaders.node_anchor_based_link_prediction_data_loader import (
    NodeAnchorBasedLinkPredictionBatch,
)
from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)


# Returns the original main batch and random negative batch, used for self-supervised training
@dataclass
class InputBatch:
    main_batch: NodeAnchorBasedLinkPredictionBatch
    random_neg_batch: RootedNodeNeighborhoodBatch


@dataclass
class BatchEmbeddings:
    """Embeddings produced by forwarding a batch through the encoder model.

    Attributes:
        query_embeddings (torch.Tensor): Anchor node embeddings.
            Expected dtype: ``torch.float32``.
        repeated_query_embeddings (dict[CondensedEdgeType, torch.Tensor]):
            Per-edge-type anchor embeddings repeated to align with their
            positive supervision targets. Expected dtype: ``torch.float32``.
        pos_embeddings (dict[CondensedEdgeType, torch.Tensor]): Per-edge-type
            positive target embeddings. Expected dtype: ``torch.float32``.
        hard_neg_embeddings (dict[CondensedEdgeType, torch.Tensor]):
            Per-edge-type hard negative target embeddings.
            Expected dtype: ``torch.float32``.
        random_neg_embeddings (dict[CondensedNodeType, torch.Tensor]):
            Per-node-type random negative embeddings.
            Expected dtype: ``torch.float32``.
    """

    query_embeddings: torch.Tensor
    repeated_query_embeddings: dict[CondensedEdgeType, torch.Tensor]
    pos_embeddings: dict[CondensedEdgeType, torch.Tensor]
    hard_neg_embeddings: dict[CondensedEdgeType, torch.Tensor]
    random_neg_embeddings: dict[CondensedNodeType, torch.Tensor]


@dataclass
class BatchScores:
    """Decoder scores for a single anchor node.

    Attributes:
        pos_scores (torch.Tensor): Scores for positive supervision targets.
            Expected dtype: ``torch.float32``.
        hard_neg_scores (torch.Tensor): Scores for hard negative targets.
            Expected dtype: ``torch.float32``.
        random_neg_scores (torch.Tensor): Scores for random negative targets.
            Expected dtype: ``torch.float32``.
    """

    pos_scores: torch.Tensor
    hard_neg_scores: torch.Tensor
    random_neg_scores: torch.Tensor


@dataclass
class BatchCombinedScores:
    """Combined decoder scores across all anchor nodes with repeated anchor embeddings.

    Used to avoid redundant calculation for retrieval-style losses.

    Attributes:
        repeated_candidate_scores (torch.Tensor): Scores between repeated
            anchor embeddings and the combined candidate set
            (positives + hard negatives + random negatives).
            Expected dtype: ``torch.float32``.
        positive_ids (torch.Tensor): Global node ids of positive targets.
            Expected dtype: ``torch.int64``.
        hard_neg_ids (torch.Tensor): Global node ids of hard negative targets.
            Expected dtype: ``torch.int64``.
        random_neg_ids (torch.Tensor): Global node ids of random negative
            targets. Expected dtype: ``torch.int64``.
        repeated_query_ids (Optional[torch.Tensor]): Global node ids of the
            anchors repeated to align with ``repeated_candidate_scores``.
            Expected dtype: ``torch.int64``.
        num_unique_query_ids (Optional[int]): Number of unique anchors before
            repetition.
    """

    repeated_candidate_scores: torch.Tensor
    positive_ids: torch.Tensor
    hard_neg_ids: torch.Tensor
    random_neg_ids: torch.Tensor
    repeated_query_ids: Optional[torch.Tensor]
    num_unique_query_ids: Optional[int]


# Combined object used for storing all outputs of forwarding through NABLP encoder and decoder, minimizing redundant calculation
@dataclass
class NodeAnchorBasedLinkPredictionTaskInputs:
    input_batch: InputBatch
    batch_embeddings: Optional[BatchEmbeddings]
    batch_scores: list[dict[CondensedEdgeType, BatchScores]]
    batch_combined_scores: dict[CondensedEdgeType, BatchCombinedScores]
