from dataclasses import dataclass
from typing import Optional

import torch
from jaxtyping import Float, Int

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


# Returns the embeddings after being forward through encoder model
@dataclass
class BatchEmbeddings:
    query_embeddings: Float[torch.FloatTensor, "queries embedding_dim"]
    repeated_query_embeddings: dict[
        CondensedEdgeType, Float[torch.FloatTensor, "_queries embedding_dim"]
    ]
    pos_embeddings: dict[
        CondensedEdgeType, Float[torch.FloatTensor, "_positives embedding_dim"]
    ]
    hard_neg_embeddings: dict[
        CondensedEdgeType, Float[torch.FloatTensor, "_hard_negatives embedding_dim"]
    ]
    random_neg_embeddings: dict[
        CondensedNodeType,
        Float[torch.FloatTensor, "_random_negatives embedding_dim"],
    ]


# Returns scores for a single anchor node
@dataclass
class BatchScores:
    pos_scores: Float[torch.FloatTensor, "1 positives"]
    hard_neg_scores: Float[torch.FloatTensor, "1 hard_negatives"]
    random_neg_scores: Float[torch.FloatTensor, "1 random_negatives"]


# Returns combined scores across all anchor nodes with repeated anchor node embeddings for each positive supervision edge
@dataclass
class BatchCombinedScores:
    repeated_candidate_scores: Float[torch.FloatTensor, "queries candidates"]
    positive_ids: Int[torch.LongTensor, " positives"]
    hard_neg_ids: Int[torch.LongTensor, " hard_negatives"]
    random_neg_ids: Int[torch.LongTensor, " random_negatives"]
    repeated_query_ids: Optional[Int[torch.LongTensor, " queries"]]
    num_unique_query_ids: Optional[int]


# Combined object used for storing all outputs of forwarding through NABLP encoder and decoder, minimizing redundant calculation
@dataclass
class NodeAnchorBasedLinkPredictionTaskInputs:
    input_batch: InputBatch
    batch_embeddings: Optional[BatchEmbeddings]
    batch_scores: list[dict[CondensedEdgeType, BatchScores]]
    batch_combined_scores: dict[CondensedEdgeType, BatchCombinedScores]
