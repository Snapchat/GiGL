from dataclasses import dataclass
from typing import Dict, List, Optional

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


# Returns the embeddings after being forward through encoder model
@dataclass
class BatchEmbeddings:
    query_embeddings: torch.FloatTensor
    repeated_query_embeddings: Dict[CondensedEdgeType, torch.FloatTensor]
    pos_embeddings: Dict[CondensedEdgeType, torch.FloatTensor]
    hard_neg_embeddings: Dict[CondensedEdgeType, torch.FloatTensor]
    random_neg_embeddings: Dict[CondensedNodeType, torch.FloatTensor]


# Returns scores for a single anchor node
@dataclass
class BatchScores:
    pos_scores: torch.FloatTensor
    hard_neg_scores: torch.FloatTensor
    random_neg_scores: torch.FloatTensor


# Returns combined scores across all anchor nodes with repeated anchor node embeddings for each positive supervision edge
@dataclass
class BatchCombinedScores:
    repeated_candidate_scores: torch.FloatTensor
    positive_ids: torch.LongTensor
    hard_neg_ids: torch.LongTensor
    random_neg_ids: torch.LongTensor
    repeated_query_ids: Optional[torch.LongTensor]
    num_unique_query_ids: Optional[int]


# Combined object used for storing all outputs of forwarding through NABLP encoder and decoder, minimizing redundant calculation
@dataclass
class NodeAnchorBasedLinkPredictionTaskInputs:
    input_batch: InputBatch
    batch_embeddings: Optional[BatchEmbeddings]
    batch_scores: List[Dict[CondensedEdgeType, BatchScores]]
    batch_combined_scores: Dict[CondensedEdgeType, BatchCombinedScores]
