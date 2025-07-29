from dataclasses import dataclass
from typing import List, Optional

import torchrec
from applied_tasks.knowledge_graph_embedding.lib.config.sampling import SamplingConfig
from applied_tasks.knowledge_graph_embedding.lib.model.types import (
    OperatorType,
    SimilarityType,
)


@dataclass
class ModelConfig:
    node_embedding_dim: int = 128
    embedding_similarity_type: SimilarityType = SimilarityType.COSINE
    src_operator: OperatorType = OperatorType.IDENTITY
    dst_operator: OperatorType = OperatorType.IDENTITY

    # Below fields are populated at runtime.
    training_sampling: Optional[SamplingConfig] = None
    validation_sampling: Optional[SamplingConfig] = None
    testing_sampling: Optional[SamplingConfig] = None
    num_edge_types: Optional[int] = None
    embeddings_config: Optional[List[torchrec.EmbeddingBagConfig]] = None
