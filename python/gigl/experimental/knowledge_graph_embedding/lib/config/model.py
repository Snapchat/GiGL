from dataclasses import dataclass
from typing import List, Optional

import torchrec
from gigl.experimental.knowledge_graph_embedding.lib.config.sampling import (
    SamplingConfig,
)
from gigl.experimental.knowledge_graph_embedding.lib.model.types import (
    OperatorType,
    SimilarityType,
)


@dataclass
class ModelConfig:
    """
    Configuration for knowledge graph embedding model architecture.

    Defines the structure and behavior of the embedding model used for link prediction
    in heterogeneous knowledge graphs.

    Attributes:
        node_embedding_dim (int): Dimensionality of node embeddings. Higher dimensions can
            capture more complex relationships but require more memory and computation.
            Defaults to 128.
        embedding_similarity_type (SimilarityType): Type of similarity function used to compute scores
            between node embeddings. Options include cosine similarity, dot product, etc.
            Defaults to SimilarityType.COSINE.
        src_operator (OperatorType): Transformation operator applied to source node embeddings before
            computing edge scores. Can be identity (no transformation) or learned operators.
            Defaults to OperatorType.IDENTITY.
        dst_operator (OperatorType): Transformation operator applied to destination node embeddings
            before computing edge scores. Can be identity (no transformation) or learned operators.
            Defaults to OperatorType.IDENTITY.
        training_sampling (Optional[SamplingConfig]): Sampling configuration used during training phase.
            Populated at runtime from training config. Defaults to None.
        validation_sampling (Optional[SamplingConfig]): Sampling configuration used during validation phase.
            Populated at runtime from validation config. Defaults to None.
        testing_sampling (Optional[SamplingConfig]): Sampling configuration used during testing phase.
            Populated at runtime from testing config. Defaults to None.
        num_edge_types (Optional[int]): Number of distinct edge types in the knowledge graph.
            Populated at runtime from graph metadata. Defaults to None.
        embeddings_config (Optional[List[torchrec.EmbeddingBagConfig]]): TorchRec embedding configuration for sparse embeddings.
            Specifies embedding tables, sharding strategies, and optimization settings.
            Populated at runtime. Defaults to None.
    """

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
