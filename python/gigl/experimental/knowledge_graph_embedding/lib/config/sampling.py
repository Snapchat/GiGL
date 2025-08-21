from dataclasses import dataclass

from applied_tasks.knowledge_graph_embedding.lib.model.types import (
    NegativeSamplingCorruptionType,
)


@dataclass
class SamplingConfig:
    """
    Configuration for negative sampling strategy during knowledge graph embedding training.

    Negative sampling is crucial for contrastive learning in knowledge graph embeddings,
    where the model learns to distinguish between true (positive) and false (negative) edges.

    Attributes:
        negative_corruption_side (NegativeSamplingCorruptionType): Which side of the edge to corrupt for negative sampling.
            NegativeSamplingCorruptionType.DST corrupts the destination node,
            NegativeSamplingCorruptionType.SRC corrupts the source node.
            Defaults to NegativeSamplingCorruptionType.DST.
        positive_edge_batch_size (int): Number of positive (true) edges to process in each batch.
            Controls memory usage and training stability. Defaults to 1024.
        num_inbatch_negatives_per_edge (int): Number of negative samples generated per positive edge
            using other edges in the same batch. This is memory-efficient but may have
            limited diversity. Defaults to 0 (disabled).
        num_random_negatives_per_edge (int): Number of negative samples generated per positive edge
            by randomly corrupting nodes. Provides high diversity but requires more computation.
            Defaults to 1024.
    """

    negative_corruption_side: NegativeSamplingCorruptionType = (
        NegativeSamplingCorruptionType.DST
    )
    positive_edge_batch_size: int = 1024
    num_inbatch_negatives_per_edge: int = 0
    num_random_negatives_per_edge: int = 1024
