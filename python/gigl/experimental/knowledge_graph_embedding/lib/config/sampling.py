from dataclasses import dataclass

from applied_tasks.knowledge_graph_embedding.lib.model.types import (
    NegativeSamplingCorruptionType,
)


@dataclass
class SamplingConfig:
    negative_corruption_side: NegativeSamplingCorruptionType = (
        NegativeSamplingCorruptionType.DST
    )
    positive_edge_batch_size: int = 1024
    num_inbatch_negatives_per_edge: int = 0
    num_random_negatives_per_edge: int = 1024
