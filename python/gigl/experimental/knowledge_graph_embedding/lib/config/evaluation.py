from dataclasses import dataclass, field
from typing import List, Optional

from applied_tasks.knowledge_graph_embedding.lib.config.dataloader import (
    DataloaderConfig,
)
from applied_tasks.knowledge_graph_embedding.lib.config.training import SamplingConfig


@dataclass
class EvaluationPhaseConfig:
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    step_frequency: Optional[int] = None
    num_batches: Optional[int] = None
    hit_rates_at_k: List[int] = field(default_factory=lambda: [1, 10, 100])
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self):
        if max(self.hit_rates_at_k) > (
            self.sampling.num_random_negatives_per_edge
            + self.sampling.num_inbatch_negatives_per_edge
        ):
            raise ValueError(
                f"""Validation `num_random_negatives_per_edge` + `num_inbatch_negatives_per_edge` must be >= max(hit_rates_at_k).
                Got ({self.sampling.num_random_negatives_per_edge} + {self.sampling.num_inbatch_negatives_per_edge}) and {max(self.hit_rates_at_k)}"""
            )
