from dataclasses import dataclass, field
from typing import List, Optional

from applied_tasks.knowledge_graph_embedding.lib.config.dataloader import (
    DataloaderConfig,
)
from applied_tasks.knowledge_graph_embedding.lib.config.training import SamplingConfig


@dataclass
class EvaluationPhaseConfig:
    """
    Configuration for evaluation phases (validation/testing) during knowledge graph embedding training.

    Controls how model performance is measured during training (validation phase) and after
    training completion (testing phase). Uses ranking-based metrics to assess link prediction quality.

    Attributes:
        dataloader (DataloaderConfig): Configuration for data loading during evaluation (workers, memory pinning).
            Defaults to DataloaderConfig() with standard settings.
        step_frequency (Optional[int]): How often to run evaluation during training (every N steps).
            If None, evaluation runs only at the end of training. Defaults to None.
        num_batches (Optional[int]): Maximum number of batches to evaluate. Useful for faster evaluation
            on large datasets by sampling a subset. If None, evaluates all data. Defaults to None.
        hit_rates_at_k (List[int]): List of k values for computing Hit@k (Hits at k) metrics.
            Hit@k measures if the correct answer appears in the top k predictions.
            Common values are [1, 10, 100]. Defaults to [1, 10, 100].
        sampling (SamplingConfig): Negative sampling configuration for evaluation. Should match or be
            compatible with training sampling to ensure fair comparison.
            Defaults to SamplingConfig() with standard settings.
    """

    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    step_frequency: Optional[int] = None
    num_batches: Optional[int] = None
    hit_rates_at_k: List[int] = field(default_factory=lambda: [1, 10, 100])
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self) -> None:
        """
        Post-initialization validation of evaluation configuration parameters.

        Validates that the total number of negative samples (random + in-batch) is
        sufficient to compute the requested Hit@k metrics. This ensures that evaluation
        can meaningfully compute metrics for all requested k values.

        Raises:
            ValueError: If max(hit_rates_at_k) exceeds the total number of negative samples
                available for ranking (num_random_negatives_per_edge + num_inbatch_negatives_per_edge).
        """
        if max(self.hit_rates_at_k) > (
            self.sampling.num_random_negatives_per_edge
            + self.sampling.num_inbatch_negatives_per_edge
        ):
            raise ValueError(
                f"""Validation `num_random_negatives_per_edge` + `num_inbatch_negatives_per_edge` must be >= max(hit_rates_at_k).
                Got ({self.sampling.num_random_negatives_per_edge} + {self.sampling.num_inbatch_negatives_per_edge}) and {max(self.hit_rates_at_k)}"""
            )
