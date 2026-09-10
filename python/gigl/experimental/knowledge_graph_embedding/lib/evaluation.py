from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import torch
import torch.distributed as dist
from torchrec.distributed import TrainPipelineSparseDist

from gigl.common.logger import Logger
from gigl.experimental.knowledge_graph_embedding.lib.config import EvaluationPhaseConfig
from gigl.experimental.knowledge_graph_embedding.lib.model.loss_utils import (
    average_pos_neg_scores,
    hit_rate_at_k,
    mean_reciprocal_rank,
)
from gigl.experimental.knowledge_graph_embedding.lib.model.types import ModelPhase
from gigl.src.common.types.graph_data import CondensedEdgeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper

logger = Logger()


@dataclass(frozen=True)
class EdgeTypeMetrics:
    """Container for evaluation metrics for a specific edge type.

    Attributes:
        avg_pos_score: Average positive score for this edge type
        avg_neg_score: Average negative score for this edge type
        avg_mrr: Average Mean Reciprocal Rank for this edge type
        avg_hit_rates: Average hit rates at different k values for this edge type
    """

    avg_pos_score: torch.Tensor
    avg_neg_score: torch.Tensor
    avg_mrr: torch.Tensor
    avg_hit_rates: torch.Tensor


# Type aliases for better readability
EvaluationResult = Tuple[torch.Tensor, Dict[CondensedEdgeType, EdgeTypeMetrics]]


class EvaluationMetricsAccumulator:
    """Maintains tensors of evaluation metrics for all edge types.

    This class uses tensors throughout to efficiently accumulate and reduce metrics
    across batches and distributed workers. Each edge type corresponds to an index
    in the metric tensors.

    Attributes:
        total_loss: Scalar tensor tracking total loss across all batches
        total_batches: Scalar tensor tracking total number of batches
        sample_counts: Tensor of sample counts per edge type [num_edge_types]
        pos_scores: Tensor of accumulated positive scores per edge type [num_edge_types]
        neg_scores: Tensor of accumulated negative scores per edge type [num_edge_types]
        mrrs: Tensor of accumulated MRRs per edge type [num_edge_types]
        hit_rates: Tensor of accumulated hit rates per edge type [num_edge_types, num_k_values]
        edge_type_to_idx: Mapping from CondensedEdgeType to tensor index
        evaluation_config: Configuration containing evaluation parameters
    """

    def __init__(
        self,
        unique_edge_types: List[CondensedEdgeType],
        evaluation_config: EvaluationPhaseConfig,
        device: torch.device,
    ):
        """Initialize the accumulator with zero tensors.

        Args:
            unique_edge_types: Sorted list of unique edge types in the graph
            evaluation_config: Configuration containing hit rate k values and other evaluation parameters
            device: Device to place tensors on
        """
        self._evaluation_config = evaluation_config
        self._edge_type_to_idx = {et: i for i, et in enumerate(unique_edge_types)}

        num_edge_types = len(unique_edge_types)
        num_k_values = len(evaluation_config.hit_rates_at_k)

        self._total_loss = torch.tensor(0.0, device=device)
        self._total_batches = torch.tensor(0, dtype=torch.long, device=device)
        self._sample_counts = torch.zeros(
            num_edge_types, dtype=torch.long, device=device
        )
        self._pos_scores = torch.zeros(num_edge_types, device=device)
        self._neg_scores = torch.zeros(num_edge_types, device=device)
        self._mrrs = torch.zeros(num_edge_types, device=device)
        self._hit_rates = torch.zeros(num_edge_types, num_k_values, device=device)

    def accumulate_batch(
        self,
        batch_loss: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor,
        condensed_edge_types: torch.Tensor,
    ) -> None:
        """Accumulate metrics from a batch for all edge types.

        Args:
            batch_loss: Loss value for this batch
            logits: Model logits for this batch
            labels: Ground truth labels for this batch
            condensed_edge_types: Edge type indices for each sample in the batch
        """
        # Accumulate batch-level metrics
        self._total_loss += batch_loss
        self._total_batches += 1

        # Process each edge type
        for edge_type, idx in self._edge_type_to_idx.items():
            edge_type_mask = condensed_edge_types == edge_type
            if not edge_type_mask.any():
                continue

            # Compute metrics for this edge type
            avg_pos_score, avg_neg_score = average_pos_neg_scores(
                logits[edge_type_mask], labels[edge_type_mask]
            )
            mrr = mean_reciprocal_rank(
                scores=logits[edge_type_mask], labels=labels[edge_type_mask]
            )
            hr_at_k = hit_rate_at_k(
                scores=logits[edge_type_mask],
                labels=labels[edge_type_mask],
                ks=self._evaluation_config.hit_rates_at_k,
            )

            # Accumulate weighted totals for this edge type
            edge_type_sample_count_in_batch = edge_type_mask.sum()
            self._sample_counts[idx] += edge_type_sample_count_in_batch
            self._pos_scores[idx] += avg_pos_score * edge_type_sample_count_in_batch
            self._neg_scores[idx] += avg_neg_score * edge_type_sample_count_in_batch
            self._mrrs[idx] += mrr * edge_type_sample_count_in_batch
            self._hit_rates[idx] += hr_at_k * edge_type_sample_count_in_batch

    def sum_metrics_over_ranks(self) -> None:
        """Perform distributed reduction (sum) on all metric tensors."""
        dist.all_reduce(self._total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._total_batches, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._sample_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._pos_scores, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._neg_scores, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._mrrs, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._hit_rates, op=dist.ReduceOp.SUM)

    def compute_final_metrics(
        self,
        unique_edge_types: List[CondensedEdgeType],
    ) -> EvaluationResult:
        """Compute final averaged metrics and format as return structure.

        Args:
            unique_edge_types: Sorted list of unique edge types in the graph

        Returns:
            Tuple of (average_loss, metrics_by_edge_type)
        """
        # Compute average loss
        avg_loss = self._total_loss / self._total_batches

        # Create mask for valid edge types (those with samples)
        mask = self._sample_counts > 0

        # Initialize metric tensors with NaN for missing edge types
        avg_pos_scores = torch.full_like(self._pos_scores, float("nan"))
        avg_neg_scores = torch.full_like(self._neg_scores, float("nan"))
        avg_mrrs = torch.full_like(self._mrrs, float("nan"))
        avg_hit_rates = torch.full_like(self._hit_rates, float("nan"))

        # Compute averages for edge types with samples
        if mask.any():
            avg_pos_scores[mask] = (
                self._pos_scores[mask] / self._sample_counts[mask].float()
            )
            avg_neg_scores[mask] = (
                self._neg_scores[mask] / self._sample_counts[mask].float()
            )
            avg_mrrs[mask] = self._mrrs[mask] / self._sample_counts[mask].float()
            # Broadcast sample counts for hit rates division
            avg_hit_rates[mask] = self._hit_rates[mask] / self._sample_counts[
                mask
            ].float().unsqueeze(-1)

        # Format into expected return structure
        metrics_by_edge_type = {}
        for edge_type in unique_edge_types:
            idx = self._edge_type_to_idx[edge_type]
            metrics_by_edge_type[edge_type] = EdgeTypeMetrics(
                avg_pos_score=avg_pos_scores[idx],
                avg_neg_score=avg_neg_scores[idx],
                avg_mrr=avg_mrrs[idx],
                avg_hit_rates=avg_hit_rates[idx],
            )

        # Log edge types with undefined metrics
        missing_edge_types = [
            et
            for et in unique_edge_types
            if self._sample_counts[self._edge_type_to_idx[et]] == 0
        ]
        if missing_edge_types:
            logger.warning(
                f"Edge types {missing_edge_types} have no samples across all ranks. "
                f"Setting metrics to NaN (undefined)."
            )

        return avg_loss, metrics_by_edge_type


def evaluate(
    pipeline: TrainPipelineSparseDist,
    val_iter: Iterator,
    phase: ModelPhase,
    evaluation_phase_config: EvaluationPhaseConfig,
    graph_metadata: GraphMetadataPbWrapper,
) -> EvaluationResult:
    """Evaluate a knowledge graph embedding model on validation data.

    This function runs the model in evaluation mode, processes validation batches,
    computes various metrics (loss, MRR, hit rates) per edge type, and aggregates
    results across distributed workers.

    Args:
        pipeline: Distributed training pipeline containing the model
        val_iter: Iterator over validation data batches
        phase: Model phase to set during evaluation (e.g., VALIDATION, TEST)
        evaluation_phase_config: Configuration specifying evaluation parameters
            like hit_rates_at_k values
        graph_metadata: Metadata containing information about condensed edge types
            in the graph

    Returns:
        A tuple containing:
        - overall_loss: Average loss across all batches and edge types
        - metrics_by_edge_type: Dictionary mapping each CondensedEdgeType to
          an EdgeTypeMetrics object containing avg_pos_score, avg_neg_score,
          avg_mrr, and avg_hit_rates (tensor with hit rates at different k values)

    Note:
        This function temporarily switches the model to evaluation mode and the
        specified phase, then restores the original state. Results are averaged
        across distributed workers using all_reduce operations.
    """
    # Set model to evaluation mode and save original state
    pipeline._model.eval()
    original_phase = pipeline._model.module.phase
    pipeline._model.module.set_phase(phase)
    device = pipeline._device

    # Initialize tensor-based metric accumulator
    unique_edge_types = sorted(graph_metadata.condensed_edge_types)

    accumulator = EvaluationMetricsAccumulator(
        unique_edge_types=unique_edge_types,
        evaluation_config=evaluation_phase_config,
        device=device,
    )

    step_count = 0

    # Process validation batches
    while True:
        try:
            batch_loss, logits, labels, edge_types = pipeline.progress(val_iter)

            # Accumulate metrics for all edge types in this batch
            accumulator.accumulate_batch(
                batch_loss=batch_loss,
                logits=logits,
                labels=labels,
                condensed_edge_types=edge_types,
            )

            step_count += 1
        except StopIteration:
            break

    logger.info(f"Completed {phase} evaluation over {step_count} steps.")

    # Perform distributed reduction on all metric tensors
    accumulator.sum_metrics_over_ranks()

    # Compute final averaged metrics and format results
    result = accumulator.compute_final_metrics(unique_edge_types=unique_edge_types)

    # Restore original model state
    pipeline._model.module.set_phase(original_phase)
    pipeline._model.train()

    return result
