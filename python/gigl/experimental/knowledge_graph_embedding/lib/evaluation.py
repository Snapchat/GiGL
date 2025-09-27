from collections import defaultdict
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

# Type aliases for better readability
EdgeTypeMetrics = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
EvaluationResult = Tuple[torch.Tensor, Dict[CondensedEdgeType, EdgeTypeMetrics]]
MetricAccumulators = Tuple[List[float], Dict, Dict, Dict, Dict]


def _compute_mean(values: List[float]) -> float:
    """Compute mean of values, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _accumulate_metrics_for_edge_type(
    condensed_edge_type: CondensedEdgeType,
    logits: torch.Tensor,
    labels: torch.Tensor,
    condensed_edge_types: torch.Tensor,
    evaluation_config: EvaluationPhaseConfig,
    accumulators: MetricAccumulators,
) -> None:
    """Accumulate evaluation metrics for a specific edge type."""
    losses, pos_logits, neg_logits, mrrs, hit_rates = accumulators

    mask = condensed_edge_types == condensed_edge_type
    if not mask.any():
        return

    # Compute metrics for this edge type
    avg_pos_score, avg_neg_score = average_pos_neg_scores(logits[mask], labels[mask])
    mrr = mean_reciprocal_rank(scores=logits[mask], labels=labels[mask])
    hr_at_k = hit_rate_at_k(
        scores=logits[mask],
        labels=labels[mask],
        ks=evaluation_config.hit_rates_at_k,
    )

    # Store metrics for this edge type
    pos_logits[condensed_edge_type].append(avg_pos_score.item())
    neg_logits[condensed_edge_type].append(avg_neg_score.item())
    mrrs[condensed_edge_type].append(mrr.item())
    hit_rates[condensed_edge_type].append(hr_at_k)


def _aggregate_metrics(
    accumulators: MetricAccumulators,
    unique_edge_types: List[CondensedEdgeType],
    device: torch.device,
) -> List[torch.Tensor]:
    """Aggregate accumulated metrics into tensors for distributed reduction."""
    losses, pos_logits, neg_logits, mrrs, hit_rates = accumulators

    # Calculate per-edge-type averages
    avg_loss = _compute_mean(losses)
    pos_logits_by_cet = {
        cet: _compute_mean(pos_logits[cet]) for cet in unique_edge_types
    }
    neg_logits_by_cet = {
        cet: _compute_mean(neg_logits[cet]) for cet in unique_edge_types
    }
    mrrs_by_cet = {cet: _compute_mean(mrrs[cet]) for cet in unique_edge_types}
    hit_rates_by_cet = {
        cet: torch.stack(hit_rates[cet]).mean(dim=0) for cet in unique_edge_types
    }

    # Convert to tensors and move to device for distributed reduction
    metrics = [
        torch.tensor(avg_loss, device=device),  # overall loss
        torch.tensor(
            [pos_logits_by_cet[cet] for cet in unique_edge_types], device=device
        ),  # positive logits by edge type
        torch.tensor(
            [neg_logits_by_cet[cet] for cet in unique_edge_types], device=device
        ),  # negative logits by edge type
        torch.tensor(
            [mrrs_by_cet[cet] for cet in unique_edge_types], device=device
        ),  # mean reciprocal ranks by edge type
        torch.stack([hit_rates_by_cet[cet] for cet in unique_edge_types], dim=0).to(
            device
        ),  # hit rates by edge type
    ]

    return metrics


def _format_output_metrics(
    metrics: List[torch.Tensor], unique_edge_types: List[CondensedEdgeType]
) -> EvaluationResult:
    """Format metrics into the expected return structure."""
    metrics_by_edge_type = {}
    for i, edge_type in enumerate(unique_edge_types):
        metrics_by_edge_type[edge_type] = (
            metrics[1][i],  # avg_pos_score
            metrics[2][i],  # avg_neg_score
            metrics[3][i],  # avg_mrr
            metrics[4][i],  # avg_hit_rate
        )

    return metrics[0], metrics_by_edge_type


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
          a tuple of (avg_pos_score, avg_neg_score, avg_mrr, avg_hit_rates)
          where avg_hit_rates is a tensor with hit rates at different k values

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

    # Initialize metric accumulators
    accumulators = (
        [],  # losses
        defaultdict(list),  # pos_logits by edge type
        defaultdict(list),  # neg_logits by edge type
        defaultdict(list),  # mrrs by edge type
        defaultdict(list),  # hit_rates by edge type
    )
    losses, _, _, _, _ = accumulators

    unique_edge_types = sorted(graph_metadata.condensed_edge_types)
    step_count = 0

    # Process validation batches
    while True:
        try:
            batch_loss, logits, labels, edge_types = pipeline.progress(val_iter)
            losses.append(batch_loss.item())

            # Accumulate metrics for each edge type in this batch
            for edge_type in unique_edge_types:
                _accumulate_metrics_for_edge_type(
                    edge_type,
                    logits,
                    labels,
                    edge_types,
                    evaluation_phase_config,
                    accumulators,
                )

            step_count += 1
        except StopIteration:
            break

    logger.info(f"Completed {phase} evaluation over {step_count} steps.")

    # Aggregate metrics and prepare for distributed reduction
    aggregated_metrics = _aggregate_metrics(accumulators, unique_edge_types, device)

    # Perform distributed reduction to average across all workers
    for metric in aggregated_metrics:
        dist.all_reduce(metric, op=dist.ReduceOp.AVG)

    # Format results and restore original model state
    result = _format_output_metrics(aggregated_metrics, unique_edge_types)
    pipeline._model.module.set_phase(original_phase)
    pipeline._model.train()

    return result
