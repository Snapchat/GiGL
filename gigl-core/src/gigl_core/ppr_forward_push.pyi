from typing import Sequence

import torch

class NeighborFetchTensors:
    node_ids: torch.Tensor
    flat_neighbor_ids: torch.Tensor
    counts: torch.Tensor
    edge_ids: torch.Tensor | None
    def __init__(
        self,
        node_ids: torch.Tensor,
        flat_neighbor_ids: torch.Tensor,
        counts: torch.Tensor,
        edge_ids: torch.Tensor | None = None,
    ) -> None: ...

class PPRExtractTensors:
    ids: torch.Tensor
    weights: torch.Tensor
    valid_counts: torch.Tensor

class OriginalEdgeExtractTensors:
    rows: torch.Tensor
    cols: torch.Tensor
    edge_ids: torch.Tensor | None

class TypedPPRQueueDrainResult:
    drained_channel_indices: list[int]
    fetch_channel_indices: list[int]
    edge_type_ids_by_fetch_channel: list[list[int]]
    unioned_node_ids_by_edge_type_id: dict[int, torch.Tensor]

class PPRForwardPush:
    def __init__(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type_id: int,
        alpha: float,
        requeue_threshold_factor: float,
        node_type_to_edge_type_ids: list[list[int]],
        edge_type_to_dst_ntype_id: list[int],
        degree_tensors: list[torch.Tensor],
    ) -> None: ...
    def drain_queue(self) -> dict[int, torch.Tensor] | None: ...
    def push_residuals(
        self,
        fetched_by_etype_id: dict[int, NeighborFetchTensors],
    ) -> None: ...
    def extract_top_k_with_residual_top_up(
        self,
        max_ppr_nodes: int,
        enable_residual_topup: bool,
    ) -> dict[int, PPRExtractTensors]: ...

def drain_typed_ppr_channel_queues(
    states: Sequence[PPRForwardPush],
    fetch_iteration_counts: Sequence[int],
    max_fetch_iterations: int = -1,
) -> TypedPPRQueueDrainResult: ...
def extract_typed_top_k_with_residual_top_up(
    states: Sequence[PPRForwardPush],
    channel_target_counts: Sequence[int],
    enable_residual_topup: bool,
) -> dict[int, PPRExtractTensors]: ...
def extract_original_edges_from_ppr_caches(
    states: Sequence[PPRForwardPush],
    selected_node_ids_by_node_type_id: dict[int, torch.Tensor],
    include_edge_ids: bool,
) -> dict[int, OriginalEdgeExtractTensors]: ...
