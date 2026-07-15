from typing import Sequence

import torch

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
        fetched_by_etype_id: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None: ...
    def extract_top_k_with_residual_top_up(
        self,
        max_ppr_nodes: int,
        enable_residual_topup: bool,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...

def drain_typed_ppr_channel_queues(
    states: Sequence[PPRForwardPush],
    fetch_iteration_counts: Sequence[int],
    max_fetch_iterations: int = -1,
) -> tuple[list[int], list[int], list[list[int]], dict[int, torch.Tensor]]: ...
def extract_typed_top_k_with_residual_top_up(
    states: Sequence[PPRForwardPush],
    channel_quotas: Sequence[int],
    max_ppr_nodes: int,
    enable_residual_topup: bool,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]: ...
