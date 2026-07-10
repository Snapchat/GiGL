import torch

TensorTriplet = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
ExtractResult = dict[int, TensorTriplet]


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
    def push_residuals(self, fetched_by_etype_id: dict[int, TensorTriplet]) -> None: ...
    def extract_top_k(self, max_ppr_nodes: int) -> ExtractResult: ...
    def extract_top_k_with_residual_top_up(
        self,
        max_ppr_nodes: int,
        max_residual_nodes: int,
    ) -> ExtractResult: ...
