from typing import Optional

import torch

def register_csr(
    sampler_id: int,
    edge_type: str,
    indptr: torch.Tensor,
    indices: torch.Tensor,
    edge_ids: Optional[torch.Tensor],
    compressed_id_offset: int,
    with_edge: bool,
) -> None: ...
def unregister_csr(sampler_id: int) -> None: ...
def set_random_seed(seed: int) -> None: ...
def lookup_neighbors(
    sampler_id: int, edge_type: str, ids: torch.Tensor, req_num: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
