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
) -> None:
    """Registers (or replaces) the CSR of one edge type of one sampler, for ``lookup_neighbors``.

    The tensors are referenced, not copied, so the caller must keep them unchanged.

    Args:
        sampler_id (int): The sampler's RPC sampling callee id, identical on every rank.
        edge_type (str): The registry key of the edge type; ``"homogeneous_edge"`` for a homogeneous graph.
        indptr (torch.Tensor): Contiguous int32 or int64 CPU row pointers, shape ``[row_count + 1]``, starting at 0.
        indices (torch.Tensor): Contiguous int32 or int64 CPU column ids, shape ``[edge_count]``.
        edge_ids (Optional[torch.Tensor]): Contiguous int64 CPU edge ids, shape ``[edge_count]``, or ``None``.
            An empty tensor is treated as ``None``, as GLT does.
        compressed_id_offset (int): The global id of row 0.
        with_edge (bool): Whether lookups return edge ids.

    Raises:
        RuntimeError: If a tensor is not on the CPU, not contiguous or not 1-D, if a dtype is not as above, if
            ``indptr`` is empty, does not start at 0 or ends past ``indices``, or if ``edge_ids`` and ``indices``
            differ in length.
    """
    ...

def unregister_csr(sampler_id: int) -> None:
    """Removes every edge type registered under ``sampler_id``.

    Lookups already in flight keep their tensors.

    Args:
        sampler_id (int): The sampler's RPC sampling callee id.
    """
    ...

def set_random_seed(seed: int) -> None:
    """Seeds the generator that sub-samples rows longer than the requested fan-out.

    Each thread seeds its own generator with this value the first time it sub-samples; threads that have already
    sub-sampled keep their generator.

    Args:
        seed (int): The seed, as GLT's ``RandomSeedManager.setSeed`` takes it.
    """
    ...
