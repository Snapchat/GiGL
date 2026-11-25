"""Inspection utilities for GiGL distributed training.

Includes per-batch fanout summaries (``summary``) for in-loop sanity and a
startup-time partition + sampler RAM logger (``log_startup_diagnostics``)
for catching silent failure modes at process boot.
"""

import torch
from torch_geometric.data import HeteroData

from gigl.analytics._inspect_impl_ai import HeteroDataSummary, _summary_impl
from gigl.common.logger import Logger
from gigl.distributed import DistDataset

logger = Logger()

# GLT allocates ~64 MB per sampling worker by default; used for the
# back-of-envelope RAM estimate logged at startup.
# TODO: (svij) - this should be auto configured i.e. not needed.
_GLT_DEFAULT_RAM_MB_PER_SAMPLING_WORKER = 64


def summary(data: HeteroData) -> HeteroDataSummary:
    """Per-seed fanout summary for a sampled HeteroData batch.

    Auto-detects the seed node type (the unique node type with ``batch_size > 0``)
    and the number of hops (from ``data.num_sampled_nodes[seed_type]``). At each
    hop, walks every edge type whose ``num_sampled_edges`` slice contains hop-K
    edges. Edges are followed from whichever end is in the current frontier
    (so the inspector works under both ``edge_dir="in"`` and ``edge_dir="out"``
    — under ``"out"`` the loader stores edges reversed, putting the seed on the
    destination side).

    ``str(summary(data))`` produces:
    ``"seeds=N hop1(min=X med=Y avg=Z max=W) hop2(...) ..."``.

    Example:
        >>> from gigl.analytics.inspect import summary
        >>> result = summary(batch)
        >>> print(result)
        'seeds=128 hop1(min=3 med=10 avg=12.5 max=25) hop2(min=12 med=80 avg=91.2 max=240)'

    Args:
        data: HeteroData batch produced by a GiGL neighbor loader. Must carry
            sampler metadata: ``batch_size`` on exactly one node type and the
            root-level dicts ``data.num_sampled_nodes`` (keyed by node type)
            and ``data.num_sampled_edges`` (keyed by edge type).

    Returns:
        ``HeteroDataSummary`` with the seed count and per-hop
        ``HeteroDataSummary.HopStats``.

    Raises:
        ValueError: zero or multiple node types have ``batch_size > 0``, or
            sampler metadata is missing for the seed type / any edge type.
    """
    return _summary_impl(data)


def log_startup_diagnostics(
    rank: int,
    world_size: int,
    dataset: DistDataset,
    sampling_workers_per_process: int,
    sampling_worker_shared_channel_size: str,
) -> None:
    """Log sampler-RAM estimate and local partition counts at startup.

    Surfaces two silent failure modes in GLT-based distributed training:
    sampler-worker RAM blowup (silent OOM) and partition misload (a rank
    receives an empty or wrong shard and silently overfits to it).

    Call once per rank from the training process bootstrap, after the
    dataset has been built and the distributed process group is initialized.
    Emits one INFO line for the sampler RAM accounting, one INFO line for
    the local node counts, and a WARNING for every node type with zero
    local nodes.

    Example:
        >>> from gigl.analytics.inspect import log_startup_diagnostics
        >>> log_startup_diagnostics(
        ...     rank=0,
        ...     world_size=8,
        ...     dataset=dataset,
        ...     sampling_workers_per_process=4,
        ...     sampling_worker_shared_channel_size="4GB",
        ... )

    Args:
        rank: Global rank of the calling process.
        world_size: Total number of ranks in the distributed group.
        dataset: Built ``DistDataset`` with ``node_ids`` populated.
        sampling_workers_per_process: Number of GLT sampling workers per
            training process; used for the RAM estimate.
        sampling_worker_shared_channel_size: Shared-channel size string
            (e.g. ``"4GB"``) passed to GLT; logged for visibility.

    Raises:
        ValueError: ``dataset.node_ids`` is ``None`` (dataset not built).
    """
    ram_mb_per_rank = (
        sampling_workers_per_process * _GLT_DEFAULT_RAM_MB_PER_SAMPLING_WORKER
    )
    logger.info(
        f"rank={rank} sampler RAM/rank: "
        f"workers={sampling_workers_per_process} "
        f"channel={sampling_worker_shared_channel_size} "
        f"≈ {ram_mb_per_rank} MB/rank × world_size={world_size}"
    )

    node_ids = dataset.node_ids
    if node_ids is None:
        raise ValueError("dataset.node_ids is None — dataset not built")

    if isinstance(node_ids, torch.Tensor):
        count = node_ids.numel()
        logger.info(f"rank={rank} local node count: {count}")
        if count == 0:
            logger.warning(f"rank={rank} has 0 nodes — partition misload?")
        return

    node_counts = {nt: node_ids[nt].numel() for nt in node_ids}
    logger.info(f"rank={rank} local node counts per type: {node_counts}")
    for nt, count in node_counts.items():
        if count == 0:
            logger.warning(f"rank={rank} has 0 {nt} nodes — partition misload?")


__all__ = ["HeteroDataSummary", "log_startup_diagnostics", "summary"]
