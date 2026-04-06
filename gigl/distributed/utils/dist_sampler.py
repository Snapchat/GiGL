"""Public helpers for creating GiGL distributed samplers.

Extracted from ``dist_sampling_producer.py`` so that both the legacy
per-producer path and the shared-backend path can reuse the same
factory and degree-tensor logic without circular imports.
"""

from typing import Optional, Union

import torch
from graphlearn_torch.channel import ChannelBase
from graphlearn_torch.distributed import (
    DistDataset,
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
)
from graphlearn_torch.sampler import EdgeSamplerInput, NodeSamplerInput, SamplingConfig
from graphlearn_torch.typing import EdgeType

from gigl.common.logger import Logger
from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler
from gigl.distributed.dist_ppr_sampler import DistPPRNeighborSampler
from gigl.distributed.sampler import ABLPNodeSamplerInput
from gigl.distributed.sampler_options import (
    KHopNeighborSamplerOptions,
    PPRSamplerOptions,
    SamplerOptions,
)

logger = Logger()

SamplerInput = Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput]
"""Union of all sampler input types supported by GiGL loaders."""

SamplerRuntime = Union[DistNeighborSampler, DistPPRNeighborSampler]
"""Union of all GiGL distributed sampler runtime types."""


def create_dist_sampler(
    *,
    data: DistDataset,
    sampling_config: SamplingConfig,
    worker_options: Union[MpDistSamplingWorkerOptions, RemoteDistSamplingWorkerOptions],
    channel: ChannelBase,
    sampler_options: SamplerOptions,
    degree_tensors: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],
    current_device: torch.device,
) -> SamplerRuntime:
    """Create a GiGL sampler runtime for one channel on one worker.

    Args:
        data: The distributed dataset to sample from.
        sampling_config: Configuration controlling neighbor counts, edge
            inclusion, feature collection, etc.
        worker_options: Worker-level options (RPC addresses, devices, concurrency).
        channel: The output channel that receives sampled messages.
        sampler_options: Sampler-type–specific options (k-hop or PPR).
        degree_tensors: Pre-computed degree tensors required by PPR sampling.
            Must be non-``None`` when *sampler_options* is :class:`PPRSamplerOptions`.
        current_device: The torch device this worker should use.

    Returns:
        A configured :class:`DistNeighborSampler` or
        :class:`DistPPRNeighborSampler` instance.

    Raises:
        NotImplementedError: If *sampler_options* is an unrecognized type.
    """
    shared_sampler_args = (
        data,
        sampling_config.num_neighbors,
        sampling_config.with_edge,
        sampling_config.with_neg,
        sampling_config.with_weight,
        sampling_config.edge_dir,
        sampling_config.collect_features,
        channel,
        worker_options.use_all2all,
        worker_options.worker_concurrency,
        current_device,
    )
    if isinstance(sampler_options, KHopNeighborSamplerOptions):
        sampler: SamplerRuntime = DistNeighborSampler(
            *shared_sampler_args,
            seed=sampling_config.seed,
        )
    elif isinstance(sampler_options, PPRSamplerOptions):
        assert degree_tensors is not None
        sampler = DistPPRNeighborSampler(
            *shared_sampler_args,
            seed=sampling_config.seed,
            alpha=sampler_options.alpha,
            eps=sampler_options.eps,
            max_ppr_nodes=sampler_options.max_ppr_nodes,
            num_neighbors_per_hop=sampler_options.num_neighbors_per_hop,
            total_degree_dtype=sampler_options.total_degree_dtype,
            degree_tensors=degree_tensors,
        )
    else:
        raise NotImplementedError(
            f"Unsupported sampler options type: {type(sampler_options)}"
        )
    return sampler


def prepare_degree_tensors(
    data: DistDataset,
    sampler_options: SamplerOptions,
) -> Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]:
    """Materialize PPR degree tensors before worker spawn when required.

    For non-PPR sampler options this is a no-op that returns ``None``.

    Args:
        data: The distributed dataset whose degree information is needed.
        sampler_options: Sampler-type–specific options.  Only
            :class:`PPRSamplerOptions` triggers tensor materialization.

    Returns:
        The degree tensor(s) if PPR sampling is configured, otherwise ``None``.
    """
    degree_tensors: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]] = None
    if isinstance(sampler_options, PPRSamplerOptions):
        degree_tensors = data.degree_tensor
        if isinstance(degree_tensors, dict):
            logger.info(
                "Pre-computed degree tensors for PPR sampling across "
                f"{len(degree_tensors)} edge types."
            )
        elif degree_tensors is not None:
            logger.info(
                "Pre-computed degree tensor for PPR sampling with "
                f"{degree_tensors.size(0)} nodes."
            )
    return degree_tensors
