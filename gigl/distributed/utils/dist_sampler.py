"""Sampler factory helpers shared across sampling producers."""

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

from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler
from gigl.distributed.dist_ppr_sampler import DistPPRNeighborSampler
from gigl.distributed.sampler import ABLPNodeSamplerInput
from gigl.distributed.sampler_options import (
    KHopNeighborSamplerOptions,
    PPRSamplerOptions,
    SamplerOptions,
)

SamplerInput = Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput]
"""Union of all supported sampler input types."""

SamplerRuntime = Union[DistNeighborSampler, DistPPRNeighborSampler]
"""Union of all supported GiGL sampler runtime types."""


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
        data: The distributed dataset containing graph topology and features.
        sampling_config: Configuration for sampling behavior (neighbors, edges, etc.).
        worker_options: Worker-level options (RPC settings, device placement, concurrency).
        channel: The communication channel for passing sampled messages.
        sampler_options: Algorithm-specific options (k-hop or PPR).
        degree_tensors: Pre-computed degree tensors required by PPR sampling.
            Must not be ``None`` when ``sampler_options`` is :class:`PPRSamplerOptions`.
        current_device: The device on which sampling will run.

    Returns:
        A configured sampler runtime, either :class:`DistNeighborSampler` or
        :class:`DistPPRNeighborSampler`.

    Raises:
        NotImplementedError: If ``sampler_options`` is an unsupported type.
    """
    shared_sampler_kwargs = dict(
        data=data,
        num_neighbors=sampling_config.num_neighbors,
        with_edge=sampling_config.with_edge,
        with_neg=sampling_config.with_neg,
        with_weight=sampling_config.with_weight,
        edge_dir=sampling_config.edge_dir,
        collect_features=sampling_config.collect_features,
        channel=channel,
        use_all2all=worker_options.use_all2all,
        concurrency=worker_options.worker_concurrency,
        device=current_device,
        seed=sampling_config.seed,
    )
    if isinstance(sampler_options, KHopNeighborSamplerOptions):
        sampler: SamplerRuntime = DistNeighborSampler(
            **shared_sampler_kwargs,
        )
    elif isinstance(sampler_options, PPRSamplerOptions):
        assert degree_tensors is not None
        sampler = DistPPRNeighborSampler(
            **shared_sampler_kwargs,
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
