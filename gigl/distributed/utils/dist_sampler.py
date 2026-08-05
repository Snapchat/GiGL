"""Sampler factory helpers shared across sampling producers."""

import random
from typing import Optional, Union

import torch
from graphlearn_torch.channel import ChannelBase
from graphlearn_torch.distributed import (
    DistDataset,
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
)
from graphlearn_torch.sampler import EdgeSamplerInput, NodeSamplerInput, SamplingConfig
from graphlearn_torch.typing import NodeType

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
    degree_tensors: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]],
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

    Note:
        When ``sampling_config.seed`` is unset we draw one here, per worker. The seed
        belongs to the process that samples rather than to the config: the config is shared
        across ranks in Graph Store mode and compared for equality, so a value drawn while
        building it would differ per rank and be rejected.

        A seed is a large performance win, not just a determinism knob. GLT's
        ``CPURandomSampler::UniformSample`` reads
        ``RandomSeedManager::getInstance().getSeed()`` on **every** call, and that call
        happens once per source row of a batch:

        - https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/csrc/cpu/random_sampler.cc#L144
        - https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/csrc/cpu/random_sampler.cc#L165

        The ``std::mt19937`` it feeds is ``thread_local static``, so after the first call
        the read is discarded -- but it still runs. With no seed set, ``getSeed()``
        constructs a ``std::random_device`` and draws from it, about 5 us of real work per
        source row for a value that is thrown away:

        - https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/include/common.h#L48-L56

        For the production use case we see up to 7x speed up, and 29x speedup in local
        testing.

        Passing the seed to the sampler is what sets it: GLT calls
        ``RandomSeedManager::getInstance().setSeed`` from ``NeighborSampler.__init__`` only
        when the seed is not ``None``. That manager is process-global and the generator it
        feeds is ``thread_local``, so the first sampler built on a worker thread fixes that
        thread's stream. Per-channel seeds therefore are not a determinism guarantee.

        TODO(kmonte): Drop this workaround if GLT ever hoists the ``getSeed()`` call out
        of ``CPURandomSampler::UniformSample`` and into the engine initializer, so the
        unseeded path stops paying per-row entropy.
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
        seed=(
            sampling_config.seed
            if sampling_config.seed is not None
            else random.getrandbits(32)
        ),
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
            enable_residual_topup=sampler_options.enable_residual_topup,
            max_fetch_iterations=sampler_options.max_fetch_iterations,
            num_neighbors_per_hop=sampler_options.num_neighbors_per_hop,
            typed_channel_ratios=sampler_options.typed_channel_ratios,
            degree_tensors=degree_tensors,
        )
    else:
        raise NotImplementedError(
            f"Unsupported sampler options type: {type(sampler_options)}"
        )
    return sampler
