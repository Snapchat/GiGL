# Significant portions of this file are taken from GraphLearn-for-PyTorch
# (graphlearn_torch/python/distributed/dist_sampling_producer.py).
# This version uses GiGL's DistNeighborSampler (which supports both standard
# neighbor sampling and ABLP) instead of GLT's DistNeighborSampler.

import datetime
import queue
from threading import Barrier
from typing import Optional, Union, cast

import torch
import torch.multiprocessing as mp
from graphlearn_torch.channel import ChannelBase
from graphlearn_torch.distributed import (
    DistDataset,
    DistMpSamplingProducer,
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from graphlearn_torch.distributed.dist_sampling_producer import (
    MP_STATUS_CHECK_INTERVAL,
    MpCommand,
)
from graphlearn_torch.sampler import (
    EdgeSamplerInput,
    NodeSamplerInput,
    SamplingConfig,
    SamplingType,
)
from graphlearn_torch.typing import EdgeType
from graphlearn_torch.utils import seed_everything
from torch._C import _set_worker_signal_handlers
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from gigl.common.logger import Logger
from gigl.distributed.dist_dataset import DistDataset as GiglDistDataset
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
SamplerRuntime = Union[DistNeighborSampler, DistPPRNeighborSampler]


def _create_dist_sampler(
    *,
    data: DistDataset,
    sampling_config: SamplingConfig,
    worker_options: Union[MpDistSamplingWorkerOptions, RemoteDistSamplingWorkerOptions],
    channel: ChannelBase,
    sampler_options: SamplerOptions,
    degree_tensors: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],
    current_device: torch.device,
) -> SamplerRuntime:
    """Create a GiGL sampler runtime for one channel on one worker."""
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


def _prepare_degree_tensors(
    data: DistDataset,
    sampler_options: SamplerOptions,
) -> Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]:
    """Materialize PPR degree tensors before worker spawn when required."""
    degree_tensors: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]] = None
    if isinstance(sampler_options, PPRSamplerOptions):
        assert isinstance(data, GiglDistDataset)
        degree_tensors = data.degree_tensor
        if isinstance(degree_tensors, dict):
            logger.info(
                "Pre-computed degree tensors for PPR sampling across "
                f"{len(degree_tensors)} edge types."
            )
        else:
            logger.info(
                "Pre-computed degree tensor for PPR sampling with "
                f"{degree_tensors.size(0)} nodes."
            )
    return degree_tensors


def _sampling_worker_loop(
    rank: int,
    data: DistDataset,
    sampler_input: Union[NodeSamplerInput, EdgeSamplerInput],
    unshuffled_index: Optional[torch.Tensor],
    sampling_config: SamplingConfig,
    worker_options: MpDistSamplingWorkerOptions,
    channel: ChannelBase,
    task_queue: mp.Queue,
    sampling_completed_worker_count,  # mp.Value
    mp_barrier: Barrier,
    sampler_options: SamplerOptions,
    degree_tensors: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],
):
    dist_sampler = None
    try:
        init_worker_group(
            world_size=worker_options.worker_world_size,
            rank=worker_options.worker_ranks[rank],
            group_name="_sampling_worker_subprocess",
        )
        if worker_options.use_all2all:
            torch.distributed.init_process_group(
                backend="gloo",
                timeout=datetime.timedelta(seconds=worker_options.rpc_timeout),
                rank=worker_options.worker_ranks[rank],
                world_size=worker_options.worker_world_size,
                init_method="tcp://{}:{}".format(
                    worker_options.master_addr, worker_options.master_port
                ),
            )

        if worker_options.num_rpc_threads is None:
            num_rpc_threads = min(data.num_partitions, 16)
        else:
            num_rpc_threads = worker_options.num_rpc_threads

        current_device = worker_options.worker_devices[rank]

        _set_worker_signal_handlers()
        torch.set_num_threads(num_rpc_threads + 1)

        init_rpc(
            master_addr=worker_options.master_addr,
            master_port=worker_options.master_port,
            num_rpc_threads=num_rpc_threads,
            rpc_timeout=worker_options.rpc_timeout,
        )

        if sampling_config.seed is not None:
            seed_everything(sampling_config.seed)

        dist_sampler = _create_dist_sampler(
            data=data,
            sampling_config=sampling_config,
            worker_options=worker_options,
            channel=channel,
            sampler_options=sampler_options,
            degree_tensors=degree_tensors,
            current_device=current_device,
        )
        dist_sampler.start_loop()

        unshuffled_index_loader: Optional[DataLoader]
        loader: DataLoader

        if unshuffled_index is not None:
            unshuffled_index_loader = DataLoader(
                cast(Dataset, unshuffled_index),
                batch_size=sampling_config.batch_size,
                shuffle=False,
                drop_last=sampling_config.drop_last,
            )
        else:
            unshuffled_index_loader = None

        mp_barrier.wait()

        keep_running = True
        while keep_running:
            try:
                command, args = task_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if command == MpCommand.SAMPLE_ALL:
                seeds_index = args
                if seeds_index is None:
                    assert unshuffled_index_loader is not None
                    loader = unshuffled_index_loader
                else:
                    loader = DataLoader(
                        seeds_index,
                        batch_size=sampling_config.batch_size,
                        shuffle=False,
                        drop_last=sampling_config.drop_last,
                    )

                if sampling_config.sampling_type == SamplingType.NODE:
                    for index in loader:
                        dist_sampler.sample_from_nodes(sampler_input[index])
                elif sampling_config.sampling_type == SamplingType.LINK:
                    for index in loader:
                        dist_sampler.sample_from_edges(sampler_input[index])
                elif sampling_config.sampling_type == SamplingType.SUBGRAPH:
                    for index in loader:
                        dist_sampler.subgraph(sampler_input[index])

                dist_sampler.wait_all()

                with sampling_completed_worker_count.get_lock():
                    sampling_completed_worker_count.value += 1

            elif command == MpCommand.STOP:
                keep_running = False
            else:
                raise RuntimeError("Unknown command type")
    except KeyboardInterrupt:
        pass

    if dist_sampler is not None:
        dist_sampler.shutdown_loop()
    shutdown_rpc(graceful=False)


class DistSamplingProducer(DistMpSamplingProducer):
    def __init__(
        self,
        data: DistDataset,
        sampler_input: Union[NodeSamplerInput, EdgeSamplerInput],
        sampling_config: SamplingConfig,
        worker_options: MpDistSamplingWorkerOptions,
        channel: ChannelBase,
        sampler_options: SamplerOptions,
    ):
        super().__init__(data, sampler_input, sampling_config, worker_options, channel)
        self._sampler_options = sampler_options

    def init(self):
        r"""Create the subprocess pool. Init samplers and rpc server."""
        degree_tensors = _prepare_degree_tensors(self.data, self._sampler_options)

        if self.sampling_config.seed is not None:
            seed_everything(self.sampling_config.seed)
        if not self.sampling_config.shuffle:
            unshuffled_indexes = self._get_seeds_indexes()
        else:
            unshuffled_indexes = [None] * self.num_workers

        mp_context = mp.get_context("spawn")
        barrier = mp_context.Barrier(self.num_workers + 1)
        for rank in range(self.num_workers):
            task_queue = mp_context.Queue(
                self.num_workers * self.worker_options.worker_concurrency
            )
            self._task_queues.append(task_queue)
            worker = mp_context.Process(
                target=_sampling_worker_loop,
                args=(
                    rank,
                    self.data,
                    self.sampler_input,
                    unshuffled_indexes[rank],
                    self.sampling_config,
                    self.worker_options,
                    self.output_channel,
                    task_queue,
                    self.sampling_completed_worker_count,
                    barrier,
                    self._sampler_options,
                    degree_tensors,
                ),
            )
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
        barrier.wait()
