# All code in this file is directly taken from GraphLearn-for-PyTorch (graphlearn_torch/python/distributed/dist_sampling_producer.py),
# with the exception that we call the GiGL DistNeighborSampler with custom link prediction logic instead of the GLT DistNeighborSampler.

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
from graphlearn_torch.utils import seed_everything
from torch._C import _set_worker_signal_handlers
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from gigl.distributed.dist_neighbor_sampler import (
    DistABLPNeighborSampler,
    DistPPRNeighborSampler,
)


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
        dist_sampler = DistABLPNeighborSampler(
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
            seed=sampling_config.seed,
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
                    sampling_completed_worker_count.value += (
                        1  # non-atomic, lock is necessary
                    )

            elif command == MpCommand.STOP:
                keep_running = False
            else:
                raise RuntimeError("Unknown command type")
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass

    if dist_sampler is not None:
        dist_sampler.shutdown_loop()
    shutdown_rpc(graceful=False)


class DistABLPSamplingProducer(DistMpSamplingProducer):
    def init(self):
        r"""Create the subprocess pool. Init samplers and rpc server."""
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
            w = mp_context.Process(
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
                ),
            )
            w.daemon = True
            w.start()
            self._workers.append(w)
        barrier.wait()


def _ppr_sampling_worker_loop(
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
    ppr_alpha: float,
    ppr_eps: float,
    ppr_max_nodes: int,
    ppr_degree_tensors: Union[torch.Tensor, dict],
):
    """
    Worker loop for PPR-based sampling. Similar to _sampling_worker_loop but uses
    DistPPRNeighborSampler instead of DistABLPNeighborSampler.

    Args:
        ppr_degree_tensors: Pre-computed degree tensors for efficient degree lookups.
            Required for PPR algorithm. Must be provided.
    """
    dist_sampler = None
    try:
        init_worker_group(
            world_size=worker_options.worker_world_size,
            rank=worker_options.worker_ranks[rank],
            group_name="_ppr_sampling_worker_subprocess",
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

        # Use PPR sampler instead of ABLP sampler
        dist_sampler = DistPPRNeighborSampler(
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
            seed=sampling_config.seed,
            alpha=ppr_alpha,
            eps=ppr_eps,
            max_ppr_nodes=ppr_max_nodes,
            degree_tensors=ppr_degree_tensors,
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
                    sampling_completed_worker_count.value += (
                        1  # non-atomic, lock is necessary
                    )

            elif command == MpCommand.STOP:
                keep_running = False
            else:
                raise RuntimeError("Unknown command type")
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass

    if dist_sampler is not None:
        dist_sampler.shutdown_loop()
    shutdown_rpc(graceful=False)


class DistPPRSamplingProducer(DistMpSamplingProducer):
    """
    Sampling producer that uses PPR-based neighbor sampling instead of standard
    uniform neighbor sampling.

    Args:
        ppr_alpha: Restart probability for PPR (default: 0.15)
        ppr_eps: Convergence threshold for PPR (default: 1e-4)
        ppr_max_nodes: Maximum number of PPR neighbors to return per seed (default: 50)
        ppr_degree_tensors: Pre-computed degree tensors for degree lookups.
            Required for PPR algorithm.
            For homogeneous graphs: torch.Tensor of shape [num_nodes] where tensor[i]
            is the degree of node i.
            For heterogeneous graphs: dict[EdgeType, torch.Tensor] where each tensor
            contains the degree of each node for that edge type.
            Degree lookups are done via in-memory tensor indexing instead of
            network calls, significantly reducing latency in the PPR computation.
    """

    def __init__(
        self,
        data: DistDataset,
        sampler_input: Union[NodeSamplerInput, EdgeSamplerInput],
        sampling_config: SamplingConfig,
        worker_options: MpDistSamplingWorkerOptions,
        output_channel: ChannelBase,
        ppr_degree_tensors: Union[torch.Tensor, dict],
        ppr_alpha: float = 0.15,
        ppr_eps: float = 1e-4,
        ppr_max_nodes: int = 50,
    ):
        super().__init__(
            data, sampler_input, sampling_config, worker_options, output_channel
        )
        self.ppr_alpha = ppr_alpha
        self.ppr_eps = ppr_eps
        self.ppr_max_nodes = ppr_max_nodes
        self.ppr_degree_tensors = ppr_degree_tensors

    def init(self):
        r"""Create the subprocess pool. Init PPR samplers and rpc server."""
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
            w = mp_context.Process(
                target=_ppr_sampling_worker_loop,
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
                    self.ppr_alpha,
                    self.ppr_eps,
                    self.ppr_max_nodes,
                    self.ppr_degree_tensors,
                ),
            )
            w.daemon = True
            w.start()
            self._workers.append(w)
        barrier.wait()
