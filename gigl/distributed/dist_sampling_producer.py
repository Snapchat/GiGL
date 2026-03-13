# Significant portions of this file are taken from GraphLearn-for-PyTorch
# (graphlearn_torch/python/distributed/dist_sampling_producer.py).
# This version uses GiGL's DistNeighborSampler (which supports both standard
# neighbor sampling and ABLP) instead of GLT's DistNeighborSampler.

import datetime
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.process import BaseProcess
from threading import Barrier
from typing import Callable, Optional, Union, cast

import torch
import torch.multiprocessing as mp
from graphlearn_torch.channel import ChannelBase, SampleMessage
from graphlearn_torch.distributed import (
    DistDataset,
    DistMpSamplingProducer,
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from graphlearn_torch.distributed.dist_context import get_context
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

from gigl.common.logger import Logger
from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler
from gigl.distributed.sampler import ABLPNodeSamplerInput
from gigl.distributed.sampler_options import KHopNeighborSamplerOptions, SamplerOptions


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

        # Resolve sampler class from options
        if isinstance(sampler_options, KHopNeighborSamplerOptions):
            sampler_cls = DistNeighborSampler
        else:
            raise NotImplementedError(
                f"Unsupported sampler options type: {type(sampler_options)}"
            )

        dist_sampler = sampler_cls(
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
                    self._sampler_options,
                ),
            )
            w.daemon = True
            w.start()
            self._workers.append(w)
        barrier.wait()


class SharedMpCommand(Enum):
    REGISTER_INPUT = auto()
    UNREGISTER_INPUT = auto()
    START_EPOCH = auto()
    STOP = auto()


EPOCH_DONE_EVENT = "EPOCH_DONE"


@dataclass(frozen=True)
class RegisterInputCmd:
    channel_id: int
    worker_key: str
    sampler_input: Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput]
    sampling_config: SamplingConfig
    channel: ChannelBase


@dataclass(frozen=True)
class StartEpochCmd:
    channel_id: int
    epoch: int
    seeds_index: Optional[torch.Tensor]


def _compute_worker_seeds_ranges(
    input_len: int,
    batch_size: int,
    num_workers: int,
) -> list[tuple[int, int]]:
    num_worker_batches = [0] * num_workers
    num_total_complete_batches = input_len // batch_size
    for rank in range(num_workers):
        num_worker_batches[rank] += num_total_complete_batches // num_workers
    for rank in range(num_total_complete_batches % num_workers):
        num_worker_batches[rank] += 1

    index_ranges: list[tuple[int, int]] = []
    start = 0
    for rank in range(num_workers):
        end = start + num_worker_batches[rank] * batch_size
        if rank == num_workers - 1:
            end = input_len
        index_ranges.append((start, end))
        start = end
    return index_ranges


def _slice_sampler_input(
    sampler_input: Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput],
    start: int,
    end: int,
) -> Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput]:
    if isinstance(sampler_input, ABLPNodeSamplerInput):
        return ABLPNodeSamplerInput(
            node=sampler_input.node[start:end],
            input_type=sampler_input.input_type,
            positive_label_by_edge_types={
                edge_type: labels[start:end]
                for edge_type, labels in sampler_input.positive_label_by_edge_types.items()
            },
            negative_label_by_edge_types={
                edge_type: labels[start:end]
                for edge_type, labels in sampler_input.negative_label_by_edge_types.items()
            },
        )
    if isinstance(sampler_input, NodeSamplerInput):
        return NodeSamplerInput(
            node=sampler_input.node[start:end],
            input_type=sampler_input.input_type,
        )
    return EdgeSamplerInput(
        row=sampler_input.row[start:end],
        col=sampler_input.col[start:end],
        label=(
            sampler_input.label[start:end] if sampler_input.label is not None else None
        ),
        input_type=sampler_input.input_type,
        neg_sampling=sampler_input.neg_sampling,
    )


def _shared_sampling_worker_loop(
    rank: int,
    data: DistDataset,
    backend_sampling_config: SamplingConfig,
    worker_options: Union[MpDistSamplingWorkerOptions, RemoteDistSamplingWorkerOptions],
    task_queue: mp.Queue,
    event_queue: mp.Queue,
    mp_barrier: Barrier,
    sampler_options: SamplerOptions,
) -> None:
    logger = Logger()
    dist_sampler = None
    channels: dict[int, ChannelBase] = {}
    inputs: dict[
        int, Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput]
    ] = {}
    cfgs: dict[int, SamplingConfig] = {}
    route_key_by_channel: dict[int, str] = {}
    pending: dict[tuple[int, int], int] = defaultdict(int)
    started_epoch: dict[int, int] = defaultdict(lambda: -1)
    removing: set[int] = set()
    channel_locks: dict[int, threading.Lock] = {}

    def _lock_for(channel_id: int) -> threading.Lock:
        lock = channel_locks.get(channel_id)
        if lock is None:
            lock = threading.Lock()
            channel_locks[channel_id] = lock
        return lock

    def _on_batch_done(channel_id: int, epoch: int) -> None:
        route_key_to_remove: Optional[str] = None
        with _lock_for(channel_id):
            key = (channel_id, epoch)
            pending[key] -= 1
            if pending[key] == 0:
                pending.pop(key, None)
                event_queue.put((EPOCH_DONE_EVENT, channel_id, epoch, rank))
                if channel_id in removing:
                    channels.pop(channel_id, None)
                    inputs.pop(channel_id, None)
                    cfgs.pop(channel_id, None)
                    started_epoch.pop(channel_id, None)
                    route_key_to_remove = route_key_by_channel.pop(channel_id, None)
                    removing.discard(channel_id)
        if route_key_to_remove is not None and dist_sampler is not None:
            dist_sampler.unregister_output(route_key_to_remove)

    def _make_batch_done_callback(
        channel_id: int, epoch: int
    ) -> Callable[[Optional[SampleMessage]], None]:
        def _callback(_res: Optional[SampleMessage]) -> None:
            _on_batch_done(channel_id, epoch)

        return _callback

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
                init_method=f"tcp://{worker_options.master_addr}:{worker_options.master_port}",
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

        if backend_sampling_config.seed is not None:
            seed_everything(backend_sampling_config.seed)

        if isinstance(sampler_options, KHopNeighborSamplerOptions):
            sampler_cls = DistNeighborSampler
        else:
            raise NotImplementedError(
                f"Unsupported sampler options type: {type(sampler_options)}"
            )

        dist_sampler = sampler_cls(
            data,
            backend_sampling_config.num_neighbors,
            backend_sampling_config.with_edge,
            backend_sampling_config.with_neg,
            backend_sampling_config.with_weight,
            backend_sampling_config.edge_dir,
            backend_sampling_config.collect_features,
            channel=None,
            use_all2all=worker_options.use_all2all,
            concurrency=worker_options.worker_concurrency,
            device=current_device,
            seed=backend_sampling_config.seed,
        )
        dist_sampler.start_loop()
        mp_barrier.wait()
        assert dist_sampler is not None

        keep_running = True
        while keep_running:
            try:
                command, payload = task_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if command == SharedMpCommand.REGISTER_INPUT:
                register_cmd: RegisterInputCmd = payload
                route_key = str(register_cmd.channel_id)
                with _lock_for(register_cmd.channel_id):
                    channels[register_cmd.channel_id] = register_cmd.channel
                    inputs[register_cmd.channel_id] = register_cmd.sampler_input
                    cfgs[register_cmd.channel_id] = register_cmd.sampling_config
                    route_key_by_channel[register_cmd.channel_id] = route_key
                    started_epoch[register_cmd.channel_id] = -1
                dist_sampler.register_output(route_key, register_cmd.channel)
                logger.info(
                    f"worker_rank={rank} registered channel_id={register_cmd.channel_id} worker_key={register_cmd.worker_key}"
                )

            elif command == SharedMpCommand.START_EPOCH:
                start_cmd: StartEpochCmd = payload
                with _lock_for(start_cmd.channel_id):
                    if start_cmd.channel_id not in inputs:
                        continue
                    if started_epoch[start_cmd.channel_id] >= start_cmd.epoch:
                        continue
                    started_epoch[start_cmd.channel_id] = start_cmd.epoch
                    local_input = inputs[start_cmd.channel_id]
                    sampling_config = cfgs[start_cmd.channel_id]
                    route_key = route_key_by_channel[start_cmd.channel_id]

                seeds_index = start_cmd.seeds_index
                if seeds_index is None:
                    seeds_index = torch.arange(len(local_input))
                loader = DataLoader(
                    cast(Dataset, seeds_index),
                    batch_size=sampling_config.batch_size,
                    shuffle=False,
                    drop_last=sampling_config.drop_last,
                )
                num_batches = len(loader)
                with _lock_for(start_cmd.channel_id):
                    pending[(start_cmd.channel_id, start_cmd.epoch)] = num_batches
                if num_batches == 0:
                    event_queue.put(
                        (
                            EPOCH_DONE_EVENT,
                            start_cmd.channel_id,
                            start_cmd.epoch,
                            rank,
                        )
                    )
                    continue

                if sampling_config.sampling_type == SamplingType.NODE:
                    callback = _make_batch_done_callback(
                        start_cmd.channel_id, start_cmd.epoch
                    )
                    for index in loader:
                        dist_sampler.sample_from_nodes(
                            local_input[index],
                            key=route_key,
                            callback=callback,
                        )
                elif sampling_config.sampling_type == SamplingType.LINK:
                    callback = _make_batch_done_callback(
                        start_cmd.channel_id, start_cmd.epoch
                    )
                    for index in loader:
                        dist_sampler.sample_from_edges(
                            local_input[index],
                            key=route_key,
                            callback=callback,
                        )
                elif sampling_config.sampling_type == SamplingType.SUBGRAPH:
                    callback = _make_batch_done_callback(
                        start_cmd.channel_id, start_cmd.epoch
                    )
                    for index in loader:
                        dist_sampler.subgraph(
                            local_input[index],
                            key=route_key,
                            callback=callback,
                        )

            elif command == SharedMpCommand.UNREGISTER_INPUT:
                channel_id = payload
                route_key = None
                with _lock_for(channel_id):
                    has_inflight = any(
                        cid == channel_id and count > 0
                        for (cid, _), count in pending.items()
                    )
                    if has_inflight:
                        removing.add(channel_id)
                    else:
                        route_key = route_key_by_channel.pop(channel_id, None)
                        channels.pop(channel_id, None)
                        inputs.pop(channel_id, None)
                        cfgs.pop(channel_id, None)
                        started_epoch.pop(channel_id, None)
                if not has_inflight and route_key is not None:
                    dist_sampler.unregister_output(route_key)

            elif command == SharedMpCommand.STOP:
                dist_sampler.wait_all()
                keep_running = False
            else:
                raise RuntimeError(f"Unknown command type: {command}")
    except KeyboardInterrupt:
        pass

    if dist_sampler is not None:
        dist_sampler.shutdown_loop()
    shutdown_rpc(graceful=False)


class SharedDistSamplingBackend:
    def __init__(
        self,
        data: DistDataset,
        worker_options: RemoteDistSamplingWorkerOptions,
        sampling_config: SamplingConfig,
        sampler_options: SamplerOptions,
    ) -> None:
        self.data = data
        self.worker_options = worker_options
        self.worker_options._assign_worker_devices()
        current_ctx = get_context()
        self.worker_options._set_worker_ranks(current_ctx)
        self.num_workers = self.worker_options.num_workers
        self._backend_sampling_config = sampling_config
        self._sampler_options = sampler_options

        self._task_queues: list[mp.Queue] = []
        self._workers: list[BaseProcess] = []
        self._event_queue: Optional[mp.Queue] = None
        self._shutdown = False
        self._initialized = False
        self._lock = threading.RLock()
        self._channel_sampling_config: dict[int, SamplingConfig] = {}
        self._channel_input_sizes: dict[int, list[int]] = {}
        self._channel_epoch: dict[int, int] = {}
        self._completed_workers: dict[tuple[int, int], set[int]] = defaultdict(set)

    def init_backend(self) -> None:
        with self._lock:
            if self._initialized:
                return
            if self._shutdown:
                raise RuntimeError("Cannot initialize a shutdown sampling backend")

            if self._backend_sampling_config.seed is not None:
                seed_everything(self._backend_sampling_config.seed)

            mp_context = mp.get_context("spawn")
            self._event_queue = mp_context.Queue()
            barrier = mp_context.Barrier(self.num_workers + 1)
            for rank in range(self.num_workers):
                task_queue = mp_context.Queue(
                    self.num_workers * self.worker_options.worker_concurrency
                )
                self._task_queues.append(task_queue)
                worker = mp_context.Process(
                    target=_shared_sampling_worker_loop,
                    args=(
                        rank,
                        self.data,
                        self._backend_sampling_config,
                        self.worker_options,
                        task_queue,
                        self._event_queue,
                        barrier,
                        self._sampler_options,
                    ),
                )
                worker.daemon = True
                worker.start()
                self._workers.append(worker)
            barrier.wait()
            self._initialized = True

    def _drain_events(self) -> None:
        if self._event_queue is None:
            return
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if (
                not isinstance(event, tuple)
                or len(event) != 4
                or event[0] != EPOCH_DONE_EVENT
            ):
                continue
            _, channel_id, epoch, worker_rank = event
            with self._lock:
                self._completed_workers[(channel_id, epoch)].add(worker_rank)

    def register_input(
        self,
        channel_id: int,
        worker_key: str,
        sampler_input: Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput],
        sampling_config: SamplingConfig,
        channel: ChannelBase,
    ) -> None:
        self.init_backend()
        if hasattr(sampler_input, "share_memory"):
            sampler_input = sampler_input.share_memory()

        worker_ranges = _compute_worker_seeds_ranges(
            input_len=len(sampler_input),
            batch_size=sampling_config.batch_size,
            num_workers=self.num_workers,
        )
        worker_inputs = [
            _slice_sampler_input(sampler_input, start, end)
            for start, end in worker_ranges
        ]
        for rank, local_input in enumerate(worker_inputs):
            if hasattr(local_input, "share_memory"):
                local_input = local_input.share_memory()
            self._task_queues[rank].put(
                (
                    SharedMpCommand.REGISTER_INPUT,
                    RegisterInputCmd(
                        channel_id=channel_id,
                        worker_key=worker_key,
                        sampler_input=local_input,
                        sampling_config=sampling_config,
                        channel=channel,
                    ),
                )
            )

        with self._lock:
            self._channel_sampling_config[channel_id] = sampling_config
            self._channel_input_sizes[channel_id] = [
                end - start for start, end in worker_ranges
            ]
            self._channel_epoch[channel_id] = -1

    def start_new_epoch_sampling(self, channel_id: int, epoch: int) -> None:
        self._drain_events()
        with self._lock:
            if channel_id not in self._channel_sampling_config:
                return
            if self._channel_epoch[channel_id] >= epoch:
                return
            self._channel_epoch[channel_id] = epoch
            sampling_config = self._channel_sampling_config[channel_id]
            worker_input_sizes = self._channel_input_sizes[channel_id]
            stale_keys = [
                key
                for key in self._completed_workers
                if key[0] == channel_id and key[1] < epoch
            ]
            for key in stale_keys:
                self._completed_workers.pop(key, None)
            self._completed_workers.pop((channel_id, epoch), None)

        for rank, input_len in enumerate(worker_input_sizes):
            seeds_index = None
            if sampling_config.shuffle:
                if input_len == 0:
                    seeds_index = torch.empty(0, dtype=torch.long)
                else:
                    seeds_index = torch.randperm(input_len)
                    seeds_index.share_memory_()
            self._task_queues[rank].put(
                (
                    SharedMpCommand.START_EPOCH,
                    StartEpochCmd(
                        channel_id=channel_id,
                        epoch=epoch,
                        seeds_index=seeds_index,
                    ),
                )
            )
        time.sleep(0.1)

    def unregister_input(self, channel_id: int) -> None:
        self._drain_events()
        with self._lock:
            self._channel_sampling_config.pop(channel_id, None)
            self._channel_input_sizes.pop(channel_id, None)
            self._channel_epoch.pop(channel_id, None)
            keys_to_delete = [
                key for key in self._completed_workers if key[0] == channel_id
            ]
            for key in keys_to_delete:
                self._completed_workers.pop(key, None)
        for queue_ in self._task_queues:
            queue_.put((SharedMpCommand.UNREGISTER_INPUT, channel_id))

    def is_channel_epoch_done(self, channel_id: int, epoch: int) -> bool:
        self._drain_events()
        with self._lock:
            return (
                len(self._completed_workers.get((channel_id, epoch), set()))
                == self.num_workers
            )

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        try:
            for queue_ in self._task_queues:
                queue_.put((SharedMpCommand.STOP, None))
            for worker in self._workers:
                worker.join(timeout=MP_STATUS_CHECK_INTERVAL)
            for queue_ in self._task_queues:
                queue_.cancel_join_thread()
                queue_.close()
            if self._event_queue is not None:
                self._event_queue.cancel_join_thread()
                self._event_queue.close()
        finally:
            for worker in self._workers:
                if worker.is_alive():
                    worker.terminate()
