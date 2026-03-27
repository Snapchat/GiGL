# Significant portions of this file are taken from GraphLearn-for-PyTorch
# (graphlearn_torch/python/distributed/dist_sampling_producer.py).
# This version uses GiGL's DistNeighborSampler (which supports both standard
# neighbor sampling and ABLP) instead of GLT's DistNeighborSampler.

import datetime
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.process import BaseProcess
from threading import Barrier
from typing import Callable, Optional, Union, cast

import psutil
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


def _log_worker_init_state(rank: int, stage: str, worker_start_time: float) -> None:
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    logger.info(
        f"sampling_worker_init rank={rank} "
        f"stage={stage} "
        f"rss_gb={rss_gb:.3f} "
        f"elapsed_s={time.perf_counter() - worker_start_time:.3f}"
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
    sampler_options: SamplerOptions,
    degree_tensors: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],
):
    dist_sampler = None
    worker_start_time = time.perf_counter()
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

        _log_worker_init_state(
            rank=rank,
            stage="before_init_rpc",
            worker_start_time=worker_start_time,
        )
        init_rpc(
            master_addr=worker_options.master_addr,
            master_port=worker_options.master_port,
            num_rpc_threads=num_rpc_threads,
            rpc_timeout=worker_options.rpc_timeout,
        )
        _log_worker_init_state(
            rank=rank,
            stage="after_init_rpc",
            worker_start_time=worker_start_time,
        )

        if sampling_config.seed is not None:
            seed_everything(sampling_config.seed)

        # Shared args for all sampler types (positional args to DistNeighborSampler.__init__)
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
            dist_sampler = DistNeighborSampler(
                *shared_sampler_args,
                seed=sampling_config.seed,
            )
        elif isinstance(sampler_options, PPRSamplerOptions):
            assert degree_tensors is not None
            dist_sampler = DistPPRNeighborSampler(
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
        _log_worker_init_state(
            rank=rank,
            stage="after_dist_neighbor_sampler",
            worker_start_time=worker_start_time,
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
        # Extract degree tensors before spawning workers.  Worker subprocesses
        # only initialize RPC (not torch.distributed), so the lazy degree
        # computation on GiglDistDataset would fail there.  Computing here —
        # where torch.distributed IS initialized — lets the tensor be shared
        # to workers via IPC.
        degree_tensors: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = None
        if isinstance(self._sampler_options, PPRSamplerOptions):
            assert isinstance(self.data, GiglDistDataset)
            degree_tensors = self.data.degree_tensor
            if isinstance(degree_tensors, dict):
                logger.info(
                    f"Pre-computed degree tensors for PPR sampling across {len(degree_tensors)} edge types."
                )
            else:
                logger.info(
                    f"Pre-computed degree tensor for PPR sampling with {degree_tensors.size(0)} nodes."
                )

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
                    degree_tensors,
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
TASK_DEQUEUED_EVENT = "TASK_DEQUEUED"
SCHEDULER_TICK_SECS = 0.05
SCHEDULER_STATE_LOG_INTERVAL_SECS = 10.0
SCHEDULER_STATE_MAX_CHANNELS = 6
SCHEDULER_SLOW_SUBMIT_SECS = 1.0


def _command_channel_id(command: SharedMpCommand, payload: object) -> Optional[int]:
    if command == SharedMpCommand.REGISTER_INPUT:
        assert isinstance(payload, RegisterInputCmd)
        return payload.channel_id
    if command == SharedMpCommand.START_EPOCH:
        assert isinstance(payload, StartEpochCmd)
        return payload.channel_id
    if command == SharedMpCommand.UNREGISTER_INPUT:
        assert isinstance(payload, int)
        return payload
    return None


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


@dataclass
class ActiveEpochState:
    channel_id: int
    epoch: int
    input_len: int
    batch_size: int
    drop_last: bool
    seeds_index: Optional[torch.Tensor]
    total_batches: int
    submitted_batches: int = 0
    completed_batches: int = 0
    cancelled: bool = False


def _compute_num_batches(
    input_len: int,
    batch_size: int,
    drop_last: bool,
) -> int:
    if input_len <= 0 or batch_size <= 0:
        return 0
    if drop_last:
        return input_len // batch_size
    return (input_len + batch_size - 1) // batch_size


def _epoch_batch_indices(state: ActiveEpochState) -> Optional[torch.Tensor]:
    if state.submitted_batches >= state.total_batches:
        return None
    start = state.submitted_batches * state.batch_size
    end = min(start + state.batch_size, state.input_len)
    if end <= start:
        return None
    if state.seeds_index is not None:
        return state.seeds_index[start:end]
    return torch.arange(start, end, dtype=torch.long)


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
    started_epoch: dict[int, int] = defaultdict(lambda: -1)
    active_epochs_by_channel: dict[int, ActiveEpochState] = {}
    runnable_channels: deque[int] = deque()
    runnable_set: set[int] = set()
    removing: set[int] = set()
    state_lock = threading.RLock()
    worker_inflight_batches = 0
    batches_submitted_total = 0
    batches_completed_total = 0
    epochs_completed_total = 0
    last_scheduler_log_time = 0.0

    def _enqueue_channel_if_runnable_locked(channel_id: int) -> None:
        state = active_epochs_by_channel.get(channel_id)
        if state is None or state.cancelled:
            return
        if state.submitted_batches >= state.total_batches:
            return
        if channel_id in runnable_set:
            return
        runnable_channels.append(channel_id)
        runnable_set.add(channel_id)

    def _clear_registered_input_locked(channel_id: int) -> Optional[str]:
        route_key = route_key_by_channel.pop(channel_id, None)
        channels.pop(channel_id, None)
        inputs.pop(channel_id, None)
        cfgs.pop(channel_id, None)
        active_epochs_by_channel.pop(channel_id, None)
        started_epoch.pop(channel_id, None)
        removing.discard(channel_id)
        runnable_set.discard(channel_id)
        return route_key

    def _format_scheduler_state_locked() -> str:
        runnable_preview: list[int] = []
        for queued_channel_id in runnable_channels:
            if queued_channel_id not in runnable_set:
                continue
            runnable_preview.append(queued_channel_id)
            if len(runnable_preview) >= SCHEDULER_STATE_MAX_CHANNELS:
                break

        active_states = sorted(
            active_epochs_by_channel.values(),
            key=lambda state: (
                state.submitted_batches - state.completed_batches,
                state.total_batches - state.completed_batches,
                -state.channel_id,
            ),
            reverse=True,
        )
        channel_progress = [
            (
                f"{state.channel_id}:e{state.epoch}:"
                f"done={state.completed_batches}/{state.total_batches}:"
                f"submitted={state.submitted_batches}:"
                f"inflight={max(0, state.submitted_batches - state.completed_batches)}:"
                f"pending_submit={max(0, state.total_batches - state.submitted_batches)}:"
                f"cancelled={int(state.cancelled)}"
            )
            for state in active_states[:SCHEDULER_STATE_MAX_CHANNELS]
        ]
        removing_preview = sorted(removing)[:SCHEDULER_STATE_MAX_CHANNELS]
        return (
            f"active_channels={len(active_epochs_by_channel)} "
            f"registered_channels={len(channels)} "
            f"runnable_depth={len(runnable_set)} "
            f"runnable_preview={runnable_preview} "
            f"removing_preview={removing_preview} "
            f"inflight={worker_inflight_batches}/{worker_options.worker_concurrency} "
            f"submitted_total={batches_submitted_total} "
            f"completed_total={batches_completed_total} "
            f"epochs_completed_total={epochs_completed_total} "
            f"channel_progress={channel_progress if channel_progress else '[]'}"
        )

    def _maybe_log_scheduler_state(reason: str, force: bool = False) -> None:
        nonlocal last_scheduler_log_time
        now = time.monotonic()
        with state_lock:
            has_work = (
                bool(active_epochs_by_channel)
                or bool(runnable_set)
                or worker_inflight_batches > 0
                or bool(removing)
            )
            if not force:
                if not has_work:
                    return
                if now - last_scheduler_log_time < SCHEDULER_STATE_LOG_INTERVAL_SECS:
                    return
            snapshot = _format_scheduler_state_locked()
            last_scheduler_log_time = now
        logger.info(
            f"sampling_scheduler_state worker_rank={rank} reason={reason} {snapshot}"
        )

    def _on_batch_done(channel_id: int, epoch: int) -> None:
        nonlocal worker_inflight_batches, batches_completed_total, epochs_completed_total
        route_key_to_remove: Optional[str] = None
        epoch_done = False
        with state_lock:
            worker_inflight_batches = max(0, worker_inflight_batches - 1)
            state = active_epochs_by_channel.get(channel_id)
            if state is not None and state.epoch == epoch:
                state.completed_batches += 1
                batches_completed_total += 1
                if state.cancelled:
                    if state.completed_batches == state.submitted_batches:
                        route_key_to_remove = _clear_registered_input_locked(channel_id)
                elif state.completed_batches >= state.total_batches:
                    active_epochs_by_channel.pop(channel_id, None)
                    runnable_set.discard(channel_id)
                    epoch_done = True
                    epochs_completed_total += 1
                    if channel_id in removing:
                        route_key_to_remove = _clear_registered_input_locked(channel_id)
                else:
                    _enqueue_channel_if_runnable_locked(channel_id)
        if route_key_to_remove is not None and dist_sampler is not None:
            dist_sampler.unregister_output(route_key_to_remove)
        if epoch_done:
            event_queue.put((EPOCH_DONE_EVENT, channel_id, epoch, rank))
        _maybe_log_scheduler_state("batch_done")

    def _make_batch_done_callback(
        channel_id: int, epoch: int
    ) -> Callable[[Optional[SampleMessage]], None]:
        def _callback(_res: Optional[SampleMessage]) -> None:
            _on_batch_done(channel_id, epoch)

        return _callback

    def _submit_one_batch(channel_id: int) -> bool:
        assert dist_sampler is not None
        nonlocal worker_inflight_batches, batches_submitted_total
        with state_lock:
            state = active_epochs_by_channel.get(channel_id)
            if state is None or state.cancelled:
                return False
            if state.submitted_batches >= state.total_batches:
                return False
            if worker_inflight_batches >= worker_options.worker_concurrency:
                return False
            local_input = inputs.get(channel_id)
            sampling_config = cfgs.get(channel_id)
            route_key = route_key_by_channel.get(channel_id)
            if local_input is None or sampling_config is None or route_key is None:
                return False
            batch_indices = _epoch_batch_indices(state)
            if batch_indices is None:
                return False
            epoch = state.epoch
            state.submitted_batches += 1
            batches_submitted_total += 1
            worker_inflight_batches += 1
            should_requeue = (
                not state.cancelled and state.submitted_batches < state.total_batches
            )

        callback = _make_batch_done_callback(channel_id, epoch)
        submit_start_time = time.monotonic()
        try:
            if sampling_config.sampling_type == SamplingType.NODE:
                dist_sampler.sample_from_nodes(
                    local_input[batch_indices],
                    key=route_key,
                    callback=callback,
                )
            elif sampling_config.sampling_type == SamplingType.LINK:
                dist_sampler.sample_from_edges(
                    local_input[batch_indices],
                    key=route_key,
                    callback=callback,
                )
            elif sampling_config.sampling_type == SamplingType.SUBGRAPH:
                dist_sampler.subgraph(
                    local_input[batch_indices],
                    key=route_key,
                    callback=callback,
                )
            else:
                raise RuntimeError(
                    f"Unsupported sampling type: {sampling_config.sampling_type}"
                )
        except Exception:
            with state_lock:
                worker_inflight_batches = max(0, worker_inflight_batches - 1)
                state = active_epochs_by_channel.get(channel_id)
                if state is not None and state.epoch == epoch:
                    state.submitted_batches = max(0, state.submitted_batches - 1)
                    _enqueue_channel_if_runnable_locked(channel_id)
            raise
        submit_elapsed = time.monotonic() - submit_start_time
        if submit_elapsed >= SCHEDULER_SLOW_SUBMIT_SECS:
            _maybe_log_scheduler_state("slow_submit", force=True)
            logger.warning(
                "sampling_scheduler_submit_slow "
                f"worker_rank={rank} channel_id={channel_id} epoch={epoch} "
                f"elapsed={submit_elapsed:.4f}s"
            )

        if should_requeue:
            with state_lock:
                state = active_epochs_by_channel.get(channel_id)
                if state is not None and state.epoch == epoch:
                    _enqueue_channel_if_runnable_locked(channel_id)
        return True

    def _pump_runnable_channels() -> bool:
        with state_lock:
            concurrency_limit = worker_options.worker_concurrency
        made_progress = False
        while True:
            with state_lock:
                if worker_inflight_batches >= concurrency_limit:
                    return made_progress
                if not runnable_channels:
                    return made_progress
                channel_id = runnable_channels.popleft()
                runnable_set.discard(channel_id)
            if _submit_one_batch(channel_id):
                made_progress = True

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
            processed_command = False

            def _handle_command(command: SharedMpCommand, payload: object) -> bool:
                assert dist_sampler is not None
                nonlocal worker_inflight_batches
                event_queue.put(
                    (
                        TASK_DEQUEUED_EVENT,
                        rank,
                        command.name,
                        _command_channel_id(command, payload),
                    )
                )

                if command == SharedMpCommand.REGISTER_INPUT:
                    register_cmd = cast(RegisterInputCmd, payload)
                    route_key = str(register_cmd.channel_id)
                    with state_lock:
                        channels[register_cmd.channel_id] = register_cmd.channel
                        inputs[register_cmd.channel_id] = register_cmd.sampler_input
                        cfgs[register_cmd.channel_id] = register_cmd.sampling_config
                        route_key_by_channel[register_cmd.channel_id] = route_key
                        started_epoch[register_cmd.channel_id] = -1
                        active_epochs_by_channel.pop(register_cmd.channel_id, None)
                        removing.discard(register_cmd.channel_id)
                        runnable_set.discard(register_cmd.channel_id)
                    dist_sampler.register_output(route_key, register_cmd.channel)
                    logger.info(
                        f"worker_rank={rank} registered channel_id={register_cmd.channel_id} worker_key={register_cmd.worker_key}"
                    )
                    _maybe_log_scheduler_state("register_input", force=True)
                    return True

                if command == SharedMpCommand.START_EPOCH:
                    start_cmd = cast(StartEpochCmd, payload)
                    epoch_done = False
                    input_len = 0
                    total_batches = 0
                    with state_lock:
                        if start_cmd.channel_id not in inputs:
                            return True
                        if started_epoch[start_cmd.channel_id] >= start_cmd.epoch:
                            return True
                        active_state = active_epochs_by_channel.get(
                            start_cmd.channel_id
                        )
                        if active_state is not None:
                            logger.warning(
                                "worker_rank=%s channel_id=%s ignoring START_EPOCH epoch=%s while epoch=%s is still active",
                                rank,
                                start_cmd.channel_id,
                                start_cmd.epoch,
                                active_state.epoch,
                            )
                            return True
                        local_input = inputs[start_cmd.channel_id]
                        sampling_config = cfgs[start_cmd.channel_id]
                        input_len = len(local_input)
                        total_batches = _compute_num_batches(
                            input_len=input_len,
                            batch_size=sampling_config.batch_size,
                            drop_last=sampling_config.drop_last,
                        )
                        started_epoch[start_cmd.channel_id] = start_cmd.epoch
                        if total_batches == 0:
                            epoch_done = True
                        else:
                            active_epochs_by_channel[
                                start_cmd.channel_id
                            ] = ActiveEpochState(
                                channel_id=start_cmd.channel_id,
                                epoch=start_cmd.epoch,
                                input_len=len(local_input),
                                batch_size=sampling_config.batch_size,
                                drop_last=sampling_config.drop_last,
                                seeds_index=start_cmd.seeds_index,
                                total_batches=total_batches,
                            )
                            _enqueue_channel_if_runnable_locked(start_cmd.channel_id)
                    logger.info(
                        "sampling_scheduler_epoch_start "
                        f"worker_rank={rank} channel_id={start_cmd.channel_id} "
                        f"epoch={start_cmd.epoch} input_len={input_len} "
                        f"batch_size={sampling_config.batch_size} "
                        f"total_batches={total_batches}"
                    )
                    _maybe_log_scheduler_state("start_epoch", force=True)
                    if epoch_done:
                        event_queue.put(
                            (
                                EPOCH_DONE_EVENT,
                                start_cmd.channel_id,
                                start_cmd.epoch,
                                rank,
                            )
                        )
                    return True

                if command == SharedMpCommand.UNREGISTER_INPUT:
                    channel_id = cast(int, payload)
                    cleared_route_key: Optional[str] = None
                    with state_lock:
                        state = active_epochs_by_channel.get(channel_id)
                        if state is not None:
                            state.cancelled = True
                            removing.add(channel_id)
                            runnable_set.discard(channel_id)
                            has_inflight = (
                                state.completed_batches < state.submitted_batches
                            )
                            if not has_inflight:
                                cleared_route_key = _clear_registered_input_locked(
                                    channel_id
                                )
                        else:
                            cleared_route_key = _clear_registered_input_locked(
                                channel_id
                            )
                    if cleared_route_key is not None:
                        dist_sampler.unregister_output(cleared_route_key)
                    _maybe_log_scheduler_state("unregister_input", force=True)
                    return True

                if command == SharedMpCommand.STOP:
                    _maybe_log_scheduler_state("stop", force=True)
                    dist_sampler.wait_all()
                    return False

                raise RuntimeError(f"Unknown command type: {command}")

            while keep_running:
                try:
                    command, payload = task_queue.get_nowait()
                except queue.Empty:
                    break
                processed_command = True
                keep_running = _handle_command(command, payload)

            if not keep_running:
                break

            made_progress = _pump_runnable_channels()
            _maybe_log_scheduler_state("steady_state")

            if processed_command or made_progress:
                continue

            try:
                command, payload = task_queue.get(timeout=SCHEDULER_TICK_SECS)
            except queue.Empty:
                _maybe_log_scheduler_state("idle_wait")
                continue
            keep_running = _handle_command(command, payload)
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
        self._queued_task_counts_by_worker: dict[int, dict[str, int]] = defaultdict(
            dict
        )
        self._task_queue_maxsize = 0  # unbounded

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
                task_queue = mp_context.Queue(self._task_queue_maxsize)
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

    @staticmethod
    def _safe_qsize(queue_: mp.Queue) -> Optional[int]:
        try:
            return queue_.qsize()
        except (AttributeError, NotImplementedError, OSError):
            return None

    def _snapshot_queue_counts(self, worker_rank: int) -> dict[str, int]:
        return dict(self._queued_task_counts_by_worker.get(worker_rank, {}))

    def describe_channel(self, channel_id: int) -> dict[str, object]:
        self._drain_events()
        with self._lock:
            epoch = self._channel_epoch.get(channel_id)
            if epoch is None:
                return {
                    "channel_id": channel_id,
                    "registered": False,
                }
            completed_workers = sorted(
                self._completed_workers.get((channel_id, epoch), set())
            )
            queue_sizes = {
                rank: qsize
                for rank, qsize in (
                    (rank, self._safe_qsize(queue_))
                    for rank, queue_ in enumerate(self._task_queues)
                )
                if qsize is not None
            }
            queued_task_counts = {
                rank: dict(counts)
                for rank, counts in self._queued_task_counts_by_worker.items()
                if counts
            }
            return {
                "channel_id": channel_id,
                "registered": True,
                "epoch": epoch,
                "worker_input_sizes": list(
                    self._channel_input_sizes.get(channel_id, [])
                ),
                "completed_workers": completed_workers,
                "completed_worker_count": len(completed_workers),
                "num_workers": self.num_workers,
                "approx_queue_sizes": queue_sizes,
                "approx_task_counts": queued_task_counts,
            }

    def _enqueue_worker_command(
        self,
        worker_rank: int,
        command: SharedMpCommand,
        payload: object,
    ) -> None:
        queue_ = self._task_queues[worker_rank]
        channel_id = _command_channel_id(command, payload)
        approx_qsize_before = self._safe_qsize(queue_)
        with self._lock:
            queued_counts_before = self._snapshot_queue_counts(worker_rank)
        logger.info(
            "task_queue enqueue_start "
            f"worker_rank={worker_rank} command={command.name} channel_id={channel_id} "
            f"approx_qsize={approx_qsize_before} maxsize={self._task_queue_maxsize} "
            f"approx_contents={queued_counts_before}"
        )
        with self._lock:
            worker_counts = self._queued_task_counts_by_worker.setdefault(
                worker_rank, {}
            )
            worker_counts[command.name] = worker_counts.get(command.name, 0) + 1
        enqueue_start_time = time.monotonic()
        try:
            queue_.put((command, payload))
        except Exception:
            with self._lock:
                existing_worker_counts = self._queued_task_counts_by_worker.get(
                    worker_rank
                )
                if existing_worker_counts is not None:
                    current_count = existing_worker_counts.get(command.name, 0)
                    if current_count <= 1:
                        existing_worker_counts.pop(command.name, None)
                    else:
                        existing_worker_counts[command.name] = current_count - 1
            raise
        enqueue_elapsed = time.monotonic() - enqueue_start_time
        approx_qsize_after = self._safe_qsize(queue_)
        with self._lock:
            queued_counts_after = self._snapshot_queue_counts(worker_rank)
        logger.info(
            "task_queue enqueue_done "
            f"worker_rank={worker_rank} command={command.name} channel_id={channel_id} "
            f"put_elapsed={enqueue_elapsed:.4f}s approx_qsize={approx_qsize_after} "
            f"maxsize={self._task_queue_maxsize} approx_contents={queued_counts_after}"
        )

    def _drain_events(self) -> None:
        if self._event_queue is None:
            return
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(event, tuple) or len(event) != 4:
                continue
            if event[0] == EPOCH_DONE_EVENT:
                _, channel_id, epoch, worker_rank = event
                with self._lock:
                    self._completed_workers[(channel_id, epoch)].add(worker_rank)
                continue
            if event[0] == TASK_DEQUEUED_EVENT:
                _, worker_rank, command_name, _channel_id = event
                with self._lock:
                    worker_counts = self._queued_task_counts_by_worker.get(worker_rank)
                    if worker_counts is None:
                        continue
                    current_count = worker_counts.get(command_name, 0)
                    if current_count <= 1:
                        worker_counts.pop(command_name, None)
                    else:
                        worker_counts[command_name] = current_count - 1
                continue

    def register_input(
        self,
        channel_id: int,
        worker_key: str,
        sampler_input: Union[NodeSamplerInput, EdgeSamplerInput, ABLPNodeSamplerInput],
        sampling_config: SamplingConfig,
        channel: ChannelBase,
    ) -> None:
        self.init_backend()
        self._drain_events()
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
            self._enqueue_worker_command(
                rank,
                SharedMpCommand.REGISTER_INPUT,
                RegisterInputCmd(
                    channel_id=channel_id,
                    worker_key=worker_key,
                    sampler_input=local_input,
                    sampling_config=sampling_config,
                    channel=channel,
                ),
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
            self._enqueue_worker_command(
                rank,
                SharedMpCommand.START_EPOCH,
                StartEpochCmd(
                    channel_id=channel_id,
                    epoch=epoch,
                    seeds_index=seeds_index,
                ),
            )
        logger.info(
            "shared_sampling_backend start_epoch "
            f"channel_id={channel_id} epoch={epoch} "
            f"state={self.describe_channel(channel_id)}"
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
        for rank in range(len(self._task_queues)):
            self._enqueue_worker_command(
                rank, SharedMpCommand.UNREGISTER_INPUT, channel_id
            )

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
            for rank in range(len(self._task_queues)):
                self._enqueue_worker_command(rank, SharedMpCommand.STOP, None)
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
