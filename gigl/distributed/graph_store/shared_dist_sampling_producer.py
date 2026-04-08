"""Shared graph-store sampling backend and fair-queued worker loop.

This module implements the multi-channel sampling backend used in graph-store
mode.  A single ``SharedDistSamplingBackend`` per loader instance manages a
pool of worker processes that service many compute-rank channels through a
fair-queued scheduler (``_shared_sampling_worker_loop``).

High-level architecture::

    ┌──────────────────────────────────────────────┐
    │      SharedDistSamplingBackend               │
    │      (main process)                          │
    ├──────────────────────────────────────────────┤
    │  register_input()                            │
    │  start_new_epoch_sampling()                  │
    │  is_channel_epoch_done()                     │
    │  unregister_input()                          │
    │  shutdown()                                  │
    └──────┬──────────────────────────────▲────────┘
           │ task_queues                  │ event_queue
           │ (SharedMpCommand, payload)   │ (EPOCH_DONE_EVENT,
           │                              │  channel_id, epoch,
           ▼                              │  worker_rank)
    ┌──────────────────────────────────────────────┐
    │  Worker 0 .. N-1                             │
    │  _shared_sampling_worker_loop()              │
    │                                              │
    │  ┌─────────────┐  sample_from_*  ┌─────────┐│
    │  │ Sampler      │───────────────▶│ Channel  ││
    │  │ (per channel)│  (results)     │ (output) ││
    │  └─────────────┘                 └─────────┘│
    └──────────────────────────────────────────────┘

Worker event-loop internals::

    ┌─────────────────────────────────────────────────┐
    │ Phase 1: Drain commands (non-blocking)          │
    │   task_queue.get_nowait() ──▶ _handle_command() │
    │     REGISTER_INPUT  ──▶ create sampler + state  │
    │     START_EPOCH     ──▶ ActiveEpochState        │
    │                          + enqueue to runnable  │
    │     UNREGISTER_INPUT ──▶ cleanup / defer        │
    │     STOP             ──▶ exit loop              │
    ├─────────────────────────────────────────────────┤
    │ Phase 2: Round-robin batch submission            │
    │   for each channel in runnable_channels:        │
    │     pop ──▶ _submit_one_batch()                 │
    │            ──▶ sampler.sample_from_*()           │
    │     if more batches: re-enqueue channel         │
    │                                                 │
    │   completion callback (_on_batch_done):         │
    │     completed_batches += 1                      │
    │     if all done ──▶ EPOCH_DONE to event_queue   │
    ├─────────────────────────────────────────────────┤
    │ Phase 3: Idle wait                              │
    │   if no commands and no batches submitted:      │
    │     task_queue.get(timeout=SCHEDULER_TICK_SECS) │
    └─────────────────────────────────────────────────┘
"""

import datetime
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing.process import BaseProcess
from threading import Barrier
from typing import Optional, Union, cast

import torch
import torch.multiprocessing as mp
from graphlearn_torch.channel import ChannelBase
from graphlearn_torch.distributed import (
    DistDataset,
    RemoteDistSamplingWorkerOptions,
    get_context,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from graphlearn_torch.distributed.dist_sampling_producer import MP_STATUS_CHECK_INTERVAL
from graphlearn_torch.sampler import (
    EdgeSamplerInput,
    NodeSamplerInput,
    SamplingConfig,
    SamplingType,
)
from graphlearn_torch.typing import EdgeType
from torch._C import _set_worker_signal_handlers

from gigl.common.logger import Logger
from gigl.distributed.sampler_options import PPRSamplerOptions, SamplerOptions
from gigl.distributed.utils.dist_sampler import (
    SamplerInput,
    SamplerRuntime,
    create_dist_sampler,
)

logger = Logger()


def _prepare_degree_tensors(
    data: DistDataset,
    sampler_options: SamplerOptions,
) -> Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]:
    """Materialize PPR degree tensors before worker spawn when required.

    Called once in the main process so that degree data is available in shared
    memory before workers fork.  Returns ``None`` for non-PPR sampler options.

    Args:
        data: The distributed dataset whose ``degree_tensor`` property is
            read.
        sampler_options: Sampler configuration.  Degree tensors are only
            materialized when this is a ``PPRSamplerOptions`` instance.

    Returns:
        A single degree tensor (homogeneous graph), a dict mapping edge types
        to degree tensors (heterogeneous graph), or ``None`` if PPR sampling
        is not configured.
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


EPOCH_DONE_EVENT = "EPOCH_DONE"
SCHEDULER_TICK_SECS = 0.05
SCHEDULER_STATE_LOG_INTERVAL_SECS = 10.0
SCHEDULER_STATE_MAX_CHANNELS = 6
SCHEDULER_SLOW_SUBMIT_SECS = 1.0


class SharedMpCommand(Enum):
    """Commands sent from the backend to worker subprocesses via task queues.

    Each command is paired with a payload in a ``(command, payload)`` tuple
    placed on the per-worker ``task_queue``.

    Attributes:
        REGISTER_INPUT: Register a new channel with its sampler input,
            sampling config, and output channel.
            Payload: ``RegisterInputCmd``.
        UNREGISTER_INPUT: Remove a channel and clean up its state.
            Payload: ``int`` (the channel_id).
        START_EPOCH: Begin sampling a new epoch for one channel.
            Payload: ``StartEpochCmd``.
        STOP: Shut down the worker process.
            Payload: ``None``.
    """

    REGISTER_INPUT = auto()
    UNREGISTER_INPUT = auto()
    START_EPOCH = auto()
    STOP = auto()


@dataclass(frozen=True)
class RegisterInputCmd:
    """Payload for ``SharedMpCommand.REGISTER_INPUT``.

    Carries everything a worker needs to set up sampling for one channel.

    Attributes:
        channel_id: Unique identifier for this channel across the backend.
        worker_key: Routing key used to identify this channel in the worker
            group (passed through to ``create_dist_sampler``).
        sampler_input: The full set of seed node/edge inputs for this channel,
            already in shared memory.
        sampling_config: Sampling parameters (batch size, num neighbors, etc.).
        channel: The output channel where sampled subgraphs are written.
    """

    channel_id: int
    worker_key: str
    sampler_input: SamplerInput
    sampling_config: SamplingConfig
    channel: ChannelBase


@dataclass(frozen=True)
class StartEpochCmd:
    """Payload for ``SharedMpCommand.START_EPOCH``.

    Attributes:
        channel_id: The channel whose epoch is starting.
        epoch: Monotonically increasing epoch number.
            Duplicate or stale epochs are silently ignored by the worker.
        seeds_index: Index tensor selecting which seeds from the channel's
            ``sampler_input`` to sample this epoch.
            ``None`` means use the full input range.
    """

    channel_id: int
    epoch: int
    seeds_index: Optional[torch.Tensor]


@dataclass
class ActiveEpochState:
    """Mutable per-channel state for an in-progress epoch inside a worker.

    Created by ``_handle_command`` on ``START_EPOCH`` and removed when all
    batches complete.

    Attributes:
        channel_id: The channel this epoch belongs to.
        epoch: The epoch number.
        input_len: Total number of seed indices assigned to this worker for
            this epoch.
        batch_size: Number of seeds per batch.
        drop_last: If True, the final incomplete batch is skipped.
        seeds_index: Index tensor into the channel's ``sampler_input``.
            ``None`` means sequential indices ``[0, input_len)``.
        total_batches: Pre-computed number of batches for this epoch.
        submitted_batches: Number of batches submitted to the sampler so far.
            Mutated by ``_submit_one_batch``.
        completed_batches: Number of batches whose sampler callbacks have
            fired.  Mutated by ``_on_batch_done``.
        cancelled: Set to True when the channel is unregistered while batches
            are still in flight.  Mutated by ``_clear_registered_input_locked``.
    """

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


def _command_channel_id(command: SharedMpCommand, payload: object) -> Optional[int]:
    """Extract the channel id from a worker command payload.

    Args:
        command: The command type.
        payload: The associated payload — one of ``RegisterInputCmd``,
            ``StartEpochCmd``, ``int`` (channel_id), or ``None``.

    Returns:
        The channel id if the command targets a specific channel,
        or ``None`` for ``STOP``.
    """
    if command == SharedMpCommand.STOP:
        return None
    if isinstance(payload, RegisterInputCmd):
        return payload.channel_id
    if isinstance(payload, StartEpochCmd):
        return payload.channel_id
    if isinstance(payload, int):
        return payload
    return None


def _compute_num_batches(input_len: int, batch_size: int, drop_last: bool) -> int:
    """Compute the number of batches emitted for an input length.

    Args:
        input_len: Total number of seed indices.
        batch_size: Number of seeds per batch.
        drop_last: If True, drops the final batch when it is smaller than
            ``batch_size``.

    Returns:
        The number of batches.  Returns 0 when ``input_len <= 0``.
    """
    if input_len <= 0:
        return 0
    if drop_last:
        return input_len // batch_size
    return (input_len + batch_size - 1) // batch_size


def _epoch_batch_indices(state: ActiveEpochState) -> Optional[torch.Tensor]:
    """Return the next batch of seed indices for an active epoch.

    Advances the logical cursor by one batch based on
    ``state.submitted_batches``.

    Args:
        state: The mutable epoch state for the channel.
            ``submitted_batches`` is read but **not** mutated here — the
            caller (``_submit_one_batch``) increments it after calling.

    Returns:
        A 1-D ``torch.long`` tensor of seed indices for the next batch,
        or ``None`` if no more batches should be submitted (epoch cancelled,
        all batches already submitted, or incomplete final batch with
        ``drop_last=True``).
    """
    if state.cancelled or state.submitted_batches >= state.total_batches:
        return None

    batch_start = state.submitted_batches * state.batch_size
    batch_end = min(batch_start + state.batch_size, state.input_len)
    if state.drop_last and batch_end - batch_start < state.batch_size:
        return None

    if state.seeds_index is None:
        return torch.arange(batch_start, batch_end, dtype=torch.long)
    return state.seeds_index[batch_start:batch_end]


def _compute_worker_seeds_ranges(
    input_len: int, batch_size: int, num_workers: int
) -> list[tuple[int, int]]:
    """Distribute seed indices across workers using GLT-compatible logic.

    Divides complete batches as evenly as possible across workers
    (lower-ranked workers get one extra batch when the division is uneven).
    The last worker's range extends to ``input_len`` so that the remainder
    (incomplete final batch) is included.

    Args:
        input_len: Total number of seed indices.
        batch_size: Number of seeds per batch.
        num_workers: Number of worker processes.

    Returns:
        A list of ``(start, end)`` index ranges, one per worker.  The ranges
        are contiguous and non-overlapping, covering ``[0, input_len)``.
    """
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


def _shared_sampling_worker_loop(
    rank: int,
    data: DistDataset,
    worker_options: RemoteDistSamplingWorkerOptions,
    task_queue: mp.Queue,
    event_queue: mp.Queue,
    mp_barrier: Barrier,
    sampler_options: SamplerOptions,
    degree_tensors: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],
) -> None:
    """Run one shared graph-store worker that schedules many input channels.

    Each worker subprocess runs this function as a fair-queued batch scheduler.
    Multiple input channels (each representing one compute rank's data stream)
    share the same sampling worker processes and graph data.

    Args:
        rank: This worker's index within the pool (``0 .. num_workers-1``).
        data: The distributed dataset, shared across all workers via the
            spawn context.
        worker_options: GLT remote sampling worker configuration (RPC
            addresses, devices, concurrency settings).
        task_queue: Per-worker command queue.  The backend enqueues
            ``(SharedMpCommand, payload)`` tuples; the worker drains them
            in Phase 1 of the event loop.
        event_queue: Shared completion queue.  Workers emit
            ``(EPOCH_DONE_EVENT, channel_id, epoch, worker_rank)`` tuples
            when all batches for an epoch have completed.
        mp_barrier: Synchronization barrier.  The worker signals it after
            RPC initialization is complete so the parent can proceed.
        sampler_options: GiGL sampler configuration (e.g. ``PPRSamplerOptions``
            for PPR-based sampling).
        degree_tensors: Pre-computed degree tensors for PPR sampling, or
            ``None`` for non-PPR samplers.  Materialized once in the parent
            process by ``_prepare_degree_tensors`` and shared across workers.

    Algorithm:
        1. Initialize RPC, sampler infrastructure, and signal the parent via barrier.
        2. Enter the main event loop which alternates between:
           a. Draining all pending commands from ``task_queue`` (register/unregister
              channels, start epochs, stop).
           b. Submitting batches round-robin from ``runnable_channels`` — a FIFO
              queue of channels that have pending work. Each channel gets one batch
              submitted per round to prevent starvation.
           c. If no commands were processed and no batches submitted, blocking on
              ``task_queue`` with a short timeout to avoid busy-waiting.
        3. Completion callbacks from the sampler update per-channel state and emit
           ``EPOCH_DONE_EVENT`` to ``event_queue`` when all batches for an epoch
           are finished.
    """
    samplers: dict[int, SamplerRuntime] = {}
    channels: dict[int, ChannelBase] = {}
    inputs: dict[int, SamplerInput] = {}
    cfgs: dict[int, SamplingConfig] = {}
    route_key_by_channel: dict[int, str] = {}
    started_epoch: dict[int, int] = {}
    active_epochs_by_channel: dict[int, ActiveEpochState] = {}
    runnable_channels: deque[int] = deque()
    runnable_set: set[int] = set()
    removing: set[int] = set()
    state_lock = threading.RLock()
    last_state_log_time = 0.0
    current_device: Optional[torch.device] = None

    # --- Scheduler helper functions ---

    def _enqueue_channel_if_runnable_locked(channel_id: int) -> None:
        """Add channel to the fair-queue if it has pending batches."""
        state = active_epochs_by_channel.get(channel_id)
        if state is None:
            return
        if state.cancelled or state.submitted_batches >= state.total_batches:
            return
        if channel_id in runnable_set:
            return
        runnable_channels.append(channel_id)
        runnable_set.add(channel_id)

    def _clear_registered_input_locked(channel_id: int) -> None:
        """Remove a channel's registration and clean up all associated state.

        If the channel still has in-flight batches (submitted but not yet
        completed), marks it for deferred removal instead of cleaning up
        immediately.
        ``_on_batch_done`` will finish the cleanup once the last in-flight
        batch completes.
        """
        state = active_epochs_by_channel.get(channel_id)
        if state is not None and state.completed_batches < state.submitted_batches:
            removing.add(channel_id)
            state.cancelled = True
            return
        sampler = samplers.pop(channel_id, None)
        if sampler is not None:
            sampler.wait_all()
            sampler.shutdown_loop()
        channels.pop(channel_id, None)
        inputs.pop(channel_id, None)
        cfgs.pop(channel_id, None)
        route_key_by_channel.pop(channel_id, None)
        started_epoch.pop(channel_id, None)
        active_epochs_by_channel.pop(channel_id, None)
        runnable_set.discard(channel_id)
        removing.discard(channel_id)

    def _format_scheduler_state_locked() -> str:
        """Format a human-readable snapshot of the scheduler for logging.

        Must be called while holding ``state_lock``.
        """
        channel_ids = sorted(channels.keys())
        preview = channel_ids[:SCHEDULER_STATE_MAX_CHANNELS]
        previews: list[str] = []
        for channel_id in preview:
            active_epoch = active_epochs_by_channel.get(channel_id)
            if active_epoch is None:
                previews.append(f"{channel_id}:idle")
            else:
                previews.append(
                    f"{channel_id}:e{active_epoch.epoch}"
                    f"/{active_epoch.submitted_batches}"
                    f"/{active_epoch.completed_batches}"
                    f"/{active_epoch.total_batches}"
                )
        extra = ""
        if len(channel_ids) > len(preview):
            extra = f" +{len(channel_ids) - len(preview)}"
        return (
            f"registered={len(channels)} active={len(active_epochs_by_channel)} "
            f"runnable={len(runnable_set)} removing={len(removing)} "
            f"channels=[{', '.join(previews)}]{extra}"
        )

    def _maybe_log_scheduler_state(reason: str, force: bool = False) -> None:
        """Log scheduler state at most once per ``SCHEDULER_STATE_LOG_INTERVAL_SECS``.

        Args:
            reason: Short tag included in the log line (e.g. "start_epoch").
            force: If True, log regardless of the time-based throttle.
        """
        nonlocal last_state_log_time
        now = time.monotonic()
        if not force and now - last_state_log_time < SCHEDULER_STATE_LOG_INTERVAL_SECS:
            return
        with state_lock:
            scheduler_state = _format_scheduler_state_locked()
        logger.info(
            f"shared_sampling_scheduler worker_rank={rank} reason={reason} "
            f"{scheduler_state}"
        )
        last_state_log_time = now

    def _on_batch_done(channel_id: int, epoch: int) -> None:
        """Sampler completion callback — invoked from sampler worker threads.

        Updates the channel's completed-batch counter.
        When all batches for the epoch are done, emits ``EPOCH_DONE_EVENT``
        to ``event_queue``.
        If the channel is pending removal, finishes cleanup via
        ``_clear_registered_input_locked``.
        """
        with state_lock:
            state = active_epochs_by_channel.get(channel_id)
            if state is None or state.epoch != epoch:
                return
            state.completed_batches += 1
            if state.completed_batches == state.total_batches:
                active_epochs_by_channel.pop(channel_id, None)
                event_queue.put((EPOCH_DONE_EVENT, channel_id, epoch, rank))
            if (
                channel_id in removing
                and state.completed_batches == state.submitted_batches
            ):
                _clear_registered_input_locked(channel_id)

    def _submit_one_batch(channel_id: int) -> bool:
        """Submit the next batch for a channel to its sampler.

        Re-enqueues the channel into ``runnable_channels`` if more batches
        remain.
        Returns True if a batch was submitted, False if the channel had no
        pending work.
        """
        with state_lock:
            state = active_epochs_by_channel.get(channel_id)
            if state is None:
                return False
            batch_indices = _epoch_batch_indices(state)
            if batch_indices is None:
                return False
            state.submitted_batches += 1
            cfg = cfgs[channel_id]
            sampler = samplers[channel_id]
            channel_input = inputs[channel_id]
            current_epoch = state.epoch
            if state.submitted_batches < state.total_batches and not state.cancelled:
                runnable_channels.append(channel_id)
                runnable_set.add(channel_id)

        sampler_input = channel_input[batch_indices]

        callback = lambda _: _on_batch_done(channel_id, current_epoch)
        if cfg.sampling_type == SamplingType.NODE:
            sampler.sample_from_nodes(
                cast(NodeSamplerInput, sampler_input), callback=callback
            )
        elif cfg.sampling_type == SamplingType.LINK:
            sampler.sample_from_edges(
                cast(EdgeSamplerInput, sampler_input), callback=callback
            )
        elif cfg.sampling_type == SamplingType.SUBGRAPH:
            sampler.subgraph(cast(NodeSamplerInput, sampler_input), callback=callback)
        else:
            raise RuntimeError(f"Unsupported sampling type: {cfg.sampling_type}")
        return True

    def _pump_runnable_channels() -> bool:
        """Submit one batch per runnable channel in round-robin order.

        Returns True if at least one batch was submitted.
        """
        made_progress = False
        with state_lock:
            num_candidates = len(runnable_channels)
        for _ in range(num_candidates):
            with state_lock:
                if not runnable_channels:
                    break
                channel_id = runnable_channels.popleft()
                runnable_set.discard(channel_id)
            made_progress = _submit_one_batch(channel_id) or made_progress
        return made_progress

    def _handle_command(command: SharedMpCommand, payload: object) -> bool:
        """Dispatch one command from the task queue.

        Returns True to keep running, False on ``STOP``.
        """
        channel_id = _command_channel_id(command, payload)
        if command == SharedMpCommand.REGISTER_INPUT:
            register = cast(RegisterInputCmd, payload)
            assert current_device is not None
            sampler = create_dist_sampler(
                data=data,
                sampling_config=register.sampling_config,
                worker_options=worker_options,
                channel=register.channel,
                sampler_options=sampler_options,
                degree_tensors=degree_tensors,
                current_device=current_device,
            )
            sampler.start_loop()
            with state_lock:
                samplers[register.channel_id] = sampler
                channels[register.channel_id] = register.channel
                inputs[register.channel_id] = register.sampler_input
                cfgs[register.channel_id] = register.sampling_config
                route_key_by_channel[register.channel_id] = register.worker_key
                started_epoch[register.channel_id] = -1
            _maybe_log_scheduler_state("register_input", force=True)
            return True

        if command == SharedMpCommand.START_EPOCH:
            start_epoch = cast(StartEpochCmd, payload)
            with state_lock:
                if channel_id not in channels:
                    return True
                if started_epoch.get(channel_id, -1) >= start_epoch.epoch:
                    return True
                started_epoch[channel_id] = start_epoch.epoch
                sampling_config = cfgs[channel_id]
                local_input_len = (
                    len(start_epoch.seeds_index)
                    if start_epoch.seeds_index is not None
                    else len(inputs[channel_id])
                )
                state = ActiveEpochState(
                    channel_id=channel_id,
                    epoch=start_epoch.epoch,
                    input_len=local_input_len,
                    batch_size=sampling_config.batch_size,
                    drop_last=sampling_config.drop_last,
                    seeds_index=start_epoch.seeds_index,
                    total_batches=_compute_num_batches(
                        local_input_len,
                        sampling_config.batch_size,
                        sampling_config.drop_last,
                    ),
                )
                active_epochs_by_channel[channel_id] = state
                if state.total_batches == 0:
                    active_epochs_by_channel.pop(channel_id, None)
                    event_queue.put(
                        (EPOCH_DONE_EVENT, channel_id, start_epoch.epoch, rank)
                    )
                    return True
                _enqueue_channel_if_runnable_locked(channel_id)
            _maybe_log_scheduler_state("start_epoch", force=True)
            return True

        if command == SharedMpCommand.UNREGISTER_INPUT:
            assert channel_id is not None
            with state_lock:
                _clear_registered_input_locked(channel_id)
            _maybe_log_scheduler_state("unregister_input", force=True)
            return True

        if command == SharedMpCommand.STOP:
            return False

        raise RuntimeError(f"Unknown command type: {command}")

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
        mp_barrier.wait()

        # --- Main event loop ---
        keep_running = True
        while keep_running:
            # Phase 1: Drain all pending commands without blocking.
            processed_command = False
            while keep_running:
                try:
                    command, payload = task_queue.get_nowait()
                except queue.Empty:
                    break
                processed_command = True
                keep_running = _handle_command(command, payload)

            # Phase 2: Submit batches round-robin from runnable channels.
            made_progress = _pump_runnable_channels()
            _maybe_log_scheduler_state("steady_state")
            if not keep_running:
                break

            # Phase 3: If idle (no commands, no batches), block until next command.
            if not (processed_command or made_progress):
                try:
                    command, payload = task_queue.get(timeout=SCHEDULER_TICK_SECS)
                except queue.Empty:
                    continue
                keep_running = _handle_command(command, payload)
    except KeyboardInterrupt:
        pass
    finally:
        for sampler in list(samplers.values()):
            sampler.wait_all()
            sampler.shutdown_loop()
        shutdown_rpc(graceful=False)


class SharedDistSamplingBackend:
    """Shared graph-store sampling backend reused across many remote channels."""

    def __init__(
        self,
        *,
        data: DistDataset,
        worker_options: RemoteDistSamplingWorkerOptions,
        sampling_config: SamplingConfig,
        sampler_options: SamplerOptions,
    ) -> None:
        self.data = data
        self.worker_options = worker_options
        self.num_workers = worker_options.num_workers
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
        self._channel_worker_seeds_ranges: dict[int, list[tuple[int, int]]] = {}
        self._channel_shuffle_generators: dict[int, Optional[torch.Generator]] = {}
        self._channel_epoch: dict[int, int] = {}
        self._completed_workers: defaultdict[tuple[int, int], set[int]] = defaultdict(
            set
        )

    def init_backend(self) -> None:
        """Initialize worker processes once for this backend."""
        with self._lock:
            if self._initialized:
                return
            self.worker_options._assign_worker_devices()
            current_ctx = get_context()
            if current_ctx is None or not current_ctx.is_server():
                raise RuntimeError(
                    "SharedDistSamplingBackend.init_backend() requires a GLT server context."
                )
            self.worker_options._set_worker_ranks(current_ctx)
            degree_tensors = _prepare_degree_tensors(
                self.data,
                self._sampler_options,
            )
            mp_context = mp.get_context("spawn")
            barrier = mp_context.Barrier(self.num_workers + 1)
            self._event_queue = mp_context.Queue()
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
                        self.worker_options,
                        task_queue,
                        self._event_queue,
                        barrier,
                        self._sampler_options,
                        degree_tensors,
                    ),
                )
                worker.daemon = True
                worker.start()
                self._workers.append(worker)
            barrier.wait()
            self._initialized = True

    def _enqueue_worker_command(
        self,
        worker_rank: int,
        command: SharedMpCommand,
        payload: object,
    ) -> None:
        queue_ = self._task_queues[worker_rank]
        enqueue_start = time.monotonic()
        queue_.put((command, payload))
        elapsed = time.monotonic() - enqueue_start
        if elapsed >= SCHEDULER_SLOW_SUBMIT_SECS:
            logger.warning(
                f"task_queue enqueue_slow worker_rank={worker_rank} "
                f"command={command.name} elapsed_secs={elapsed:.2f}"
            )

    def register_input(
        self,
        channel_id: int,
        worker_key: str,
        sampler_input: SamplerInput,
        sampling_config: SamplingConfig,
        channel: ChannelBase,
    ) -> None:
        """Register a channel-specific input on all backend workers."""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("SharedDistSamplingBackend is not initialized.")
            if channel_id in self._channel_sampling_config:
                raise ValueError(f"channel_id {channel_id} is already registered.")
            if sampling_config != self._backend_sampling_config:
                raise ValueError(
                    "Sampling config must match the backend sampling config for shared backends."
                )

            shared_sampler_input = sampler_input.share_memory()
            worker_ranges = _compute_worker_seeds_ranges(
                len(shared_sampler_input),
                sampling_config.batch_size,
                self.num_workers,
            )
            self._channel_sampling_config[channel_id] = sampling_config
            self._channel_input_sizes[channel_id] = [
                end - start for start, end in worker_ranges
            ]
            self._channel_worker_seeds_ranges[channel_id] = worker_ranges
            if sampling_config.shuffle:
                generator = torch.Generator()
                if sampling_config.seed is None:
                    generator.manual_seed(torch.seed())
                else:
                    generator.manual_seed(sampling_config.seed)
                self._channel_shuffle_generators[channel_id] = generator
            else:
                self._channel_shuffle_generators[channel_id] = None
            self._channel_epoch[channel_id] = -1
            for worker_rank in range(self.num_workers):
                self._enqueue_worker_command(
                    worker_rank,
                    SharedMpCommand.REGISTER_INPUT,
                    RegisterInputCmd(
                        channel_id=channel_id,
                        worker_key=worker_key,
                        sampler_input=shared_sampler_input,
                        sampling_config=sampling_config,
                        channel=channel,
                    ),
                )

    def _drain_events(self) -> None:
        """Drain worker completion events into the backend-local state."""
        if self._event_queue is None:
            return
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                return
            if event[0] == EPOCH_DONE_EVENT:
                _, channel_id, epoch, worker_rank = event
                self._completed_workers[(channel_id, epoch)].add(worker_rank)

    def start_new_epoch_sampling(self, channel_id: int, epoch: int) -> None:
        """Start one new epoch for one registered channel."""
        with self._lock:
            self._drain_events()
            sampling_config = self._channel_sampling_config[channel_id]
            if self._channel_epoch[channel_id] >= epoch:
                return
            previous_epoch = self._channel_epoch[channel_id]
            self._channel_epoch[channel_id] = epoch
            stale_keys = [
                k
                for k in self._completed_workers
                if k[0] == channel_id and k[1] <= epoch
            ]
            for k in stale_keys:
                del self._completed_workers[k]
            input_len = sum(self._channel_input_sizes[channel_id])
            worker_ranges = self._channel_worker_seeds_ranges[channel_id]
            if sampling_config.shuffle:
                generator = self._channel_shuffle_generators[channel_id]
                assert generator is not None
                full_index = torch.randperm(input_len, generator=generator)
                for worker_rank, (start, end) in enumerate(worker_ranges):
                    worker_index = full_index[start:end]
                    worker_index.share_memory_()
                    self._enqueue_worker_command(
                        worker_rank,
                        SharedMpCommand.START_EPOCH,
                        StartEpochCmd(
                            channel_id=channel_id,
                            epoch=epoch,
                            seeds_index=worker_index,
                        ),
                    )
            else:
                for worker_rank, (start, end) in enumerate(worker_ranges):
                    worker_index = torch.arange(start, end, dtype=torch.long)
                    worker_index.share_memory_()
                    self._enqueue_worker_command(
                        worker_rank,
                        SharedMpCommand.START_EPOCH,
                        StartEpochCmd(
                            channel_id=channel_id,
                            epoch=epoch,
                            seeds_index=worker_index,
                        ),
                    )

    def unregister_input(self, channel_id: int) -> None:
        """Unregister a channel from the backend workers."""
        with self._lock:
            if channel_id not in self._channel_sampling_config:
                return
            self._drain_events()
            self._channel_sampling_config.pop(channel_id, None)
            self._channel_input_sizes.pop(channel_id, None)
            self._channel_worker_seeds_ranges.pop(channel_id, None)
            self._channel_shuffle_generators.pop(channel_id, None)
            self._channel_epoch.pop(channel_id, None)
            stale_keys = [k for k in self._completed_workers if k[0] == channel_id]
            for k in stale_keys:
                del self._completed_workers[k]
            for worker_rank in range(self.num_workers):
                self._enqueue_worker_command(
                    worker_rank,
                    SharedMpCommand.UNREGISTER_INPUT,
                    channel_id,
                )

    def is_channel_epoch_done(self, channel_id: int, epoch: int) -> bool:
        """Return whether every worker finished the epoch for one channel."""
        with self._lock:
            self._drain_events()
            return (
                len(self._completed_workers.get((channel_id, epoch), set()))
                == self.num_workers
            )

    def describe_channel(self, channel_id: int) -> dict[str, object]:
        """Return lightweight diagnostics for one registered channel."""
        with self._lock:
            self._drain_events()
            epoch = self._channel_epoch.get(channel_id, -1)
            completed_workers = len(
                self._completed_workers.get((channel_id, epoch), set())
            )
            return {
                "epoch": epoch,
                "input_sizes": self._channel_input_sizes.get(channel_id, []),
                "completed_workers": completed_workers,
            }

    def shutdown(self) -> None:
        """Stop all worker processes and release backend resources."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            try:
                for worker_rank in range(len(self._task_queues)):
                    self._enqueue_worker_command(
                        worker_rank,
                        SharedMpCommand.STOP,
                        None,
                    )
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
