# Significant portions of this file are taken from GraphLearn-for-PyTorch
# (graphlearn_torch/python/distributed/dist_sampling_producer.py).
# This version uses GiGL's sampler hierarchy (BaseGiGLSampler subclasses:
# DistNeighborSampler for k-hop, DistPPRNeighborSampler for PPR) instead of
# GLT's DistNeighborSampler directly.

import datetime
import os
import queue
import socket
import time
import traceback
from dataclasses import dataclass, field, replace
from multiprocessing.connection import Connection
from threading import Barrier, BrokenBarrierError
from typing import Optional, Union, cast

import torch
import torch.multiprocessing as mp
from graphlearn_torch.channel import ChannelBase
from graphlearn_torch.distributed import (
    DistDataset,
    DistMpSamplingProducer,
    MpDistSamplingWorkerOptions,
    all_gather,
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
from graphlearn_torch.typing import NodeType
from graphlearn_torch.utils import seed_everything
from torch._C import _set_worker_signal_handlers
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from gigl.common.logger import Logger
from gigl.distributed.base_sampler import _get_sampler_timing_log_every_n
from gigl.distributed.sampler_options import SamplerOptions
from gigl.distributed.utils.dist_sampler import create_dist_sampler

logger = Logger()

SAMPLING_PORT_LEASE_CLOSE_ATTEMPTS = 3
UINT32_MODULUS = 1 << 32


def derive_sampling_worker_seed(
    *,
    run_seed: int,
    parent_global_rank: int,
    parent_world_size: int,
    worker_index: int,
    workers_per_parent: int,
) -> tuple[int, int]:
    """Return a collision-free uint32 seed and stable global sampler id.

    ``run_seed`` is one explicit, persisted value for the whole run.  The global
    sampler id uses the parent inference rank rather than an RPC-group rank because
    colocated local-rank cohorts have independent RPC groups whose ranks repeat.
    Addition modulo ``2**32`` is a permutation, so distinct sampler ids remain
    distinct for every supported run.
    """
    values = {
        "run_seed": run_seed,
        "parent_global_rank": parent_global_rank,
        "parent_world_size": parent_world_size,
        "worker_index": worker_index,
        "workers_per_parent": workers_per_parent,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values.values()
    ):
        raise TypeError(f"sampling seed identity values must be integers, got {values}")
    if run_seed not in range(UINT32_MODULUS):
        raise ValueError(f"run_seed must be uint32, got {run_seed}")
    if parent_world_size <= 0:
        raise ValueError(f"parent_world_size must be positive, got {parent_world_size}")
    if parent_global_rank not in range(parent_world_size):
        raise ValueError(
            "parent_global_rank must be in "
            f"[0, {parent_world_size}), got {parent_global_rank}"
        )
    if workers_per_parent <= 0:
        raise ValueError(
            f"workers_per_parent must be positive, got {workers_per_parent}"
        )
    if worker_index not in range(workers_per_parent):
        raise ValueError(
            f"worker_index must be in [0, {workers_per_parent}), got {worker_index}"
        )
    sampler_world_size = parent_world_size * workers_per_parent
    if sampler_world_size > UINT32_MODULUS:
        raise ValueError(
            "sampling worker world cannot exceed the uint32 seed space, got "
            f"{sampler_world_size}"
        )
    global_sampler_id = parent_global_rank * workers_per_parent + worker_index
    return (run_seed + global_sampler_id) % UINT32_MODULUS, global_sampler_id


@dataclass(frozen=True)
class SamplingWorkerRpcSpec:
    """RPC identity for one isolated sampling-worker subprocess."""

    worker_index: int
    group_name: str
    world_size: int
    rank: int
    master_port: int


@dataclass(frozen=True)
class SamplingWorkerSeedSpec:
    """Stable run and subprocess identity for one installed sampler seed."""

    run_seed: int
    parent_global_rank: int
    parent_world_size: int
    worker_index: int
    workers_per_parent: int
    global_sampler_id: int
    worker_seed: int


@dataclass(frozen=True)
class SamplingWorkerStatus:
    """Structured child-to-parent initialization status."""

    state: str
    worker_index: int
    group_name: str
    rank: int
    master_port: int
    pid: Optional[int]
    phase: str
    elapsed_seconds: float
    error: Optional[str] = None


@dataclass
class SamplingPortLease:
    """Parent-owned locks and master-side sockets for isolated RPC ports."""

    ports: tuple[int, ...]
    lock_file_descriptors: tuple[int, ...] = ()
    reservations: dict[int, socket.socket] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False, repr=False)

    def release_reservation(self, port: int) -> None:
        reservation = self.reservations.get(port)
        if reservation is not None:
            reservation.close()
            del self.reservations[port]

    def close(self) -> None:
        if self._closed:
            return
        errors: list[str] = []
        for port, reservation in list(self.reservations.items()):
            try:
                reservation.close()
            except OSError as error:
                errors.append(f"reservation port {port}: {error}")
            else:
                del self.reservations[port]
        remaining_file_descriptors: list[int] = []
        for file_descriptor in self.lock_file_descriptors:
            try:
                os.close(file_descriptor)
            except OSError as error:
                errors.append(f"lock fd {file_descriptor}: {error}")
                remaining_file_descriptors.append(file_descriptor)
        self.lock_file_descriptors = tuple(remaining_file_descriptors)
        self._closed = not self.reservations and not self.lock_file_descriptors
        if errors:
            logger.warning(
                "isolated sampling port lease cleanup was incomplete; "
                f"retryable_errors={errors}"
            )


def close_sampling_port_lease_with_retries(
    lease: SamplingPortLease,
    *,
    context: str,
    max_attempts: int = SAMPLING_PORT_LEASE_CLOSE_ATTEMPTS,
) -> bool:
    """Boundedly retry a lease that has no later lifecycle owner."""
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    for attempt in range(1, max_attempts + 1):
        try:
            lease.close()
        except BaseException:
            logger.exception(
                "isolated sampling port lease cleanup attempt raised "
                f"context={context} attempt={attempt}/{max_attempts}"
            )
        if lease._closed:
            return True
    logger.error(
        "isolated sampling port lease remains open after bounded cleanup "
        f"context={context} attempts={max_attempts} ports={lease.ports} "
        f"reservations={sorted(lease.reservations)} "
        f"lock_fds={lease.lock_file_descriptors}"
    )
    return False


def validate_isolated_sampling_group_readiness(
    readiness: dict[str, dict[str, int]], rpc_spec: SamplingWorkerRpcSpec
) -> None:
    """Require an exact worker-name, identity, and partition map for one group."""
    expected_worker_names = {
        f"{rpc_spec.group_name}_{rank}" for rank in range(rpc_spec.world_size)
    }
    if set(readiness) != expected_worker_names:
        raise RuntimeError(
            f"isolated sampler group {rpc_spec.worker_index} has invalid worker "
            f"names: expected {expected_worker_names}, got {set(readiness)}"
        )
    invalid_identity = {
        worker_name: payload
        for worker_name, payload in readiness.items()
        if payload.get("worker_index") != rpc_spec.worker_index
        or payload.get("port") != rpc_spec.master_port
        or worker_name != f"{rpc_spec.group_name}_{payload.get('rank')}"
    }
    if invalid_identity:
        raise RuntimeError(
            f"isolated sampler group {rpc_spec.worker_index} has invalid worker "
            f"identity payloads: {invalid_identity}"
        )
    observed = {
        (payload.get("rank"), payload.get("partition"))
        for payload in readiness.values()
    }
    expected = {(group_rank, group_rank) for group_rank in range(rpc_spec.world_size)}
    if observed != expected:
        raise RuntimeError(
            f"isolated sampler group {rpc_spec.worker_index} has invalid "
            f"rank/partition coverage: expected {expected}, got {observed}"
        )


def resolve_isolated_sampling_worker_rpc_specs(
    *,
    parent_world_size: int,
    parent_rank: int,
    parent_group_name: str,
    data_num_partitions: int,
    data_partition_idx: int,
    num_workers: int,
    master_ports: list[int],
) -> tuple[SamplingWorkerRpcSpec, ...]:
    """Build and validate one independent RPC group per sampling worker."""
    if parent_world_size <= 0:
        raise ValueError(f"parent_world_size must be positive, got {parent_world_size}")
    if parent_rank not in range(parent_world_size):
        raise ValueError(
            f"parent_rank must be in [0, {parent_world_size}), got {parent_rank}"
        )
    if data_num_partitions != parent_world_size:
        raise ValueError(
            "isolated sampling RPC groups require one parent per data partition, "
            f"got {data_num_partitions} partitions and parent world {parent_world_size}"
        )
    if data_partition_idx != parent_rank:
        raise ValueError(
            "isolated sampling RPC groups require parent rank to match data "
            f"partition, got rank {parent_rank} and partition {data_partition_idx}"
        )
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    if len(master_ports) != num_workers:
        raise ValueError(
            f"expected {num_workers} isolated sampling ports, got {master_ports}"
        )
    if len(set(master_ports)) != len(master_ports):
        raise ValueError(
            f"isolated sampling ports must be distinct, got {master_ports}"
        )
    if any(port <= 0 or port > 65535 for port in master_ports):
        raise ValueError(
            f"isolated sampling ports must be valid TCP ports: {master_ports}"
        )

    return tuple(
        SamplingWorkerRpcSpec(
            worker_index=worker_index,
            group_name=f"{parent_group_name}_sampling_worker_{worker_index}",
            world_size=parent_world_size,
            rank=parent_rank,
            master_port=master_port,
        )
        for worker_index, master_port in enumerate(master_ports)
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
    degree_tensors: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]],
    sampling_worker_seed_spec: Optional[SamplingWorkerSeedSpec] = None,
    rpc_spec: Optional[SamplingWorkerRpcSpec] = None,
    status_connection: Optional[Connection] = None,
):
    dist_sampler = None
    initialization_start = time.monotonic()
    phase = "worker_group"
    try:
        worker_world_size = (
            rpc_spec.world_size
            if rpc_spec is not None
            else worker_options.worker_world_size
        )
        worker_rank = (
            rpc_spec.rank if rpc_spec is not None else worker_options.worker_ranks[rank]
        )
        group_name = (
            rpc_spec.group_name
            if rpc_spec is not None
            else "_sampling_worker_subprocess"
        )
        master_port = (
            rpc_spec.master_port if rpc_spec is not None else worker_options.master_port
        )
        init_worker_group(
            world_size=worker_world_size,
            rank=worker_rank,
            group_name=group_name,
        )
        if worker_options.use_all2all:
            torch.distributed.init_process_group(
                backend="gloo",
                timeout=datetime.timedelta(seconds=worker_options.rpc_timeout),
                rank=worker_rank,
                world_size=worker_world_size,
                init_method="tcp://{}:{}".format(
                    worker_options.master_addr, master_port
                ),
            )

        if worker_options.num_rpc_threads is None:
            num_rpc_threads = min(data.num_partitions, 16)
        else:
            num_rpc_threads = worker_options.num_rpc_threads

        current_device = worker_options.worker_devices[rank]

        _set_worker_signal_handlers()
        torch.set_num_threads(num_rpc_threads + 1)

        phase = "rpc_init"
        init_rpc(
            master_addr=worker_options.master_addr,
            master_port=master_port,
            num_rpc_threads=num_rpc_threads,
            rpc_timeout=worker_options.rpc_timeout,
        )

        if sampling_config.seed is not None:
            seed_everything(sampling_config.seed)

        phase = "sampler_construction"
        dist_sampler = create_dist_sampler(
            data=data,
            sampling_config=sampling_config,
            worker_options=worker_options,
            channel=channel,
            sampler_options=sampler_options,
            degree_tensors=degree_tensors,
            current_device=current_device,
        )
        if sampling_worker_seed_spec is not None:
            if sampling_config.seed != sampling_worker_seed_spec.worker_seed:
                raise RuntimeError(
                    "sampling worker seed/config mismatch: "
                    f"spec={sampling_worker_seed_spec} "
                    f"config_seed={sampling_config.seed}"
                )
            logger.info(
                "sampling_worker_seed_installed "
                f"run_seed={sampling_worker_seed_spec.run_seed} "
                f"parent_global_rank={sampling_worker_seed_spec.parent_global_rank}/"
                f"{sampling_worker_seed_spec.parent_world_size} "
                f"worker_index={sampling_worker_seed_spec.worker_index}/"
                f"{sampling_worker_seed_spec.workers_per_parent} "
                f"global_sampler_id={sampling_worker_seed_spec.global_sampler_id} "
                f"worker_seed={sampling_worker_seed_spec.worker_seed} pid={os.getpid()}"
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

        if rpc_spec is not None:
            phase = "group_readiness"
            readiness = all_gather(
                {
                    "worker_index": rpc_spec.worker_index,
                    "rank": rpc_spec.rank,
                    "partition": data.partition_idx,
                    "port": rpc_spec.master_port,
                },
                timeout=worker_options.rpc_timeout,
            )
            validate_isolated_sampling_group_readiness(readiness, rpc_spec)
            assert status_connection is not None
            status_connection.send(
                SamplingWorkerStatus(
                    state="READY",
                    worker_index=rpc_spec.worker_index,
                    group_name=rpc_spec.group_name,
                    rank=rpc_spec.rank,
                    master_port=rpc_spec.master_port,
                    pid=os.getpid(),
                    phase=phase,
                    elapsed_seconds=time.monotonic() - initialization_start,
                )
            )
            logger.info(
                "isolated_sampling_worker_ready "
                f"worker_index={rpc_spec.worker_index} group={rpc_spec.group_name} "
                f"rank={rpc_spec.rank}/{rpc_spec.world_size} "
                f"partition={data.partition_idx} port={rpc_spec.master_port} "
                f"hostname={socket.gethostname()} pid={os.getpid()}"
            )

        phase = "local_barrier"
        if rpc_spec is None:
            mp_barrier.wait()
        else:
            mp_barrier.wait(timeout=worker_options.rpc_timeout)

        phase = "sampling_loop"
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
    except BaseException:
        if rpc_spec is not None and status_connection is not None:
            try:
                status_connection.send(
                    SamplingWorkerStatus(
                        state="ERROR",
                        worker_index=rpc_spec.worker_index,
                        group_name=rpc_spec.group_name,
                        rank=rpc_spec.rank,
                        master_port=rpc_spec.master_port,
                        pid=os.getpid(),
                        phase=phase,
                        elapsed_seconds=time.monotonic() - initialization_start,
                        error=traceback.format_exc(),
                    )
                )
            except (BrokenPipeError, EOFError, OSError):
                pass
        raise
    finally:
        try:
            if dist_sampler is not None:
                dist_sampler.shutdown_loop()
        except BaseException:
            logger.exception("isolated sampling worker sampler shutdown failed")
        finally:
            try:
                shutdown_rpc(graceful=False)
            except BaseException:
                logger.exception("isolated sampling worker RPC shutdown failed")
            finally:
                if status_connection is not None:
                    status_connection.close()


class DistSamplingProducer(DistMpSamplingProducer):
    def __init__(
        self,
        data: DistDataset,
        sampler_input: Union[NodeSamplerInput, EdgeSamplerInput],
        sampling_config: SamplingConfig,
        worker_options: MpDistSamplingWorkerOptions,
        channel: ChannelBase,
        sampler_options: SamplerOptions,
        degree_tensors: Optional[
            Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        ] = None,
        isolated_rpc_specs: Optional[tuple[SamplingWorkerRpcSpec, ...]] = None,
        isolated_port_lease: Optional[SamplingPortLease] = None,
        sampling_run_seed: Optional[int] = None,
        parent_global_rank: Optional[int] = None,
        parent_world_size: Optional[int] = None,
    ):
        self._isolated_port_lease = isolated_port_lease
        try:
            super().__init__(
                data, sampler_input, sampling_config, worker_options, channel
            )
            self._sampler_options = sampler_options
            self._degree_tensors = degree_tensors
            self._isolated_rpc_specs = isolated_rpc_specs
            self._sampling_run_seed = sampling_run_seed
            self._parent_global_rank = parent_global_rank
            self._parent_world_size = parent_world_size
            self._isolated_status_connections: list[Connection] = []
            self._isolated_ready_workers: set[int] = set()
            self._isolated_barrier: Optional[Barrier] = None
            self._isolated_resources_closed = False
            self._isolated_cleanup_complete = False
            if isolated_rpc_specs is not None:
                if worker_options.use_all2all:
                    raise ValueError(
                        "one RPC group per sampling worker does not yet support "
                        "use_all2all"
                    )
                if len(isolated_rpc_specs) != self.num_workers:
                    raise ValueError(
                        f"expected {self.num_workers} isolated RPC specs, got "
                        f"{len(isolated_rpc_specs)}"
                    )
                if worker_options.rpc_timeout <= 0:
                    raise ValueError(
                        f"rpc_timeout must be positive, got {worker_options.rpc_timeout}"
                    )
            elif isolated_port_lease is not None:
                raise ValueError("an isolated port lease requires isolated RPC specs")
            if sampling_run_seed is not None:
                if sampling_config.seed is not None:
                    raise ValueError(
                        "sampling_run_seed cannot be combined with a pre-seeded "
                        "SamplingConfig"
                    )
                if parent_global_rank is None or parent_world_size is None:
                    raise ValueError(
                        "sampling_run_seed requires parent_global_rank and "
                        "parent_world_size"
                    )
                derive_sampling_worker_seed(
                    run_seed=sampling_run_seed,
                    parent_global_rank=parent_global_rank,
                    parent_world_size=parent_world_size,
                    worker_index=0,
                    workers_per_parent=self.num_workers,
                )
        except BaseException:
            if isolated_port_lease is not None:
                close_sampling_port_lease_with_retries(
                    isolated_port_lease,
                    context="DistSamplingProducer.__init__",
                )
            raise

    def _sampling_config_for_worker(
        self, worker_index: int
    ) -> tuple[SamplingConfig, Optional[SamplingWorkerSeedSpec]]:
        if self._sampling_run_seed is None:
            return self.sampling_config, None
        assert self._parent_global_rank is not None
        assert self._parent_world_size is not None
        worker_seed, global_sampler_id = derive_sampling_worker_seed(
            run_seed=self._sampling_run_seed,
            parent_global_rank=self._parent_global_rank,
            parent_world_size=self._parent_world_size,
            worker_index=worker_index,
            workers_per_parent=self.num_workers,
        )
        seed_spec = SamplingWorkerSeedSpec(
            run_seed=self._sampling_run_seed,
            parent_global_rank=self._parent_global_rank,
            parent_world_size=self._parent_world_size,
            worker_index=worker_index,
            workers_per_parent=self.num_workers,
            global_sampler_id=global_sampler_id,
            worker_seed=worker_seed,
        )
        logger.info(
            "sampling_worker_seed_plan "
            f"run_seed={self._sampling_run_seed} "
            f"parent_global_rank={self._parent_global_rank}/"
            f"{self._parent_world_size} worker_index={worker_index}/"
            f"{self.num_workers} global_sampler_id={global_sampler_id} "
            f"worker_seed={worker_seed}"
        )
        return replace(self.sampling_config, seed=worker_seed), seed_spec

    def _close_isolated_resources(self) -> None:
        if self._isolated_resources_closed:
            return
        errors: list[str] = []
        remaining_connections: list[Connection] = []
        for connection in self._isolated_status_connections:
            try:
                connection.close()
            except (OSError, ValueError) as error:
                errors.append(f"status connection: {error}")
                remaining_connections.append(connection)
        self._isolated_status_connections = remaining_connections
        remaining_task_queues = []
        for task_queue in self._task_queues:
            try:
                task_queue.cancel_join_thread()
            except (OSError, ValueError) as error:
                errors.append(f"task queue cancel_join_thread: {error}")
            try:
                task_queue.close()
            except (OSError, ValueError) as error:
                errors.append(f"task queue close: {error}")
                remaining_task_queues.append(task_queue)
        self._task_queues = remaining_task_queues
        if self._isolated_port_lease is not None:
            self._isolated_port_lease.close()
            if not self._isolated_port_lease._closed:
                errors.append("port lease still owns retryable resources")
        self._isolated_resources_closed = (
            not self._isolated_status_connections
            and not self._task_queues
            and (self._isolated_port_lease is None or self._isolated_port_lease._closed)
        )
        if errors:
            logger.warning(
                "isolated sampling parent resource cleanup was incomplete; "
                f"retryable_errors={errors}"
            )

    def _cleanup_isolated_workers(
        self, *, graceful: bool, suppress_errors: bool = False
    ) -> None:
        """Idempotently reap a complete or partially initialized worker prefix."""
        if self._isolated_cleanup_complete:
            return
        errors: list[str] = []
        self._shutdown = True
        if self._isolated_barrier is not None:
            try:
                self._isolated_barrier.abort()
            except (BrokenBarrierError, OSError, ValueError) as error:
                errors.append(f"barrier abort: {error}")
        if graceful:
            for task_queue in self._task_queues:
                try:
                    task_queue.put_nowait((MpCommand.STOP, None))
                except (OSError, ValueError, queue.Full) as error:
                    errors.append(f"STOP enqueue: {error}")

        def join_worker(worker, phase: str) -> None:
            if worker.pid is None:
                return
            try:
                worker.join(timeout=MP_STATUS_CHECK_INTERVAL)
            except (OSError, ValueError) as error:
                errors.append(f"{phase} join pid={worker.pid}: {error}")

        def is_worker_alive(worker, phase: str) -> bool:
            if worker.pid is None:
                return False
            try:
                return worker.is_alive()
            except (OSError, ValueError) as error:
                errors.append(f"{phase} is_alive pid={worker.pid}: {error}")
                return True

        for worker in self._workers:
            join_worker(worker, "initial")
        for worker in self._workers:
            if is_worker_alive(worker, "terminate"):
                try:
                    worker.terminate()
                except (OSError, ValueError) as error:
                    errors.append(f"terminate pid={worker.pid}: {error}")
        for worker in self._workers:
            join_worker(worker, "post-terminate")
        for worker in self._workers:
            if is_worker_alive(worker, "kill"):
                try:
                    worker.kill()
                except (OSError, ValueError) as error:
                    errors.append(f"kill pid={worker.pid}: {error}")
        for worker in self._workers:
            join_worker(worker, "post-kill")
        survivors = [
            worker.pid
            for worker in self._workers
            if is_worker_alive(worker, "survivor-check")
        ]
        self._close_isolated_resources()
        if not self._isolated_resources_closed:
            errors.append("parent resources remain open after cleanup")
        if survivors:
            errors.append(f"live child PIDs remain: {survivors}")
        self._isolated_cleanup_complete = (
            not survivors and self._isolated_resources_closed
        )
        if errors:
            cleanup_error = RuntimeError(
                "isolated sampling cleanup was incomplete: " + "; ".join(errors)
            )
            if suppress_errors:
                logger.error(str(cleanup_error))
            else:
                raise cleanup_error

    def _poll_isolated_worker_statuses(self) -> None:
        for index, (worker, connection) in enumerate(
            zip(self._workers, self._isolated_status_connections)
        ):
            while connection.poll():
                try:
                    status = connection.recv()
                except EOFError as error:
                    raise RuntimeError(
                        f"isolated sampling worker {index} closed its status "
                        "pipe before initialization completed"
                    ) from error
                if not isinstance(status, SamplingWorkerStatus):
                    raise RuntimeError(
                        f"sampling worker {index} sent invalid status {status!r}"
                    )
                assert self._isolated_rpc_specs is not None
                expected_spec = self._isolated_rpc_specs[index]
                expected_identity = (
                    expected_spec.worker_index,
                    expected_spec.group_name,
                    expected_spec.rank,
                    expected_spec.master_port,
                    worker.pid,
                )
                observed_identity = (
                    status.worker_index,
                    status.group_name,
                    status.rank,
                    status.master_port,
                    status.pid,
                )
                if observed_identity != expected_identity:
                    raise RuntimeError(
                        f"sampling worker {index} sent invalid identity: "
                        f"expected={expected_identity}, observed={observed_identity}"
                    )
                if status.state == "ERROR":
                    raise RuntimeError(
                        "isolated sampling worker failed: "
                        f"{status}; traceback:\n{status.error}"
                    )
                if status.state != "READY":
                    raise RuntimeError(
                        f"sampling worker {index} sent unknown state {status.state}"
                    )
                self._isolated_ready_workers.add(index)
            if worker.exitcode is not None:
                raise RuntimeError(
                    f"isolated sampling worker {index} exited with code "
                    f"{worker.exitcode} before initialization completed"
                )

    def _wait_for_isolated_worker_ready(self, worker_index: int) -> None:
        started_at = time.monotonic()
        deadline = started_at + self.worker_options.rpc_timeout
        while True:
            self._poll_isolated_worker_statuses()
            if worker_index in self._isolated_ready_workers:
                return
            if time.monotonic() >= deadline:
                spec = self._isolated_rpc_specs[worker_index]
                raise TimeoutError(
                    "timed out waiting for isolated sampling worker: "
                    f"worker_index={worker_index} group={spec.group_name} "
                    f"rank={spec.rank}/{spec.world_size} port={spec.master_port} "
                    f"elapsed={time.monotonic() - started_at:.3f}s "
                    f"pid={self._workers[worker_index].pid} "
                    f"exitcode={self._workers[worker_index].exitcode}"
                )
            time.sleep(min(MP_STATUS_CHECK_INTERVAL, 0.1))

    def _wait_for_isolated_workers_at_barrier(self) -> None:
        assert self._isolated_barrier is not None
        deadline = time.monotonic() + self.worker_options.rpc_timeout
        while self._isolated_barrier.n_waiting < self.num_workers:
            self._poll_isolated_worker_statuses()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for isolated sampling workers at the "
                    f"local barrier: waiting={self._isolated_barrier.n_waiting}/"
                    f"{self.num_workers}"
                )
            time.sleep(min(MP_STATUS_CHECK_INTERVAL, 0.1))
        self._isolated_barrier.wait(
            timeout=min(float(self.worker_options.rpc_timeout), 5.0)
        )

    def init(self):
        r"""Create the subprocess pool. Init samplers and rpc server."""
        # Fail in the parent before creating a barrier. A worker-side config
        # error before ``barrier.wait()`` would otherwise strand this process.
        _get_sampler_timing_log_every_n()
        if self.sampling_config.seed is not None:
            seed_everything(self.sampling_config.seed)
        if not self.sampling_config.shuffle:
            unshuffled_indexes = self._get_seeds_indexes()
        else:
            unshuffled_indexes = [None] * self.num_workers

        mp_context = mp.get_context("spawn")
        barrier = mp_context.Barrier(self.num_workers + 1)
        if self._isolated_rpc_specs is None:
            for rank in range(self.num_workers):
                worker_sampling_config, worker_seed_spec = (
                    self._sampling_config_for_worker(rank)
                )
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
                        worker_sampling_config,
                        self.worker_options,
                        self.output_channel,
                        task_queue,
                        self.sampling_completed_worker_count,
                        barrier,
                        self._sampler_options,
                        self._degree_tensors,
                        worker_seed_spec,
                    ),
                )
                worker.daemon = True
                worker.start()
                self._workers.append(worker)
            barrier.wait()
            return

        self._isolated_barrier = barrier
        try:
            for rank, rpc_spec in enumerate(self._isolated_rpc_specs):
                worker_sampling_config, worker_seed_spec = (
                    self._sampling_config_for_worker(rank)
                )
                task_queue = mp_context.Queue(
                    self.num_workers * self.worker_options.worker_concurrency
                )
                self._task_queues.append(task_queue)
                parent_connection, child_connection = mp_context.Pipe(duplex=False)
                self._isolated_status_connections.append(parent_connection)
                try:
                    worker = mp_context.Process(
                        target=_sampling_worker_loop,
                        args=(
                            rank,
                            self.data,
                            self.sampler_input,
                            unshuffled_indexes[rank],
                            worker_sampling_config,
                            self.worker_options,
                            self.output_channel,
                            task_queue,
                            self.sampling_completed_worker_count,
                            barrier,
                            self._sampler_options,
                            self._degree_tensors,
                            worker_seed_spec,
                            rpc_spec,
                            child_connection,
                        ),
                    )
                    self._workers.append(worker)
                    worker.daemon = True
                    if self._isolated_port_lease is not None:
                        self._isolated_port_lease.release_reservation(
                            rpc_spec.master_port
                        )
                    worker.start()
                finally:
                    try:
                        child_connection.close()
                    except BaseException:
                        logger.exception(
                            "failed to close isolated sampling child status pipe"
                        )
                self._wait_for_isolated_worker_ready(rank)
            self._wait_for_isolated_workers_at_barrier()
        except BaseException:
            self._cleanup_isolated_workers(graceful=False, suppress_errors=True)
            raise

    def shutdown(self) -> None:
        if self._isolated_rpc_specs is None:
            super().shutdown()
            return
        self._cleanup_isolated_workers(graceful=True)
