# Significant portions of this file are taken from GraphLearn-for-PyTorch
# (graphlearn_torch/python/distributed/dist_sampling_producer.py).
# This version uses GiGL's sampler hierarchy (BaseGiGLSampler subclasses:
# DistNeighborSampler for k-hop, DistPPRNeighborSampler for PPR) instead of
# GLT's DistNeighborSampler directly.

import datetime
import queue
from threading import Barrier, BrokenBarrierError
from typing import Optional, Union, cast

import torch
import torch.distributed.rpc
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
from graphlearn_torch.typing import NodeType
from graphlearn_torch.utils import seed_everything
from torch._C import _set_worker_signal_handlers
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from gigl.common.logger import Logger
from gigl.distributed.constants import (
    SAMPLING_RPC_INIT_TIMEOUT_ENV,
    SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS,
    sampling_worker_init_timeout_seconds,
)
from gigl.distributed.sampler_options import SamplerOptions
from gigl.distributed.utils.dist_sampler import create_dist_sampler

logger = Logger()


def _narrow_rpc_timeout_after_init(rank: int, bringup_rpc_timeout: float) -> None:
    """Reset the RPC agent's request timeout once a sampling worker is past its init barrier.

    ``init_rpc`` sets the agent's default request timeout to the bring-up tolerance, which is
    deliberately generous so cross-rank skew does not kill the gather -- but leaving it there
    would mean every steady-state sampling and feature-collection RPC also waits that long
    before a stuck request surfaces. The init barrier is precisely the boundary between the two
    regimes, so the worker narrows the timeout immediately after passing it.

    A no-op when the bring-up value already equals the steady-state value.
    """
    if SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS != bringup_rpc_timeout:
        torch.distributed.rpc._set_rpc_timeout(
            float(SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS)
        )
        logger.info(
            f"sampling worker {rank} past init barrier; RPC request timeout narrowed from "
            f"{bringup_rpc_timeout}s (bring-up) to {SAMPLING_STEADY_STATE_RPC_TIMEOUT_SECONDS}s"
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

        dist_sampler = create_dist_sampler(
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

        _narrow_rpc_timeout_after_init(rank, worker_options.rpc_timeout)

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
    finally:
        # In a finally so a worker released through a broken init barrier still tears down
        # its sampler loop and RPC agent instead of leaking them
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
        degree_tensors: Optional[
            Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        ] = None,
    ):
        super().__init__(data, sampler_input, sampling_config, worker_options, channel)
        self._sampler_options = sampler_options
        self._degree_tensors = degree_tensors

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
                    self._degree_tensors,
                ),
            )
            worker.daemon = True
            worker.start()
            self._workers.append(worker)
        self._wait_for_workers_at_barrier(barrier)

    def _wait_for_workers_at_barrier(self, barrier: Barrier) -> None:
        """Wait for every sampling worker to reach the init barrier, but not forever.

        An unbounded ``barrier.wait()`` here is the difference between a job that fails and a job
        that hangs. Each worker must complete GLT's ``init_rpc``, which itself gathers all
        ``num_workers x world_size`` workers within the bring-up tolerance; a worker that loses
        that gather dies, never reaches this barrier, and leaves this process blocked forever
        while its peers sit in collectives until the training process group's own timeout fires.
        Bounding this wait is what turns that wedge into a fast, attributable failure.

        The barrier is waited on ONCE with the full timeout rather than polled: a timed-out
        ``wait`` puts a multiprocessing barrier into the broken state permanently, for the children
        too, so polling it would destroy the very rendezvous it is checking.

        Raises:
            RuntimeError: if the barrier is not reached within the timeout, naming the workers
                that died so the failure is diagnosable from one replica's log.
        """
        timeout_seconds = sampling_worker_init_timeout_seconds()
        try:
            barrier.wait(timeout=timeout_seconds)
        except BrokenBarrierError:
            # Diagnose BEFORE shutdown(): it terminates the surviving workers, which would
            # make every worker look dead and erase the exit codes that matter
            dead = [
                rank
                for rank, worker in enumerate(self._workers)
                if not worker.is_alive()
            ]
            exit_codes = {rank: self._workers[rank].exitcode for rank in dead}
            diagnosis = (
                f"dead workers (rank -> exitcode): {exit_codes}"
                if exit_codes
                else "no worker has exited, so at least one is still initialising"
            )
            # Tear down the queues and processes this producer already created; GLT's
            # shutdown() joins with a timeout and terminates stragglers, so it cannot
            # replace the hang this bound exists to remove
            self.shutdown()
            raise RuntimeError(
                f"sampling workers did not reach the init barrier within {timeout_seconds}s; "
                f"{diagnosis}. A worker that loses GLT's init_rpc gather dies before this "
                f"barrier. Raise {SAMPLING_RPC_INIT_TIMEOUT_ENV} (this bound is a multiple of "
                f"it) if init is legitimately slower than this on a cold cache."
            ) from None
