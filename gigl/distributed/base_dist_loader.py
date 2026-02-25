"""
Base distributed loader that consolidates shared initialization logic
from DistNeighborLoader and DistABLPLoader.

Subclasses GLT's DistLoader and handles:
- Dataset metadata storage
- Colocated mode: DistLoader attribute setting + staggered producer init
- Graph Store mode: barrier loop + async RPC dispatch + channel creation
"""

import queue
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Final, Optional, Union

import torch
from graphlearn_torch.channel import RemoteReceivingChannel, ShmChannel
from graphlearn_torch.distributed import (
    DistLoader,
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
    get_context,
)
from graphlearn_torch.distributed.dist_client import async_request_server
from graphlearn_torch.distributed.dist_sampling_producer import DistMpSamplingProducer
from graphlearn_torch.distributed.rpc import rpc_is_initialized
from graphlearn_torch.sampler import (
    NodeSamplerInput,
    RemoteSamplerInput,
    SamplingConfig,
    SamplingType,
)
from torch_geometric.typing import EdgeType
from typing_extensions import Self

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.constants import DEFAULT_MASTER_INFERENCE_PORT
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store.dist_server import DistServer
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.utils.neighborloader import (
    DatasetSchema,
    patch_fanout_for_sampling,
)
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE

logger = Logger()

_COLLATION_SENTINEL: Final = object()  # Signals end-of-epoch to consumer


class TimingStats:
    """Accumulates timing measurements for profiling and outputs a summary."""

    def __init__(self, name: str):
        self._name = name
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}
        self._order: list[str] = []

    def record(self, key: str, elapsed: float) -> None:
        if key not in self._totals:
            self._order.append(key)
        self._totals[key] += elapsed
        self._counts[key] += 1
        if key not in self._mins or elapsed < self._mins[key]:
            self._mins[key] = elapsed
        if key not in self._maxs or elapsed > self._maxs[key]:
            self._maxs[key] = elapsed

    def summary(self) -> str:
        lines = [
            f"\n{'=' * 80}",
            f"  {self._name} — Timing Summary",
            f"{'=' * 80}",
        ]
        for key in self._order:
            total = self._totals[key]
            count = self._counts[key]
            avg = total / count if count > 0 else 0
            min_v = self._mins.get(key, 0)
            max_v = self._maxs.get(key, 0)
            if count == 1:
                lines.append(f"  {key:<45s} {total:>10.4f}s")
                continue
            lines.append(
                f"  {key:<45s} total={total:>10.4f}s  n={count:>6d}  "
                f"avg={avg:>8.4f}s  min={min_v:>8.4f}s  max={max_v:>8.4f}s"
            )
        lines.append(f"{'=' * 80}\n")
        return "\n".join(lines)


# We don't see logs for graph store mode for whatever reason.
# TOOD(#442): Revert this once the GCP issues are resolved.
def _flush() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


@dataclass(frozen=True)
class DistributedRuntimeInfo:
    """Plain data container for resolved distributed context information."""

    node_world_size: int
    node_rank: int
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    master_ip_address: str
    should_cleanup_distributed_context: bool


class BaseDistLoader(DistLoader):
    """Base class for GiGL distributed loaders.

    Consolidates shared initialization logic from DistNeighborLoader and DistABLPLoader.
    Subclasses GLT's DistLoader but does NOT call its ``__init__`` — instead, it
    replicates the relevant attribute-setting logic to allow configurable producer classes.

    Subclasses should:
    1. Call ``resolve_runtime()`` to get runtime context.
    2. Determine mode (colocated vs graph store).
    3. Call ``create_sampling_config()`` to build the SamplingConfig.
    4. For colocated: call ``create_colocated_channel()`` and construct the
       ``DistMpSamplingProducer`` (or subclass), then pass the producer as ``sampler``.
    5. For graph store: pass the RPC function (e.g. ``DistServer.create_sampling_producer``)
       as ``sampler``.
    6. Call ``super().__init__()`` with the prepared data.

    Args:
        dataset: ``DistDataset`` (colocated) or ``RemoteDistDataset`` (graph store).
        sampler_input: Prepared by the subclass. Single input for colocated mode,
            list (one per server) for graph store mode.
        dataset_schema: Contains edge types, feature info, edge dir, etc.
        worker_options: ``MpDistSamplingWorkerOptions`` (colocated) or
            ``RemoteDistSamplingWorkerOptions`` (graph store).
        sampling_config: Configuration for the sampler (created via ``create_sampling_config``).
        device: Target device for sampled results.
        runtime: Resolved distributed runtime information.
        sampler: Either a pre-constructed ``DistMpSamplingProducer`` (colocated mode)
            or a callable to dispatch on the ``DistServer`` (graph store mode).
        process_start_gap_seconds: Delay between each process for staggered colocated init.
        background_collation_queue_size: If set to a positive integer, enables
            background collation in a daemon thread. The collation of sampled
            messages (via ``_collate_fn``) is performed in a background thread,
            overlapping with GPU training. The value controls the maximum number
            of pre-collated batches buffered in memory. ``None`` disables
            background collation (default behavior).
    """

    @staticmethod
    def resolve_runtime(
        context: Optional[DistributedContext] = None,
        local_process_rank: Optional[int] = None,
        local_process_world_size: Optional[int] = None,
    ) -> DistributedRuntimeInfo:
        """Resolves distributed context from either a DistributedContext or torch.distributed.

        Args:
            context: (Deprecated) If provided, derives rank info from the DistributedContext.
                Requires local_process_rank and local_process_world_size.
            local_process_rank: (Deprecated) Required when context is provided.
            local_process_world_size: (Deprecated) Required when context is provided.

        Returns:
            A DistributedRuntimeInfo containing all resolved rank/topology information.
        """
        should_cleanup_distributed_context: bool = False

        if context:
            assert (
                local_process_world_size is not None
            ), "context: DistributedContext provided, so local_process_world_size must be provided."
            assert (
                local_process_rank is not None
            ), "context: DistributedContext provided, so local_process_rank must be provided."

            master_ip_address = context.main_worker_ip_address
            node_world_size = context.global_world_size
            node_rank = context.global_rank
            local_world_size = local_process_world_size
            local_rank = local_process_rank

            rank = node_rank * local_world_size + local_rank
            world_size = node_world_size * local_world_size

            if not torch.distributed.is_initialized():
                logger.info(
                    "process group is not available, trying to torch.distributed.init_process_group "
                    "to communicate necessary setup information."
                )
                should_cleanup_distributed_context = True
                logger.info(
                    f"Initializing process group with master ip address: {master_ip_address}, "
                    f"rank: {rank}, world size: {world_size}, "
                    f"local_rank: {local_rank}, local_world_size: {local_world_size}."
                )
                torch.distributed.init_process_group(
                    backend="gloo",
                    init_method=f"tcp://{master_ip_address}:{DEFAULT_MASTER_INFERENCE_PORT}",
                    rank=rank,
                    world_size=world_size,
                )
        else:
            assert torch.distributed.is_initialized(), (
                "context: DistributedContext is None, so process group must be "
                "initialized before constructing the loader."
            )
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            rank_ip_addresses = gigl.distributed.utils.get_internal_ip_from_all_ranks()
            master_ip_address = rank_ip_addresses[0]

            count_ranks_per_ip_address = Counter(rank_ip_addresses)
            local_world_size = count_ranks_per_ip_address[master_ip_address]
            for rank_ip_address, count in count_ranks_per_ip_address.items():
                if count != local_world_size:
                    raise ValueError(
                        f"All ranks must have the same number of processes, but found "
                        f"{count} processes for rank {rank} on ip {rank_ip_address}, "
                        f"expected {local_world_size}. "
                        f"count_ranks_per_ip_address = {count_ranks_per_ip_address}"
                    )

            node_world_size = len(count_ranks_per_ip_address)
            local_rank = rank % local_world_size
            node_rank = rank // local_world_size

        return DistributedRuntimeInfo(
            node_world_size=node_world_size,
            node_rank=node_rank,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            master_ip_address=master_ip_address,
            should_cleanup_distributed_context=should_cleanup_distributed_context,
        )

    def __init__(
        self,
        dataset: Union[DistDataset, RemoteDistDataset],
        sampler_input: Union[NodeSamplerInput, list[NodeSamplerInput]],
        dataset_schema: DatasetSchema,
        worker_options: Union[
            MpDistSamplingWorkerOptions, RemoteDistSamplingWorkerOptions
        ],
        sampling_config: SamplingConfig,
        device: torch.device,
        runtime: DistributedRuntimeInfo,
        sampler: Union[DistMpSamplingProducer, Callable[..., int]],
        process_start_gap_seconds: float = 60.0,
        background_collation_queue_size: Optional[int] = None,
    ):
        # Set right away so __del__ can clean up if we throw during init.
        # Will be set to False once connections are initialized.
        self._shutdowned = True

        # --- Background collation setup (validate early, before heavy init) ---
        if (
            background_collation_queue_size is not None
            and background_collation_queue_size < 1
        ):
            raise ValueError(
                f"background_collation_queue_size must be >= 1 if provided, "
                f"got {background_collation_queue_size}"
            )
        self._background_collation_queue_size = background_collation_queue_size
        self._collation_thread: Optional[threading.Thread] = None
        self._collated_queue: Optional[queue.Queue] = None
        self._collation_stop_event: Optional[threading.Event] = None

        # --- Timing instrumentation ---
        mode_label = (
            "background_collation"
            if background_collation_queue_size is not None
            else "synchronous"
        )
        self._timing = TimingStats(f"BaseDistLoader ({mode_label})")
        self._epoch_start_time: Optional[float] = None
        self._log_timing_every_n_batches: Final[int] = 10

        # Store dataset metadata for subclass _collate_fn usage
        self._is_homogeneous_with_labeled_edge_type = (
            dataset_schema.is_homogeneous_with_labeled_edge_type
        )
        self._node_feature_info = dataset_schema.node_feature_info
        self._edge_feature_info = dataset_schema.edge_feature_info

        # --- Attributes shared by both modes (mirrors GLT DistLoader.__init__) ---
        self.input_data = sampler_input
        self.sampling_type = sampling_config.sampling_type
        self.num_neighbors = sampling_config.num_neighbors
        self.batch_size = sampling_config.batch_size
        self.shuffle = sampling_config.shuffle
        self.drop_last = sampling_config.drop_last
        self.with_edge = sampling_config.with_edge
        self.with_weight = sampling_config.with_weight
        self.collect_features = sampling_config.collect_features
        self.edge_dir = sampling_config.edge_dir
        self.sampling_config = sampling_config
        self.to_device = device
        self.worker_options = worker_options

        self._is_collocated_worker = False
        self._with_channel = True
        self._num_recv = 0
        self._epoch = 0

        # --- Mode-specific attributes and connection initialization ---
        if isinstance(sampler, DistMpSamplingProducer):
            assert isinstance(dataset, DistDataset)
            assert isinstance(worker_options, MpDistSamplingWorkerOptions)
            assert isinstance(sampler_input, NodeSamplerInput)

            self.data: Optional[DistDataset] = dataset
            self._is_mp_worker = True
            self._is_remote_worker = False

            self.num_data_partitions = dataset.num_partitions
            self.data_partition_idx = dataset.partition_idx
            self._set_ntypes_and_etypes(
                dataset.get_node_types(), dataset.get_edge_types()
            )

            self._input_len = len(sampler_input)
            self._input_type = sampler_input.input_type
            self._num_expected = self._input_len // self.batch_size
            if not self.drop_last and self._input_len % self.batch_size != 0:
                self._num_expected += 1

            self._shutdowned = False
            self._init_colocated_connections(
                dataset=dataset,
                producer=sampler,
                runtime=runtime,
                process_start_gap_seconds=process_start_gap_seconds,
            )
        else:
            assert isinstance(dataset, RemoteDistDataset)
            assert isinstance(worker_options, RemoteDistSamplingWorkerOptions)
            assert isinstance(sampler_input, list)
            assert callable(sampler)

            self.data = None
            self._is_mp_worker = False
            self._is_remote_worker = True
            self._num_expected = float("inf")

            self._server_rank_list: list[int] = (
                worker_options.server_rank
                if isinstance(worker_options.server_rank, list)
                else [worker_options.server_rank]
            )
            self._input_data_list = sampler_input
            self._input_type = self._input_data_list[0].input_type

            self.num_data_partitions = dataset.cluster_info.num_storage_nodes
            self.data_partition_idx = dataset.cluster_info.compute_node_rank
            edge_types = dataset_schema.edge_types or []
            if edge_types:
                node_types = list(
                    set([et[0] for et in edge_types] + [et[2] for et in edge_types])
                )
            else:
                node_types = [DEFAULT_HOMOGENEOUS_NODE_TYPE]
            self._set_ntypes_and_etypes(node_types, edge_types)

            self._shutdowned = False
            self._init_graph_store_connections(
                dataset=dataset,
                create_producer_fn=sampler,
            )

    @staticmethod
    def create_sampling_config(
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        dataset_schema: DatasetSchema,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> SamplingConfig:
        """Creates a SamplingConfig with patched fanout.

        Patches ``num_neighbors`` to zero-out label edge types, then creates
        the SamplingConfig used by both colocated and graph store modes.

        Args:
            num_neighbors: Fanout per hop.
            dataset_schema: Contains edge types and edge dir.
            batch_size: How many samples per batch.
            shuffle: Whether to shuffle input nodes.
            drop_last: Whether to drop the last incomplete batch.

        Returns:
            A fully configured SamplingConfig.
        """
        num_neighbors = patch_fanout_for_sampling(
            edge_types=dataset_schema.edge_types,
            num_neighbors=num_neighbors,
        )
        return SamplingConfig(
            sampling_type=SamplingType.NODE,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            with_edge=True,
            collect_features=True,
            with_neg=False,
            with_weight=False,
            edge_dir=dataset_schema.edge_dir,
            seed=None,
        )

    @staticmethod
    def create_colocated_channel(
        worker_options: MpDistSamplingWorkerOptions,
    ) -> ShmChannel:
        """Creates a ShmChannel for colocated mode.

        Creates and optionally pin-memories the shared-memory channel.

        Args:
            worker_options: The colocated worker options (must already be fully configured).

        Returns:
            A ShmChannel ready to be passed to a DistMpSamplingProducer.
        """
        channel = ShmChannel(
            worker_options.channel_capacity, worker_options.channel_size
        )
        if worker_options.pin_memory:
            channel.pin_memory()
        return channel

    def _init_colocated_connections(
        self,
        dataset: DistDataset,
        producer: DistMpSamplingProducer,
        runtime: DistributedRuntimeInfo,
        process_start_gap_seconds: float,
    ) -> None:
        """Initialize colocated mode connections.

        Validates the GLT distributed context, stores the pre-constructed producer,
        and performs staggered initialization to avoid memory OOM.

        All DistLoader attributes are already set by ``__init__`` before this is called.

        Args:
            dataset: The local DistDataset.
            producer: A pre-constructed DistMpSamplingProducer (or subclass).
            runtime: Resolved distributed runtime info (used for staggered sleep).
            process_start_gap_seconds: Delay multiplier for staggered init.
        """
        # Validate context and store the pre-constructed producer and its channel
        current_ctx = get_context()
        if not current_ctx.is_worker():
            raise RuntimeError(
                f"'{self.__class__.__name__}': only supports "
                f"launching multiprocessing sampling workers with "
                f"a non-server distribution mode, current role of "
                f"distributed context is {current_ctx.role}."
            )
        if dataset is None:
            raise ValueError(
                f"'{self.__class__.__name__}': missing input dataset "
                f"when launching multiprocessing sampling workers."
            )
        self.worker_options._set_worker_ranks(current_ctx)
        self._channel = producer.output_channel
        self._mp_producer = producer

        # Staggered init — sleep proportional to local_rank to avoid
        # concurrent initialization spikes that cause CPU memory OOM.
        logger.info(
            f"---Machine {runtime.rank} local process number {runtime.local_rank} "
            f"preparing to sleep for {process_start_gap_seconds * runtime.local_rank} seconds"
        )
        time.sleep(process_start_gap_seconds * runtime.local_rank)
        self._mp_producer.init()

    def _init_graph_store_connections(
        self,
        dataset: RemoteDistDataset,
        create_producer_fn: Callable[..., int],
    ) -> None:
        """Initialize Graph Store mode connections.

        Validates the GLT distributed context, performs a sequential barrier loop
        across compute nodes, dispatches async RPCs to create sampling producers on
        storage nodes, and creates a RemoteReceivingChannel.

        All DistLoader attributes are already set by ``__init__`` before this is called.

        Uses ``async_request_server`` instead of ``ThreadPoolExecutor`` to avoid
        TensorPipe rendezvous deadlock with many servers.

        For Graph Store mode it's important to distinguish "compute node" (e.g. physical compute machine) from "compute process" (e.g. process running on the compute node).
        Since in practice we have multiple compute processes per compute node, and each compute process needs to initialize the connection to the storage nodes.
        E.g. if there are 4 gpus per compute node, then there will be 4 connections from each compute node to each storage node.

        See below for a connection setup.
        ╔═══════════════════════════════════════════════════════════════════════════════════════╗
        ║                         COMPUTE TO STORAGE NODE CONNECTIONS                            ║
        ╚═══════════════════════════════════════════════════════════════════════════════════════╝

             COMPUTE NODES                                              STORAGE NODES
             ═════════════                                              ═════════════

          ┌──────────────────────┐          (1)                      ┌───────────────┐
          │    COMPUTE NODE 0    │                                   │               │
          │  ┌────┬────┬────┬────┤ ══════════════════════════════════│   STORAGE 0   │
          │  │GPU │GPU │GPU │GPU │                                 ╱ │               │
          │  │ 0  │ 1  │ 2  │ 3  │ ════════════════════╲         ╱   └───────────────┘
          │  └────┴────┴────┴────┤          (2)          ╲     ╱
          └──────────────────────┘                         ╲ ╱
                                                            ╳
                                                  (3)     ╱   ╲     (4)
          ┌──────────────────────┐                      ╱       ╲    ┌───────────────┐
          │    COMPUTE NODE 1    │                    ╱           ╲  │               │
          │  ┌────┬────┬────┬────┤ ═════════════════╱               ═│   STORAGE 1   │
          │  │GPU │GPU │GPU │GPU │                                   │               │
          │  │ 0  │ 1  │ 2  │ 3  │ ══════════════════════════════════│               │
          │  └────┴────┴────┴────┤                                   └───────────────┘
          └──────────────────────┘

          ┌─────────────────────────────────────────────────────────────────────────────┐
          │  (1) Compute Node 0  →  Storage 0   (4 connections, one per GPU)            │
          │  (2) Compute Node 0  →  Storage 1   (4 connections, one per GPU)            │
          │  (3) Compute Node 1  →  Storage 0   (4 connections, one per GPU)            │
          │  (4) Compute Node 1  →  Storage 1   (4 connections, one per GPU)            │
          └─────────────────────────────────────────────────────────────────────────────┘
        """
        # Validate distributed context
        ctx = get_context()
        if ctx is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}': the distributed context "
                f"has not been initialized."
            )
        if not ctx.is_client():
            raise RuntimeError(
                f"'{self.__class__.__name__}': must be used on a client "
                f"worker process."
            )

        # Move input to CPU before sending to server
        for inp in self._input_data_list:
            if not isinstance(inp, RemoteSamplerInput):
                inp.to(torch.device("cpu"))

        node_rank = dataset.cluster_info.compute_node_rank

        _flush()
        start_time = time.time()
        rpc_futures: list[tuple[int, torch.futures.Future[int]]] = []
        # Dispatch ALL create_producer RPCs async.
        # async_request_server queues the RPC in TensorPipe and returns
        # immediately, allowing all storage nodes to start their worker
        # rendezvous simultaneously.
        logger.info(
            f"node_rank={node_rank} dispatching create_sampling_producer to "
            f"{len(self._server_rank_list)} servers"
        )
        _flush()
        t_dispatch = time.time()
        for server_rank, inp_data in zip(self._server_rank_list, self._input_data_list):
            fut = async_request_server(
                server_rank,
                create_producer_fn,
                inp_data,
                self.sampling_config,
                self.worker_options,
            )
            rpc_futures.append((server_rank, fut))
        logger.info(
            f"node_rank={node_rank} all {len(rpc_futures)} RPCs dispatched in "
            f"{time.time() - t_dispatch:.3f}s, waiting for responses"
        )
        _flush()

        # Wait for all results
        self._producer_id_list: list[int] = []
        for server_rank, fut in rpc_futures:
            t_wait = time.time()
            producer_id: int = fut.wait()
            logger.info(
                f"node_rank={node_rank} create_sampling_producer"
                f"(server_rank={server_rank}) returned "
                f"producer_id={producer_id} in {time.time() - t_wait:.2f}s"
            )
            _flush()
            self._producer_id_list.append(producer_id)
        logger.info(
            f"node_rank={node_rank} all {len(self._producer_id_list)} producers "
            f"created in {time.time() - t_dispatch:.2f}s total"
        )
        _flush()
        # Create remote receiving channel for cross-machine message passing
        self._channel = RemoteReceivingChannel(
            self._server_rank_list,
            self._producer_id_list,
            self.worker_options.prefetch_size,
        )

        logger.info(
            f"node_rank {node_rank} initialized the dist loader in "
            f"{time.time() - start_time:.2f}s"
        )
        _flush()

    # --- Background collation methods ---

    def _maybe_log_timing(self) -> None:
        """Log timing summary periodically and at end of epoch."""
        if self._num_recv % self._log_timing_every_n_batches == 0:
            logger.info(self._timing.summary())
            _flush()

    def __next__(self):  # type: ignore[override]
        """Returns the next collated batch.

        When background collation is enabled, retrieves pre-collated results
        from the bounded queue. Otherwise, falls back to the synchronous
        path (replicated from GLT ``DistLoader``).

        Returns:
            A ``Data`` or ``HeteroData`` batch.

        Raises:
            StopIteration: When the epoch is exhausted.
        """
        if self._background_collation_queue_size is not None:
            return self._next_from_background_collation()
        # Original synchronous path (replicated from GLT DistLoader)
        if self._num_recv == self._num_expected:
            logger.info(
                f"[sync] Epoch done. Total batches: {self._num_recv}, "
                f"epoch wall time: {time.time() - self._epoch_start_time:.2f}s"
            )
            logger.info(self._timing.summary())
            _flush()
            raise StopIteration
        t0 = time.time()
        if self._with_channel:
            msg = self._channel.recv()
        else:
            msg = self._collocated_producer.sample()
        t_recv = time.time()
        self._timing.record("sync/recv", t_recv - t0)

        result = self._collate_fn(msg)
        t_collate = time.time()
        self._timing.record("sync/collate_fn", t_collate - t_recv)
        self._timing.record("sync/total_next", t_collate - t0)

        self._num_recv += 1
        self._maybe_log_timing()
        return result

    def _next_from_background_collation(self):
        """Retrieves the next pre-collated batch from the background queue.

        Returns:
            A ``Data`` or ``HeteroData`` batch.

        Raises:
            StopIteration: On sentinel or when expected count is reached.
        """
        assert self._collated_queue is not None
        t0 = time.time()
        qsize_before = self._collated_queue.qsize()
        item = self._collated_queue.get()
        t_get = time.time()
        self._timing.record("bg_consumer/queue_get", t_get - t0)
        self._timing.record("bg_consumer/queue_size_at_get", qsize_before)

        if item is _COLLATION_SENTINEL:
            logger.info(
                f"[bg] Epoch done. Total batches: {self._num_recv}, "
                f"epoch wall time: {time.time() - self._epoch_start_time:.2f}s"
            )
            logger.info(self._timing.summary())
            _flush()
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        self._num_recv += 1
        self._maybe_log_timing()
        return item

    def _collation_worker(self) -> None:
        """Target function for the background collation daemon thread.

        Continuously receives messages from the channel (or collocated
        producer) and runs ``_collate_fn``, placing collated results into
        ``_collated_queue``. Exits when the epoch batch count is reached,
        a ``StopIteration`` is received from the channel, or the stop event
        is set.
        """
        assert self._collated_queue is not None
        assert self._collation_stop_event is not None
        num_produced = 0
        try:
            while True:
                # For finite epochs, exit after producing all expected batches
                if (
                    self._num_expected != float("inf")
                    and num_produced >= self._num_expected
                ):
                    self._collated_queue.put(_COLLATION_SENTINEL)
                    return

                # Receive next sampled message
                t0 = time.time()
                try:
                    if self._with_channel:
                        msg = self._channel.recv()
                    else:
                        msg = self._collocated_producer.sample()
                except StopIteration:
                    self._collated_queue.put(_COLLATION_SENTINEL)
                    return
                t_recv = time.time()
                self._timing.record("bg_producer/recv", t_recv - t0)

                # Check stop event between recv and collate
                if self._collation_stop_event.is_set():
                    return

                result = self._collate_fn(msg)
                t_collate = time.time()
                self._timing.record("bg_producer/collate_fn", t_collate - t_recv)

                self._collated_queue.put(result)
                t_put = time.time()
                self._timing.record("bg_producer/queue_put", t_put - t_collate)
                self._timing.record("bg_producer/total_iteration", t_put - t0)

                num_produced += 1
        except Exception as e:
            self._collated_queue.put(e)

    def _start_collation_thread(self) -> None:
        """Creates and starts a fresh background collation thread."""
        assert self._background_collation_queue_size is not None
        self._collation_stop_event = threading.Event()
        self._collated_queue = queue.Queue(
            maxsize=self._background_collation_queue_size
        )
        self._collation_thread = threading.Thread(
            target=self._collation_worker, daemon=True
        )
        self._collation_thread.start()

    def _stop_collation_thread(self) -> None:
        """Stops the background collation thread if it is running.

        Sets the stop event and drains the queue to unblock the worker
        if it is blocked on ``queue.put()``. Joins the thread with a
        10-second timeout.
        """
        if self._collation_thread is None or not self._collation_thread.is_alive():
            return
        assert self._collation_stop_event is not None
        assert self._collated_queue is not None

        self._collation_stop_event.set()
        # Drain the queue to unblock the worker if it's blocked on put()
        while True:
            try:
                self._collated_queue.get_nowait()
            except queue.Empty:
                break
        self._collation_thread.join(timeout=10.0)
        if self._collation_thread.is_alive():
            logger.warning(
                "Background collation thread did not terminate within 10 seconds."
            )

    # Overwrite DistLoader.shutdown to so we can use our own shutdown and rpc calls
    def shutdown(self) -> None:
        if self._shutdowned:
            return
        if self._background_collation_queue_size is not None:
            self._stop_collation_thread()
        if self._is_collocated_worker:
            self._collocated_producer.shutdown()
        elif self._is_mp_worker:
            self._mp_producer.shutdown()
        elif rpc_is_initialized() is True:
            rpc_futures: list[torch.futures.Future[None]] = []
            for server_rank, producer_id in zip(
                self._server_rank_list, self._producer_id_list
            ):
                fut = async_request_server(
                    server_rank, DistServer.destroy_sampling_producer, producer_id
                )
                rpc_futures.append(fut)
            torch.futures.wait_all(rpc_futures)
        self._shutdowned = True

    # Overwrite DistLoader.__iter__ to so we can use our own __iter__ and rpc calls
    def __iter__(self) -> Self:
        if self._background_collation_queue_size is not None:
            self._stop_collation_thread()

        # Log previous epoch timing (if any) and reset for new epoch
        if self._epoch > 0:
            logger.info(
                f"[iter] Resetting for epoch {self._epoch}. " f"Previous epoch timing:"
            )
            logger.info(self._timing.summary())
            _flush()

        mode_label = (
            "background_collation"
            if self._background_collation_queue_size is not None
            else "synchronous"
        )
        self._timing = TimingStats(f"BaseDistLoader ({mode_label}) epoch={self._epoch}")
        self._epoch_start_time = time.time()

        self._num_recv = 0
        if self._is_collocated_worker:
            self._collocated_producer.reset()
        elif self._is_mp_worker:
            self._mp_producer.produce_all()
        else:
            rpc_futures: list[torch.futures.Future[None]] = []
            for server_rank, producer_id in zip(
                self._server_rank_list, self._producer_id_list
            ):
                fut = async_request_server(
                    server_rank,
                    DistServer.start_new_epoch_sampling,
                    producer_id,
                    self._epoch,
                )
                rpc_futures.append(fut)
            torch.futures.wait_all(rpc_futures)
            self._channel.reset()
        self._epoch += 1

        if self._background_collation_queue_size is not None:
            self._start_collation_thread()

        return self
