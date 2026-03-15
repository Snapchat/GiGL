"""
Base distributed loader that consolidates shared initialization logic
from DistNeighborLoader and DistABLPLoader.

Subclasses GLT's DistLoader and handles:
- Dataset metadata storage
- Colocated mode: DistLoader attribute setting + staggered producer init
- Graph Store mode: barrier loop + async RPC dispatch + channel creation
"""

import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
from graphlearn_torch.channel import ShmChannel
from graphlearn_torch.distributed import (
    DistLoader,
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
    get_context,
)
from graphlearn_torch.distributed.dist_client import async_request_server
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
from gigl.distributed.dist_sampling_producer import DistSamplingProducer
from gigl.distributed.graph_store.dist_server import (
    DistServer,
    InitSamplingBackendOpts,
    RegisterBackendOpts,
)
from gigl.distributed.graph_store.remote_channel import RemoteReceivingChannel
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.sampler_options import SamplerOptions
from gigl.distributed.utils.neighborloader import (
    DatasetSchema,
    patch_fanout_for_sampling,
)
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE

logger = Logger()


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
       ``DistSamplingProducer`` (or subclass), then pass the producer as ``producer``.
    5. For graph store: pass a non-producer sentinel as ``producer``; this base
       class will run two-phase RPC init (`init_sampling_backend` then
       `register_sampling_input`).
    6. Call ``super().__init__()`` with the prepared data.

    Args:
        dataset: ``DistDataset`` (colocated) or ``RemoteDistDataset`` (graph store).
        sampler_input: Prepared by the subclass. Single input for colocated mode,
            list (one per server) for graph store mode.
        dataset_schema: Contains edge types, feature info, edge dir, etc.
        worker_options: ``MpDistSamplingWorkerOptions`` (colocated) or
            ``RemoteDistSamplingWorkerOptions`` (graph store).
        sampling_config: Configuration for sampling (created via ``create_sampling_config``).
        device: Target device for sampled results.
        runtime: Resolved distributed runtime information.
        producer: A pre-constructed ``DistSamplingProducer`` in colocated mode.
            In graph store mode this value is ignored by initialization logic.
        sampler_options: Controls which sampler class is instantiated.
        process_start_gap_seconds: Delay between each process for staggered colocated init.
            In graph store mode, this is the delay between each batch of concurrent
            producer initializations.
        max_concurrent_producer_inits: Maximum number of leader ranks that may
            dispatch graph-store init RPCs concurrently in graph store mode.
            Leaders are grouped into batches of this size; each batch sleeps
            ``batch_index * process_start_gap_seconds`` before dispatching.
            Only applies to graph store mode. Defaults to ``None``
            (no staggering).
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
        producer: Union[DistSamplingProducer, Callable[..., int]],
        sampler_options: SamplerOptions,
        process_start_gap_seconds: float = 60.0,
        max_concurrent_producer_inits: Optional[int] = None,
    ):
        if max_concurrent_producer_inits is None:
            max_concurrent_producer_inits = sys.maxsize

        # Set right away so __del__ can clean up if we throw during init.
        # Will be set to False once connections are initialized.
        self._shutdowned = True

        # Store dataset metadata for subclass _collate_fn usage
        self._is_homogeneous_with_labeled_edge_type = (
            dataset_schema.is_homogeneous_with_labeled_edge_type
        )
        self._node_feature_info = dataset_schema.node_feature_info
        self._edge_feature_info = dataset_schema.edge_feature_info

        self._sampler_options = sampler_options

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
        if isinstance(producer, DistSamplingProducer):
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
                producer=producer,
                runtime=runtime,
                process_start_gap_seconds=process_start_gap_seconds,
            )
        else:
            assert isinstance(dataset, RemoteDistDataset)
            assert isinstance(worker_options, RemoteDistSamplingWorkerOptions)
            assert isinstance(sampler_input, list)

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
                runtime=runtime,
                process_start_gap_seconds=process_start_gap_seconds,
                max_concurrent_producer_inits=max_concurrent_producer_inits,
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
            A ShmChannel ready to be passed to a DistSamplingProducer.
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
        producer: DistSamplingProducer,
        runtime: DistributedRuntimeInfo,
        process_start_gap_seconds: float,
    ) -> None:
        """Initialize colocated mode connections.

        Validates the GLT distributed context, stores the pre-constructed producer,
        and performs staggered initialization to avoid memory OOM.

        All DistLoader attributes are already set by ``__init__`` before this is called.

        Args:
            dataset: The local DistDataset.
            producer: A pre-constructed DistSamplingProducer (or subclass).
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
        runtime: DistributedRuntimeInfo,
        process_start_gap_seconds: float = 60.0,
        max_concurrent_producer_inits: int = sys.maxsize,  # Already resolved from None by __init__
    ) -> None:
        """Initialize Graph Store mode connections.

        Uses a two-step lifecycle:
        1) initialize shared backend per backend key
        2) register per-worker-key inputs/channels
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

        backend_id_list = self._init_graph_store_sampling_backends(
            runtime=runtime,
            process_start_gap_seconds=process_start_gap_seconds,
            max_concurrent_producer_inits=max_concurrent_producer_inits,
        )
        channel_id_list = self._register_graph_store_sampling_inputs(
            runtime=runtime,
            backend_id_list=backend_id_list,
            process_start_gap_seconds=process_start_gap_seconds,
            max_concurrent_producer_inits=max_concurrent_producer_inits,
        )
        self._channel_id_list = channel_id_list
        self._channel = RemoteReceivingChannel(
            self._server_rank_list,
            self._channel_id_list,
            self.worker_options.prefetch_size,
        )

        logger.info(
            f"node_rank {node_rank} rank={runtime.rank} initialized "
            f"the dist loader in {time.time() - start_time:.2f}s"
        )
        _flush()

    @staticmethod
    def _compute_group_leader(
        my_key: str,
        all_keys: list[Optional[str]],
        rank: int,
        process_start_gap_seconds: float,
        max_concurrent_producer_inits: int,
    ) -> tuple[int, bool, int, int, float, int]:
        key_to_ranks: dict[str, list[int]] = defaultdict(list)
        for r, key in enumerate(all_keys):
            assert key is not None, f"Rank {r} did not provide key"
            key_to_ranks[key].append(r)

        leader_rank = min(key_to_ranks[my_key])
        is_leader = rank == leader_rank
        unique_keys = sorted(key_to_ranks.keys())
        my_key_index = unique_keys.index(my_key)
        num_unique_keys = len(unique_keys)
        num_batches = math.ceil(num_unique_keys / max_concurrent_producer_inits)
        my_batch = my_key_index // max_concurrent_producer_inits
        stagger_sleep_seconds = my_batch * process_start_gap_seconds
        return (
            leader_rank,
            is_leader,
            my_batch,
            num_batches,
            stagger_sleep_seconds,
            len(key_to_ranks[my_key]),
        )

    def _build_graph_store_backend_key(self) -> str:
        worker_options = self.worker_options
        assert isinstance(worker_options, RemoteDistSamplingWorkerOptions)
        worker_devices = ",".join(
            str(device) for device in worker_options.worker_devices
        )
        return "|".join(
            [
                f"master_addr={worker_options.master_addr}",
                f"master_port={worker_options.master_port}",
                f"num_workers={worker_options.num_workers}",
                f"worker_concurrency={worker_options.worker_concurrency}",
                f"num_rpc_threads={worker_options.num_rpc_threads}",
                f"rpc_timeout={worker_options.rpc_timeout}",
                f"use_all2all={worker_options.use_all2all}",
                f"worker_devices={worker_devices}",
                f"sampler_options={type(self._sampler_options).__name__}",
                f"num_neighbors={self.sampling_config.num_neighbors}",
                f"with_edge={self.sampling_config.with_edge}",
                f"with_neg={self.sampling_config.with_neg}",
                f"with_weight={self.sampling_config.with_weight}",
                f"edge_dir={self.sampling_config.edge_dir}",
                f"collect_features={self.sampling_config.collect_features}",
                f"seed={self.sampling_config.seed}",
            ]
        )

    def _init_graph_store_sampling_backends(
        self,
        runtime: DistributedRuntimeInfo,
        process_start_gap_seconds: float,
        max_concurrent_producer_inits: int,
    ) -> list[int]:
        backend_key = self._build_graph_store_backend_key()
        all_backend_keys: list[Optional[str]] = [None] * runtime.world_size
        torch.distributed.all_gather_object(all_backend_keys, backend_key)
        (
            leader_rank,
            is_leader,
            my_batch,
            num_batches,
            stagger_sleep_seconds,
            group_size,
        ) = self._compute_group_leader(
            my_key=backend_key,
            all_keys=all_backend_keys,
            rank=runtime.rank,
            process_start_gap_seconds=process_start_gap_seconds,
            max_concurrent_producer_inits=max_concurrent_producer_inits,
        )
        logger.info(
            f"rank={runtime.rank} backend_key={backend_key} is_leader={is_leader} "
            f"leader_rank={leader_rank} group_size={group_size} "
            f"batch={my_batch}/{num_batches} stagger_sleep={stagger_sleep_seconds:.1f}s"
        )
        if is_leader and stagger_sleep_seconds > 0:
            time.sleep(stagger_sleep_seconds)

        backend_id_list: list[int] = []
        if is_leader:
            rpc_futures: list[tuple[int, torch.futures.Future[int]]] = []
            worker_options = self.worker_options
            assert isinstance(worker_options, RemoteDistSamplingWorkerOptions)
            for server_rank in self._server_rank_list:
                fut = async_request_server(
                    server_rank,
                    DistServer.init_sampling_backend,
                    InitSamplingBackendOpts(
                        backend_key=backend_key,
                        worker_options=worker_options,
                        sampler_options=self._sampler_options,
                        sampling_config=self.sampling_config,
                    ),
                )
                rpc_futures.append((server_rank, fut))
            for _server_rank, fut in rpc_futures:
                backend_id_list.append(fut.wait())

        all_backend_ids: list[list[int]] = [[] for _ in range(runtime.world_size)]
        torch.distributed.all_gather_object(all_backend_ids, backend_id_list)
        return all_backend_ids[leader_rank]

    def _register_graph_store_sampling_inputs(
        self,
        runtime: DistributedRuntimeInfo,
        backend_id_list: list[int],
        process_start_gap_seconds: float,
        max_concurrent_producer_inits: int,
    ) -> list[int]:
        worker_options = self.worker_options
        assert isinstance(worker_options, RemoteDistSamplingWorkerOptions)
        my_worker_key = worker_options.worker_key
        all_worker_keys: list[Optional[str]] = [None] * runtime.world_size
        torch.distributed.all_gather_object(all_worker_keys, my_worker_key)
        (
            leader_rank,
            is_leader,
            my_batch,
            num_batches,
            stagger_sleep_seconds,
            group_size,
        ) = self._compute_group_leader(
            my_key=my_worker_key,
            all_keys=all_worker_keys,
            rank=runtime.rank,
            process_start_gap_seconds=process_start_gap_seconds,
            max_concurrent_producer_inits=max_concurrent_producer_inits,
        )
        logger.info(
            f"rank={runtime.rank} worker_key={my_worker_key} is_leader={is_leader} "
            f"leader_rank={leader_rank} group_size={group_size} "
            f"batch={my_batch}/{num_batches} stagger_sleep={stagger_sleep_seconds:.1f}s"
        )
        if is_leader and stagger_sleep_seconds > 0:
            time.sleep(stagger_sleep_seconds)

        channel_id_list: list[int] = []
        if is_leader:
            rpc_futures: list[tuple[int, torch.futures.Future[int]]] = []
            for server_rank, backend_id, inp_data in zip(
                self._server_rank_list, backend_id_list, self._input_data_list
            ):
                fut = async_request_server(
                    server_rank,
                    DistServer.register_sampling_input,
                    RegisterBackendOpts(
                        backend_id=backend_id,
                        worker_key=my_worker_key,
                        sampler_input=inp_data,
                        sampling_config=self.sampling_config,
                        buffer_capacity=worker_options.buffer_capacity,
                        buffer_size=worker_options.buffer_size,
                    ),
                )
                rpc_futures.append((server_rank, fut))
            for _server_rank, fut in rpc_futures:
                channel_id_list.append(fut.wait())

        all_channel_ids: list[list[int]] = [[] for _ in range(runtime.world_size)]
        torch.distributed.all_gather_object(all_channel_ids, channel_id_list)
        return all_channel_ids[leader_rank]

    # Overwrite DistLoader.shutdown to so we can use our own shutdown and rpc calls
    def shutdown(self) -> None:
        if self._shutdowned:
            return
        if self._is_collocated_worker:
            self._collocated_producer.shutdown()
        elif self._is_mp_worker:
            self._mp_producer.shutdown()
        elif rpc_is_initialized() is True:
            rpc_futures: list[torch.futures.Future[None]] = []
            for server_rank, channel_id in zip(
                self._server_rank_list, self._channel_id_list
            ):
                fut = async_request_server(
                    server_rank, DistServer.destroy_sampling_input, channel_id
                )
                rpc_futures.append(fut)
            torch.futures.wait_all(rpc_futures)
        self._shutdowned = True

    # Overwrite DistLoader.__iter__ to so we can use our own __iter__ and rpc calls
    def __iter__(self) -> Self:
        self._num_recv = 0
        if self._is_collocated_worker:
            self._collocated_producer.reset()
        elif self._is_mp_worker:
            self._mp_producer.produce_all()
        else:
            rpc_futures: list[torch.futures.Future[None]] = []
            for server_rank, channel_id in zip(
                self._server_rank_list, self._channel_id_list
            ):
                fut = async_request_server(
                    server_rank,
                    DistServer.start_new_epoch_sampling,
                    channel_id,
                    self._epoch,
                )
                rpc_futures.append(fut)
            torch.futures.wait_all(rpc_futures)
            self._channel.reset()
        self._epoch += 1
        return self
