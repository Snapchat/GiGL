"""
Base distributed loader that consolidates shared initialization logic
from DistNeighborLoader and DistABLPLoader.

Subclasses GLT's DistLoader and handles:
- Dataset metadata storage
- Colocated mode: DistLoader attribute setting + staggered producer init
- Graph Store mode: barrier loop + async RPC dispatch + channel creation
"""

import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage, ShmChannel
from graphlearn_torch.distributed import (
    DistLoader,
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
    get_context,
)
from graphlearn_torch.distributed.rpc import rpc_is_initialized
from graphlearn_torch.sampler import (
    EdgeSamplerInput,
    NodeSamplerInput,
    RemoteSamplerInput,
    SamplingConfig,
    SamplingType,
)
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType
from typing_extensions import Self

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.constants import DEFAULT_MASTER_INFERENCE_PORT
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_sampling_producer import DistSamplingProducer
from gigl.distributed.graph_store.compute import async_request_server
from gigl.distributed.graph_store.dist_server import DistServer
from gigl.distributed.graph_store.messages import (
    InitSamplingBackendRequest,
    RegisterBackendRequest,
)
from gigl.distributed.graph_store.remote_channel import RemoteReceivingChannel
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.sampler_options import PPRSamplerOptions, SamplerOptions
from gigl.distributed.utils.neighborloader import (
    DatasetSchema,
    patch_fanout_for_sampling,
)
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE

logger = Logger()

DEFAULT_NUM_CPU_THREADS = 2


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
    5. For graph store: prepare remote inputs and worker options; the base class
       handles the two-phase backend/channel RPC initialization.
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
        producer: Optional pre-constructed ``DistSamplingProducer`` for colocated mode.
        sampler_options: Controls which sampler class is instantiated.
        backend_key: Unique key identifying the shared sampling backend for this
            loader instance. Required for graph store mode; must be ``None`` for
            colocated mode.
        process_start_gap_seconds: Delay between each process for staggered colocated init.
            Only applies to colocated mode.
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
            assert local_process_world_size is not None, (
                "context: DistributedContext provided, so local_process_world_size must be provided."
            )
            assert local_process_rank is not None, (
                "context: DistributedContext provided, so local_process_rank must be provided."
            )

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
        producer: Optional[DistSamplingProducer],
        sampler_options: SamplerOptions,
        backend_key: Optional[str] = None,
        process_start_gap_seconds: float = 60.0,
        non_blocking_transfers: bool = True,
    ):
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
        self._non_blocking_transfers = non_blocking_transfers
        self._backend_key = backend_key

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
        if (
            isinstance(dataset, DistDataset)
            and isinstance(worker_options, MpDistSamplingWorkerOptions)
            and isinstance(producer, DistSamplingProducer)
        ):
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

            self._remote_input_has_batches: list[bool] = []
            self._shutdowned = False
            self._init_colocated_connections(
                dataset=dataset,
                producer=producer,
                runtime=runtime,
                process_start_gap_seconds=process_start_gap_seconds,
            )
        elif isinstance(dataset, RemoteDistDataset) and isinstance(
            worker_options, RemoteDistSamplingWorkerOptions
        ):
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

            self._remote_input_has_batches = [
                self._sampler_input_has_batches(inp) for inp in self._input_data_list
            ]
            self._shutdowned = False
            self._init_graph_store_connections(dataset=dataset)
        else:
            raise TypeError(
                "Invalid loader construction. Expected either "
                "(DistDataset, MpDistSamplingWorkerOptions, DistSamplingProducer) "
                "for colocated mode or (RemoteDistDataset, RemoteDistSamplingWorkerOptions) "
                "for graph-store mode."
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

    @staticmethod
    def create_mp_producer(
        dataset: DistDataset,
        sampler_input: Union[NodeSamplerInput, EdgeSamplerInput],
        sampling_config: SamplingConfig,
        worker_options: MpDistSamplingWorkerOptions,
        sampler_options: SamplerOptions,
    ) -> DistSamplingProducer:
        """Create a colocated-mode DistSamplingProducer with pre-computed degree tensors.

        Creates the shared-memory channel and, for PPR sampling, pre-computes
        degree tensors via all-reduce before constructing the producer.  The
        all-reduce must happen here — before the staggered sleep in
        ``_init_colocated_connections`` — so that all ranks complete the
        collective together and the stagger applies only to worker spawning.

        Args:
            dataset: The local DistDataset for this rank.
            sampler_input: Node or edge sampler input (ABLPNodeSamplerInput is
                also accepted as it extends NodeSamplerInput).
            sampling_config: Sampling configuration.
            worker_options: Colocated worker options (must be fully configured).
            sampler_options: Controls which sampler class is instantiated.

        Returns:
            A fully constructed DistSamplingProducer, ready to be passed to
            ``_init_colocated_connections``.
        """
        channel = BaseDistLoader.create_colocated_channel(worker_options)
        if isinstance(sampler_options, PPRSamplerOptions):
            degree_tensors = dataset.degree_tensor
            if isinstance(degree_tensors, dict):
                logger.info(
                    f"Pre-computed degree tensors for PPR sampling across "
                    f"{len(degree_tensors)} edge types."
                )
            else:
                logger.info(
                    f"Pre-computed degree tensor for PPR sampling with "
                    f"{degree_tensors.size(0)} nodes."
                )
        else:
            degree_tensors = None
        return DistSamplingProducer(
            data=dataset,
            sampler_input=sampler_input,
            sampling_config=sampling_config,
            worker_options=worker_options,
            channel=channel,
            sampler_options=sampler_options,
            degree_tensors=degree_tensors,
        )

    @staticmethod
    def initialize_colocated_sampling_worker(
        *,
        local_rank: int,
        local_world_size: int,
        node_rank: int,
        node_world_size: int,
        master_ip_address: str,
        device: torch.device,
        num_cpu_threads: Optional[int],
    ) -> None:
        """Initialize the colocated GLT worker group for the current process.

        Args:
            local_rank: Local rank of the current process on this machine.
            local_world_size: Total number of local processes on this machine.
            node_rank: Rank of the current machine.
            node_world_size: Total number of machines in the cluster.
            master_ip_address: Master node IP address used for worker-group setup.
            device: Device assigned to this loader process.
            num_cpu_threads: Optional PyTorch CPU thread count override.
        """
        neighbor_loader_ports = gigl.distributed.utils.get_free_ports_from_master_node(
            num_ports=local_world_size
        )
        neighbor_loader_port_for_current_rank = neighbor_loader_ports[local_rank]
        logger.info(
            f"Initializing neighbor loader worker in process: {local_rank}/{local_world_size} "
            f"using device: {device} on port {neighbor_loader_port_for_current_rank}."
        )

        should_use_cpu_workers = device.type == "cpu"
        if should_use_cpu_workers and num_cpu_threads is None:
            logger.info(
                "Using CPU workers, but found num_cpu_threads to be None. "
                f"Will default setting num_cpu_threads to {DEFAULT_NUM_CPU_THREADS}."
            )
            num_cpu_threads = DEFAULT_NUM_CPU_THREADS

        gigl.distributed.utils.init_neighbor_loader_worker(
            master_ip_address=master_ip_address,
            local_process_rank=local_rank,
            local_process_world_size=local_world_size,
            rank=node_rank,
            world_size=node_world_size,
            master_worker_port=neighbor_loader_port_for_current_rank,
            device=device,
            should_use_cpu_workers=should_use_cpu_workers,
            num_cpu_threads=num_cpu_threads,
        )
        logger.info(
            f"Finished initializing neighbor loader worker: {local_rank}/{local_world_size}"
        )

    @staticmethod
    def create_colocated_worker_options(
        *,
        dataset_num_partitions: int,
        num_workers: int,
        worker_concurrency: int,
        master_ip_address: str,
        master_port: int,
        channel_size: str,
        pin_memory: bool,
    ) -> MpDistSamplingWorkerOptions:
        """Create worker options for colocated sampling workers.

        Args:
            dataset_num_partitions: Number of graph partitions in the colocated dataset.
            num_workers: Number of sampling worker processes.
            worker_concurrency: Max sampling concurrency per worker.
            master_ip_address: Master node IP address used by GLT RPC.
            master_port: Port for the GLT sampling worker group.
            channel_size: Shared-memory channel size.
            pin_memory: Whether the output channel should be pinned.

        Returns:
            Fully configured worker options for colocated sampling.
        """
        return MpDistSamplingWorkerOptions(
            num_workers=num_workers,
            worker_devices=[torch.device("cpu") for _ in range(num_workers)],
            worker_concurrency=worker_concurrency,
            # Each worker will spawn several sampling workers, and all sampling workers spawned by workers in one group
            # need to be connected. Thus, we need master ip address and master port to
            # initate the connection.
            # Note that different groups of workers are independent, and thus
            # the sampling processes in different groups should be independent, and should
            # use different master ports.
            master_addr=master_ip_address,
            master_port=master_port,
            # Load testing shows that when num_rpc_threads exceed 16, the performance
            # will degrade.
            num_rpc_threads=min(dataset_num_partitions, 16),
            rpc_timeout=600,
            channel_size=channel_size,
            pin_memory=pin_memory,
        )

    @staticmethod
    def create_graph_store_worker_options(
        *,
        dataset: RemoteDistDataset,
        worker_key: str,
        num_workers: int,
        worker_concurrency: int,
        channel_size: str,
        prefetch_size: int,
    ) -> RemoteDistSamplingWorkerOptions:
        """Create worker options for graph-store sampling workers.

        Args:
            dataset: Remote dataset proxy used to discover storage-cluster topology.
            worker_key: Unique key used by the storage cluster to deduplicate producers.
            num_workers: Number of sampling worker processes.
            worker_concurrency: Max sampling concurrency per worker.
            channel_size: Remote shared-memory buffer size.
            prefetch_size: Max prefetched messages per storage server.

        Returns:
            Fully configured worker options for graph-store sampling.
        """
        sampling_ports = dataset.fetch_free_ports_on_storage_cluster(num_ports=1)
        sampling_port = sampling_ports[0]
        return RemoteDistSamplingWorkerOptions(
            server_rank=list(range(dataset.cluster_info.num_storage_nodes)),
            num_workers=num_workers,
            worker_devices=[torch.device("cpu") for _ in range(num_workers)],
            worker_concurrency=worker_concurrency,
            master_addr=dataset.cluster_info.storage_cluster_master_ip,
            buffer_size=channel_size,
            master_port=sampling_port,
            worker_key=worker_key,
            prefetch_size=prefetch_size,
        )

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

    def _init_graph_store_sampling_backends(self) -> list[int]:
        """Initialize or reuse one shared backend per storage server.

        Every compute rank issues one RPC per storage server.
        ``DistServer.init_sampling_backend`` deduplicates concurrent calls
        with the same ``backend_key`` internally (the first caller creates
        the backend; subsequent callers block on ``backend_state.lock``
        until init completes), so all ranks observe the same ``backend_id``
        on each server without explicit rank-level coordination.

        Returns:
            List of backend IDs, one per storage server.

        Raises:
            RuntimeError: If ``_backend_key`` was not set.
        """
        if self._backend_key is None:
            raise RuntimeError(
                f"{type(self).__name__} was constructed without a backend_key. "
                "Graph-store mode requires a non-None backend_key."
            )
        futures: list[torch.futures.Future[int]] = [
            async_request_server(
                server_rank,
                DistServer.init_sampling_backend,
                InitSamplingBackendRequest(
                    backend_key=self._backend_key,
                    worker_options=self.worker_options,
                    sampler_options=self._sampler_options,
                    sampling_config=self.sampling_config,
                ),
            )
            for server_rank in self._server_rank_list
        ]
        return torch.futures.wait_all(futures)

    def _register_graph_store_sampling_inputs(
        self, backend_id_list: list[int]
    ) -> list[int]:
        """Register this compute rank's inputs on existing shared backends.

        Each compute rank has a unique ``worker_key``, so registrations are
        naturally per-rank and do not need cross-rank coordination.

        Args:
            backend_id_list: Backend IDs from ``_init_graph_store_sampling_backends``.

        Returns:
            List of channel IDs, one per storage server.
        """
        assert (
            len(self._server_rank_list)
            == len(backend_id_list)
            == len(self._input_data_list)
        ), (
            f"Mismatched lengths: server_rank_list={len(self._server_rank_list)}, "
            f"backend_id_list={len(backend_id_list)}, "
            f"input_data_list={len(self._input_data_list)}"
        )
        worker_key = self.worker_options.worker_key
        futures: list[torch.futures.Future[int]] = [
            async_request_server(
                server_rank,
                DistServer.register_sampling_input,
                RegisterBackendRequest(
                    backend_id=backend_id,
                    worker_key=worker_key,
                    sampler_input=input_data,
                    sampling_config=self.sampling_config,
                    buffer_capacity=self.worker_options.buffer_capacity,
                    buffer_size=self.worker_options.buffer_size,
                ),
            )
            for server_rank, backend_id, input_data in zip(
                self._server_rank_list,
                backend_id_list,
                self._input_data_list,
            )
        ]
        return torch.futures.wait_all(futures)

    def _sampler_input_has_batches(self, sampler_input: NodeSamplerInput) -> bool:
        """Return whether this sampler input can produce at least one batch.

        Args:
            sampler_input: The sampler input to check.

        Returns:
            True if the input has enough elements for at least one batch.
        """
        input_len = len(sampler_input)
        return input_len > 0 and not (self.drop_last and input_len < self.batch_size)

    def _init_graph_store_connections(self, dataset: RemoteDistDataset) -> None:
        """Initialize graph-store mode with shared backends and per-rank channels.

        Populates two parallel lists indexed by storage server. A compute rank
        corresponds to one loader process; with N compute machines and P
        processes per machine there are N*P compute ranks.

        ``_backend_id_list``
            One backend per storage server, shared across all compute ranks
            using the same loader instance (keyed by ``_backend_key``).
            Used when the operation targets the backend itself (e.g.
            ``init_sampling_backend``).
            Note that when there are multiple loader instances
            (e.g. and ABLP and DistLoader for training), each instance
            will have it's own backend id, per server.

        ``_channel_id_list``
            One channel per storage server, unique to this compute rank and
            loader instance. Used for per-rank operations (e.g.
            ``start_new_epoch_sampling``, ``fetch_one_sampled_message``,
            ``destroy_sampling_input``).

        Invariants:

        * ``len(backend_id_list) == len(channel_id_list) == num_storage_servers``.
        * All compute ranks sharing a loader instance see the same
          ``backend_id_list``. Server-side dedup on ``backend_key`` (via
          ``DistServer._backend_id_by_backend_key``) guarantees that every
          rank's concurrent ``init_sampling_backend`` RPC returns the same
          ID on a given server.
        * Each storage server maintains its own ``_next_backend_id`` and
          ``_next_channel_id`` counters. Values on different servers advance
          independently; cross-server numeric equality is not guaranteed
          (partial init failures or other operations on one server can
          desynchronize counters).
        * Within a single server, channel IDs are unique across all
          registrations (no dedup; each ``register_sampling_input`` call
          allocates a fresh monotonic ID). A single compute rank may hold
          multiple channel IDs on the same server if it owns multiple
          concurrent loader instances.
        """
        ctx = get_context()
        if ctx is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}': the distributed context "
                f"has not been initialized."
            )
        if not ctx.is_client():
            raise RuntimeError(
                f"'{self.__class__.__name__}': must be used on a client worker process."
            )

        for inp in self._input_data_list:
            if not isinstance(inp, RemoteSamplerInput):
                inp.to(torch.device("cpu"))

        start_time = time.time()
        self._backend_id_list = self._init_graph_store_sampling_backends()
        self._channel_id_list = self._register_graph_store_sampling_inputs(
            backend_id_list=self._backend_id_list,
        )
        self._channel = RemoteReceivingChannel(
            server_rank=self._server_rank_list,
            channel_id=self._channel_id_list,
            prefetch_size=self.worker_options.prefetch_size,
            active_mask=self._remote_input_has_batches,
            pin_memory=self.to_device is not None and self.to_device.type == "cuda",
        )
        logger.info(
            f"node_rank {dataset.cluster_info.compute_node_rank} "
            f"rank={torch.distributed.get_rank()} "
            f"initialized shared graph-store loader in {time.time() - start_time:.2f}s"
        )
        _flush()

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

    def _collate_fn(self, msg: SampleMessage) -> Union[Data, HeteroData]:
        """Override GLT's _collate_fn to optionally batch-transfer tensors with non_blocking=True.

        When ``_non_blocking_transfers`` is enabled (default), moves all tensors
        in the SampleMessage to the target CUDA device using non-blocking copies
        before delegating to the parent ``_collate_fn``.  This is effective when
        source tensors reside in pinned memory, allowing host-to-device transfers
        to overlap with other work on the default CUDA stream.

        When ``_non_blocking_transfers`` is disabled, the bulk transfer is skipped
        entirely and GLT's default (blocking) device placement is used instead.

        See https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
        for background on pinned memory and non-blocking transfers.
        """
        if (
            self._non_blocking_transfers
            and self.to_device is not None
            and self.to_device.type == "cuda"
        ):
            for k, v in msg.items():
                if isinstance(v, torch.Tensor) and v.device != self.to_device:
                    msg[k] = v.to(self.to_device, non_blocking=True)
            # Synchronize the current CUDA stream to ensure all non-blocking
            # transfers are complete before the parent _collate_fn processes
            # the message.
            torch.cuda.current_stream().synchronize()
        return super()._collate_fn(msg)

    # Overwrite DistLoader.__iter__ to so we can use our own __iter__ and rpc calls
    def __iter__(self) -> Self:
        self._num_recv = 0
        if self._is_collocated_worker:
            self._collocated_producer.reset()
        elif self._is_mp_worker:
            self._mp_producer.produce_all()
        else:
            rpc_futures: list[torch.futures.Future[None]] = []
            for server_rank, channel_id, has_batches in zip(
                self._server_rank_list,
                self._channel_id_list,
                self._remote_input_has_batches,
            ):
                if not has_batches:
                    continue
                fut = async_request_server(
                    server_rank,
                    DistServer.start_new_epoch_sampling,
                    channel_id,
                    self._epoch,
                )
                rpc_futures.append(fut)
            # Match GLT's remote-loader ordering: do not begin fetching until
            # every storage server has acknowledged the epoch start for this
            # channel. Otherwise a fetch RPC can race ahead and block the
            # corresponding start-epoch RPC on the server-side channel lock.
            torch.futures.wait_all(rpc_futures)
            self._channel.reset()
        self._epoch += 1
        return self
