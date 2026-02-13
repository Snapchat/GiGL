import sys
import time
from collections import Counter, abc
from typing import Optional, Tuple, Union

import torch
from graphlearn_torch.channel import RemoteReceivingChannel, SampleMessage
from graphlearn_torch.distributed import (
    DistLoader,
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
)
from graphlearn_torch.distributed.dist_client import (
    async_request_server,
    request_server,
)
from graphlearn_torch.distributed.dist_context import get_context
from graphlearn_torch.sampler import (
    NodeSamplerInput,
    RemoteSamplerInput,
    SamplingConfig,
    SamplingType,
)
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.constants import DEFAULT_MASTER_INFERENCE_PORT
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.graph_store.dist_server import DistServer as GiglDistServer
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.utils.neighborloader import (
    DatasetSchema,
    SamplingClusterSetup,
    labeled_to_homogeneous,
    patch_fanout_for_sampling,
    set_missing_features,
    shard_nodes_by_process,
    strip_label_edges,
)
from gigl.src.common.types.graph_data import (
    NodeType,  # TODO (mkolodner-sc): Change to use torch_geometric.typing
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
)

logger = Logger()

# When using CPU based inference/training, we default cpu threads for neighborloading on top of the per process parallelism.
DEFAULT_NUM_CPU_THREADS = 2


def flush():
    sys.stdout.flush()
    sys.stderr.flush()


class DistNeighborLoader(DistLoader):
    def __init__(
        self,
        dataset: Union[DistDataset, RemoteDistDataset],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        input_nodes: Optional[
            Union[
                torch.Tensor,
                Tuple[NodeType, torch.Tensor],
                abc.Mapping[int, torch.Tensor],
                Tuple[NodeType, abc.Mapping[int, torch.Tensor]],
            ]
        ] = None,
        num_workers: int = 1,
        batch_size: int = 1,
        context: Optional[DistributedContext] = None,  # TODO: (svij) Deprecate this
        local_process_rank: Optional[int] = None,  # TODO: (svij) Deprecate this
        local_process_world_size: Optional[int] = None,  # TODO: (svij) Deprecate this
        pin_memory_device: Optional[torch.device] = None,
        worker_concurrency: int = 4,
        channel_size: str = "4GB",
        process_start_gap_seconds: float = 60.0,
        num_cpu_threads: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        """
        Note: We try to adhere to pyg dataloader api as much as possible.
        See the following for reference:
        https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/loader/node_loader.html#NodeLoader
        https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/distributed/dist_neighbor_loader.html#DistNeighborLoader

        Args:
            dataset (DistDataset | RemoteDistDataset): The dataset to sample from.
            If this is a `RemoteDistDataset`, then we assumed to be in "Graph Store" mode.
            num_neighbors (list[int] or dict[Tuple[str, str, str], list[int]]):
                The number of neighbors to sample for each node in each iteration.
                If an entry is set to `-1`, all neighbors will be included.
                In heterogeneous graphs, may also take in a dictionary denoting
                the amount of neighbors to sample for each individual edge type.
            context (deprecated - will be removed soon) (DistributedContext): Distributed context information of the current process.
            local_process_rank (deprecated - will be removed soon) (int): Required if context provided. The local rank of the current process within a node.
            local_process_world_size (deprecated - will be removed soon)(int): Required if context provided. The total number of processes within a node.
            input_nodes (Tensor | Tuple[NodeType, Tensor] | dict[int, Tensor] | Tuple[NodeType, dict[int, Tensor]]):
                The nodes to start sampling from.
                It is of type `torch.LongTensor` for homogeneous graphs.
                If set to `None` for homogeneous settings, all nodes will be considered.
                In heterogeneous graphs, this flag must be passed in as a tuple that holds
                the node type and node indices. (default: `None`)
                For Graph Store mode, this must be a tuple of (NodeType, dict[int, Tensor]) or dict[int, Tensor].
                Where each Tensor in the dict is the node ids to sample from, by server.
                e.g. {0: [10, 20], 1: [30, 40]} means sample from nodes 10 and 20 on server 0, and nodes 30 and 40 on server 1.
                If a Graph Store input (e.g. list[Tensor]) is provided to colocated mode, or colocated input (e.g. Tensor) is provided to Graph Store mode,
                then an error will be raised.
            num_workers (int): How many workers to use (subprocesses to spwan) for
                    distributed neighbor sampling of the current process. (default: ``1``).
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            pin_memory_device (str, optional): The target device that the sampled
                results should be copied to. If set to ``None``, the device is inferred based off of
                (got by ``gigl.distributed.utils.device.get_available_device``). Which uses the
                local_process_rank and torch.cuda.device_count() to assign the device. If cuda is not available,
                the cpu device will be used. (default: ``None``).
            worker_concurrency (int): The max sampling concurrency for each sampling
                worker. Load testing has showed that setting worker_concurrency to 4 yields the best performance
                for sampling. Although, you may whish to explore higher/lower settings when performance tuning.
                (default: `4`).
            channel_size (int or str): The shared-memory buffer size (bytes) allocated
                for the channel. Can be modified for performance tuning; a good starting point is: ``num_workers * 64MB``
                (default: "4GB").
            process_start_gap_seconds (float): Delay between each process for initializing neighbor loader. At large scales,
                it is recommended to set this value to be between 60 and 120 seconds -- otherwise multiple processes may
                attempt to initialize dataloaders at overlapping times, which can cause CPU memory OOM.
            num_cpu_threads (Optional[int]): Number of cpu threads PyTorch should use for CPU training/inference
                neighbor loading; on top of the per process parallelism.
                Defaults to `2` if set to `None` when using cpu training/inference.
            shuffle (bool): Whether to shuffle the input nodes. (default: ``False``).
            drop_last (bool): Whether to drop the last incomplete batch. (default: ``False``).
        """

        # Set self._shutdowned right away, that way if we throw here, and __del__ is called,
        # then we can properly clean up and don't get extraneous error messages.
        # We set to `True` as we don't need to cleanup right away, and this will get set
        # to `False` in super().__init__()` e.g.
        # https://github.com/alibaba/graphlearn-for-pytorch/blob/26fe3d4e050b081bc51a79dc9547f244f5d314da/graphlearn_torch/python/distributed/dist_loader.py#L125C1-L126C1
        self._shutdowned = True

        node_world_size: int
        node_rank: int
        rank: int
        world_size: int
        local_rank: int
        local_world_size: int

        master_ip_address: str
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
                    "process group is not available, trying to torch.distributed.init_process_group to communicate necessary setup information."
                )
                should_cleanup_distributed_context = True
                logger.info(
                    f"Initializing process group with master ip address: {master_ip_address}, rank: {rank}, world size: {world_size}, local_rank: {local_rank}, local_world_size: {local_world_size}."
                )
                torch.distributed.init_process_group(
                    backend="gloo",  # We just default to gloo for this temporary process group
                    init_method=f"tcp://{master_ip_address}:{DEFAULT_MASTER_INFERENCE_PORT}",
                    rank=rank,
                    world_size=world_size,
                )

        else:
            assert (
                torch.distributed.is_initialized()
            ), f"context: DistributedContext is None, so process group must be initialized before constructing this object {self.__class__.__name__}."
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

            rank_ip_addresses = gigl.distributed.utils.get_internal_ip_from_all_ranks()
            master_ip_address = rank_ip_addresses[0]

            count_ranks_per_ip_address = Counter(rank_ip_addresses)
            local_world_size = count_ranks_per_ip_address[master_ip_address]
            for rank_ip_address, count in count_ranks_per_ip_address.items():
                if count != local_world_size:
                    raise ValueError(
                        f"All ranks must have the same number of processes, but found {count} processes for rank {rank} on ip {rank_ip_address}, expected {local_world_size}."
                        + f"count_ranks_per_ip_address = {count_ranks_per_ip_address}"
                    )

            node_world_size = len(count_ranks_per_ip_address)
            local_rank = rank % local_world_size
            node_rank = rank // local_world_size

        del (
            context,
            local_process_rank,
            local_process_world_size,
        )  # delete deprecated vars so we don't accidentally use them.
        if isinstance(dataset, RemoteDistDataset):
            self._sampling_cluster_setup = SamplingClusterSetup.GRAPH_STORE
        else:
            self._sampling_cluster_setup = SamplingClusterSetup.COLOCATED
        logger.info(f"Sampling cluster setup: {self._sampling_cluster_setup.value}")
        device = (
            pin_memory_device
            if pin_memory_device
            else gigl.distributed.utils.get_available_device(
                local_process_rank=local_rank
            )
        )

        # Determines if the node ids passed in are heterogeneous or homogeneous.
        if self._sampling_cluster_setup == SamplingClusterSetup.COLOCATED:
            assert isinstance(
                dataset, DistDataset
            ), "When using colocated mode, dataset must be a DistDataset."
            input_data, worker_options, dataset_metadata = self._setup_for_colocated(
                input_nodes,
                dataset,
                local_rank,
                local_world_size,
                device,
                master_ip_address,
                node_rank,
                node_world_size,
                num_workers,
                worker_concurrency,
                channel_size,
                num_cpu_threads,
            )
        else:  # Graph Store mode
            assert isinstance(
                dataset, RemoteDistDataset
            ), "When using Graph Store mode, dataset must be a RemoteDistDataset."
            input_data, worker_options, dataset_metadata = self._setup_for_graph_store(
                input_nodes,
                dataset,
                num_workers,
                channel_size,
            )

        self._is_homogeneous_with_labeled_edge_type = (
            dataset_metadata.is_homogeneous_with_labeled_edge_type
        )
        self._node_feature_info = dataset_metadata.node_feature_info
        self._edge_feature_info = dataset_metadata.edge_feature_info

        logger.info(f"num_neighbors before patch: {num_neighbors}")
        num_neighbors = patch_fanout_for_sampling(
            edge_types=dataset_metadata.edge_types,
            num_neighbors=num_neighbors,
        )
        logger.info(
            f"num_neighbors: {num_neighbors}, edge_types: {dataset_metadata.edge_types}"
        )
        sampling_config = SamplingConfig(
            sampling_type=SamplingType.NODE,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            with_edge=True,
            collect_features=True,
            with_neg=False,
            with_weight=False,
            edge_dir=dataset_metadata.edge_dir,
            seed=None,  # it's actually optional - None means random.
        )

        if should_cleanup_distributed_context and torch.distributed.is_initialized():
            logger.info(
                f"Cleaning up process group as it was initialized inside {self.__class__.__name__}.__init__."
            )
            torch.distributed.destroy_process_group()

        if self._sampling_cluster_setup == SamplingClusterSetup.COLOCATED:
            # When initiating data loader(s), there will be a spike of memory usage lasting for ~30s.
            # The current hypothesis is making connections across machines require a lot of memory.
            # If we start all data loaders in all processes simultaneously, the spike of memory
            # usage will add up and cause CPU memory OOM. Hence, we initiate the data loaders group by group
            # to smooth the memory usage. The definition of group is discussed in init_neighbor_loader_worker.
            logger.info(
                f"---Machine {rank} local process number {local_rank} preparing to sleep for {process_start_gap_seconds * local_rank} seconds"
            )
            time.sleep(process_start_gap_seconds * local_rank)
            super().__init__(
                dataset,  # Pass in the dataset for colocated mode.
                input_data,
                sampling_config,
                device,
                worker_options,
            )
        else:
            # Graph Store mode — initialize DistLoader attributes directly instead of
            # calling super().__init__() to avoid the ThreadPoolExecutor deadlock at scale.
            #
            # GLT's DistLoader.__init__() dispatches create_sampling_producer RPCs via
            # ThreadPoolExecutor(max_workers=32). With 60+ servers, only 32 threads run,
            # causing a TensorPipe rendezvous deadlock. Instead, we inline the DistLoader
            # init code and dispatch all RPCs asynchronously in a simple loop.
            #
            # See docs/graph_store_scale_debug.md for full analysis.

            node_rank = dataset.cluster_info.compute_node_rank
            num_storage_nodes = dataset.cluster_info.num_storage_nodes

            # --- Set all DistLoader attributes (mirrors GLT DistLoader.__init__) ---
            # These are required by inherited methods: shutdown(), __iter__(), __next__(),
            # __del__(), _collate_fn(), _set_ntypes_and_etypes().
            self.data = None  # No local data in Graph Store mode
            self.input_data = input_data
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
            self._shutdowned = False

            self._is_collocated_worker = False
            self._is_mp_worker = False
            self._is_remote_worker = True

            self._num_recv = 0
            self._epoch = 0

            # Context validation
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

            # Remote worker attributes
            self._num_expected = float("inf")
            self._with_channel = True

            self._server_rank_list: list[int] = (
                self.worker_options.server_rank
                if isinstance(self.worker_options.server_rank, list)
                else [self.worker_options.server_rank]
            )
            self._input_data_list: list[NodeSamplerInput] = (
                self.input_data
                if isinstance(self.input_data, list)
                else [self.input_data]
            )
            self._input_type = self._input_data_list[0].input_type

            # --- Barrier loop: one compute node at a time ---
            logger.info(
                f"node_rank {node_rank} starting barrier loop with "
                f"{dataset.cluster_info.num_compute_nodes} compute nodes"
            )
            flush()
            # For Graph Store mode, we need to start the communcation between compute and storage nodes sequentially, by compute node.
            # E.g. intialize connections between compute node 0 and storage nodes 0, 1, 2, 3, then compute node 1 and storage nodes 0, 1, 2, 3, etc.
            # Note that each compute node may have multiple connections to each storage node, once per compute process.
            # It's important to distinguish "compute node" (e.g. physical compute machine) from "compute process" (e.g. process running on the compute node).
            # Since in practice we have multiple compute processes per compute node, and each compute process needs to initialize the connection to the storage nodes.
            # E.g. if there are 4 gpus per compute node, then there will be 4 connections from each compute node to each storage node.
            # We need to this because if we don't, then there is a race condition when initalizing the samplers on the storage nodes [1]
            # Where since the lock is per *server* (e.g. per storage node), if we try to start one connection from compute node 0, and compute node 1
            # Then we deadlock and fail.
            # Specifically, the race condition happens in `DistLoader.__init__` when it initializes the sampling producers on the storage nodes. [2]
            # [1]: https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_server.py#L129-L167
            # [2]: https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/python/distributed/dist_loader.py#L187-L193

            # See below for a connection setup.
            # ╔═══════════════════════════════════════════════════════════════════════════════════════╗
            # ║                         COMPUTE TO STORAGE NODE CONNECTIONS                            ║
            # ╚═══════════════════════════════════════════════════════════════════════════════════════╝

            #      COMPUTE NODES                                              STORAGE NODES
            #      ═════════════                                              ═════════════

            #   ┌──────────────────────┐          (1)                      ┌───────────────┐
            #   │    COMPUTE NODE 0    │                                   │               │
            #   │  ┌────┬────┬────┬────┤ ══════════════════════════════════│   STORAGE 0   │
            #   │  │GPU │GPU │GPU │GPU │                                 ╱ │               │
            #   │  │ 0  │ 1  │ 2  │ 3  │ ════════════════════╲         ╱   └───────────────┘
            #   │  └────┴────┴────┴────┤          (2)          ╲     ╱
            #   └──────────────────────┘                         ╲ ╱
            #                                                     ╳
            #                                           (3)     ╱   ╲     (4)
            #   ┌──────────────────────┐                      ╱       ╲    ┌───────────────┐
            #   │    COMPUTE NODE 1    │                    ╱           ╲  │               │
            #   │  ┌────┬────┬────┬────┤ ═════════════════╱               ═│   STORAGE 1   │
            #   │  │GPU │GPU │GPU │GPU │                                   │               │
            #   │  │ 0  │ 1  │ 2  │ 3  │ ══════════════════════════════════│               │
            #   │  └────┴────┴────┴────┤                                   └───────────────┘
            #   └──────────────────────┘

            #   ┌─────────────────────────────────────────────────────────────────────────────┐
            #   │  (1) Compute Node 0  →  Storage 0   (4 connections, one per GPU)            │
            #   │  (2) Compute Node 0  →  Storage 1   (4 connections, one per GPU)            │
            #   │  (3) Compute Node 1  →  Storage 0   (4 connections, one per GPU)            │
            #   │  (4) Compute Node 1  →  Storage 1   (4 connections, one per GPU)            │
            #   └─────────────────────────────────────────────────────────────────────────────┘
            for target_node_rank in range(dataset.cluster_info.num_compute_nodes):
                start_time = time.time()
                if node_rank == target_node_rank:
                    # Step 1: Get dataset metadata via RPC (single call, fast)
                    (
                        self.num_data_partitions,
                        self.data_partition_idx,
                        ntypes,
                        etypes,
                    ) = request_server(
                        self._server_rank_list[0],
                        GiglDistServer.get_dataset_meta,
                    )
                    self._set_ntypes_and_etypes(ntypes, etypes)

                    # Step 2: Move input data to CPU if needed
                    for i, inp in enumerate(self._input_data_list):
                        if not isinstance(inp, RemoteSamplerInput):
                            self._input_data_list[i] = inp.to(torch.device("cpu"))

                    # Step 3: Dispatch ALL create_sampling_producer RPCs async.
                    #
                    # This is the key fix: async_request_server queues the RPC
                    # in TensorPipe and returns immediately. By dispatching all
                    # N RPCs in a loop BEFORE waiting for any response, all
                    # storage nodes receive the RPC and start their worker
                    # rendezvous simultaneously. No ThreadPoolExecutor needed.
                    logger.info(
                        f"node_rank={node_rank} dispatching "
                        f"create_sampling_producer to "
                        f"{num_storage_nodes} servers"
                    )
                    flush()
                    t_dispatch = time.time()
                    rpc_futures: list[tuple[int, torch.futures.Future[int]]] = []
                    for server_rank, inp_data in zip(
                        self._server_rank_list, self._input_data_list
                    ):
                        fut = async_request_server(
                            server_rank,
                            GiglDistServer.create_sampling_producer,
                            inp_data,
                            self.sampling_config,
                            self.worker_options,
                        )
                        rpc_futures.append((server_rank, fut))
                    logger.info(
                        f"node_rank={node_rank} all "
                        f"{len(rpc_futures)} RPCs dispatched in "
                        f"{time.time() - t_dispatch:.3f}s, "
                        f"waiting for responses"
                    )
                    flush()

                    # Step 4: Wait for all results
                    self._producer_id_list: list[int] = []
                    for server_rank, fut in rpc_futures:
                        t_wait = time.time()
                        producer_id: int = fut.wait()
                        logger.info(
                            f"node_rank={node_rank} "
                            f"create_sampling_producer"
                            f"(server_rank={server_rank}) returned "
                            f"producer_id={producer_id} in "
                            f"{time.time() - t_wait:.2f}s"
                        )
                        flush()
                        self._producer_id_list.append(producer_id)
                    logger.info(
                        f"node_rank={node_rank} all "
                        f"{len(self._producer_id_list)} producers created "
                        f"in {time.time() - t_dispatch:.2f}s total"
                    )
                    flush()

                    # Step 5: Create remote receiving channel
                    self._channel = RemoteReceivingChannel(
                        self._server_rank_list,
                        self._producer_id_list,
                        self.worker_options.prefetch_size,
                    )

                    logger.info(
                        f"node_rank {node_rank} initialized the dist loader in "
                        f"{time.time() - start_time:.2f}s"
                    )
                    flush()
                else:
                    logger.info(
                        f"node_rank {node_rank} waiting for barrier "
                        f"for rank {target_node_rank}"
                    )
                    flush()
                torch.distributed.barrier(device_ids=torch.device("cpu"))
                logger.info(
                    f"node_rank {node_rank} barrier for rank "
                    f"{target_node_rank} in {time.time() - start_time:.2f}s"
                )
                flush()

            torch.distributed.barrier(device_ids=torch.device("cpu"))
            logger.info(
                f"node_rank {node_rank}: all "
                f"{dataset.cluster_info.num_compute_nodes} node ranks "
                f"initialized the dist loader"
            )
            flush()

    def _setup_for_graph_store(
        self,
        input_nodes: Optional[
            Union[
                torch.Tensor,
                tuple[NodeType, torch.Tensor],
                abc.Mapping[int, torch.Tensor],
                tuple[NodeType, abc.Mapping[int, torch.Tensor]],
            ]
        ],
        dataset: RemoteDistDataset,
        num_workers: int,
        channel_size: str,
    ) -> tuple[NodeSamplerInput, RemoteDistSamplingWorkerOptions, DatasetSchema]:
        if input_nodes is None:
            raise ValueError(
                f"When using Graph Store mode, input nodes must be provided, received {input_nodes}"
            )
        elif isinstance(input_nodes, torch.Tensor):
            raise ValueError(
                f"When using Graph Store mode, input nodes must be of type (abc.Mapping[int, torch.Tensor] | (NodeType, abc.Mapping[int, torch.Tensor]), received {type(input_nodes)}"
            )
        elif isinstance(input_nodes, tuple) and isinstance(
            input_nodes[1], torch.Tensor
        ):
            raise ValueError(
                f"When using Graph Store mode, input nodes must be of type (dict[int, torch.Tensor] | (NodeType, dict[int, torch.Tensor])), received {type(input_nodes)} ({type(input_nodes[0])}, {type(input_nodes[1])})"
            )

        is_homogeneous_with_labeled_edge_type = False
        node_feature_info = dataset.get_node_feature_info()
        edge_feature_info = dataset.get_edge_feature_info()
        edge_types = dataset.get_edge_types()
        node_rank = dataset.cluster_info.compute_node_rank

        # Get sampling ports for compute-storage connections.
        sampling_ports = dataset.get_free_ports_on_storage_cluster(
            num_ports=dataset.cluster_info.num_compute_nodes
        )
        sampling_port = sampling_ports[node_rank]

        worker_options = RemoteDistSamplingWorkerOptions(
            server_rank=list(range(dataset.cluster_info.num_storage_nodes)),
            num_workers=num_workers,
            worker_devices=[torch.device("cpu") for i in range(num_workers)],
            master_addr=dataset.cluster_info.storage_cluster_master_ip,
            buffer_size=channel_size,
            master_port=sampling_port,
            worker_key=f"compute_rank_{node_rank}",
        )
        logger.info(
            f"Rank {torch.distributed.get_rank()}! init for sampling rpc: {f'tcp://{dataset.cluster_info.storage_cluster_master_ip}:{sampling_port}'}"
        )

        # Setup input data for the dataloader.

        # Determine nodes list and fallback input_type based on input_nodes structure
        if isinstance(input_nodes, abc.Mapping):
            nodes = input_nodes
            fallback_input_type = None
            require_edge_feature_info = False
        elif isinstance(input_nodes, tuple) and isinstance(input_nodes[1], abc.Mapping):
            nodes = input_nodes[1]
            fallback_input_type = input_nodes[0]
            require_edge_feature_info = True
        else:
            raise ValueError(
                f"When using Graph Store mode, input nodes must be of type (abc.Mapping[int, torch.Tensor] | (NodeType, abc.Mapping[int, torch.Tensor])), received {type(input_nodes)}"
            )

        # Determine input_type based on edge_feature_info
        if isinstance(edge_types, list):
            if edge_types == [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                input_type: Optional[NodeType] = DEFAULT_HOMOGENEOUS_NODE_TYPE
            else:
                input_type = fallback_input_type
        elif require_edge_feature_info:
            raise ValueError(
                "When using Graph Store mode, edge types must be provided for heterogeneous graphs."
            )
        else:
            input_type = None

        # Convert from dict to list which is what the GLT DistNeighborLoader expects.
        servers = nodes.keys()
        if max(servers) >= dataset.cluster_info.num_storage_nodes or min(servers) < 0:
            raise ValueError(
                f"When using Graph Store mode, the server ranks must be less than the number of storage nodes and greater than 0, received inputs for servers: {list(nodes.keys())}"
            )
        input_data: list[NodeSamplerInput] = []
        for server_rank in range(dataset.cluster_info.num_storage_nodes):
            if server_rank in nodes:
                input_data.append(
                    NodeSamplerInput(node=nodes[server_rank], input_type=input_type)
                )
            else:
                input_data.append(
                    NodeSamplerInput(
                        node=torch.empty(0, dtype=torch.long), input_type=input_type
                    )
                )
        return (
            input_data,
            worker_options,
            DatasetSchema(
                is_homogeneous_with_labeled_edge_type=is_homogeneous_with_labeled_edge_type,
                edge_types=edge_types,
                node_feature_info=node_feature_info,
                edge_feature_info=edge_feature_info,
                edge_dir=dataset.get_edge_dir(),
            ),
        )

    def _setup_for_colocated(
        self,
        input_nodes: Optional[
            Union[
                torch.Tensor,
                Tuple[NodeType, torch.Tensor],
                abc.Mapping[int, torch.Tensor],
                Tuple[NodeType, abc.Mapping[int, torch.Tensor]],
            ]
        ],
        dataset: DistDataset,
        local_rank: int,
        local_world_size: int,
        device: torch.device,
        master_ip_address: str,
        node_rank: int,
        node_world_size: int,
        num_workers: int,
        worker_concurrency: int,
        channel_size: str,
        num_cpu_threads: Optional[int],
    ) -> tuple[NodeSamplerInput, MpDistSamplingWorkerOptions, DatasetSchema]:
        if input_nodes is None:
            if dataset.node_ids is None:
                raise ValueError(
                    "Dataset must have node ids if input_nodes are not provided."
                )
            if isinstance(dataset.node_ids, abc.Mapping):
                raise ValueError(
                    f"input_nodes must be provided for heterogeneous datasets, received node_ids of type: {dataset.node_ids.keys()}"
                )
            input_nodes = dataset.node_ids
        if isinstance(input_nodes, abc.Mapping):
            raise ValueError(
                f"When using Colocated mode, input nodes must be of type (torch.Tensor | (NodeType, torch.Tensor)), received {type(input_nodes)}"
            )
        elif isinstance(input_nodes, tuple) and isinstance(input_nodes[1], abc.Mapping):
            raise ValueError(
                f"When using Colocated mode, input nodes must be of type (torch.Tensor | (NodeType, torch.Tensor)), received {type(input_nodes)} ({type(input_nodes[0])}, {type(input_nodes[1])})"
            )
        is_homogeneous_with_labeled_edge_type = False
        if isinstance(input_nodes, torch.Tensor):
            node_ids = input_nodes

            # If the dataset is heterogeneous, we may be in the "labeled homogeneous" setting,
            # if so, then we should use DEFAULT_HOMOGENEOUS_NODE_TYPE.
            if isinstance(dataset.node_ids, abc.Mapping):
                if (
                    len(dataset.node_ids) == 1
                    and DEFAULT_HOMOGENEOUS_NODE_TYPE in dataset.node_ids
                ):
                    node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
                    is_homogeneous_with_labeled_edge_type = True
                else:
                    raise ValueError(
                        f"For heterogeneous datasets, input_nodes must be a tuple of (node_type, node_ids) OR if it is a labeled homogeneous dataset, input_nodes may be a torch.Tensor. Received node types: {dataset.node_ids.keys()}"
                    )
            else:
                node_type = None
        else:
            node_type, node_ids = input_nodes
            assert isinstance(
                dataset.node_ids, abc.Mapping
            ), "Dataset must be heterogeneous if provided input nodes are a tuple."

        curr_process_nodes = shard_nodes_by_process(
            input_nodes=node_ids,
            local_process_rank=local_rank,
            local_process_world_size=local_world_size,
        )

        input_data = NodeSamplerInput(node=curr_process_nodes, input_type=node_type)

        # Sets up processes and torch device for initializing the GLT DistNeighborLoader, setting up RPC and worker groups to minimize
        # the memory overhead and CPU contention.
        logger.info(
            f"Initializing neighbor loader worker in process: {local_rank}/{local_world_size} using device: {device}"
        )
        should_use_cpu_workers = device.type == "cpu"
        if should_use_cpu_workers and num_cpu_threads is None:
            logger.info(
                "Using CPU workers, but found num_cpu_threads to be None. "
                f"Will default setting num_cpu_threads to {DEFAULT_NUM_CPU_THREADS}."
            )
            num_cpu_threads = DEFAULT_NUM_CPU_THREADS

        neighbor_loader_ports = gigl.distributed.utils.get_free_ports_from_master_node(
            num_ports=local_world_size
        )
        neighbor_loader_port_for_current_rank = neighbor_loader_ports[local_rank]

        gigl.distributed.utils.init_neighbor_loader_worker(
            master_ip_address=master_ip_address,
            local_process_rank=local_rank,
            local_process_world_size=local_world_size,
            rank=node_rank,
            world_size=node_world_size,
            master_worker_port=neighbor_loader_port_for_current_rank,
            device=device,
            should_use_cpu_workers=should_use_cpu_workers,
            # Lever to explore tuning for CPU based inference
            num_cpu_threads=num_cpu_threads,
        )
        logger.info(
            f"Finished initializing neighbor loader worker:  {local_rank}/{local_world_size}"
        )

        # Sets up worker options for the dataloader
        dist_sampling_ports = gigl.distributed.utils.get_free_ports_from_master_node(
            num_ports=local_world_size
        )
        dist_sampling_port_for_current_rank = dist_sampling_ports[local_rank]

        worker_options = MpDistSamplingWorkerOptions(
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
            master_port=dist_sampling_port_for_current_rank,
            # Load testing show that when num_rpc_threads exceed 16, the performance
            # will degrade.
            num_rpc_threads=min(dataset.num_partitions, 16),
            rpc_timeout=600,
            channel_size=channel_size,
            pin_memory=device.type == "cuda",
        )

        if isinstance(dataset.graph, dict):
            edge_types = list(dataset.graph.keys())
        else:
            edge_types = None

        return (
            input_data,
            worker_options,
            DatasetSchema(
                is_homogeneous_with_labeled_edge_type=is_homogeneous_with_labeled_edge_type,
                edge_types=edge_types,
                node_feature_info=dataset.node_feature_info,
                edge_feature_info=dataset.edge_feature_info,
                edge_dir=dataset.edge_dir,
            ),
        )

    def _collate_fn(self, msg: SampleMessage) -> Union[Data, HeteroData]:
        data = super()._collate_fn(msg)
        data = set_missing_features(
            data=data,
            node_feature_info=self._node_feature_info,
            edge_feature_info=self._edge_feature_info,
            device=self.to_device,
        )
        if isinstance(data, HeteroData):
            data = strip_label_edges(data)
        if self._is_homogeneous_with_labeled_edge_type:
            data = labeled_to_homogeneous(DEFAULT_HOMOGENEOUS_EDGE_TYPE, data)
        return data
