import sys
from collections import abc
from typing import Optional, Tuple, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import (
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
)
from graphlearn_torch.sampler import NodeSamplerInput
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.base_dist_loader import BaseDistLoader
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_ppr_sampler import (
    PPR_EDGE_INDEX_METADATA_KEY,
    PPR_WEIGHT_METADATA_KEY,
)
from gigl.distributed.dist_sampling_producer import DistSamplingProducer
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.sampler_options import (
    PPRSamplerOptions,
    SamplerOptions,
    resolve_sampler_options,
)
from gigl.distributed.utils.neighborloader import (
    DatasetSchema,
    SamplingClusterSetup,
    attach_ppr_outputs,
    extract_edge_type_metadata,
    extract_metadata,
    labeled_to_homogeneous,
    set_missing_features,
    shard_nodes_by_process,
    strip_label_edges,
    strip_non_ppr_edge_types,
)
from gigl.src.common.types.graph_data import (
    NodeType,  # TODO (mkolodner-sc): Change to use torch_geometric.typing
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
)

logger = Logger()


# We don't see logs for graph store mode for whatever reason.
# TOOD(#442): Revert this once the GCP issues are resolved.
def flush():
    sys.stdout.flush()
    sys.stderr.flush()


class DistNeighborLoader(BaseDistLoader):
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
        prefetch_size: Optional[int] = None,
        process_start_gap_seconds: float = 60.0,
        max_concurrent_producer_inits: Optional[int] = None,
        num_cpu_threads: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler_options: Optional[SamplerOptions] = None,
        non_blocking_transfers: bool = True,
    ):
        """
        Distributed Neighbor Loader.
        Takes in some input nodes and samples neighbors from the dataset.
        This loader should be used if you do not have any specially sampling needs,
        e.g. you need to generate *training* examples for Anchor Based Link Prediction (ABLP) tasks.
        Though this loader is useful for generating random negative examples for ABLP training.

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
                If ``KHopNeighborSamplerOptions`` is also provided, they must match.
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
            prefetch_size (Optional[int]): Max number of sampled messages to prefetch on the
                client side, per server. Only applies to Graph Store mode (remote workers).
                Lower values reduce server-side RPC thread contention when multiple loaders
                are active concurrently. (default: ``None``).
                Only applicable in Graph Store mode.
                If supplied and not it Graph Store mode, an error will be raised.
            process_start_gap_seconds (float): Delay between each process for initializing neighbor loader.
                In colocated mode, each process sleeps ``local_rank * process_start_gap_seconds``
                before initializing. In graph store mode, leader ranks are grouped into batches
                of ``max_concurrent_producer_inits`` and each batch sleeps
                ``batch_index * process_start_gap_seconds`` before dispatching RPCs.
            max_concurrent_producer_inits (int): Maximum number of leader ranks that may
                dispatch create-producer RPCs concurrently in graph store mode. Leaders are
                grouped into batches of this size; each batch is staggered by
                ``process_start_gap_seconds``. Only applies to graph store mode.
                Defaults to ``None`` (no staggering).
            num_cpu_threads (Optional[int]): Number of cpu threads PyTorch should use for CPU training/inference
                neighbor loading; on top of the per process parallelism.
                Defaults to `2` if set to `None` when using cpu training/inference.
            shuffle (bool): Whether to shuffle the input nodes. (default: ``False``).
            drop_last (bool): Whether to drop the last incomplete batch. (default: ``False``).
            sampler_options (Optional[SamplerOptions]): Controls which sampler class is
                instantiated. Pass ``KHopNeighborSamplerOptions`` to use the built-in sampler,
                or ``CustomSamplerOptions`` to dynamically import a custom sampler class.
                If ``None``, defaults to ``KHopNeighborSamplerOptions(num_neighbors)``.
            non_blocking_transfers (bool): If True (default), batch-transfers all
                sampled tensors to the target CUDA device using non-blocking copies
                before collation, which can overlap data transfer with computation
                when source tensors reside in pinned memory.  If False, the bulk
                transfer is skipped and GLT's default (blocking) device placement
                is used instead.
                See https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
                for background on pinned memory and non-blocking transfers.
        """

        # Set self._shutdowned right away, that way if we throw here, and __del__ is called,
        # then we can properly clean up and don't get extraneous error messages.
        self._shutdowned = True

        sampler_options = resolve_sampler_options(num_neighbors, sampler_options)

        # Resolve distributed context
        runtime = BaseDistLoader.resolve_runtime(
            context, local_process_rank, local_process_world_size
        )
        del context, local_process_rank, local_process_world_size

        # Determine mode
        if isinstance(dataset, RemoteDistDataset):
            self._sampling_cluster_setup = SamplingClusterSetup.GRAPH_STORE
        else:
            self._sampling_cluster_setup = SamplingClusterSetup.COLOCATED
            if prefetch_size is not None:
                raise ValueError(
                    f"prefetch_size must be None when using Colocated mode, received {prefetch_size}"
                )
            if max_concurrent_producer_inits is not None:
                raise ValueError(
                    f"max_concurrent_producer_inits must be None when using Colocated mode, received {max_concurrent_producer_inits}"
                )
        logger.info(f"Sampling cluster setup: {self._sampling_cluster_setup.value}")

        self._instance_count = next(BaseDistLoader._global_loader_counter)
        device = (
            pin_memory_device
            if pin_memory_device
            else gigl.distributed.utils.get_available_device(
                local_process_rank=runtime.local_rank
            )
        )

        # Mode-specific setup
        if self._sampling_cluster_setup == SamplingClusterSetup.COLOCATED:
            assert isinstance(
                dataset, DistDataset
            ), "When using colocated mode, dataset must be a DistDataset."
            input_data, worker_options, dataset_schema = self._setup_for_colocated(
                input_nodes=input_nodes,
                dataset=dataset,
                local_rank=runtime.local_rank,
                local_world_size=runtime.local_world_size,
                device=device,
                master_ip_address=runtime.master_ip_address,
                node_rank=runtime.node_rank,
                node_world_size=runtime.node_world_size,
                num_workers=num_workers,
                worker_concurrency=worker_concurrency,
                channel_size=channel_size,
                num_cpu_threads=num_cpu_threads,
            )
        else:
            assert isinstance(
                dataset, RemoteDistDataset
            ), "When using Graph Store mode, dataset must be a RemoteDistDataset."
            if prefetch_size is None:
                logger.info(f"prefetch_size is not provided, using default of 4")
                prefetch_size = 4
            input_data, worker_options, dataset_schema = self._setup_for_graph_store(
                input_nodes=input_nodes,
                dataset=dataset,
                num_workers=num_workers,
                worker_concurrency=worker_concurrency,
                prefetch_size=prefetch_size,
                channel_size=channel_size,
            )

        # Cleanup temporary process group if needed
        if (
            runtime.should_cleanup_distributed_context
            and torch.distributed.is_initialized()
        ):
            logger.info(
                f"Cleaning up process group as it was initialized inside {self.__class__.__name__}.__init__."
            )
            torch.distributed.destroy_process_group()

        # Create SamplingConfig (with patched fanout)
        sampling_config = BaseDistLoader.create_sampling_config(
            num_neighbors=num_neighbors,
            dataset_schema=dataset_schema,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        producer: Optional[DistSamplingProducer] = None
        if self._sampling_cluster_setup == SamplingClusterSetup.COLOCATED:
            assert isinstance(dataset, DistDataset)
            assert isinstance(worker_options, MpDistSamplingWorkerOptions)
            channel = BaseDistLoader.create_colocated_channel(worker_options)
            producer = DistSamplingProducer(
                data=dataset,
                sampler_input=input_data,
                sampling_config=sampling_config,
                worker_options=worker_options,
                channel=channel,
                sampler_options=sampler_options,
            )

        # Call base class — handles metadata storage and connection initialization
        # (including staggered init for colocated mode).
        super().__init__(
            dataset=dataset,
            sampler_input=input_data,
            dataset_schema=dataset_schema,
            worker_options=worker_options,
            sampling_config=sampling_config,
            device=device,
            runtime=runtime,
            producer=producer,
            sampler_options=sampler_options,
            process_start_gap_seconds=process_start_gap_seconds,
            max_concurrent_producer_inits=max_concurrent_producer_inits,
            non_blocking_transfers=non_blocking_transfers,
        )

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
        worker_concurrency: int,
        prefetch_size: int,
        channel_size: str,
    ) -> tuple[list[NodeSamplerInput], RemoteDistSamplingWorkerOptions, DatasetSchema]:
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

        node_feature_info = dataset.fetch_node_feature_info()
        edge_feature_info = dataset.fetch_edge_feature_info()
        edge_types = dataset.fetch_edge_types()
        compute_rank = torch.distributed.get_rank()

        self._backend_key = f"dist_neighbor_loader_{self._instance_count}"
        worker_key = f"{self._backend_key}_compute_rank_{compute_rank}"
        logger.info(f"Rank {compute_rank} worker key: {worker_key}")
        worker_options = BaseDistLoader.create_graph_store_worker_options(
            dataset=dataset,
            loader_port_index=self._instance_count,
            worker_key=worker_key,
            num_workers=num_workers,
            worker_concurrency=worker_concurrency,
            channel_size=channel_size,
            prefetch_size=prefetch_size,
        )
        logger.info(
            f"Rank {torch.distributed.get_rank()}! init for sampling rpc: "
            f"tcp://{worker_options.master_addr}:{worker_options.master_port}"
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
            if DEFAULT_HOMOGENEOUS_EDGE_TYPE in edge_types:
                input_type: Optional[NodeType] = DEFAULT_HOMOGENEOUS_NODE_TYPE
            else:
                input_type = fallback_input_type
        elif require_edge_feature_info:
            raise ValueError(
                "When using Graph Store mode, edge types must be provided for heterogeneous graphs."
            )
        else:
            input_type = None

        is_homogeneous_with_labeled_edge_type = (
            input_type == DEFAULT_HOMOGENEOUS_NODE_TYPE
        )

        # Convert from dict to list which is what the GLT DistNeighborLoader expects.
        servers = nodes.keys()
        if max(servers) >= dataset.cluster_info.num_storage_nodes or min(servers) < 0:
            raise ValueError(
                f"When using Graph Store mode, the server ranks must be in range [0, num_servers ({dataset.cluster_info.num_storage_nodes})), received inputs for servers: {list(nodes.keys())}"
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
                edge_dir=dataset.fetch_edge_dir(),
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

        BaseDistLoader.initialize_colocated_sampling_worker(
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_rank=node_rank,
            node_world_size=node_world_size,
            master_ip_address=master_ip_address,
            device=device,
            num_cpu_threads=num_cpu_threads,
        )

        # Sets up worker options for the dataloader
        dist_sampling_ports = gigl.distributed.utils.get_free_ports_from_master_node(
            num_ports=local_world_size
        )
        dist_sampling_port_for_current_rank = dist_sampling_ports[local_rank]

        worker_options = BaseDistLoader.create_colocated_worker_options(
            dataset_num_partitions=dataset.num_partitions,
            num_workers=num_workers,
            worker_concurrency=worker_concurrency,
            master_ip_address=master_ip_address,
            master_port=dist_sampling_port_for_current_rank,
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
        # Extract user-defined metadata before super()._collate_fn, which
        # calls GLT's to_hetero_data.  to_hetero_data misinterprets #META. keys
        # as edge types and fails when edge_dir="out" (tries to call
        # reverse_edge_type on them).  We strip them here and re-apply after.
        # TODO (mkolodner-sc): Remove once GLT's to_hetero_data is fixed.
        metadata, stripped_msg = extract_metadata(msg, self.to_device)
        data = super()._collate_fn(stripped_msg)
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

        if isinstance(self._sampler_options, PPRSamplerOptions):
            matched, metadata = extract_edge_type_metadata(
                metadata=metadata,
                prefixes=[PPR_EDGE_INDEX_METADATA_KEY, PPR_WEIGHT_METADATA_KEY],
            )
            ppr_edge_indices = matched[PPR_EDGE_INDEX_METADATA_KEY]
            ppr_weights = matched[PPR_WEIGHT_METADATA_KEY]
            attach_ppr_outputs(data, ppr_edge_indices, ppr_weights)
            if isinstance(data, HeteroData):
                data = strip_non_ppr_edge_types(data, set(ppr_edge_indices.keys()))

        # Attach any remaining metadata (e.g. custom user-defined keys) directly onto the
        # data object so downstream code can access them via attribute lookup.
        for key, value in metadata.items():
            data[key] = value
        return data
