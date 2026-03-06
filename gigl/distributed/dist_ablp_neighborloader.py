import ast
from collections import abc, defaultdict
from itertools import count
from typing import Callable, Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import (
    MpDistSamplingWorkerOptions,
    RemoteDistSamplingWorkerOptions,
)
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.base_dist_loader import BaseDistLoader
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_sampling_producer import DistSamplingProducer
from gigl.distributed.distributed_neighborloader import DEFAULT_NUM_CPU_THREADS
from gigl.distributed.graph_store.dist_server import DistServer
from gigl.distributed.graph_store.remote_dist_dataset import RemoteDistDataset
from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
    metadata_key_with_prefix,
)
from gigl.distributed.sampler_options import SamplerOptions, resolve_sampler_options
from gigl.distributed.utils.neighborloader import (
    DatasetSchema,
    SamplingClusterSetup,
    labeled_to_homogeneous,
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
    label_edge_type_to_message_passing_edge_type,
    message_passing_to_negative_label,
    message_passing_to_positive_label,
    reverse_edge_type,
    select_label_edge_types,
)
from gigl.utils.data_splitters import get_labels_for_anchor_nodes
from gigl.utils.sampling import ABLPInputNodes

logger = Logger()


class DistABLPLoader(BaseDistLoader):
    # Counts instantiations of this class, per process.
    # This is needed so we can generate unique worker key for each instance, for graph store mode.
    # NOTE: This is per-class, not per-instance.
    _counter = count(0)

    def __init__(
        self,
        dataset: Union[DistDataset, RemoteDistDataset],
        num_neighbors: Optional[Union[list[int], dict[EdgeType, list[int]]]] = None,
        input_nodes: Optional[
            Union[
                torch.Tensor,
                tuple[NodeType, torch.Tensor],
                # Graph Store mode inputs
                dict[int, ABLPInputNodes],
            ]
        ] = None,
        supervision_edge_type: Optional[Union[EdgeType, list[EdgeType]]] = None,
        num_workers: int = 1,
        batch_size: int = 1,
        pin_memory_device: Optional[torch.device] = None,
        worker_concurrency: int = 4,
        prefetch_size: Optional[int] = None,
        channel_size: str = "4GB",
        process_start_gap_seconds: float = 60.0,
        num_cpu_threads: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler_options: Optional[SamplerOptions] = None,
        context: Optional[DistributedContext] = None,  # TODO: (svij) Deprecate this
        local_process_rank: Optional[int] = None,  # TODO: (svij) Deprecate this
        local_process_world_size: Optional[int] = None,  # TODO: (svij) Deprecate this
    ):
        """
        Neighbor loader for Anchor Based Link Prediction (ABLP) tasks.

        Note that for this class, the dataset must *always* be heterogeneous,
        as we need separate edge types for positive and negative labels.

        By default, the loader will return {py:class} `torch_geometric.data.HeteroData` (heterogeneous) objects,
        but will return a {py:class}`torch_geometric.data.Data` (homogeneous) object if the dataset is "labeled homogeneous".

        The following fields may also be present:
        - `y_positive`: `dict[int, torch.Tensor]` mapping from local anchor node id to a tensor of positive
                label node ids.
        - `y_negative`: (Optional) `dict[int, torch.Tensor]` mapping from local anchor node id to a tensor of negative
                label node ids. This will only be present if the supervision edge type has negative labels.


        NOTE: for both y_positive, and y_negative, the values represented in both the key and value of the dicts are
        the *local* node ids of the sampled nodes, not the global node ids.
        In order to get the global node ids, you can use the `node` field of the Data/HeteroData object.
        e.g. global_positive_node_id_labels = data.node[data.y_positive[local_anchor_node_id]].

        The underlying graph engine may also add the following fields to the output Data object:
            - num_sampled_nodes: If heterogeneous. a dictionary mapping from node type to the number of sampled nodes for that type, by hop.
            if homogeneous, a tensor the number of sampled nodes, by hop.
            - num_sampled_edges: If heterogeneous, a dictionary mapping from edge type to the number of sampled edges for that type, by hop.
            If homogeneous, a tensor denoting the number of sampled edges, by hop.

        Let's use the following homogeneous graph (https://is.gd/a8DK15) as an example:
            0 -> 1 [label="Positive example" color="green"]
            0 -> 2 [label="Negative example" color="red"]

            0 -> {3, 4}
            3 -> {5, 6}
            4 -> {7, 8}

            1 -> 9 # shouldn't be sampled
            2 -> 10 # shouldn't be sampled

        For sampling around node `0`, the fields on the output Data object will be:
            - `y_positive`: {0: torch.tensor([1])} # 1 is the only positive label for node 0
            - `y_negative`: {0: torch.tensor([2])} # 2 is the only negative label for node 0

        NOTE: both label fields will instead be `dict[EdgeType, dict[int, torch.Tensor]]` if multiple supervision edge types are provided.
        e.g. if there are supervision edge types: (a, to, b) and (a, to, c), then the label fields could be:
            - `y_positive`: {(a, to, b): {0: torch.tensor([1])}, (a, to, c): {0: torch.tensor([2])}}
            - `y_negative`: {(a, to, b): {0: torch.tensor([3])}, (a, to, c): {0: torch.tensor([4])}}

        Args:
            dataset (Union[DistDataset, RemoteDistDataset]): The dataset to sample from.
                If this is a `RemoteDistDataset`, then we are in "Graph Store" mode.
            num_neighbors (Optional[list[int] or dict[tuple[str, str, str], list[int]]]):
                The number of neighbors to sample for each node in each iteration.
                If an entry is set to `-1`, all neighbors will be included.
                In heterogeneous graphs, may also take in a dictionary denoting
                the amount of neighbors to sample for each individual edge type.
                Required — either directly or via ``KHopNeighborSamplerOptions``.
                If both are provided with ``KHopNeighborSamplerOptions``, they must match.
            input_nodes: Indices of seed nodes to start sampling from.
                For Colocated mode: `torch.Tensor` or `tuple[NodeType, torch.Tensor]`.
                    If set to `None` for homogeneous settings, all nodes will be considered.
                    In heterogeneous graphs, this flag must be passed in as a tuple that holds
                    the node type and node indices.
                    NOTE: We intend to migrate colocated mode to have a similar input format to Graph Store mode in the future.
                    We want to do this so that users can easily control labels per anchor.
                For Graph Store mode: `dict[int, ABLPInputNodes]`
                    Maps server_rank to an ABLPInputNodes dataclass containing anchor nodes,
                    positive labels, and negative labels with explicit node type and edge type info.
                    This is the return type of `RemoteDistDataset.fetch_ablp_input()`.
            supervision_edge_type (Optional[Union[EdgeType, list[EdgeType]]]):
                The edge type(s) to use for supervision.
                For Colocated mode: Must be None iff the dataset is labeled homogeneous.
                    If set to a single EdgeType, the positive and negative labels will be stored in the `y_positive` and `y_negative` fields of the Data object.
                    If set to a list of EdgeTypes, the positive and negative labels will be stored in the `y_positive` and `y_negative` fields of the Data object,
                    with the key being the EdgeType. (default: `None`)
                For Graph Store mode: Must not be provided (must be None). The supervision edge types are
                    inferred from the label edge type keys in ABLPInputNodes.
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
            prefetch_size (Optional[int]): Max number of sampled messages to prefetch on the
                client side, per server. Only applies to Graph Store mode (remote workers).
                Lower values reduce server-side RPC thread contention when multiple loaders
                are active concurrently. (default: ``None``).
                If supplied and not it Graph Store mode, an error will be raised.
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
            sampler_options (Optional[SamplerOptions]): Controls which sampler class is
                instantiated. Pass ``KHopNeighborSamplerOptions`` to use the built-in sampler,
                or ``CustomSamplerOptions`` to dynamically import a custom sampler class.
                If ``None``, defaults to ``KHopNeighborSamplerOptions(num_neighbors)``.
            context (deprecated - will be removed soon) (Optional[DistributedContext]): Distributed context information of the current process.
            local_process_rank (deprecated - will be removed soon) (int): The local rank of the current process within a node.
            local_process_world_size (deprecated - will be removed soon) (int): The total number of processes within a node.
        """

        # Set self._shutdowned right away, that way if we throw here, and __del__ is called,
        # then we can properly clean up and don't get extraneous error messages.
        self._shutdowned = True

        num_neighbors, sampler_options = resolve_sampler_options(
            num_neighbors, sampler_options
        )

        # Determine sampling cluster setup based on dataset type
        if isinstance(dataset, RemoteDistDataset):
            self._sampling_cluster_setup = SamplingClusterSetup.GRAPH_STORE
            if supervision_edge_type is not None:
                raise ValueError(
                    "supervision_edge_type must not be provided when using Graph Store mode. "
                    "The supervision edge types are inferred from the ABLPInputNodes label keys in input_nodes."
                )
            # self._supervision_edge_types will be set in _setup_for_graph_store
        else:
            self._sampling_cluster_setup = SamplingClusterSetup.COLOCATED
            if supervision_edge_type is None:
                self._supervision_edge_types: list[EdgeType] = [
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE
                ]
            elif isinstance(supervision_edge_type, list):
                if not supervision_edge_type:
                    raise ValueError(
                        "supervision_edge_type must be a non-empty list when providing multiple supervision edge types."
                    )
                self._supervision_edge_types = supervision_edge_type
            else:
                self._supervision_edge_types = [supervision_edge_type]
            if prefetch_size is not None:
                raise ValueError(
                    f"prefetch_size must be None when using Colocated mode, received {prefetch_size}"
                )
        logger.info(f"Sampling cluster setup: {self._sampling_cluster_setup.value}")

        del supervision_edge_type
        self._instance_count = next(self._counter)

        # Resolve distributed context
        runtime = BaseDistLoader.resolve_runtime(
            context, local_process_rank, local_process_world_size
        )
        del context, local_process_rank, local_process_world_size

        device = (
            pin_memory_device
            if pin_memory_device
            else gigl.distributed.utils.get_available_device(
                local_process_rank=runtime.local_rank
            )
        )
        self.to_device = device

        # Mode-specific setup
        if self._sampling_cluster_setup == SamplingClusterSetup.COLOCATED:
            assert isinstance(
                dataset, DistDataset
            ), "When using colocated mode, dataset must be a DistDataset."
            # Validate input_nodes type for colocated mode
            if isinstance(input_nodes, dict):
                raise ValueError(
                    f"When using Colocated mode, input_nodes must be of type "
                    f"(torch.Tensor | tuple[NodeType, torch.Tensor] | None), "
                    f"received {type(input_nodes)}"
                )
            setup_info = self._setup_for_colocated(
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
            sampler_input: Union[
                ABLPNodeSamplerInput, list[ABLPNodeSamplerInput]
            ] = setup_info[0]
            worker_options: Union[
                MpDistSamplingWorkerOptions, RemoteDistSamplingWorkerOptions
            ] = setup_info[1]
            dataset_schema: DatasetSchema = setup_info[2]
        else:  # Graph Store mode
            assert isinstance(
                dataset, RemoteDistDataset
            ), "When using Graph Store mode, dataset must be a RemoteDistDataset."
            # Validate input_nodes type for Graph Store mode
            if not isinstance(input_nodes, dict):
                raise ValueError(
                    f"When using Graph Store mode, input_nodes must be of type "
                    f"dict[int, ABLPInputNodes], "
                    f"received {type(input_nodes)}"
                )
            if prefetch_size is None:
                logger.info(f"prefetch_size is not provided, using default of 4")
                prefetch_size = 4
            (
                sampler_input,
                worker_options,
                dataset_schema,
            ) = self._setup_for_graph_store(
                input_nodes=input_nodes,
                dataset=dataset,
                num_workers=num_workers,
                worker_concurrency=worker_concurrency,
                prefetch_size=prefetch_size,
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

        # Build the producer: a pre-constructed producer for colocated mode,
        # or an RPC callable for graph store mode.
        if self._sampling_cluster_setup == SamplingClusterSetup.COLOCATED:
            assert isinstance(dataset, DistDataset)
            assert isinstance(worker_options, MpDistSamplingWorkerOptions)
            channel = BaseDistLoader.create_colocated_channel(worker_options)
            producer: Union[
                DistSamplingProducer, Callable[..., int]
            ] = DistSamplingProducer(
                dataset,
                sampler_input,
                sampling_config,
                worker_options,
                channel,
                sampler_options,
            )
        else:
            producer = DistServer.create_sampling_producer

        # Call base class — handles metadata storage and connection initialization
        # (including staggered init for colocated mode).
        super().__init__(
            dataset=dataset,
            sampler_input=sampler_input,
            dataset_schema=dataset_schema,
            worker_options=worker_options,
            sampling_config=sampling_config,
            device=device,
            runtime=runtime,
            producer=producer,
            sampler_options=sampler_options,
            process_start_gap_seconds=process_start_gap_seconds,
        )

    def _setup_for_colocated(
        self,
        input_nodes: Optional[
            Union[
                torch.Tensor,
                tuple[NodeType, torch.Tensor],
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
    ) -> tuple[ABLPNodeSamplerInput, MpDistSamplingWorkerOptions, DatasetSchema]:
        """
        Setup method for colocated (non-Graph Store) mode.

        Args:
            input_nodes: Input nodes for sampling (tensor or tuple of node type and tensor).
            dataset: The DistDataset to sample from.
            local_rank: Local rank of the current process.
            local_world_size: Total number of processes on this machine.
            device: Target device for sampled data.
            master_ip_address: IP address of the master node.
            node_rank: Rank of the current machine.
            node_world_size: Total number of machines.
            num_workers: Number of sampling workers.
            worker_concurrency: Max sampling concurrency per worker.
            channel_size: Size of shared memory channel.
            num_cpu_threads: Number of CPU threads for PyTorch.

        Returns:
            Tuple of (ABLPNodeSamplerInput, MpDistSamplingWorkerOptions, DatasetSchema).
        """
        # Validate input format - should not be Graph Store format
        if isinstance(input_nodes, abc.Mapping):
            raise ValueError(
                f"When using Colocated mode, input_nodes must be of type (torch.Tensor | tuple[NodeType, torch.Tensor]), "
                f"received {type(input_nodes)}"
            )
        elif isinstance(input_nodes, tuple) and isinstance(input_nodes[1], abc.Mapping):
            raise ValueError(
                f"When using Colocated mode, input_nodes must be of type (torch.Tensor | tuple[NodeType, torch.Tensor]), "
                f"received tuple with second element of type {type(input_nodes[1])}"
            )

        if not isinstance(dataset.graph, abc.Mapping):
            raise ValueError(
                f"The dataset must be heterogeneous for ABLP. Received dataset with graph of type: {type(dataset.graph)}"
            )

        is_homogeneous_with_labeled_edge_type: bool = True
        if isinstance(input_nodes, tuple):
            if self._supervision_edge_types == [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                raise ValueError(
                    "When using heterogeneous ABLP, you must provide supervision_edge_types."
                )
            is_homogeneous_with_labeled_edge_type = False
            anchor_node_type, anchor_node_ids = input_nodes
            # TODO (mkolodner-sc): We currently assume supervision edges are directed outward, revisit in future if
            # this assumption is no longer valid and/or is too opinionated
            for supervision_edge_type in self._supervision_edge_types:
                assert (
                    supervision_edge_type[0] == anchor_node_type
                ), f"Label EdgeType are currently expected to be provided in outward edge direction as tuple (`anchor_node_type`,`relation`,`supervision_node_type`), \
                    got supervision edge type {supervision_edge_type} with anchor node type {anchor_node_type}"
            if dataset.edge_dir == "in":
                self._supervision_edge_types = [
                    reverse_edge_type(supervision_edge_type)
                    for supervision_edge_type in self._supervision_edge_types
                ]
        elif isinstance(input_nodes, torch.Tensor):
            if self._supervision_edge_types != [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                raise ValueError(
                    f"Expected supervision edge type to be None for homogeneous input nodes, got {self._supervision_edge_types}"
                )
            anchor_node_ids = input_nodes
            anchor_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        elif input_nodes is None:
            if dataset.node_ids is None:
                raise ValueError(
                    "Dataset must have node ids if input_nodes are not provided."
                )
            if isinstance(dataset.node_ids, abc.Mapping):
                raise ValueError(
                    f"input_nodes must be provided for heterogeneous datasets, received node_ids of type: {dataset.node_ids.keys()}"
                )
            if self._supervision_edge_types != [DEFAULT_HOMOGENEOUS_EDGE_TYPE]:
                raise ValueError(
                    f"Expected supervision edge type to be None for homogeneous input nodes, got {self._supervision_edge_types}"
                )
            anchor_node_ids = dataset.node_ids
            anchor_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        else:
            raise ValueError(f"Unexpected input_nodes type: {type(input_nodes)}")

        missing_edge_types = set(self._supervision_edge_types) - set(
            dataset.graph.keys()
        )
        if missing_edge_types:
            raise ValueError(
                f"Missing edge types in dataset: {missing_edge_types}. Edge types in dataset: {dataset.graph.keys()}"
            )

        # Type narrowing - anchor_node_ids is always a Tensor in colocated mode
        assert isinstance(anchor_node_ids, torch.Tensor)

        if len(anchor_node_ids.shape) != 1:
            raise ValueError(
                f"input_nodes must be a 1D tensor, got {anchor_node_ids.shape}."
            )

        curr_process_nodes = shard_nodes_by_process(
            input_nodes=anchor_node_ids,
            local_process_rank=local_rank,
            local_process_world_size=local_world_size,
        )

        self._positive_label_edge_types: list[EdgeType] = []
        self._negative_label_edge_types: list[EdgeType] = []
        positive_labels_by_label_edge_type: dict[EdgeType, torch.Tensor] = {}
        negative_labels_by_label_edge_type: dict[EdgeType, torch.Tensor] = {}
        for supervision_edge_type in self._supervision_edge_types:
            (
                positive_label_edge_type,
                negative_label_edge_type,
            ) = select_label_edge_types(supervision_edge_type, dataset.graph.keys())
            self._positive_label_edge_types.append(positive_label_edge_type)
            if negative_label_edge_type is not None:
                self._negative_label_edge_types.append(negative_label_edge_type)

            positive_labels, negative_labels = get_labels_for_anchor_nodes(
                dataset=dataset,
                node_ids=curr_process_nodes,
                positive_label_edge_type=positive_label_edge_type,
                negative_label_edge_type=negative_label_edge_type,
            )
            positive_labels_by_label_edge_type[
                positive_label_edge_type
            ] = positive_labels
            if negative_label_edge_type is not None and negative_labels is not None:
                negative_labels_by_label_edge_type[
                    negative_label_edge_type
                ] = negative_labels

        sampler_input = ABLPNodeSamplerInput(
            node=curr_process_nodes,
            input_type=anchor_node_type,
            positive_label_by_edge_types=positive_labels_by_label_edge_type,
            negative_label_by_edge_types=negative_labels_by_label_edge_type,
        )

        # Sets up processes and torch device for initializing the GLT DistNeighborLoader,
        # setting up RPC and worker groups to minimize the memory overhead and CPU contention.
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
            # Lever to explore tuning for CPU based inference
            num_cpu_threads=num_cpu_threads,
        )
        logger.info(
            f"Finished initializing neighbor loader worker: {local_rank}/{local_world_size}"
        )

        # Sets up worker options for the dataloader
        dist_sampling_ports = gigl.distributed.utils.get_free_ports_from_master_node(
            num_ports=local_world_size
        )
        dist_sampling_port_for_current_rank = dist_sampling_ports[local_rank]
        worker_options = MpDistSamplingWorkerOptions(
            num_workers=num_workers,
            # Each worker will spawn several sampling workers, and all sampling workers spawned by workers in one group
            # need to be connected. Thus, we need master ip address and master port to
            # initate the connection.
            # Note that different groups of workers are independent, and thus
            # the sampling processes in different groups should be independent, and should
            # use different master ports.
            worker_devices=[torch.device("cpu") for _ in range(num_workers)],
            worker_concurrency=worker_concurrency,
            master_addr=master_ip_address,
            master_port=dist_sampling_port_for_current_rank,
            # Load testing shows that when num_rpc_threads exceed 16, the performance
            # will degrade.
            num_rpc_threads=min(dataset.num_partitions, 16),
            rpc_timeout=600,
            channel_size=channel_size,
            pin_memory=device.type == "cuda",
        )

        edge_types = list(dataset.graph.keys())

        return (
            sampler_input,
            worker_options,
            DatasetSchema(
                is_homogeneous_with_labeled_edge_type=is_homogeneous_with_labeled_edge_type,
                edge_types=edge_types,
                node_feature_info=dataset.node_feature_info,
                edge_feature_info=dataset.edge_feature_info,
                edge_dir=dataset.edge_dir,
            ),
        )

    def _setup_for_graph_store(
        self,
        input_nodes: dict[int, ABLPInputNodes],
        dataset: RemoteDistDataset,
        num_workers: int,
        worker_concurrency: int = 4,
        prefetch_size: int = 4,
    ) -> tuple[
        list[ABLPNodeSamplerInput], RemoteDistSamplingWorkerOptions, DatasetSchema
    ]:
        """
        Setup method for Graph Store mode.

        Args:
            input_nodes: ABLP input from RemoteDistDataset.fetch_ablp_input().
                Maps server_rank to ABLPInputNodes containing anchor nodes, positive/negative
                labels with explicit node type and edge type information.
            dataset: The RemoteDistDataset to sample from.
            num_workers: Number of sampling workers.
            worker_concurrency: Max sampling concurrency per worker. (default: ``4``).
            prefetch_size: Max prefetched sampled messages per server on client side.
                (default: ``4``).

        Returns:
            Tuple of (list[ABLPNodeSamplerInput], RemoteDistSamplingWorkerOptions, DatasetSchema).
        """
        node_feature_info = dataset.fetch_node_feature_info()
        edge_feature_info = dataset.fetch_edge_feature_info()
        edge_types = dataset.fetch_edge_types()
        node_rank = dataset.cluster_info.compute_node_rank

        # Get sampling ports for compute-storage connections.
        sampling_ports = dataset.fetch_free_ports_on_storage_cluster(
            num_ports=dataset.cluster_info.num_compute_nodes
        )
        sampling_port = sampling_ports[node_rank]
        worker_key = (
            f"compute_ablp_loader_rank_{node_rank}_worker_{self._instance_count}"
        )
        logger.info(f"rank: {torch.distributed.get_rank()}, worker_key: {worker_key}")
        worker_options = RemoteDistSamplingWorkerOptions(
            server_rank=list(range(dataset.cluster_info.num_storage_nodes)),
            num_workers=num_workers,
            worker_devices=[torch.device("cpu") for _ in range(num_workers)],
            worker_concurrency=worker_concurrency,
            master_addr=dataset.cluster_info.storage_cluster_master_ip,
            master_port=sampling_port,
            worker_key=worker_key,
            prefetch_size=prefetch_size,
        )
        logger.info(
            f"Rank {torch.distributed.get_rank()}! init for sampling rpc: "
            f"tcp://{dataset.cluster_info.storage_cluster_master_ip}:{sampling_port}"
        )

        # Validate server ranks
        servers = input_nodes.keys()
        if len(servers) > 0:
            if (
                max(servers) >= dataset.cluster_info.num_storage_nodes
                or min(servers) < 0
            ):
                raise ValueError(
                    f"When using Graph Store mode, the server ranks must be in range "
                    f"[0, {dataset.cluster_info.num_storage_nodes}), "
                    f"received inputs for servers: {list(servers)}"
                )

        # Extract node type and label edge types from the ABLPInputNodes dataclass.
        # All entries should have the same anchor_node_type and edge type keys.
        first_input = next(iter(input_nodes.values()))
        input_type = first_input.anchor_node_type
        is_homogeneous_with_labeled_edge_type = (
            input_type == DEFAULT_HOMOGENEOUS_NODE_TYPE
        )

        # Extract supervision edge types and derive label edge types from the
        # ABLPInputNodes.labels dict (keyed by supervision edge type).
        self._supervision_edge_types = list(first_input.labels.keys())
        has_negatives = any(neg is not None for _, neg in first_input.labels.values())

        self._positive_label_edge_types = [
            message_passing_to_positive_label(et) for et in self._supervision_edge_types
        ]
        self._negative_label_edge_types = (
            [
                message_passing_to_negative_label(et)
                for et in self._supervision_edge_types
            ]
            if has_negatives
            else []
        )

        # Graph Store mode currently only supports a single supervision edge type,
        # so the labels dict must have exactly 1 entry.
        if len(self._supervision_edge_types) != 1:
            raise ValueError(
                f"Graph Store mode currently only supports a single supervision edge type, "
                f"but ABLPInputNodes.labels has {len(self._supervision_edge_types)} "
                f"entries: {self._supervision_edge_types}"
            )

        logger.info(f"Positive label edge types: {self._positive_label_edge_types}")
        logger.info(f"Negative label edge types: {self._negative_label_edge_types}")

        # Convert from ABLPInputNodes to list of ABLPNodeSamplerInput (one per server)
        input_data: list[ABLPNodeSamplerInput] = []
        for server_rank in range(dataset.cluster_info.num_storage_nodes):
            positive_label_by_edge_type: dict[EdgeType, torch.Tensor] = {}
            negative_label_by_edge_type: dict[EdgeType, torch.Tensor] = {}
            if server_rank in input_nodes:
                ablp_input_nodes = input_nodes[server_rank]
                anchors = ablp_input_nodes.anchor_nodes
                for supervision_edge_type, (
                    positive_labels,
                    negative_labels,
                ) in ablp_input_nodes.labels.items():
                    positive_label_by_edge_type[
                        message_passing_to_positive_label(supervision_edge_type)
                    ] = positive_labels
                    if negative_labels is not None:
                        negative_label_by_edge_type[
                            message_passing_to_negative_label(supervision_edge_type)
                        ] = negative_labels
            else:
                # Empty input for servers with no data for this rank
                anchors = torch.empty(0, dtype=torch.long)
                positive_label_by_edge_type = {
                    et: torch.empty(0, 0, dtype=torch.long)
                    for et in self._positive_label_edge_types
                }
                if has_negatives:
                    negative_label_by_edge_type = {
                        et: torch.empty(0, 0, dtype=torch.long)
                        for et in self._negative_label_edge_types
                    }

            logger.info(
                f"Rank: {torch.distributed.get_rank()}! Building ABLPNodeSamplerInput for server rank: {server_rank} "
                f"with input type: {input_type}. anchors: {anchors.shape}, "
                f"positive_labels edge types: {list(positive_label_by_edge_type.keys())}, "
                f"negative_labels edge types: {list(negative_label_by_edge_type.keys())}"
            )
            ablp_input = ABLPNodeSamplerInput(
                node=anchors,
                input_type=input_type,
                positive_label_by_edge_types=positive_label_by_edge_type,
                negative_label_by_edge_types=negative_label_by_edge_type,
            )
            input_data.append(ablp_input)

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

    def _get_labels(
        self, msg: SampleMessage
    ) -> tuple[
        SampleMessage,
        dict[EdgeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
    ]:
        # TODO (mkolodner-sc): Remove the need to modify metadata once GLT's `to_hetero_data` function is fixed
        f"""
        Gets the labels from the output SampleMessage and removes them from the metadata. We need to remove the labels from GLT's metadata since the
        `to_hetero_data` function strangely assumes that we are doing edge-based sampling if the metadata is not empty at the time of
        building the HeteroData object.

        Args:
            msg (SampleMessage): All possible results from a sampler, including subgraph data, features, and used defined metadata
        Returns:
            SampleMessage: Updated sample messsage with the label fields removed
            dict[EdgeType, torch.Tensor]: Dict[positive label edge type, label ID tensor],
                where the ith row  of the tensor corresponds to the ith anchor node ID.
            dict[EdgeType, torch.Tensor]: Dict[negative label edge type, label ID tensor],
                where the ith row  of the tensor corresponds to the ith anchor node ID.
                May be empty if no negative labels are present.
        """
        metadata: dict[str, torch.Tensor] = {}
        positive_labels_by_label_edge_type: dict[EdgeType, torch.Tensor] = {}
        negative_labels_by_label_edge_type: dict[EdgeType, torch.Tensor] = {}
        # We update metadata with sepcial POSITIVE_LABEL_METADATA_KEY and NEGATIVE_LABEL_METADATA_KEY keys
        # in gigl/distributed/dist_neighbor_sampler.py.
        # We need to encode the tuples as strings because GLT requires the keys to be strings.
        # As such, we decode the strings back into tuples,
        # And then pop those keys out of the metadata as they are not needed otherwise.
        # If edge_dir is "in", we need to reverse the edge type because GLT swaps src/dst for edge_dir = "out".
        # NOTE: GLT *prepends* the keys with "#META."
        positive_label_metadata_key_prefix = metadata_key_with_prefix(
            POSITIVE_LABEL_METADATA_KEY
        )
        negative_label_metadata_key_prefix = metadata_key_with_prefix(
            NEGATIVE_LABEL_METADATA_KEY
        )
        for k in list(msg.keys()):
            if k.startswith(positive_label_metadata_key_prefix):
                edge_type_str = k[len(positive_label_metadata_key_prefix) :]
                edge_type = ast.literal_eval(edge_type_str)
                if self.edge_dir == "in":
                    edge_type = reverse_edge_type(edge_type)
                positive_labels_by_label_edge_type[edge_type] = msg[k].to(
                    self.to_device
                )
                del msg[k]
            elif k.startswith(negative_label_metadata_key_prefix):
                edge_type_str = k[len(negative_label_metadata_key_prefix) :]
                edge_type = ast.literal_eval(edge_type_str)
                if self.edge_dir == "in":
                    edge_type = reverse_edge_type(edge_type)
                negative_labels_by_label_edge_type[edge_type] = msg[k].to(
                    self.to_device
                )
                del msg[k]
            elif k.startswith("#META."):
                meta_key = str(k[len("#META.") :])
                metadata[meta_key] = msg[k].to(self.to_device)
                del msg[k]
        return (
            msg,
            positive_labels_by_label_edge_type,
            negative_labels_by_label_edge_type,
        )

    def _set_labels(
        self,
        data: Union[Data, HeteroData],
        positive_labels_by_label_edge_type: dict[EdgeType, torch.Tensor],
        negative_labels_by_label_edge_type: dict[EdgeType, torch.Tensor],
    ) -> Union[Data, HeteroData]:
        """
        Sets the labels and relevant fields in the torch_geometric Data object, converting the global node ids for labels to their
        local index. Removes inserted supervision edge type from the data variables, since this is an implementation detail and should not be
        exposed in the final HeteroData/Data object.
        Args:
            data (Union[Data, HeteroData]): Graph to provide labels for
            positive_labels_by_label_edge_type (dict[EdgeType, torch.Tensor]): Dict[positive label edge type, label ID tensor],
                where the ith row  of the tensor corresponds to the ith anchor node ID.
            negative_labels_by_label_edge_type (dict[EdgeType, torch.Tensor]): Dict[negative label edge type, label ID tensor],
                where the ith row  of the tensor corresponds to the ith anchor node ID.
        Returns:
            Union[Data, HeteroData]: torch_geometric HeteroData/Data object with the filtered edge fields and labels set as properties of the instance
        """
        # shape [N], where N is the number of nodes in the subgraph, and local_node_to_global_node[i] gives the global node id for local node id `i`
        node_type_to_local_node_to_global_node: dict[NodeType, torch.Tensor] = {}
        if isinstance(data, HeteroData):
            for e_type in self._supervision_edge_types:
                node_type_to_local_node_to_global_node[e_type[0]] = data[e_type[0]].node
                node_type_to_local_node_to_global_node[e_type[2]] = data[e_type[2]].node
        else:
            node_type_to_local_node_to_global_node[
                DEFAULT_HOMOGENEOUS_NODE_TYPE
            ] = data.node
        output_positive_labels: dict[EdgeType, dict[int, torch.Tensor]] = defaultdict(
            dict
        )
        output_negative_labels: dict[EdgeType, dict[int, torch.Tensor]] = defaultdict(
            dict
        )
        # We always have supervision edge types of the form (anchor_node_type, to, supervision_node_type)
        # So we can index into the edge type accordingly.
        edge_index = 2
        for edge_type, label_tensor in positive_labels_by_label_edge_type.items():
            for local_anchor_node_id in range(label_tensor.size(0)):
                positive_mask = (
                    node_type_to_local_node_to_global_node[
                        edge_type[edge_index]
                    ].unsqueeze(1)
                    == label_tensor[local_anchor_node_id]
                )  # shape [N, P], where N is the number of nodes and P is the number of positive labels for the current anchor node

                # Gets the indexes of the items in local_node_to_global_node which match any of the positive labels for the current anchor node
                output_positive_labels[
                    label_edge_type_to_message_passing_edge_type(edge_type)
                ][local_anchor_node_id] = torch.nonzero(positive_mask)[:, 0].to(
                    self.to_device
                )
                # Shape [X], where X is the number of indexes in the original local_node_to_global_node which match a node in the positive labels for the current anchor node

        for edge_type, label_tensor in negative_labels_by_label_edge_type.items():
            for local_anchor_node_id in range(label_tensor.size(0)):
                negative_mask = (
                    node_type_to_local_node_to_global_node[
                        edge_type[edge_index]
                    ].unsqueeze(1)
                    == label_tensor[local_anchor_node_id]
                )  # shape [N, M], where N is the number of nodes and M is the number of negative labels for the current anchor node

                # Gets the indexes of the items in local_node_to_global_node which match any of the negative labels for the current anchor node
                output_negative_labels[
                    label_edge_type_to_message_passing_edge_type(edge_type)
                ][local_anchor_node_id] = torch.nonzero(negative_mask)[:, 0].to(
                    self.to_device
                )
                # Shape [X], where X is the number of indexes in the original local_node_to_global_node which match a node in the negative labels for the current anchor node
        if not output_positive_labels:
            raise ValueError("No positive labels were found in the data!")
        elif len(output_positive_labels) == 1:
            data.y_positive = next(iter(output_positive_labels.values()))
        else:
            data.y_positive = output_positive_labels

        if len(output_negative_labels) == 1:
            data.y_negative = next(iter(output_negative_labels.values()))
        elif len(output_negative_labels) > 0:
            data.y_negative = output_negative_labels
        return data

    def _collate_fn(self, msg: SampleMessage) -> Union[Data, HeteroData]:
        msg, positive_labels, negative_labels = self._get_labels(msg)
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
            if len(self._supervision_edge_types) != 1:
                raise ValueError(
                    f"Expected 1 supervision edge type, got {len(self._supervision_edge_types)}"
                )
            data = labeled_to_homogeneous(self._supervision_edge_types[0], data)
        data = self._set_labels(data, positive_labels, negative_labels)
        return data
