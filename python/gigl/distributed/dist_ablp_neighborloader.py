from collections import abc
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from graphlearn_torch.channel import ShmChannel
from graphlearn_torch.distributed import MpDistSamplingWorkerOptions, get_context
from graphlearn_torch.sampler import SamplingConfig, SamplingType
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed.constants import (
    DEFAULT_MASTER_INFERENCE_PORT,
    DEFAULT_MASTER_SAMPLING_PORT,
)
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.dist_sampling_producer import DistSamplingProducer
from gigl.distributed.distributed_neighborloader import (
    DEFAULT_NUM_CPU_THREADS,
    DistNeighborLoader,
    shard_nodes_by_process,
)
from gigl.src.common.types.graph_data import (
    NodeType,  # TODO (mkolodner-sc): Change to use torch_geometric.typing
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    is_label_edge_type,
    reverse_edge_type,
    select_label_edge_types,
    to_heterogeneous_edge,
)
from gigl.types.sampler import LabeledNodeSamplerInput
from gigl.utils.data_splitters import PADDING_NODE, get_labels_for_anchor_nodes

logger = Logger()


class DistABLPLoader(DistNeighborLoader):
    def __init__(
        self,
        dataset: DistLinkPredictionDataset,
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        context: DistributedContext,
        local_process_rank: int,  # TODO: Move this to DistributedContext
        local_process_world_size: int,  # TODO: Move this to DistributedContext
        input_nodes: Optional[
            Union[
                torch.Tensor,
                Tuple[NodeType, torch.Tensor],
            ]
        ] = None,
        # TODO(kmonte): Support multiple supervision edge types.
        supervision_edge_type: Optional[EdgeType] = None,
        num_workers: int = 1,
        batch_size: int = 1,
        pin_memory_device: Optional[torch.device] = None,
        worker_concurrency: int = 4,
        channel_size: str = "4GB",
        process_start_gap_seconds: float = 60.0,
        num_cpu_threads: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        _main_inference_port: int = DEFAULT_MASTER_INFERENCE_PORT,
        _main_sampling_port: int = DEFAULT_MASTER_SAMPLING_PORT,
    ):
        """
        Neighbor loader for Anchor Based Link Prediction (ABLP) tasks.

        Note that for this class, the dataset must *always* be heterogeneous,
        as we need separate edge types for positive and negative labels.

        If you provide `input_nodes` for homogeneous input (only as a Tensor),
        Then we will attempt to infer the positive and optional negative labels
        from the dataset.
        In this case, the output of the loader will be a torch_geometric.data.Data object.
        Otherwise, the output will be a torch_geometric.data.HeteroData object.

           Args:
            dataset (DistLinkPredictionDataset): The dataset to sample from.
            num_neighbors (List[int] or Dict[Tuple[str, str, str], List[int]]):
                The number of neighbors to sample for each node in each iteration.
                If an entry is set to `-1`, all neighbors will be included.
                In heterogeneous graphs, may also take in a dictionary denoting
                the amount of neighbors to sample for each individual edge type.
            context (DistributedContext): Distributed context information of the current process.
            local_process_rank (int): The local rank of the current process within a node.
            local_process_world_size (int): The total number of processes within a node.
            input_nodes (torch.Tensor or Tuple[str, torch.Tensor]): The
                indices of seed nodes to start sampling from.
                It is of type `torch.LongTensor` for homogeneous graphs.
                If set to `None` for homogeneous settings, all nodes will be considered.
                In heterogeneous graphs, this flag must be passed in as a tuple that holds
                the node type and node indices. (default: `None`)
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

        if not isinstance(dataset.graph, abc.Mapping):
            raise ValueError(
                f"The dataset must be heterogeneous for ABLP. Recieved dataset with graph of type: {type(dataset.graph)}"
            )
        # TODO(kmonte): Remove these checks and support properly heterogeneous NABLP.
        is_heterogeneous: bool = False
        if isinstance(input_nodes, tuple):
            if supervision_edge_type is None:
                raise ValueError(
                    "When using heterogeneous ABLP, you must provide supervision_edge_types."
                )
            is_heterogeneous = True
            node_type, node_ids = input_nodes
            assert (
                supervision_edge_type[0] == node_type
            ), f"Label EdgeType are currently expected to be provided as `anchor_node_type`_`relation`_`supervision_node_type`, \
                got supervision edge type {supervision_edge_type} with anchor node type {node_type}"
            supervision_node_type = supervision_edge_type[2]
            if dataset.edge_dir == "in":
                supervision_edge_type = reverse_edge_type(supervision_edge_type)

        elif isinstance(input_nodes, torch.Tensor):
            node_ids = input_nodes
            node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
            supervision_edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE
            supervision_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        elif input_nodes is None:
            if dataset.node_ids is None:
                raise ValueError(
                    "Dataset must have node ids if input_nodes are not provided."
                )
            if isinstance(dataset.node_ids, abc.Mapping):
                raise ValueError(
                    f"input_nodes must be provided for heterogeneous datasets, received node_ids of type: {dataset.node_ids.keys()}"
                )
            node_ids = dataset.node_ids
            node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
            supervision_edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE
            supervision_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE

        missing_edge_types = set([supervision_edge_type]) - set(dataset.graph.keys())
        if missing_edge_types:
            raise ValueError(
                f"Missing edge types in dataset: {missing_edge_types}. Edge types in dataset: {dataset.graph.keys()}"
            )

        if len(node_ids.shape) != 1:
            raise ValueError(f"input_nodes must be a 1D tensor, got {node_ids.shape}.")
        positive_label_edge_type, negative_label_edge_type = select_label_edge_types(
            supervision_edge_type, dataset.graph.keys()
        )

        positive_labels, negative_labels = get_labels_for_anchor_nodes(
            dataset=dataset,
            node_ids=node_ids,
            positive_label_edge_type=positive_label_edge_type,
            negative_label_edge_type=negative_label_edge_type,
        )

        transforms: list[
            Callable[[Union[Data, HeteroData]], Union[Data, HeteroData]]
        ] = []

        if not is_heterogeneous:
            transforms.append(_SupervisedToHomogeneous(supervision_edge_type))
        transforms.append(
            _SetLabels(
                input_node_type=node_type,
                positive_label_edge_type=positive_label_edge_type,
                negative_label_edge_type=negative_label_edge_type,
            )
        )
        self._transforms = transforms

        # TODO(kmonte): stop setting fanout for positive/negative once GLT sampling is fixed.
        if isinstance(num_neighbors, dict):
            num_hop = len(list(num_neighbors.values())[0])
        else:
            num_hop = len(num_neighbors)
        zero_samples = [0] * num_hop
        num_neighbors = to_heterogeneous_edge(num_neighbors)
        for edge_type in dataset.graph.keys():
            if is_label_edge_type(edge_type):
                num_neighbors[edge_type] = zero_samples
            elif edge_type not in num_neighbors:
                num_neighbors[edge_type] = zero_samples
        logger.info(f"Overwrote num_neighbors to: {num_neighbors}.")

        if num_neighbors.keys() != dataset.graph.keys():
            raise ValueError(
                f"num_neighbors must have all edge types in the graph, received: {num_neighbors.keys()} with for graph with edge types {dataset.graph.keys()}"
            )
        hops = len(next(iter(num_neighbors.values())))
        if not all(len(fanout) == hops for fanout in num_neighbors.values()):
            raise ValueError(
                f"num_neighbors must be a dict of edge types with the same number of hops. Received: {num_neighbors}"
            )

        curr_process_nodes = shard_nodes_by_process(
            input_nodes=node_ids,
            local_process_rank=local_process_rank,
            local_process_world_size=local_process_world_size,
        )

        device = (
            pin_memory_device
            if pin_memory_device
            else gigl.distributed.utils.get_available_device(
                local_process_rank=local_process_rank
            )
        )
        # Sets up processes and torch device for initializing the GLT DistNeighborLoader, setting up RPC and worker groups to minimize
        # the memory overhead and CPU contention.
        logger.info(
            f"Initializing neighbor loader worker in process: {local_process_rank}/{local_process_world_size} using device: {device}"
        )
        should_use_cpu_workers = device.type == "cpu"
        if should_use_cpu_workers and num_cpu_threads is None:
            logger.info(
                "Using CPU workers, but found num_cpu_threads to be None. "
                f"Will default setting num_cpu_threads to {DEFAULT_NUM_CPU_THREADS}."
            )
            num_cpu_threads = DEFAULT_NUM_CPU_THREADS
        gigl.distributed.utils.init_neighbor_loader_worker(
            master_ip_address=context.main_worker_ip_address,
            local_process_rank=local_process_rank,
            local_process_world_size=local_process_world_size,
            rank=context.global_rank,
            world_size=context.global_world_size,
            master_worker_port=_main_inference_port,
            device=device,
            should_use_cpu_workers=should_use_cpu_workers,
            # Lever to explore tuning for CPU based inference
            num_cpu_threads=num_cpu_threads,
            process_start_gap_seconds=process_start_gap_seconds,
        )
        logger.info(
            f"Finished initializing neighbor loader worker:  {local_process_rank}/{local_process_world_size}"
        )

        # Sets up worker options for the dataloader
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
            master_addr=context.main_worker_ip_address,
            master_port=_main_sampling_port + local_process_rank,
            # Load testing show that when num_rpc_threads exceed 16, the performance
            # will degrade.
            num_rpc_threads=min(dataset.num_partitions, 16),
            rpc_timeout=600,
            channel_size=channel_size,
            pin_memory=device.type == "cuda",
        )

        sampler_input = LabeledNodeSamplerInput(
            node=curr_process_nodes,
            input_type=node_type,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            supervision_node_type=supervision_node_type,
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
            edge_dir=dataset.edge_dir,
            seed=None,  # it's actually optional - None means random.
        )

        self.data = dataset
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

        # We can set shutdowned to false now
        self._shutdowned = False

        self._is_mp_worker = True
        self._is_collocated_worker = False
        self._is_remote_worker = False

        if self.data is not None:
            self.num_data_partitions = self.data.num_partitions
            self.data_partition_idx = self.data.partition_idx
            self._set_ntypes_and_etypes(
                self.data.get_node_types(), self.data.get_edge_types()
            )

        self._num_recv = 0
        self._epoch = 0

        current_ctx = get_context()

        self._input_len = len(self.input_data)
        self._input_type = self.input_data.input_type
        self._num_expected = self._input_len // self.batch_size
        if not self.drop_last and self._input_len % self.batch_size != 0:
            self._num_expected += 1

        if not current_ctx.is_worker():
            raise RuntimeError(
                f"'{self.__class__.__name__}': only supports "
                f"launching multiprocessing sampling workers with "
                f"a non-server distribution mode, current role of "
                f"distributed context is {current_ctx.role}."
            )
        if self.data is None:
            raise ValueError(
                f"'{self.__class__.__name__}': missing input dataset "
                f"when launching multiprocessing sampling workers."
            )

        # Launch multiprocessing sampling workers
        self._with_channel = True
        self.worker_options._set_worker_ranks(current_ctx)

        self._channel = ShmChannel(
            self.worker_options.channel_capacity, self.worker_options.channel_size
        )
        if self.worker_options.pin_memory:
            self._channel.pin_memory()

        self._mp_producer = DistSamplingProducer(
            self.data,
            self.input_data,
            self.sampling_config,
            self.worker_options,
            self._channel,
        )
        self._mp_producer.init()


class _SupervisedToHomogeneous:
    """Transform class to convert a heterogeneous graph to a homogeneous graph."""

    def __init__(
        self,
        message_passing_edge_type: EdgeType,
    ):
        """
        Args:
            message_passing_edge_type (EdgeType): The edge type to use for message passing.
        """

        self._message_passing_edge_type = message_passing_edge_type

    def __call__(self, data: HeteroData) -> Data:
        """Transform the heterogeneous graph to a homogeneous graph."""
        homogeneous_data = data.edge_type_subgraph(
            [self._message_passing_edge_type]
        ).to_homogeneous(add_edge_type=False, add_node_type=False)

        return homogeneous_data


class _SetLabels:
    """Transform class to set labels for the nodes in the graph."""

    def __init__(
        self,
        input_node_type,
        positive_label_edge_type,
        negative_label_edge_type,
    ):
        self._input_node_type = input_node_type
        self._positive_label_edge_type = positive_label_edge_type
        self._negative_label_edge_type = negative_label_edge_type

    def __call__(self, data: Data) -> Data:
        is_heterogeneous = isinstance(data, HeteroData)
        positive_labels: dict[int, torch.Tensor] = {}
        has_negative_label = hasattr(data, "negative_labels")
        if is_heterogeneous:
            anchor_nodes = data.batch_dict[self._input_node_type]
        else:
            anchor_nodes = data.batch
        if has_negative_label:
            assert self._negative_label_edge_type is not None
            negative_labels: dict[int, torch.Tensor] = {}

        for i in range(data.positive_labels.shape[0]):
            positive_example = data.positive_labels[i]
            positive_labels[anchor_nodes[i].item()] = positive_example[
                positive_example != PADDING_NODE
            ]
            if has_negative_label:
                negative_example = data.negative_labels[i]
                negative_labels[anchor_nodes[i].item()] = negative_example[
                    negative_example != PADDING_NODE
                ]

        data.y_positive = positive_labels
        del data.positive_labels
        if has_negative_label:
            data.y_negative = negative_labels
            del data.negative_labels
        if is_heterogeneous:
            del data.num_sampled_edges[self._positive_label_edge_type]
            del data._edge_store_dict[self._positive_label_edge_type]
            if has_negative_label:
                del data.num_sampled_edges[self._negative_label_edge_type]
                del data._edge_store_dict[self._negative_label_edge_type]

        return data
