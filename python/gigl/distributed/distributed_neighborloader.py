import itertools
from collections import abc
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import DistLoader, MpDistSamplingWorkerOptions
from graphlearn_torch.sampler import NodeSamplerInput, SamplingConfig, SamplingType
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

import gigl.distributed.utils
from gigl.common.logger import Logger
from gigl.distributed import DistributedContext
from gigl.distributed.constants import (
    DEFAULT_MASTER_INFERENCE_PORT,
    DEFAULT_MASTER_SAMPLING_PORT,
)
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.src.common.types.graph_data import (
    NodeType,  # TODO (mkolodner-sc): Change to use torch_geometric.typing
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    select_label_edge_types,
    to_heterogeneous_edge,
)
from gigl.utils.data_splitters import PADDING_NODE, get_labels_for_anchor_nodes

logger = Logger()

# When using CPU based inference/training, we default cpu threads for neighborloading on top of the per process parallelism.
DEFAULT_NUM_CPU_THREADS = 2


class DistNeighborLoader(DistLoader):
    _transforms: Sequence[Callable[[Union[Data, HeteroData]], Union[Data, HeteroData]]]

    def __init__(
        self,
        dataset: DistLinkPredictionDataset,
        num_neighbors: Union[List[int], Dict[EdgeType, List[int]]],
        context: DistributedContext,
        local_process_rank: int,  # TODO: Move this to DistributedContext
        local_process_world_size: int,  # TODO: Move this to DistributedContext
        input_nodes: Optional[
            Union[torch.Tensor, Tuple[NodeType, torch.Tensor]]
        ] = None,
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
        Note: We try to adhere to pyg dataloader api as much as possible.
        See the following for reference:
        https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/loader/node_loader.html#NodeLoader
        https://pytorch-geometric.readthedocs.io/en/2.5.2/_modules/torch_geometric/distributed/dist_neighbor_loader.html#DistNeighborLoader

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
            _main_inference_port (int): WARNING: You don't need to configure this unless port conflict issues. Slotted for refactor.
                The port number to use for inference processes.
                In future, the port will be automatically assigned based on availability.
                Currently defaults to: gigl.distributed.constants.DEFAULT_MASTER_INFERENCE_PORT
            _main_sampling_port (int): WARNING: You don't need to configure this unless port conflict issues. Slotted for refactor.
                The port number to use for sampling processes.
                In future, the port will be automatically assigned based on availability.
                Currently defaults to: gigl.distributed.constants.DEFAULT_MASTER_SAMPLING_PORT
        """

        # Set self._shutdowned right away, that way if we throw here, and __del__ is called,
        # then we can properly clean up and don't get extraneous error messages.
        # We set to `True` as we don't need to cleanup right away, and this will get set
        # to `False` in super().__init__()` e.g.
        # https://github.com/alibaba/graphlearn-for-pytorch/blob/26fe3d4e050b081bc51a79dc9547f244f5d314da/graphlearn_torch/python/distributed/dist_loader.py#L125C1-L126C1
        self._shutdowned = True

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

        if isinstance(num_neighbors, abc.Mapping):
            # TODO(kmonte): We should enable this. We have two blockers:
            # 1. We need to treat `EdgeType` as a proper tuple, not the GiGL`EdgeType`.
            # 2. There are (likely) some GLT bugs around https://github.com/alibaba/graphlearn-for-pytorch/blob/26fe3d4e050b081bc51a79dc9547f244f5d314da/graphlearn_torch/python/distributed/dist_neighbor_sampler.py#L317-L318
            # Where if num_neighbors is a dict then we index into it improperly.
            if not isinstance(dataset.graph, abc.Mapping):
                raise ValueError(
                    "When num_neighbors is a dict, the dataset must be heterogeneous."
                )
            if num_neighbors.keys() != dataset.graph.keys():
                raise ValueError(
                    f"num_neighbors must have all edge types in the graph, received: {num_neighbors.keys()} with for graph with edge types {dataset.graph.keys()}"
                )
            hops = len(next(iter(num_neighbors.values())))
            if not all(len(fanout) == hops for fanout in num_neighbors.values()):
                raise ValueError(
                    f"num_neighbors must be a dict of edge types with the same number of hops. Received: {num_neighbors}"
                )

        curr_process_nodes = _shard_nodes_by_process(
            input_nodes=input_nodes,
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

        # May be set by base classes.
        if not hasattr(self, "_transforms"):
            self._transforms = []

        # Determines if the node ids passed in are heterogeneous or homogeneous.
        if isinstance(curr_process_nodes, torch.Tensor):
            node_ids = curr_process_nodes
            node_type = None
        else:
            node_type, node_ids = curr_process_nodes

        input_data = self._get_node_sampler_input(node_ids, node_type)

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
        super().__init__(dataset, input_data, sampling_config, device, worker_options)

    def _get_node_sampler_input(
        self, node: torch.Tensor, input_type: Optional[Union[NodeType, str]]
    ) -> NodeSamplerInput:
        if len(node.shape) != 1:
            raise ValueError(f"Input nodes must be a 1D tensor, got {node.shape}.")
        return NodeSamplerInput(node, input_type)

    def _collate_fn(self, msg: SampleMessage) -> Union[Data, HeteroData]:
        data = super()._collate_fn(msg)
        for transform in self._transforms:
            data = transform(data)
        return data


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
        # TODO(kmonte): Remove these checks and support properly heterogeneous NABLP.
        if isinstance(input_nodes, tuple):
            raise ValueError(
                "Heterogeneous ABLP is not supported yet. Provide node_ids as a tensor."
            )
        if isinstance(num_neighbors, abc.Mapping):
            raise ValueError(
                "Heterogeneous ABLP is not supported yet. Provide num_neighbors as a list of integers."
            )
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

        if not isinstance(dataset.graph, abc.Mapping):
            raise ValueError(
                f"The dataset must be heterogeneous for ABLP. Recieved dataset with graph of type: {type(dataset.graph)}"
            )

        if DEFAULT_HOMOGENEOUS_EDGE_TYPE not in dataset.graph:
            raise ValueError(
                f"With Homogeneous ABLP, the graph must have {DEFAULT_HOMOGENEOUS_EDGE_TYPE} edge type, received {dataset.graph.keys()}."
            )

        if isinstance(input_nodes, abc.Mapping):
            input_nodes = input_nodes[DEFAULT_HOMOGENEOUS_NODE_TYPE]

        if len(input_nodes.shape) != 1:
            raise ValueError(
                f"input_nodes must be a 1D tensor, got {input_nodes.shape}."
            )
        positive_label_edge_type, negative_label_edge_type = select_label_edge_types(
            DEFAULT_HOMOGENEOUS_EDGE_TYPE, dataset.graph.keys()
        )
        # We want to setup "input_nodes" as an approppriately shaped tensor for sampling.
        # Our goal here is to:
        # * Sample against anchor nodes and their labels together
        # * Later dissambiguate what anchors have which labels.
        # Let's say we have the following graph:
        # Message passing edges:
        # A -> B -> C
        # Positive label edges:
        # A -> D
        # B -> F
        # Negative label edges:
        # A -> E
        # B -> G
        # Then we want to sample `{A, D, E}, {B, F, G}` together.
        # Therefor we want to create `input_nodes` as:
        # `[[A, -2, D, -3, E, -4], [B, -2, F, -3, G, -4]]`
        # We use -2, -3, -4 as sentinels to distinguish between the anchor node, positive labels, and negative labels.
        # The sentinels will later be stripped out by _LabeledNodeSamplerInput.
        positive_labels, negative_labels = get_labels_for_anchor_nodes(
            dataset, input_nodes, positive_label_edge_type, negative_label_edge_type
        )  # (num_nodes X num_positive_labels), (num_nodes X num_negative_labels)
        # TODO (kmonte): Potentially - we can avoid using the sentinel values with either nested/jagged tensors
        # or upstreaming changes to GLT to allow us to disasmbiguate between anchors and labels.
        extracted_input_nodes = input_nodes.unsqueeze(1)  # (num_nodes X 1)
        if negative_labels is not None:
            anchor_sentinel = -2
            positive_sentinel = -3
            negative_sentinel = -4
            input_nodes = torch.cat(
                [
                    extracted_input_nodes,
                    torch.full_like(extracted_input_nodes, anchor_sentinel),
                    positive_labels,
                    torch.full_like(extracted_input_nodes, positive_sentinel),
                    negative_labels,
                    torch.full_like(extracted_input_nodes, negative_sentinel),
                ],
                dim=1,
            )  # (num_nodes X (num_positive_labels + num_negative_labels + 4))
        else:
            anchor_sentinel = -2
            positive_sentinel = -3
            negative_sentinel = None
            input_nodes = torch.cat(
                [
                    extracted_input_nodes,
                    torch.full_like(extracted_input_nodes, anchor_sentinel),
                    positive_labels,
                    torch.full_like(extracted_input_nodes, positive_sentinel),
                ],
                dim=1,
            )  # (num_nodes X (num_positive_labels + 3))

        # _batch_store is a multi-process shared dict that stores the sampled nodes for each process.
        # It is a mapping of (node_type, sampled_nodes) to the "full batch".
        # Since the full batch may contain invalid node ids (sentinels and padding),
        # we need to strip out invalid node ids (< 0), in `_LabeledNodeSamplerInput`
        # before we can sample.
        # But since we do care about the sentinels and padding, we need to
        # keep the "full batch" as the value of the dict.
        # We use a multiprocessing dict as the sampling and transforms happen in different
        # processes.
        # TODO (kmonte): We can avoid _batch_store if we have GLT output the "full batch".
        node_batches_by_sampled_nodes: abc.MutableMapping[
            tuple[NodeType, tuple[int, ...]], torch.Tensor
        ] = torch.multiprocessing.Manager().dict()
        self._batch_store = node_batches_by_sampled_nodes

        input_nodes = (DEFAULT_HOMOGENEOUS_NODE_TYPE, input_nodes)
        logger.info(
            f"Converted input nodes to tuple of ({DEFAULT_HOMOGENEOUS_NODE_TYPE}, {input_nodes[1].shape})."
        )
        transforms: Sequence[
            Callable[[Union[Data, HeteroData]], Union[Data, HeteroData]]
        ] = [
            _SupervisedToHomogeneous(
                message_passing_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
            ),
            _SetLabels(
                batch_store=node_batches_by_sampled_nodes,
                anchor_node_sentinel=anchor_sentinel,
                positive_label_sentinel=positive_sentinel,
                negative_label_sentinel=negative_sentinel,
            ),
        ]
        self._transforms = transforms
        # TODO(kmonte): stop setting fanout for positive/negative once GLT sampling is fixed.
        zero_samples = [0] * len(num_neighbors)
        num_neighbors = to_heterogeneous_edge(num_neighbors)
        num_neighbors[positive_label_edge_type] = zero_samples
        if negative_label_edge_type is not None:
            num_neighbors[negative_label_edge_type] = zero_samples
        logger.info(f"Overwrote num_neighbors to: {num_neighbors}.")

        super().__init__(
            dataset=dataset,
            num_neighbors=num_neighbors,
            context=context,
            local_process_rank=local_process_rank,
            local_process_world_size=local_process_world_size,
            input_nodes=input_nodes,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory_device=pin_memory_device,
            worker_concurrency=worker_concurrency,
            channel_size=channel_size,
            process_start_gap_seconds=process_start_gap_seconds,
            num_cpu_threads=num_cpu_threads,
            shuffle=shuffle,
            drop_last=drop_last,
            _main_inference_port=_main_inference_port,
            _main_sampling_port=_main_sampling_port,
        )

    def _get_node_sampler_input(
        self, node: torch.Tensor, input_type: Optional[Union[NodeType, str]]
    ) -> NodeSamplerInput:
        return _LabeledNodeSamplerInput(node, input_type, self._batch_store)


def _shard_nodes_by_process(
    input_nodes: Union[torch.Tensor, Tuple[str, torch.Tensor]],
    local_process_rank: int,
    local_process_world_size: int,
) -> Union[torch.Tensor, Tuple[str, torch.Tensor]]:
    def shard(nodes: torch.Tensor) -> torch.Tensor:
        num_node_ids_per_process = nodes.size(0) // local_process_world_size
        start_index = local_process_rank * num_node_ids_per_process
        end_index = (
            nodes.size(0)
            if local_process_rank == local_process_world_size - 1
            else start_index + num_node_ids_per_process
        )
        nodes_for_current_process = nodes[start_index:end_index]
        return nodes_for_current_process

    if isinstance(input_nodes, torch.Tensor):
        return shard(input_nodes)
    else:
        node_type, node_ids = input_nodes
        node_ids = shard(node_ids)
        return (node_type, node_ids)


_GraphData = TypeVar("_GraphData", Data, HeteroData)


class _LabeledNodeSamplerInput(NodeSamplerInput):
    """
    Allows for guaranteeing that all "label sets" are sampled together.

    A "label set" is an anchor node, and it's positive and negative labels.
    For instance if we have the graph:
        # Message passing
        A -> B -> C
        # Positive label
        A -> D
        # Negative label
        A -> E

    Then the label set for A is (A, D, E).

    If this is being used for labeled input, then the `node` input should be a tensor of shape N x M
    where N is the number of label sets, and M is the number of nodes in each label set.

    This class may also be given a tensor of shape N x 1, in which case there is no guarantee
    a node and it's labels are sampled together.
    """

    def __init__(
        self,
        node: torch.Tensor,
        input_type: Optional[Union[str, NodeType]],
        batch_store: Optional[
            abc.MutableMapping[tuple[NodeType, tuple[int, ...]], torch.Tensor]
        ] = None,
    ):
        """
        Args:
            node (torch.Tensor): The node ids to sample.
            input_type (Union[str, NodeType]): The type of the node.
            batch_store (Optional[abc.MutableMapping[tuple[NodeType, tuple[int, ...]]]]): The batch store to use for the sampled nodes. Should be a multiprocessing dict.
        """
        super().__init__(node, input_type)
        self._batch_store = batch_store

    def __len__(self) -> int:
        return self.node.shape[0]

    def __getitem__(
        self, index: Union[torch.Tensor, Any]
    ) -> "_LabeledNodeSamplerInput":
        if not isinstance(index, torch.Tensor):
            index = torch.tensor(index, dtype=torch.long)
        index = index.to(self.node.device)
        full_batch = self.node[index].view(-1)
        nodes_to_sample = full_batch[full_batch >= 0]
        if self._batch_store is not None:
            # Dedup the nodes to sample.
            # We use a `dict` here to dedup the nodes, as `set` does not preserve
            # insertion order, which need to keep we can differentiate between:
            # anchor: A, positive: B, negative: C
            # and
            # anchor: C, positive: B, negative: A
            # We use a tuple so it can be a hash key.
            nodes = tuple(
                dict(zip(nodes_to_sample.tolist(), itertools.cycle([None]))).keys()
            )
            self._batch_store[self.input_type, nodes] = full_batch
        return _LabeledNodeSamplerInput(nodes_to_sample, self.input_type)


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
        batch_store: abc.MutableMapping[tuple[NodeType, tuple[int, ...]], torch.Tensor],
        anchor_node_sentinel: int,
        positive_label_sentinel: int,
        negative_label_sentinel: Optional[int] = None,
    ):
        """
        Sets the labels for nodes in Data.y_positive and Data.y_negative.
        The labels are set based on the anchor node and positive label sentinels.
        We assume that the input batch is a 1D tensor of node ids, or fornat:
            `[<anchor node>, <anchor node sentinel>, <positive labels>, <positive label sentinel>, <negative label>, <negative label sentinel>]`
        Will also set anchor node ids.


        The returned data will have the following additional attributes:
            * `y_positive`: A mapping from anchor node id -> positive label node ids.
            * (Optionally)`y_negative`: A mapping from anchor node id -> negative label node ids.

        We expect that the batch store contains:
            * `(node_type, node_ids)` as the key
            * where `node_ids` is a 1D tensor of node ids
            * where the sentinel values have been removed,
            * the node ids have been de-duped
            * and the node ids are in the same order as the input batch.

        For example, if the input batch is:
            `[0, -1, 1, -2, 3, 1 -3]`
            for anchor node 0, positive label 1, and negative labels 3, 1,
            and sentinal values are -1, -2, and -3,
        then the batch store will have the key `(node_type, (0, 1, 3))`.

        Args:
            batch_store (abc.MutableMapping): The batch store to use for the graph.
            anchor_node_sentinel (int): The sentinel value for the anchor node.
            positive_label_sentinel (int): The sentinel value for the positive label.
            negative_label_sentinel (Optional[int]): The sentinel value for the negative label.
            If not provided, then negative labels will not be set.
        """
        self._batch_store = batch_store
        self._anchor_node_sentinel = anchor_node_sentinel
        self._positive_label_sentinel = positive_label_sentinel
        self._negative_label_sentinel = negative_label_sentinel

    def __call__(self, data: _GraphData) -> _GraphData:
        """Transform the heterogeneous graph to a homogeneous graph."""
        is_heterogeneous = isinstance(data, HeteroData)
        positive_labels: dict[NodeType, dict[int, torch.Tensor]] = {}
        negative_labels: dict[NodeType, dict[int, torch.Tensor]] = {}
        if is_heterogeneous:
            node_types = data.node_types
        else:
            node_types = [DEFAULT_HOMOGENEOUS_NODE_TYPE]

        # The approach here is:
        # For each node type:
        # 1. Get the "full batch" of node ids, containing the sentinels.
        # e.g. [<anchor node>, <anchor node sentinel>, <positive labels>, <positive label sentinel>, <negative labels>, <negative label sentinel>]
        # or [0, -1, 1, 2, -3, 3, -4]
        # Where 0 is the anchor node, -1 is the anchor node sentinel, 1 and 2 are the positive labels,
        # -3 is the positive label sentinel, 3 is the negative label, and -4 is the negative label sentinel.
        # 2. Select the indices of all the sentinels.
        # 3. Get the labels between the sentinels.
        # 4. Set the labels in the data object.
        # TODO(kmonte): Since the labels are padded, we should be able to vectorize this.
        for node_type in node_types:
            full_batch: torch.Tensor
            # data.batch is already de-duped.
            # Represents all nodes that were sampled in the batch.
            if is_heterogeneous:
                batch = tuple(data[node_type].batch.tolist())
            else:
                batch = tuple(data.batch.tolist())
            full_batch = self._batch_store.pop((node_type, batch))  # [N]
            # Get indices of the sentinels.
            achor_node_sentinels = torch.nonzero(
                full_batch == self._anchor_node_sentinel
            ).squeeze(
                1
            )  # [Num sampled anchors]
            positive_label_sentinels = torch.nonzero(
                full_batch == self._positive_label_sentinel
            ).squeeze(
                1
            )  # [Num sampled anchors]
            pos_labels: dict[int, torch.Tensor] = {}
            if self._negative_label_sentinel is not None:
                negative_label_sentinels = torch.nonzero(
                    full_batch == self._negative_label_sentinel
                ).squeeze(
                    1
                )  # [Num sampled anchors]
                neg_labels = {}
                for anchor, positive, negative in zip(
                    achor_node_sentinels,
                    positive_label_sentinels,
                    negative_label_sentinels,
                ):
                    anchor_node = int(full_batch[anchor - 1].item())
                    pos_batch = full_batch[anchor + 1 : positive].view(
                        -1
                    )  # [max num positive labels]
                    pos_labels[anchor_node] = pos_batch[
                        pos_batch != PADDING_NODE
                    ]  # [num positive labels for $anchor_node]
                    neg_batch = full_batch[positive + 1 : negative].view(
                        -1
                    )  # [max num negative labels]
                    neg_labels[anchor_node] = neg_batch[
                        neg_batch != PADDING_NODE
                    ]  # [num negative labels for $anchor_node]
            else:
                neg_labels = None
                for anchor, positive in zip(
                    achor_node_sentinels, positive_label_sentinels
                ):
                    anchor_node = int(full_batch[anchor - 1].item())
                    pos_batch = full_batch[anchor + 1 : positive].view(
                        -1
                    )  # [max num positive labels]
                    pos_labels[anchor_node] = pos_batch[
                        pos_batch != PADDING_NODE
                    ]  # [num positive labels for $anchor_node]
            positive_labels[node_type] = pos_labels
            if neg_labels is not None:
                negative_labels[node_type] = neg_labels
        if is_heterogeneous:
            data.y_positive = positive_labels
            if negative_labels:
                data.y_negative = negative_labels
        else:
            # Saddly, can't use `to_homogeneous` here for mypy reasons as the values are dicts :(
            data.y_positive = next(iter(positive_labels.values()))
            if negative_labels:
                data.y_negative = next(iter(negative_labels.values()))
        return data
