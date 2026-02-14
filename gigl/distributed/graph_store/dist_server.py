"""
GiGL implementation of GLT DistServer.

Main change here is that we use gigl DistAblpSamplingProducer instead of GLT DistMpSamplingProducer.

Based on https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_server.py
"""

import logging
import threading
import time
import warnings
from collections import abc
from typing import Literal, Optional, Union

import graphlearn_torch.distributed.dist_server as glt_dist_server
import torch
from graphlearn_torch.channel import QueueTimeoutError, ShmChannel
from graphlearn_torch.distributed import (
    DistMpSamplingProducer,
    RemoteDistSamplingWorkerOptions,
    barrier,
    init_rpc,
    shutdown_rpc,
)
from graphlearn_torch.partition import PartitionBook
from graphlearn_torch.sampler import (
    EdgeSamplerInput,
    NodeSamplerInput,
    RemoteSamplerInput,
    SamplingConfig,
)

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_sampling_producer import DistABLPSamplingProducer
from gigl.distributed.sampler import ABLPNodeSamplerInput
from gigl.distributed.utils.neighborloader import shard_nodes_by_process
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureInfo,
    select_label_edge_types,
)
from gigl.utils.data_splitters import get_labels_for_anchor_nodes

SERVER_EXIT_STATUS_CHECK_INTERVAL = 5.0
r""" Interval (in seconds) to check exit status of server.
"""


# TODO(kmonte): Migrate graph_store/storage_utils to this class.
class DistServer(object):
    r"""A server that supports launching remote sampling workers for
    training clients.

    Note that this server is enabled only when the distribution mode is a
    server-client framework, and the graph and feature store will be partitioned
    and managed by all server nodes.

    Args:
      dataset (DistDataset): The ``DistDataset`` object of a partition of graph
        data and feature data, along with distributed patition books.
    """

    def __init__(self, dataset: DistDataset) -> None:
        self.dataset = dataset
        self._lock = threading.RLock()
        self._exit = False
        self._cur_producer_idx = 0  # auto incremental index (same as producer count)
        # The mapping from the key in worker options (such as 'train', 'test')
        # to producer id
        self._worker_key2producer_id: dict[str, int] = {}
        self._producer_pool: dict[int, DistABLPSamplingProducer] = {}
        self._msg_buffer_pool: dict[int, ShmChannel] = {}
        self._epoch: dict[int, int] = {}  # last epoch for the producer

    def shutdown(self) -> None:
        for producer_id in list(self._producer_pool.keys()):
            self.destroy_sampling_producer(producer_id)
        assert len(self._producer_pool) == 0
        assert len(self._msg_buffer_pool) == 0

    def wait_for_exit(self) -> None:
        r"""Block until the exit flag been set to ``True``."""
        while not self._exit:
            time.sleep(SERVER_EXIT_STATUS_CHECK_INTERVAL)

    def exit(self) -> bool:
        r"""Set the exit flag to ``True``."""
        self._exit = True
        return self._exit

    def get_dataset_meta(
        self,
    ) -> tuple[int, int, Optional[list[NodeType]], Optional[list[EdgeType]]]:
        r"""Get the meta info of the distributed dataset managed by the current
        server, including partition info and graph types.
        """
        return (
            self.dataset.num_partitions,
            self.dataset.partition_idx,
            self.dataset.get_node_types(),
            self.dataset.get_edge_types(),
        )

    def get_node_partition_id(
        self, node_type: Optional[NodeType], index: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if isinstance(self.dataset.node_pb, PartitionBook):
            if node_type is not None:
                raise ValueError(
                    f"node_type must be None for homogeneous dataset. Received: {node_type}"
                )
            partition_id = self.dataset.node_pb[index]
            return partition_id
        elif isinstance(self.dataset.node_pb, dict):
            if node_type is None:
                raise ValueError(
                    f"node_type must be provided for heterogeneous dataset. Received: {node_type}"
                )
            partition_id = self.dataset.node_pb[node_type][index]
            return partition_id
        return None

    def get_node_feature(
        self, node_type: Optional[NodeType], index: torch.Tensor
    ) -> torch.Tensor:
        feature = self.dataset.get_node_feature(node_type)
        return feature[index].cpu()

    def get_tensor_size(self, node_type: Optional[NodeType]) -> torch.Size:
        feature = self.dataset.get_node_feature(node_type)
        return feature.shape

    def get_node_label(
        self, node_type: Optional[NodeType], index: torch.Tensor
    ) -> torch.Tensor:
        label = self.dataset.get_node_label(node_type)
        return label[index]

    def get_edge_index(
        self, edge_type: Optional[EdgeType], layout: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.dataset.get_graph(edge_type)
        row = None
        col = None
        result = None
        if layout == "coo":
            row, col, _, _ = graph.topo.to_coo()
            result = (row, col)
        else:
            raise ValueError(f"Invalid layout {layout}")
        return result

    def get_edge_size(
        self, edge_type: Optional[EdgeType], layout: str
    ) -> tuple[int, int]:
        graph = self.dataset.get_graph(edge_type)
        if layout == "coo":
            row_count = graph.row_count
            col_count = graph.col_count
        else:
            raise ValueError(f"Invalid layout {layout}")
        return (row_count, col_count)

    def get_node_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[NodeType, FeatureInfo], None]:
        """Get node feature information from the dataset.

        Returns:
            Node feature information, which can be:
            - A single FeatureInfo object for homogeneous graphs
            - A dict mapping NodeType to FeatureInfo for heterogeneous graphs
            - None if no node features are available
        """
        return self.dataset.node_feature_info

    def get_edge_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[EdgeType, FeatureInfo], None]:
        """Get edge feature information from the dataset.

        Returns:
            Edge feature information, which can be:
            - A single FeatureInfo object for homogeneous graphs
            - A dict mapping EdgeType to FeatureInfo for heterogeneous graphs
            - None if no edge features are available
        """
        return self.dataset.edge_feature_info

    def get_edge_dir(self) -> Literal["in", "out"]:
        """Get the edge direction from the dataset.

        Returns:
            The edge direction.
        """
        return self.dataset.edge_dir

    def get_node_ids(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        split: Optional[Union[Literal["train", "val", "test"], str]] = None,
        node_type: Optional[NodeType] = None,
    ) -> torch.Tensor:
        """Get the node ids from the dataset.

        Args:
            rank: The rank of the process requesting node ids. Must be provided if world_size is provided.
            world_size: The total number of processes in the distributed setup. Must be provided if rank is provided.
            split: The split of the dataset to get node ids from. If provided, the dataset must have
                `train_node_ids`, `val_node_ids`, and `test_node_ids` properties.
            node_type: The type of nodes to get node ids for. Must be provided if the dataset is heterogeneous.

        Returns:
            The node ids.

        Raises:
            ValueError:
                * If the rank and world_size are not provided together
                * If the split is invalid
                * If the node ids are not a torch.Tensor or a dict[NodeType, torch.Tensor]
                * If the node type is provided for a homogeneous dataset
                * If the node ids are not a dict[NodeType, torch.Tensor] when no node type is provided

        Examples:
            Suppose the dataset has 100 nodes total: train=[0..59], val=[60..79], test=[80..99].

            Get all node ids (no split filtering):

            >>> server.get_node_ids()
            tensor([0, 1, 2, ..., 99])  # All 100 nodes

            Get only training nodes:

            >>> server.get_node_ids(split="train")
            tensor([0, 1, 2, ..., 59])  # 60 training nodes

            Shard all nodes across 4 processes (each gets ~25 nodes):

            >>> server.get_node_ids(rank=0, world_size=4)
            tensor([0, 1, 2, ..., 24])  # First 25 of all 100 nodes

            Shard training nodes across 4 processes (each gets ~15 nodes):

            >>> server.get_node_ids(rank=0, world_size=4, split="train")
            tensor([0, 1, 2, ..., 14])  # First 15 of the 60 training nodes

            Note: When `split=None`, all nodes are queryable. This means nodes from any
            split (train, val, or test) may be returned. This is useful when you need
            to sample neighbors during inference, as neighbor nodes may belong to any split.
        """
        if (rank is None) ^ (world_size is None):
            raise ValueError(
                f"rank and world_size must be provided together. Received rank: {rank}, world_size: {world_size}"
            )
        if split == "train":
            nodes = self.dataset.train_node_ids
        elif split == "val":
            nodes = self.dataset.val_node_ids
        elif split == "test":
            nodes = self.dataset.test_node_ids
        elif split is None:
            nodes = self.dataset.node_ids
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be one of 'train', 'val', 'test', or None."
            )

        if node_type is not None:
            if not isinstance(nodes, abc.Mapping):
                raise ValueError(
                    f"node_type was provided as {node_type}, so node ids must be a dict[NodeType, torch.Tensor] (e.g. a heterogeneous dataset), got {type(nodes)}"
                )
            nodes = nodes[node_type]
        elif not isinstance(nodes, torch.Tensor):
            raise ValueError(
                f"node_type was not provided, so node ids must be a torch.Tensor (e.g. a homogeneous dataset), got {type(nodes)}."
            )

        if rank is not None and world_size is not None:
            return shard_nodes_by_process(nodes, rank, world_size)
        return nodes

    def get_edge_types(self) -> Optional[list[EdgeType]]:
        """Get the edge types from the dataset.

        Returns:
            The edge types in the dataset, None if the dataset is homogeneous.
        """
        if isinstance(self.dataset.graph, dict):
            return list(self.dataset.graph.keys())
        else:
            return None

    def get_ablp_input(
        self,
        split: Union[Literal["train", "val", "test"], str],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE,
        supervision_edge_type: EdgeType = DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get the ABLP (Anchor Based Link Prediction) input for a specific rank in distributed processing.

        Note: rank and world_size here are for the process group we're *fetching for*, not the process group we're *fetching from*.
        e.g. if our compute cluster is of world size 4, and we have 2 storage nodes, then the world size this gets called with is 4, not 2.

        Args:
            split: The split to get the training input for.
            rank: The rank of the process requesting the training input. Defaults to None, in which case all nodes are returned.
                Must be provided if world_size is provided.
            world_size: The total number of processes in the distributed setup. Defaults to None, in which case all nodes are returned.
                Must be provided if rank is provided.
            node_type: The type of nodes to retrieve. Defaults to the default homogeneous node type.
            supervision_edge_type: The edge type to use for the supervision. Defaults to the default homogeneous edge type.

        Returns:
            A tuple containing the anchor nodes for the rank, the positive labels, and the negative labels.
            The positive labels are of shape [N, M], where N is the number of anchor nodes and M is the number of positive labels.
            The negative labels are of shape [N, M], where N is the number of anchor nodes and M is the number of negative labels.
            The negative labels may be None if no negative labels are available.

        Raises:
            ValueError: If the split is invalid.
        """
        anchors = self.get_node_ids(
            split=split, rank=rank, world_size=world_size, node_type=node_type
        )
        positive_label_edge_type, negative_label_edge_type = select_label_edge_types(
            supervision_edge_type, self.dataset.get_edge_types()
        )
        positive_labels, negative_labels = get_labels_for_anchor_nodes(
            self.dataset, anchors, positive_label_edge_type, negative_label_edge_type
        )
        return anchors, positive_labels, negative_labels

    def create_sampling_ablp_producer(
        self,
        sampler_input: Union[
            NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput
        ],
        sampling_config: SamplingConfig,
        worker_options: RemoteDistSamplingWorkerOptions,
    ) -> int:
        r"""Create and initialize an instance of ``DistABLPSamplingProducer`` with
        a group of subprocesses for distributed sampling.

        Args:
          sampler_input (NodeSamplerInput or EdgeSamplerInput): The input data
            for sampling.
          sampling_config (SamplingConfig): Configuration of sampling meta info.
          worker_options (RemoteDistSamplingWorkerOptions): Options for launching
            remote sampling workers by this server.

        Returns:
          A unique id of created sampling producer on this server.
        """

        if not isinstance(sampler_input, ABLPNodeSamplerInput):
            raise ValueError(
                f"Sampler input must be an instance of ABLPNodeSamplerInput. Received: {type(sampler_input)}"
            )

        return self._create_producer(
            sampler_input=sampler_input,
            sampling_config=sampling_config,
            worker_options=worker_options,
            producer_cls=DistABLPSamplingProducer,
        )

    def create_sampling_producer(
        self,
        sampler_input: Union[
            NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput
        ],
        sampling_config: SamplingConfig,
        worker_options: RemoteDistSamplingWorkerOptions,
    ) -> int:
        r"""Create and initialize an instance of ``DistSamplingProducer`` with
        a group of subprocesses for distributed sampling.

        Args:
          sampler_input (NodeSamplerInput or EdgeSamplerInput): The input data
            for sampling.
          sampling_config (SamplingConfig): Configuration of sampling meta info.
          worker_options (RemoteDistSamplingWorkerOptions): Options for launching
            remote sampling workers by this server.

        Returns:
          A unique id of created sampling producer on this server.
        """
        return self._create_producer(
            sampler_input=sampler_input,
            sampling_config=sampling_config,
            worker_options=worker_options,
            producer_cls=DistMpSamplingProducer,
        )

    def _create_producer(
        self,
        sampler_input: Union[
            NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput, ABLPNodeSamplerInput
        ],
        sampling_config: SamplingConfig,
        worker_options: RemoteDistSamplingWorkerOptions,
        producer_cls: type[Union[DistABLPSamplingProducer, DistMpSamplingProducer]],
    ) -> int:
        r"""Shared logic to create and initialize a sampling producer.

        Converts remote sampler inputs to local, creates a ``ShmChannel`` buffer,
        instantiates the given ``producer_cls``, and registers it in the internal pools.

        Args:
          sampler_input (NodeSamplerInput, EdgeSamplerInput, RemoteSamplerInput,
            or ABLPNodeSamplerInput): The input data for sampling.
          sampling_config (SamplingConfig): Configuration of sampling meta info.
          worker_options (RemoteDistSamplingWorkerOptions): Options for launching
            remote sampling workers by this server.
          producer_cls: The producer class to instantiate
            (``DistABLPSamplingProducer`` or ``DistMpSamplingProducer``).

        Returns:
          int: A unique id of created sampling producer on this server.
        """
        if isinstance(sampler_input, RemoteSamplerInput):
            sampler_input = sampler_input.to_local_sampler_input(dataset=self.dataset)

        with self._lock:
            producer_id = self._worker_key2producer_id.get(worker_options.worker_key)
            if producer_id is None:
                producer_id = self._cur_producer_idx
                self._worker_key2producer_id[worker_options.worker_key] = producer_id
                self._cur_producer_idx += 1
                buffer = ShmChannel(
                    worker_options.buffer_capacity, worker_options.buffer_size
                )
                producer = producer_cls(
                    self.dataset, sampler_input, sampling_config, worker_options, buffer
                )
                producer.init()
                self._producer_pool[producer_id] = producer
                self._msg_buffer_pool[producer_id] = buffer
                self._epoch[producer_id] = -1
        return producer_id

    def destroy_sampling_producer(self, producer_id: int) -> None:
        r"""Shutdown and destroy a sampling producer managed by this server with
        its producer id.
        """
        with self._lock:
            producer = self._producer_pool.get(producer_id, None)
            if producer is not None:
                producer.shutdown()
                self._producer_pool.pop(producer_id)
                self._msg_buffer_pool.pop(producer_id)
                self._epoch.pop(producer_id)

    def start_new_epoch_sampling(self, producer_id: int, epoch: int) -> None:
        r"""Start a new epoch sampling tasks for a specific sampling producer
        with its producer id.
        """
        with self._lock:
            cur_epoch = self._epoch[producer_id]
            if cur_epoch < epoch:
                self._epoch[producer_id] = epoch
                producer = self._producer_pool.get(producer_id, None)
                if producer is not None:
                    producer.produce_all()

    def fetch_one_sampled_message(
        self, producer_id: int
    ) -> tuple[Optional[bytes], bool]:
        r"""Fetch a sampled message from the buffer of a specific sampling
        producer with its producer id.
        """
        producer = self._producer_pool.get(producer_id, None)
        if producer is None:
            warnings.warn("invalid producer_id {producer_id}")
            return None, False
        if producer.is_all_sampling_completed_and_consumed():
            return None, True
        buffer = self._msg_buffer_pool.get(producer_id, None)
        while True:
            try:
                msg = buffer.recv(timeout_ms=500)
                return msg, False
            except QueueTimeoutError as e:
                if producer.is_all_sampling_completed():
                    return None, True


_dist_server: Optional[DistServer] = None
r"""``DistServer`` instance of the current process."""


def get_server() -> DistServer:
    r"""Get the ``DistServer`` instance on the current process."""
    if _dist_server is None:
        raise RuntimeError("DistServer not initialized! Call init_server() first.")
    return _dist_server


def init_server(
    num_servers: int,
    server_rank: int,
    dataset: DistDataset,
    master_addr: str,
    master_port: int,
    num_clients: int = 0,
    num_rpc_threads: int = 16,
    request_timeout: int = 180,
    server_group_name: Optional[str] = None,
    is_dynamic: bool = False,
) -> None:
    r"""Initialize the current process as a server and establish connections
    with all other servers and clients. Note that this method should be called
    only in the server-client distribution mode.

    Args:
      num_servers (int): Number of processes participating in the server group.
      server_rank (int): Rank of the current process withing the server group (it
        should be a number between 0 and ``num_servers``-1).
      dataset (DistDataset): The ``DistDataset`` object of a partition of graph
        data and feature data, along with distributed patition book info.
      master_addr (str): The master TCP address for RPC connection between all
        servers and clients, the value of this parameter should be same for all
        servers and clients.
      master_port (int): The master TCP port for RPC connection between all
        servers and clients, the value of this parameter should be same for all
        servers and clients.
      num_clients (int): Number of processes participating in the client group.
        if ``is_dynamic`` is ``True``, this parameter will be ignored.
      num_rpc_threads (int): The number of RPC worker threads used for the
        current server to respond remote requests. (Default: ``16``).
      request_timeout (int): The max timeout seconds for remote requests,
        otherwise an exception will be raised. (Default: ``16``).
      server_group_name (str): A unique name of the server group that current
        process belongs to. If set to ``None``, a default name will be used.
        (Default: ``None``).
      is_dynamic (bool): Whether the world size is dynamic. (Default: ``False``).
    """
    if server_group_name:
        server_group_name = server_group_name.replace("-", "_")
    glt_dist_server._set_server_context(
        num_servers, server_rank, server_group_name, num_clients
    )
    global _dist_server
    _dist_server = DistServer(dataset=dataset)
    # Also set GLT's _dist_server so that GLT's RPC mechanism routes to GiGL's server
    glt_dist_server._dist_server = _dist_server
    init_rpc(
        master_addr,
        master_port,
        num_rpc_threads,
        request_timeout,
        is_dynamic=is_dynamic,
    )


def wait_and_shutdown_server() -> None:
    r"""Block until all client have been shutdowned, and further shutdown the
    server on the current process and destroy all RPC connections.
    """
    current_context = glt_dist_server.get_context()
    if current_context is None:
        logging.warning(
            "'wait_and_shutdown_server': try to shutdown server when "
            "the current process has not been initialized as a server."
        )
        return
    if not current_context.is_server():
        raise RuntimeError(
            f"'wait_and_shutdown_server': role type of "
            f"the current process context is not a server, "
            f"got {current_context.role}."
        )
    global _dist_server
    if _dist_server is not None:
        _dist_server.wait_for_exit()
        _dist_server.shutdown()
        _dist_server = None
        # Also clear GLT's _dist_server
        glt_dist_server._dist_server = None
    barrier()
    shutdown_rpc()


def _call_func_on_server(func, *args, **kwargs):
    r"""A callee entry for remote requests on the server side."""
    if not callable(func):
        logging.warning(
            f"'_call_func_on_server': receive a non-callable " f"function target {func}"
        )
        return None

    server = get_server()
    if hasattr(server, func.__name__):
        return func(server, *args, **kwargs)

    return func(*args, **kwargs)
