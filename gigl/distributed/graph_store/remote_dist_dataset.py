import time
from collections.abc import MutableMapping
from multiprocessing.managers import DictProxy
from typing import Literal, Optional, Union

import torch
from graphlearn_torch.distributed import async_request_server, request_server

from gigl.common.logger import Logger
from gigl.distributed.graph_store.storage_utils import (
    get_ablp_input,
    get_edge_dir,
    get_edge_feature_info,
    get_edge_types,
    get_node_feature_info,
    get_node_ids,
)
from gigl.distributed.utils.networking import get_free_ports
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureInfo,
)

logger = Logger()


class RemoteDistDataset:
    def __init__(
        self,
        cluster_info: GraphStoreInfo,
        local_rank: int,
        mp_sharing_dict: Optional[MutableMapping[str, torch.Tensor]] = None,
    ):
        """
        Represents a dataset that is stored on a difference storage cluster.
        *Must* be used in the GiGL graph-store distributed setup.

        This class *must* be used on the compute (client) side of the graph-store distributed setup.

        Args:
            cluster_info (GraphStoreInfo): The cluster information.
            local_rank (int): The local rank of the process on the compute node.
            mp_sharing_dict (Optional[MutableMapping[str, torch.Tensor]]):
                (Optional) If provided, will be used to share tensors across the local machine.
                e.g. for `get_node_ids`.
                If provided, *must* be a `DictProxy` e.g. the return value of a mp.Manager.
                ex. torch.multiprocessing.Manager().dict().
        """
        self._cluster_info = cluster_info
        self._local_rank = local_rank
        self._mp_sharing_dict = mp_sharing_dict
        # We accept mp_sharing_dict as a `MutableMapping` as if we directly annotate as `DictProxy` (which is what we want)
        # Then we will have runtime failures e.g.:
        #   File "/.../GiGL/python/gigl/distributed/graph_store/remote_dist_dataset.py", line 30, in RemoteDistDataset
        #   mp_sharing_dict: Optional[DictProxy[str, torch.Tensor]] = None,
        #                              ~~~~~~~~~^^^^^^^^^^^^^^^^^^^
        # TypeError: type 'DictProxy' is not subscriptable

        if self._mp_sharing_dict is not None and not isinstance(
            self._mp_sharing_dict, DictProxy
        ):
            raise ValueError(
                f"When using mp_sharing_dict, you must pass in a `DictProxy` e.g. mp.manager().dict(). Recieved a {type(self._mp_sharing_dict)}"
            )

    @property
    def cluster_info(self) -> GraphStoreInfo:
        return self._cluster_info

    def get_node_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[NodeType, FeatureInfo], None]:
        """Get node feature information from the registered dataset.

        Returns:
            Node feature information, which can be:
            - A single FeatureInfo object for homogeneous graphs
            - A dict mapping NodeType to FeatureInfo for heterogeneous graphs
            - None if no node features are available
        """
        return request_server(
            0,
            get_node_feature_info,
        )

    def get_edge_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[EdgeType, FeatureInfo], None]:
        """Get edge feature information from the registered dataset.

        Returns:
            Edge feature information, which can be:
            - A single FeatureInfo object for homogeneous graphs
            - A dict mapping EdgeType to FeatureInfo for heterogeneous graphs
            - None if no edge features are available
        """
        return request_server(
            0,
            get_edge_feature_info,
        )

    def get_edge_dir(self) -> Union[str, Literal["in", "out"]]:
        """Get the edge direction from the registered dataset.

        Returns:
            The edge direction.
        """
        return request_server(
            0,
            get_edge_dir,
        )

    def _get_node_ids(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        node_type: Optional[NodeType] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
    ) -> dict[int, torch.Tensor]:
        """Fetches node ids from the storage nodes for the current compute node (machine)."""
        futures: list[torch.futures.Future[torch.Tensor]] = []
        logger.info(
            f"Getting node ids for rank {rank} / {world_size} with node type {node_type} and split {split}"
        )

        for server_rank in range(self.cluster_info.num_storage_nodes):
            futures.append(
                async_request_server(
                    server_rank,
                    get_node_ids,
                    rank=rank,
                    world_size=world_size,
                    split=split,
                    node_type=node_type,
                )
            )
            node_ids = torch.futures.wait_all(futures)
        return {server_rank: node_ids for server_rank, node_ids in enumerate(node_ids)}

    def get_node_ids(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        node_type: Optional[NodeType] = None,
    ) -> dict[int, torch.Tensor]:
        """
        Fetches node ids from the storage nodes for the current compute node (machine).

        The returned dict maps storage rank to the node ids stored on that storage node,
        filtered and sharded according to the provided arguments.

        Args:
            rank (Optional[int]): The rank of the process requesting node ids. Must be provided if world_size is provided.
            world_size (Optional[int]): The total number of processes in the distributed setup. Must be provided if rank is provided.
            split (Optional[Literal["train", "val", "test"]]):
                The split of the dataset to get node ids from.
                If provided, the dataset must have `train_node_ids`, `val_node_ids`, and `test_node_ids` properties.
            node_type (Optional[NodeType]): The type of nodes to get.
                Must be provided for heterogeneous datasets.

        Returns:
            dict[int, torch.Tensor]: A dict mapping storage rank to node ids.

        Examples:
            Suppose we have 2 storage nodes and 2 compute nodes, with 16 total nodes.
            Nodes are partitioned across storage nodes, with splits defined as:

                Storage rank 0: [0, 1, 2, 3, 4, 5, 6, 7]
                    train=[0, 1, 2, 3], val=[4, 5], test=[6, 7]
                Storage rank 1: [8, 9, 10, 11, 12, 13, 14, 15]
                    train=[8, 9, 10, 11], val=[12, 13], test=[14, 15]

            Get all nodes (no split filtering, no sharding):

            >>> dataset.get_node_ids()
            {
                0: tensor([0, 1, 2, 3, 4, 5, 6, 7]),      # All 8 nodes from storage rank 0
                1: tensor([8, 9, 10, 11, 12, 13, 14, 15]) # All 8 nodes from storage rank 1
            }

            Shard all nodes across 2 compute nodes (compute rank 0 gets first half from each storage):

            >>> dataset.get_node_ids(rank=0, world_size=2)
            {
                0: tensor([0, 1, 2, 3]),   # First 4 of all 8 nodes from storage rank 0
                1: tensor([8, 9, 10, 11])  # First 4 of all 8 nodes from storage rank 1
            }

            Get only training nodes (no sharding):

            >>> dataset.get_node_ids(split="train")
            {
                0: tensor([0, 1, 2, 3]),   # 4 training nodes from storage rank 0
                1: tensor([8, 9, 10, 11])  # 4 training nodes from storage rank 1
            }

            Combine split and sharding (training nodes, sharded for compute rank 0):

            >>> dataset.get_node_ids(rank=0, world_size=2, split="train")
            {
                0: tensor([0, 1]),  # First 2 of 4 training nodes from storage rank 0
                1: tensor([8, 9])   # First 2 of 4 training nodes from storage rank 1
            }

        Note:
            When `split=None`, all nodes are queryable. This means nodes from any split
            (train, val, or test) may be returned. This is useful when you need to sample
            neighbors during inference, as neighbor nodes may belong to any split.

            The GLT sampling engine expects all processes on a given compute machine to have
            the same sampling input (node ids). As such, the input tensors may be duplicated
            across all processes on a given compute machine. To save on CPU memory, pass
            `mp_sharing_dict` to the `RemoteDistDataset` constructor.
        """

        def server_key(server_rank: int) -> str:
            return f"node_ids_from_server_{server_rank}"

        if self._mp_sharing_dict is not None:
            if self._local_rank == 0:
                start_time = time.time()
                logger.info(
                    f"Compute rank {torch.distributed.get_rank()} is getting node ids from storage nodes"
                )
                node_ids = self._get_node_ids(rank, world_size, node_type, split)
                for server_rank, node_id in node_ids.items():
                    node_id.share_memory_()
                    self._mp_sharing_dict[server_key(server_rank)] = node_id
                logger.info(
                    f"Compute rank {torch.distributed.get_rank()} got node ids from storage nodes in {time.time() - start_time:.2f} seconds"
                )
            torch.distributed.barrier()
            node_ids = {
                server_rank: self._mp_sharing_dict[server_key(server_rank)]
                for server_rank in range(self.cluster_info.num_storage_nodes)
            }
            return node_ids
        else:
            return self._get_node_ids(rank, world_size, node_type, split)

    def get_free_ports_on_storage_cluster(self, num_ports: int) -> list[int]:
        """
        Get free ports from the storage master node.

        This *must* be used with a torch.distributed process group initialized, for the *entire* training cluster.

        All compute ranks will receive the same free ports.

        Args:
            num_ports (int): Number of free ports to get.

        Returns:
            list[int]: A list of free port numbers on the storage master node.
        """
        if not torch.distributed.is_initialized():
            raise ValueError(
                "torch.distributed process group must be initialized for the entire training cluster"
            )
        compute_cluster_rank = (
            self.cluster_info.compute_node_rank
            * self.cluster_info.num_processes_per_compute
            + self._local_rank
        )
        if compute_cluster_rank == 0:
            ports = request_server(
                0,
                get_free_ports,
                num_ports=num_ports,
            )
            logger.info(
                f"Compute rank {compute_cluster_rank} found free ports: {ports}"
            )
        else:
            ports = [None] * num_ports
        torch.distributed.broadcast_object_list(ports, src=0)
        logger.info(f"Compute rank {compute_cluster_rank} received free ports: {ports}")
        return ports

    def _get_ablp_input(
        self,
        split: Literal["train", "val", "test"],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE,
        supervision_edge_type: EdgeType = DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Fetches ABLP input from the storage nodes for the current compute node (machine)."""
        futures: list[
            torch.futures.Future[
                tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            ]
        ] = []
        logger.info(
            f"Getting ABLP input for rank {rank} / {world_size} with node type {node_type}, "
            f"split {split}, and supervision edge type {supervision_edge_type}"
        )

        for server_rank in range(self.cluster_info.num_storage_nodes):
            futures.append(
                async_request_server(
                    server_rank,
                    get_ablp_input,
                    split=split,
                    rank=rank,
                    world_size=world_size,
                    node_type=node_type,
                    supervision_edge_type=supervision_edge_type,
                )
            )
            ablp_inputs = torch.futures.wait_all(futures)
        return {
            server_rank: ablp_input
            for server_rank, ablp_input in enumerate(ablp_inputs)
        }

    def get_ablp_input(
        self,
        split: Literal["train", "val", "test"],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE,
        supervision_edge_type: EdgeType = DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Fetches ABLP (Anchor Based Link Prediction) input from the storage nodes.

        The returned dict maps storage rank to a tuple of (anchor_nodes, positive_labels, negative_labels)
        for that storage node, filtered and sharded according to the provided arguments.

        Args:
            split (Literal["train", "val", "test"]): The split to get the input for.
            rank (Optional[int]): The rank of the process requesting the input.
                Must be provided if world_size is provided.
            world_size (Optional[int]): The total number of processes in the distributed setup.
                Must be provided if rank is provided.
            node_type (NodeType): The type of nodes to retrieve.
                Defaults to DEFAULT_HOMOGENEOUS_NODE_TYPE.
            supervision_edge_type (EdgeType): The edge type for supervision.
                Defaults to DEFAULT_HOMOGENEOUS_EDGE_TYPE.

        Returns:
            dict[int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
                A dict mapping storage rank to a tuple of:
                - anchor_nodes: The anchor node ids for the split
                - positive_labels: Positive label node ids of shape [N, M] where N is the number
                  of anchor nodes and M is the number of positive labels per anchor
                - negative_labels: Negative label node ids of shape [N, M], or None if unavailable

        Examples:
            Suppose we have 1 storage node with users [0, 1, 2, 3, 4] where:
                train=[0, 1, 2], val=[3], test=[4]
            And positive/negative labels defined for link prediction.

            Get training ABLP input:

            >>> dataset.get_ablp_input(split="train", node_type=USER)
            {
                0: (
                    tensor([0, 1, 2]),           # anchor nodes
                    tensor([[0, 1], [1, 2], [2, 3]]),  # positive labels
                    tensor([[2], [3], [4]])     # negative labels
                )
            }

            With sharding across 2 compute nodes (rank 0 gets first portion):

            >>> dataset.get_ablp_input(split="train", rank=0, world_size=2, node_type=USER)
            {
                0: (
                    tensor([0]),                # first anchor node
                    tensor([[0, 1]]),           # its positive labels
                    tensor([[2]])               # its negative labels
                )
            }

        Note:
            The GLT sampling engine expects all processes on a given compute machine to have
            the same sampling input (node ids). As such, the input tensors may be duplicated
            across all processes on a given compute machine. To save on CPU memory, pass
            `mp_sharing_dict` to the `RemoteDistDataset` constructor.
        """

        def anchors_key(server_rank: int) -> str:
            return f"ablp_server_{server_rank}_anchors"

        def positive_labels_key(server_rank: int) -> str:
            return f"ablp_server_{server_rank}_positive_labels"

        def negative_labels_key(server_rank: int) -> str:
            return f"ablp_server_{server_rank}_negative_labels"

        if self._mp_sharing_dict is not None:
            if self._local_rank == 0:
                start_time = time.time()
                logger.info(
                    f"Compute rank {torch.distributed.get_rank()} is getting ABLP input from storage nodes"
                )
                ablp_inputs = self._get_ablp_input(
                    split, rank, world_size, node_type, supervision_edge_type
                )
                for server_rank, (
                    anchors,
                    positive_labels,
                    negative_labels,
                ) in ablp_inputs.items():
                    anchors.share_memory_()
                    positive_labels.share_memory_()
                    self._mp_sharing_dict[anchors_key(server_rank)] = anchors
                    self._mp_sharing_dict[
                        positive_labels_key(server_rank)
                    ] = positive_labels
                    if negative_labels is not None:
                        negative_labels.share_memory_()
                        self._mp_sharing_dict[
                            negative_labels_key(server_rank)
                        ] = negative_labels
                logger.info(
                    f"Compute rank {torch.distributed.get_rank()} got ABLP input from storage nodes "
                    f"in {time.time() - start_time:.2f} seconds"
                )
            torch.distributed.barrier()
            returned_ablp_inputs: dict[
                int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            ] = {}
            for server_rank in range(self.cluster_info.num_storage_nodes):
                anchors = self._mp_sharing_dict[anchors_key(server_rank)]
                positive_labels = self._mp_sharing_dict[
                    positive_labels_key(server_rank)
                ]
                neg_key = negative_labels_key(server_rank)
                negative_labels = (
                    self._mp_sharing_dict[neg_key]
                    if neg_key in self._mp_sharing_dict
                    else None
                )
                returned_ablp_inputs[server_rank] = (
                    anchors,
                    positive_labels,
                    negative_labels,
                )
            return returned_ablp_inputs
        else:
            return self._get_ablp_input(
                split, rank, world_size, node_type, supervision_edge_type
            )

    def get_edge_types(self) -> Optional[list[EdgeType]]:
        """Get the edge types from the registered dataset.

        Returns:
            The edge types in the dataset, None if the dataset is homogeneous.
        """
        return request_server(
            0,
            get_edge_types,
        )
