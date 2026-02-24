import time
from collections.abc import MutableMapping
from multiprocessing.managers import DictProxy
from typing import Literal, Optional, Union, cast

import torch
from graphlearn_torch.partition import PartitionBook

from gigl.common.logger import Logger
from gigl.distributed.graph_store.compute import async_request_server, request_server
from gigl.distributed.graph_store.dist_server import DistServer
from gigl.distributed.utils.networking import get_free_ports
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureInfo,
)
from gigl.utils.sampling import ABLPInputNodes

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
            DistServer.get_node_feature_info,
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
            DistServer.get_edge_feature_info,
        )

    def get_edge_dir(self) -> Union[str, Literal["in", "out"]]:
        """Get the edge direction from the registered dataset.

        Returns:
            The edge direction.
        """
        return request_server(
            0,
            DistServer.get_edge_dir,
        )

    def get_node_partition_book(
        self, node_type: Optional[NodeType] = None
    ) -> Optional[PartitionBook]:
        """
        Gets the partition book for the specified node type.

        Args:
            node_type: The node type to look up.  Must be ``None`` for
                homogeneous datasets and non-``None`` for heterogeneous ones.

        Returns:
            The partition book for the requested node type, or ``None`` if
            no partition book is available.
        """
        node_type = self._infer_node_type_for_homogeneous_with_label_edges(node_type)
        return request_server(
            0,
            DistServer.get_node_partition_book,
            node_type=node_type,
        )

    def get_edge_partition_book(
        self, edge_type: Optional[EdgeType] = None
    ) -> Optional[PartitionBook]:
        """
        Gets the partition book for the specified edge type.

        Args:
            edge_type: The edge type to look up.  Must be ``None`` for
                homogeneous datasets and non-``None`` for heterogeneous ones.

        Returns:
            The partition book for the requested edge type, or ``None`` if
            no partition book is available.
        """
        edge_type = self._maybe_infer_edge_type(edge_type)
        return request_server(
            0,
            DistServer.get_edge_partition_book,
            edge_type=edge_type,
        )

    def _infer_node_type_for_homogeneous_with_label_edges(
        self, node_type: Optional[NodeType]
    ) -> Optional[NodeType]:
        """
        Auto-infers the default homogeneous node type for homogeneous datasets with label edges.
        """
        if node_type is None:
            node_types = self.get_node_types()
            if node_types is not None and DEFAULT_HOMOGENEOUS_NODE_TYPE in node_types:
                node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
                logger.info(
                    f"Auto-inferred default node type {node_type} for homogeneous dataset with label edges "
                    f"as {DEFAULT_HOMOGENEOUS_NODE_TYPE} is in the node types: {node_types}"
                )
        return node_type

    def _maybe_infer_edge_type(
        self, edge_type: Optional[EdgeType]
    ) -> Optional[EdgeType]:
        """
        Auto-infers the default homogeneous edge type for homogeneous datasets with label edges.
        """
        if edge_type is None:
            edge_types = self.get_edge_types()
            if edge_types is not None and DEFAULT_HOMOGENEOUS_EDGE_TYPE in edge_types:
                edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE
                logger.info(
                    f"Auto-inferred default edge type {edge_type} for homogeneous dataset with label edges "
                    f"as {DEFAULT_HOMOGENEOUS_EDGE_TYPE} is in the edge types: {edge_types}"
                )
        return edge_type

    def _get_node_ids(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        node_type: Optional[NodeType] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
    ) -> dict[int, torch.Tensor]:
        """Fetches node ids from the storage nodes for the current compute node (machine)."""
        futures: list[torch.futures.Future[torch.Tensor]] = []
        node_type = self._infer_node_type_for_homogeneous_with_label_edges(node_type)

        logger.info(
            f"Getting node ids for rank {rank} / {world_size} with node type {node_type} and split {split}"
        )

        for server_rank in range(self.cluster_info.num_storage_nodes):
            futures.append(
                async_request_server(
                    server_rank,
                    DistServer.get_node_ids,
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
            ports: Union[list[int], list[None]] = request_server(
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
        return cast(list[int], ports)

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
                    DistServer.get_ablp_input,
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

    # TODO(#488) - support multiple supervision edge types
    def get_ablp_input(
        self,
        split: Literal["train", "val", "test"],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        anchor_node_type: Optional[NodeType] = None,
        supervision_edge_type: Optional[EdgeType] = None,
    ) -> dict[int, ABLPInputNodes]:
        """
        Fetches ABLP (Anchor Based Link Prediction) input from the storage nodes.

        The returned dict maps storage rank to an :class:`ABLPInputNodes` dataclass
        for that storage node. If (rank, world_size) is provided, the input will be
        sharded across the compute nodes. If no (rank, world_size) is provided, the
        input will be returned for all storage nodes.

        The ``ABLPInputNodes`` dataclass carries explicit node type information and
        keys the label tensors by their label ``EdgeType``, making it unambiguous which
        node types the positive/negative labels correspond to.

        Args:
            split (Literal["train", "val", "test"]): The split to get the input for.
            rank (Optional[int]): The rank of the process requesting the input.
                Must be provided if world_size is provided.
            world_size (Optional[int]): The total number of processes in the distributed setup.
                Must be provided if rank is provided.
            anchor_node_type (Optional[NodeType]): The type of the anchor nodes to retrieve.
                Must be provided for heterogeneous graphs.
                Must be None for labeled homogeneous graphs.
                Defaults to None.
            supervision_edge_type (Optional[EdgeType]): The edge type for supervision.
                Must be provided for heterogeneous graphs.
                Must be None for labeled homogeneous graphs.
                Defaults to None.

        Returns:
            dict[int, ABLPInputNodes]:
                A dict mapping storage rank to an ABLPInputNodes containing:
                - anchor_node_type: The node type of the anchor nodes, or None for labeled homogeneous.
                - anchor_nodes: 1D tensor of anchor node IDs for the split.
                - positive_labels: Dict mapping positive label EdgeType to a 2D tensor [N, M].
                - negative_labels: Optional dict mapping negative label EdgeType to a 2D tensor [N, M].

        Examples:
            Suppose we have 1 storage node with users [0, 1, 2, 3, 4] where:
                train=[0, 1, 2], val=[3], test=[4]
            And positive/negative labels defined for link prediction.

            Get training ABLP input (heterogeneous):

            >>> dataset.get_ablp_input(split="train", node_type=USER, supervision_edge_type=USER_TO_ITEM)
            {
                0: ABLPInputNodes(
                    anchor_nodes=tensor([0, 1, 2]),
                    positive_labels={("user", "to_positive", "item"): tensor([[0, 1], [1, 2], [2, 3]])},
                    anchor_node_type="user",
                    negative_labels={("user", "to_negative", "item"): tensor([[2], [3], [4]])},
                )
            }

            For labeled homogeneous graphs, anchor_node_type will be DEFAULT_HOMOGENEOUS_NODE_TYPE.

        Note:
            The GLT sampling engine expects all processes on a given compute machine to have
            the same sampling input (node ids). As such, the input tensors may be duplicated
            across all processes on a given compute machine. To save on CPU memory, pass
            `mp_sharing_dict` to the `RemoteDistDataset` constructor.
        """

        if (anchor_node_type is None) != (supervision_edge_type is None):
            raise ValueError(
                f"anchor_node_type and supervision_edge_type must both be provided or both be None, received: "
                f"anchor_node_type: {anchor_node_type}, supervision_edge_type: {supervision_edge_type}"
            )
        if anchor_node_type is None:
            evaluated_anchor_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        else:
            evaluated_anchor_node_type = anchor_node_type
        if supervision_edge_type is None:
            evaluated_supervision_edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE
        else:
            evaluated_supervision_edge_type = supervision_edge_type
        del anchor_node_type, supervision_edge_type

        def anchors_key(server_rank: int) -> str:
            return f"ablp_server_{server_rank}_anchors"

        def positive_labels_key(server_rank: int) -> str:
            return f"ablp_server_{server_rank}_positive_labels"

        def negative_labels_key(server_rank: int) -> str:
            return f"ablp_server_{server_rank}_negative_labels"

        def wrap_ablp_input(
            anchors: torch.Tensor,
            anchor_node_type: NodeType,
            positive_labels: torch.Tensor,
            negative_labels: Optional[torch.Tensor],
        ) -> ABLPInputNodes:
            """Convert raw tensors into an ABLPInputNodes dataclass."""
            return ABLPInputNodes(
                anchor_node_type=anchor_node_type,
                anchor_nodes=anchors,
                labels={
                    evaluated_supervision_edge_type: (positive_labels, negative_labels)
                },
            )

        if self._mp_sharing_dict is not None:
            if self._local_rank == 0:
                start_time = time.time()
                logger.info(
                    f"Compute rank {torch.distributed.get_rank()} is getting ABLP input from storage nodes"
                )
                raw_ablp_inputs = self._get_ablp_input(
                    split=split,
                    rank=rank,
                    world_size=world_size,
                    node_type=evaluated_anchor_node_type,
                    supervision_edge_type=evaluated_supervision_edge_type,
                )
                for server_rank, (
                    anchors,
                    positive_labels,
                    negative_labels,
                ) in raw_ablp_inputs.items():
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
            returned_ablp_inputs: dict[int, ABLPInputNodes] = {}
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
                returned_ablp_inputs[server_rank] = wrap_ablp_input(
                    anchors=anchors,
                    anchor_node_type=evaluated_anchor_node_type,
                    positive_labels=positive_labels,
                    negative_labels=negative_labels,
                )
            return returned_ablp_inputs
        else:
            raw_inputs = self._get_ablp_input(
                split=split,
                rank=rank,
                world_size=world_size,
                node_type=evaluated_anchor_node_type,
                supervision_edge_type=evaluated_supervision_edge_type,
            )
            return {
                server_rank: wrap_ablp_input(
                    anchor_node_type=evaluated_anchor_node_type,
                    anchors=anchors,
                    positive_labels=positive_labels,
                    negative_labels=negative_labels,
                )
                for server_rank, (
                    anchors,
                    positive_labels,
                    negative_labels,
                ) in raw_inputs.items()
            }

    def get_edge_types(self) -> Optional[list[EdgeType]]:
        """Get the edge types from the registered dataset.

        Returns:
            The edge types in the dataset, None if the dataset is homogeneous.
        """
        return request_server(
            0,
            DistServer.get_edge_types,
        )

    def get_node_types(self) -> Optional[list[NodeType]]:
        """Get the node types from the registered dataset.

        Returns:
            The node types in the dataset, None if the dataset is homogeneous.
        """
        return request_server(
            0,
            DistServer.get_node_types,
        )
