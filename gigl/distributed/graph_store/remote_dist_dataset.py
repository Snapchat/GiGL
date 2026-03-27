from typing import Literal, Optional, Union, cast

import torch
from graphlearn_torch.partition import PartitionBook

from gigl.common.logger import Logger
from gigl.distributed.graph_store.compute import async_request_server, request_server
from gigl.distributed.graph_store.dist_server import DistServer
from gigl.distributed.graph_store.messages import (
    FetchABLPInputRequest,
    FetchNodesRequest,
)
from gigl.distributed.utils.neighborloader import (
    ShardStrategy,
    compute_server_assignments,
)
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
    ):
        """
        Represents a dataset that is stored on a different storage cluster.
        *Must* be used in the GiGL graph-store distributed setup.

        This class *must* be used on the compute (client) side of the graph-store distributed setup.

        Args:
            cluster_info (GraphStoreInfo): The cluster information.
            local_rank (int): The local rank of the process on the compute node.
        """
        self._cluster_info = cluster_info
        self._local_rank = local_rank

    @property
    def cluster_info(self) -> GraphStoreInfo:
        return self._cluster_info

    def fetch_node_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[NodeType, FeatureInfo], None]:
        """Fetch node feature information from the registered dataset.

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

    def fetch_edge_feature_info(
        self,
    ) -> Union[FeatureInfo, dict[EdgeType, FeatureInfo], None]:
        """Fetch edge feature information from the registered dataset.

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

    def fetch_edge_dir(self) -> Union[str, Literal["in", "out"]]:
        """Fetch the edge direction from the registered dataset.

        Returns:
            The edge direction.
        """
        return request_server(
            0,
            DistServer.get_edge_dir,
        )

    def fetch_node_partition_book(
        self, node_type: Optional[NodeType] = None
    ) -> Optional[PartitionBook]:
        """
        Fetches the partition book for the specified node type.

        Args:
            node_type: The node type to look up.  Must be ``None`` for
                homogeneous datasets and non-``None`` for heterogeneous ones.

        Returns:
            The partition book for the requested node type, or ``None`` if
            no partition book is available.
        """
        node_type = self._infer_node_type_if_homogeneous_with_label_edges(node_type)
        return request_server(
            0,
            DistServer.get_node_partition_book,
            node_type=node_type,
        )

    def fetch_edge_partition_book(
        self, edge_type: Optional[EdgeType] = None
    ) -> Optional[PartitionBook]:
        """
        Fetches the partition book for the specified edge type.

        Args:
            edge_type: The edge type to look up.  Must be ``None`` for
                homogeneous datasets and non-``None`` for heterogeneous ones.

        Returns:
            The partition book for the requested edge type, or ``None`` if
            no partition book is available.
        """
        edge_type = self._infer_edge_type_if_homogeneous_with_label_edges(edge_type)
        return request_server(
            0,
            DistServer.get_edge_partition_book,
            edge_type=edge_type,
        )

    def _infer_node_type_if_homogeneous_with_label_edges(
        self, node_type: Optional[NodeType]
    ) -> Optional[NodeType]:
        """
        Auto-infers the default homogeneous node type for homogeneous datasets with label edges.
        """
        if node_type is None:
            node_types = self.fetch_node_types()
            if node_types is not None and DEFAULT_HOMOGENEOUS_NODE_TYPE in node_types:
                node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
                logger.info(
                    f"Auto-inferred default node type {node_type} for homogeneous dataset with label edges "
                    f"as {DEFAULT_HOMOGENEOUS_NODE_TYPE} is in the node types: {node_types}"
                )
        return node_type

    def _infer_edge_type_if_homogeneous_with_label_edges(
        self, edge_type: Optional[EdgeType]
    ) -> Optional[EdgeType]:
        """
        Auto-infers the default homogeneous edge type for homogeneous datasets with label edges.
        """
        if edge_type is None:
            edge_types = self.fetch_edge_types()
            if edge_types is not None and DEFAULT_HOMOGENEOUS_EDGE_TYPE in edge_types:
                edge_type = DEFAULT_HOMOGENEOUS_EDGE_TYPE
                logger.info(
                    f"Auto-inferred default edge type {edge_type} for homogeneous dataset with label edges "
                    f"as {DEFAULT_HOMOGENEOUS_EDGE_TYPE} is in the edge types: {edge_types}"
                )
        return edge_type

    def _fetch_node_ids(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        node_type: Optional[NodeType] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
    ) -> dict[int, torch.Tensor]:
        """Fetches node ids from the storage nodes for the current compute node (machine)."""
        futures: list[torch.futures.Future[torch.Tensor]] = []
        node_type = self._infer_node_type_if_homogeneous_with_label_edges(node_type)

        logger.info(
            f"Getting node ids for rank {rank} / {world_size} with node type {node_type} and split {split}"
        )

        for server_rank in range(self.cluster_info.num_storage_nodes):
            futures.append(
                async_request_server(
                    server_rank,
                    DistServer.get_node_ids,
                    FetchNodesRequest(
                        rank=rank,
                        world_size=world_size,
                        split=split,
                        node_type=node_type,
                    ),
                )
            )
            node_ids = torch.futures.wait_all(futures)
        return {server_rank: node_ids for server_rank, node_ids in enumerate(node_ids)}

    def _fetch_node_ids_by_server(
        self,
        rank: int,
        world_size: int,
        node_type: Optional[NodeType] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
    ) -> dict[int, torch.Tensor]:
        """Fetches node ids using contiguous server assignment.

        Each compute node is assigned a contiguous range of servers. Only
        assigned servers are RPCed; unassigned servers get empty tensors.
        Boundary servers are sliced fractionally when servers don't divide
        evenly across compute nodes.

        Args:
            rank: The rank of the compute node requesting node ids.
            world_size: The total number of compute nodes.
            node_type: The type of nodes to get. Must be provided for heterogeneous datasets.
            split: The split of the dataset to get node ids from.

        Returns:
            A dict mapping every server rank to a tensor of node ids.
        """
        node_type = self._infer_node_type_if_homogeneous_with_label_edges(node_type)

        assignments = compute_server_assignments(
            num_servers=self.cluster_info.num_storage_nodes,
            num_compute_nodes=world_size,
            compute_rank=rank,
        )

        logger.info(
            f"Getting node ids via CONTIGUOUS strategy for rank {rank} / {world_size} "
            f"with node type {node_type} and split {split}. "
            f"Assigned servers: {list(assignments.keys())}"
        )

        # RPC only assigned servers (fetch ALL nodes, no server-side sharding)
        futures: dict[int, torch.futures.Future[torch.Tensor]] = {}
        for server_rank in assignments:
            futures[server_rank] = async_request_server(
                server_rank,
                DistServer.get_node_ids,
                FetchNodesRequest(
                    rank=None,
                    world_size=None,
                    split=split,
                    node_type=node_type,
                ),
            )

        # Build result: slice assigned servers, empty tensors for unassigned
        result: dict[int, torch.Tensor] = {}
        for server_rank in range(self.cluster_info.num_storage_nodes):
            if server_rank in futures:
                all_nodes = futures[server_rank].wait()
                result[server_rank] = assignments[server_rank].slice_tensor(all_nodes)
            else:
                result[server_rank] = torch.empty(0, dtype=torch.long)

        return result

    def fetch_node_ids(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        node_type: Optional[NodeType] = None,
        shard_strategy: ShardStrategy = ShardStrategy.ROUND_ROBIN,
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
            shard_strategy (ShardStrategy): Strategy for sharding node IDs across compute nodes.
                ``ROUND_ROBIN`` (default) shards each server's nodes across all compute nodes.
                ``CONTIGUOUS`` assigns entire servers to compute nodes, producing empty tensors
                for unassigned servers. ``CONTIGUOUS`` requires both ``rank`` and ``world_size``.

        Raises:
            ValueError: If ``shard_strategy`` is ``CONTIGUOUS`` but ``rank`` or ``world_size`` is None.

        Returns:
            dict[int, torch.Tensor]: A dict mapping storage rank to node ids.

        Examples:
            See :class:`~gigl.distributed.utils.neighborloader.ShardStrategy` for
            concrete examples of how each strategy distributes node IDs across
            compute nodes.

        Note:
            When `split=None`, all nodes are queryable. This means nodes from any split
            (train, val, or test) may be returned. This is useful when you need to sample
            neighbors during inference, as neighbor nodes may belong to any split.
        """

        if shard_strategy == ShardStrategy.CONTIGUOUS:
            if rank is None or world_size is None:
                raise ValueError(
                    "Both rank and world_size must be provided when using "
                    f"ShardStrategy.CONTIGUOUS. Got rank={rank}, world_size={world_size}"
                )

        if shard_strategy == ShardStrategy.CONTIGUOUS:
            assert rank is not None and world_size is not None
            return self._fetch_node_ids_by_server(rank, world_size, node_type, split)
        return self._fetch_node_ids(rank, world_size, node_type, split)

    def fetch_free_ports_on_storage_cluster(self, num_ports: int) -> list[int]:
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

    def _fetch_ablp_input(
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
                    FetchABLPInputRequest(
                        split=split,
                        rank=rank,
                        world_size=world_size,
                        node_type=node_type,
                        supervision_edge_type=supervision_edge_type,
                    ),
                )
            )
            ablp_inputs = torch.futures.wait_all(futures)
        return {
            server_rank: ablp_input
            for server_rank, ablp_input in enumerate(ablp_inputs)
        }

    def _fetch_ablp_input_by_server(
        self,
        split: Literal["train", "val", "test"],
        rank: int,
        world_size: int,
        node_type: NodeType = DEFAULT_HOMOGENEOUS_NODE_TYPE,
        supervision_edge_type: EdgeType = DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Fetches ABLP input using contiguous server assignment.

        Each compute node is assigned a contiguous range of servers. Only
        assigned servers are RPCed; unassigned servers get empty tensors.
        Boundary servers are sliced fractionally when servers don't divide
        evenly across compute nodes.

        Args:
            split: The split of the dataset to get ABLP input from.
            rank: The rank of the compute node requesting ABLP input.
            world_size: The total number of compute nodes.
            node_type: The type of anchor nodes to retrieve.
            supervision_edge_type: The edge type for supervision.

        Returns:
            A dict mapping every server rank to a tuple of
            (anchors, positive_labels, negative_labels).
        """
        assignments = compute_server_assignments(
            num_servers=self.cluster_info.num_storage_nodes,
            num_compute_nodes=world_size,
            compute_rank=rank,
        )

        logger.info(
            f"Getting ABLP input via CONTIGUOUS strategy for rank {rank} / {world_size} "
            f"with node type {node_type}, split {split}, and "
            f"supervision edge type {supervision_edge_type}. "
            f"Assigned servers: {list(assignments.keys())}"
        )

        # RPC only assigned servers (fetch ALL ABLP data, no server-side sharding)
        futures: dict[
            int,
            torch.futures.Future[
                tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            ],
        ] = {}
        for server_rank in assignments:
            futures[server_rank] = async_request_server(
                server_rank,
                DistServer.get_ablp_input,
                FetchABLPInputRequest(
                    split=split,
                    rank=None,
                    world_size=None,
                    node_type=node_type,
                    supervision_edge_type=supervision_edge_type,
                ),
            )

        # Build result: slice assigned servers, empty tensors for unassigned
        result: dict[
            int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
        ] = {}
        for server_rank in range(self.cluster_info.num_storage_nodes):
            if server_rank in futures:
                anchors, positive_labels, negative_labels = futures[server_rank].wait()
                server_slice = assignments[server_rank]
                sliced_anchors = server_slice.slice_tensor(anchors)
                sliced_positive = server_slice.slice_tensor(positive_labels)
                sliced_negative = (
                    server_slice.slice_tensor(negative_labels)
                    if negative_labels is not None
                    else None
                )
                result[server_rank] = (
                    sliced_anchors,
                    sliced_positive,
                    sliced_negative,
                )
            else:
                result[server_rank] = (
                    torch.empty(0, dtype=torch.long),
                    torch.empty((0, 0), dtype=torch.long),
                    None,
                )

        return result

    # TODO(#488) - support multiple supervision edge types
    def fetch_ablp_input(
        self,
        split: Literal["train", "val", "test"],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        anchor_node_type: Optional[NodeType] = None,
        supervision_edge_type: Optional[EdgeType] = None,
        shard_strategy: ShardStrategy = ShardStrategy.ROUND_ROBIN,
    ) -> dict[int, ABLPInputNodes]:
        """Fetches ABLP (Anchor Based Link Prediction) input from the storage nodes.

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
            shard_strategy (ShardStrategy): Strategy for sharding ABLP input across compute
                nodes. ``ROUND_ROBIN`` (default) shards each server's data across all compute
                nodes. ``CONTIGUOUS`` assigns entire servers to compute nodes, producing empty
                tensors for unassigned servers. ``CONTIGUOUS`` requires both ``rank`` and
                ``world_size``.

        Returns:
            dict[int, ABLPInputNodes]:
                A dict mapping storage rank to an ABLPInputNodes containing:
                - anchor_node_type: The node type of the anchor nodes, or None for labeled homogeneous.
                - anchor_nodes: 1D tensor of anchor node IDs for the split.
                - positive_labels: Dict mapping positive label EdgeType to a 2D tensor [N, M].
                - negative_labels: Optional dict mapping negative label EdgeType to a 2D tensor [N, M].

        Raises:
            ValueError: If ``shard_strategy`` is ``CONTIGUOUS`` but ``rank`` or ``world_size`` is None.

        Examples:
            See :class:`~gigl.distributed.utils.neighborloader.ShardStrategy` for
            concrete examples of how each strategy distributes data across
            compute nodes.

        """

        if shard_strategy == ShardStrategy.CONTIGUOUS:
            if rank is None or world_size is None:
                raise ValueError(
                    "Both rank and world_size must be provided when using "
                    f"ShardStrategy.CONTIGUOUS. Got rank={rank}, world_size={world_size}"
                )

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

        if shard_strategy == ShardStrategy.CONTIGUOUS:
            assert rank is not None and world_size is not None
            raw_inputs = self._fetch_ablp_input_by_server(
                split=split,
                rank=rank,
                world_size=world_size,
                node_type=evaluated_anchor_node_type,
                supervision_edge_type=evaluated_supervision_edge_type,
            )
        else:
            raw_inputs = self._fetch_ablp_input(
                split=split,
                rank=rank,
                world_size=world_size,
                node_type=evaluated_anchor_node_type,
                supervision_edge_type=evaluated_supervision_edge_type,
            )
        return {
            server_rank: ABLPInputNodes(
                anchor_node_type=evaluated_anchor_node_type,
                anchor_nodes=anchors,
                labels={
                    evaluated_supervision_edge_type: (
                        positive_labels,
                        negative_labels,
                    )
                },
            )
            for server_rank, (
                anchors,
                positive_labels,
                negative_labels,
            ) in raw_inputs.items()
        }

    def fetch_edge_types(self) -> Optional[list[EdgeType]]:
        """Fetch the edge types from the registered dataset.

        Returns:
            The edge types in the dataset, None if the dataset is homogeneous.
        """
        return request_server(
            0,
            DistServer.get_edge_types,
        )

    def fetch_node_types(self) -> Optional[list[NodeType]]:
        """Fetch the node types from the registered dataset.

        Returns:
            The node types in the dataset, None if the dataset is homogeneous.
        """
        return request_server(
            0,
            DistServer.get_node_types,
        )
