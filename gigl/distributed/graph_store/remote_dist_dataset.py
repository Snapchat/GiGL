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
from gigl.distributed.graph_store.sharding import (
    ServerSlice,
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

    def _compute_assignments_if_needed(
        self,
        rank: Optional[int],
        world_size: Optional[int],
        shard_strategy: ShardStrategy,
    ) -> Optional[dict[int, ServerSlice]]:
        """Compute contiguous server assignments when that shard strategy is requested.

        Returns ``None`` for ``ROUND_ROBIN``.
        Raises ``ValueError`` for ``CONTIGUOUS`` if rank or world_size is ``None``.
        """
        if shard_strategy != ShardStrategy.CONTIGUOUS:
            return None

        if rank is None or world_size is None:
            raise ValueError(
                "Both rank and world_size must be provided when using "
                f"ShardStrategy.CONTIGUOUS. Got rank={rank}, world_size={world_size}"
            )
        return compute_server_assignments(
            num_servers=self.cluster_info.num_storage_nodes,
            num_compute_nodes=world_size,
            compute_rank=rank,
        )

    def _fetch_node_ids(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        node_type: Optional[NodeType] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        assignments: Optional[dict[int, ServerSlice]] = None,
    ) -> dict[int, torch.Tensor]:
        """Fetches node ids from the storage nodes for the current compute node (machine)."""
        node_type = self._infer_node_type_if_homogeneous_with_label_edges(node_type)

        # Build per-server requests
        requests: dict[int, FetchNodesRequest] = {}
        if assignments is None:
            for server_rank in range(self.cluster_info.num_storage_nodes):
                requests[server_rank] = FetchNodesRequest(
                    rank=rank,
                    world_size=world_size,
                    split=split,
                    node_type=node_type,
                )
        else:
            for server_rank, server_slice in assignments.items():
                requests[server_rank] = FetchNodesRequest(
                    split=split,
                    node_type=node_type,
                    server_slice=server_slice,
                )

        strategy = "CONTIGUOUS" if assignments is not None else "ROUND_ROBIN"
        logger.info(
            f"Fetching node ids via {strategy} for rank {rank} / {world_size} "
            f"with node type {node_type} and split {split}. "
            f"Requesting from servers: {sorted(requests.keys())}"
        )

        # Dispatch all futures
        futures: dict[int, torch.futures.Future[torch.Tensor]] = {
            server_rank: async_request_server(
                server_rank, DistServer.get_node_ids, request
            )
            for server_rank, request in requests.items()
        }

        # Collect results, filling empty tensors for unrequested servers
        return {
            server_rank: futures[server_rank].wait()
            if server_rank in futures
            else torch.empty(0, dtype=torch.long)
            for server_rank in range(self.cluster_info.num_storage_nodes)
        }

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
            rank (Optional[int]): The requested shard rank.
                When `None` with `ROUND_ROBIN`, all data is returned unsharded
                e.g. returns all node ids from all storage nodes.
            world_size (Optional[int]): The requested shard world size.
                When `None` with `ROUND_ROBIN`, all data is returned unsharded
                e.g. returns all node ids from all storage nodes.
            split (Optional[Literal["train", "val", "test"]]):
                The split of the dataset to get node ids from.
                If provided, the dataset must have `train_node_ids`, `val_node_ids`, and `test_node_ids` properties.
            node_type (Optional[NodeType]): The type of nodes to get.
                Must be provided for heterogeneous datasets.
                Must be None for labeled homogeneous graphs.
            shard_strategy (ShardStrategy): Strategy for sharding node IDs across compute nodes.
                See the documentation for `ShardStrategy` for more details.
                `ROUND_ROBIN` (default) is the default strategy.
        Raises:
            ValueError: If `shard_strategy` is `CONTIGUOUS` but `rank` or `world_size` is `None`.

        Returns:
            dict[int, torch.Tensor]: A dict mapping storage rank to node ids.

        Examples:
            See :class:`~gigl.distributed.graph_store.sharding.ShardStrategy` for
            concrete examples of how each strategy distributes node IDs across
            compute nodes.

        Note:
            When `split=None`, all nodes are queryable. This means nodes from any split
            (train, val, or test) may be returned. This is useful when you need to sample
            neighbors during inference, as neighbor nodes may belong to any split.
        """
        assignments = self._compute_assignments_if_needed(
            rank=rank,
            world_size=world_size,
            shard_strategy=shard_strategy,
        )
        return self._fetch_node_ids(
            rank=rank,
            world_size=world_size,
            node_type=node_type,
            split=split,
            assignments=assignments,
        )

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
        assignments: Optional[dict[int, ServerSlice]] = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Fetches ABLP input from the storage nodes for the current compute node (machine)."""
        # Build per-server requests
        requests: dict[int, FetchABLPInputRequest] = {}
        if assignments is None:
            for server_rank in range(self.cluster_info.num_storage_nodes):
                requests[server_rank] = FetchABLPInputRequest(
                    split=split,
                    rank=rank,
                    world_size=world_size,
                    node_type=node_type,
                    supervision_edge_type=supervision_edge_type,
                )
        else:
            for server_rank, server_slice in assignments.items():
                requests[server_rank] = FetchABLPInputRequest(
                    split=split,
                    node_type=node_type,
                    supervision_edge_type=supervision_edge_type,
                    server_slice=server_slice,
                )

        strategy = "CONTIGUOUS" if assignments is not None else "ROUND_ROBIN"
        logger.info(
            f"Fetching ABLP input via {strategy} for rank {rank} / {world_size} "
            f"with node type {node_type}, split {split}, and "
            f"supervision edge type {supervision_edge_type}. "
            f"Requesting from servers: {sorted(requests.keys())}"
        )

        # Dispatch all futures
        futures: dict[
            int,
            torch.futures.Future[
                tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            ],
        ] = {
            server_rank: async_request_server(
                server_rank, DistServer.get_ablp_input, request
            )
            for server_rank, request in requests.items()
        }

        def _empty_ablp_result() -> tuple[
            torch.Tensor, torch.Tensor, Optional[torch.Tensor]
        ]:
            """Return an empty ABLP result tuple: (anchor_nodes, positive_labels, negative_labels)."""
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty((0, 0), dtype=torch.long),
                None,
            )

        # Collect results, filling empty tuples for unrequested servers
        return {
            server_rank: futures[server_rank].wait()
            if server_rank in futures
            else _empty_ablp_result()
            for server_rank in range(self.cluster_info.num_storage_nodes)
        }

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
            rank (Optional[int]): The requested shard rank.
                When `None` with `ROUND_ROBIN`, all data is returned unsharded
                e.g. returns all ABLP input from all storage nodes.
            world_size (Optional[int]): The requested shard world size.
                When `None` with `ROUND_ROBIN`, all data is returned unsharded
                e.g. returns all ABLP input from all storage nodes.
            anchor_node_type (Optional[NodeType]): The type of the anchor nodes to retrieve.
                Must be provided for heterogeneous graphs.
                Must be None for labeled homogeneous graphs.
                Defaults to None.
            supervision_edge_type (Optional[EdgeType]): The edge type for supervision.
                Must be provided for heterogeneous graphs.
                Must be None for labeled homogeneous graphs.
                Defaults to None.
            shard_strategy (ShardStrategy):
                Strategy for sharding ABLP input across compute nodes.
                See the documentation for `ShardStrategy` for more details.
                `ROUND_ROBIN` (default) is the default strategy.

        Returns:
            dict[int, ABLPInputNodes]:
                A dict mapping storage rank to an ABLPInputNodes containing:
                - anchor_node_type: The node type of the anchor nodes, or ``DEFAULT_HOMOGENEOUS_NODE_TYPE`` for labeled homogeneous.
                - anchor_nodes: 1D tensor of anchor node IDs for the split.
                - positive_labels: Dict mapping positive label EdgeType to a 2D tensor [N, M].
                - negative_labels: Optional dict mapping negative label EdgeType to a 2D tensor [N, M].

        Raises:
            ValueError: If ``shard_strategy`` is ``CONTIGUOUS`` but ``rank`` or ``world_size`` is None.

        Examples:
            See :class:`~gigl.distributed.graph_store.sharding.ShardStrategy` for
            concrete examples of how each strategy distributes data across
            compute nodes.

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

        assignments = self._compute_assignments_if_needed(
            rank=rank,
            world_size=world_size,
            shard_strategy=shard_strategy,
        )
        raw_inputs = self._fetch_ablp_input(
            split=split,
            rank=rank,
            world_size=world_size,
            node_type=evaluated_anchor_node_type,
            supervision_edge_type=evaluated_supervision_edge_type,
            assignments=assignments,
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
