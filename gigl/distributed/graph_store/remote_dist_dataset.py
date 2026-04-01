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


def _plan_storage_rank_shards_for_compute_rank(
    rank: int,
    world_size: int,
    num_storage_nodes: int,
    num_assigned_storage_ranks: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, tuple[int, int]]]:
    """Plan which storage ranks a compute rank contacts and its local shard within each.

    Each compute rank is assigned ``num_assigned_storage_ranks`` storage ranks
    using a round-robin scheme starting from an evenly-spaced offset
    (``compute_rank * num_storage_nodes // world_size``).

    The reverse mapping tells us how many compute ranks share each storage rank.
    For the given ``rank``, its local shard within a shared storage rank is its
    position in the sorted list of compute ranks assigned to that storage rank.

    The constraint ``world_size * num_assigned_storage_ranks >= num_storage_nodes``
    guarantees that every storage rank is contacted by at least one compute rank,
    ensuring exact global coverage.

    Args:
        rank: The current compute rank.
        world_size: Total number of compute ranks.
        num_storage_nodes: Total number of storage nodes in the cluster.
        num_assigned_storage_ranks: Number of storage ranks each compute rank contacts.

    Returns:
        A 3-tuple:
            - compute_rank_to_storage_ranks: Maps every compute rank to its assigned storage ranks.
            - storage_rank_to_compute_ranks: Maps every storage rank to the compute ranks that contact it.
            - storage_rank_to_local_shard: For the given ``rank`` only, maps each assigned storage rank
              to ``(shard_index, num_shards)`` where ``shard_index`` is this rank's position among
              the compute ranks sharing that storage rank.

    Raises:
        ValueError: If arguments are out of range or coverage cannot be guaranteed.
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be > 0, received {world_size}")
    if num_storage_nodes <= 0:
        raise ValueError(f"num_storage_nodes must be > 0, received {num_storage_nodes}")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"rank must be in [0, world_size), received rank={rank}, world_size={world_size}"
        )
    if (
        num_assigned_storage_ranks <= 0
        or num_assigned_storage_ranks > num_storage_nodes
    ):
        raise ValueError(
            "num_assigned_storage_ranks must be in [1, num_storage_nodes], "
            f"received num_assigned_storage_ranks={num_assigned_storage_ranks}, num_storage_nodes={num_storage_nodes}"
        )
    if world_size * num_assigned_storage_ranks < num_storage_nodes:
        raise ValueError(
            "world_size * num_assigned_storage_ranks must be >= num_storage_nodes "
            "to guarantee all storage nodes are sampled from. "
            f"Received world_size={world_size}, num_assigned_storage_ranks={num_assigned_storage_ranks}, "
            f"num_storage_nodes={num_storage_nodes}"
        )

    compute_rank_to_storage_ranks: dict[int, list[int]] = {}
    for compute_rank in range(world_size):
        start_storage_rank = (compute_rank * num_storage_nodes) // world_size
        compute_rank_to_storage_ranks[compute_rank] = [
            (start_storage_rank + offset) % num_storage_nodes
            for offset in range(num_assigned_storage_ranks)
        ]

    storage_rank_to_compute_ranks: dict[int, list[int]] = {
        storage_rank: [] for storage_rank in range(num_storage_nodes)
    }
    for compute_rank, storage_ranks in compute_rank_to_storage_ranks.items():
        for storage_rank in storage_ranks:
            storage_rank_to_compute_ranks[storage_rank].append(compute_rank)

    storage_rank_to_local_shard: dict[int, tuple[int, int]] = {}
    for storage_rank in compute_rank_to_storage_ranks[rank]:
        assigned_compute_ranks = storage_rank_to_compute_ranks[storage_rank]
        storage_rank_to_local_shard[storage_rank] = (
            assigned_compute_ranks.index(rank),
            len(assigned_compute_ranks),
        )

    return (
        compute_rank_to_storage_ranks,
        storage_rank_to_compute_ranks,
        storage_rank_to_local_shard,
    )


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
        num_assigned_storage_ranks: Optional[int] = None,
    ) -> dict[int, torch.Tensor]:
        """Fetches node ids from the storage nodes for the current compute node (machine)."""
        requested_storage_ranks: list[int]
        requests: list[FetchNodesRequest] = []

        if num_assigned_storage_ranks is not None:
            if rank is None or world_size is None:
                raise ValueError(
                    "num_assigned_storage_ranks requires rank and world_size. "
                    f"Received rank={rank}, world_size={world_size}, "
                    f"num_assigned_storage_ranks={num_assigned_storage_ranks}"
                )
            (
                compute_rank_to_storage_ranks,
                _,
                storage_rank_to_local_shard,
            ) = _plan_storage_rank_shards_for_compute_rank(
                rank=rank,
                world_size=world_size,
                num_storage_nodes=self.cluster_info.num_storage_nodes,
                num_assigned_storage_ranks=num_assigned_storage_ranks,
            )
            requested_storage_ranks = compute_rank_to_storage_ranks[rank]
        else:
            requested_storage_ranks = list(range(self.cluster_info.num_storage_nodes))

        node_type = self._infer_node_type_if_homogeneous_with_label_edges(node_type)

        logger.info(
            f"Getting node ids for rank {rank} / {world_size} with node type {node_type}, "
            f"split {split}, and num_assigned_storage_ranks {num_assigned_storage_ranks}"
        )

        if num_assigned_storage_ranks is None:
            for storage_rank in requested_storage_ranks:
                request = FetchNodesRequest(
                    rank=rank,
                    world_size=world_size,
                    split=split,
                    node_type=node_type,
                )
                request.validate()
                requests.append(request)
        else:
            for storage_rank in requested_storage_ranks:
                shard_index, num_shards = storage_rank_to_local_shard[storage_rank]
                request = FetchNodesRequest(
                    split=split,
                    node_type=node_type,
                    rank=shard_index,
                    world_size=num_shards,
                )
                request.validate()
                requests.append(request)

        futures: list[torch.futures.Future[torch.Tensor]] = []
        for storage_rank, request in zip(requested_storage_ranks, requests):
            futures.append(
                async_request_server(
                    storage_rank,
                    DistServer.get_node_ids,
                    request,
                )
            )

        node_ids = torch.futures.wait_all(futures)
        return {
            storage_rank: node_id
            for storage_rank, node_id in zip(requested_storage_ranks, node_ids)
        }

    def fetch_node_ids(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        node_type: Optional[NodeType] = None,
        num_assigned_storage_ranks: Optional[int] = None,
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
            num_assigned_storage_ranks (Optional[int]): If provided, limit this compute rank
                to exactly this many storage ranks while preserving exact global coverage.
                Requires ``rank`` and ``world_size``.

                Must satisfy ``world_size * num_assigned_storage_ranks >= num_storage_nodes``
                to guarantee all storage nodes are sampled from.

                Typical values are 1-4. Lower values reduce cross-cluster network fanout
                at the cost of potentially less balanced data distribution per compute rank.
                ``None`` (the default) contacts all storage nodes.

        Returns:
            dict[int, torch.Tensor]: A dict mapping storage rank to node ids. When
                ``num_assigned_storage_ranks`` is set, only the assigned storage
                ranks are returned.

        Examples:
            Suppose we have 2 storage nodes and 2 compute nodes, with 16 total nodes.
            Nodes are partitioned across storage nodes, with splits defined as:

                Storage rank 0: [0, 1, 2, 3, 4, 5, 6, 7]
                    train=[0, 1, 2, 3], val=[4, 5], test=[6, 7]
                Storage rank 1: [8, 9, 10, 11, 12, 13, 14, 15]
                    train=[8, 9, 10, 11], val=[12, 13], test=[14, 15]

            Get all nodes (no split filtering, no sharding):

            >>> dataset.fetch_node_ids()
            {
                0: tensor([0, 1, 2, 3, 4, 5, 6, 7]),      # All 8 nodes from storage rank 0
                1: tensor([8, 9, 10, 11, 12, 13, 14, 15]) # All 8 nodes from storage rank 1
            }

            Shard all nodes across 2 compute nodes (compute rank 0 gets first half from each storage):

            >>> dataset.fetch_node_ids(rank=0, world_size=2)
            {
                0: tensor([0, 1, 2, 3]),   # First 4 of all 8 nodes from storage rank 0
                1: tensor([8, 9, 10, 11])  # First 4 of all 8 nodes from storage rank 1
            }

            Get only training nodes (no sharding):

            >>> dataset.fetch_node_ids(split="train")
            {
                0: tensor([0, 1, 2, 3]),   # 4 training nodes from storage rank 0
                1: tensor([8, 9, 10, 11])  # 4 training nodes from storage rank 1
            }

            Combine split and sharding (training nodes, sharded for compute rank 0):

            >>> dataset.fetch_node_ids(rank=0, world_size=2, split="train")
            {
                0: tensor([0, 1]),  # First 2 of 4 training nodes from storage rank 0
                1: tensor([8, 9])   # First 2 of 4 training nodes from storage rank 1
            }

            Limit each compute rank to 1 assigned storage rank (4 compute ranks, 2 storage nodes).
            Ranks 0-1 are assigned to storage 0; ranks 2-3 are assigned to storage 1.
            Each storage node's data is contiguously split among its assigned compute ranks:

            >>> dataset.fetch_node_ids(rank=0, world_size=4, num_assigned_storage_ranks=1)
            {
                0: tensor([0, 1, 2, 3])       # Storage 0, shard 0 of 2
            }
            >>> dataset.fetch_node_ids(rank=1, world_size=4, num_assigned_storage_ranks=1)
            {
                0: tensor([4, 5, 6, 7])       # Storage 0, shard 1 of 2
            }
            >>> dataset.fetch_node_ids(rank=2, world_size=4, num_assigned_storage_ranks=1)
            {
                1: tensor([8, 9, 10, 11])     # Storage 1, shard 0 of 2
            }
            >>> dataset.fetch_node_ids(rank=3, world_size=4, num_assigned_storage_ranks=1)
            {
                1: tensor([12, 13, 14, 15])   # Storage 1, shard 1 of 2
            }

        Note:
            When `split=None`, all nodes are queryable. This means nodes from any split
            (train, val, or test) may be returned. This is useful when you need to sample
            neighbors during inference, as neighbor nodes may belong to any split.
        """
        return self._fetch_node_ids(
            rank=rank,
            world_size=world_size,
            node_type=node_type,
            split=split,
            num_assigned_storage_ranks=num_assigned_storage_ranks,
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
        num_assigned_storage_ranks: Optional[int] = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Fetches ABLP input from the storage nodes for the current compute node (machine)."""
        requested_storage_ranks: list[int]
        requests: list[FetchABLPInputRequest] = []

        if num_assigned_storage_ranks is not None:
            if rank is None or world_size is None:
                raise ValueError(
                    "num_assigned_storage_ranks requires rank and world_size. "
                    f"Received rank={rank}, world_size={world_size}, "
                    f"num_assigned_storage_ranks={num_assigned_storage_ranks}"
                )
            (
                compute_rank_to_storage_ranks,
                _,
                storage_rank_to_local_shard,
            ) = _plan_storage_rank_shards_for_compute_rank(
                rank=rank,
                world_size=world_size,
                num_storage_nodes=self.cluster_info.num_storage_nodes,
                num_assigned_storage_ranks=num_assigned_storage_ranks,
            )
            requested_storage_ranks = compute_rank_to_storage_ranks[rank]
        else:
            requested_storage_ranks = list(range(self.cluster_info.num_storage_nodes))

        logger.info(
            f"Getting ABLP input for rank {rank} / {world_size} with node type {node_type}, "
            f"split {split}, supervision edge type {supervision_edge_type}, "
            f"and num_assigned_storage_ranks {num_assigned_storage_ranks}"
        )

        if num_assigned_storage_ranks is None:
            for storage_rank in requested_storage_ranks:
                request = FetchABLPInputRequest(
                    split=split,
                    rank=rank,
                    world_size=world_size,
                    node_type=node_type,
                    supervision_edge_type=supervision_edge_type,
                )
                request.validate()
                requests.append(request)
        else:
            for storage_rank in requested_storage_ranks:
                shard_index, num_shards = storage_rank_to_local_shard[storage_rank]
                request = FetchABLPInputRequest(
                    split=split,
                    node_type=node_type,
                    supervision_edge_type=supervision_edge_type,
                    rank=shard_index,
                    world_size=num_shards,
                )
                request.validate()
                requests.append(request)

        futures: list[
            torch.futures.Future[
                tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            ]
        ] = []
        for storage_rank, request in zip(requested_storage_ranks, requests):
            futures.append(
                async_request_server(
                    storage_rank,
                    DistServer.get_ablp_input,
                    request,
                )
            )

        ablp_inputs = torch.futures.wait_all(futures)
        return {
            storage_rank: ablp_input
            for storage_rank, ablp_input in zip(requested_storage_ranks, ablp_inputs)
        }

    # TODO(#488) - support multiple supervision edge types
    def fetch_ablp_input(
        self,
        split: Literal["train", "val", "test"],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        anchor_node_type: Optional[NodeType] = None,
        supervision_edge_type: Optional[EdgeType] = None,
        num_assigned_storage_ranks: Optional[int] = None,
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
            num_assigned_storage_ranks (Optional[int]): If provided, limit this compute
                rank to exactly this many storage ranks while preserving exact global
                coverage. Requires ``rank`` and ``world_size``.

        Returns:
            dict[int, ABLPInputNodes]:
                A dict mapping storage rank to an ABLPInputNodes containing:
                - anchor_node_type: The node type of the anchor nodes, or None for labeled homogeneous.
                - anchor_nodes: 1D tensor of anchor node IDs for the split.
                - positive_labels: Dict mapping positive label EdgeType to a 2D tensor [N, M].
                - negative_labels: Optional dict mapping negative label EdgeType to a 2D tensor [N, M].
                When ``num_assigned_storage_ranks`` is set, only the assigned storage
                ranks are returned.

        Examples:
            Suppose we have 1 storage node with users [0, 1, 2, 3, 4] where:
                train=[0, 1, 2], val=[3], test=[4]
            And positive/negative labels defined for link prediction.

            Get training ABLP input (heterogeneous):

            >>> dataset.fetch_ablp_input(split="train", node_type=USER, supervision_edge_type=USER_TO_ITEM)
            {
                0: ABLPInputNodes(
                    anchor_nodes=tensor([0, 1, 2]),
                    positive_labels={("user", "to_positive", "item"): tensor([[0, 1], [1, 2], [2, 3]])},
                    anchor_node_type="user",
                    negative_labels={("user", "to_negative", "item"): tensor([[2], [3], [4]])},
                )
            }

            For labeled homogeneous graphs, anchor_node_type will be DEFAULT_HOMOGENEOUS_NODE_TYPE.

            Limit each compute rank to 1 assigned storage rank (2 compute ranks, 2 storage nodes).
            Suppose we have 2 storage nodes with train anchors and labels:

                Storage rank 0: train anchors=[0, 1, 2]
                Storage rank 1: train anchors=[3, 4]

            >>> dataset.fetch_ablp_input(
            ...     split="train", rank=0, world_size=2,
            ...     anchor_node_type=USER, supervision_edge_type=USER_TO_ITEM,
            ...     num_assigned_storage_ranks=1,
            ... )
            {
                0: ABLPInputNodes(
                    anchor_nodes=tensor([0, 1, 2]),
                    labels={("user", "to_positive", "item"): (pos_labels, neg_labels)},
                    anchor_node_type="user",
                )
            }
            >>> dataset.fetch_ablp_input(
            ...     split="train", rank=1, world_size=2,
            ...     anchor_node_type=USER, supervision_edge_type=USER_TO_ITEM,
            ...     num_assigned_storage_ranks=1,
            ... )
            {
                1: ABLPInputNodes(
                    anchor_nodes=tensor([3, 4]),
                    labels={("user", "to_positive", "item"): (pos_labels, neg_labels)},
                    anchor_node_type="user",
                )
            }

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

        raw_inputs = self._fetch_ablp_input(
            split=split,
            rank=rank,
            world_size=world_size,
            node_type=evaluated_anchor_node_type,
            supervision_edge_type=evaluated_supervision_edge_type,
            num_assigned_storage_ranks=num_assigned_storage_ranks,
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
