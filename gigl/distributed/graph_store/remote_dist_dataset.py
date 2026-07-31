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
    compute_server_assignments,
)
from gigl.distributed.utils.networking import get_free_ports
from gigl.env.distributed import GraphStoreInfo
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureInfo,
    reverse_edge_type,
    select_label_edge_types,
)
from gigl.utils.sampling import ABLPInputNodes

logger = Logger()


def _resolve_registered_supervision_edge_types(
    supervision_edge_types: tuple[EdgeType, ...],
    anchor_node_type: NodeType,
    edge_dir: Literal["in", "out"],
    registered_edge_types: list[EdgeType],
) -> tuple[
    tuple[EdgeType, ...],
    set[EdgeType],
]:
    """Resolve ABLP supervision types to registered sampling orientation.

    Args:
        supervision_edge_types: Ordered supervision types to resolve.
        anchor_node_type: Node type of the anchor nodes.
        edge_dir: Registered graph sampling direction.
        registered_edge_types: Edge types registered in GraphStore.

    Returns:
        The ordered registered supervision types and the subset with negative
        label topology.

    Raises:
        ValueError: If the direction, tuple, orientation, or registered label
            topology is invalid.
    """
    if edge_dir not in ("in", "out"):
        raise ValueError(f"Expected edge_dir to be 'in' or 'out', got {edge_dir!r}.")
    if not supervision_edge_types:
        raise ValueError("supervision_edge_types must not be empty.")
    if len(set(supervision_edge_types)) != len(supervision_edge_types):
        raise ValueError("supervision_edge_types must not contain duplicates.")

    all_anchor_outward = all(
        edge_type[0] == anchor_node_type for edge_type in supervision_edge_types
    )
    if edge_dir == "out":
        if not all_anchor_outward:
            raise ValueError(
                "For edge_dir='out', every supervision edge type must have "
                f"source {anchor_node_type!r}."
            )
        resolved_edge_types = supervision_edge_types
    else:
        all_legacy_incoming = all(
            edge_type[2] == anchor_node_type for edge_type in supervision_edge_types
        )
        if all_anchor_outward:
            resolved_edge_types = tuple(
                reverse_edge_type(edge_type) for edge_type in supervision_edge_types
            )
        elif all_legacy_incoming:
            resolved_edge_types = supervision_edge_types
        else:
            raise ValueError(
                "For edge_dir='in', supervision edge types must uniformly be "
                "anchor-outward or uniformly have the anchor as destination."
            )

    edge_types_with_negatives: set[EdgeType] = set()
    for edge_type in resolved_edge_types:
        _, negative_label_edge_type = select_label_edge_types(
            edge_type,
            registered_edge_types,
        )
        if negative_label_edge_type is not None:
            edge_types_with_negatives.add(edge_type)
    return resolved_edge_types, edge_types_with_negatives


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
    ) -> Optional[dict[int, ServerSlice]]:
        """Compute contiguous server assignments when rank and world_size are provided.

        Returns ``None`` when both ``rank`` and ``world_size`` are ``None``,
        meaning all data should be fetched unsharded.

        Raises:
            ValueError: If only one of ``rank`` or ``world_size`` is provided.
        """
        if rank is None and world_size is None:
            return None

        if rank is None or world_size is None:
            raise ValueError(
                "Both rank and world_size must be provided together, or both "
                f"must be None. Got rank={rank}, world_size={world_size}"
            )
        return compute_server_assignments(
            num_servers=self.cluster_info.num_storage_nodes,
            num_compute_nodes=world_size,
            compute_rank=rank,
        )

    def _fetch_node_ids(
        self,
        node_type: Optional[NodeType] = None,
        split: Optional[Literal["train", "val", "test"]] = None,
        assignments: Optional[dict[int, ServerSlice]] = None,
    ) -> dict[int, torch.Tensor]:
        """Fetches node ids from the storage nodes for the current compute node (machine)."""
        node_type = self._infer_node_type_if_homogeneous_with_label_edges(node_type)

        # Build per-server requests
        requests: dict[int, FetchNodesRequest] = {}
        if assignments is None:
            # No assignments means fetch all data from all servers (unsharded).
            for server_rank in range(self.cluster_info.num_storage_nodes):
                requests[server_rank] = FetchNodesRequest(
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

        sharded = assignments is not None
        logger.info(
            f"Fetching node ids (sharded={sharded}) "
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
    ) -> dict[int, torch.Tensor]:
        """Fetch node ids from the storage nodes for the current compute node (machine).

        The returned dict maps storage rank to the node ids stored on that storage node,
        filtered and sharded according to the provided arguments.

        Storage servers are assigned to compute nodes in contiguous blocks.
        Each compute node fetches all data from its assigned server(s) and receives
        empty tensors for unassigned ones.
        When both ``rank`` and ``world_size`` are ``None``, all data is returned
        unsharded from every storage server.

        Args:
            rank: The compute rank requesting data.
                When ``None`` (together with ``world_size``), all data is
                returned unsharded from all storage nodes.
            world_size: The total number of compute processes.
                When ``None`` (together with ``rank``), all data is
                returned unsharded from all storage nodes.
            split: The split of the dataset to get node ids from.
                If provided, the dataset must have ``train_node_ids``,
                ``val_node_ids``, and ``test_node_ids`` properties.
            node_type: The type of nodes to get.
                Must be provided for heterogeneous datasets.
                Must be ``None`` for labeled homogeneous graphs.

        Raises:
            ValueError: If only one of ``rank`` or ``world_size`` is provided.

        Returns:
            A dict mapping storage rank to node ids.

        Example:
            Suppose we have 2 storage nodes and 2 compute nodes, with 16 total
            nodes. Nodes are partitioned across storage nodes, with splits
            defined as::

                Storage rank 0: [0, 1, 2, 3, 4, 5, 6, 7]
                    train=[0, 1, 2, 3], val=[4, 5], test=[6, 7]
                Storage rank 1: [8, 9, 10, 11, 12, 13, 14, 15]
                    train=[8, 9, 10, 11], val=[12, 13], test=[14, 15]

            Get all nodes (no split filtering, no sharding)::

                >>> dataset.fetch_node_ids()
                {
                    0: tensor([0, 1, 2, 3, 4, 5, 6, 7]),
                    1: tensor([8, 9, 10, 11, 12, 13, 14, 15]),
                }

            Shard training nodes across 2 compute nodes (contiguous — each rank
            gets entire servers)::

                >>> dataset.fetch_node_ids(rank=0, world_size=2, split="train")
                {
                    0: tensor([0, 1, 2, 3]),  # All training nodes from storage 0
                    1: tensor([]),             # Nothing from storage 1
                }
                >>> dataset.fetch_node_ids(rank=1, world_size=2, split="train")
                {
                    0: tensor([]),             # Nothing from storage 0
                    1: tensor([8, 9, 10, 11]), # All training nodes from storage 1
                }

            With 3 storage nodes and 2 compute nodes, server 1 is fractionally
            split::

                >>> dataset.fetch_node_ids(rank=0, world_size=2, split="train")
                {
                    0: tensor([0, 1, 2, 3]),  # All of storage 0
                    1: tensor([8, 9]),         # First half of storage 1
                    2: tensor([]),             # Nothing from storage 2
                }

        Note:
            When ``split=None``, all nodes are queryable. This means nodes from
            any split (train, val, or test) may be returned. This is useful when
            you need to sample neighbors during inference, as neighbor nodes may
            belong to any split.
        """
        assignments = self._compute_assignments_if_needed(
            rank=rank,
            world_size=world_size,
        )
        return self._fetch_node_ids(
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
        node_type: NodeType,
        supervision_edge_types: tuple[EdgeType, ...],
        supervision_edge_types_with_negatives: set[EdgeType],
        assignments: Optional[dict[int, ServerSlice]],
    ) -> dict[int, ABLPInputNodes]:
        """Fetches ABLP input from the storage nodes for the current compute node (machine)."""
        # Build per-server requests
        requests: dict[int, FetchABLPInputRequest] = {}
        if assignments is None:
            # No assignments means fetch all data from all servers (unsharded).
            for server_rank in range(self.cluster_info.num_storage_nodes):
                requests[server_rank] = FetchABLPInputRequest(
                    split=split,
                    node_type=node_type,
                    supervision_edge_types=supervision_edge_types,
                )
        else:
            for server_rank, server_slice in assignments.items():
                requests[server_rank] = FetchABLPInputRequest(
                    split=split,
                    node_type=node_type,
                    supervision_edge_types=supervision_edge_types,
                    server_slice=server_slice,
                )

        sharded = assignments is not None
        logger.info(
            f"Fetching ABLP input (sharded={sharded}) "
            f"with node type {node_type}, split {split}, and "
            f"registered supervision edge types {supervision_edge_types}. "
            f"Requesting from servers: {sorted(requests.keys())}"
        )

        # Dispatch all futures
        futures: dict[int, torch.futures.Future[ABLPInputNodes]] = {
            server_rank: async_request_server(
                server_rank, DistServer.get_ablp_input, request
            )
            for server_rank, request in requests.items()
        }

        def _empty_ablp_result() -> ABLPInputNodes:
            """Return schema-complete empty ABLP input for an unassigned server."""
            return ABLPInputNodes(
                anchor_nodes=torch.empty(0, dtype=torch.long),
                anchor_node_type=node_type,
                labels={
                    edge_type: (
                        torch.empty((0, 0), dtype=torch.long),
                        torch.empty((0, 0), dtype=torch.long)
                        if edge_type in supervision_edge_types_with_negatives
                        else None,
                    )
                    for edge_type in supervision_edge_types
                },
            )

        # Collect results, filling schema-complete inputs for unrequested servers.
        return {
            server_rank: futures[server_rank].wait()
            if server_rank in futures
            else _empty_ablp_result()
            for server_rank in range(self.cluster_info.num_storage_nodes)
        }

    def fetch_ablp_input(
        self,
        split: Literal["train", "val", "test"],
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        anchor_node_type: Optional[NodeType] = None,
        supervision_edge_type: Optional[Union[EdgeType, list[EdgeType]]] = None,
    ) -> dict[int, ABLPInputNodes]:
        """Fetch ABLP (Anchor Based Link Prediction) input from the storage nodes.

        The returned dict maps storage rank to an :class:`ABLPInputNodes` dataclass
        for that storage node. If (rank, world_size) is provided, the input will be
        sharded across the compute nodes using contiguous server assignments.
        If both are ``None``, the input will be returned unsharded for all storage nodes.

        The ``ABLPInputNodes`` dataclass carries explicit node type information and
        keys the label tensors by their label ``EdgeType``, making it unambiguous which
        node types the positive/negative labels correspond to.

        Args:
            split: The split to get the input for.
            rank: The compute rank requesting data.
                When ``None`` (together with ``world_size``), all data is
                returned unsharded from all storage nodes.
            world_size: The total number of compute processes.
                When ``None`` (together with ``rank``), all data is
                returned unsharded from all storage nodes.
            anchor_node_type: The type of the anchor nodes to retrieve.
                Must be provided for heterogeneous graphs.
                Must be ``None`` for labeled homogeneous graphs.
            supervision_edge_type: One or more edge types for supervision.
                Must be provided for heterogeneous graphs.
                Must be ``None`` for labeled homogeneous graphs.

        Returns:
            A dict mapping storage rank to an :class:`ABLPInputNodes` containing:

            - ``anchor_node_type``: The node type of the anchor nodes, or
              ``DEFAULT_HOMOGENEOUS_NODE_TYPE`` for labeled homogeneous.
            - ``anchor_nodes``: 1D tensor of anchor node IDs for the split.
            - ``positive_labels``: Dict mapping positive label EdgeType to a 2D tensor [N, M].
            - ``negative_labels``: Optional dict mapping negative label EdgeType to a 2D tensor [N, M].

        Raises:
            ValueError: If rank/world size or anchor/supervision type arguments
                are not paired, or if registered direction/label topology is
                invalid.

        Note:
            Edge direction and registered edge types are fetched once per call.
            ABLP payloads are fetched once per selected storage server.

        Example:
            Suppose we have 2 storage nodes and 2 compute nodes.
            Storage rank 0 has anchor nodes [0, 1, 2] (train), storage rank 1
            has anchor nodes [3, 4, 5] (train), with positive/negative labels
            for link prediction.

            Shard training ABLP input across 2 compute nodes (contiguous — each
            rank gets entire servers)::

                >>> dataset.fetch_ablp_input(split="train", rank=0, world_size=2)
                {
                    0: ABLPInputNodes(
                        anchor_nodes=tensor([0, 1, 2]),
                        labels={...},
                    ),
                    1: ABLPInputNodes(
                        anchor_nodes=tensor([]),
                        labels={...},
                    ),
                }
                >>> dataset.fetch_ablp_input(split="train", rank=1, world_size=2)
                {
                    0: ABLPInputNodes(
                        anchor_nodes=tensor([]),
                        labels={...},
                    ),
                    1: ABLPInputNodes(
                        anchor_nodes=tensor([3, 4, 5]),
                        labels={...},
                    ),
                }

            With 3 storage nodes and 2 compute nodes, server 1 is fractionally
            split. Storage rank 0 has anchors [0, 1], rank 1 has [2, 3],
            rank 2 has [4, 5]::

                >>> dataset.fetch_ablp_input(split="train", rank=0, world_size=2)
                {
                    0: ABLPInputNodes(
                        anchor_nodes=tensor([0, 1]),
                        labels={...},
                    ),
                    1: ABLPInputNodes(
                        anchor_nodes=tensor([2]),    # First half of storage 1
                        labels={...},
                    ),
                    2: ABLPInputNodes(
                        anchor_nodes=tensor([]),     # Nothing from storage 2
                        labels={...},
                    ),
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
            evaluated_supervision_edge_types = (DEFAULT_HOMOGENEOUS_EDGE_TYPE,)
        elif isinstance(supervision_edge_type, list):
            evaluated_supervision_edge_types = tuple(supervision_edge_type)
        else:
            evaluated_supervision_edge_types = (supervision_edge_type,)
        del anchor_node_type, supervision_edge_type

        if not evaluated_supervision_edge_types:
            raise ValueError("supervision_edge_type must be a non-empty list.")
        if len(set(evaluated_supervision_edge_types)) != len(
            evaluated_supervision_edge_types
        ):
            raise ValueError("supervision_edge_type must not contain duplicates.")
        if (
            evaluated_anchor_node_type == DEFAULT_HOMOGENEOUS_NODE_TYPE
            and evaluated_supervision_edge_types != (DEFAULT_HOMOGENEOUS_EDGE_TYPE,)
        ):
            raise ValueError(
                "Labeled homogeneous GraphStore input supports only the default "
                "homogeneous supervision edge type."
            )

        assignments = self._compute_assignments_if_needed(
            rank=rank,
            world_size=world_size,
        )
        # These two metadata lookups happen once per public fetch. ABLP payload
        # requests remain one per selected storage server.
        edge_dir = self.fetch_edge_dir()
        if edge_dir == "in":
            evaluated_edge_dir: Literal["in", "out"] = "in"
        elif edge_dir == "out":
            evaluated_edge_dir = "out"
        else:
            raise ValueError(
                f"Expected GraphStore edge_dir to be 'in' or 'out', got {edge_dir!r}."
            )
        registered_edge_types = self.fetch_edge_types()
        if registered_edge_types is None:
            raise ValueError(
                "ABLP GraphStore input requires registered label edge types."
            )
        (
            evaluated_supervision_edge_types,
            supervision_edge_types_with_negatives,
        ) = _resolve_registered_supervision_edge_types(
            supervision_edge_types=evaluated_supervision_edge_types,
            anchor_node_type=evaluated_anchor_node_type,
            edge_dir=evaluated_edge_dir,
            registered_edge_types=registered_edge_types,
        )

        return self._fetch_ablp_input(
            split=split,
            node_type=evaluated_anchor_node_type,
            supervision_edge_types=evaluated_supervision_edge_types,
            supervision_edge_types_with_negatives=supervision_edge_types_with_negatives,
            assignments=assignments,
        )

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

    def fetch_edge_weights_registered(self) -> bool:
        """Fetch whether edge weights were registered in the remote dataset.

        Returns:
            True if edge weights were registered via ``DistPartitioner.register_edge_weights()``.
        """
        return request_server(
            0,
            DistServer.get_edge_weights_registered,
        )
