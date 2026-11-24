import gc
import math
import time
from collections import abc, defaultdict
from typing import Callable, Final, Optional, Union

import graphlearn_torch.distributed.rpc as glt_rpc
import torch
from graphlearn_torch.distributed.dist_context import get_context
from graphlearn_torch.distributed.dist_random_partitioner import DistPartitionManager
from graphlearn_torch.partition import PartitionBook, RangePartitionBook
from graphlearn_torch.utils import convert_to_tensor, index_select

from gigl.common.logger import Logger
from gigl.distributed.utils.partition_book import (
    build_partition_book,
    get_ids_on_rank,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
    to_heterogeneous_edge,
    to_heterogeneous_node,
    to_homogeneous,
)
from gigl.utils.share_memory import allocate_disk_backed, is_disk_backed

logger = Logger()


# Size of the RAM staging block used to write rows in id order. It bounds the staging buffer's RAM;
# larger blocks mean fewer, larger sequential writes to the disk-backed destinations.
_ORDERED_BLOCK_BYTES: Final = 256 * 1024 * 1024


def _empty_rows(num_rows: int, sample: torch.Tensor) -> torch.Tensor:
    """Returns an uninitialized tensor shaped like ``sample`` with ``num_rows`` rows.

    It is disk-backed when spilling is enabled and the tensor is large enough, otherwise in RAM.
    """
    shape = (num_rows, *sample.shape[1:])
    destination = allocate_disk_backed(shape, sample.dtype)
    return (
        torch.empty(shape, dtype=sample.dtype) if destination is None else destination
    )


def _copy_chunks_into(
    chunks: list[tuple[torch.Tensor, ...]], destinations: dict[int, torch.Tensor]
) -> None:
    """Copies each chunk's fields into ``destinations`` (keyed by field), in chunk order, and empties ``chunks``.

    ``chunks`` is mutated on purpose: each chunk is released as soon as it is copied, so the
    destinations never coexist in RAM with every chunk the way ``torch.cat``'s results do.
    """
    first_field = next(iter(destinations))
    chunks.reverse()
    offset = 0
    while chunks:
        chunk = chunks.pop()
        chunk_rows = chunk[first_field].shape[0]
        for field, destination in destinations.items():
            destination[offset : offset + chunk_rows] = chunk[field]
        offset += chunk_rows
        del chunk


class DistPartitioner:
    """
    This class is based on GLT's DistRandomPartitioner class (https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_random_partitioner.py)
    and has been optimized for better flexibility and memory management. We assume that init_rpc() and init_worker_group have been called to initialize the rpc and context,
    respectively, prior to this class. This class aims to partition homogeneous and heterogeneous input data, such as nodes,
    node features, edges, edge features, and any supervision labels across multiple machines. This class also produces range-based partition books for edges and
    nodes. Rather than storing the rank of every id in a tensor, a range partition book stores the upper bound of ids for each rank. For example, a range partition book [4, 8, 12]
    stores ids 0-3 on the 0th rank, 4-7 on the 1st rank, and 8-11 on the 2nd rank. To get the rank of ids, the partition book does a search through
    these partition bounds rather than a direct index lookup. For example, to get the rank of node ids 1 and 6 by doing node_pb[[1, 6]], the range partition book
    uses torch.searchsorted on the partition bounds to return [0, 1], the ranks of each of these ids. As a result, the range-based partition book trades off
    a slower lookup time for indices for a much more efficient memory footprint.

    Nodes are partitioned into continuous ranges so that the difference between the lengths of any two ranges is no greater than 1. Since the node ids on each rank are
    then fully described by the partition book, the partitioned node features and labels are sorted by node id and do not store their ids. Edge partition books are
    built from the number of edges assigned to each rank, and edge ids are inferred from the ranges of the edge partition book.

    In this class, node and edge id and feature tensors can be passed in either through the constructor or the public register functions. It is required to have
    registered these tensors to the class prior to partitioning. For optimal memory management, it is recommended that the reference to these large tensors be deleted
    after being registered to the class but before partitioning, as maintaining both original and intermediate tensors can cause OOM concerns. Registering these tensors is available through both the constructor and the register functions to support
    the multiple use ways customers can use partitioning:

    Option 1: User wants to Partition just the nodes of a graph

    ```
    partitioner = DistPartitioner()
    # Customer doesn't have to pass in excessive amounts of parameters to the constructor to partition only nodes
    partitioner.register_nodes(node_ids)
    del node_ids # Del reference to node_ids outside of DistPartitioner to allow memory cleanup within the class
    partitioner.partition_nodes()
    # We may optionally want to call gc.collect() to ensure that any lingering memory is cleaned up, which may happen in cases where only a subset of inputs are partitioned (i.e no feats or labels)
    gc.collect()
    ```

    Option 2: User wants to partition all parts of a graph together and in sequence

    ```
    partitioner = DistPartitioner(node_ids, edge_index, node_features, edge_features, pos_labels, neg_labels)
    # Register is called in the __init__ functions and doesn't need to be called at all outside the class.
    del (
        node_ids,
        edge_index,
        node_features,
        edge_features,
        pos_labels,
        neg_labels,
        node_labels,
    ) # Del reference to tensors outside of DistPartitioner to allow memory cleanup within the class
    partitioner.partition()
    # We may optionally want to call gc.collect() to ensure that any lingering memory is cleaned up, which may happen in cases where only a subset of inputs are partitioned (i.e no feats or labels)
    gc.collect()
    ```

    The use case for only partitioning one entity through Option 1 may be in cases where we want to further parallelize some of the workload,
    since the previous GLT use case only had access to Partition() which calls partitioning of entities in sequence.

    For optimal memory management, it is recommended that the reference to these large tensors be deleted
    after being registered to the class but before partitioning, as maintaining both original and intermediate tensors can cause OOM concerns.

    Once all desired tensors are registered, you can either call the `partition` function to partition all registered fields or partition each field individually
    through the public `partition_{entity_type}` functions. With the `partition` function, fields which are not registered will return `None`. Note that each entity type
    should only be partitioned once, since registered fields are cleaned up after partitioning for optimal memory impact.

    From GLT's description of DistRandomPartitioner:
        Each distributed partitioner will process a part of the full graph and feature data, and partition them. A distributed partitioner's
        rank is corresponding to a partition index, and the number of all distributed partitioners must be same with the number of output partitions. During
        partitioning, the partitioned results will be sent to other distributed partitioners according to their ranks. After partitioning, each distributed
        partitioner will own a partitioned graph with its corresponding rank and further save the partitioned results into the local output directory.
    """

    def __init__(
        self,
        should_assign_edges_by_src_node: bool = False,
        node_ids: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]] = None,
        node_features: Optional[
            Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        ] = None,
        node_packed_features: Optional[
            Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        ] = None,
        edge_index: Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]] = None,
        edge_features: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = None,
        positive_labels: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = None,
        negative_labels: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = None,
        node_labels: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]] = None,
    ):
        """
        Initializes the parameters of the partitioner. Also optionally takes in node and edge tensors as arguments and registers them to the partitioner. Registered
        entities should be a dictionary of dict[[NodeType or EdgeType], torch.Tensor] if heterogeneous or a torch.Tensor if homogeneous.
        This class assumes the distributed context has already been initialized outside of this class with the glt.distributed.init_worker_group()
        function and that rpc has been initialized with glt_distributed.init_rpc().
        Args:
            should_assign_edges_by_src_node (bool): Whether edges should be assigned to the machine of the source nodes during partitioning
            node_ids (Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]): Optionally registered node ids from input. Tensors should be of shape [num_nodes_on_current_rank]
            node_features (Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]): Optionally registered node feats from input. Tensors should be of shope [num_nodes_on_current_rank, node_feat_dim]
            node_packed_features (Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]): Optionally registered packed uint8 node features from input. Tensors should be of shape [num_nodes_on_current_rank, packed_node_feat_dim]
            edge_index (Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]): Optionally registered edge indexes from input. Tensors should be of shape [2, num_edges_on_current_rank]
            edge_features (Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]): Optionally registered edge features from input. Tensors should be of shape [num_edges_on_current_rank, edge_feat_dim]
            positive_labels (Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]): Optionally registered positive labels from input. Tensors should be of shape [2, num_pos_labels_on_current_rank]
            negative_labels (Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]): Optionally registered negative labels from input. Tensors should be of shape [2, num_neg_labels_on_current_rank]
            node_labels (Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]): Optionally registered node labels from input. Tensors should be of shape [num_nodes_on_current_rank, node_label_dim]
        """

        self._world_size: int
        self._rank: int
        self._partition_mgr: DistPartitionManager

        self._is_rpc_initialized: bool = False

        self._is_input_homogeneous: Optional[bool] = None
        self._should_assign_edges_by_src_node: bool = should_assign_edges_by_src_node
        self._edge_types: list[EdgeType] = []
        self._node_types: list[NodeType] = []
        self._num_nodes: Optional[dict[NodeType, int]] = None
        self._num_edges: Optional[dict[EdgeType, int]] = None

        self._node_ids: Optional[dict[NodeType, torch.Tensor]] = None
        self._max_node_ids: Optional[dict[NodeType, int]] = None
        self._node_feat: Optional[dict[NodeType, torch.Tensor]] = None
        self._node_feat_dim: Optional[dict[NodeType, int]] = None
        self._node_packed_feat: Optional[dict[NodeType, torch.Tensor]] = None
        self._node_packed_feat_dim: Optional[dict[NodeType, int]] = None
        self._node_labels: Optional[dict[NodeType, torch.Tensor]] = None
        self._node_labels_dim: Optional[dict[NodeType, int]] = None

        self._edge_index: Optional[dict[EdgeType, torch.Tensor]] = None
        self._edge_feat: Optional[dict[EdgeType, torch.Tensor]] = None
        self._edge_feat_dim: Optional[dict[EdgeType, int]] = None
        self._edge_packed_feat: Optional[dict[EdgeType, torch.Tensor]] = None
        self._edge_packed_feat_dim: Optional[dict[EdgeType, int]] = None
        self._edge_weights: Optional[dict[EdgeType, torch.Tensor]] = None

        # TODO (mkolodner-sc): Deprecate the need for explicitly storing labels are part of this class, leveraging
        # heterogeneous support instead
        self._positive_label_edge_index: Optional[dict[EdgeType, torch.Tensor]] = None
        self._negative_label_edge_index: Optional[dict[EdgeType, torch.Tensor]] = None

        if node_ids is not None:
            self.register_node_ids(node_ids=node_ids)

        if edge_index is not None:
            self.register_edge_index(edge_index=edge_index)

        if node_features is not None:
            self.register_node_features(node_features=node_features)

        if node_packed_features is not None:
            self.register_node_packed_features(
                node_packed_features=node_packed_features
            )

        if edge_features is not None:
            self.register_edge_features(edge_features=edge_features)

        if positive_labels is not None:
            self.register_labels(label_edge_index=positive_labels, is_positive=True)

        if negative_labels is not None:
            self.register_labels(label_edge_index=negative_labels, is_positive=False)
        if node_labels is not None:
            self.register_node_labels(node_labels=node_labels)

    def _assert_data_type_consistency(
        self,
        input_entity: abc.Mapping,
        is_node_entity: bool,
        is_subset: bool,
    ) -> None:
        """
        Checks that the keys of the input_entity, which must be a dictionary, align with registered node or edge types.

        This function will check that the registered node/edge types align with the input tensor's node/edge types.
        The function determines whether to check node or edge types through the provided `is_node_entity` argument.

        Args:
            input_entity (abc.Mapping): Input entity, which must be a dictionary
            is_node_entity (bool): Whether the current input entity containing node information, if False the input entity is assumed to be for edges.
            is_subset (bool): Whether the current input should be a subset of the entity types, if False will check that it is an exact match
        """
        if is_node_entity:
            # We check that the input tensor node types match the registered node types.
            if is_subset:
                node_diff = set(input_entity.keys()) - set(self._node_types)
                assert not node_diff, (
                    f"Found node types {node_diff} not contained in registered node types {self._node_types}"
                )
            else:
                assert self._node_types == sorted(input_entity.keys()), (
                    f"Found different node input types {sorted(input_entity.keys())} from registered node types {self._node_types}"
                )
        else:
            # We check that the input tensor edge types match the registered edge types.
            if is_subset:
                edge_diff = set(input_entity.keys()) - set(self._edge_types)
                assert not edge_diff, (
                    f"Found edge types {edge_diff} not contained in registered edge types {self._edge_types}"
                )
            else:
                assert self._edge_types == sorted(input_entity.keys()), (
                    f"Found different edge input types {sorted(input_entity.keys())} from registered edge types {self._edge_types}"
                )

    def _assert_and_get_rpc_setup(self) -> None:
        """
        Asserts RPC and worker context have been initialized. If this is the first time this is called,
        sets the rank, world_size, and partition manager fields of the partitioner class from the distributed context.
        """
        assert get_context() is not None, (
            "Distributed context must be initialized by the user prior to partitioning or registering fields by calling glt.distributed.init_worker_group()"
        )

        assert glt_rpc.rpc_is_initialized(), (
            "rpc must be initialized by the user prior to partitioning or registering fields to partitioning by calling glt.distributed.init_rpc()"
        )

        # If rank, world size, and partition_manager are not set, we set them once we know the context and rpc has been initialized.
        if not self._is_rpc_initialized:
            logger.info(
                f"Machine {get_context().rank} setting up partition manager ..."
            )
            self._rank = get_context().rank
            self._world_size = get_context().world_size
            self._partition_mgr = DistPartitionManager(
                total_val_size=0, generate_pb=False
            )
        self._is_rpc_initialized = True

    def _convert_node_entity_to_heterogeneous_format(
        self,
        input_node_entity: Union[
            torch.Tensor,
            PartitionBook,
            dict[NodeType, torch.Tensor],
            dict[NodeType, PartitionBook],
        ],
    ) -> Union[dict[NodeType, torch.Tensor], dict[NodeType, PartitionBook]]:
        """
        Converts input_node_entity into heterogeneous format if it is not already. If input is homogeneous, this will be a dictionary with Node Type DEFAULT_HOMOGENEOUS_NODE_TYPE.
        This is done so that the logical can be simplified for partitioning to just the heterogeneous case. Homogeneous inputs are re-converted back to non-dictionary
        formats when returning the outputs of partitioning through the `self._is_input_homogeneous` variable.
        """

        if isinstance(input_node_entity, (PartitionBook, torch.Tensor)):
            if (
                self._is_input_homogeneous is not None
                and not self._is_input_homogeneous
            ):
                raise ValueError(
                    "Registering homogeneous field when previously registered entity was heterogeneous"
                )
            self._is_input_homogeneous = True
        else:
            if self._is_input_homogeneous is not None and self._is_input_homogeneous:
                raise ValueError(
                    "Registering heterogeneous field when previously registered entity was heterogeneous"
                )
            self._is_input_homogeneous = False
        return to_heterogeneous_node(input_node_entity)

    def _edge_index_dtype(self, edge_type: EdgeType) -> torch.dtype:
        """int32 when both endpoint types' ids fit, halving the edge index; int64 otherwise.

        ``_max_node_ids`` is gathered across ranks in ``register_node_ids``, so every rank picks the
        same width.
        """
        int32_max = torch.iinfo(torch.int32).max
        if self._max_node_ids is None:
            return torch.int64
        # A node type that was never registered counts as too large.
        max_id = max(
            self._max_node_ids.get(node_type, int32_max + 1)
            for node_type in (edge_type.src_node_type, edge_type.dst_node_type)
        )
        return torch.int32 if max_id <= int32_max else torch.int64

    def _convert_edge_entity_to_heterogeneous_format(
        self,
        input_edge_entity: Union[
            torch.Tensor,
            PartitionBook,
            dict[EdgeType, torch.Tensor],
            dict[EdgeType, PartitionBook],
        ],
    ) -> Union[dict[EdgeType, torch.Tensor], dict[EdgeType, PartitionBook]]:
        """
        Converts input_edge_entity into heterogeneous format if it is not already. If input is homogeneous, this will be a dictionary with Edge Type DEFAULT_HOMOGENEOUS_EDGE_TYPE.
        """

        if isinstance(input_edge_entity, (PartitionBook, torch.Tensor)):
            if not self._is_input_homogeneous:
                raise ValueError(
                    "Registering homogeneous field when previously registered entity was heterogeneous"
                )
            self._is_input_homogeneous = True
        else:
            if self._is_input_homogeneous:
                raise ValueError(
                    "Registering heterogeneous field when previously registered entity was heterogeneous"
                )
            self._is_input_homogeneous = False
        return to_heterogeneous_edge(input_edge_entity)

    def register_node_ids(
        self, node_ids: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    ) -> None:
        """
        Registers the node ids to the partitioner. Also computes additional fields for partitioning such as the total number of nodes across all ranks.

        Note that node ids provided must be unique and contiguous from 0 to total_num_nodes - 1 for each provided node type across all machines.

        For optimal memory management, it is recommended that the reference to the node_id tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        Args:
            node_ids (Union[torch.Tensor, dict[NodeType, torch.Tensor]]): Input node_ids which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        # Check if node data has already been registered
        if self._node_ids is not None:
            raise ValueError(
                "Node IDs have already been registered. Cannot re-register node data."
            )

        logger.info("Registering Nodes ...")
        input_node_ids = self._convert_node_entity_to_heterogeneous_format(
            input_node_entity=node_ids
        )

        assert input_node_ids, (
            "Node ids is an empty dictionary. Please provide node ids to register."
        )

        self._node_types = sorted(input_node_ids.keys())

        self._node_ids = convert_to_tensor(input_node_ids, dtype=torch.int64)

        self._num_nodes = defaultdict(int)
        self._max_node_ids = defaultdict(int)

        node_type_to_num_nodes_and_max_node: dict[NodeType, tuple[int, int]] = {}
        for node_type in self._node_types:
            if input_node_ids[node_type].numel() > 0:
                node_type_to_num_nodes_and_max_node[node_type] = (
                    input_node_ids[node_type].size(0),
                    int(input_node_ids[node_type].max().item()),
                )
            else:
                node_type_to_num_nodes_and_max_node[node_type] = (0, 0)

        # This tuple here represents a (rank, (num_nodes_on_rank, max_node_id_on_rank)) pair on a given partition,
        # specified by the str key of the dictionary of format `distributed_random_partitoner_{rank}`.
        # num_nodes_on_rank and max_node_id_on_rank are ints.
        # Gathered_node_info is then used to identify the number of nodes and max node id on each rank,
        # allowing us to access the total number of nodes and max node id across all ranks
        gathered_node_info: dict[str, tuple[int, dict[NodeType, tuple[int, int]]]] = (
            glt_rpc.all_gather((self._rank, node_type_to_num_nodes_and_max_node))
        )

        for (
            _,
            gathered_node_type_to_num_nodes_and_max_node,
        ) in gathered_node_info.values():
            for node_type, (
                num_nodes,
                max_id,
            ) in gathered_node_type_to_num_nodes_and_max_node.items():
                self._num_nodes[node_type] += num_nodes
                self._max_node_ids[node_type] = max(
                    self._max_node_ids[node_type], max_id
                )

    def register_edge_index(
        self, edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ) -> None:
        """
        Registers the edge_index to the partitioner. Also computes additional fields for partitioning such as the total number of edges across all ranks.
        Edge ids are not inferred here, since they are not needed for partitioning.

        For optimal memory management, it is recommended that the reference to edge_index tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        Args:
            edge_index (Union[torch.Tensor, dict[EdgeType, torch.Tensor]]): Input edge index which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        Raises:
            ValueError: If edge indices have already been registered.
        """

        self._assert_and_get_rpc_setup()

        # Check if edge data has already been registered
        if self._edge_index is not None:
            raise ValueError(
                "Edge indices have already been registered. Cannot re-register edge data."
            )

        logger.info("Registering Edge Indices ...")

        input_edge_index = self._convert_edge_entity_to_heterogeneous_format(
            input_edge_entity=edge_index
        )

        assert input_edge_index, (
            "Edge Index is an empty dictionary. Please provide edge indices to register."
        )

        self._edge_types = sorted(input_edge_index.keys())

        self._edge_index = {
            edge_type: convert_to_tensor(
                edge_index, dtype=self._edge_index_dtype(edge_type)
            )
            for edge_type, edge_index in input_edge_index.items()
        }

        # The tuple here represents a (rank, num_edges_on_rank) pair on a given partition, specified by the str key of the dictionary of format `distributed_random_partitoner_{rank}`
        # num_edges_on_rank is a dict[EdgeType, int].
        # Gathered_num_edges is then used to identify the number of edges on each rank, allowing us to access the total number of edges across all ranks
        gathered_edge_info: dict[str, tuple[int, dict[EdgeType, int]]]
        self._num_edges = {}
        edge_type_to_num_edges: dict[EdgeType, int] = {
            edge_type: input_edge_index[edge_type].size(1)
            for edge_type in sorted(input_edge_index.keys())
        }

        # Gathering to compute the number of edges on each rank for each edge type
        gathered_edge_info = glt_rpc.all_gather((self._rank, edge_type_to_num_edges))

        # Looping through registered edge types in graph
        for edge_type in self._edge_types:
            # Populating num_edges_all_ranks list, where num_edges_all_ranks[i] = num_edges means that rank `i`` has `num_edges` edges
            num_edges_all_ranks = [0] * self._world_size
            for (
                rank,
                gathered_edge_type_to_num_edges,
            ) in gathered_edge_info.values():
                num_edges_all_ranks[rank] = gathered_edge_type_to_num_edges[edge_type]

            # Setting total number of edges across all ranks
            self._num_edges[edge_type] = sum(num_edges_all_ranks)

    def register_node_features(
        self, node_features: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    ) -> None:
        """
        Registers the node features to the partitioner.

        Given a
            - node feature tensor on current machine of shape [num_nodes_on_current_rank, node_feat_dim]
            - node id tensor on current machine of shape [num_nodes_on_current_rank] which has been registered through `register_node_ids`
        We assume that the node feature at node_feature[n] corresponds to the node id at node_id[n].

        Note that the order of calling register_node_ids and register_node_features does not matter, but both must be called
        prior to partitioning node features.

        For optimal memory management, it is recommended that the reference to node_features tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        We do not need to perform `all_gather` calls here since register_node_ids is responsible for determining total number of nodes
        across all ranks.
        Args:
            node_features(Union[torch.Tensor, dict[NodeType, torch.Tensor]]): Input node features which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        # Check if node features have already been registered
        if self._node_feat is not None:
            raise ValueError(
                "Node features have already been registered. Cannot re-register node feature data."
            )

        logger.info("Registering Node Features ...")

        input_node_features = self._convert_node_entity_to_heterogeneous_format(
            input_node_entity=node_features
        )

        assert input_node_features, (
            "Node features is an empty dictionary. Please provide node features to register."
        )

        self._node_feat = convert_to_tensor(input_node_features, dtype=torch.float32)
        self._node_feat_dim = {}
        for node_type in input_node_features:
            self._node_feat_dim[node_type] = input_node_features[node_type].shape[1]

    def register_node_packed_features(
        self,
        node_packed_features: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
    ) -> None:
        """Registers packed uint8 node features to the partitioner."""

        self._assert_and_get_rpc_setup()

        if self._node_packed_feat is not None:
            raise ValueError(
                "Node packed features have already been registered. Cannot re-register node packed feature data."
            )

        logger.info("Registering Node Packed Features ...")

        input_node_packed_features = self._convert_node_entity_to_heterogeneous_format(
            input_node_entity=node_packed_features
        )

        assert input_node_packed_features, (
            "Node packed features is an empty dictionary. Please provide node packed features to register."
        )

        self._node_packed_feat = convert_to_tensor(
            input_node_packed_features, dtype=torch.uint8
        )
        self._node_packed_feat_dim = {}
        for node_type in input_node_packed_features:
            self._node_packed_feat_dim[node_type] = input_node_packed_features[
                node_type
            ].shape[1]

    def register_node_labels(
        self, node_labels: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    ) -> None:
        """
        Registers the node labels to the partitioner.

        Given a
            - node label tensor on current machine of shape [num_nodes_on_current_rank, label_dim]
            - node id tensor on current machine of shape [num_nodes_on_current_rank] which has been registered through `register_node_ids`
        We assume that the node label at node_labels[n] corresponds to the node id at node_id[n].

        Note that the order of calling register_node_ids and register_node_labels does not matter, but both must be called
        prior to partitioning node labels.

        For optimal memory management, it is recommended that the reference to node_labels tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        We do not need to perform `all_gather` calls here since register_node_ids is responsible for determining total number of nodes
        across all ranks.
        Args:
            node_labels (Union[torch.Tensor, dict[NodeType, torch.Tensor]]): Input node labels which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        # Check if node labels have already been registered
        if self._node_labels is not None:
            raise ValueError(
                "Node labels have already been registered. Cannot re-register node label data."
            )

        logger.info("Registering Node Labels ...")

        input_node_labels = self._convert_node_entity_to_heterogeneous_format(
            input_node_entity=node_labels
        )

        assert input_node_labels, (
            "Node labels is an empty dictionary. Please provide node labels to register."
        )

        self._node_labels = convert_to_tensor(input_node_labels, dtype=torch.int64)

        self._node_labels_dim = {}
        for node_type in input_node_labels:
            self._node_labels_dim[node_type] = input_node_labels[node_type].size(1)

    def register_edge_features(
        self, edge_features: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ) -> None:
        """
        Registers the edge features to the partitioner.

        For optimal memory management, it is recommended that the reference to edge_features tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        We do not need to perform `all_gather` calls here since register_edge_index is responsible for determining total number of edges
        across all ranks and inferrring edge ids.
        Args:
            edge_features(Union[torch.Tensor, dict[EdgeType, torch.Tensor]]): Input edge features which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        # Check if edge features have already been registered
        if self._edge_feat is not None:
            raise ValueError(
                "Edge features have already been registered. Cannot re-register edge feature data."
            )

        logger.info("Registering Edge Features ...")

        input_edge_features = self._convert_edge_entity_to_heterogeneous_format(
            input_edge_entity=edge_features
        )

        assert input_edge_features, (
            "Edge features is an empty dictionary. Please provide edge features to register."
        )

        self._edge_feat = convert_to_tensor(input_edge_features, dtype=torch.float32)
        self._edge_feat_dim = {}
        for edge_type in input_edge_features:
            self._edge_feat_dim[edge_type] = input_edge_features[edge_type].shape[1]

    def register_edge_packed_features(
        self, edge_packed_features: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ) -> None:
        """Register packed uint8 main-edge features for co-partitioning."""

        self._assert_and_get_rpc_setup()
        if self._edge_packed_feat is not None:
            raise ValueError(
                "Edge packed features have already been registered. Cannot re-register edge packed feature data."
            )
        logger.info("Registering Edge Packed Features ...")

        input_edge_packed_features = self._convert_edge_entity_to_heterogeneous_format(
            input_edge_entity=edge_packed_features
        )

        assert input_edge_packed_features, (
            "Edge packed features is an empty dictionary. Please provide edge packed features to register."
        )

        self._edge_packed_feat = convert_to_tensor(
            input_edge_packed_features, dtype=torch.uint8
        )
        self._edge_packed_feat_dim = {
            edge_type: features.shape[1]
            for edge_type, features in input_edge_packed_features.items()
        }

    def register_edge_weights(
        self, edge_weights: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ) -> None:
        """Registers per-edge sampling weights to the partitioner.

        Weights must be a 1-D float tensor of shape ``[num_edges]``, one scalar per edge.
        Must be called before ``partition()``. May be called before or after
        ``register_edge_index()``; shape validation against the edge index is performed
        only when ``register_edge_index()`` has already been called.

        Weights are kept separate from edge features — do not include the weight column
        in edge features passed to ``register_edge_features()``.

        For optimal memory management, delete the reference to ``edge_weights`` after
        calling this function.

        Args:
            edge_weights: Per-edge weights, either a ``[num_edges]`` tensor if homogeneous
                or a ``dict[EdgeType, Tensor]`` if heterogeneous.
        """
        self._assert_and_get_rpc_setup()

        if self._edge_weights is not None:
            raise ValueError(
                "Edge weights have already been registered. Cannot re-register edge weight data."
            )

        logger.info("Registering Edge Weights ...")

        input_edge_weights = self._convert_edge_entity_to_heterogeneous_format(
            input_edge_entity=edge_weights
        )

        if not input_edge_weights:
            raise ValueError(
                "Edge weights is an empty dictionary. Please provide edge weights to register."
            )

        for edge_type, weight_tensor in input_edge_weights.items():
            if weight_tensor.ndim != 1:
                raise ValueError(
                    f"Edge weights for edge type {edge_type} must be a 1-D tensor of shape "
                    f"[num_edges], got shape {tuple(weight_tensor.shape)}."
                )
        self._edge_weights = convert_to_tensor(input_edge_weights, dtype=torch.float32)

    def register_labels(
        self,
        label_edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
        is_positive: bool,
    ) -> None:
        """
        Registers the positive or negative label to the partitioner. Note that for the homogeneous case,
        all edge types of the graph must be present in the label edge index dictionary.

        For optimal memory management, it is recommended that the reference to the label tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        We do not need to perform `all_gather` calls here since register_edge_index is responsible for determining total number of edges
        across all ranks and inferring edge ids.
        Args:
            label_edge_index (Union[torch.Tensor, dict[EdgeType, torch.Tensor]]): Input positive or negative labels which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
            is_positive (bool): Whether positive labels are currently being registered. If False, labels will be registered as negative
        """

        self._assert_and_get_rpc_setup()

        # Check if labels have already been registered
        if is_positive:
            if self._positive_label_edge_index is not None:
                raise ValueError(
                    "Positive labels have already been registered. Cannot re-register positive label data."
                )
        else:
            if self._negative_label_edge_index is not None:
                raise ValueError(
                    "Negative labels have already been registered. Cannot re-register negative label data."
                )

        input_label_edge_index = self._convert_edge_entity_to_heterogeneous_format(
            input_edge_entity=label_edge_index
        )

        assert input_label_edge_index, (
            "Label edge index is an empty dictionary. Please provide label edge indices to register."
        )

        if is_positive:
            logger.info("Registering Positive Labels ...")
            self._positive_label_edge_index = convert_to_tensor(
                input_label_edge_index, dtype=torch.int64
            )
        else:
            logger.info("Registering Negative Labels ...")
            self._negative_label_edge_index = convert_to_tensor(
                input_label_edge_index, dtype=torch.int64
            )

    def _assemble_node_chunks(
        self,
        chunks: list[tuple[torch.Tensor, ...]],
        fields: list[int],
        node_id_field: int,
    ) -> dict[int, torch.Tensor]:
        """Join ``fields`` of the received chunks sorted by node id, emptying ``chunks``.

        ``node_id_field`` is the position of the node ids within each chunk tuple, and must be in ``fields``.
        The range-based ``id2index`` maps node id to row by offset, so rows must be sorted by id.
        If every destination is in RAM, the chunks are joined and then gathered by id.
        Otherwise each destination is filled one block of rows at a time, so a spilled field
        never needs a second full copy in RAM.

        Returns:
            The sorted tensor for each field in ``fields``, keyed by field.
        """
        chunk_node_ids = [chunk[node_id_field] for chunk in chunks]
        chunk_row_counts = [node_ids.shape[0] for node_ids in chunk_node_ids]
        num_rows = sum(chunk_row_counts)
        destinations: dict[int, torch.Tensor] = {}
        for field in fields:
            destinations[field] = _empty_rows(num_rows, chunks[0][field])
        if not any(is_disk_backed(tensor) for tensor in destinations.values()):
            # The block fill below handles any mix of disk and RAM destinations. This path is only for
            # the all-RAM case, where freeing each chunk as it is copied peaks lower. The destinations
            # allocated above are reused: torch.empty does not touch pages, so they cost nothing until written.
            # chunk_node_ids must go first, or it would keep every chunk's ids alive.
            del chunk_node_ids
            _copy_chunks_into(chunks, destinations)
            order = torch.argsort(destinations[node_id_field])
            # Gather one field at a time so each joined tensor is freed once its sorted copy exists.
            for field in fields:
                destinations[field] = destinations[field][order]
            return destinations
        chunk_starts = torch.tensor([0] + chunk_row_counts).cumsum(0)
        order = torch.argsort(torch.cat(chunk_node_ids))
        del chunk_node_ids
        row_bytes = 1
        for tensor in destinations.values():
            row_bytes = max(
                row_bytes, math.prod(tensor.shape[1:]) * tensor.element_size()
            )
        block_rows = max(1, _ORDERED_BLOCK_BYTES // row_bytes)
        for block_start in range(0, num_rows, block_rows):
            # Rows of the concatenated chunks, in the order they belong in this block.
            positions = order[block_start : block_start + block_rows]
            # Locate the chunk holding each row once per block, so the per-field work below is
            # one gather per chunk rather than per row.
            chunk_of_position = (
                torch.searchsorted(chunk_starts, positions, right=True) - 1
            )
            by_chunk = torch.argsort(chunk_of_position, stable=True)
            counts = torch.bincount(chunk_of_position, minlength=len(chunks)).tolist()
            block_rows_by_chunk = by_chunk.split(counts)
            row_in_chunk = (positions - chunk_starts[chunk_of_position])[by_chunk]
            source_rows_by_chunk = row_in_chunk.split(counts)
            # Only the chunks that contribute rows to this block, shared by every field.
            contributions: list[tuple[int, torch.Tensor, torch.Tensor]] = []
            for chunk_index, count in enumerate(counts):
                if count > 0:
                    contributions.append(
                        (
                            chunk_index,
                            block_rows_by_chunk[chunk_index],
                            source_rows_by_chunk[chunk_index],
                        )
                    )
            for field, destination in destinations.items():
                # Gather into RAM first: random writes to a disk-backed destination are slow, a
                # single sequential copy of the finished block is not.
                staging = torch.empty(
                    (positions.numel(), *destination.shape[1:]),
                    dtype=destination.dtype,
                )
                for chunk_index, rows_in_block, source_rows in contributions:
                    staging[rows_in_block] = chunks[chunk_index][field][source_rows]
                destination[block_start : block_start + positions.numel()] = staging
        chunks.clear()
        return destinations

    def _remove_key_from_member_dict(
        self, field_name: str, entity_key: Union[NodeType, EdgeType]
    ) -> None:
        """
        This method safely removes a key (node type or edge type) from a dictionary attribute of the partitioner.
        If the dictionary becomes empty after removing the key, the entire attribute is set to None to maintain consistency.

        Args:
            field_name (str): The name of the instance attribute (e.g., '_node_feat')
            entity_key (Union[NodeType, EdgeType]): The key to remove from the dictionary
        Raises:
            AttributeError: If the field_name does not exist in the class
        """
        field = getattr(self, field_name)
        if field is not None:
            assert isinstance(field, dict), (
                f"Field {field_name} is not a dict, got {type(field)}"
            )
            if entity_key in field:
                del field[entity_key]
                if len(field) == 0:
                    setattr(self, field_name, None)

    def _partition_single_chunk_data(
        self,
        input_data: Optional[tuple[torch.Tensor, ...]],
        rank_indices: torch.Tensor,
        partition_function: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor],
        chunk_start_pos: int,
        chunk_end_pos: int,
    ) -> None:
        """
        Partitions a single chunk of data across multiple machines. First, the partition function is used to lookup or compute the rank of the current input.
        Then, the positions are stably sorted by rank and split into one compact piece per rank. We then send each piece to its machine using the partition manager.

        Flow for a chunk of 6 items on 3 machines::

            chunk_rank                  [2, 0, 1, 0, 2, 1]   rank of each item, from partition_function
            counts                      [2, 2, 2]            items per rank, from bincount
            order (stable argsort)      [1, 3, 2, 5, 0, 4]   positions grouped by rank, input order kept within a rank
            order.split(counts)         [1, 3] [2, 5] [0, 4] one position tensor per rank
            index_select(input_data, p) gathers each rank's items into its own compact tensor
            partition_mgr.process(...)  sends piece i to machine i
        Args:
            input_data (Optional[tuple[torch.Tensor, ...]]): generic data type of items to be partitioned on the current chunk, which any information that should be partitioned across machines.
            rank_indices (torch.Tensor): torch tensor of indices which are used to determine the rank of each item to be partitioned on the current chunk
            partition_function (Callable): Function for determining ranks of current chunk. The first argument to this function is
                the specified indices in the chunk range while the second argument is the chunk start and end values. It returns a tuple indicating the rank
                of each item in the chunk.
            chunk_start_pos (int): The starting position of the current chunk being partitioned
            chunk_end_pos (int): The ending position of the current chunk being partitioned
        """
        chunk_rank = partition_function(rank_indices, (chunk_start_pos, chunk_end_pos))

        # Splits the chunk by destination rank with one stable sort.
        # The stable sort keeps the within-rank order of the input, so each rank's positions equal what masking would select.
        # Valid ranks are in [0, world_size), so the key dtypes below cannot wrap; out-of-range ranks are undefined.
        counts = torch.bincount(chunk_rank, minlength=self._world_size).tolist()
        if self._world_size <= 256:
            sort_key = chunk_rank.to(torch.uint8)
        elif self._world_size <= 2**15:
            sort_key = chunk_rank.to(torch.int16)
        else:
            sort_key = chunk_rank.to(torch.int32)
        order = torch.argsort(sort_key, stable=True)
        del sort_key

        # Each rank's piece is gathered from the unsorted input into its own compact tensor.
        # Views of one sorted buffer would be sent whole by rpc when a piece covers more than half of it,
        # and would pin the buffer on the receiver until the final concatenation.
        # The second element of each pair is the partition book update, which is unused since we do not generate a partition book here.
        chunk_res: list[
            tuple[Optional[tuple[torch.Tensor, ...]], Optional[torch.Tensor]]
        ] = []
        for rank_positions in order.split(counts):
            rank_data = index_select(input_data, rank_positions)
            chunk_res.append((rank_data, None))
        del order

        self._partition_mgr.process(chunk_res)

    def _partition_by_chunk(
        self,
        input_data: Optional[tuple[torch.Tensor, ...]],
        rank_indices: torch.Tensor,
        partition_function: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor],
    ) -> list[tuple[torch.Tensor, ...]]:
        r"""Partitions input data chunk by chunk.
        Args:
            input_data (Optional[tuple[torch.Tensor, ...]]): generic data type of items to be partitioned across machine, which any information that should be partitioned across machines.
            rank_indices (torch.Tensor): torch tensor of indices which are used to determine the rank of each item to be partitioned
            partition_function (Callable): Function for determining ranks of current chunk. The first argument to this function is
                the specified indices in the chunk range while the second argument is the chunk start and end values. It returns a tuple indicating the rank
                of each item in the chunk.
        Return:
            list[tuple[torch.Tensor, ...]]: Partitioned results of the input generic data type
        """
        num_items = len(rank_indices)

        # We currently hard-code the chunk_num to be 8 unless the number of items is less than 8, and determine the chunk size based on this value.
        # If this is not performant, we may revisit this in the future.
        chunk_num = min(num_items, 8)
        if chunk_num != 0:
            chunk_size, remainder = divmod(num_items, chunk_num)
        else:
            chunk_size = 0
            remainder = 0

        # This is set to 0 since the the data that is provided is already per-rank, and we begin at index 0 of this local data.
        chunk_start_pos = 0

        # Resets the partition manager's partition list. Range-based partitioning does not generate a partition book from the partitioned results.
        self._partition_mgr.reset(total_val_size=0, generate_pb=False)
        glt_rpc.barrier()

        # Rather than processing all of the tensors at once, we batch the tensors into chunks and process them separately.
        # Doing so yields performance improvement over processesing all of the tensor at once or processing each item individually.
        for i in range(chunk_num):
            # We set `remainder` number of chunks to have at most one more item.
            if i < remainder:
                chunk_end_pos = chunk_start_pos + chunk_size + 1
            else:
                chunk_end_pos = chunk_start_pos + chunk_size
            self._partition_single_chunk_data(
                input_data=index_select(
                    input_data, index=(chunk_start_pos, chunk_end_pos)
                ),
                rank_indices=rank_indices[chunk_start_pos:chunk_end_pos],
                partition_function=partition_function,
                chunk_start_pos=chunk_start_pos,
                chunk_end_pos=chunk_end_pos,
            )

            chunk_start_pos = chunk_end_pos

            # The garbage collection here significantly reduces peak memory usage.
            gc.collect()

        glt_rpc.barrier()

        return self._partition_mgr.cur_part_val_list

    def _partition_node(self, node_type: NodeType) -> PartitionBook:
        """
        Partition graph nodes of a specific node type. For range-based partitioning, we partition all
        the nodes into continuous ranges so that the diff between lengths of any two ranges is no greater
        than 1. This function gets called by the `partition_node` API, which handles
        the node partitioning across all node types.

        Args:
            node_type (NodeType): The node type for input nodes

        Returns:
            PartitionBook: The partition book of graph nodes.
        """

        start_time = time.time()

        assert self._num_nodes is not None, (
            "Must have registered nodes prior to partitioning them"
        )

        num_nodes = self._num_nodes[node_type]

        node_partition_book = build_partition_book(
            num_nodes, self._rank, self._world_size
        )

        logger.info(
            f"Got node range-based partition book for node type {node_type} on rank {self._rank} with partition bounds: {node_partition_book.partition_bounds}"
        )

        logger.info(
            f"Node Partitioning for node type {node_type} finished, took {time.time() - start_time:.3f}s"
        )

        return node_partition_book

    # TODO: Create a dataclass for this positional three-value partition result.
    def _partition_node_features_and_labels(
        self,
        node_partition_book: dict[NodeType, PartitionBook],
        node_type: NodeType,
    ) -> tuple[
        Optional[FeaturePartitionData],
        Optional[FeaturePartitionData],
        Optional[FeaturePartitionData],
    ]:
        """
        Partitions node features and labels according to the node partition book.

        We should partition node features and labels together so that the ids of the partitioned node features and labels
        are the same, leading to an overall smaller memory footprint and faster partitioning.

        The partitioned rows are sorted by node id, so that the range-based id2idx corresponds correctly to the rows.
        The partitioned node ids are not returned (`ids` is None), since they are available from the node partition book.

        Args:
            node_partition_book (dict[NodeType, PartitionBook]): The partition book of nodes
            node_type (NodeType): Node type of input data

        Returns:
            Optional[FeaturePartitionData]: Partitioned data of node features for current node type.
            Optional[FeaturePartitionData]: Partitioned data of packed uint8 node features for current node type.
            Optional[FeaturePartitionData]: Partitioned data of node labels for current node type.
        """

        start_time = time.time()

        assert (
            self._num_nodes is not None
            and self._node_ids is not None
            and self._max_node_ids is not None
        ), "Node ids must be registered prior to partitioning."

        target_node_partition_book = node_partition_book[node_type]
        node_ids = self._node_ids[node_type]
        num_nodes = self._num_nodes[node_type]
        max_node_ids = self._max_node_ids[node_type]

        if not torch.all(node_ids >= 0):
            raise ValueError(f"Found negative node ids for node type {node_type}")

        if num_nodes > 0 and max_node_ids + 1 != num_nodes:
            raise ValueError(
                f"Found max_id_on_rank which must be exactly one smaller than total number of nodes, but got max id: {max_node_ids} and total_num_nodes: {num_nodes} for node type {node_type}. This is beceause \
                Node IDS provided must be unique and contiguous from 0 to total_num_nodes - 1. Thus, we require and enforce that the max id is one smaller than the total number of nodes."
            )

        # Since we partition node features and labels together for each node type, there are possible scenarios where a
        # given node type may have features but not labels or labels but not features. Therefore, we need to check for these
        # scenarios and still partition the entity which exists.

        node_features = (
            self._node_feat[node_type]
            if self._node_feat is not None and node_type in self._node_feat
            else None
        )
        node_feat_dim = (
            self._node_feat_dim[node_type]
            if self._node_feat_dim is not None and node_type in self._node_feat_dim
            else None
        )
        node_packed_features = (
            self._node_packed_feat[node_type]
            if self._node_packed_feat is not None
            and node_type in self._node_packed_feat
            else None
        )
        node_packed_feat_dim = (
            self._node_packed_feat_dim[node_type]
            if self._node_packed_feat_dim is not None
            and node_type in self._node_packed_feat_dim
            else None
        )

        node_labels = (
            self._node_labels[node_type]
            if self._node_labels is not None and node_type in self._node_labels
            else None
        )
        node_labels_dim = (
            self._node_labels_dim[node_type]
            if self._node_labels_dim is not None and node_type in self._node_labels_dim
            else None
        )

        input_parts: list[torch.Tensor] = []
        node_feature_ind: Optional[int] = None
        node_packed_feature_ind: Optional[int] = None
        node_label_ind: Optional[int] = None

        if node_features is not None:
            node_feature_ind = len(input_parts)
            input_parts.append(node_features)
        if node_packed_features is not None:
            node_packed_feature_ind = len(input_parts)
            input_parts.append(node_packed_features)
        if node_labels is not None:
            node_label_ind = len(input_parts)
            input_parts.append(node_labels)
        if not input_parts:
            raise ValueError(
                f"Found no node features, packed node features, or node labels to partition for node type {node_type}"
            )
        input_parts.append(node_ids)
        input_data: tuple[torch.Tensor, ...] = tuple(input_parts)

        has_node_features = node_features is not None
        has_node_packed_features = node_packed_features is not None
        has_node_labels = node_labels is not None

        def _node_feature_partition_fn(node_feature_ids, _):
            return target_node_partition_book[node_feature_ids]

        # partitioned_results is a list of tuples. Each tuple correpsonds
        # to a chunk of data. A tuple contains node features, node labels, and node ids.
        partitioned_results = self._partition_by_chunk(
            input_data=input_data,
            rank_indices=node_ids,
            partition_function=_node_feature_partition_fn,
        )

        self._remove_key_from_member_dict("_node_ids", node_type)
        self._remove_key_from_member_dict("_num_nodes", node_type)
        self._remove_key_from_member_dict("_max_node_ids", node_type)
        self._remove_key_from_member_dict("_node_feat", node_type)
        self._remove_key_from_member_dict("_node_feat_dim", node_type)
        self._remove_key_from_member_dict("_node_packed_feat", node_type)
        self._remove_key_from_member_dict("_node_packed_feat_dim", node_type)
        self._remove_key_from_member_dict("_node_labels", node_type)
        self._remove_key_from_member_dict("_node_labels_dim", node_type)

        # Since the unpartitioned node ids, features and labels are large, we would like to delete them when
        # they are no longer needed to free memory.
        del (
            node_ids,
            num_nodes,
            max_node_ids,
            node_features,
            node_packed_features,
            node_labels,
        )

        gc.collect()

        if len(partitioned_results) > 0:
            # Partitioned node ids are stored at the last index in each tuple of the partitioned results.
            node_id_ind = len(partitioned_results[0]) - 1
            fields: list[int] = []
            for ind in (
                node_feature_ind,
                node_packed_feature_ind,
                node_label_ind,
                node_id_ind,
            ):
                if ind is not None:
                    fields.append(ind)
            # Rows come back sorted by node id, which the range-based id2idx requires.
            assembled = self._assemble_node_chunks(
                partitioned_results, fields, node_id_ind
            )
            partitioned_ids = assembled[node_id_ind]
            if partitioned_ids.numel() != torch.unique(partitioned_ids).numel():
                raise ValueError(
                    f"Node ids are not unique for node type {node_type}. Please ensure that node ids are unique and contiguous across all machines from 0 to total_num_nodes - 1."
                )

            # We don't store the partitioned node ids in the returned data, since they are available from the range-based node partition book.
            del partitioned_ids

            gc.collect()

            if has_node_features:
                assert node_feature_ind is not None
                node_feature_partition_data = FeaturePartitionData(
                    feats=assembled[node_feature_ind],
                    ids=None,
                )

            else:
                node_feature_partition_data = None

            if has_node_packed_features:
                assert node_packed_feature_ind is not None
                node_packed_feature_partition_data = FeaturePartitionData(
                    feats=assembled[node_packed_feature_ind],
                    ids=None,
                )

            else:
                node_packed_feature_partition_data = None

            if has_node_labels:
                assert node_label_ind is not None
                node_label_partition_data = FeaturePartitionData(
                    feats=assembled[node_label_ind],
                    ids=None,
                )

            else:
                node_label_partition_data = None
            del assembled
        else:  # if len(partitioned_results) == 0
            if node_feat_dim is not None:
                node_feature_partition_data = FeaturePartitionData(
                    feats=torch.empty((0, node_feat_dim)),
                    ids=None,
                )
            else:
                node_feature_partition_data = None

            if node_packed_feat_dim is not None:
                node_packed_feature_partition_data = FeaturePartitionData(
                    feats=torch.empty((0, node_packed_feat_dim), dtype=torch.uint8),
                    ids=None,
                )
            else:
                node_packed_feature_partition_data = None

            if node_labels_dim is not None:
                node_label_partition_data = FeaturePartitionData(
                    feats=torch.empty((0, node_labels_dim)),
                    ids=None,
                )
            else:
                node_label_partition_data = None

        partitioned_results.clear()

        gc.collect()

        logger.info(
            f"Node feature and label partitioning for node type {node_type} finished, took {time.time() - start_time:.3f}s"
        )

        return (
            node_feature_partition_data,
            node_packed_feature_partition_data,
            node_label_partition_data,
        )

    # TODO: Create a dataclass for this positional four-value partition result.
    def _partition_edge_index_and_edge_features(
        self,
        node_partition_book: dict[NodeType, PartitionBook],
        edge_type: EdgeType,
    ) -> tuple[
        GraphPartitionData,
        Optional[FeaturePartitionData],
        Optional[FeaturePartitionData],
        Optional[PartitionBook],
    ]:
        """
        Partition graph topology of a specific edge type. We partition edges and edge features (if they exist)
        together. Once they have been partitioned across machines, we build the range-based edge partition book
        based on the number of edges assigned to each machine. Then, we infer the edge IDs from the edge partition
        book's ranges.

        If there are no edge features for the current edge type, both the returned edge feature and edge partition book will be None.

        Args:
            node_partition_book (dict[NodeType, PartitionBook]): The partition books of all graph nodes.
            edge_type (EdgeType): The edge type for input edges

        Returns:
            GraphPartitionData: The graph data of the current partition.
            Optional[FeaturePartitionData]: The edge features on the current partition, will be None if there are no edge features for the current edge type
            Optional[FeaturePartitionData]: The packed edge features on the current partition, will be None if there are no packed edge features for the current edge type
            Optional[PartitionBook]: The partition book of graph edges, will be None if there are no edge features for the current edge type
        """

        start_time = time.time()

        assert self._edge_index is not None, (
            "Must have registered edges prior to partitioning them"
        )

        edge_index = self._edge_index[edge_type]
        has_edge_feats = self._edge_feat is not None and edge_type in self._edge_feat
        has_edge_packed_feats = (
            self._edge_packed_feat is not None and edge_type in self._edge_packed_feat
        )
        has_edge_weights = (
            self._edge_weights is not None and edge_type in self._edge_weights
        )

        if not has_edge_feats:
            logger.info(
                f"No edge features detected for edge type {edge_type}, will only partition edge indices for this edge type."
            )

        edge_feat: Optional[torch.Tensor] = None
        edge_feat_dim: Optional[int] = None
        edge_weights_tensor: Optional[torch.Tensor] = None
        edge_packed_features: Optional[torch.Tensor] = None
        edge_packed_feature_dim: Optional[int] = None

        if has_edge_feats:
            assert self._edge_feat is not None and self._edge_feat_dim is not None
            assert edge_type in self._edge_feat_dim
            edge_feat = self._edge_feat[edge_type]
            edge_feat_dim = self._edge_feat_dim[edge_type]
        if has_edge_packed_feats:
            assert self._edge_packed_feat is not None
            assert self._edge_packed_feat_dim is not None
            edge_packed_features = self._edge_packed_feat[edge_type]
            edge_packed_feature_dim = self._edge_packed_feat_dim[edge_type]
        if has_edge_weights:
            assert self._edge_weights is not None
            edge_weights_tensor = self._edge_weights[edge_type]

        # Build input_data as (src, dst[, feat][, packed feat][, weights]).
        # Recorded indices keep result unpacking aligned with optional inputs.
        input_parts: list[torch.Tensor] = [edge_index[0], edge_index[1]]
        feat_idx: Optional[int] = None
        weight_idx: Optional[int] = None
        if edge_feat is not None:
            feat_idx = len(input_parts)
            input_parts.append(edge_feat)
        packed_feat_idx: Optional[int] = None
        if edge_packed_features is not None:
            packed_feat_idx = len(input_parts)
            input_parts.append(edge_packed_features)
        if edge_weights_tensor is not None:
            weight_idx = len(input_parts)
            input_parts.append(edge_weights_tensor)
        input_data: tuple[torch.Tensor, ...] = tuple(input_parts)

        if self._should_assign_edges_by_src_node:
            target_node_partition_book = node_partition_book[edge_type.src_node_type]
            target_indices = edge_index[0]
        else:
            target_node_partition_book = node_partition_book[edge_type.dst_node_type]
            target_indices = edge_index[1]

        def edge_partition_fn(rank_indices, _):
            return target_node_partition_book[rank_indices]

        res_list = self._partition_by_chunk(
            input_data=input_data,
            rank_indices=target_indices,
            partition_function=edge_partition_fn,
        )

        input_parts.clear()
        del (
            input_parts,
            input_data,
            edge_index,
            target_indices,
            edge_feat,
            edge_packed_features,
            edge_weights_tensor,
        )
        del self._edge_index[edge_type]
        if self._edge_feat is not None and edge_type in self._edge_feat:
            assert self._edge_feat_dim is not None
            del self._edge_feat[edge_type], self._edge_feat_dim[edge_type]
        if self._edge_weights is not None and edge_type in self._edge_weights:
            del self._edge_weights[edge_type]
        if self._edge_packed_feat is not None and edge_type in self._edge_packed_feat:
            assert self._edge_packed_feat_dim is not None
            del (
                self._edge_packed_feat[edge_type],
                self._edge_packed_feat_dim[edge_type],
            )

        # We check if edge_index or edge_feat dict is empty after deleting the tensor. If so, we set these fields to None.
        if not self._edge_index:
            self._edge_index = None
        if not self._edge_feat and not self._edge_feat_dim:
            self._edge_feat = None
            self._edge_feat_dim = None
        if self._edge_weights is not None and not self._edge_weights:
            self._edge_weights = None
        if self._edge_packed_feat is not None and not self._edge_packed_feat:
            self._edge_packed_feat = None
            self._edge_packed_feat_dim = None

        gc.collect()

        if len(res_list) == 0:
            partitioned_edge_index = torch.empty((2, 0))
            partitioned_edge_features = (
                torch.empty(0, edge_feat_dim) if edge_feat_dim is not None else None
            )
            partitioned_weights = torch.empty(0) if has_edge_weights else None
            partitioned_edge_packed_features = (
                torch.empty(0, edge_packed_feature_dim, dtype=torch.uint8)
                if edge_packed_feature_dim is not None
                else None
            )
        else:
            partitioned_edge_index = torch.stack(
                (
                    torch.cat([r[0] for r in res_list]),
                    torch.cat([r[1] for r in res_list]),
                ),
                dim=0,
            )
            partitioned_edge_features = (
                torch.cat([r[feat_idx] for r in res_list])
                if feat_idx is not None
                else None
            )
            partitioned_weights = (
                torch.cat([r[weight_idx] for r in res_list])
                if weight_idx is not None
                else None
            )
            partitioned_edge_packed_features = (
                torch.cat([r[packed_feat_idx] for r in res_list])
                if packed_feat_idx is not None
                else None
            )

        res_list.clear()
        gc.collect()

        # Generate range-based edge partition book and infer edge IDs for every
        # sidecar that requires sampled edge lookup.
        num_edges_on_each_rank: list[tuple[int, int]] = sorted(
            glt_rpc.all_gather((self._rank, partitioned_edge_index.size(1))).values(),
            key=lambda x: x[0],
        )
        partition_ranges: list[tuple[int, int]] = []
        start = 0
        for _, num_edges in num_edges_on_each_rank:
            end = start + num_edges
            partition_ranges.append((start, end))
            start = end

        if (
            edge_feat_dim is not None
            or edge_packed_feature_dim is not None
            or has_edge_weights
        ):
            edge_partition_book = RangePartitionBook(
                partition_ranges=partition_ranges, partition_idx=self._rank
            )
            partitioned_edge_ids = get_ids_on_rank(
                partition_book=edge_partition_book, rank=self._rank
            )
            current_graph_part = GraphPartitionData(
                edge_index=partitioned_edge_index,
                edge_ids=partitioned_edge_ids,
                weights=partitioned_weights,
            )
            current_feat_part = (
                FeaturePartitionData(feats=partitioned_edge_features, ids=None)
                if partitioned_edge_features is not None
                else None
            )
            logger.info(
                f"Got edge range-based partition book for edge type {edge_type} on rank {self._rank} with partition bounds: {edge_partition_book.partition_bounds}"
            )
        else:
            current_feat_part = None
            current_graph_part = GraphPartitionData(
                edge_index=partitioned_edge_index,
                edge_ids=None,
                weights=partitioned_weights,
            )
            edge_partition_book = None

        logger.info(
            f"Edge Index and Feature Partitioning for edge type {edge_type} finished, took {time.time() - start_time:.3f}s"
        )

        current_packed_feat_part = (
            FeaturePartitionData(feats=partitioned_edge_packed_features, ids=None)
            if partitioned_edge_packed_features is not None
            else None
        )
        return (
            current_graph_part,
            current_feat_part,
            current_packed_feat_part,
            edge_partition_book,
        )

    def _partition_label_edge_index(
        self,
        node_partition_book: dict[NodeType, PartitionBook],
        is_positive: bool,
        edge_type: EdgeType,
    ) -> torch.Tensor:
        """
        Partitions labels according to the node partition book.

        Args:
            node_partition_book (dict[NodeType, PartitionBook]): The partition book of nodes
            is_positive (bool): Whether positive labels are currently being registered. If False, negative labels will be partitioned.
            edge_type (EdgeType): Edge type of input data, must be specified if heterogeneous

        Returns:
            torch.Tensor: Edge index tensor of positive or negative labels, depending on is_positive flag
        """
        start_time = time.time()

        if edge_type.src_node_type not in node_partition_book:
            raise ValueError(
                f"Edge type {edge_type} source node type {edge_type.src_node_type} not found in the node partition book node keys: {node_partition_book.keys()}"
            )

        target_node_partition_book = node_partition_book[edge_type.src_node_type]
        if is_positive:
            assert self._positive_label_edge_index is not None, (
                "Must register positive labels prior to partitioning them"
            )
            label_edge_index = self._positive_label_edge_index[edge_type]
        else:
            assert self._negative_label_edge_index is not None, (
                "Must register negative labels prior to partitioning them"
            )
            label_edge_index = self._negative_label_edge_index[edge_type]

        def _label_pfn(source_node_ids, _):
            return target_node_partition_book[source_node_ids]

        # partitioned_chunks is a list of tuples. Each tuple is the the partitioned
        # result of a chunk of input data. The schema of each tuple is defined
        # by 'val'. In this case, each tuple contains source node IDs and destination
        # node IDs.
        partitioned_chunks = self._partition_by_chunk(
            input_data=(
                label_edge_index[0],
                label_edge_index[1],
            ),
            # 'partition_fn' takes 'val_indices' as input, uses it as keys for partition,
            # and returns the partition index.
            rank_indices=label_edge_index[0],
            partition_function=_label_pfn,
        )

        del label_edge_index

        if is_positive:
            # This assert is added to pass mypy type check, in practice we will not see this fail
            assert self._positive_label_edge_index is not None, (
                "Must register positive labels prior to partitioning them"
            )

            del self._positive_label_edge_index[edge_type]
            if len(self._positive_label_edge_index) == 0:
                self._positive_label_edge_index = None
        else:
            # This assert is added to pass mypy type check, in practice we will not see this fail
            assert self._negative_label_edge_index is not None, (
                "Must register negative labels prior to partitioning them"
            )

            del self._negative_label_edge_index[edge_type]
            if len(self._negative_label_edge_index) == 0:
                self._negative_label_edge_index = None

        gc.collect()

        # Combine the partitioned source and destination node IDs into a single 2D tensor
        if len(partitioned_chunks) == 0:
            partitioned_label_edge_index = torch.empty((2, 0))
        else:
            partitioned_label_edge_index = torch.stack(
                [
                    torch.cat([src_ids for src_ids, _ in partitioned_chunks]),
                    torch.cat([dst_ids for _, dst_ids in partitioned_chunks]),
                ],
                dim=0,
            )

        partitioned_chunks.clear()

        gc.collect()

        logger.info(
            f"Edge label partitioning for edge type {edge_type} finished, took {time.time() - start_time:.3f}s"
        )

        return partitioned_label_edge_index

    def partition_node(self) -> Union[PartitionBook, dict[NodeType, PartitionBook]]:
        """
        Partitions nodes of a graph. If heterogeneous, partitions nodes for all node types.

        Returns:
            Union[PartitionBook, dict[NodeType, PartitionBook]]: Partition Book of input nodes or Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        assert self._num_nodes is not None, (
            "Must have registered nodes prior to partitioning them"
        )

        logger.info("Partitioning Nodes ...")
        start_time = time.time()

        self._assert_data_type_consistency(
            input_entity=self._num_nodes, is_node_entity=True, is_subset=False
        )

        node_partition_book: dict[NodeType, PartitionBook] = {}
        for node_type in self._node_types:
            node_partition_book[node_type] = self._partition_node(node_type=node_type)

        elapsed_time = time.time() - start_time
        logger.info(f"Node Partitioning finished, took {elapsed_time:.3f}s")

        formatted_num_nodes = {
            node_type: f"{num_nodes:,}"
            for node_type, num_nodes in self._num_nodes.items()
        }

        if self._is_input_homogeneous:
            logger.info(
                f"Partitioned {to_homogeneous(formatted_num_nodes)} nodes for homogeneous dataset"
            )
            return to_homogeneous(node_partition_book)
        else:
            logger.info(f"Partitioned {formatted_num_nodes} nodes per node type")
            return node_partition_book

    def partition_node_features_and_labels(
        self, node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]]
    ) -> tuple[
        Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]],
        Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]],
        Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]],
    ]:
        """
        Partitions node features and labels of a graph. If heterogeneous, partitions features and labels for all node type.
        Must call `partition_node` first to get the node partition book as input.

        We should partition node features and labels together so that the ids of the partitioned node features and labels
        are the same, leading to an overall smaller memory footprint and faster partitioning.

        Note that we must have registered node ids with `register_node_ids` and obtaining a node partition book with `partition_node` before calling
        this function. If labels are not registered (i.e we don't call `register_node_labels`), we will only partition node features.
        Likewise, if features are not registered (i.e we don't call `register_node_features`), we will only partition node labels.
        At least one of node features or node labels must be registered prior to this function call.

        Args:
            node_partition_book (Union[PartitionBook, dict[NodeType, PartitionBook]]): The Computed Node Partition Book
        Returns:
            Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]]: Partitioned data of node features.
            Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]]: Partitioned data of packed uint8 node features.
            Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]]: Partitioned data of node labels.
        """
        assert self._num_nodes is not None and self._node_ids is not None, (
            "Node ids must be registered prior to partitioning."
        )

        self._assert_and_get_rpc_setup()

        logger.info("Partitioning Node Features and Labels ...")
        start_time = time.time()

        transformed_node_partition_book = (
            self._convert_node_entity_to_heterogeneous_format(
                input_node_entity=node_partition_book
            )
        )

        self._assert_data_type_consistency(
            input_entity=transformed_node_partition_book,
            is_node_entity=True,
            is_subset=False,
        )
        self._assert_data_type_consistency(
            input_entity=self._node_ids, is_node_entity=True, is_subset=False
        )

        node_feature_types: set[NodeType] = set()
        if self._node_feat is not None:
            self._assert_data_type_consistency(
                input_entity=self._node_feat, is_node_entity=True, is_subset=True
            )
            for node_type in self._node_feat.keys():
                node_feature_types.add(node_type)
        if self._node_packed_feat is not None:
            self._assert_data_type_consistency(
                input_entity=self._node_packed_feat,
                is_node_entity=True,
                is_subset=True,
            )
            for node_type in self._node_packed_feat.keys():
                node_feature_types.add(node_type)
        if self._node_labels is not None:
            self._assert_data_type_consistency(
                input_entity=self._node_labels, is_node_entity=True, is_subset=True
            )
            for node_type in self._node_labels.keys():
                node_feature_types.add(node_type)
        if not node_feature_types:
            raise ValueError(
                "Node features, packed node features, or labels must be registered prior to partitioning."
            )

        partitioned_node_features: dict[NodeType, FeaturePartitionData] = {}
        partitioned_node_packed_features: dict[NodeType, FeaturePartitionData] = {}
        partitioned_node_labels: dict[NodeType, FeaturePartitionData] = {}
        for node_type in sorted(node_feature_types):
            (
                partitioned_node_features_for_node_type,
                partitioned_node_packed_features_for_node_type,
                partitioned_node_labels_for_node_type,
            ) = self._partition_node_features_and_labels(
                node_partition_book=transformed_node_partition_book, node_type=node_type
            )
            if partitioned_node_features_for_node_type is not None:
                partitioned_node_features[node_type] = (
                    partitioned_node_features_for_node_type
                )
            if partitioned_node_packed_features_for_node_type is not None:
                partitioned_node_packed_features[node_type] = (
                    partitioned_node_packed_features_for_node_type
                )
            if partitioned_node_labels_for_node_type is not None:
                partitioned_node_labels[node_type] = (
                    partitioned_node_labels_for_node_type
                )

        elapsed_time = time.time() - start_time
        logger.info(f"Node Feature Partitioning finished, took {elapsed_time:.3f}s")

        if self._is_input_homogeneous:
            return (
                partitioned_node_features[DEFAULT_HOMOGENEOUS_NODE_TYPE]
                if partitioned_node_features
                else None,
                partitioned_node_packed_features[DEFAULT_HOMOGENEOUS_NODE_TYPE]
                if partitioned_node_packed_features
                else None,
                partitioned_node_labels[DEFAULT_HOMOGENEOUS_NODE_TYPE]
                if partitioned_node_labels
                else None,
            )
        else:
            return (
                partitioned_node_features if partitioned_node_features else None,
                partitioned_node_packed_features
                if partitioned_node_packed_features
                else None,
                partitioned_node_labels if partitioned_node_labels else None,
            )

    def partition_edge_index_and_edge_features(
        self, node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]]
    ) -> Union[
        tuple[
            GraphPartitionData,
            Optional[FeaturePartitionData],
            Optional[FeaturePartitionData],
            Optional[PartitionBook],
        ],
        tuple[
            dict[EdgeType, GraphPartitionData],
            Optional[dict[EdgeType, FeaturePartitionData]],
            Optional[dict[EdgeType, FeaturePartitionData]],
            Optional[dict[EdgeType, PartitionBook]],
        ],
    ]:
        """
        Partitions edges of a graph, including edge indices and edge features. If there are no edge features, only edge indices are partitioned.
        If heterogeneous, partitions edges/features for all edge types. Must call `partition_node` first to get the node partition book as input.

        Args:
            node_partition_book (Union[PartitionBook, dict[NodeType, PartitionBook]]): The computed Node Partition Book

        Returns:
            Union[
                tuple[GraphPartitionData, Optional[FeaturePartitionData], Optional[FeaturePartitionData], Optional[PartitionBook]],
                tuple[dict[EdgeType, GraphPartitionData], Optional[dict[EdgeType, FeaturePartitionData]], Optional[dict[EdgeType, FeaturePartitionData]], Optional[dict[EdgeType, PartitionBook]]],
            ]: Partitioned Graph Data, Feature Data, and corresponding edge partition book, is a dictionary if heterogeneous.
        """

        self._assert_and_get_rpc_setup()

        assert self._edge_index is not None and self._num_edges is not None, (
            "Must have registered edges prior to partitioning them"
        )

        logger.info("Partitioning Edges ...")
        start_time = time.time()

        transformed_node_partition_book = (
            self._convert_node_entity_to_heterogeneous_format(
                input_node_entity=node_partition_book
            )
        )

        self._assert_data_type_consistency(
            input_entity=transformed_node_partition_book,
            is_node_entity=True,
            is_subset=False,
        )

        self._assert_data_type_consistency(
            input_entity=self._edge_index, is_node_entity=False, is_subset=False
        )

        if self._edge_feat is not None:
            self._assert_data_type_consistency(
                input_entity=self._edge_feat, is_node_entity=False, is_subset=True
            )

        edge_partition_book: dict[EdgeType, PartitionBook] = {}
        partitioned_edge_index: dict[EdgeType, GraphPartitionData] = {}
        partitioned_edge_features: dict[EdgeType, FeaturePartitionData] = {}
        partitioned_edge_packed_features: dict[EdgeType, FeaturePartitionData] = {}
        for edge_type in self._edge_types:
            (
                partitioned_edge_index_per_edge_type,
                partitioned_edge_features_per_edge_type,
                partitioned_edge_packed_features_per_edge_type,
                edge_partition_book_per_edge_type,
            ) = self._partition_edge_index_and_edge_features(
                node_partition_book=transformed_node_partition_book, edge_type=edge_type
            )
            partitioned_edge_index[edge_type] = partitioned_edge_index_per_edge_type
            if edge_partition_book_per_edge_type is not None:
                edge_partition_book[edge_type] = edge_partition_book_per_edge_type
            if partitioned_edge_features_per_edge_type is not None:
                partitioned_edge_features[edge_type] = (
                    partitioned_edge_features_per_edge_type
                )
            if partitioned_edge_packed_features_per_edge_type is not None:
                partitioned_edge_packed_features[edge_type] = (
                    partitioned_edge_packed_features_per_edge_type
                )

        elapsed_time = time.time() - start_time
        logger.info(f"Edge Partitioning finished, took {elapsed_time:.3f}s")

        formatted_num_edges = {
            edge_type: f"{num_edges:,}"
            for edge_type, num_edges in self._num_edges.items()
        }

        # If partitioned_edge_features or edge_partition_book is empty, we return None. This is becauuse we assert
        # that any registered edge feature is non-empty, so if we encounter an empty dictionary in this case, this means
        # we never registered edge features and we can safely return None here.
        if self._is_input_homogeneous:
            logger.info(
                f"Partitioned {to_homogeneous(formatted_num_edges)} edges for homogeneous dataset"
            )
            return (
                to_homogeneous(partitioned_edge_index),
                to_homogeneous(partitioned_edge_features)
                if partitioned_edge_features
                else None,
                to_homogeneous(partitioned_edge_packed_features)
                if partitioned_edge_packed_features
                else None,
                to_homogeneous(edge_partition_book) if edge_partition_book else None,
            )
        else:
            logger.info(f"Partitioned {formatted_num_edges} edges per edge type")
            return (
                partitioned_edge_index,
                partitioned_edge_features if partitioned_edge_features else None,
                (
                    partitioned_edge_packed_features
                    if partitioned_edge_packed_features
                    else None
                ),
                edge_partition_book if edge_partition_book else None,
            )

    def partition_labels(
        self,
        node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]],
        is_positive: bool,
    ) -> Union[torch.Tensor, dict[EdgeType, torch.Tensor]]:
        """
        Partitions positive or negative labels of a graph. If heterogeneous, partitions labels for all edge type.
        Must call `partition_node` first to get the node partition book as input.
        Note that labels are always partitioned by the source node, since the edges are always assumed to be anchor_node -> to -> supervision node.
        Args:
            node_partition_book (Union[PartitionBook, dict[NodeType, PartitionBook]]): The computed Node Partition Book
            is_positive (bool): Whether positive labels are currently being registered. If False, negative labels will be partitioned.
        Returns:
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]: Returns the edge indices for partitioned positive or negative label, dependent on the is_positive flag
        """

        self._assert_and_get_rpc_setup()

        if is_positive:
            assert self._positive_label_edge_index is not None, (
                "Must register positive labels prior to partitioning them"
            )

            edge_label_types = sorted(self._positive_label_edge_index.keys())

            logger.info("Partitioning Positive Labels ...")

        else:
            assert self._negative_label_edge_index is not None, (
                "Must register negative labels partitioning them"
            )

            edge_label_types = sorted(self._negative_label_edge_index.keys())

            logger.info("Partitioning Negative Labels ...")

        start_time = time.time()

        transformed_node_partition_book = (
            self._convert_node_entity_to_heterogeneous_format(
                input_node_entity=node_partition_book
            )
        )

        self._assert_data_type_consistency(
            input_entity=transformed_node_partition_book,
            is_node_entity=True,
            is_subset=False,
        )

        partitioned_label_edge_index: dict[EdgeType, torch.Tensor] = {}
        for edge_type in edge_label_types:
            partitioned_label_edge_index[edge_type] = self._partition_label_edge_index(
                node_partition_book=transformed_node_partition_book,
                is_positive=is_positive,
                edge_type=edge_type,
            )

        elapsed_time = time.time() - start_time
        if is_positive:
            logger.info(
                f"Positive Label Partitioning finished, took {elapsed_time:.3f}s"
            )
        else:
            logger.info(
                f"Negative Label Partitioning finished, took {elapsed_time:.3f}s"
            )

        if self._is_input_homogeneous:
            return partitioned_label_edge_index[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
        else:
            return partitioned_label_edge_index

    def partition(
        self,
    ) -> PartitionOutput:
        """
        Calls partition on all registered fields. Note that at minimum nodes and edges must be registered when using this function.
        Returns:
            PartitionOutput: Reshuffled Outputs of Partitioning
        """

        self._assert_and_get_rpc_setup()

        logger.info(f"Rank {self._rank} starting partitioning ...")
        start_time = time.time()

        # The imported modules and registered inputs hold on the order of a million tracked objects, and every `gc.collect()` call during
        # partitioning scans them while holding the GIL, which also stalls the rpc handlers serving other machines.
        # Freezing moves them out of the collector's view; the collections that remain only scan objects created afterwards.
        # Objects frozen by the caller are left frozen.
        gc_frozen_by_caller = gc.get_freeze_count() > 0
        gc.collect()
        if not gc_frozen_by_caller:
            gc.freeze()

        # The try/finally does not handle errors; it guarantees the gc.unfreeze() below runs
        # when any partitioning step raises, so the process is not left with a frozen collector.
        try:
            # Node partition should happen at the very beginning, as edge partition
            # and label partition depends on node partition book.
            node_partition_book = self.partition_node()

            # Partition edge and clean up input edge data.
            (
                partitioned_edge_index,
                partitioned_edge_features,
                partitioned_edge_packed_features,
                edge_partition_book,
            ) = self.partition_edge_index_and_edge_features(
                node_partition_book=node_partition_book
            )

            if (
                self._node_feat is not None
                or self._node_packed_feat is not None
                or self._node_labels is not None
            ):
                (
                    partitioned_node_features,
                    partitioned_node_packed_features,
                    partitioned_node_labels,
                ) = self.partition_node_features_and_labels(
                    node_partition_book=node_partition_book
                )
            else:
                partitioned_node_features = None
                partitioned_node_packed_features = None
                partitioned_node_labels = None

            if self._positive_label_edge_index is not None:
                partitioned_positive_edge_index = self.partition_labels(
                    node_partition_book=node_partition_book, is_positive=True
                )
            else:
                partitioned_positive_edge_index = None

            if self._negative_label_edge_index is not None:
                partitioned_negative_edge_index = self.partition_labels(
                    node_partition_book=node_partition_book, is_positive=False
                )
            else:
                partitioned_negative_edge_index = None
        finally:
            if not gc_frozen_by_caller:
                gc.unfreeze()

        logger.info(
            f"Rank {self._rank} finished partitioning in {time.time() - start_time:.2f} seconds"
        )

        return PartitionOutput(
            node_partition_book=node_partition_book,
            edge_partition_book=edge_partition_book,
            partitioned_edge_index=partitioned_edge_index,
            partitioned_node_features=partitioned_node_features,
            partitioned_node_packed_features=partitioned_node_packed_features,
            partitioned_edge_features=partitioned_edge_features,
            partitioned_edge_packed_features=partitioned_edge_packed_features,
            partitioned_positive_labels=partitioned_positive_edge_index,
            partitioned_negative_labels=partitioned_negative_edge_index,
            partitioned_node_labels=partitioned_node_labels,
        )
