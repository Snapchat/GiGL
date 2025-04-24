import gc
import math
import time
from collections import abc, defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import graphlearn_torch.distributed.rpc as glt_rpc
import torch
from graphlearn_torch.distributed.dist_context import get_context
from graphlearn_torch.distributed.dist_random_partitioner import DistPartitionManager
from graphlearn_torch.partition import PartitionBook
from graphlearn_torch.utils import convert_to_tensor, index_select

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
    to_heterogeneous_edge,
    to_heterogeneous_node,
)

logger = Logger()


class _DistLinkPredicitonPartitionManager(DistPartitionManager):
    """
    Inherited from GLT's DistPartitionManager, this class is responsible for handling partitioning for tensor-based partition books.
    We implement this here to override the reset function.
    This is because we can save a lot of memory by using the minimum required dtype for our partition books.
    """

    def __init__(
        self, world_size: int, total_val_size: int = 1, generate_pb: bool = True
    ):
        """
        Initializes the partition manager.
        Args:
            world_size (int): The number of partitions (the total number of machines).
            total_val_size (int): The number of entities we should generate a partition book for.
            generate_pb (bool): Whether we should generate a partition book
        """
        if world_size <= 255:
            self._pb_dtype = torch.uint8
        elif world_size <= 65535:  # 2 ^ 16 - 1
            self._pb_dtype = torch.uint16
        elif world_size <= 4294967295:  # 2 ^ 32 - 1
            self._pb_dtype = torch.uint32
        else:
            self._pb_dtype = torch.uint64
        logger.info(
            f"Since the world size is {world_size}, using dtype of {self._pb_dtype} for partition book"
        )
        super().__init__(total_val_size, generate_pb)

    def reset(self, total_val_size: int, generate_pb: bool = True):
        """
        Resets the partition book and current values for partitioning.
        Args:
            total_val_size (int): Total size of partition book to generate
            generate_pb (bool): Whether we should generate a partition book
        """
        self.partition_book: Optional[torch.Tensor]

        with self._lock:
            self.generate_pb: bool = generate_pb
            self.cur_part_val_list: List[Tuple[torch.Tensor, ...]] = []
            if self.generate_pb:
                # This is the only difference from DistPartitionManager's reset() function.
                self.partition_book = torch.zeros(total_val_size, dtype=self._pb_dtype)
            else:
                self.partition_book = None


class DistLinkPredictionDataPartitioner:
    """
    This class is based on GLT's DistRandomPartitioner class (https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_random_partitioner.py)
    and has been optimized for better flexibility and memory management. We assume that init_rpc() and init_worker_group have been called to initialize the rpc and context,
    respectively, prior to this class. This class aims to partition homogeneous and heterogeneous input data, such as nodes,
    node features, edges, edge features, and any supervision labels across multiple machines. This class also produces partition books for edges and
    nodes, which are 1-d tensors that indicate which rank each node id and edge id are stored on. For example, the node partition book

    [0, 0, 1, 2]

    Means that node 0 is on rank 0, node 1 is on rank 0, node 2 is on rank 1, and node 3 is on rank 2.

    In this class, node and edge id and feature tensors can be passed in either through the constructor or the public register functions. It is required to have
    registered these tensors to the class prior to partitioning. For optimal memory management, it is recommended that the reference to these large tensors be deleted
    after being registered to the class but before partitioning, as maintaining both original and intermediate tensors can cause OOM concerns. Registering these tensors is available through both the constructor and the register functions to support
    the multiple use ways customers can use partitioning:

    Option 1: User wants to Partition just the nodes of a graph

    ```
    partitioner = DistLinkPredictionDataPartitioner()
    # Customer doesn't have to pass in excessive amounts of parameters to the constructor to partition only nodes
    partitioner.register_nodes(node_ids)
    del node_ids # Del reference to node_ids outside of DistLinkPredictionDataPartitioner to allow memory cleanup within the class
    partitioner.partition_nodes()
    # We may optionally want to call gc.collect() to ensure that any lingering memory is cleaned up, which may happen in cases where only a subset of inputs are partitioned (i.e no feats or labels)
    gc.collect()
    ```

    Option 2: User wants to partition all parts of a graph together and in sequence

    ```
    partitioner = DistLinkPredictionDataPartitioner(node_ids, edge_index, node_features, edge_features, pos_labels, neg_labels)
    # Register is called in the __init__ functions and doesn't need to be called at all outside the class.
    del (
        node_ids,
        edge_index,
        node_features,
        edge_features,
        pos_labels,
        neg_labels
    ) # Del reference to tensors outside of DistLinkPredictionDataPartitioner to allow memory cleanup within the class
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
        node_ids: Optional[Union[torch.Tensor, Dict[NodeType, torch.Tensor]]] = None,
        node_features: Optional[
            Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
        ] = None,
        edge_index: Optional[Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]] = None,
        edge_features: Optional[
            Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
        ] = None,
        positive_labels: Optional[
            Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
        ] = None,
        negative_labels: Optional[
            Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
        ] = None,
    ):
        """
        Initializes the parameters of the partitioner. Also optionally takes in node and edge tensors as arguments and registers them to the partitioner. Registered
        entities should be a dictionary of Dict[[NodeType or EdgeType], torch.Tensor] if heterogeneous or a torch.Tensor if homogeneous.
        This class assumes the distributed context has already been initialized outside of this class with the glt.distributed.init_worker_group()
        function and that rpc has been initialized with glt_distributed.init_rpc().
        Args:
            should_assign_edges_by_src_node (bool): Whether edges should be assigned to the machine of the source nodes during partitioning
            node_ids (Optional[Union[torch.Tensor, Dict[NodeType, torch.Tensor]]]): Optionally registered node ids from input. Tensors should be of shape [num_nodes_on_current_rank]
            node_features (Optional[Union[torch.Tensor, Dict[NodeType, torch.Tensor]]]): Optionally registered node feats from input. Tensors should be of shope [num_nodes_on_current_rank, node_feat_dim]
            edge_index (Optional[Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]]): Optionally registered edge indexes from input. Tensors should be of shape [2, num_edges_on_current_rank]
            edge_features (Optional[Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]]): Optionally registered edge features from input. Tensors should be of shape [num_edges_on_current_rank, edge_feat_dim]
            positive_labels (Optional[Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]]): Optionally registered positive labels from input. Tensors should be of shape [2, num_pos_labels_on_current_rank]
            negative_labels (Optional[Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]]): Optionally registered negative labels from input. Tensors should be of shape [2, num_neg_labels_on_current_rank]
        """

        self._world_size: int
        self._rank: int
        self._partition_mgr: _DistLinkPredicitonPartitionManager

        self._is_rpc_initialized: bool = False

        self._is_input_homogeneous: Optional[bool] = None
        self._should_assign_edges_by_src_node: bool = should_assign_edges_by_src_node
        self._edge_types: List[EdgeType] = []
        self._node_types: List[NodeType] = []
        self._num_nodes: Optional[Dict[NodeType, int]] = None
        self._num_edges: Optional[Dict[EdgeType, int]] = None

        self._node_ids: Optional[Dict[NodeType, torch.Tensor]] = None
        self._node_feat: Optional[Dict[NodeType, torch.Tensor]] = None
        self._node_feat_dim: Optional[Dict[NodeType, int]] = None

        self._edge_index: Optional[Dict[EdgeType, torch.Tensor]] = None
        self._edge_ids: Optional[Dict[EdgeType, torch.Tensor]] = None
        self._edge_feat: Optional[Dict[EdgeType, torch.Tensor]] = None
        self._edge_feat_dim: Optional[Dict[EdgeType, int]] = None

        self._positive_label_edge_index: Optional[Dict[EdgeType, torch.Tensor]] = None
        self._negative_label_edge_index: Optional[Dict[EdgeType, torch.Tensor]] = None

        if node_ids is not None:
            self.register_node_ids(node_ids=node_ids)

        if edge_index is not None:
            self.register_edge_index(edge_index=edge_index)

        if node_features is not None:
            self.register_node_features(node_features=node_features)

        if edge_features is not None:
            self.register_edge_features(edge_features=edge_features)

        if positive_labels is not None:
            self.register_labels(label_edge_index=positive_labels, is_positive=True)

        if negative_labels is not None:
            self.register_labels(label_edge_index=negative_labels, is_positive=False)

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
                assert (
                    not node_diff
                ), f"Found node types {node_diff} not contained in registered node types {self._node_types}"
            else:
                assert self._node_types == sorted(
                    input_entity.keys()
                ), f"Found different node input types {sorted(input_entity.keys())} from registered node types {self._node_types}"
        else:
            # We check that the input tensor edge types match the registered edge types.
            if is_subset:
                edge_diff = set(input_entity.keys()) - set(self._edge_types)
                assert (
                    not edge_diff
                ), f"Found edge types {edge_diff} not contained in registered edge types {self._edge_types}"
            else:
                assert self._edge_types == sorted(
                    input_entity.keys()
                ), f"Found different edge input types {sorted(input_entity.keys())} from registered edge types {self._edge_types}"

    def _assert_and_get_rpc_setup(self) -> None:
        """
        Asserts RPC and worker context have been initialized. If this is the first time this is called,
        sets the rank, world_size, and partition manager fields of the partitioner class from the distributed context.
        """
        assert (
            get_context() is not None
        ), "Distributed context must be initialized by the user prior to partitioning or registering fields by calling glt.distributed.init_worker_group()"

        assert (
            glt_rpc.rpc_is_initialized()
        ), "rpc must be initialized by the user prior to partitioning or registering fields to partitioning by calling glt.distributed.init_rpc()"

        # If rank, world size, and partition_manager are not set, we set them once we know the context and rpc has been initialized.
        if not self._is_rpc_initialized:
            logger.info(
                f"Machine {get_context().rank} setting up partition manager ..."
            )
            self._rank = get_context().rank
            self._world_size = get_context().world_size
            self._partition_mgr = _DistLinkPredicitonPartitionManager(
                world_size=self._world_size
            )
        self._is_rpc_initialized = True

    def _convert_node_entity_to_heterogeneous_format(
        self,
        input_node_entity: Union[
            torch.Tensor,
            PartitionBook,
            Dict[NodeType, torch.Tensor],
            Dict[NodeType, PartitionBook],
        ],
    ) -> Union[Dict[NodeType, torch.Tensor], Dict[NodeType, PartitionBook]]:
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

    def _convert_edge_entity_to_heterogeneous_format(
        self,
        input_edge_entity: Union[
            torch.Tensor,
            PartitionBook,
            Dict[EdgeType, torch.Tensor],
            Dict[EdgeType, PartitionBook],
        ],
    ) -> Union[Dict[EdgeType, torch.Tensor], Dict[EdgeType, PartitionBook]]:
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
        self, node_ids: Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
    ) -> None:
        """
        Registers the node ids to the partitioner. Also computes additional fields for partitioning such as the total number of nodes across all ranks.

        For optimal memory management, it is recommended that the reference to the node_id tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        Args:
            node_ids (Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): Input node_ids which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        logger.info("Registering Nodes ...")
        input_node_ids = self._convert_node_entity_to_heterogeneous_format(
            input_node_entity=node_ids
        )

        assert (
            input_node_ids
        ), "Node ids is an empty dictionary. Please provide node ids to register."

        self._node_types = sorted(input_node_ids.keys())

        self._node_ids = convert_to_tensor(input_node_ids, dtype=torch.int64)

        # This tuple here represents a (rank, num_nodes_on_rank) pair on a given partition, specified by the str key of the dictionary of format `distributed_random_partitoner_{rank}`.
        # num_nodes_on_rank is a Dict[NodeType, int].
        # Gathered_num_nodes is then used to identify the number of nodes on each rank, allowing us to access the total number of nodes across all ranks
        gathered_node_info: Dict[str, Tuple[int, Dict[NodeType, int]]]
        self._num_nodes = defaultdict(int)

        node_type_to_num_nodes: Dict[NodeType, int] = {
            node_type: input_node_ids[node_type].size(0)
            for node_type in sorted(input_node_ids.keys())
        }

        # Gathering to compute the number of nodes on each rank for each node type
        gathered_node_info = glt_rpc.all_gather((self._rank, node_type_to_num_nodes))

        # Looping through each of the registered node types in the graph
        for node_type in self._node_types:
            # Computing total number of nodes across all ranks of type `node_type`
            for (
                _,
                gathered_node_type_to_num_nodes,
            ) in gathered_node_info.values():
                self._num_nodes[node_type] += gathered_node_type_to_num_nodes[node_type]

    def register_edge_index(
        self, edge_index: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    ) -> None:
        """
        Registers the edge_index to the partitioner. Also computes additional fields for partitioning such as the total number of edges across all ranks and the number of
        edges on the current rnak.

        For optimal memory management, it is recommended that the reference to edge_index tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        Args:
            edge_index (Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]): Input edge index which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        logger.info("Registering Edge Indices ...")

        input_edge_index = self._convert_edge_entity_to_heterogeneous_format(
            input_edge_entity=edge_index
        )

        assert (
            input_edge_index
        ), "Edge Index is an empty dictionary. Please provide edge indices to register."

        self._edge_types = sorted(input_edge_index.keys())

        self._edge_index = convert_to_tensor(input_edge_index, dtype=torch.int64)

        # The tuple here represents a (rank, num_edges_on_rank) pair on a given partition, specified by the str key of the dictionary of format `distributed_random_partitoner_{rank}`
        # num_edges_on_rank is a Dict[EdgeType, int].
        # Gathered_num_edges is then used to identify the number of edges on each rank, allowing us to access the total number of edges across all ranks
        gathered_edge_info: Dict[str, Tuple[int, Dict[EdgeType, int]]]
        self._num_edges = {}
        edge_ids: Dict[EdgeType, torch.Tensor] = {}
        edge_type_to_num_edges: Dict[EdgeType, int] = {
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

            # Calculating the first edge id on the current rank by calculating the total number of edges prior to current rank
            start = sum(num_edges_all_ranks[: self._rank])

            # Calculating the last edge id on current rank by adding adding number of edges on the current rank to the start id
            end = start + num_edges_all_ranks[self._rank]

            # Setting total number of edges across all ranks
            self._num_edges[edge_type] = sum(num_edges_all_ranks)

            # Setting all the edge ids on the current rank
            edge_ids[edge_type] = torch.arange(start, end)

        self._edge_ids = convert_to_tensor(edge_ids, dtype=torch.int64)

    def register_node_features(
        self, node_features: Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
    ) -> None:
        """
        Registers the node features to the partitioner.

        For optimal memory management, it is recommended that the reference to node_features tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        We do not need to perform `all_gather` calls here since register_node_ids is responsible for determining total number of nodes
        across all ranks.
        Args:
            node_features(Union[torch.Tensor, Dict[NodeType, torch.Tensor]]): Input node features which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        logger.info("Registering Node Features ...")

        input_node_features = self._convert_node_entity_to_heterogeneous_format(
            input_node_entity=node_features
        )

        assert (
            input_node_features
        ), "Node features is an empty dictionary. Please provide node features to register."

        self._node_feat = convert_to_tensor(input_node_features, dtype=torch.float32)
        self._node_feat_dim = {}
        for node_type in input_node_features:
            self._node_feat_dim[node_type] = input_node_features[node_type].shape[1]

    def register_edge_features(
        self, edge_features: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    ) -> None:
        """
        Registers the edge features to the partitioner.

        For optimal memory management, it is recommended that the reference to edge_features tensor be deleted
        after calling this function using del <tensor>, as maintaining both original and intermediate tensors can cause OOM concerns.
        We do not need to perform `all_gather` calls here since register_edge_index is responsible for determining total number of edges
        across all ranks and inferrring edge ids.
        Args:
            edge_features(Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]): Input edge features which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        logger.info("Registering Edge Features ...")

        input_edge_features = self._convert_edge_entity_to_heterogeneous_format(
            input_edge_entity=edge_features
        )

        assert (
            input_edge_features
        ), "Edge features is an empty dictionary. Please provide edge features to register."

        self._edge_feat = convert_to_tensor(input_edge_features, dtype=torch.float32)
        self._edge_feat_dim = {}
        for edge_type in input_edge_features:
            self._edge_feat_dim[edge_type] = input_edge_features[edge_type].shape[1]

    def register_labels(
        self,
        label_edge_index: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]],
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
            label_edge_index (Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]): Input positive or negative labels which is either a torch.Tensor if homogeneous or a Dict if heterogeneous
            is_positive (bool): Whether positive labels are currently being registered. If False, labels will be registered as negative
        """

        self._assert_and_get_rpc_setup()

        input_label_edge_index = self._convert_edge_entity_to_heterogeneous_format(
            input_edge_entity=label_edge_index
        )

        assert (
            input_label_edge_index
        ), "Label edge index is an empty dictionary. Please provide label edge indices to register."

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

    def _partition_single_chunk_data(
        self,
        input_data: Optional[Tuple[torch.Tensor, ...]],
        rank_indices: torch.Tensor,
        partition_function: Callable[[torch.Tensor, Tuple[int, int]], torch.Tensor],
        chunk_start_pos: int,
        chunk_end_pos: int,
    ) -> None:
        """
        Partitions a single chunk of data across multiple machines. First, the partition function is used to lookup or compute the rank of the current input.
        Then, we loop over all the ranks and, for each rank, the inputs are masked to only contain the information belonging to that rank. We then send that
        information to other machines using the partition manager.
        Args:
            input_data (Optional[Tuple[torch.Tensor, ...]]): generic data type of items to be partitioned on the current chunk, which any information that should be partitioned across machines.
            rank_indices (torch.Tensor): torch tensor of indices which are used to determine the rank of each item to be partitioned on the current chunk
            partition_function (Callable): Function for determining ranks of current chunk. The first argument to this function is
                the specified indices in the chunk range while the second argument is the chunk start and end values. It returns a tuple indicating the rank
                of each item in the chunk.
            chunk_start_pos (int): The starting position of the current chunk being partitioned
            chunk_end_pos (int): The ending position of the current chunk being partitioned
        """
        # chunk_res is a list where index `i` corresponds to Tuple[input_data_on_i, rank_indices_on_i]
        chunk_res: List[Tuple[Optional[Tuple[torch.Tensor, ...]], torch.Tensor]] = []
        chunk_length = chunk_end_pos - chunk_start_pos
        chunk_rank = partition_function(rank_indices, (chunk_start_pos, chunk_end_pos))

        for rank in range(self._world_size):
            # Filtering for items in chunk which are on the current partition `rank`
            current_rank_mask = chunk_rank == rank
            per_rank_indices = torch.masked_select(
                torch.arange(chunk_length, dtype=torch.long), current_rank_mask
            )
            chunk_res.append(
                (
                    index_select(input_data, per_rank_indices),
                    rank_indices[per_rank_indices],
                )
            )
        self._partition_mgr.process(chunk_res)

    def _partition_by_chunk(
        self,
        input_data: Optional[Tuple[torch.Tensor, ...]],
        rank_indices: torch.Tensor,
        partition_function: Callable[[torch.Tensor, Tuple[int, int]], torch.Tensor],
        total_val_size: int,
        generate_pb: bool = True,
    ) -> Tuple[List[Tuple[torch.Tensor, ...]], Optional[PartitionBook]]:
        r"""Partitions input data chunk by chunk.
        Args:
            input_data (Optional[Tuple[torch.Tensor, ...]]): generic data type of items to be partitioned across machine, which any information that should be partitioned across machines.
            rank_indices (torch.Tensor): torch tensor of indices which are used to determine the rank of each item to be partitioned
            partition_function (Callable): Function for determining ranks of current chunk. The first argument to this function is
                the specified indices in the chunk range while the second argument is the chunk start and end values. It returns a tuple indicating the rank
                of each item in the chunk.
            total_val_size (int): The size of the partition book
            generate_pb (bool): Whether a partition book should be generated, defaults to True. This should only be set to true if partitioning nodes or edges,
                and should be false if partitioning node features or edge features.
        Return:
            List[Tuple[torch.Tensor, ...]]: Partitioned results of the input generic data type
            Optional[PartitionBook]: Partition Book if `generate_pb` is True, returns None if `generate_pb` is False
        """
        # TODO (mkolodner-sc): Investigate range-based partitioning
        num_items = len(rank_indices)

        # We currently hard-code the chunk_num to be 4 unless the number of items is less than 4, and determine the chunk size based on this value.
        # If this is not performant, we may revisit this in the future.
        chunk_num = min(num_items, 4)
        if chunk_num != 0:
            chunk_size = math.ceil(num_items / chunk_num)
        else:
            chunk_size = 0

        # This is set to 0 since the the data that is provided is already per-rank, and we begin at index 0 of this local data.
        chunk_start_pos = 0

        # Resets the partition manager's partition list and partition book fields
        # If generate_pb is False, self._partition_mgr.partition_book is set to None, otherwise it is set to torch.zeros(total_val_size, dtype=torch.int64)
        self._partition_mgr.reset(
            total_val_size=total_val_size, generate_pb=generate_pb
        )
        glt_rpc.barrier()

        # Rather than processing all of the tensors at once, we batch the tensors into chunks and process them separately.
        # Doing so yields performance improvement over processesing all of the tensor at once or processing each item individually.
        for _ in range(chunk_num):
            chunk_end_pos = min(num_items, chunk_start_pos + chunk_size)
            self._partition_single_chunk_data(
                input_data=index_select(
                    input_data, index=(chunk_start_pos, chunk_end_pos)
                ),
                rank_indices=rank_indices[chunk_start_pos:chunk_end_pos],
                partition_function=partition_function,
                chunk_start_pos=chunk_start_pos,
                chunk_end_pos=chunk_end_pos,
            )

            chunk_start_pos += chunk_size

            # The garbage collection here significantly reduces peak memory usage.
            gc.collect()

        glt_rpc.barrier()

        return (
            self._partition_mgr.cur_part_val_list,
            self._partition_mgr.partition_book,
        )

    def _partition_node(self, node_type: NodeType) -> PartitionBook:
        r"""Partition graph nodes of a specific node type.

        Args:
        node_type (NodeType): The node type for input nodes

        Returns:
        PartitionBook: The partition book of graph nodes.
        """

        assert (
            self._num_nodes is not None
        ), "Must have registered nodes prior to partitioning them"

        num_nodes = self._num_nodes[node_type]

        per_node_num = num_nodes // self._world_size
        local_node_start = per_node_num * self._rank

        local_node_end = min(num_nodes, per_node_num * (self._rank + 1))

        local_node_ids = torch.arange(
            local_node_start, local_node_end, dtype=torch.int64
        )

        # TODO (mkolodner-sc): Explore other node partitioning strategies here beyond random permutation
        def _node_pfn(n_ids, _):
            partition_idx = n_ids % self._world_size
            rand_order = torch.randperm(len(n_ids))
            return partition_idx[rand_order]

        partitioned_results, node_partition_book = self._partition_by_chunk(
            input_data=None,
            rank_indices=local_node_ids,
            partition_function=_node_pfn,
            total_val_size=num_nodes,
            generate_pb=True,
        )

        assert (
            node_partition_book is not None
        ), "Ensure `generate_pb` is set to true prior to calling _partition_by_chunk for node partitioning"

        del local_node_ids

        partitioned_results.clear()
        gc.collect()

        return node_partition_book

    def _partition_node_features(
        self,
        node_partition_book: Dict[NodeType, PartitionBook],
        node_type: NodeType,
    ) -> FeaturePartitionData:
        """
        Partitions node features according to the node partition book.

        Args:
            node_partition_book (Dict[NodeType, PartitionBook]): The partition book of nodes
            node_type (NodeType): Node type of input data

        Returns:
            FeaturePartitionData: Ids and Features of input nodes
        """

        assert (
            self._node_feat is not None
            and self._num_nodes is not None
            and self._node_ids is not None
            and self._node_feat_dim is not None
        ), "Node features and ids must be registered prior to partitioning."

        target_node_partition_book = node_partition_book[node_type]
        node_features = self._node_feat[node_type]
        node_ids = self._node_ids[node_type]
        num_nodes = self._num_nodes[node_type]
        node_feat_dim = self._node_feat_dim[node_type]

        def _node_feature_partition_fn(node_feature_ids, _):
            return target_node_partition_book[node_feature_ids]

        # partitioned_results is a list of tuples. Each tuple correpsonds
        # to a chunk of data. A tuple contains node features and node ids.
        partitioned_results, _ = self._partition_by_chunk(
            input_data=(node_features, node_ids),
            rank_indices=node_ids,
            partition_function=_node_feature_partition_fn,
            total_val_size=num_nodes,
            generate_pb=False,
        )

        # Since node features are large, we would like to delete them whenever
        # they are not used to free memory.
        del node_features, node_ids, num_nodes

        del (
            self._node_feat[node_type],
            self._node_ids[node_type],
            self._num_nodes[node_type],
        )

        if len(self._node_feat) == 0:
            self._node_feat = None
            self._node_ids = None
            self._num_nodes = None

        gc.collect()

        if len(partitioned_results) == 0:
            feature_partition_data = FeaturePartitionData(
                feats=torch.empty((0, node_feat_dim)),
                ids=torch.empty(0),
            )
        else:
            feature_partition_data = FeaturePartitionData(
                feats=torch.cat([r[0] for r in partitioned_results]),
                ids=torch.cat([r[1] for r in partitioned_results]),
            )

        del self._node_feat_dim[node_type], node_feat_dim
        if len(self._node_feat_dim) == 0:
            self._node_feat_dim = None

        partitioned_results.clear()

        gc.collect()

        return feature_partition_data

    def _partition_edge_index_and_edge_features(
        self,
        node_partition_book: Dict[NodeType, PartitionBook],
        edge_type: EdgeType,
    ) -> Tuple[GraphPartitionData, Optional[FeaturePartitionData], PartitionBook]:
        r"""Partition graph topology and edge features of a specific edge type.

        Args:
            node_partition_book (Dict[NodeType, PartitionBook]): The partition books of all graph nodes.
            edge_type (EdgeType): The edge type for input edges

        Returns:
            GraphPartitionData: The graph data of the current partition.
            FeaturePartitionData: The edge features on the current partition
            PartitionBook: The partition book of graph edges.
        """

        assert (
            self._edge_index is not None
            and self._edge_ids is not None
            and self._num_edges is not None
        ), "Must have registered edges prior to partitioning them"

        # Partitioning Edge Indices

        edge_index = self._edge_index[edge_type]
        edge_ids = self._edge_ids[edge_type]
        num_edges = self._num_edges[edge_type]

        if self._should_assign_edges_by_src_node:
            target_node_partition_book = node_partition_book[edge_type.src_node_type]
            target_indices = edge_index[0]
        else:
            target_node_partition_book = node_partition_book[edge_type.dst_node_type]
            target_indices = edge_index[1]

        def _edge_pfn(_, chunk_range):
            chunk_target_indices = index_select(target_indices, chunk_range)
            return target_node_partition_book[chunk_target_indices]

        # TODO (mkolodner-sc): Investigate partitioning edge features as part of this input_data tuple
        edge_res_list, edge_partition_book = self._partition_by_chunk(
            input_data=(edge_index[0], edge_index[1], edge_ids),
            rank_indices=edge_ids,
            partition_function=_edge_pfn,
            total_val_size=num_edges,
            generate_pb=True,
        )

        # We add this check both to ensure generate_pb was set to True for above call and to correctly type edge_partition_book as a torch tensor
        assert (
            edge_partition_book is not None
        ), "Ensure `generate_pb` is set to true prior to calling _partition_by_chunk for edge partitioning"

        del edge_index, target_indices
        del self._edge_index[edge_type]

        if len(self._edge_index) == 0:
            self._edge_index = None

        gc.collect()

        if len(edge_res_list) == 0:
            partitioned_edge_index = torch.empty((2, 0))
            partitioned_edge_ids = torch.empty(0)
        else:
            partitioned_edge_index = torch.stack(
                (
                    torch.cat([r[0] for r in edge_res_list]),
                    torch.cat([r[1] for r in edge_res_list]),
                ),
                dim=0,
            )
            partitioned_edge_ids = torch.cat([r[2] for r in edge_res_list])

        current_graph_part = GraphPartitionData(
            edge_index=partitioned_edge_index,
            edge_ids=partitioned_edge_ids,
        )

        edge_res_list.clear()

        gc.collect()

        # Partitioning Edge Features

        if self._edge_feat is None or edge_type not in self._edge_feat:
            logger.info(
                f"No edge features detected for edge type {edge_type}, will only partition edge indices for this edge type."
            )
            current_feat_part = None
            del edge_ids
            del self._edge_ids[edge_type]
            if len(self._edge_ids) == 0:
                self._edge_ids = None
            gc.collect()
        else:
            assert self._edge_feat_dim is not None and edge_type in self._edge_feat_dim
            edge_feat = self._edge_feat[edge_type]
            edge_feat_dim = self._edge_feat_dim[edge_type]

            def _edge_feature_pfn(edge_feature_ids, _):
                return edge_partition_book[edge_feature_ids]

            # partitioned_results is a list of tuples. Each tuple correpsonds
            # to a chunk of data. A tuple contains edge features and edge ids.
            edge_feat_res_list, _ = self._partition_by_chunk(
                input_data=(edge_feat, edge_ids),
                rank_indices=edge_ids,
                partition_function=_edge_feature_pfn,
                total_val_size=num_edges,
                generate_pb=False,
            )
            del edge_feat, edge_ids
            del (
                self._edge_feat[edge_type],
                self._edge_feat_dim[edge_type],
                self._edge_ids[edge_type],
            )
            if len(self._edge_ids) == 0:
                self._edge_ids = None
            if len(self._edge_feat) == 0 and len(self._edge_feat_dim) == 0:
                self._edge_feat = None
                self._edge_feat_dim = None
            gc.collect()
            if len(edge_feat_res_list) == 0:
                partitioned_edge_features = torch.empty(0, edge_feat_dim)
                partitioned_edge_feat_ids = torch.empty(0)
            else:
                partitioned_edge_features = torch.cat(
                    [r[0] for r in edge_feat_res_list]
                )
                partitioned_edge_feat_ids = torch.cat(
                    [r[1] for r in edge_feat_res_list]
                )

            current_feat_part = FeaturePartitionData(
                feats=partitioned_edge_features, ids=partitioned_edge_feat_ids
            )

        return current_graph_part, current_feat_part, edge_partition_book

    def _partition_label_edge_index(
        self,
        node_partition_book: Dict[NodeType, PartitionBook],
        is_positive: bool,
        edge_type: EdgeType,
    ) -> torch.Tensor:
        """
        Partitions labels according to the node partition book.

        Args:
            node_partition_book (Dict[NodeType, PartitionBook]): The partition book of nodes
            is_positive (bool): Whether positive labels are currently being registered. If False, negative labels will be partitioned.
            edge_type (EdgeType): Edge type of input data, must be specified if heterogeneous

        Returns:
            torch.Tensor: Edge index tensor of positive or negative labels, depending on is_positive flag
        """

        target_node_partition_book = node_partition_book[edge_type.src_node_type]
        if is_positive:
            assert (
                self._positive_label_edge_index is not None
            ), "Must register positive labels prior to partitioning them"
            label_edge_index = self._positive_label_edge_index[edge_type]
        else:
            assert (
                self._negative_label_edge_index is not None
            ), "Must register negative labels prior to partitioning them"
            label_edge_index = self._negative_label_edge_index[edge_type]

        def _label_pfn(source_node_ids, _):
            return target_node_partition_book[source_node_ids]

        # partitioned_chunks is a list of tuples. Each tuple is the the partitioned
        # result of a chunk of input data. The schema of each tuple is defined
        # by 'val'. In this case, each tuple contains source node IDs and destination
        # node IDs.
        partitioned_chunks, _ = self._partition_by_chunk(
            input_data=(
                label_edge_index[0],
                label_edge_index[1],
            ),
            # 'partition_fn' takes 'val_indices' as input, uses it as keys for partition,
            # and returns the partition index.
            rank_indices=label_edge_index[0],
            partition_function=_label_pfn,
            total_val_size=label_edge_index[0].size(0),
            generate_pb=False,
        )

        del label_edge_index

        if is_positive:
            # This assert is added to pass mypy type check, in practice we will not see this fail
            assert (
                self._positive_label_edge_index is not None
            ), "Must register positive labels prior to partitioning them"

            del self._positive_label_edge_index[edge_type]
            if len(self._positive_label_edge_index) == 0:
                self._positive_label_edge_index = None
        else:
            # This assert is added to pass mypy type check, in practice we will not see this fail
            assert (
                self._negative_label_edge_index is not None
            ), "Must register negative labels prior to partitioning them"

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

        return partitioned_label_edge_index

    def partition_node(self) -> Union[PartitionBook, Dict[NodeType, PartitionBook]]:
        """
        Partitions nodes of a graph. If heterogeneous, partitions nodes for all node types.

        Returns:
            Union[PartitionBook, Dict[NodeType, PartitionBook]]: Partition Book of input nodes or Dict if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        assert (
            self._num_nodes is not None
        ), "Must have registered nodes prior to partitioning them"

        logger.info("Partitioning Nodes ...")
        start_time = time.time()

        self._assert_data_type_consistency(
            input_entity=self._num_nodes, is_node_entity=True, is_subset=False
        )

        node_partition_book: Dict[NodeType, PartitionBook] = {}
        for node_type in self._node_types:
            node_partition_book[node_type] = self._partition_node(node_type=node_type)

        elapsed_time = time.time() - start_time
        logger.info(f"Node Partitioning finished, took {elapsed_time:.3f}s")

        if self._is_input_homogeneous:
            # Converting heterogeneous input back to homogeneous
            return node_partition_book[DEFAULT_HOMOGENEOUS_NODE_TYPE]
        else:
            return node_partition_book

    def partition_node_features(
        self, node_partition_book: Union[PartitionBook, Dict[NodeType, PartitionBook]]
    ) -> Union[FeaturePartitionData, Dict[NodeType, FeaturePartitionData]]:
        """
        Partitions node features of a graph. If heterogeneous, partitions features for all node type.
        Must call `partition_node` first to get the node partition book as input.

        Args:
            node_partition_book (Union[PartitionBook, Dict[NodeType, PartitionBook]]): The Computed Node Partition Book
        Returns:
            Union[FeaturePartitionData, Dict[NodeType, FeaturePartitionData]]: Feature Partition Data of ids and features or Dict if heterogeneous.
        """
        assert (
            self._node_feat is not None
            and self._num_nodes is not None
            and self._node_ids is not None
        ), "Node features and ids must be registered prior to partitioning."

        self._assert_and_get_rpc_setup()

        logger.info("Partitioning Node Feats ...")
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
            input_entity=self._node_feat, is_node_entity=True, is_subset=True
        )
        self._assert_data_type_consistency(
            input_entity=self._node_ids, is_node_entity=True, is_subset=False
        )

        node_feature_types = sorted(self._node_feat.keys())

        partitioned_node_features: Dict[NodeType, FeaturePartitionData] = {}
        for node_type in node_feature_types:
            partitioned_node_features[node_type] = self._partition_node_features(
                node_partition_book=transformed_node_partition_book, node_type=node_type
            )

        elapsed_time = time.time() - start_time
        logger.info(f"Node Feature Partitioning finished, took {elapsed_time:.3f}s")

        if self._is_input_homogeneous:
            return partitioned_node_features[DEFAULT_HOMOGENEOUS_NODE_TYPE]
        else:
            return partitioned_node_features

    def partition_edge_index_and_edge_features(
        self, node_partition_book: Union[PartitionBook, Dict[NodeType, PartitionBook]]
    ) -> Union[
        Tuple[GraphPartitionData, Optional[FeaturePartitionData], PartitionBook],
        Tuple[
            Dict[EdgeType, GraphPartitionData],
            Optional[Dict[EdgeType, FeaturePartitionData]],
            Dict[EdgeType, PartitionBook],
        ],
    ]:
        """
        Partitions edges of a graph, including edge indices and edge features. If there are no edge features, only edge indices are partitioned.
        If heterogeneous, partitions edges/features for all edge types. Must call `partition_node` first to get the node partition book as input.
        Args:
            node_partition_book (Union[PartitionBook, Dict[NodeType, PartitionBook]]): The computed Node Partition Book
        Returns:
            Union[
                Tuple[GraphPartitionData, FeaturePartitionData, PartitionBook],
                Tuple[Dict[EdgeType, GraphPartitionData], Dict[EdgeType, FeaturePartitionData], Dict[EdgeType, PartitionBook]],
            ]: Partitioned Graph Data, Feature Data, and corresponding edge partition book, is a dictionary if heterogeneous
        """

        self._assert_and_get_rpc_setup()

        assert (
            self._edge_index is not None
            and self._edge_ids is not None
            and self._num_edges is not None
        ), "Must have registered edges prior to partitioning them"

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

        self._assert_data_type_consistency(
            input_entity=self._edge_ids, is_node_entity=False, is_subset=False
        )

        self._assert_data_type_consistency(
            input_entity=self._num_edges, is_node_entity=False, is_subset=False
        )

        if self._edge_feat is not None:
            self._assert_data_type_consistency(
                input_entity=self._edge_feat, is_node_entity=False, is_subset=True
            )

        edge_partition_book: Dict[EdgeType, PartitionBook] = {}
        partitioned_edge_index: Dict[EdgeType, GraphPartitionData] = {}
        partitioned_edge_features: Dict[EdgeType, FeaturePartitionData] = {}
        for edge_type in self._edge_types:
            (
                partitioned_edge_index_per_edge_type,
                partitioned_edge_features_per_edge_type,
                edge_partition_book_per_edge_type,
            ) = self._partition_edge_index_and_edge_features(
                node_partition_book=transformed_node_partition_book, edge_type=edge_type
            )
            partitioned_edge_index[edge_type] = partitioned_edge_index_per_edge_type
            edge_partition_book[edge_type] = edge_partition_book_per_edge_type
            if partitioned_edge_features_per_edge_type is not None:
                partitioned_edge_features[
                    edge_type
                ] = partitioned_edge_features_per_edge_type

        elapsed_time = time.time() - start_time
        logger.info(f"Edge Partitioning finished, took {elapsed_time:.3f}s")

        if self._is_input_homogeneous:
            return_edge_features = (
                partitioned_edge_features[DEFAULT_HOMOGENEOUS_EDGE_TYPE]
                if partitioned_edge_features
                else None
            )
            return (
                partitioned_edge_index[DEFAULT_HOMOGENEOUS_EDGE_TYPE],
                return_edge_features,
                edge_partition_book[DEFAULT_HOMOGENEOUS_EDGE_TYPE],
            )
        else:
            return (
                partitioned_edge_index,
                partitioned_edge_features,
                edge_partition_book,
            )

    def partition_labels(
        self,
        node_partition_book: Union[PartitionBook, Dict[NodeType, PartitionBook]],
        is_positive: bool,
    ) -> Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]:
        """
        Partitions positive or negative labels of a graph. If heterogeneous, partitions labels for all edge type.
        Must call `partition_node` first to get the node partition book as input.
        Args:
            node_partition_book (Union[PartitionBook, Dict[NodeType, PartitionBook]]): The computed Node Partition Book
            is_positive (bool): Whether positive labels are currently being registered. If False, negative labels will be partitioned.
        Returns:
            Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]: Returns the edge indices for partitioned positive or negative label, dependent on the is_positive flag
        """

        self._assert_and_get_rpc_setup()

        if is_positive:
            assert (
                self._positive_label_edge_index is not None
            ), "Must register positive labels prior to partitioning them"

            self._assert_data_type_consistency(
                input_entity=self._positive_label_edge_index,
                is_node_entity=False,
                is_subset=True,
            )

            edge_label_types = sorted(self._positive_label_edge_index.keys())

            logger.info("Partitioning Positive Labels ...")

        else:
            assert (
                self._negative_label_edge_index is not None
            ), "Must register negative labels partitioning them"

            self._assert_data_type_consistency(
                input_entity=self._negative_label_edge_index,
                is_node_entity=False,
                is_subset=True,
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

        partitioned_label_edge_index: Dict[EdgeType, torch.Tensor] = {}
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

        # Node partition should happen at the very beginning, as edge partition
        # and label partition depends on node partition book.
        node_partition_book = self.partition_node()

        # Partition edge and clean up input edge data.
        (
            partitioned_edge_index,
            partitioned_edge_features,
            edge_partition_book,
        ) = self.partition_edge_index_and_edge_features(
            node_partition_book=node_partition_book
        )

        if self._node_feat is not None:
            partitioned_node_features = self.partition_node_features(
                node_partition_book=node_partition_book
            )
        else:
            partitioned_node_features = None

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

        logger.info(
            f"Rank {self._rank} finished partitioning in {time.time() - start_time:.2f} seconds"
        )

        return PartitionOutput(
            node_partition_book=node_partition_book,
            edge_partition_book=edge_partition_book,
            partitioned_edge_index=partitioned_edge_index,
            partitioned_node_features=partitioned_node_features,
            partitioned_edge_features=partitioned_edge_features,
            partitioned_positive_labels=partitioned_positive_edge_index,
            partitioned_negative_labels=partitioned_negative_edge_index,
        )
