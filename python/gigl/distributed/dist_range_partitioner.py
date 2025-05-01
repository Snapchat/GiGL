import gc
import time
from typing import Dict, Optional, Union

import torch
from graphlearn_torch.distributed.rpc import all_gather
from graphlearn_torch.partition import PartitionBook, RangePartitionBook
from graphlearn_torch.utils import convert_to_tensor

from gigl.common.logger import Logger
from gigl.distributed.dist_partitioner import DistPartitioner
from gigl.distributed.utils.partition_book import get_ids_on_rank
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import FeaturePartitionData, GraphPartitionData, to_homogeneous

logger = Logger()


class DistRangePartitioner(DistPartitioner):
    """
    This class is responsible for implementing range-based partitioning. Rather than using a tensor-based partition
    book, this approach stores the upper bound of ids for each rank. For example, a range partition book [4, 8, 12]
    stores edge ids 0-3 on the 0th rank, 4-7 on the 1st rank, and 8-11 on the 2nd rank. While keeping the same
    id-indexing pattern for rank lookup as the tensor-based partitioning, this partition book does a search through
    these partition bounds to fetch the ranks, rather than using a direct index lookup. For example, to get the rank
    of node ids 1 and 6 by doing node_pb[[1, 6]], the range partition book uses torch.searchsorted on the partition
    bounds to return [0, 1], the ranks of each of these ids. As a result, the range-based partition book trades off
    more efficient memory storage for a slower lookup time for indices.
    """

    def register_edge_index(
        self, edge_index: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    ) -> None:
        """
        Registers the edge_index to the partitioner. Unlike the tensor-based partitioner, this register pattern
        does not automatically infer edge ids,as they are not needed for partitioning.

        For optimal memory management, it is recommended that the reference to edge_index tensor be deleted after
        calling this function using del <tensor>, as maintaining both original and intermediate tensors can
        cause OOM concerns.

        Args:
            edge_index (Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]): Input edge index which is either a
                torch.Tensor if homogeneous or a Dict if heterogeneous
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

    def _partition_node(self, node_type: NodeType) -> PartitionBook:
        """
        Partition graph nodes of a specific node type. For range-based partitioning, we partition all
        the nodes into continuous ranges so that the diff between lengths of any two ranges is no greater
        than 1. This function gets called by the `partition_node` API from the parent class, which handles
        the node partitioning across all node types.

        Args:
            node_type (NodeType): The node type for input nodes

        Returns:
            PartitionBook: The partition book of graph nodes.
        """

        assert (
            self._num_nodes is not None
        ), "Must have registered nodes prior to partitioning them"

        num_nodes = self._num_nodes[node_type]

        per_node_num, remainder = divmod(num_nodes, self._world_size)

        # We set `remainder` number of partitions to have at most one more item.

        start = 0
        partition_ranges: list[tuple[int, int]] = []
        for partition_index in range(self._world_size):
            if partition_index < remainder:
                end = start + per_node_num + 1
            else:
                end = start + per_node_num
            partition_ranges.append((start, end))
            start = end

        # Store and return partitioned ranges as GLT's RangePartitionBook
        node_partition_book = RangePartitionBook(
            partition_ranges=partition_ranges, partition_idx=self._rank
        )

        logger.info(
            f"Got node range-based partition book for node type {node_type} on rank {self._rank} with partition bounds: {node_partition_book.partition_bounds}"
        )

        return node_partition_book

    def _partition_node_features(
        self,
        node_partition_book: Dict[NodeType, PartitionBook],
        node_type: NodeType,
    ) -> FeaturePartitionData:
        """
        Partitions node features according to the node partition book. We rely on the functionality from the parent tensor-based partitioner here,
        and add logic to sort the node features by node indices which is specific to range-based partitioning. This is done so that the range-based
        id2idx corresponds correctly to the node features.

        Args:
            node_partition_book (Dict[NodeType, PartitionBook]): The partition book of nodes
            node_type (NodeType): Node type of input data

        Returns:
            FeaturePartitionData: Ids and Features of input nodes
        """
        features_partition_data = super()._partition_node_features(
            node_partition_book=node_partition_book, node_type=node_type
        )
        # The parent class always returns ids in the feature_partition_data, but we don't need to store the partitioned node feature ids for
        # range-based partitioning, since this is available from the node partition book.
        assert features_partition_data.ids is not None
        sorted_node_ids_indices = torch.argsort(features_partition_data.ids)
        partitioned_node_features = features_partition_data.feats[
            sorted_node_ids_indices
        ]
        return FeaturePartitionData(feats=partitioned_node_features, ids=None)

    def _partition_edge_index_and_edge_features(
        self,
        node_partition_book: Dict[NodeType, PartitionBook],
        edge_type: EdgeType,
    ) -> tuple[GraphPartitionData, Optional[FeaturePartitionData], PartitionBook]:
        """
        Partition graph topology of a specific edge type. For range-based partitioning, we partition
        edges and edge features (if they exist) together. Once they have been partitioned across machines,
        we build the edge partition book based on the number of edges assigned to each machine. Then, we infer
        the edge IDs from the edge partition book's ranges.

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
        ), "Must have registered edges prior to partitioning them"

        edge_index = self._edge_index[edge_type]

        input_data: tuple[torch.Tensor, ...]

        if self._edge_feat is None or edge_type not in self._edge_feat:
            logger.info(
                f"No edge features detected for edge type {edge_type}, will only partition edge indices for this edge type."
            )
            edge_feat = None
            edge_feat_dim = None
            input_data = (edge_index[0], edge_index[1])
        else:
            assert self._edge_feat_dim is not None and edge_type in self._edge_feat_dim
            edge_feat = self._edge_feat[edge_type]
            edge_feat_dim = self._edge_feat_dim[edge_type]
            input_data = (edge_index[0], edge_index[1], edge_feat)

        if self._should_assign_edges_by_src_node:
            target_node_partition_book = node_partition_book[edge_type.src_node_type]
            target_indices = edge_index[0]
        else:
            target_node_partition_book = node_partition_book[edge_type.dst_node_type]
            target_indices = edge_index[1]

        def edge_partition_fn(rank_indices, _):
            return target_node_partition_book[rank_indices]

        res_list, _ = self._partition_by_chunk(
            input_data=input_data,
            rank_indices=target_indices,
            partition_function=edge_partition_fn,
        )

        del edge_index, target_indices, edge_feat
        del self._edge_index[edge_type]
        if self._edge_feat is not None and edge_type in self._edge_feat:
            del self._edge_feat[edge_type]

        # We check if edge_index or edge_feat dict is empty after deleting the tensor. If so, we set these fields to None.
        if not self._edge_index:
            self._edge_index = None
        if not self._edge_feat and not self._edge_feat_dim:
            self._edge_feat = None
            self._edge_feat_dim = None

        gc.collect()

        if len(res_list) == 0:
            partitioned_edge_index = torch.empty((2, 0))
        else:
            partitioned_edge_index = torch.stack(
                (
                    torch.cat([r[0] for r in res_list]),
                    torch.cat([r[1] for r in res_list]),
                ),
                dim=0,
            )

        # Generating edge partition book

        num_of_edges_on_current_rank = partitioned_edge_index.size(1)
        num_edges_on_each_rank: list[tuple[int, int]] = sorted(
            all_gather((self._rank, num_of_edges_on_current_rank)).values(),
            key=lambda x: x[0],
        )

        partition_ranges: list[tuple[int, int]] = []
        start = 0
        for _, num_edges in num_edges_on_each_rank:
            end = start + num_edges
            partition_ranges.append((start, end))
            start = end

        edge_partition_book = RangePartitionBook(
            partition_ranges=partition_ranges, partition_idx=self._rank
        )
        partitioned_edge_ids = get_ids_on_rank(
            partition_book=edge_partition_book, rank=self._rank
        )

        current_graph_part = GraphPartitionData(
            edge_index=partitioned_edge_index,
            edge_ids=partitioned_edge_ids,
        )

        if edge_feat_dim is None:
            current_feat_part = None
        else:
            if len(res_list) == 0:
                partitioned_edge_features = torch.empty(0, edge_feat_dim)
            else:
                partitioned_edge_features = torch.cat([r[2] for r in res_list])
            # We don't need to store the partitioned edge feature ids for range-based partitioning, since this is available from the edge partition book
            current_feat_part = FeaturePartitionData(
                feats=partitioned_edge_features, ids=None
            )

        res_list.clear()

        gc.collect()

        logger.info(
            f"Got edge range-based partition book for edge type {edge_type} on rank {self._rank} with partition bounds: {edge_partition_book.partition_bounds}"
        )

        return current_graph_part, current_feat_part, edge_partition_book

    def partition_edge_index_and_edge_features(
        self, node_partition_book: Union[PartitionBook, Dict[NodeType, PartitionBook]]
    ) -> Union[
        tuple[GraphPartitionData, Optional[FeaturePartitionData], PartitionBook],
        tuple[
            Dict[EdgeType, GraphPartitionData],
            Optional[Dict[EdgeType, FeaturePartitionData]],
            Dict[EdgeType, PartitionBook],
        ],
    ]:
        """
        Partitions edges of a graph, including edge indices and edge features. If heterogeneous, partitions edges
        for all edge types. You must call `partition_node` first to get the node partition book as input. The difference
        between this function and its parent is that we no longer need to check that the `edge_ids` have been
        pre-computed as a prerequisite for partitioning edges and edge features.

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

        return_edge_features = (
            partitioned_edge_features if partitioned_edge_features else None
        )
        if self._is_input_homogeneous:
            return (
                to_homogeneous(partitioned_edge_index),
                to_homogeneous(return_edge_features),
                to_homogeneous(edge_partition_book),
            )
        else:
            return (
                partitioned_edge_index,
                return_edge_features,
                edge_partition_book,
            )
