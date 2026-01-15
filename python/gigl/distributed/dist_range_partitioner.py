import gc
import time
from typing import Optional, Union

import torch
from graphlearn_torch.distributed.rpc import all_gather
from graphlearn_torch.partition import PartitionBook, RangePartitionBook
from graphlearn_torch.utils import convert_to_tensor

from gigl.common.logger import Logger
from gigl.distributed.dist_partitioner import DistPartitioner
from gigl.distributed.utils.partition_book import build_partition_book, get_ids_on_rank
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
        self, edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    ) -> None:
        """
        Registers the edge_index to the partitioner. Unlike the tensor-based partitioner, this register pattern
        does not automatically infer edge ids,as they are not needed for partitioning.

        For optimal memory management, it is recommended that the reference to edge_index tensor be deleted after
        calling this function using del <tensor>, as maintaining both original and intermediate tensors can
        cause OOM concerns.

        Args:
            edge_index (Union[torch.Tensor, dict[EdgeType, torch.Tensor]]): Input edge index which is either a
                torch.Tensor if homogeneous or a dict if heterogeneous
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

        # Logging information about number of edges across the machines

        edge_type_to_num_edges: dict[EdgeType, int] = {
            edge_type: input_edge_index[edge_type].size(1)
            for edge_type in sorted(input_edge_index.keys())
        }

        # The tuple here represents a (rank, num_edges_on_rank) pair on a given partition, specified by the str key of the dictionary of format `distributed_random_partitoner_{rank}`
        # num_edges_on_rank is a dict[EdgeType, int].
        # Gathered_num_edges is then used to identify the number of edges on each rank, allowing us to access the total number of edges across all ranks
        gathered_edge_info: dict[str, tuple[int, dict[EdgeType, int]]]

        # Gathering to compute the number of edges on each rank for each edge type
        gathered_edge_info = all_gather((self._rank, edge_type_to_num_edges))

        self._num_edges = {}

        # Looping through registered edge types in graph
        for edge_type in self._edge_types:
            # Populating num_edges_all_ranks list, where num_edges_all_ranks[i] = num_edges means that rank `i`` has `num_edges` edges
            num_edges_all_ranks = [0] * self._world_size
            for (
                rank,
                gathered_edge_type_to_num_edges,
            ) in gathered_edge_info.values():
                num_edges_all_ranks[rank] = gathered_edge_type_to_num_edges[edge_type]

            self._num_edges[edge_type] = sum(num_edges_all_ranks)

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

        start_time = time.time()

        assert (
            self._num_nodes is not None
        ), "Must have registered nodes prior to partitioning them"

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

    def _partition_node_features_and_labels(
        self,
        node_partition_book: dict[NodeType, PartitionBook],
        node_type: NodeType,
    ) -> tuple[Optional[FeaturePartitionData], Optional[FeaturePartitionData]]:
        """
        Partitions node features according to the node partition book. We rely on the functionality from the parent tensor-based partitioner here,
        and add logic to sort the node features by node indices which is specific to range-based partitioning. This is done so that the range-based
        id2idx corresponds correctly to the node features.

        Args:
            node_partition_book (dict[NodeType, PartitionBook]): The partition book of nodes
            node_type (NodeType): Node type of input data

        Returns:
            FeaturePartitionData: Ids and Features of input nodes
        """
        (
            feature_partition_data,
            labels_partition_data,
        ) = super()._partition_node_features_and_labels(
            node_partition_book=node_partition_book, node_type=node_type
        )

        # The parent class always returns ids in the feature_partition_data, but we don't need to store the partitioned node feature ids for
        # range-based partitioning, since this is available from the node partition book.

        if feature_partition_data is not None:
            ids = feature_partition_data.ids
            assert ids is not None
            sorted_node_ids_indices = torch.argsort(ids)
            partitioned_node_features = feature_partition_data.feats[
                sorted_node_ids_indices
            ]
            partitioned_node_feature_data = FeaturePartitionData(
                feats=partitioned_node_features, ids=None
            )

            del sorted_node_ids_indices
            gc.collect()
        else:
            partitioned_node_feature_data = None

        if labels_partition_data is not None:
            ids = labels_partition_data.ids
            assert ids is not None
            sorted_node_ids_indices = torch.argsort(ids)
            partitioned_node_labels = labels_partition_data.feats[
                sorted_node_ids_indices
            ]
            partitioned_node_label_data = FeaturePartitionData(
                feats=partitioned_node_labels, ids=None
            )

            del sorted_node_ids_indices
            gc.collect()
        else:
            partitioned_node_label_data = None

        return partitioned_node_feature_data, partitioned_node_label_data

    def _partition_edge_index_and_edge_features(
        self,
        node_partition_book: dict[NodeType, PartitionBook],
        edge_type: EdgeType,
    ) -> tuple[
        GraphPartitionData, Optional[FeaturePartitionData], Optional[PartitionBook]
    ]:
        """
        Partition graph topology of a specific edge type. For range-based partitioning, we partition
        edges and edge features (if they exist) together. Once they have been partitioned across machines,
        we build the edge partition book based on the number of edges assigned to each machine. Then, we infer
        the edge IDs from the edge partition book's ranges.

        If there are no edge features for the current edge type, both the returned edge feature and edge partition book will be None.

        Args:
            node_partition_book (dict[NodeType, PartitionBook]): The partition books of all graph nodes.
            edge_type (EdgeType): The edge type for input edges

        Returns:
            GraphPartitionData: The graph data of the current partition.
            Optional[FeaturePartitionData]: The edge features on the current partition, will be None if there are no edge features for the current edge type
            Optional[PartitionBook]: The partition book of graph edges, will be None if there are no edge features for the current edge type
        """

        start_time = time.time()

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

        del input_data, edge_index, target_indices, edge_feat
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

        if edge_feat_dim is not None:
            if len(res_list) == 0:
                partitioned_edge_features = torch.empty(0, edge_feat_dim)
            else:
                partitioned_edge_features = torch.cat([r[2] for r in res_list])

        res_list.clear()

        gc.collect()

        # Generating edge partition book

        num_edges_on_each_rank: list[tuple[int, int]] = sorted(
            all_gather((self._rank, partitioned_edge_index.size(1))).values(),
            key=lambda x: x[0],
        )
        partition_ranges: list[tuple[int, int]] = []
        start = 0
        for _, num_edges in num_edges_on_each_rank:
            end = start + num_edges
            partition_ranges.append((start, end))
            start = end

        if edge_feat_dim is not None:
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
            current_feat_part = FeaturePartitionData(
                feats=partitioned_edge_features, ids=None
            )
            logger.info(
                f"Got edge range-based partition book for edge type {edge_type} on rank {self._rank} with partition bounds: {edge_partition_book.partition_bounds}"
            )
        else:
            current_feat_part = None
            current_graph_part = GraphPartitionData(
                edge_index=partitioned_edge_index,
                edge_ids=None,
            )
            edge_partition_book = None

        logger.info(
            f"Edge Index and Feature Partitioning for edge type {edge_type} finished, took {time.time() - start_time:.3f}s"
        )

        return current_graph_part, current_feat_part, edge_partition_book

    def partition_edge_index_and_edge_features(
        self, node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]]
    ) -> Union[
        tuple[
            GraphPartitionData, Optional[FeaturePartitionData], Optional[PartitionBook]
        ],
        tuple[
            dict[EdgeType, GraphPartitionData],
            Optional[dict[EdgeType, FeaturePartitionData]],
            Optional[dict[EdgeType, PartitionBook]],
        ],
    ]:
        """
        Partitions edges of a graph, including edge indices and edge features. If heterogeneous, partitions edges
        for all edge types. You must call `partition_node` first to get the node partition book as input. The difference
        between this function and its parent is that we no longer need to check that the `edge_ids` have been
        pre-computed as a prerequisite for partitioning edges and edge features.

        Args:
            node_partition_book (Union[PartitionBook, dict[NodeType, PartitionBook]]): The computed Node Partition Book
        Returns:
            Union[
                Tuple[GraphPartitionData, Optional[FeaturePartitionData], Optional[PartitionBook]],
                Tuple[dict[EdgeType, GraphPartitionData], Optional[dict[EdgeType, FeaturePartitionData]], Optional[dict[EdgeType, PartitionBook]]],
            ]: Partitioned Graph Data, Feature Data, and corresponding edge partition book, is a dictionary if heterogeneous.
        """

        self._assert_and_get_rpc_setup()

        assert (
            self._edge_index is not None and self._num_edges is not None
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

        edge_partition_book: dict[EdgeType, PartitionBook] = {}
        partitioned_edge_index: dict[EdgeType, GraphPartitionData] = {}
        partitioned_edge_features: dict[EdgeType, FeaturePartitionData] = {}
        for edge_type in self._edge_types:
            (
                partitioned_edge_index_per_edge_type,
                partitioned_edge_features_per_edge_type,
                edge_partition_book_per_edge_type,
            ) = self._partition_edge_index_and_edge_features(
                node_partition_book=transformed_node_partition_book, edge_type=edge_type
            )
            partitioned_edge_index[edge_type] = partitioned_edge_index_per_edge_type
            if partitioned_edge_features_per_edge_type is not None:
                assert edge_partition_book_per_edge_type is not None
                edge_partition_book[edge_type] = edge_partition_book_per_edge_type
                partitioned_edge_features[
                    edge_type
                ] = partitioned_edge_features_per_edge_type

        elapsed_time = time.time() - start_time
        logger.info(f"Edge Partitioning finished, took {elapsed_time:.3f}s")

        formatted_num_edges = {
            edge_type: f"{num_edges:,}"
            for edge_type, num_edges in self._num_edges.items()
        }

        if self._is_input_homogeneous:
            logger.info(
                f"Partitioned {to_homogeneous(formatted_num_edges)} edges for homogeneous dataset"
            )
            return (
                to_homogeneous(partitioned_edge_index),
                to_homogeneous(partitioned_edge_features)
                if partitioned_edge_features
                else None,
                to_homogeneous(edge_partition_book) if edge_partition_book else None,
            )
        else:
            logger.info(f"Partitioned {formatted_num_edges} edges per edge type")
            return (
                partitioned_edge_index,
                partitioned_edge_features if partitioned_edge_features else None,
                edge_partition_book if edge_partition_book else None,
            )
