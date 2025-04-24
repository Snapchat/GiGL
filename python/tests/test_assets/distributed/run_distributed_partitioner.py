from enum import Enum
from typing import Dict, Type, Union

import torch
from graphlearn_torch.distributed import init_rpc, init_worker_group

from gigl.distributed import DistLinkPredictionDataPartitioner
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import PartitionOutput
from tests.test_assets.distributed.constants import (
    MOCKED_NUM_PARTITIONS,
    USER_NODE_TYPE,
    USER_TO_USER_EDGE_TYPE,
    TestGraphData,
)


class InputDataStrategy(Enum):
    REGISTER_ALL_ENTITIES_SEPARATELY = "REGISTER_ALL_ENTITIES_SEPARATELY"
    REGISTER_ALL_ENTITIES_TOGETHER = "REGISTER_ALL_ENTITIES_TOGETHER"
    REGISTER_MINIMAL_ENTITIES_SEPARATELY = "REGISTER_MINIMAL_ENTITIES_SEPARATELY"


def run_distributed_partitioner(
    rank: int,
    output_dict: Dict[int, PartitionOutput],
    is_heterogeneous: bool,
    rank_to_input_graph: Dict[int, TestGraphData],
    should_assign_edges_by_src_node: bool,
    master_addr: str,
    master_port: int,
    input_data_strategy: InputDataStrategy,
    partitioner_class: Type[DistLinkPredictionDataPartitioner],
) -> None:
    """
    Runs the distributed partitioner on a specific rank.
    Args:
        rank (int): Current rank of process
        output_dict: Dict[int, PartitionOutput]: Dict initialized by mp.Manager().dict() in which outputs of partitioner will be written to. This is a mapping of rank to Partition output.
        is_heterogeneous (bool): Whether homogeneous or heterogeneous inputs should be used
        rank_to_input_graph (Dict[int, TestGraphData]): Mapping of rank to mocked input graph for testing partitioning
        should_assign_edges_by_src_node (bool): Whether to partion edges according to the partition book of the source node or destination node
        master_addr (str): Master address for initializing rpc for partitioning
        master_port (int): Master port for initializing rpc for partitioning
        input_data_strategy (InputDataStrategy): Strategy for registering inputs to the partitioner
        partitioner_class (Type[DistLinkPredictionDataPartitioner]): The class to use for partitioning
    """

    input_graph = rank_to_input_graph[rank]
    node_ids: Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
    edge_index: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    node_features: Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
    edge_features: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    positive_labels: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    negative_labels: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]

    if not is_heterogeneous:
        node_ids = input_graph.node_ids[USER_NODE_TYPE]
        edge_index = input_graph.edge_index[USER_TO_USER_EDGE_TYPE]
        node_features = input_graph.node_features[USER_NODE_TYPE]
        edge_features = input_graph.edge_features[USER_TO_USER_EDGE_TYPE]
        positive_labels = input_graph.positive_labels[USER_TO_USER_EDGE_TYPE]
        negative_labels = input_graph.negative_labels[USER_TO_USER_EDGE_TYPE]
    else:
        node_ids = input_graph.node_ids
        edge_index = input_graph.edge_index
        node_features = input_graph.node_features
        edge_features = input_graph.edge_features
        positive_labels = input_graph.positive_labels
        negative_labels = input_graph.negative_labels
    partition_output: PartitionOutput

    init_worker_group(world_size=MOCKED_NUM_PARTITIONS, rank=rank)
    init_rpc(master_addr=master_addr, master_port=master_port, num_rpc_threads=4)
    dist_partitioner: DistLinkPredictionDataPartitioner

    if input_data_strategy == InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY:
        dist_partitioner = partitioner_class(
            should_assign_edges_by_src_node=should_assign_edges_by_src_node,
        )
        # We call del to mimic the real use case for handling these input tensors
        dist_partitioner.register_node_ids(node_ids=node_ids)
        del node_ids
        output_node_partition_book = dist_partitioner.partition_node()

        dist_partitioner.register_edge_index(edge_index=edge_index)
        dist_partitioner.register_edge_features(edge_features=edge_features)
        del edge_index
        del edge_features
        (
            output_edge_index,
            output_edge_features,
            output_edge_partition_book,
        ) = dist_partitioner.partition_edge_index_and_edge_features(
            node_partition_book=output_node_partition_book
        )

        dist_partitioner.register_node_features(node_features=node_features)
        del node_features
        output_node_features = dist_partitioner.partition_node_features(
            node_partition_book=output_node_partition_book
        )

        dist_partitioner.register_labels(
            label_edge_index=positive_labels, is_positive=True
        )
        del positive_labels
        output_positive_labels = dist_partitioner.partition_labels(
            node_partition_book=output_node_partition_book, is_positive=True
        )

        dist_partitioner.register_labels(
            label_edge_index=negative_labels, is_positive=False
        )
        del negative_labels
        output_negative_labels = dist_partitioner.partition_labels(
            node_partition_book=output_node_partition_book, is_positive=False
        )

        partition_output = PartitionOutput(
            node_partition_book=output_node_partition_book,
            edge_partition_book=output_edge_partition_book,
            partitioned_edge_index=output_edge_index,
            partitioned_node_features=output_node_features,
            partitioned_edge_features=output_edge_features,
            partitioned_positive_labels=output_positive_labels,
            partitioned_negative_labels=output_negative_labels,
        )
    elif input_data_strategy == InputDataStrategy.REGISTER_MINIMAL_ENTITIES_SEPARATELY:
        dist_partitioner = partitioner_class(
            should_assign_edges_by_src_node=should_assign_edges_by_src_node,
        )
        # We call del to mimic the real use case for handling these input tensors
        dist_partitioner.register_node_ids(node_ids=node_ids)
        del node_ids
        output_node_partition_book = dist_partitioner.partition_node()

        dist_partitioner.register_edge_index(edge_index=edge_index)
        del edge_index
        (
            output_graph,
            output_edge_features,
            output_edge_partition_book,
        ) = dist_partitioner.partition_edge_index_and_edge_features(
            node_partition_book=output_node_partition_book
        )

        partition_output = PartitionOutput(
            node_partition_book=output_node_partition_book,
            edge_partition_book=output_edge_partition_book,
            partitioned_edge_index=output_graph,
            partitioned_node_features=None,
            partitioned_edge_features=None,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
        )

    else:
        dist_partitioner = partitioner_class(
            should_assign_edges_by_src_node=should_assign_edges_by_src_node,
            node_ids=node_ids,
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
        )
        # We call del to mimic the real use case for handling these input tensors
        del (
            node_ids,
            node_features,
            edge_index,
            edge_features,
            positive_labels,
            negative_labels,
        )
        partition_output = dist_partitioner.partition()

    output_dict[rank] = partition_output
