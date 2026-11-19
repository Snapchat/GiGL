from enum import Enum
from typing import Any, Optional, Union, cast

import torch
from graphlearn_torch.distributed import init_rpc, init_worker_group, shutdown_rpc

from gigl.common.utils.garbage_collection import freeze_gc
from gigl.distributed import DistPartitioner
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
    REGISTER_EDGE_WEIGHTS_WITHOUT_EDGE_FEATURES = (
        "REGISTER_EDGE_WEIGHTS_WITHOUT_EDGE_FEATURES"
    )
    REGISTER_EDGE_QUANTIZED_FEATURES = "REGISTER_EDGE_QUANTIZED_FEATURES"
    REGISTER_EDGE_QUANTIZED_FEATURES_WITHOUT_EDGE_FEATURES = (
        "REGISTER_EDGE_QUANTIZED_FEATURES_WITHOUT_EDGE_FEATURES"
    )


@freeze_gc()
def run_distributed_partitioner_cases(
    rank: int,
    output_dicts: list[dict[int, PartitionOutput]],
    cases: list[dict[str, Any]],
    master_addr: str,
    master_port: int,
) -> None:
    """
    Runs several partitioning cases in sequence on a specific rank, initializing rpc only once.
    Spawning the ranks and importing dominates the cost of one case, so tests run many cases in one spawn.
    Args:
        rank (int): Current rank of process
        output_dicts (list[dict[int, PartitionOutput]]): One mp.Manager().dict() per case, in which the outputs of that case are written to.
        cases (list[dict[str, Any]]): One dict per case with the keyword arguments of `_partition_on_rank` other than `rank`.
        master_addr (str): Master address for initializing rpc for partitioning
        master_port (int): Master port for initializing rpc for partitioning
    """
    init_worker_group(world_size=MOCKED_NUM_PARTITIONS, rank=rank)
    init_rpc(master_addr=master_addr, master_port=master_port, num_rpc_threads=4)
    for output_dict, case in zip(output_dicts, cases):
        output_dict[rank] = _partition_on_rank(rank=rank, **case)
    shutdown_rpc()


def _partition_on_rank(
    rank: int,
    is_heterogeneous: bool,
    rank_to_input_graph: dict[int, TestGraphData],
    should_assign_edges_by_src_node: bool,
    input_data_strategy: InputDataStrategy,
    rank_to_edge_weights: Optional[
        dict[int, Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]
    ] = None,
) -> PartitionOutput:
    """Partitions the mocked input graph of `rank`. Requires rpc to already be initialized."""
    input_graph = rank_to_input_graph[rank]
    node_ids: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    node_features: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    edge_features: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    positive_labels: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    negative_labels: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    node_labels: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    node_packed_features: Union[torch.Tensor, dict[NodeType, torch.Tensor]]

    if not is_heterogeneous:
        node_ids = input_graph.node_ids[USER_NODE_TYPE]
        edge_index = input_graph.edge_index[USER_TO_USER_EDGE_TYPE]
        node_features = input_graph.node_features[USER_NODE_TYPE]
        edge_features = input_graph.edge_features[USER_TO_USER_EDGE_TYPE]
        positive_labels = input_graph.positive_labels[USER_TO_USER_EDGE_TYPE]
        negative_labels = input_graph.negative_labels[USER_TO_USER_EDGE_TYPE]
        node_labels = input_graph.node_labels[USER_NODE_TYPE]
    else:
        node_ids = input_graph.node_ids
        edge_index = input_graph.edge_index
        node_features = input_graph.node_features
        edge_features = input_graph.edge_features
        positive_labels = input_graph.positive_labels
        negative_labels = input_graph.negative_labels
        node_labels = input_graph.node_labels

    if isinstance(node_ids, dict):
        node_ids_by_type = cast(dict[NodeType, torch.Tensor], node_ids)
        node_packed_features = {
            node_type: node_type_ids.to(torch.uint8).unsqueeze(1)
            for node_type, node_type_ids in node_ids_by_type.items()
        }
    else:
        node_packed_features = node_ids.to(torch.uint8).unsqueeze(1)

    partition_output: PartitionOutput

    dist_partitioner: DistPartitioner

    if input_data_strategy in (
        InputDataStrategy.REGISTER_EDGE_QUANTIZED_FEATURES,
        InputDataStrategy.REGISTER_EDGE_QUANTIZED_FEATURES_WITHOUT_EDGE_FEATURES,
    ):
        dist_partitioner = DistPartitioner(
            should_assign_edges_by_src_node=should_assign_edges_by_src_node,
        )
        dist_partitioner.register_node_ids(node_ids=node_ids)
        dist_partitioner.register_edge_index(edge_index=edge_index)
        edge_packed_features: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        if isinstance(edge_index, dict):
            edge_index_by_type = cast(dict[EdgeType, torch.Tensor], edge_index)
            if (
                input_data_strategy
                == InputDataStrategy.REGISTER_EDGE_QUANTIZED_FEATURES
            ):
                assert isinstance(edge_features, dict)
                edge_packed_features = {
                    edge_type: torch.stack(
                        (indices[0] * 3 + 17, indices[0] * 5 + 29), dim=1
                    ).to(torch.uint8)
                    for edge_type, indices in edge_index_by_type.items()
                    if edge_type in edge_features
                }
            else:
                edge_packed_features = {
                    edge_type: indices[0].to(torch.uint8).unsqueeze(1)
                    for edge_type, indices in edge_index_by_type.items()
                }
        else:
            if (
                input_data_strategy
                == InputDataStrategy.REGISTER_EDGE_QUANTIZED_FEATURES
            ):
                edge_packed_features = torch.stack(
                    (edge_index[0] * 3 + 17, edge_index[0] * 5 + 29), dim=1
                ).to(torch.uint8)
            else:
                edge_packed_features = edge_index[0].to(torch.uint8).unsqueeze(1)
        if input_data_strategy == InputDataStrategy.REGISTER_EDGE_QUANTIZED_FEATURES:
            dist_partitioner.register_edge_features(edge_features=edge_features)
        dist_partitioner.register_edge_packed_features(
            edge_packed_features=edge_packed_features
        )
        partition_output = dist_partitioner.partition()
    elif input_data_strategy in (
        InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY,
        InputDataStrategy.REGISTER_EDGE_WEIGHTS_WITHOUT_EDGE_FEATURES,
    ):
        dist_partitioner = DistPartitioner(
            should_assign_edges_by_src_node=should_assign_edges_by_src_node,
        )
        # We call del to mimic the real use case for handling these input tensors
        dist_partitioner.register_node_ids(node_ids=node_ids)
        del node_ids
        output_node_partition_book = dist_partitioner.partition_node()

        dist_partitioner.register_edge_index(edge_index=edge_index)
        del edge_index
        if input_data_strategy == InputDataStrategy.REGISTER_ALL_ENTITIES_SEPARATELY:
            dist_partitioner.register_edge_features(edge_features=edge_features)
        del edge_features
        if rank_to_edge_weights is not None and rank in rank_to_edge_weights:
            dist_partitioner.register_edge_weights(
                edge_weights=rank_to_edge_weights[rank]
            )
        (
            output_edge_index,
            output_edge_features,
            output_edge_packed_features,
            output_edge_partition_book,
        ) = dist_partitioner.partition_edge_index_and_edge_features(
            node_partition_book=output_node_partition_book
        )

        dist_partitioner.register_node_features(node_features=node_features)
        dist_partitioner.register_node_packed_features(
            node_packed_features=node_packed_features
        )
        dist_partitioner.register_node_labels(node_labels=node_labels)
        del node_labels
        del node_features
        del node_packed_features
        (
            output_node_features,
            output_node_packed_features,
            output_node_labels,
        ) = dist_partitioner.partition_node_features_and_labels(
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
            partitioned_node_packed_features=output_node_packed_features,
            partitioned_node_labels=output_node_labels,
            partitioned_edge_features=output_edge_features,
            partitioned_edge_packed_features=output_edge_packed_features,
            partitioned_positive_labels=output_positive_labels,
            partitioned_negative_labels=output_negative_labels,
        )
    elif input_data_strategy == InputDataStrategy.REGISTER_MINIMAL_ENTITIES_SEPARATELY:
        dist_partitioner = DistPartitioner(
            should_assign_edges_by_src_node=should_assign_edges_by_src_node,
        )
        # We call del to mimic the real use case for handling these input tensors
        dist_partitioner.register_node_ids(node_ids=node_ids)
        del node_ids
        output_node_partition_book = dist_partitioner.partition_node()

        dist_partitioner.register_edge_index(edge_index=edge_index)
        del edge_index
        (
            output_edge_index,
            output_edge_features,
            output_edge_packed_features,
            output_edge_partition_book,
        ) = dist_partitioner.partition_edge_index_and_edge_features(
            node_partition_book=output_node_partition_book
        )

        partition_output = PartitionOutput(
            node_partition_book=output_node_partition_book,
            edge_partition_book=output_edge_partition_book,
            partitioned_edge_index=output_edge_index,
            partitioned_node_features=None,
            partitioned_node_labels=None,
            partitioned_edge_features=output_edge_features,
            partitioned_edge_packed_features=output_edge_packed_features,
            partitioned_positive_labels=None,
            partitioned_negative_labels=None,
        )

    else:
        dist_partitioner = DistPartitioner(
            should_assign_edges_by_src_node=should_assign_edges_by_src_node,
            node_ids=node_ids,
            node_features=node_features,
            node_packed_features=node_packed_features,
            edge_index=edge_index,
            edge_features=edge_features,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            node_labels=node_labels,
        )
        # We call del to mimic the real use case for handling these input tensors
        del (
            node_ids,
            node_features,
            node_packed_features,
            edge_index,
            edge_features,
            positive_labels,
            negative_labels,
            node_labels,
        )
        partition_output = dist_partitioner.partition()

    return partition_output
