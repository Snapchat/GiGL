import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.utils.data
from torch.utils.data._utils.worker import WorkerInfo
from torch_geometric.data import Data
from torch_geometric.data.hetero_data import HeteroData

from gigl.common.logger import Logger
from gigl.common.utils.torch_training import get_rank, get_world_size
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)

logger = Logger()


def get_data_split_for_current_worker(data_list: np.ndarray) -> np.ndarray:
    """Split list of data per worker
    Selects a subset of data based on Torch get_worker_info.
    Used as a shard selection function in Dataset.
    """
    # Worker info only available if in a worker i.e. not in main process
    worker_info: Optional[WorkerInfo] = torch.utils.data.get_worker_info()
    if worker_info is None:
        return data_list  # Just using main process for training, use all urls
    else:
        worker_id: int = worker_info.id
        num_workers: int = worker_info.num_workers
        # Accounting for distributed training
        global_worker_id = get_rank() * num_workers + worker_id
        global_num_workers = num_workers * get_world_size()
        global_num_worker_to_data_list_ratio = global_num_workers / len(data_list)
        if global_num_worker_to_data_list_ratio > 1:
            logger.warning(
                f"Number of workers ({global_num_workers}) is greater than number of elements ({len(data_list)}). "
                f"Data will be replicated, which may lead to increased memory usage. "
                f"Consider reducing the number of workers or increasing the dataset size for better efficiency."
            )
            data_list = np.tile(
                data_list, math.ceil(global_num_worker_to_data_list_ratio)
            )

        # Starting at the url at index `worker_id`, return every url that is `num_workers` index away from prior
        # i.e. if worker_id = 2, and num workers = 3, then urls returned have following indeces:
        # [3, 6, 9, ...] ; worker_id starts at 0 thus worker_id = 2, is really the 3rd worker
        worker_data = data_list[global_worker_id::global_num_workers]
        logger.debug(
            f"Worker {global_worker_id} has {len(worker_data)} elements: {worker_data}."
        )
        return worker_data


def cast_graph_for_training(
    batch_graph_data: PygGraphData,
    graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
    preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
    batch_type: str,
    should_register_edge_features: Optional[bool],
) -> Union[Data, HeteroData]:
    """
    Casts the PygGraphData object into a Data or HeteroData object. Also fills in any missing fields from graph
    builder with empty tensors in cases where there are no edges for a graph or given edge type.
    Args:
        batch_graph_data (PygGraphData): Coalesced batch graph
        graph_metadata_pb_wrapper (GraphMetadataPbWrapper): Graph Metadata Pb Wrapper for this training job
        preprocessed_metadata_pb_wrapper (PreprocessedMetadataPbWrapper): Preprocessed Metadata Pb Wrapper for this training job
        should_register_edge_features (bool): Whether we should register edge features for the built graph
    """
    if graph_metadata_pb_wrapper.is_heterogeneous:
        casted_graph = batch_graph_data
        missing_node_types_list: List[NodeType] = []
        # If we have a node type missing in the graph, insert that node type into the x_dict field
        for (
            condensed_node_type,
            node_feature_dim,
        ) in (
            preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_dim_map.items()
        ):
            node_type = graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                condensed_node_type
            ]
            if node_type not in casted_graph.node_types:
                missing_node_types_list.append(node_type)
                casted_graph[node_type].x = torch.empty(
                    (0, node_feature_dim), dtype=torch.float32
                )
        has_any_missing_node_types: bool = len(missing_node_types_list) > 0
        if has_any_missing_node_types:
            logger.info(
                f"Found the follow node types missing from heterogeneous {batch_type} batched graph: {missing_node_types_list}. If you are seeing multiple of this log across batches and this isn't expected, please revisit the graph definition and sampling strategy."
            )

        missing_edge_types_list: List[EdgeType] = []
        for (
            condensed_edge_type,
            edge_feature_dim,
        ) in (
            preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_dim_map.items()
        ):
            edge_type = graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                condensed_edge_type
            ]
            is_edge_type_missing_from_casted_graph = (
                edge_type not in casted_graph.edge_types
                or casted_graph[edge_type].edge_index is None
            )
            if is_edge_type_missing_from_casted_graph:
                missing_edge_types_list.append(edge_type)
                casted_graph[edge_type].edge_index = torch.empty(
                    (2, 0), dtype=torch.int64
                )
                if should_register_edge_features:
                    casted_graph[edge_type].edge_attr = torch.empty(
                        (0, edge_feature_dim), dtype=torch.float32
                    )
        has_any_missing_edge_types: bool = len(missing_edge_types_list) > 0
        if has_any_missing_edge_types:
            logger.info(
                f"Found the follow edge types missing from heterogeneous {batch_type} batched graph: {missing_edge_types_list}. If you are seeing multiple of this log across batches and this isn't expected, please revisit the graph definition and sampling strategy."
            )

    else:
        casted_graph = batch_graph_data.to_homogeneous()
        if casted_graph.num_nodes == 0:
            logger.warning(
                f"Found no nodes in homogeneous {batch_type} batched graph. "
            )
        condensed_edge_type = graph_metadata_pb_wrapper.condensed_edge_types[0]
        is_edge_missing = casted_graph.edge_index is None
        if is_edge_missing:
            logger.warning(f"Found no edges in homogeneous {batch_type} batched graph.")
            casted_graph.edge_index = torch.empty((2, 0), dtype=torch.int64)
            if should_register_edge_features:
                edge_feature_dim = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_dim_map[
                    condensed_edge_type
                ]
                casted_graph.edge_attr = torch.empty(
                    (0, edge_feature_dim), dtype=torch.float32
                )
    return casted_graph
