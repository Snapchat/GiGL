import time
import traceback
from dataclasses import dataclass
from typing import Dict, MutableMapping, Optional, Union

import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed.rpc import barrier, rpc_is_initialized
from torch.multiprocessing import Manager

from gigl.common.data.dataloaders import (
    SerializedTFRecordInfo,
    TFDatasetOptions,
    TFRecordDataLoader,
)
from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    LoadedGraphTensors,
)
from gigl.utils.share_memory import share_memory

logger = Logger()

_ID_FMT = "{entity}_ids"
_FEATURE_FMT = "{entity}_features"
_NODE_KEY = "node"
_EDGE_KEY = "edge"
_POSITIVE_LABEL_KEY = "positive_label"
_NEGATIVE_LABEL_KEY = "negative_label"


@dataclass(frozen=True)
class SerializedGraphMetadata:
    """
    Stores information for all entities. If homogeneous, all types are of type SerializedTFRecordInfo. Otherwise, they are dictionaries with the corresponding mapping.
    """

    # Node Entity Info for loading node tensors, a SerializedTFRecordInfo for homogeneous and Dict[NodeType, SerializedTFRecordInfo] for heterogeneous cases
    node_entity_info: Union[
        SerializedTFRecordInfo, Dict[NodeType, SerializedTFRecordInfo]
    ]
    # Edge Entity Info for loading edge tensors, a SerializedTFRecordInfo for homogeneous and Dict[EdgeType, SerializedTFRecordInfo] for heterogeneous cases
    edge_entity_info: Union[
        SerializedTFRecordInfo, Dict[EdgeType, SerializedTFRecordInfo]
    ]
    # Positive Label Entity Info, if present, a SerializedTFRecordInfo for homogeneous and Dict[EdgeType, SerializedTFRecordInfo] for heterogeneous cases. May be None
    # for specific edge types. If data has no positive labels across all edge types, this value is None
    positive_label_entity_info: Optional[
        Union[SerializedTFRecordInfo, Dict[EdgeType, Optional[SerializedTFRecordInfo]]]
    ] = None
    # Negative Label Entity Info, if present, a SerializedTFRecordInfo for homogeneous and Dict[EdgeType, SerializedTFRecordInfo] for heterogeneous cases. May be None
    # for specific edge types. If input has no negative labels across all edge types, this value is None.
    negative_label_entity_info: Optional[
        Union[SerializedTFRecordInfo, Dict[EdgeType, Optional[SerializedTFRecordInfo]]]
    ] = None


def _data_loading_process(
    tf_record_dataloader: TFRecordDataLoader,
    output_dict: MutableMapping[
        str, Union[torch.Tensor, Dict[Union[NodeType, EdgeType], torch.Tensor]]
    ],
    error_dict: MutableMapping[str, str],
    entity_type: str,
    serialized_tf_record_info: Union[
        SerializedTFRecordInfo,
        Dict[Union[NodeType, EdgeType], SerializedTFRecordInfo],
    ],
    rank: int,
    tf_dataset_options: TFDatasetOptions = TFDatasetOptions(),
) -> None:
    """
    Spawned multiprocessing.Process which loads homogeneous or heterogeneous information for a specific entity type [node, edge, positive_label, negative_label]
    and moves to shared memory. Also logs timing information for duration of loading. If an exception is thrown, its traceback will be stored in
    the error_dict "error" field, since exceptions for spawned processes won't properly be raised to the parent process.

    Args:
        tf_record_dataloader (TFRecordDataLoader): TFRecordDataloader used for loading tensors from serialized tfrecords
        output_dict (MutableMapping[str, Union[torch.Tensor, Dict[Union[NodeType, EdgeType], torch.Tensor]]]):
            Dictionary initialized by mp.Manager().dict() in which outputs of tensor loading will be written to
        error_dict (MutableMapping[str, str]): Dictionary initialized by mp.Manager().dict() in which error of errors in current process will be written to
        entity_type (str): Entity type to prefix ids, features, and error keys with when
            writing to the output_dict and error_dict fields
        serialized_tf_record_info (Union[SerializedTFRecordInfo, Dict[NodeType, SerializedTFRecordInfo], Dict[EdgeType, SerializedTFRecordInfo]]):
            Serialized information for current entity
        rank (int): Rank of the current machine
        tf_dataset_options (TFDatasetOptions): The options to use when building the dataset.
    """
    # We add a try - except clause here to ensure that exceptions are properly circulated back to the parent process
    try:
        # To simplify the logic to proceed on a singular code path, we convert homogeneous inputs to heterogeneous just within the scope of this function
        if isinstance(serialized_tf_record_info, SerializedTFRecordInfo):
            serialized_tf_record_info = (
                {DEFAULT_HOMOGENEOUS_NODE_TYPE: serialized_tf_record_info}
                if serialized_tf_record_info.is_node_entity
                else {DEFAULT_HOMOGENEOUS_EDGE_TYPE: serialized_tf_record_info}
            )
            is_input_homogeneous = True
        else:
            is_input_homogeneous = False

        all_tf_record_uris = [
            serialized_entity.tfrecord_uri_prefix.uri
            for serialized_entity in serialized_tf_record_info.values()
        ]

        start_time = time.time()

        logger.info(
            f"Rank {rank} has begun to load data from tfrecord directories: {all_tf_record_uris}"
        )

        ids: Dict[Union[NodeType, EdgeType], torch.Tensor] = {}
        features: Dict[Union[NodeType, EdgeType], torch.Tensor] = {}
        for (
            graph_type,
            serialized_entity_tf_record_info,
        ) in serialized_tf_record_info.items():
            (
                entity_ids,
                entity_features,
            ) = tf_record_dataloader.load_as_torch_tensors(
                serialized_tf_record_info=serialized_entity_tf_record_info,
                tf_dataset_options=tf_dataset_options,
            )
            ids[graph_type] = entity_ids
            logger.info(
                f"Rank {rank} finished loading {entity_type} ids of shape {entity_ids.shape} for graph type {graph_type} from {serialized_entity_tf_record_info.tfrecord_uri_prefix.uri}"
            )
            if entity_features is not None:
                features[graph_type] = entity_features
                logger.info(
                    f"Rank {rank} finished loading {entity_type} features of shape {entity_features.shape} for graph type {graph_type} from {serialized_entity_tf_record_info.tfrecord_uri_prefix.uri}"
                )
            else:
                logger.info(
                    f"Rank {rank} did not detect {entity_type} features for graph type {graph_type} from {serialized_entity_tf_record_info.tfrecord_uri_prefix.uri}"
                )

        logger.info(
            f"Rank {rank} is attempting to share {entity_type} id memory for tfrecord directories: {all_tf_record_uris}"
        )
        share_memory(ids)
        # We convert the ids back to homogeneous from the default heterogeneous setup if our provided input was homogeneous

        if features:
            logger.info(
                f"Rank {rank} is attempting to share {entity_type} feature memory for tfrecord directories: {all_tf_record_uris}"
            )
            share_memory(features)
            # We convert the features back to homogeneous from the default heterogeneous setup if our provided input was homogeneous

        output_dict[_ID_FMT.format(entity=entity_type)] = (
            list(ids.values())[0] if is_input_homogeneous else ids
        )
        if features:
            output_dict[_FEATURE_FMT.format(entity=entity_type)] = (
                list(features.values())[0] if is_input_homogeneous else features
            )

        logger.info(
            f"Rank {rank} has finished loading {entity_type} data from tfrecord directories: {all_tf_record_uris}, elapsed time: {time.time() - start_time:.2f} seconds"
        )

    except Exception:
        error_dict[entity_type] = traceback.format_exc()


def load_torch_tensors_from_tf_record(
    tf_record_dataloader: TFRecordDataLoader,
    serialized_graph_metadata: SerializedGraphMetadata,
    should_load_tensors_in_parallel: bool,
    rank: int = 0,
    tf_dataset_options: TFDatasetOptions = TFDatasetOptions(),
) -> LoadedGraphTensors:
    """
    Loads all torch tensors from a SerializedGraphMetadata object for all entity [node, edge, positive_label, negative_label] and edge / node types.

    Running these processes in parallel slows the runtime of each individual process, but may still result in a net speedup across all entity types. As a result,
    there is a tradeoff that needs to be made between parallel and sequential tensor loading, which is why we don't parallelize across node and edge types. We enable
    the `should_load_tensors_in_parallel` to allow some customization for loading strategies based on the input data.

    Args:
        tf_record_dataloader (TFRecordDataLoader): TFRecordDataloader used for loading tensors from serialized tfrecords
        serialized_graph_metadata (SerializedGraphMetadata): Serialized graph metadata contained serialized information for loading tfrecords across node and edge types
        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        rank (int): Rank on current machine
        tf_dataset_options (TFDatasetOptions): The options to use when building the dataset.
    Returns:
        loaded_graph_tensors (LoadedGraphTensors): Unpartitioned Graph Tensors
    """

    logger.info(f"Rank {rank} starting loading torch tensors from serialized info ...")
    start_time = time.time()

    manager = Manager()

    # By default, torch processes are created using the `fork` method, which makes a copy of the entire process. This can be problematic in multi-threaded settings,
    # especially when working with TensorFlow, since this includes all threads, which can lead to deadlocks or other synchronization issues. As a result, we set the
    # start method to spawn, which creates a new Python interpreter process and is much safer with multi-threading applications.
    ctx = mp.get_context("spawn")

    node_output_dict: MutableMapping[
        str, Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
    ] = manager.dict()

    edge_output_dict: MutableMapping[
        str, Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
    ] = manager.dict()

    error_dict: MutableMapping[str, str] = manager.dict()

    node_data_loading_process = ctx.Process(
        target=_data_loading_process,
        kwargs={
            "tf_record_dataloader": tf_record_dataloader,
            "output_dict": node_output_dict,
            "error_dict": error_dict,
            "entity_type": _NODE_KEY,
            "serialized_tf_record_info": serialized_graph_metadata.node_entity_info,
            "rank": rank,
            "tf_dataset_options": tf_dataset_options,
        },
    )

    edge_data_loading_process = ctx.Process(
        target=_data_loading_process,
        kwargs={
            "tf_record_dataloader": tf_record_dataloader,
            "output_dict": edge_output_dict,
            "error_dict": error_dict,
            "entity_type": _EDGE_KEY,
            "serialized_tf_record_info": serialized_graph_metadata.edge_entity_info,
            "rank": rank,
            "tf_dataset_options": tf_dataset_options,
        },
    )

    if serialized_graph_metadata.positive_label_entity_info is not None:
        positive_label_data_loading_process = ctx.Process(
            target=_data_loading_process,
            kwargs={
                "tf_record_dataloader": tf_record_dataloader,
                "output_dict": edge_output_dict,
                "error_dict": error_dict,
                "entity_type": _POSITIVE_LABEL_KEY,
                "serialized_tf_record_info": serialized_graph_metadata.positive_label_entity_info,
                "rank": rank,
                "tf_dataset_options": tf_dataset_options,
            },
        )
    else:
        logger.info(f"No positive labels detected from input data")

    if serialized_graph_metadata.negative_label_entity_info is not None:
        negative_label_data_loading_process = ctx.Process(
            target=_data_loading_process,
            kwargs={
                "tf_record_dataloader": tf_record_dataloader,
                "output_dict": edge_output_dict,
                "error_dict": error_dict,
                "entity_type": _NEGATIVE_LABEL_KEY,
                "serialized_tf_record_info": serialized_graph_metadata.negative_label_entity_info,
                "rank": rank,
                "tf_dataset_options": tf_dataset_options,
            },
        )
    else:
        logger.info(f"No negative labels detected from input data")

    if should_load_tensors_in_parallel:
        # In this setting, we start all the processes at once and join them at the end to achieve parallelized tensor loading
        logger.info("Loading Serialized TFRecord Data in Parallel ...")
        node_data_loading_process.start()
        edge_data_loading_process.start()
        if serialized_graph_metadata.positive_label_entity_info is not None:
            positive_label_data_loading_process.start()
        if serialized_graph_metadata.negative_label_entity_info is not None:
            negative_label_data_loading_process.start()

        node_data_loading_process.join()
        edge_data_loading_process.join()
        if serialized_graph_metadata.positive_label_entity_info is not None:
            positive_label_data_loading_process.join()
        if serialized_graph_metadata.negative_label_entity_info is not None:
            negative_label_data_loading_process.join()
    else:
        # In this setting, we start and join each process one-at-a-time in order to achieve sequential tensor loading
        logger.info("Loading Serialized TFRecord Data in Sequence ...")
        node_data_loading_process.start()
        node_data_loading_process.join()
        edge_data_loading_process.start()
        edge_data_loading_process.join()
        if serialized_graph_metadata.positive_label_entity_info is not None:
            positive_label_data_loading_process.start()
            positive_label_data_loading_process.join()
        if serialized_graph_metadata.negative_label_entity_info is not None:
            negative_label_data_loading_process.start()
            negative_label_data_loading_process.join()

    if error_dict:
        for entity_type, traceback in error_dict.items():
            logger.error(
                f"Identified error in {entity_type} data loading process: \n{traceback}"
            )
        raise ValueError(
            f"Raised error in data loading processes for entity types {error_dict.keys()}."
        )

    node_ids = node_output_dict[_ID_FMT.format(entity=_NODE_KEY)]
    node_features = node_output_dict.get(_FEATURE_FMT.format(entity=_NODE_KEY), None)

    edge_index = edge_output_dict[_ID_FMT.format(entity=_EDGE_KEY)]
    edge_features = edge_output_dict.get(_FEATURE_FMT.format(entity=_EDGE_KEY), None)

    positive_labels = edge_output_dict.get(
        _ID_FMT.format(entity=_POSITIVE_LABEL_KEY), None
    )

    negative_labels = edge_output_dict.get(
        _ID_FMT.format(entity=_NEGATIVE_LABEL_KEY), None
    )

    if rpc_is_initialized():
        logger.info(
            f"Rank {rank} has finished loading data in {time.time() - start_time:.2f} seconds. Wait for other ranks to finish loading data from tfrecords"
        )
        barrier()

    logger.info(
        f"All ranks have finished loading data from tfrecords, rank {rank} finished in {time.time() - start_time:.2f} seconds"
    )

    return LoadedGraphTensors(
        node_ids=node_ids,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        positive_label=positive_labels,
        negative_label=negative_labels,
    )
