"""
DatasetFactory is responsible for building and returning a DistLinkPredictionDataset class or subclass. It does this by spawning a
process which initializes rpc + worker group, loads and builds a partitioned dataset, and shuts down the rpc + worker group.
"""
import time
from collections import abc
from distutils.util import strtobool
from typing import Dict, Literal, MutableMapping, Optional, Type, Union

import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed import (
    barrier,
    get_context,
    init_rpc,
    init_worker_group,
    rpc_is_initialized,
    shutdown_rpc,
)

from gigl.common import UriFactory
from gigl.common.data.dataloaders import TFRecordDataLoader
from gigl.common.data.load_torch_tensors import (
    SerializedGraphMetadata,
    TFDatasetOptions,
    load_torch_tensors_from_tf_record,
)
from gigl.common.logger import Logger
from gigl.common.utils.decorator import tf_on_cpu
from gigl.distributed.constants import DEFAULT_MASTER_DATA_BUILDING_PORT
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_link_prediction_data_partitioner import (
    DistLinkPredictionDataPartitioner,
)
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.dist_link_prediction_range_partitioner import (
    DistLinkPredictionRangePartitioner,
)
from gigl.distributed.utils import get_process_group_name
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.types.graph import GraphPartitionData
from gigl.utils.data_splitters import (
    NodeAnchorLinkSplitter,
    select_ssl_positive_label_edges,
)

logger = Logger()


@tf_on_cpu
def _load_and_build_partitioned_dataset(
    serialized_graph_metadata: SerializedGraphMetadata,
    should_load_tensors_in_parallel: bool,
    edge_dir: Literal["in", "out"],
    partitioner_class: Optional[Type[DistLinkPredictionDataPartitioner]],
    tf_dataset_options: TFDatasetOptions,
    splitter: Optional[NodeAnchorLinkSplitter] = None,
    _ssl_positive_label_percentage: Optional[float] = None,
) -> DistLinkPredictionDataset:
    """
    Given some information about serialized TFRecords, loads and builds a partitioned dataset into a DistLinkPredictionDataset class.
    We require init_rpc and init_worker_group have been called to set up the rpc and context, respectively, prior to calling this function. If this is not
    set up beforehand, this function will throw an error.
    Args:
        serialized_graph_metadata (SerializedGraphMetadata): Serialized Graph Metadata contains serialized information for loading TFRecords across node and edge types
        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        edge_dir (Literal["in", "out"]): Edge direction of the provided graph
        partitioner_class (Optional[Type[DistLinkPredictionDataPartitioner]]): Partitioner class to partition the graph inputs. If provided, this must be a
            DistLinkPredictionDataPartitioner or subclass of it. If not provided, will initialize use the DistLinkPredictionDataPartitioner class.
        tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized data is read.
        splitter (Optional[NodeAnchorLinkSplitter]): Optional splitter to use for splitting the graph data into train, val, and test sets. If not provided (None), no splitting will be performed.
        _ssl_positive_label_percentage (Optional[float]): Percentage of edges to select as self-supervised labels. Must be None if supervised edge labels are provided in advance.
            Slotted for refactor once this functionality is available in the transductive `splitter` directly
    Returns:
        DistLinkPredictionDataset: Initialized dataset with partitioned graph information

    """
    assert (
        get_context() is not None
    ), "Context must be setup prior to calling `load_and_build_partitioned_dataset` through glt.distributed.init_worker_group()"
    assert (
        rpc_is_initialized()
    ), "RPC must be setup prior to calling `load_and_build_partitioned_dataset` through glt.distributed.init_rpc()"

    rank: int = get_context().rank
    world_size: int = get_context().world_size

    tfrecord_data_loader = TFRecordDataLoader(rank=rank, world_size=world_size)
    loaded_graph_tensors = load_torch_tensors_from_tf_record(
        tf_record_dataloader=tfrecord_data_loader,
        serialized_graph_metadata=serialized_graph_metadata,
        should_load_tensors_in_parallel=should_load_tensors_in_parallel,
        rank=rank,
        tf_dataset_options=tf_dataset_options,
    )

    should_assign_edges_by_src_node: bool = False if edge_dir == "in" else True

    if partitioner_class is None:
        partitioner_class = DistLinkPredictionDataPartitioner

    if should_assign_edges_by_src_node:
        logger.info(
            f"Initializing {partitioner_class.__name__} instance while partitioning edges to its source node machine"
        )
    else:
        logger.info(
            f"Initializing {partitioner_class.__name__} instance while partitioning edges to its destination node machine"
        )
    partitioner = partitioner_class(
        should_assign_edges_by_src_node=should_assign_edges_by_src_node
    )

    partitioner.register_node_ids(node_ids=loaded_graph_tensors.node_ids)
    partitioner.register_edge_index(edge_index=loaded_graph_tensors.edge_index)
    if loaded_graph_tensors.node_features is not None:
        partitioner.register_node_features(
            node_features=loaded_graph_tensors.node_features
        )
    if loaded_graph_tensors.edge_features is not None:
        partitioner.register_edge_features(
            edge_features=loaded_graph_tensors.edge_features
        )
    if loaded_graph_tensors.positive_label is not None:
        partitioner.register_labels(
            label_edge_index=loaded_graph_tensors.positive_label, is_positive=True
        )
    if loaded_graph_tensors.negative_label is not None:
        partitioner.register_labels(
            label_edge_index=loaded_graph_tensors.negative_label, is_positive=False
        )

    # We call del so that the reference count of these registered fields is 1,
    # allowing these intermediate assets to be cleaned up as necessary inside of the partitioner.partition() call

    del (
        loaded_graph_tensors.node_ids,
        loaded_graph_tensors.node_features,
        loaded_graph_tensors.edge_index,
        loaded_graph_tensors.edge_features,
        loaded_graph_tensors.positive_label,
        loaded_graph_tensors.negative_label,
    )
    del loaded_graph_tensors

    partition_output = partitioner.partition()

    # TODO (mkolodner-sc): Move this code block to transductive splitter once that is ready
    if _ssl_positive_label_percentage is not None:
        assert (
            partition_output.partitioned_positive_labels is None
            and partition_output.partitioned_negative_labels is None
        ), "Cannot have partitioned positive and negative labels when attempting to select self-supervised positive edges from edge index."
        positive_label_edges: Union[torch.Tensor, Dict[EdgeType, torch.Tensor]]
        # TODO (mkolodner-sc): Only add necessary edge types to positive label dictionary, rather than all of the keys in the partitioned edge index
        if isinstance(partition_output.partitioned_edge_index, abc.Mapping):
            positive_label_edges = {}
            for (
                edge_type,
                graph_partition_data,
            ) in partition_output.partitioned_edge_index.items():
                edge_index = graph_partition_data.edge_index
                positive_label_edges[edge_type] = select_ssl_positive_label_edges(
                    edge_index=edge_index,
                    positive_label_percentage=_ssl_positive_label_percentage,
                )
        elif isinstance(partition_output.partitioned_edge_index, GraphPartitionData):
            positive_label_edges = select_ssl_positive_label_edges(
                edge_index=partition_output.partitioned_edge_index.edge_index,
                positive_label_percentage=_ssl_positive_label_percentage,
            )
        else:
            raise ValueError(
                "Found no partitioned edge index when attempting to select positive labels"
            )

        partition_output.partitioned_positive_labels = positive_label_edges

    logger.info(
        f"Initializing DistLinkPredictionDataset instance with edge direction {edge_dir}"
    )
    dataset = DistLinkPredictionDataset(
        rank=rank, world_size=world_size, edge_dir=edge_dir
    )

    dataset.build(
        partition_output=partition_output,
        splitter=splitter,
    )

    return dataset


def _build_dataset_process(
    process_number_on_current_machine: int,
    output_dict: MutableMapping[str, DistLinkPredictionDataset],
    serialized_graph_metadata: SerializedGraphMetadata,
    distributed_context: DistributedContext,
    dataset_building_port: int,
    sample_edge_direction: Literal["in", "out"],
    should_load_tensors_in_parallel: bool,
    partitioner_class: Optional[Type[DistLinkPredictionDataPartitioner]],
    tf_dataset_options: TFDatasetOptions,
    splitter: Optional[NodeAnchorLinkSplitter] = None,
    _ssl_positive_label_percentage: Optional[float] = None,
) -> None:
    """
    This function is spawned by a single process per machine and is responsible for:
        1. Initializing worker group and rpc connections
        2. Loading Torch tensors from serialized TFRecords
        3. Partition loaded Torch tensors across multiple machines
        4. Loading and formatting graph and feature partition data into a `DistLinkPredictionDataset` class, which will be used during inference
        5. Tearing down these connections
    Steps 2-4 are done by the `load_and_build_partitioned_dataset` function.

    We wrap this logic inside of a `mp.spawn` process so that that assets from these steps are properly cleaned up after the dataset has been built. Without
    it, we observe inference performance degradation via cached entities that remain during the inference loop. As such, using a `mp.spawn` process is an easy
    way to ensure all cached entities are cleaned up. We use `mp.spawn` instead of `mp.Process` so that any exceptions thrown in this function will be correctly
    propogated to the parent process.

    This step currently only is supported on CPU.

    Args:
        process_number_on_current_machine (int): Process number on current machine. This parameter is required and provided by mp.spawn.
            This is always set to 1 for dataset building.
        output_dict (MutableMapping[str, DistLinkPredictionDataset]): A dictionary spawned by a mp.manager which the built dataset
            will be written to for use by the parent process
        serialized_graph_metadata (SerializedGraphMetadata): Metadata about TFRecords that are serialized to disk
        distributed_context (DistributedContext): Distributed context containing information for master_ip_address, rank, and world size
        dataset_building_port (int): RPC port to use to build the dataset
        sample_edge_direction (Literal["in", "out"]): Whether edges in the graph are directed inward or outward
        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        partitioner_class (Optional[Type[DistLinkPredictionDataPartitioner]]): Partitioner class to partition the graph inputs. If provided, this must be a
            DistLinkPredictionDataPartitioner or subclass of it. If not provided, will initialize use the DistLinkPredictionDataPartitioner class.
        tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized data is read.
        splitter (Optional[NodeAnchorLinkSplitter]): Optional splitter to use for splitting the graph data into train, val, and test sets. If not provided (None), no splitting will be performed.
        _ssl_positive_label_percentage (Optional[float]): Percentage of edges to select as self-supervised labels. Must be None if supervised edge labels are provided in advance.
            Slotted for refactor once this functionality is available in the transductive `splitter` directly
    """

    # Sets up the worker group and rpc connection. We need to ensure we cleanup by calling shutdown_rpc() after we no longer need the rpc connection.
    init_worker_group(
        world_size=distributed_context.global_world_size,
        rank=distributed_context.global_rank,
        group_name=get_process_group_name(process_number_on_current_machine),
    )

    init_rpc(
        master_addr=distributed_context.main_worker_ip_address,
        master_port=dataset_building_port,
        num_rpc_threads=4,
    )

    output_dataset: DistLinkPredictionDataset = _load_and_build_partitioned_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        should_load_tensors_in_parallel=should_load_tensors_in_parallel,
        edge_dir=sample_edge_direction,
        partitioner_class=partitioner_class,
        tf_dataset_options=tf_dataset_options,
        splitter=splitter,
        _ssl_positive_label_percentage=_ssl_positive_label_percentage,
    )

    output_dict["dataset"] = output_dataset

    # We add a barrier here so that all processes end and exit this function at the same time. Without this, we may have some machines call shutdown_rpc() while other
    # machines may require rpc setup for partitioning, which will result in failure.
    barrier()
    shutdown_rpc()


def build_dataset(
    serialized_graph_metadata: SerializedGraphMetadata,
    distributed_context: DistributedContext,
    sample_edge_direction: Union[Literal["in", "out"], str],
    should_load_tensors_in_parallel: bool = True,
    partitioner_class: Optional[Type[DistLinkPredictionDataPartitioner]] = None,
    tf_dataset_options: TFDatasetOptions = TFDatasetOptions(),
    splitter: Optional[NodeAnchorLinkSplitter] = None,
    _ssl_positive_label_percentage: Optional[float] = None,
    _dataset_building_port: int = DEFAULT_MASTER_DATA_BUILDING_PORT,
) -> DistLinkPredictionDataset:
    """
    Launches a spawned process for building and returning a DistLinkPredictionDataset instance provided some SerializedGraphMetadata
    Args:
        serialized_graph_metadata (SerializedGraphMetadata): Metadata about TFRecords that are serialized to disk
        distributed_context (DistributedContext): Distributed context containing information for master_ip_address, rank, and world size
        sample_edge_direction (Union[Literal["in", "out"], str]): Whether edges in the graph are directed inward or outward. Note that this is
            listed as a possible string to satisfy type check, but in practice must be a Literal["in", "out"].
        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        partitioner_class (Optional[Type[DistLinkPredictionDataPartitioner]]): Partitioner class to partition the graph inputs. If provided, this must be a
            DistLinkPredictionDataPartitioner or subclass of it. If not provided, will initialize use the DistLinkPredictionDataPartitioner class.
        tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized data is read.
        splitter (Optional[NodeAnchorLinkSplitter]): Optional splitter to use for splitting the graph data into train, val, and test sets. If not provided (None), no splitting will be performed.
        _ssl_positive_label_percentage (Optional[float]): Percentage of edges to select as self-supervised labels. Must be None if supervised edge labels are provided in advance.
            Slotted for refactor once this functionality is available in the transductive `splitter` directly
        _dataset_building_port (int): WARNING: You don't need to configure this unless port conflict issues. Slotted for refactor.
            The RPC port to use to build the dataset. In future, the port will be automatically assigned based on availability.
            Currently defaults to: gigl.distributed.constants.DEFAULT_MASTER_DATA_BUILDING_PORT

    Returns:
        DistLinkPredictionDataset: Built GraphLearn-for-PyTorch Dataset class
    """
    assert (
        sample_edge_direction == "in" or sample_edge_direction == "out"
    ), f"Provided edge direction from inference args must be one of `in` or `out`, got {sample_edge_direction}"

    manager = mp.Manager()

    dataset_building_start_time = time.time()

    # Used for directing the outputs of the dataset building process back to the parent process
    output_dict = manager.dict()

    # Launches process for loading serialized TFRecords from disk into memory, partitioning the data across machines, and storing data inside a GLT dataset class
    mp.spawn(
        fn=_build_dataset_process,
        args=(
            output_dict,
            serialized_graph_metadata,
            distributed_context,
            _dataset_building_port,
            sample_edge_direction,
            should_load_tensors_in_parallel,
            partitioner_class,
            tf_dataset_options,
            splitter,
            _ssl_positive_label_percentage,
        ),
    )

    output_dataset: DistLinkPredictionDataset = output_dict["dataset"]

    logger.info(
        f"--- Dataset Building finished on rank {distributed_context.global_rank}, which took {time.time()-dataset_building_start_time:.2f} seconds"
    )

    return output_dataset


def build_dataset_from_task_config_uri(
    task_config_uri: str,
    distributed_context: DistributedContext,
    is_inference: bool = True,
) -> DistLinkPredictionDataset:
    """
    Builds a dataset from a provided `task_config_uri` as part of GiGL orchestration. Parameters to
    this step should be provided in the `inferenceArgs` field of the GbmlConfig for inference or the
    trainerArgs field of the GbmlConfig for training. The current parsable arguments are here are
    - sample_edge_direction: Direction of the graph
    - should_use_range_partitioning: Whether we should be using range-based partitioning
    - should_load_tensors_in_parallel: Whether TFRecord loading should happen in parallel across entities
    Args:
        task_config_uri (str): URI to a GBML Config
        distributed_context (DistributedContext): Distributed context containing information for
            master_ip_address, rank, and world size
        is_inference (bool): Whether the run is for inference or training. If True, arguments will
            be read from inferenceArgs. Otherwise, arguments witll be read from trainerArgs.
    """
    # Read from GbmlConfig for preprocessed data metadata, GNN model uri, and bigquery embedding table path
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )
    if is_inference:
        args = dict(gbml_config_pb_wrapper.inferencer_config.inferencer_args)
        args_path = "inferencerConfig.inferencerArgs"
    else:
        args = dict(gbml_config_pb_wrapper.trainer_config.trainer_args)
        args_path = "trainerConfig.trainerArgs"

    # Should be a string which is either "in" or "out"
    sample_edge_direction = args.get("sample_edge_direction", "in")

    assert sample_edge_direction in (
        "in",
        "out",
    ), f"Provided edge direction from args must be one of `in` or `out`, got {sample_edge_direction}"

    should_use_range_partitioning = bool(
        strtobool(args.get("should_use_range_partitioning", "True"))
    )

    should_load_tensors_in_parallel = bool(
        strtobool(args.get("should_load_tensors_in_parallel", "True"))
    )

    logger.info(
        f"Inferred 'sample_edge_direction' argument as : {sample_edge_direction} from argument path {args_path}. To override, please provide 'sample_edge_direction' flag."
    )
    logger.info(
        f"Inferred 'should_use_range_partitioning' argument as : {should_use_range_partitioning} from argument path {args_path}. To override, please provide 'should_use_range_partitioning' flag."
    )
    logger.info(
        f"Inferred 'should_load_tensors_in_parallel' argument as : {should_load_tensors_in_parallel} from argument path {args_path}. To override, please provide 'should_load_tensors_in_parallel' flag."
    )

    # We use a `SerializedGraphMetadata` object to store and organize information for loading serialized TFRecords from disk into memory.
    # We provide a convenience utility `convert_pb_to_serialized_graph_metadata` to build the
    # `SerializedGraphMetadata` object when using GiGL orchestration, leveraging fields of the GBMLConfigPbWrapper

    serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
        preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
        graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
    )

    if should_use_range_partitioning:
        partitioner_class = DistLinkPredictionDataPartitioner
    else:
        partitioner_class = DistLinkPredictionRangePartitioner

    dataset = build_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        distributed_context=distributed_context,
        sample_edge_direction=sample_edge_direction,
        partitioner_class=partitioner_class,
    )

    return dataset
