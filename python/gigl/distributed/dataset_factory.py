"""
DatasetFactory is responsible for building and returning a DistLinkPredictionDataset class or subclass. It does this by spawning a
process which initializes rpc + worker group, loads and builds a partitioned dataset, and shuts down the rpc + worker group.
"""
import time
from collections import abc
from typing import Dict, Literal, MutableMapping, Optional, Union

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
from gigl.distributed.utils import get_process_group_name
from gigl.src.common.types.graph_data import EdgeType
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
    partitioner: Optional[DistLinkPredictionDataPartitioner],
    dataset: Optional[DistLinkPredictionDataset],
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
        partitioner (Optional[DistLinkPredictionDataPartitioner]): Initialized partitioner to partition the graph inputs. If provided, this must be a
            DistLinkPredictionDataPartitioner or subclass of it. If not provided, will initialize a DistLinkPredictionDataPartitioner instance
            using provided edge assign strategy.
        dataset (Optional[DistLinkPredictionDataset]): Initialized dataset class to store the graph inputs. If provided, this must be a
            DistLinkPredictionDataset or subclass of it. If not provided, will initialize a DistLinkPredictionDataset instance using provided edge_dir.
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

    if partitioner is None:
        if should_assign_edges_by_src_node:
            logger.info(
                f"Initializing DistLinkPredictionDataPartitioner instance while partitioning edges to its source node machine"
            )
        else:
            logger.info(
                f"Initializing DistLinkPredictionDataPartitioner instance while partitioning edges to its destination node machine"
            )
        partitioner = DistLinkPredictionDataPartitioner(
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

    if dataset is None:
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
    partitioner: Optional[DistLinkPredictionDataPartitioner],
    dataset: Optional[DistLinkPredictionDataset],
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
        partitioner (Optional[DistLinkPredictionDataPartitioner]): Initialized partitioner to partition the graph inputs. If provided, this must be a
            DistLinkPredictionDataPartitioner or subclass of it. If not provided, will initialize a DistLinkPredictionDataPartitioner instance
            using provided edge assign strategy.
        dataset (Optional[DistLinkPredictionDataset]): Initialized dataset class to store the graph inputs. If provided, this must be a
            DistLinkPredictionDataset or subclass of it. If not provided, will initialize a DistLinkPredictionDataset instance using provided edge_dir.
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
        partitioner=partitioner,
        dataset=dataset,
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
    partitioner: Optional[DistLinkPredictionDataPartitioner] = None,
    dataset: Optional[DistLinkPredictionDataset] = None,
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
        partitioner (Optional[DistLinkPredictionDataPartitioner]): Initialized partitioner to partition the graph inputs. If provided, this must be a
            DistLinkPredictionDataPartitioner or subclass of it. If not provided, will initialize a DistLinkPredictionDataPartitioner instance
            using provided edge assign strategy.
        dataset (Optional[DistLinkPredictionDataset]): Initialized dataset class to store the graph inputs. If provided, this must be a
            DistLinkPredictionDataset or subclass of it. If not provided, will initialize a DistLinkPredictionDataset instance using provided edge_dir.
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
            partitioner,
            dataset,
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
