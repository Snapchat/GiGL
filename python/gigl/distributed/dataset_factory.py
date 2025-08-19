"""
DatasetFactory is responsible for building and returning a DistLinkPredictionDataset class or subclass. It does this by spawning a
process which initializes rpc + worker group, loads and builds a partitioned dataset, and shuts down the rpc + worker group.
"""
import gc
import time
from collections import abc
from distutils.util import strtobool
from typing import Literal, MutableMapping, Optional, Tuple, Type, Union

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
from gigl.types.graph import LoadedGraphTensors
from gigl.common import Uri, UriFactory
from gigl.common.data.dataloaders import SerializedTFRecordInfo, TFRecordDataLoader
from gigl.common.data.load_torch_tensors import (
    SerializedGraphMetadata,
    TFDatasetOptions,
    load_torch_tensors_from_tf_record,
)
from gigl.common.logger import Logger
from gigl.common.utils.decorator import tf_on_cpu
from gigl.distributed.constants import DEFAULT_MASTER_DATA_BUILDING_PORT
from gigl.distributed.dist_context import DistributedContext
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.dist_partitioner import DistPartitioner
from gigl.distributed.dist_range_partitioner import DistRangePartitioner
from gigl.distributed.utils import (
    get_free_ports_from_master_node,
    get_internal_ip_from_master_node,
    get_process_group_name,
)
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.types.graph import DEFAULT_HOMOGENEOUS_EDGE_TYPE, FeaturePartitionData, PartitionOutput
from gigl.utils.data_splitters import (
    HashedNodeAnchorLinkSplitter,
    NodeAnchorLinkSplitter,
    select_ssl_positive_label_edges,
)

logger = Logger()


def _get_labels_from_features(
    feature_and_label_tensor: torch.Tensor, label_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a combined tensor of features and labels, returns the features and labels separately.
    Args:
        feature_and_label_tensor (torch.Tensor): Tensor of features and labels
        label_dim (int): Dimension of the labels
    Returns:
        feature_tensor (torch.Tensor): Tensor of features
        label_tensor (torch.Tensor): Tensor of labels
    """

    if len(feature_and_label_tensor.shape) != 2:
        raise ValueError(
            f"Expected tensor to be 2D for extracting labels, but got shape {feature_and_label_tensor.shape}"
        )

    _, feature_and_label_dim = feature_and_label_tensor.shape

    if label_dim > feature_and_label_dim:
        raise ValueError(
            f"Got invalid label dim {label_dim} for extracting labels from tensor of shape {feature_and_label_dim}"
        )

    feature_dim = feature_and_label_dim - label_dim

    return (
        feature_and_label_tensor[:, :feature_dim],
        feature_and_label_tensor[:, feature_dim:],
    )


def _partition_graph_data(
    loaded_graph_tensors: LoadedGraphTensors,
    edge_dir: Literal["in", "out"],
    partitioner_class: Optional[Type[DistPartitioner]],
) -> PartitionOutput:
    """
    Partitions graph data using the specified partitioner.

    Args:
        loaded_graph_tensors: The loaded graph tensors containing node_ids, edge_index, features, and labels
        edge_dir: Edge direction of the provided graph
        partitioner_class: Partitioner class to partition the graph inputs

    Returns:
        partition_output: The result of partitioning the graph data
    """
    should_assign_edges_by_src_node: bool = False if edge_dir == "in" else True

    if partitioner_class is None:
        partitioner_class = DistPartitioner

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
    return partition_output


def _extract_node_labels(
    partition_output: PartitionOutput,
    serialized_graph_metadata: SerializedGraphMetadata,
) -> Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]:
    """
    Extracts node labels from partitioned node features.

    Args:
        partition_output: The partitioned graph data
        serialized_graph_metadata: Metadata containing label information

    Returns:
        node_labels: Extracted node labels or None if no labels are present
    """
    node_labels: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]] = None
    if isinstance(partition_output.partitioned_node_features, abc.Mapping):
        node_labels = {}
        for (
            node_type,
            node_feature,
        ) in partition_output.partitioned_node_features.items():
            if isinstance(serialized_graph_metadata.node_entity_info, abc.Mapping):
                label_dim = len(
                    serialized_graph_metadata.node_entity_info[node_type].label_keys
                )
            else:
                label_dim = len(serialized_graph_metadata.node_entity_info.label_keys)
            if label_dim > 0:
                node_features, node_labels[node_type] = _get_labels_from_features(
                    node_feature.feats, label_dim=label_dim
                )
                partition_output.partitioned_node_features[
                    node_type
                ] = FeaturePartitionData(feats=node_features, ids=node_feature.ids)
                del node_feature
                gc.collect()

    elif isinstance(partition_output.partitioned_node_features, FeaturePartitionData):
        if not isinstance(
            serialized_graph_metadata.node_entity_info, SerializedTFRecordInfo
        ):
            raise ValueError(
                f"Expected partitioned node features to be type SerializedTFRecordInfo, got {type(partition_output.partitioned_node_features)}"
            )
        label_dim = len(serialized_graph_metadata.node_entity_info.label_keys)
        if label_dim > 0:
            node_features, node_labels = _get_labels_from_features(
                partition_output.partitioned_node_features.feats, label_dim=label_dim
            )
            partition_output.partitioned_node_features = FeaturePartitionData(
                feats=node_features, ids=partition_output.partitioned_node_features.ids
            )
            gc.collect()
    else:
        raise ValueError(
            f"Expected to have partitioned node features if labels are present, but got node features {partition_output.partitioned_node_features}"
        )

    return node_labels


def _process_ssl_positive_labels(
    loaded_graph_tensors: LoadedGraphTensors,
    ssl_positive_label_percentage: float,
    splitter: Optional[NodeAnchorLinkSplitter],
) -> None:
    """
    Processes SSL positive label selection from edge indices.

    Args:
        loaded_graph_tensors: The loaded graph tensors to modify
        ssl_positive_label_percentage: Percentage of edges to select as self-supervised labels
        splitter: Optional splitter used for heterogeneous graphs

    Raises:
        ValueError: If positive/negative labels already exist or unknown edge index type
    """
    if (
        loaded_graph_tensors.positive_label is not None
        or loaded_graph_tensors.negative_label is not None
    ):
        raise ValueError(
            "Cannot have loaded positive and negative labels when attempting to select self-supervised positive edges from edge index."
        )

    positive_label_edges: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
    if isinstance(loaded_graph_tensors.edge_index, abc.Mapping):
        # This assert is required while `select_ssl_positive_label_edges` exists out of any splitter. Once this is in transductive splitter,
        # we can remove this assert.
        assert isinstance(
            splitter, HashedNodeAnchorLinkSplitter
        ), f"GiGL only supports {HashedNodeAnchorLinkSplitter.__name__} currently, got {type(splitter)}"
        positive_label_edges = {}
        for supervision_edge_type in splitter._supervision_edge_types:
            positive_label_edges[
                supervision_edge_type
            ] = select_ssl_positive_label_edges(
                edge_index=loaded_graph_tensors.edge_index[supervision_edge_type],
                positive_label_percentage=ssl_positive_label_percentage,
            )
    elif isinstance(loaded_graph_tensors.edge_index, torch.Tensor):
        positive_label_edges = select_ssl_positive_label_edges(
            edge_index=loaded_graph_tensors.edge_index,
            positive_label_percentage=ssl_positive_label_percentage,
        )
    else:
        raise ValueError(
            f"Found an unknown edge index type: {type(loaded_graph_tensors.edge_index)} when attempting to select positive labels"
        )

    loaded_graph_tensors.positive_label = positive_label_edges


@tf_on_cpu
def _load_and_build_partitioned_dataset(
    serialized_graph_metadata: SerializedGraphMetadata,
    should_load_tensors_in_parallel: bool,
    edge_dir: Literal["in", "out"],
    partitioner_class: Optional[Type[DistPartitioner]],
    node_tf_dataset_options: TFDatasetOptions,
    edge_tf_dataset_options: TFDatasetOptions,
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
        partitioner_class (Optional[Type[DistPartitioner]]): Partitioner class to partition the graph inputs. If provided, this must be a
            DistPartitioner or subclass of it. If not provided, will initialize use the DistPartitioner class.
        node_tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized node data is read.
        edge_tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized edge data is read.
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
        node_tf_dataset_options=node_tf_dataset_options,
        edge_tf_dataset_options=edge_tf_dataset_options,
    )

    # TODO (mkolodner-sc): Move SSL code block to transductive splitter once that is ready
    if _ssl_positive_label_percentage is not None:
        _process_ssl_positive_labels(
            loaded_graph_tensors=loaded_graph_tensors,
            ssl_positive_label_percentage=_ssl_positive_label_percentage,
            splitter=splitter,
        )

    if splitter is not None and splitter.should_convert_labels_to_edges:
        loaded_graph_tensors.treat_labels_as_edges(edge_dir=edge_dir)

    partition_output = _partition_graph_data(
        loaded_graph_tensors=loaded_graph_tensors,
        edge_dir=edge_dir,
        partitioner_class=partitioner_class,
    )

    node_labels: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]] = _extract_node_labels(
        partition_output=partition_output,
        serialized_graph_metadata=serialized_graph_metadata,
    )

    # TODO (mkolodner-sc): Add node labels to the dataset

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
    master_ip_address: str,
    master_dataset_building_ports: Tuple[int, int],
    node_rank: int,
    node_world_size: int,
    sample_edge_direction: Literal["in", "out"],
    should_load_tensors_in_parallel: bool,
    partitioner_class: Optional[Type[DistPartitioner]],
    node_tf_dataset_options: TFDatasetOptions,
    edge_tf_dataset_options: TFDatasetOptions,
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
        master_ip_address (str): IP address of the master node
        master_dataset_building_ports (Tuple[int, int]): Free ports on the master node to use to build the dataset, the first port is used for partitioning and the second is used for splitting
        node_rank (int): Rank of the node (machine) on which this process is running
        node_world_size (int): World size (total #) of the nodes participating in hosting the dataset
        sample_edge_direction (Literal["in", "out"]): Whether edges in the graph are directed inward or outward
        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        partitioner_class (Optional[Type[DistPartitioner]]): Partitioner class to partition the graph inputs. If provided, this must be a
            DistPartitioner or subclass of it. If not provided, will initialize use the DistPartitioner class.
        node_tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized node data is read.
        edge_tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized edge data is read.
        splitter (Optional[NodeAnchorLinkSplitter]): Optional splitter to use for splitting the graph data into train, val, and test sets. If not provided (None), no splitting will be performed.
        _ssl_positive_label_percentage (Optional[float]): Percentage of edges to select as self-supervised labels. Must be None if supervised edge labels are provided in advance.
            Slotted for refactor once this functionality is available in the transductive `splitter` directly
    """

    # Sets up the worker group and rpc connection. We need to ensure we cleanup by calling shutdown_rpc() after we no longer need the rpc connection.
    init_worker_group(
        world_size=node_world_size,
        rank=node_rank,
        group_name=get_process_group_name(process_number_on_current_machine),
    )

    assert (
        len(master_dataset_building_ports) == 2
    ), f"Expected master_dataset_building_ports to be a tuple of two ports, got {master_dataset_building_ports}"
    rpc_port = master_dataset_building_ports[0]
    splitter_port = master_dataset_building_ports[1]

    init_rpc(
        master_addr=master_ip_address,
        master_port=rpc_port,
        num_rpc_threads=16,
    )
    # HashedNodeAnchorLinkSplitter requires rpc to be initialized, so we initialize it here.
    should_teardown_process_group = False
    if isinstance(splitter, HashedNodeAnchorLinkSplitter):
        should_teardown_process_group = True
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_ip_address}:{splitter_port}",
            world_size=node_world_size,
            rank=node_rank,
        )

    output_dataset: DistLinkPredictionDataset = _load_and_build_partitioned_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        should_load_tensors_in_parallel=should_load_tensors_in_parallel,
        edge_dir=sample_edge_direction,
        partitioner_class=partitioner_class,
        node_tf_dataset_options=node_tf_dataset_options,
        edge_tf_dataset_options=edge_tf_dataset_options,
        splitter=splitter,
        _ssl_positive_label_percentage=_ssl_positive_label_percentage,
    )

    output_dict["dataset"] = output_dataset

    # We add a barrier here so that all processes end and exit this function at the same time. Without this, we may have some machines call shutdown_rpc() while other
    # machines may require rpc setup for partitioning, which will result in failure.
    barrier()
    shutdown_rpc()
    if should_teardown_process_group and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def build_dataset(
    serialized_graph_metadata: SerializedGraphMetadata,
    sample_edge_direction: Union[Literal["in", "out"], str],
    distributed_context: Optional[DistributedContext] = None,
    should_load_tensors_in_parallel: bool = True,
    partitioner_class: Optional[Type[DistPartitioner]] = None,
    node_tf_dataset_options: TFDatasetOptions = TFDatasetOptions(),
    edge_tf_dataset_options: TFDatasetOptions = TFDatasetOptions(),
    splitter: Optional[NodeAnchorLinkSplitter] = None,
    _ssl_positive_label_percentage: Optional[float] = None,
    _dataset_building_port: Optional[
        int
    ] = None,  # WARNING: This field will be deprecated in the future
) -> DistLinkPredictionDataset:
    """
    Launches a spawned process for building and returning a DistLinkPredictionDataset instance provided some
    SerializedGraphMetadata.

    It is expected that there is only one `build_dataset` call per node (machine).
    This requirement exists to ensure each machine only participates once in housing a parition of a dataset; otherwise
    a machine may end up housing multiple partitions of the same dataset which may cause memory issues.

    This function expects that there is a process group initialized between the process' for the nodes participating in
    hosting the dataset partition. This is so necessary information can be communicated between the nodes
    i.e. free port information, master IP address, et al. to enable configure RPC. If there is no process group initialized,
    the function will initialize one using `env://` config. See :py:obj:`torch.distributed.init_process_group` for more info.

    Args:
        serialized_graph_metadata (SerializedGraphMetadata): Metadata about TFRecords that are serialized to disk
        distributed_context (deprecated field - will be removed soon) (Optional[DistributedContext]): Distributed context containing information for master_ip_address, rank, and world size.
            Defaults to None, in which case it will be initialized from the current torch.distributed context. If provided,
            you need not initialized a process_group, one will be initialized.
        sample_edge_direction (Union[Literal["in", "out"], str]): Whether edges in the graph are directed inward or outward. Note that this is
            listed as a possible string to satisfy type check, but in practice must be a Literal["in", "out"].
        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        partitioner_class (Optional[Type[DistPartitioner]]): Partitioner class to partition the graph inputs. If provided, this must be a
            DistPartitioner or subclass of it. If not provided, will initialize use the DistPartitioner class.
        node_tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized node data is read.
        edge_tf_dataset_options (TFDatasetOptions): Options provided to a tf.data.Dataset to tune how serialized edge data is read.
        splitter (Optional[NodeAnchorLinkSplitter]): Optional splitter to use for splitting the graph data into train, val, and test sets.
            If not provided (None), no splitting will be performed.
        _ssl_positive_label_percentage (Optional[float]): Percentage of edges to select as self-supervised labels. Must be None if supervised edge labels are provided in advance.
            Slotted for refactor once this functionality is available in the transductive `splitter` directly
        _dataset_building_port (deprecated field - will be removed soon) (Optional[int]): Contains information about master port. Defaults to None, in which case it will
            be initialized from the current torch.distributed context.

    Returns:
        DistLinkPredictionDataset: Built GraphLearn-for-PyTorch Dataset class
    """
    if distributed_context is not None:
        logger.warning(
            "The `distributed_context` argument is deprecated and will be removed in a future release. "
            "Please setup the `torch.distributed.init_process_group` in the caller context and prevent "
            "passing `distributed_context` argument."
        )

    assert (
        sample_edge_direction == "in" or sample_edge_direction == "out"
    ), f"Provided edge direction from inference args must be one of `in` or `out`, got {sample_edge_direction}"

    if splitter is not None:
        logger.info(f"Received splitter {type(splitter)}.")

    manager = mp.Manager()

    dataset_building_start_time = time.time()

    # Used for directing the outputs of the dataset building process back to the parent process
    output_dict = manager.dict()

    node_world_size: int
    node_rank: int
    master_ip_address: str
    master_dataset_building_ports: Tuple[int, int]
    if distributed_context is None:
        should_cleanup_distributed_context: bool = False
        if _dataset_building_port is not None:
            logger.warning(
                f"Found specified dataset building port {_dataset_building_port} but no distributed context is provided, will instead infer free ports automatically for dataset building"
            )

        if not torch.distributed.is_initialized():
            logger.info(
                "Distributed context is None, and no process group detected; will try to "
                + "`init_process_group` to communicate necessary setup information."
            )
            should_cleanup_distributed_context = True
            torch.distributed.init_process_group(backend="gloo")

        node_world_size = torch.distributed.get_world_size()
        node_rank = torch.distributed.get_rank()
        master_ip_address = get_internal_ip_from_master_node()
        master_dataset_building_ports = tuple(get_free_ports_from_master_node(num_ports=2))  # type: ignore[assignment]

        if should_cleanup_distributed_context and torch.distributed.is_initialized():
            logger.info(
                "Cleaning up process group as it was initialized inside build_dataset."
            )
            torch.distributed.destroy_process_group()
    else:
        node_world_size = distributed_context.global_world_size
        node_rank = distributed_context.global_rank
        master_ip_address = distributed_context.main_worker_ip_address
        if _dataset_building_port is None:
            master_dataset_building_ports = (
                DEFAULT_MASTER_DATA_BUILDING_PORT,  # Used for partitioning rpc
                DEFAULT_MASTER_DATA_BUILDING_PORT
                + 1,  # Used for distributed communication with the splitter, will not be used if no splitter is provided,
            )
        else:
            master_dataset_building_ports = (
                _dataset_building_port,  # Used for partitioning rpc
                _dataset_building_port
                + 1,  # Used for distributed communication with the splitter, will not be used if no splitter is provided
            )

    logger.info(
        f"Dataset Building started on {node_rank} of {node_world_size} nodes, using following node as main: "
        + f"{master_ip_address}:{master_dataset_building_ports}"
    )

    # Launches process for loading serialized TFRecords from disk into memory, partitioning the data across machines, and storing data inside a GLT dataset class
    mp.spawn(
        fn=_build_dataset_process,
        args=(
            output_dict,
            serialized_graph_metadata,
            master_ip_address,
            master_dataset_building_ports,
            node_rank,
            node_world_size,
            sample_edge_direction,
            should_load_tensors_in_parallel,
            partitioner_class,
            node_tf_dataset_options,
            edge_tf_dataset_options,
            splitter,
            _ssl_positive_label_percentage,
        ),
    )

    output_dataset: DistLinkPredictionDataset = output_dict["dataset"]

    logger.info(
        f"Dataset Building finished on rank {node_rank} of {node_world_size}, which took {time.time()-dataset_building_start_time:.2f} seconds"
    )

    return output_dataset


def build_dataset_from_task_config_uri(
    task_config_uri: Union[str, Uri],
    distributed_context: Optional[DistributedContext] = None,
    is_inference: bool = True,
    _tfrecord_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$",
) -> DistLinkPredictionDataset:
    """
    Builds a dataset from a provided `task_config_uri` as part of GiGL orchestration. Parameters to
    this step should be provided in the `inferenceArgs` field of the GbmlConfig for inference or the
    trainerArgs field of the GbmlConfig for training.

    It is expected that there is only one `build_dataset_from_task_config_uri` call per node (machine).
    This requirement exists to ensure each machine only participates once in housing a parition of a dataset; otherwise
    a machine may end up housing multiple partitions of the same dataset which may cause memory issues.


    This function expects that there is a process group initialized between the process' for the nodes participating in
    hosting the dataset partition. This is so necessary information can be communicated between the nodes
    i.e. free port information, master IP address, et al. to configure RPC. If there is no process group initialized,
    the function will initialize one using `env://` config. See :py:obj:`torch.distributed.init_process_group` for more info.


    The current parsable arguments are here are
    - sample_edge_direction (Literal["in", "out"]): Direction of the graph
    - should_use_range_partitioning (bool): Whether we should be using range-based partitioning
    - should_load_tensors_in_parallel (bool): Whether TFRecord loading should happen in parallel across entities
        Must be None if supervised edge labels are provided in advance.
        Slotted for refactor once this functionality is available in the transductive `splitter` directly.
    If training there are two additional arguments:
    - num_val (float): Percentage of edges to use for validation, defaults to 0.1. Must in in range [0, 1].
    - num_test (float): Percentage of edges to use for testing, defaults to 0.1. Must be in range [0, 1].
    - ssl_positive_label_percentage (Optional[float]): Percentage of edges to select as self-supervised labels.

    Args:
        task_config_uri (str): URI to a GBML Config
        distributed_context (Optional[DistributedContext]): Distributed context containing information for
            master_ip_address, rank, and world size. Defaults to None, in which case it will be initialized
            from the current torch.distributed context.
        is_inference (bool): Whether the run is for inference or training. If True, arguments will
            be read from inferenceArgs. Otherwise, arguments witll be read from trainerArgs.
        _tfrecord_uri_pattern (str): INTERNAL ONLY. Regex pattern for loading serialized tf records. Defaults to ".*-of-.*\.tfrecord(\.gz)?$".
    """

    if distributed_context is not None:
        logger.warning(
            "The `distributed_context` argument is deprecated and will be removed in a future release. "
            "Please setup the `torch.distributed.init_process_group` in the caller context and prevent "
            "passing `distributed_context` argument."
        )

    # Read from GbmlConfig for preprocessed data metadata, GNN model uri, and bigquery embedding table path
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )

    ssl_positive_label_percentage: Optional[float] = None
    if is_inference:
        args = dict(gbml_config_pb_wrapper.inferencer_config.inferencer_args)

        sample_edge_direction = args.get("sample_edge_direction", "in")
        args_path = "inferencerConfig.inferencerArgs"
        splitter = None
    else:
        args = dict(gbml_config_pb_wrapper.trainer_config.trainer_args)
        num_val = float(args.get("num_val", "0.1"))
        num_test = float(args.get("num_test", "0.1"))
        supervision_edge_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
            if gbml_config_pb_wrapper.graph_metadata_pb_wrapper.is_heterogeneous
            else [DEFAULT_HOMOGENEOUS_EDGE_TYPE]
        )
        sample_edge_direction = args.get("sample_edge_direction", "in")
        args_path = "trainerConfig.trainerArgs"
        # TODO(kmonte): Maybe we should enable `should_convert_labels_to_edges` as a flag?
        splitter = HashedNodeAnchorLinkSplitter(
            sampling_direction=sample_edge_direction,
            supervision_edge_types=supervision_edge_types,
            should_convert_labels_to_edges=True,
            num_val=num_val,
            num_test=num_test,
        )
        if "ssl_positive_label_percentage" in args:
            ssl_positive_label_percentage = float(args["ssl_positive_label_percentage"])

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
        tfrecord_uri_pattern=_tfrecord_uri_pattern,
    )

    # Need to do this "backwards" so the parent class can be defined first.
    # Otherwise, mypy complains that:
    # "expression has type "type[DistPartitioner]", variable has type "type[DistRangePartitioner]"
    if not should_use_range_partitioning:
        partitioner_class = DistPartitioner
    else:
        partitioner_class = DistRangePartitioner

    dataset = build_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        sample_edge_direction=sample_edge_direction,
        distributed_context=distributed_context,
        partitioner_class=partitioner_class,
        splitter=splitter,
        _ssl_positive_label_percentage=ssl_positive_label_percentage,
    )

    return dataset
