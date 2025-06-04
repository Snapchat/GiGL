import time
from functools import lru_cache
from typing import Dict, Optional, Tuple, Union

import psutil
import torch
from graphlearn_torch.distributed import init_rpc, init_worker_group
from graphlearn_torch.partition import PartitionBook, RangePartitionBook

from gigl.common import UriFactory
from gigl.common.data.dataloaders import SerializedTFRecordInfo
from gigl.common.data.load_torch_tensors import SerializedGraphMetadata
from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict
from gigl.types.graph import to_homogeneous
from snapchat.research.gbml.preprocessed_metadata_pb2 import PreprocessedMetadata

logger = Logger()


def get_available_device(local_process_rank: int) -> torch.device:
    r"""Returns the available device for the current process.

    Args:
        local_process_rank (int): The local rank of the current process within a node.
    Returns:
        torch.device: The device to use.
    """
    device = torch.device(
        "cpu"
        if not torch.cuda.is_available()
        # If the number of processes are larger than the available GPU,
        # we assign each process to one GPU in a round robin manner.
        else f"cuda:{local_process_rank % torch.cuda.device_count()}"
    )
    return device


_is_cpu_env_initialized = False


def get_process_group_name(process_rank: int) -> str:
    """
    Returns the name of the process group for the given process rank.
    Args:
        process_rank (int): The rank of the process.
    Returns:
        str: The name of the process group.
    """
    return f"distributed-process-{process_rank}"


# torch.set_num_interop_threads() can only be called once, otherwise we see:
# RuntimeError: Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called
# Since we don't need to re-setup the identical worker pools, etc, we can just "cache" this call.
# That way the "side-effects" of the call are only executed once.
@lru_cache(maxsize=1)
def init_neighbor_loader_worker(
    master_ip_address: str,
    local_process_rank: int,
    local_process_world_size: int,
    rank: int,
    world_size: int,
    master_worker_port: int,
    device: torch.device,
    should_use_cpu_workers: bool = False,
    num_cpu_threads: Optional[int] = None,
    process_start_gap_seconds: float = 60.0,
) -> None:
    """
    Sets up processes and torch device for initializing the GLT DistNeighborLoader, setting up RPC and worker groups to minimize
    the memory overhead and CPU contention. Returns the torch device which current worker is assigned to.
    Args:
        master_ip_address (str): Master IP Address to manage processes
        local_process_rank (int): Process number on the current machine
        local_process_world_size (int): Total number of processes on the current machine
        rank (int): Rank of current machine
        world_size (int): Total number of machines
        master_worker_port (int): Master port to use for communicating between workers during training or inference
        device (torch.device): The device where you want to load the data onto - i.e. where is your model?
        should_use_cpu_workers (bool): Whether we should do CPU training or inference.
        num_cpu_threads (Optional[int]): Number of cpu threads PyTorch should use for CPU training or inference.
            Must be set if should_use_cpu_workers is True.
        process_start_gap_seconds (float): Delay between each process for initializing neighbor loader. At large scales, it is recommended to set
            this value to be between 60 and 120 seconds -- otherwise multiple processes may attempt to initialize dataloaders at overlapping timesß,
            which can cause CPU memory OOM.
    Returns:
        torch.device: Device which current worker is assigned to
    """

    global _is_cpu_env_initialized

    # When initiating data loader(s), there will be a spike of memory usage lasting for ~30s.
    # The current hypothesis is making connections across machines require a lot of memory.
    # If we start all data loaders in all processes simultaneously, the spike of memory
    # usage will add up and cause CPU memory OOM. Hence, we initiate the data loaders group by group
    # to smooth the memory usage. The definition of group is discussed below.
    logger.info(
        f"---Machine {rank} local process number {local_process_rank} preparing to sleep for {process_start_gap_seconds * local_process_rank} seconds"
    )
    time.sleep(process_start_gap_seconds * local_process_rank)
    logger.info(f"---Machine {rank} local process number {local_process_rank} started")
    if not should_use_cpu_workers:
        assert (
            torch.cuda.device_count() > 0
        ), f"Must have at least 1 GPU available for GPU Training or inference, got {torch.cuda.device_count()}"

    if should_use_cpu_workers:
        assert (
            num_cpu_threads is not None
        ), "Must provide number of cpu threads when using cpu workers"
        # Assign processes to disjoint physical cores. Since training or inference is computation
        # bound instead of I/O bound, logical core segmentation is not enough, as two
        # hyperthreads on the same physical core could still compete for resources.

        # Compute the range of physical cores the process should run on.
        total_physical_cores = psutil.cpu_count(logical=False)
        if total_physical_cores is None:
            raise ValueError("Was unable to determine the number of physical cores")
        physical_cores_per_process = total_physical_cores // local_process_world_size
        start_physical_core = local_process_rank * physical_cores_per_process
        end_physical_core = (
            total_physical_cores
            if local_process_rank == local_process_world_size - 1
            else start_physical_core + physical_cores_per_process
        )

        # Essentially we could only specify the logical cores the process should run
        # on, so we have to map physical cores to logical cores. For GCP machines,
        # logical cores are assigned to physical cores in a round robin manner, i.e.,
        # if there are 4 physical cores, logical cores 0, 1, 2, 3, will be assigned
        # to physical cores 0, 1, 2, 3. Logical core 4 will be assigned to physical
        # core 0, logical core 5 will be assigned to physical core 1, etc. However,
        # this mapping does not always hold. Some VM assigns logical cores 0 and 1 to
        # physical core 0, and assigns logical cores 2, 3 to physical core 1. We could
        # to check it by running `lscpu -p` command in the terminal.
        first_logical_core_range = list(range(start_physical_core, end_physical_core))
        second_logical_core_range = list(
            range(
                start_physical_core + total_physical_cores,
                end_physical_core + total_physical_cores,
            )
        )
        logical_cores = first_logical_core_range + second_logical_core_range

        if not _is_cpu_env_initialized:
            torch.set_num_interop_threads(num_cpu_threads)
            torch.set_num_threads(num_cpu_threads)

            # Set the logical cpu cores the current process shoud run on. Note
            # that the sampling process spawned by the process will inherit
            # this setting, meaning that sampling process will run on the same group
            # of logical cores. However, the sampling process is network bound so
            # it may not heavily compete resouce with model training or inference.
            p = psutil.Process()
            p.cpu_affinity(logical_cores)
            _is_cpu_env_initialized = True
        else:
            logger.info("Logical CPU cores has already been set for current process.")

    else:
        # Setting the default CUDA device for the current process to be the
        # device. Without it, there will be a process created on cuda:0 device, and
        # another process created on the device. Consequently, there will be
        # more processes running on cuda:0 than other cuda devices. The processes on
        # cuda:0 will compete for memory and could cause CUDA OOM.
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        logger.info(
            f"Machine {rank} local rank {local_process_rank} uses device {torch.cuda.current_device()} by default"
        )

    # Group of workers. Each process is a worker. Each
    # worker will initiate one model and at least one data loader. Each data loader
    # will spawn several sampling processes (a.k.a. sampling workers).
    # Instead of combining all workers into one group, we define N groups where
    # N is the number of processes on each machine. Specifically, we have
    # Group 0: (Machine 0, process 0), (Machine 1, process 0),..., (Machine M, process 0)
    # Group 1: (Machine 0, process 1), (Machine 1, process 1),..., (Machine M, process 1)
    # ...
    # Group N-1: (Machine 0, process N-1), (Machine 1, process N-1),..., (Machine M, process N-1)
    # We do this as we want to start different groups in different times to smooth
    # the spike of memory usage as mentioned above.

    group_name = get_process_group_name(local_process_rank)
    logger.info(
        f"Init worker group with: world_size={world_size}, rank={rank}, group_name={group_name}, "
    )
    init_worker_group(
        world_size=world_size,
        rank=rank,
        group_name=group_name,
    )

    # Initialize the communication channel across all workers in one group, so
    # that we could add barrier and wait all workers to finish before quitting.
    # Note that all sampling workers across all processeses in one group need to
    # be connected for graph sampling. Thus, a worker needs to wait others even
    # if it finishes, as quiting process will shutdown the correpsonding sampling
    # workers, and break the connection with other sampling workers.
    # Note that different process groups are independent of each other. Therefore,
    # they have to use different master ports.
    logger.info(
        f"Initing worker group with: world_size={world_size}, rank={rank}, group_name={group_name}, "
    )
    init_rpc(
        master_addr=master_ip_address,
        master_port=master_worker_port + local_process_rank,
        rpc_timeout=600,
    )

    logger.info(f"Group {group_name} with rpc is initiated")


def _get_ids_from_range_partition_book(
    range_partition_book: PartitionBook, rank: int
) -> torch.Tensor:
    """
    This function is very similar to RangePartitionBook.id_filter(). However, we re-implement this here, since the usage-pattern for that is a bit strange
    i.e. range_partition_book.id_filter(node_pb=range_partition_book, partition_idx=rank).
    """
    assert isinstance(range_partition_book, RangePartitionBook)
    start_node_id = range_partition_book.partition_bounds[rank - 1] if rank > 0 else 0
    end_node_id = range_partition_book.partition_bounds[rank]
    return torch.arange(start_node_id, end_node_id, dtype=torch.int64)


def get_ids_on_rank(
    partition_book: Union[torch.Tensor, PartitionBook],
    rank: int,
) -> torch.Tensor:
    """
    Provided a tensor-based partition book or a range-based bartition book and a rank, returns all the ids that are stored on that rank.
    Args:
        partition_book (Union[torch.Tensor, PartitionBook]): Tensor or range-based partition book
        rank (int): Rank of current machine
    """
    if isinstance(partition_book, torch.Tensor):
        return torch.nonzero(partition_book == rank).squeeze(dim=1)
    else:
        return _get_ids_from_range_partition_book(
            range_partition_book=partition_book, rank=rank
        )


def _build_serialized_tfrecord_entity_info(
    preprocessed_metadata: Union[
        PreprocessedMetadata.NodeMetadataOutput, PreprocessedMetadata.EdgeMetadataInfo
    ],
    feature_spec_dict: FeatureSpecDict,
    entity_key: Union[str, Tuple[str, str]],
    tfrecord_uri_pattern: str,
) -> SerializedTFRecordInfo:
    """
    Populates a SerializedTFRecordInfo field from provided arguments for either a node or edge entity of a single node/edge type.
    Args:
        preprocessed_metadata(Union[
            PreprocessedMetadata.NodeMetadataOutput, PreprocessedMetadata.EdgeMetadataInfo
        ]): Preprocessed metadata pb for either NodeMetadataOutput or EdgeMetadataInfo
        feature_spec_dict (FeatureSpecDict): Feature spec to register to SerializedTFRecordInfo
        entity_key (Union[str, Tuple[str, str]]): Entity key to register to SerializedTFRecordInfo, is a str if Node entity or Tuple[str, str] if Edge entity
        tfrecord_uri_pattern (str): Regex pattern for loading serialized tf records
    Returns:
        SerializedTFRecordInfo: Stored metadata for current entity
    """
    return SerializedTFRecordInfo(
        tfrecord_uri_prefix=UriFactory.create_uri(
            preprocessed_metadata.tfrecord_uri_prefix
        ),
        feature_keys=list(preprocessed_metadata.feature_keys),
        feature_spec=feature_spec_dict,
        feature_dim=preprocessed_metadata.feature_dim,
        entity_key=entity_key,
        tfrecord_uri_pattern=tfrecord_uri_pattern,
    )


def convert_pb_to_serialized_graph_metadata(
    preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
    graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
    tfrecord_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$",
) -> SerializedGraphMetadata:
    """
    Populates a SerializedGraphMetadata field from PreprocessedMetadataPbWrapper and GraphMetadataPbWrapper, containing information for loading tensors for all entities and node/edge types.
    Args:
        preprocessed_metadata_pb_wrapper (PreprocessedMetadataPbWrapper): Preprocessed Metadata Pb Wrapper to translate into SerializedGraphMetadata
        graph_metadata_pb_wrapper (GraphMetadataPbWrapper): Graph Metadata Pb Wrapper to translate into Dataset Metadata
        tfrecord_uri_pattern (str): Regex pattern for loading serialized tf records
    Returns:
        SerializedGraphMetadata: Dataset Metadata for all entity and node/edge types.
    """

    node_entity_info: Dict[NodeType, SerializedTFRecordInfo] = {}
    edge_entity_info: Dict[EdgeType, SerializedTFRecordInfo] = {}
    positive_label_entity_info: Dict[EdgeType, Optional[SerializedTFRecordInfo]] = {}
    negative_label_entity_info: Dict[EdgeType, Optional[SerializedTFRecordInfo]] = {}

    preprocessed_metadata_pb = preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb

    for node_type in graph_metadata_pb_wrapper.node_types:
        condensed_node_type = (
            graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[node_type]
        )
        node_metadata = (
            preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
                condensed_node_type
            ]
        )

        node_feature_spec_dict = (
            preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_schema_map[
                condensed_node_type
            ].feature_spec
        )

        node_key = node_metadata.node_id_key

        node_entity_info[node_type] = _build_serialized_tfrecord_entity_info(
            preprocessed_metadata=node_metadata,
            feature_spec_dict=node_feature_spec_dict,
            entity_key=node_key,
            tfrecord_uri_pattern=tfrecord_uri_pattern,
        )

    for edge_type in graph_metadata_pb_wrapper.edge_types:
        condensed_edge_type = (
            graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[edge_type]
        )

        edge_metadata = (
            preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[
                condensed_edge_type
            ]
        )

        edge_feature_spec_dict = (
            preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_schema_map[
                condensed_edge_type
            ].feature_spec
        )

        edge_key = (
            edge_metadata.src_node_id_key,
            edge_metadata.dst_node_id_key,
        )

        edge_entity_info[edge_type] = _build_serialized_tfrecord_entity_info(
            preprocessed_metadata=edge_metadata.main_edge_info,
            feature_spec_dict=edge_feature_spec_dict,
            entity_key=edge_key,
            tfrecord_uri_pattern=tfrecord_uri_pattern,
        )

        if preprocessed_metadata_pb_wrapper.has_pos_edge_features(
            condensed_edge_type=condensed_edge_type
        ):
            pos_edge_feature_spec_dict = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_pos_edge_feature_schema_map[
                condensed_edge_type
            ].feature_spec

            positive_label_entity_info[
                edge_type
            ] = _build_serialized_tfrecord_entity_info(
                preprocessed_metadata=edge_metadata.positive_edge_info,
                feature_spec_dict=pos_edge_feature_spec_dict,
                entity_key=edge_key,
                tfrecord_uri_pattern=tfrecord_uri_pattern,
            )
        else:
            positive_label_entity_info[edge_type] = None

        if preprocessed_metadata_pb_wrapper.has_hard_neg_edge_features(
            condensed_edge_type=condensed_edge_type
        ):
            hard_neg_edge_feature_spec_dict = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_hard_neg_edge_feature_schema_map[
                condensed_edge_type
            ].feature_spec

            negative_label_entity_info[
                edge_type
            ] = _build_serialized_tfrecord_entity_info(
                preprocessed_metadata=edge_metadata.negative_edge_info,
                feature_spec_dict=hard_neg_edge_feature_spec_dict,
                entity_key=edge_key,
                tfrecord_uri_pattern=tfrecord_uri_pattern,
            )
        else:
            negative_label_entity_info[edge_type] = None

    if not graph_metadata_pb_wrapper.is_heterogeneous:
        # If our input is homogeneous, we remove the node/edge type component of the metadata fields.
        return SerializedGraphMetadata(
            node_entity_info=to_homogeneous(node_entity_info),
            edge_entity_info=to_homogeneous(edge_entity_info),
            positive_label_entity_info=to_homogeneous(positive_label_entity_info),
            negative_label_entity_info=to_homogeneous(negative_label_entity_info),
        )
    else:
        return SerializedGraphMetadata(
            node_entity_info=node_entity_info,
            edge_entity_info=edge_entity_info,
            positive_label_entity_info=positive_label_entity_info
            if not all(
                entity_info is None
                for entity_info in positive_label_entity_info.values()
            )
            else None,
            negative_label_entity_info=negative_label_entity_info
            if not all(
                entity_info is None
                for entity_info in negative_label_entity_info.values()
            )
            else None,
        )
