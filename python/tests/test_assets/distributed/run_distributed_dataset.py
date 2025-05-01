from typing import Literal, MutableMapping, Optional, Type

from gigl.common.utils.vertex_ai_context import DistributedContext
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_link_prediction_data_partitioner import (
    DistLinkPredictionDataPartitioner,
)
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    get_mocked_dataset_artifact_metadata,
)
from gigl.utils.data_splitters import NodeAnchorLinkSplitter


def run_distributed_dataset(
    rank: int,
    world_size: int,
    mocked_dataset_info: MockedDatasetInfo,
    output_dict: MutableMapping[int, DistLinkPredictionDataset],
    should_load_tensors_in_parallel: bool,
    master_ip_address: str,
    master_port: int,
    partitioner_class: Optional[Type[DistLinkPredictionDataPartitioner]] = None,
    splitter: Optional[NodeAnchorLinkSplitter] = None,
) -> DistLinkPredictionDataset:
    """
    Runs DistLinkPredictionDataset Load() __init__ and load() functions provided a mocked dataset info
    Args:
        rank (int): Rank of the current process
        world_size (int): World size of the current process
        mocked_dataset_info (MockedDatasetInfo): Mocked Dataset Metadata for current run
        output_dict (MutableMapping[int, DistLinkPredictionDataset]): Dict initialized by mp.Manager().dict() in which outputs will be written to
        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        master_ip_address (str): Master IP Address for performing distributed operations.
        master_port (int) Master Port for performing distributed operations
        partitioner_class (Optional[Type[DistLinkPredictionDataPartitioner]]): Optional partitioner class to pass into `build_dataset`
        splitter (Optional[NodeAnchorLinkSplitter]): Provided splitter for testing
    """
    mocked_dataset_artifact_metadata: MockedDatasetArtifactMetadata = (
        get_mocked_dataset_artifact_metadata()[mocked_dataset_info.name]
    )
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=mocked_dataset_artifact_metadata.frozen_gbml_config_uri
    )
    preprocessed_metadata_pb_wrapper = (
        gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
    )
    graph_metadata_pb_wrapper = gbml_config_pb_wrapper.graph_metadata_pb_wrapper

    # When loading mocked inputs to inferencer, the TFRecords are read from format `data.tfrecord`. We update the
    # tfrecord_uri_pattern to expect this input.
    serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
        preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
        graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
        tfrecord_uri_pattern=".*\.tfrecord(\.gz)?$",
    )

    distributed_context = DistributedContext(
        main_worker_ip_address=master_ip_address,
        global_rank=rank,
        global_world_size=world_size,
    )

    sample_edge_direction: Literal["in", "out"] = "out"
    dataset = build_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        distributed_context=distributed_context,
        sample_edge_direction=sample_edge_direction,
        should_load_tensors_in_parallel=should_load_tensors_in_parallel,
        partitioner_class=partitioner_class,
        splitter=splitter,
        _dataset_building_port=master_port,
    )
    output_dict[rank] = dataset
    return dataset
