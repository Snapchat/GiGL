from typing import MutableMapping, Optional, Type

import torch.distributed as dist

from gigl.common.data.load_torch_tensors import SerializedGraphMetadata
from gigl.common.utils.vertex_ai_context import DistributedContext
from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_link_prediction_dataset import DistLinkPredictionDataset
from gigl.distributed.dist_partitioner import DistPartitioner
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


def convert_mocked_dataset_info_to_serialized_graph_metadata(
    mocked_dataset_info: MockedDatasetInfo,
) -> SerializedGraphMetadata:
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

    # When loading mocked inputs, the TFRecords are read from format `data.tfrecord`. We update the
    # tfrecord_uri_pattern to expect this input.
    serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
        preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
        graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
        tfrecord_uri_pattern=".*\.tfrecord(\.gz)?$",
    )

    return serialized_graph_metadata


def run_distributed_dataset(
    rank: int,
    world_size: int,
    mocked_dataset_info: MockedDatasetInfo,
    should_load_tensors_in_parallel: bool,
    output_dict: Optional[MutableMapping[int, DistLinkPredictionDataset]] = None,
    partitioner_class: Optional[Type[DistPartitioner]] = None,
    splitter: Optional[NodeAnchorLinkSplitter] = None,
    _use_process_group: bool = True,  # TODO: (svij) Marked for deprecation, use_process_group will default to be True in the future
    _port: Optional[int] = None,  # TODO: (svij) Marked for deprecation
) -> DistLinkPredictionDataset:
    """
    Runs DistLinkPredictionDataset Load() __init__ and load() functions provided a mocked dataset info
    Args:
        rank (int): Rank of the current process
        world_size (int): World size of the current process
        mocked_dataset_info (MockedDatasetInfo): Mocked Dataset Metadata for current run

        should_load_tensors_in_parallel (bool): Whether tensors should be loaded from serialized information in parallel or in sequence across the [node, edge, pos_label, neg_label] entity types.
        output_dict (Optional[MutableMapping[int, DistLinkPredictionDataset]]): Dict initialized by mp.Manager().dict() in which outputs will be written to
        partitioner_class (Optional[Type[DistPartitioner]]): Optional partitioner class to pass into `build_dataset`
        splitter (Optional[NodeAnchorLinkSplitter]): Provided splitter for testing
    """
    try:
        distributed_context: Optional[DistributedContext] = None
        if _use_process_group:
            assert _port is not None, "Port must be provided when using process group."
            init_process_group_init_method = f"tcp://127.0.0.1:{_port}"
            dist.init_process_group(
                backend="gloo",
                init_method=init_process_group_init_method,
                rank=rank,
                world_size=world_size,
            )
        else:
            distributed_context = DistributedContext(
                main_worker_ip_address="localhost",
                global_rank=rank,
                global_world_size=world_size,
            )

        serialized_graph_metadata = (
            convert_mocked_dataset_info_to_serialized_graph_metadata(
                mocked_dataset_info=mocked_dataset_info
            )
        )

        sample_edge_direction = "out"
        dataset = build_dataset(
            serialized_graph_metadata=serialized_graph_metadata,
            distributed_context=distributed_context,
            sample_edge_direction=sample_edge_direction,
            should_load_tensors_in_parallel=should_load_tensors_in_parallel,
            partitioner_class=partitioner_class,
            splitter=splitter,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    if output_dict is not None:
        output_dict[rank] = dataset
    return dataset
