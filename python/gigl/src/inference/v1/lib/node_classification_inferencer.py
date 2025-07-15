from functools import partial
from typing import Callable, Dict, List

import apache_beam as beam

from gigl.common import Uri, UriFactory
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.inference.v1.lib.base_inference_blueprint import BaseInferenceBlueprint
from gigl.src.inference.v1.lib.base_inferencer import (
    BaseInferencer,
    SupervisedNodeClassificationBaseInferencer,
)
from gigl.src.inference.v1.lib.transforms.utils import cache_mappings
from gigl.src.training.v1.lib.data_loaders.supervised_node_classification_data_loader import (
    SupervisedNodeClassificationBatch,
)
from snapchat.research.gbml import (
    flattened_graph_metadata_pb2,
    training_samples_schema_pb2,
)


class NodeClassificationInferenceBlueprint(
    BaseInferenceBlueprint[
        training_samples_schema_pb2.SupervisedNodeClassificationSample,
        SupervisedNodeClassificationBatch,
    ]
):
    """
    Concrete NodeClassificationInferenceBlueprint class that implements functions in order
    to correctly compute and save inference results for SupervisedNodeClassification tasks.

    Implements Generics:
        RawSampleType = training_samples_schema_pb2.SupervisedNodeClassificationSample
        BatchType = SupervisedNodeClassificationBatch
    """

    def __init__(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        inferencer: BaseInferencer,
        graph_builder: GraphBuilder,
    ) -> None:
        # TODO (tzhao-sc): change these to args of functions s.t. we only initialize a property
        #              at the top level components
        self.__builder = graph_builder
        self.__gbml_config_pb_wrapper = gbml_config_pb_wrapper
        cache_mappings(gbml_config_pb_wrapper=self.__gbml_config_pb_wrapper)
        assert isinstance(inferencer, SupervisedNodeClassificationBaseInferencer)
        super().__init__(inferencer=inferencer)

    def get_inference_data_tf_record_uri_prefixes(self) -> Dict[NodeType, List[Uri]]:
        flattened_graph_metadata_pb_wrapper = (
            self.__gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper
        )
        assert isinstance(
            flattened_graph_metadata_pb_wrapper.output_metadata,
            flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput,
        )
        task_metadata_pb_wrapper = (
            self.__gbml_config_pb_wrapper.task_metadata_pb_wrapper
        )
        assert (
            task_metadata_pb_wrapper.task_metadata_type
            == TaskMetadataType.NODE_BASED_TASK
        ), f"Expected task metadata to be node based task, got {TaskMetadataType.NODE_BASED_TASK}"
        inferencer_node_types = (
            task_metadata_pb_wrapper.task_metadata_pb.node_based_task_metadata.supervision_node_types
        )
        if len(inferencer_node_types) != 1:
            raise NotImplementedError(
                f"Supervised node classification task expects one output node type, found {len(inferencer_node_types)} node types: {inferencer_node_types}"
            )
        return {
            NodeType(inferencer_node_types[0]): [
                UriFactory.create_uri(
                    flattened_graph_metadata_pb_wrapper.output_metadata.labeled_tfrecord_uri_prefix
                ),
                UriFactory.create_uri(
                    flattened_graph_metadata_pb_wrapper.output_metadata.unlabeled_tfrecord_uri_prefix
                ),
            ]
        }

    def get_tf_record_coder(self) -> beam.coders.ProtoCoder:
        coder = beam.coders.ProtoCoder(
            proto_message_type=training_samples_schema_pb2.SupervisedNodeClassificationSample
        )
        return coder

    def get_batch_generator_fn(self) -> Callable:
        return partial(
            SupervisedNodeClassificationBatch.process_raw_pyg_samples_and_collate_fn,
            builder=self.__builder,
            graph_metadata_pb_wrapper=self.__gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=self.__gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
        )
