from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List

import apache_beam as beam

from gigl.common import Uri, UriFactory
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.inference.v1.lib.base_inference_blueprint import BaseInferenceBlueprint
from gigl.src.inference.v1.lib.base_inferencer import (
    BaseInferencer,
    NodeAnchorBasedLinkPredictionBaseInferencer,
)
from gigl.src.inference.v1.lib.transforms.utils import cache_mappings
from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)
from snapchat.research.gbml import (
    flattened_graph_metadata_pb2,
    training_samples_schema_pb2,
)


class NodeAnchorBasedLinkPredictionInferenceBlueprint(
    BaseInferenceBlueprint[
        training_samples_schema_pb2.RootedNodeNeighborhood,
        RootedNodeNeighborhoodBatch,
    ]
):
    """
    Concrete NodeAnchorBasedLinkPredictionInferenceBlueprint class that implements functions in order
    to correctly compute and save inference results for NodeAnchorBasedLinkPrediction task.

    Implements Generics:
        RawSampleType = training_samples_schema_pb2.RootedNodeNeighborhood
        BatchType = RootedNodeNeighborhoodBatch

    Note that this sample does inference on RootedNodeNeighborhood pbs as these are the full node neighborhoods
    required for inference. The NodeAnchorBasedLinkPrediction samples contain more information which
    is useful for training but unnecessary for inference.
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
        assert isinstance(inferencer, NodeAnchorBasedLinkPredictionBaseInferencer)
        super().__init__(inferencer=inferencer)

    def get_inference_data_tf_record_uri_prefixes(self) -> Dict[NodeType, List[Uri]]:
        flattened_graph_metadata_pb_wrapper = (
            self.__gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper
        )
        assert isinstance(
            flattened_graph_metadata_pb_wrapper.output_metadata,
            flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput,
        )
        node_type_to_tf_record_uri_prefixes: Dict[NodeType, List[Uri]] = defaultdict(
            list
        )
        node_type_to_random_negative_tfrecord_uri_prefix = (
            flattened_graph_metadata_pb_wrapper.output_metadata.node_type_to_random_negative_tfrecord_uri_prefix
        )
        for (
            node_type,
            uri_prefix,
        ) in node_type_to_random_negative_tfrecord_uri_prefix.items():
            uri: Uri = UriFactory.create_uri(uri_prefix)
            node_type_to_tf_record_uri_prefixes[NodeType(node_type)].append(uri)
        return node_type_to_tf_record_uri_prefixes

    def get_tf_record_coder(self) -> beam.coders.ProtoCoder:
        coder = beam.coders.ProtoCoder(
            proto_message_type=training_samples_schema_pb2.RootedNodeNeighborhood
        )
        return coder

    def get_batch_generator_fn(self) -> Callable:
        return partial(
            RootedNodeNeighborhoodBatch.process_raw_pyg_samples_and_collate_fn,
            builder=self.__builder,
            graph_metadata_pb_wrapper=self.__gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=self.__gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
        )
