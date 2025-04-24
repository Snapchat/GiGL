from dataclasses import dataclass
from typing import Dict, List, Type, cast

from gigl.common import Uri, UriFactory
from gigl.src.common.types.pb_wrappers.types import (
    DatasetMetadataPb,
    FlattenedGraphMetadataOutputPb,
    TrainingSamplePb,
)
from snapchat.research.gbml import (
    dataset_metadata_pb2,
    flattened_graph_metadata_pb2,
    training_samples_schema_pb2,
)

FLATTENED_GRAPH_TO_TRAINING_SAMPLE_TYPE: Dict[
    Type[FlattenedGraphMetadataOutputPb], Type[TrainingSamplePb]
] = {
    flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput: training_samples_schema_pb2.SupervisedNodeClassificationSample,
    flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput: training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
    flattened_graph_metadata_pb2.SupervisedLinkBasedTaskOutput: training_samples_schema_pb2.SupervisedLinkBasedTaskSample,
}


FLATTENED_GRAPH_TO_DATASET_TYPE: Dict[
    Type[FlattenedGraphMetadataOutputPb], Type[DatasetMetadataPb]
] = {
    flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput: dataset_metadata_pb2.SupervisedNodeClassificationDataset,
    flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput: dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset,
    flattened_graph_metadata_pb2.SupervisedLinkBasedTaskOutput: dataset_metadata_pb2.SupervisedLinkBasedTaskSplitDataset,
}


@dataclass
class FlattenedGraphMetadataPbWrapper:
    flattened_graph_metadata_pb: flattened_graph_metadata_pb2.FlattenedGraphMetadata

    @property
    def output_metadata(self) -> FlattenedGraphMetadataOutputPb:
        """
        Returns the relevant FlattenedGraphMetadataOutputPb instance
        (e.g. an instance of SupervisedNodeClassificationOutput).
        :return:
        """
        field = cast(
            str, self.flattened_graph_metadata_pb.WhichOneof("output_metadata")
        )
        output = getattr(self.flattened_graph_metadata_pb, field)
        return output

    @property
    def output_metadata_type(self) -> Type[FlattenedGraphMetadataOutputPb]:
        """
        Returns the type of the output from SubgraphSampler. (e.g. SupervisedNodeClassificationOutput)
        :return:
        """
        flattened_graph_metadata_output_type = type(self.output_metadata)
        return flattened_graph_metadata_output_type

    @property
    def training_sample_type(self) -> Type[TrainingSamplePb]:
        """
        Returns the corresponding type of TrainingSamplePb to parse protos from SubgraphSampler as.
        (e.g. SupervisedNodeClassificationSample)
        :return:
        """
        training_sample_type = FLATTENED_GRAPH_TO_TRAINING_SAMPLE_TYPE[
            self.output_metadata_type
        ]
        return training_sample_type

    @property
    def dataset_type(self) -> Type[DatasetMetadataPb]:
        """
        Returns the corresponding type of output for SplitGenerator to generate
        (e.g. SupervisedNodeClassificationSplitOutput)
        :return:
        """
        dataset_type = FLATTENED_GRAPH_TO_DATASET_TYPE[self.output_metadata_type]
        return dataset_type

    def get_output_paths(self) -> List[Uri]:
        """
        Returns a list of output paths referenced by the output metadata.
        :return:
        """
        metadata_pb = self.output_metadata
        paths = list()
        if isinstance(
            metadata_pb, flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput
        ):
            paths += [
                metadata_pb.labeled_tfrecord_uri_prefix,
                metadata_pb.unlabeled_tfrecord_uri_prefix,
            ]
        elif isinstance(
            metadata_pb,
            flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput,
        ):
            paths += list(
                metadata_pb.node_type_to_random_negative_tfrecord_uri_prefix.values()
            ) + [metadata_pb.tfrecord_uri_prefix]
        elif isinstance(
            metadata_pb, flattened_graph_metadata_pb2.SupervisedLinkBasedTaskOutput
        ):
            paths += [
                metadata_pb.labeled_tfrecord_uri_prefix,
                metadata_pb.unlabeled_tfrecord_uri_prefix,
            ]

        uri_paths = [UriFactory.create_uri(uri=path) for path in paths if path]
        return uri_paths
