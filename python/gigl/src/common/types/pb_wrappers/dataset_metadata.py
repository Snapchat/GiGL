from dataclasses import dataclass
from typing import Dict, List, Type, cast

from gigl.common import Uri, UriFactory
from gigl.src.common.types.pb_wrappers.types import DatasetMetadataPb, TrainingSamplePb
from snapchat.research.gbml import dataset_metadata_pb2, training_samples_schema_pb2

DATASET_TO_TRAINING_SAMPLE_TYPE: Dict[
    Type[DatasetMetadataPb], Type[TrainingSamplePb]
] = {
    dataset_metadata_pb2.SupervisedNodeClassificationDataset: training_samples_schema_pb2.SupervisedNodeClassificationSample,
    dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset: training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
    dataset_metadata_pb2.SupervisedLinkBasedTaskSplitDataset: training_samples_schema_pb2.SupervisedLinkBasedTaskSample,
}


@dataclass(frozen=True)
class DatasetMetadataPbWrapper:
    dataset_metadata_pb: dataset_metadata_pb2.DatasetMetadata

    @property
    def output_metadata(self) -> DatasetMetadataPb:
        """
        Returns the relevant SplitSample instance
        (e.g. an instance of SupervisedNodeClassificationDataset).
        :return:
        """
        field = cast(str, self.dataset_metadata_pb.WhichOneof("output_metadata"))
        output_metadata = getattr(self.dataset_metadata_pb, field)
        return output_metadata

    @property
    def output_metadata_type(self) -> Type[DatasetMetadataPb]:
        """
        Returns the type of the dataset from SplitGenerator.
        (e.g. SupervisedNodeClassificationSplitDataset)
        :return:
        """
        output_metadata_type = type(self.output_metadata)
        return output_metadata_type

    @property
    def training_sample_type(self) -> Type[TrainingSamplePb]:
        """
        Returns the corresponding type of TrainingSample to parse protos from SplitGenerator as.
        (e.g. SupervisedNodeClassificationSample)
        :return:
        """
        training_sample_type = DATASET_TO_TRAINING_SAMPLE_TYPE[
            self.output_metadata_type
        ]
        return training_sample_type

    def get_output_paths(self) -> List[Uri]:
        """
        Returns a list of output paths referenced by the output metadata.
        :return:
        """
        metadata_pb = self.output_metadata
        paths = list()
        if isinstance(
            metadata_pb, dataset_metadata_pb2.SupervisedNodeClassificationDataset
        ):
            paths += [
                metadata_pb.train_data_uri,
                metadata_pb.val_data_uri,
                metadata_pb.test_data_uri,
            ]
        elif isinstance(
            metadata_pb, dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset
        ):
            paths += [
                metadata_pb.train_main_data_uri,
                metadata_pb.val_main_data_uri,
                metadata_pb.test_main_data_uri,
            ]
            paths += list(
                metadata_pb.train_node_type_to_random_negative_data_uri.values()
            )
            paths += list(
                metadata_pb.val_node_type_to_random_negative_data_uri.values()
            )
            paths += list(
                metadata_pb.test_node_type_to_random_negative_data_uri.values()
            )
        elif isinstance(
            metadata_pb, dataset_metadata_pb2.SupervisedLinkBasedTaskSplitDataset
        ):
            paths += [
                metadata_pb.train_data_uri,
                metadata_pb.val_data_uri,
                metadata_pb.test_data_uri,
            ]

        uri_paths = [UriFactory.create_uri(uri=path) for path in paths if path]
        return uri_paths
