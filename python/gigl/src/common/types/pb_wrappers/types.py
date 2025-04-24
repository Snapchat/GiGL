from typing import Union

from snapchat.research.gbml import (
    dataset_metadata_pb2,
    flattened_graph_metadata_pb2,
    gbml_config_pb2,
    training_samples_schema_pb2,
)

FlattenedGraphMetadataOutputPb = Union[
    flattened_graph_metadata_pb2.SupervisedNodeClassificationOutput,
    flattened_graph_metadata_pb2.NodeAnchorBasedLinkPredictionOutput,
    flattened_graph_metadata_pb2.SupervisedLinkBasedTaskOutput,
]

# TODO: (svij-sc) Clean this up and make it reflective of tests/integration/pipeline_tests/split_generator_pipeline_test.py#TSample
TrainingSamplePb = Union[
    training_samples_schema_pb2.SupervisedNodeClassificationSample,
    training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
    training_samples_schema_pb2.RootedNodeNeighborhood,
    training_samples_schema_pb2.SupervisedLinkBasedTaskSample,
]
DatasetMetadataPb = Union[
    dataset_metadata_pb2.SupervisedNodeClassificationDataset,
    dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset,
    dataset_metadata_pb2.SupervisedLinkBasedTaskSplitDataset,
]

TaskMetadataPb = Union[
    gbml_config_pb2.GbmlConfig.TaskMetadata.NodeBasedTaskMetadata,
    gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata,
    gbml_config_pb2.GbmlConfig.TaskMetadata.LinkBasedTaskMetadata,
]
