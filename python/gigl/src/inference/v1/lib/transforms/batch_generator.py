from typing import Iterable, List, Union

import apache_beam as beam

from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)
from gigl.src.training.v1.lib.data_loaders.supervised_node_classification_data_loader import (
    SupervisedNodeClassificationBatch,
)
from snapchat.research.gbml import training_samples_schema_pb2

RawBatchType = Union[
    training_samples_schema_pb2.RootedNodeNeighborhood,
    training_samples_schema_pb2.SupervisedNodeClassificationSample,
]

InferenceBatchType = Union[
    SupervisedNodeClassificationBatch, RootedNodeNeighborhoodBatch
]


class BatchProcessorDoFn(beam.DoFn):
    def __init__(
        self,
        batch_generator_fn,
    ):
        self.batch_generator_fn = batch_generator_fn

    def process(self, element: List[RawBatchType]) -> Iterable[InferenceBatchType]:
        yield self.batch_generator_fn(
            batch=element,
        )
