from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Iterable, List, Optional, TypeVar

import apache_beam as beam
from apache_beam import pvalue

from gigl.common import Uri
from gigl.src.common.types.graph_data import NodeType

# Raw data format that will be read from TFRecord files i.e. a proto class
from gigl.src.inference.v1.lib.base_inferencer import BaseInferencer
from gigl.src.inference.v1.lib.inference_output_schema import (
    DEFAULT_EMBEDDING_FIELD,
    DEFAULT_EMBEDDINGS_TABLE_SCHEMA,
    DEFAULT_NODE_ID_FIELD,
    DEFAULT_PREDICTION_FIELD,
    DEFAULT_PREDICTIONS_TABLE_SCHEMA,
    UNENUMERATED_EMBEDDINGS_TABLE_SCHEMA,
    UNENUMERATED_PREDICTIONS_TABLE_SCHEMA,
    InferenceOutputBigqueryTableSchema,
)

RawSampleType = TypeVar("RawSampleType")
# A batch representation of samples above that can be used to make inference more efficient.
BatchType = TypeVar("BatchType")

PREDICTION_TAGGED_OUTPUT_KEY = "predictions"
EMBEDDING_TAGGED_OUTPUT_KEY = "embeddings"


class BaseInferenceBlueprint(
    ABC,
    Generic[
        RawSampleType,
        BatchType,
    ],
):
    """
    Abstract Base Class that needs to be implemented for inference dataflow pipelines
    to correctly compute and save inference results for GBML tasks, such as
    Supervised Node Classification, Node Anchor-Based Link Prediction,
    Supervised Link-Based Task Split, etc.

    Implements Generics:
    - RawSampleType: The raw sample that will be parsed from get_tf_record_coder.
    - BatchType: The batch type needed for model inference (forward pass) for the specific task at hand (e.g RootedNodeNeighborhoodBatch).
    """

    def __init__(self, inferencer: BaseInferencer):
        self._inferencer = inferencer

    def get_inferer(
        self,
    ) -> Callable[[BatchType], Iterable[pvalue.TaggedOutput]]:
        """
        Returns a function that takes a DigestableBatchType object instance as input and yields TaggedOutputs
        with tags of either PREDICTION_TAGGED_OUTPUT_KEY or EMBEDDING_TAGGED_OUTPUT_KEY. The value is a Dict
        that can be directly written to BQ following the schemas defined in get_emb_table_schema for outputs
        with tag "embeddings" and get_pred_table_schema for outputs with tag PREDICTION_TAGGED_OUTPUT_KEY.

        For example, the following will be mapped to the predictions table:
        pvalue.TaggedOutput(
            PREDICTION_TAGGED_OUTPUT_KEY,
            {
                'source': 'Mahatma Gandhi', 'quote': 'My life is my message.'
            }
        )

        Note that the output follows the schema presented in get_pred_table_schema.
        """

        def _make_inference(
            batch: BatchType,
        ) -> Iterable[pvalue.TaggedOutput]:
            infer_batch_results = self._inferencer.infer_batch(batch=batch)
            for i, node in enumerate(batch.root_nodes):  # type: ignore
                pred: Optional[List[int]] = None
                emb: Optional[List[float]] = None
                predictions = infer_batch_results.predictions
                embeddings = infer_batch_results.embeddings
                if predictions is not None:
                    pred = predictions[i].tolist()
                if embeddings is not None:
                    emb = embeddings[i].tolist()

                if pred is not None:
                    yield pvalue.TaggedOutput(
                        PREDICTION_TAGGED_OUTPUT_KEY,
                        {
                            DEFAULT_NODE_ID_FIELD: node.id,
                            DEFAULT_PREDICTION_FIELD: pred,
                        },
                    )
                if emb is not None:
                    yield pvalue.TaggedOutput(
                        EMBEDDING_TAGGED_OUTPUT_KEY,
                        {
                            DEFAULT_NODE_ID_FIELD: node.id,
                            DEFAULT_EMBEDDING_FIELD: emb,
                        },
                    )

        return _make_inference

    @staticmethod
    def get_emb_table_schema(
        should_run_unenumeration: bool = False,
    ) -> InferenceOutputBigqueryTableSchema:
        """
        Returns the schema for the BQ table that will house embeddings.

        Returns:
        InferenceOutputBQTableSchema: Instance containing the schema and registered node field.
        See: https://beam.apache.org/documentation/io/built-in/google-bigquery/#creating-a-table-schema
        Example schema:
            'fields': [
                {'name': 'source', 'type': 'STRING', 'mode': 'NULLABLE'},
                {'name': 'quote', 'type': 'STRING', 'mode': 'REQUIRED'}
            ]
        """
        if should_run_unenumeration:
            return UNENUMERATED_EMBEDDINGS_TABLE_SCHEMA
        else:
            return DEFAULT_EMBEDDINGS_TABLE_SCHEMA

    @staticmethod
    def get_pred_table_schema(
        should_run_unenumeration: bool = False,
    ) -> InferenceOutputBigqueryTableSchema:
        """
        Returns the schema for the BQ table that will house predictions.

        Returns:
        InferenceOutputBQTableSchema: Instance containing the schema and registered node field.
        See: https://beam.apache.org/documentation/io/built-in/google-bigquery/#creating-a-table-schema
        Example schema:
            'fields': [
                {'name': 'source', 'type': 'STRING', 'mode': 'NULLABLE'},
                {'name': 'quote', 'type': 'STRING', 'mode': 'REQUIRED'}
            ]
        """
        if should_run_unenumeration:
            return UNENUMERATED_PREDICTIONS_TABLE_SCHEMA
        else:
            return DEFAULT_PREDICTIONS_TABLE_SCHEMA

    @abstractmethod
    def get_inference_data_tf_record_uri_prefixes(self) -> Dict[NodeType, List[Uri]]:
        """
        Returns:
            Dict[NodeType, List[Uri]]: Dictionary of node type to the list of uri prefixes where to find tf record files
            that will be used for inference
        """
        raise NotImplementedError

    @abstractmethod
    def get_tf_record_coder(self) -> beam.coders.ProtoCoder:
        """
        Returns:
            beam.coders.ProtoCoder: The coder used to parse the TFRecords to raw data samples of
            type RawSampleType
        """
        raise NotImplementedError

    @abstractmethod
    def get_batch_generator_fn(
        self,
    ) -> Callable:
        """
        Returns:
            Callable: The function specific to the batch type needed for the inference task at hand.
        """
        raise NotImplementedError
