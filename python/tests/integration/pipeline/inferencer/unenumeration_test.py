import unittest

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from gigl.common.logger import Logger
from gigl.src.data_preprocessor.lib.enumerate.queries import (
    DEFAULT_ENUMERATED_NODE_ID_FIELD,
    DEFAULT_ORIGINAL_NODE_ID_FIELD,
)
from gigl.src.inference.v1.lib.base_inference_blueprint import (
    EMBEDDING_TAGGED_OUTPUT_KEY,
    PREDICTION_TAGGED_OUTPUT_KEY,
)
from gigl.src.inference.v1.lib.inference_output_schema import (
    DEFAULT_EMBEDDING_FIELD,
    DEFAULT_NODE_ID_FIELD,
    DEFAULT_PREDICTION_FIELD,
)
from gigl.src.inference.v1.lib.utils import UnenumerateAssets

logger = Logger()


class UnenumeratorTest(unittest.TestCase):
    """Tests un-enumeration functionality in Inferencer by validating that
    we can un-enumerate an asset using an id mapping similar to that produced
    by the enumerator inside Data Preprocessor.
    """

    def setUp(self) -> None:
        self.toy_inferencer_prediction_output = [
            {DEFAULT_NODE_ID_FIELD: 1, DEFAULT_PREDICTION_FIELD: 10},
            {DEFAULT_NODE_ID_FIELD: 2, DEFAULT_PREDICTION_FIELD: 15},
            {DEFAULT_NODE_ID_FIELD: 3, DEFAULT_PREDICTION_FIELD: 20},
        ]
        self.toy_inferencer_embedding_output = [
            {DEFAULT_NODE_ID_FIELD: 1, DEFAULT_EMBEDDING_FIELD: [0.1, 0.2, 0.3]},
            {DEFAULT_NODE_ID_FIELD: 2, DEFAULT_EMBEDDING_FIELD: [0.4, 0.5, 0.6]},
            {DEFAULT_NODE_ID_FIELD: 3, DEFAULT_EMBEDDING_FIELD: [0.7, 0.8, 0.9]},
        ]

        self.toy_inferencer_id_mapping = [
            {DEFAULT_ENUMERATED_NODE_ID_FIELD: 1, DEFAULT_ORIGINAL_NODE_ID_FIELD: 100},
            {DEFAULT_ENUMERATED_NODE_ID_FIELD: 2, DEFAULT_ORIGINAL_NODE_ID_FIELD: 1000},
            {DEFAULT_ENUMERATED_NODE_ID_FIELD: 3, DEFAULT_ORIGINAL_NODE_ID_FIELD: 5000},
        ]

        self.expected_output_predictions = [
            {DEFAULT_NODE_ID_FIELD: 100, DEFAULT_PREDICTION_FIELD: 10},
            {DEFAULT_NODE_ID_FIELD: 1000, DEFAULT_PREDICTION_FIELD: 15},
            {DEFAULT_NODE_ID_FIELD: 5000, DEFAULT_PREDICTION_FIELD: 20},
        ]

        self.expected_output_embeddings = [
            {DEFAULT_NODE_ID_FIELD: 100, DEFAULT_EMBEDDING_FIELD: [0.1, 0.2, 0.3]},
            {DEFAULT_NODE_ID_FIELD: 1000, DEFAULT_EMBEDDING_FIELD: [0.4, 0.5, 0.6]},
            {DEFAULT_NODE_ID_FIELD: 5000, DEFAULT_EMBEDDING_FIELD: [0.7, 0.8, 0.9]},
        ]

    def tearDown(self) -> None:
        pass

    def test_can_unenumerate_asset(self):
        with TestPipeline() as p:
            predictions = p | "Create Predictions" >> beam.Create(
                self.toy_inferencer_prediction_output
            )
            embeddings = p | "Create Embeddings" >> beam.Create(
                self.toy_inferencer_embedding_output
            )
            id_mapping = (
                p
                | "Create Id Mapping" >> beam.Create(self.toy_inferencer_id_mapping)
                | "Prepare id_mapping for join"
                >> beam.Map(
                    lambda row: (
                        row[DEFAULT_ENUMERATED_NODE_ID_FIELD],
                        row[DEFAULT_ORIGINAL_NODE_ID_FIELD],
                    )
                )
            )

            logger.info(
                "Finished creating all prediction, embedding, and id_mapping assets"
            )

            unenumerated_predictions = (
                predictions,
                id_mapping,
            ) | "Unenumerate Predictions" >> UnenumerateAssets(
                tagged_output_key=PREDICTION_TAGGED_OUTPUT_KEY
            )

            logger.info("Finished unenumerating predictions")

            unenumerated_embeddings = (
                embeddings,
                id_mapping,
            ) | "Unenumerate Embeddings" >> UnenumerateAssets(
                tagged_output_key=EMBEDDING_TAGGED_OUTPUT_KEY
            )

            logger.info("Finished unenumerating embeddings")

            assert_that(
                unenumerated_predictions,
                equal_to(self.expected_output_predictions),
                label="assert_predictions",
            )
            assert_that(
                unenumerated_embeddings,
                equal_to(self.expected_output_embeddings),
                label="assert_embeddings",
            )
