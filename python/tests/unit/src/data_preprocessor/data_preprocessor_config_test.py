from copy import deepcopy
from typing import List

import tensorflow_transform.beam as tft_beam
from tensorflow_transform.beam import tft_unit
from tensorflow_transform.tf_metadata.dataset_metadata import DatasetMetadata

from gigl.src.data_preprocessor.lib.data_preprocessor_config import (
    build_ingestion_feature_spec_fn,
    build_passthrough_transform_preprocessing_fn,
)
from gigl.src.data_preprocessor.lib.types import InstanceDict


class DataPreprocessorConfigTest(tft_unit.TransformTestCase):
    def test_passthrough_ingestion_and_preprocessing_fn(self):
        """
        Checks whether ingestion and transform utils from DataPreprocessorConfig enable us to ingest toy data and run
        TFTransform with expected output results.
        :return:
        """
        # Create some mock input data.
        input_data: List[InstanceDict] = [
            {"a": 1, "b": 1.5, "c": "first", "d": [0.1, 0.1]},
            {"a": 2, "b": 2.5, "c": "second", "d": [0.2, 0.2]},
            {"a": 3, "b": 3.5, "c": "third", "d": [0.3, 0.3]},
            {"a": 4, "b": 4.5, "c": "fourth", "d": [0.4, 0.4]},
        ]

        # Use DataPreprocessorConfig utils to generate a feature spec.
        input_feature_spec = build_ingestion_feature_spec_fn(
            fixed_int_fields=["a"],
            fixed_float_fields=["b", "d"],
            fixed_float_field_shapes={"d": [2]},
            fixed_string_fields=["c"],
        )()

        # Build some input metadata from this feature spec.
        input_metadata = DatasetMetadata.from_feature_spec(
            feature_spec=input_feature_spec
        )

        # Create expected outputs to be equivalent to inputs for no-op / passthrough TFTransform.
        expected_data = deepcopy(input_data)
        expected_metadata = deepcopy(input_metadata)

        # Use DataPreprocessorConfig utils to generate a passthrough TFTransform preprocessing_fn.
        preprocess_fn = build_passthrough_transform_preprocessing_fn()

        # Check that the pipeline runs and transform outputs match inputs.
        with tft_beam.Context(use_deep_copy_optimization=True):
            self.assertAnalyzeAndTransformResults(
                input_data,
                input_metadata,
                preprocess_fn,
                expected_data,
                expected_metadata,
            )
