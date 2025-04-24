from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

import tensorflow as tf

from gigl.common.logger import Logger
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import (
    DEFAULT_TF_FLOAT_DTYPE,
    DEFAULT_TF_INT_DTYPE,
    DEFAULT_TF_STRING_DTYPE,
    EdgeDataPreprocessingSpec,
    FeatureSpecDict,
    NodeDataPreprocessingSpec,
    TFTensorDict,
)

logger = Logger()


class DataPreprocessorConfig(ABC):
    """
    Users should inherit from this and define the relevant specs for their preprocessing job.
    """

    def prepare_for_pipeline(
        self, applied_task_identifier: AppliedTaskIdentifier
    ) -> None:
        """
        This function is called at the very start of the pipeline before enumerator and datapreprocessor.
        This function does not return anything. It can be overwritten to perform any operation needed
        before running the pipeline, such as gathering data for node and edge sources
        """

        logger.info(
            "No prepare_for_pipeline() override specified. Continue to running preprocessing logic"
        )

    @abstractmethod
    def get_nodes_preprocessing_spec(
        self,
    ) -> Dict[NodeDataReference, NodeDataPreprocessingSpec]:
        raise NotImplementedError

    @abstractmethod
    def get_edges_preprocessing_spec(
        self,
    ) -> Dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
        raise NotImplementedError


def build_ingestion_feature_spec_fn(
    fixed_string_fields: Optional[List[str]] = None,
    fixed_string_field_shapes: Dict[str, List[int]] = {},
    fixed_float_fields: Optional[List[str]] = None,
    fixed_float_field_shapes: Dict[str, List[int]] = {},
    fixed_int_fields: Optional[List[str]] = None,
    fixed_int_field_shapes: Dict[str, List[int]] = {},
    varlen_string_fields: Optional[List[str]] = None,
    varlen_float_fields: Optional[List[str]] = None,
    varlen_int_fields: Optional[List[str]] = None,
) -> Callable[[], FeatureSpecDict]:
    """
    Returns a callable, which when called, generates the FeatureSpecDict which lets TFTransform know how to
    construe input data as tensors.

    :param fixed_string_fields: Fixed-length string features.
    :param fixed_string_field_shapes: Data shape lookup for fixed-length string features.
    :param fixed_float_fields: Fixed-length float features.
    :param fixed_float_field_shapes: Data shape lookup for fixed-length float features.
    :param fixed_int_fields: Fixed-length int features.
    :param fixed_int_field_shapes: Data shape lookup for fixed-length int features.
    :param varlen_string_fields: Variable-length string features.
    :param varlen_float_fields: Variable-length float features.
    :param varlen_int_fields: Variable-length int features.
    :return:
    """

    def get_ingestion_feature_spec() -> FeatureSpecDict:
        feature_spec_dict: FeatureSpecDict = dict()
        if fixed_string_fields:
            feature_spec_dict.update(
                {
                    col: tf.io.FixedLenFeature(
                        shape=fixed_string_field_shapes.get(col, []),
                        dtype=DEFAULT_TF_STRING_DTYPE,
                    )
                    for col in fixed_string_fields
                }
            )

        if fixed_float_fields:
            feature_spec_dict.update(
                {
                    col: tf.io.FixedLenFeature(
                        shape=fixed_float_field_shapes.get(col, []),
                        dtype=DEFAULT_TF_FLOAT_DTYPE,
                    )
                    for col in fixed_float_fields
                }
            )

        if fixed_int_fields:
            feature_spec_dict.update(
                {
                    col: tf.io.FixedLenFeature(
                        shape=fixed_int_field_shapes.get(col, []),
                        dtype=DEFAULT_TF_INT_DTYPE,
                    )
                    for col in fixed_int_fields
                }
            )

        if varlen_string_fields:
            feature_spec_dict.update(
                {
                    col: tf.io.VarLenFeature(dtype=DEFAULT_TF_STRING_DTYPE)
                    for col in varlen_string_fields
                }
            )

        if varlen_float_fields:
            feature_spec_dict.update(
                {
                    col: tf.io.VarLenFeature(dtype=DEFAULT_TF_FLOAT_DTYPE)
                    for col in varlen_float_fields
                }
            )

        if varlen_int_fields:
            feature_spec_dict.update(
                {
                    col: tf.io.VarLenFeature(dtype=DEFAULT_TF_INT_DTYPE)
                    for col in varlen_int_fields
                }
            )

        return feature_spec_dict

    return get_ingestion_feature_spec


def build_passthrough_transform_preprocessing_fn() -> (
    Callable[[TFTensorDict], TFTensorDict]
):
    """
    Produces a callable which acts as a pass-through preprocessing_fn for TFT to use.  In other words, it simply
    passes all keys available in the input onwards to the output.

    See https://www.tensorflow.org/tfx/tutorials/transform/census#create_a_tftransform_preprocessing_fn/ for details.
    :return:
    """

    def preprocessing_fn(inputs: TFTensorDict) -> TFTensorDict:
        return inputs

    return preprocessing_fn
