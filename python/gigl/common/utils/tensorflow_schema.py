from typing import Tuple

import absl
import tensorflow as tf
from tensorflow_data_validation import load_schema_text
from tensorflow_metadata.proto.v0.schema_pb2 import Schema
from tensorflow_transform.tf_metadata import schema_utils

from gigl.common import GcsUri, LocalUri, Uri
from gigl.src.data_preprocessor.lib.types import FeatureIndexDict, FeatureSpecDict

# We suppress noisy tensorflow logs to minimize unintentional clutter in logging:
# https://stackoverflow.com/questions/69485127/disabling-useless-logs-ouputs-from-tfx-setuptools
absl.logging.set_verbosity(absl.logging.FATAL)


def load_tf_schema_uri_str_to_feature_spec(uri: Uri) -> Tuple[Schema, FeatureSpecDict]:
    if not (GcsUri.is_valid(uri) or LocalUri.is_valid(uri)):
        raise ValueError(
            f"Invalid uri: {uri}. Uri has to either be a GCS or local uri string."
        )
    schema = load_schema_text(uri.uri)
    feature_spec = schema_utils.schema_as_feature_spec(schema).feature_spec
    return schema, feature_spec


def get_feature_len_from_fixed_len_feature(
    feature_config: tf.io.FixedLenFeature,
) -> int:
    feature_shape = feature_config.shape
    # shape=[] for length 1 feature, shape=[k] for k-length feature
    feature_len: int = feature_shape[0] if feature_shape else 1
    return feature_len


def feature_spec_to_feature_index_map(
    feature_spec: FeatureSpecDict,
) -> FeatureIndexDict:
    # from python 3.7 order in dict is guaranteed
    feature_to_index_map: FeatureIndexDict = {}
    index = 0
    for feature_name, feature_config in feature_spec.items():
        feature_len: int = get_feature_len_from_fixed_len_feature(
            feature_config=feature_config
        )
        start, end = index, index + feature_len
        feature_to_index_map[feature_name] = (start, end)
        index += feature_len

    return feature_to_index_map
