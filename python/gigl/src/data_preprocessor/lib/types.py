from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import apache_beam as beam
import tensorflow as tf
from apache_beam import PCollection
from tensorflow_metadata.proto.v0.schema_pb2 import Feature
from tensorflow_transform import common_types

from gigl.common import Uri

# TODO (mkolodner-sc): Move these variables to a more general location, as they are used even outside of context of data preprocessor

InstanceDict = Dict[str, Any]
TFTensorDict = Dict[str, common_types.TensorType]
FeatureSpecDict = Dict[str, common_types.FeatureSpecType]
FeatureIndexDict = Dict[str, Tuple[int, int]]  # feature_name -> (start, end) index
FeatureSchemaDict = Dict[str, Feature]
FeatureVocabDict = Dict[str, List[str]]


# Only these 3 dtypes are supported in TFTransform
# See https://www.tensorflow.org/tfx/guide/transform#understanding_the_inputs_to_the_preprocessing_fn
DEFAULT_TF_INT_DTYPE = tf.int64
DEFAULT_TF_FLOAT_DTYPE = tf.float32
DEFAULT_TF_STRING_DTYPE = tf.string


class FeatureSchema(NamedTuple):
    """
    FeatureSchema stores the following
    1. tf schema: a dict of feature name -> Feature (tensorflow_metadata.proto.v0.schema_pb2.Feature)
    2. feature_spec: a dict of feature name -> FeatureSpec (eg. FixedLenFeature, VarlenFeature, SparseFeature, RaggedFeature)
    3. feature_index: a dict of feature name -> (start, end) index
    4. feature_vocab: a dict of feature name -> list of vocab values
    """

    schema: FeatureSchemaDict
    feature_spec: FeatureSpecDict
    feature_index: FeatureIndexDict
    feature_vocab: FeatureVocabDict


class NodeOutputIdentifier(str):
    """
    References the TFTransform output field / column name for a node identifier.
    e.g. for Cora, this would be "paper_id".
    """


class EdgeOutputIdentifier(NamedTuple):
    """
    References the TFTransform output fields / column names for src and dst node ids of an edge.
    e.g. for Cora, this would be "from_paper_id" and "to_paper_id".
    """

    src_node: NodeOutputIdentifier
    dst_node: NodeOutputIdentifier


class NodeDataPreprocessingSpec(NamedTuple):
    """
    `feature_spec_fn` should reflect that the `identifier_output` field is designated as tf.int64.
    `preprocessing_fn` should not manipulate the type of `identifier_output`.

    These caveats are to support enumeration of node / edge data.
    """

    feature_spec_fn: Callable[[], FeatureSpecDict]
    preprocessing_fn: Callable[[TFTensorDict], TFTensorDict]
    identifier_output: NodeOutputIdentifier
    pretrained_tft_model_uri: Optional[Uri] = None
    features_outputs: Optional[List[str]] = None
    labels_outputs: Optional[List[str]] = None

    def __repr__(self) -> str:
        return f"""NodeDataPreprocessingSpec(
            identifier_output={self.identifier_output},
            feature_spec={self.feature_spec_fn()},
            preprocessing_fn={self.preprocessing_fn},
            pretrained_tft_model_uri={self.pretrained_tft_model_uri},
            features_outputs={self.features_outputs},
            labels_outputs={self.labels_outputs})
        """


class EdgeDataPreprocessingSpec(NamedTuple):
    """
    `feature_spec_fn` should reflect that the `identifier_output` fields are designated as tf.int64.
    `preprocessing_fn` should not manipulate the types of fields in `identifier_output`.
    """

    feature_spec_fn: Callable[[], FeatureSpecDict]
    preprocessing_fn: Callable[[TFTensorDict], TFTensorDict]
    identifier_output: EdgeOutputIdentifier
    pretrained_tft_model_uri: Optional[Uri] = None
    features_outputs: Optional[List[str]] = None
    labels_outputs: Optional[List[str]] = None

    def __repr__(self) -> str:
        return f"""EdgeDataPreprocessingSpec(
            identifier_output={self.identifier_output},
            feature_spec={self.feature_spec_fn()},
            preprocessing_fn={self.preprocessing_fn},
            pretrained_tft_model_uri={self.pretrained_tft_model_uri},
            features_outputs={self.features_outputs},
            labels_outputs={self.labels_outputs})
        """


class InstanceDictPTransform(beam.PTransform, ABC):
    @abstractmethod
    def expand(self, input_or_inputs: PCollection[Any]) -> PCollection[InstanceDict]:
        raise NotImplementedError
