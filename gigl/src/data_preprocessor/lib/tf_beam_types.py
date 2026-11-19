"""Data preprocessor types that need TensorFlow or Apache Beam at import time.

These live apart from `gigl.src.data_preprocessor.lib.types` so that importing the light-weight types does not pull in
TensorFlow and Apache Beam.
"""

from abc import ABC, abstractmethod
from typing import Any

import apache_beam as beam
import tensorflow as tf
from apache_beam import PCollection

from gigl.src.data_preprocessor.lib.types import InstanceDict

# Only these 3 dtypes are supported in TFTransform
# See https://www.tensorflow.org/tfx/guide/transform#understanding_the_inputs_to_the_preprocessing_fn
DEFAULT_TF_INT_DTYPE = tf.int64
DEFAULT_TF_FLOAT_DTYPE = tf.float32
DEFAULT_TF_STRING_DTYPE = tf.string


class InstanceDictPTransform(beam.PTransform, ABC):
    @abstractmethod
    def expand(self, input_or_inputs: PCollection[Any]) -> PCollection[InstanceDict]:
        raise NotImplementedError
