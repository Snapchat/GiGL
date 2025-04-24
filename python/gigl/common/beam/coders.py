from typing import Any, Dict, Iterable

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
import tensorflow_transform.tf_metadata.dataset_metadata
from apache_beam import coders
from tensorflow_transform import common_types
from tfx_bsl.public import tfxio


class PassthroughCoder(coders.Coder):
    """Used as a dummy coder to just pass through the value without any special processing"""

    def is_deterministic(self) -> bool:
        return True

    def encode(self, value: Any) -> bytes:
        return value

    def decode(self, encoded):
        return encoded


class RuntimeTFExampleProtoCoderFn(beam.DoFn):
    """Can be used on runtime to encode msgs to tf.Example proto msgs"""

    def __init__(self):
        self._coder = None

    def process(
        self,
        element: Dict[str, common_types.TensorType],
        transformed_metadata: tensorflow_transform.tf_metadata.dataset_metadata.DatasetMetadata,
        *args,
        **kwargs,
    ) -> Iterable[tf.train.Example]:
        """Note that transformed_metadata actually needs to be passed in as part of process rather
        than class init. This is because the transformed_metadata that gets passed in is a side input
        which only materializes as the true transformed-metadata when passed in as part of process.

        Args:
            sample (Dict[str, common_types.TensorType]): TfExample Instance Dict
            transformed_metadata (tensorflow_transform.tf_metadata.dataset_metadata.DatasetMetadata):
                Used to generate the ExampleProtoCoder

        Yields:
            tf.Example: Encoded tf.Example
        """
        if not self._coder:
            self._coder = tensorflow_transform.coders.ExampleProtoCoder(
                transformed_metadata.schema
            )
        yield self._coder.encode(element)


class RecordBatchToTFExampleCoderFn(beam.DoFn):
    """Encode pyarrow.RecordBatch to serialized tf.train.Example(s)"""

    def __init__(self):
        self._coder = None

    def process(
        self,
        element: pa.RecordBatch,
        transformed_metadata: tensorflow_transform.tf_metadata.dataset_metadata.DatasetMetadata,
        *args,
        **kwargs,
    ) -> Iterable[bytes]:
        """Note that transformed_metadata needs to be passed in as side input, i.e., as an argument
        of process function, instead of being passed to class init, since it could potentially materialize
        (depending on whether it is read from file or built by tft_beam.AnalyzeDataset) after the
        class is constructed.

        Args:
            element (pa.RecordBatch): A batch of records, e.g., a batch of transformed features
            transformed_metadata (tensorflow_transform.tf_metadata.dataset_metadata.DatasetMetadata):
                containing the schema needed by RecordBatchToExamplesEncoder for encoding

        Yields:
            bytes: serialized tf.Example
        """
        if not self._coder:
            self._coder = tfxio.RecordBatchToExamplesEncoder(
                schema=transformed_metadata.schema
            )
        encoded_examples = self._coder.encode(element)
        for example in encoded_examples:
            yield example
