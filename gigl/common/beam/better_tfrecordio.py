"""Internal fork of WriteToTFRecord with improved TFRecord sink.
Specifically we add functionality to cap the max bytes per shard -
a feature supported by file based sinks but something not implemented
for tensorflow sinks. Also has support for specifying deferred
tft.tf_metadata.dataset_metadata.DatasetMetadata, so it can be used
in pipelines where DatasetMetadata is derived on runtime.
"""

from typing import Optional, Union

import apache_beam as beam
import apache_beam.pvalue
import apache_beam.transforms.window
import apache_beam.utils.windowed_value
import tensorflow_transform.tf_metadata.dataset_metadata as dataset_metadata
from apache_beam.io import filebasedsink
from apache_beam.io.filesystem import CompressionTypes
from apache_beam.io.iobase import Write
from apache_beam.io.tfrecordio import _TFRecordUtil
from apache_beam.transforms import PTransform

from gigl.common.beam.coders import PassthroughCoder, RecordBatchToTFExampleCoderFn
from gigl.common.logger import Logger

logger = Logger()


class _BetterTFRecordSink(filebasedsink.FileBasedSink):
    """Sink for writing TFRecords files.
    Utilizing changes introduced here: https://github.com/apache/beam/pull/22130

    For detailed TFRecord format description see:
      https://www.tensorflow.org/tutorials/load_data/tfrecord
    """

    def __init__(
        self,
        file_path_prefix,
        coder,
        file_name_suffix,
        num_shards,
        shard_name_template,
        compression_type,
        max_bytes_per_shard,
    ):
        """Initialize a TFRecordSink. See BetterWriteToTFRecord for details."""

        super().__init__(
            file_path_prefix=file_path_prefix,
            coder=coder,
            file_name_suffix=file_name_suffix,
            num_shards=num_shards,
            shard_name_template=shard_name_template,
            mime_type="application/octet-stream",
            compression_type=compression_type,
            max_bytes_per_shard=max_bytes_per_shard,
        )

    def write_encoded_record(self, file_handle, value):
        _TFRecordUtil.write_record(file_handle, value)


class BetterWriteToTFRecord(PTransform):
    """Transform for writing to TFRecord sinks."""

    def __init__(
        self,
        file_path_prefix: str,
        transformed_metadata: Optional[
            Union[dataset_metadata.DatasetMetadata, apache_beam.pvalue.AsSingleton]
        ] = None,
        file_name_suffix: Optional[str] = ".tfrecord",
        compression_type: Optional[str] = CompressionTypes.AUTO,
        max_bytes_per_shard: Optional[int] = int(2e8),  # 200mb
        num_shards: Optional[int] = 0,
    ):
        """
        Initialize BetterWriteToTFRecord transform.

        We improve the default WriteToTFRecord implementation by first simplifying needed params,
        adding functionality to cap the max bytes per shard. And, adding support for both easily
        serializing generic protobuff messages and tf.train.Example messages with capacity to
        specify deferred (computed at runtime) tft.tf_metadata.dataset_metadata.DatasetMetadata.

        Args:
            file_path_prefix (str): The file path to write to. The files written will begin
                with this prefix, followed by a shard identifier, and end in a common extension,
                if given by file_name_suffix.
            transformed_metadata (Optional[Union[dataset_metadata.DatasetMetadata, apache_beam.pvalue.AsSingleton]]):
                Useful for encoding tf.train.Example, when reading a TFTransform fn
                (dataset_metadata.DatasetMetadata) or when building it for the first time
                (apache_beam.pvalue.AsSingleton[dataset_metadata.DatasetMetadata]). Defaults
                to None, meaning a generic protobuf message is assumed which will be
                encoded using `SerializeToString()`.
            file_name_suffix (str, optional): Suffix for the files written. Defaults to ".tfrecord".
            compression_type (str, optional): Used to handle compressed output files. Typical value
                is CompressionTypes.AUTO, in which case the file_path's extension will
                be used to detect the compression.
            max_bytes_per_shard (int, optional): The data is sharded into separate files to promote
                faster/distributed writes. This parameter controls the max size of these shards.
                Defaults to int(2e8), or ~200 Mb.
            num_shards (int, optional): The number of files (shards) used for output. If not set,
                the service will decide on the optimal number of shards based off of max_bytes_per_shard.
                WARNING: Constraining the number of shards is likely to reduce the performance of a
                pipeline - only use if you know what you are doing.
        """
        super().__init__()
        self._transformed_metadata = transformed_metadata
        self._sink = _BetterTFRecordSink(
            file_path_prefix=file_path_prefix,
            coder=PassthroughCoder(),
            file_name_suffix=file_name_suffix,
            num_shards=num_shards,
            shard_name_template=None,
            compression_type=compression_type,
            max_bytes_per_shard=max_bytes_per_shard,
        )

    def expand(self, pcoll):
        if self._transformed_metadata:
            logger.info("Using transformed_metadata to encode samples.")
            pcoll = (
                pcoll
                | "Encode pyarrow.RecordBatch as serialized tf.train.Example"
                >> beam.ParDo(
                    RecordBatchToTFExampleCoderFn(),
                    transformed_metadata=self._transformed_metadata,
                )
            )
        else:
            logger.info("Using default proto serialization to encode samples.")
            pcoll = pcoll | "Serialize To String" >> beam.Map(
                lambda msg: msg.SerializeToString()
            )
        return pcoll | Write(self._sink)
