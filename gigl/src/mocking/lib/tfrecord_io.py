from typing import Iterable, Optional, Sequence, TypeVar
from uuid import uuid4

import tensorflow as tf
from google.protobuf import message

from gigl.common import Uri
from gigl.common.logger import Logger

logger = Logger()

T = TypeVar("T")


def write_pb_tfrecord_shards_to_uri(
    pb_samples: Sequence[message.Message],
    uri_prefix: Uri,
    filename_prefix: str = "data",
    chunk_size=100,
    sample_type_for_logging: Optional[str] = "",
    raise_exception_if_no_pb_samples: bool = True,
):
    """
    Given a list of protobufs, chunk them and write them out to TFRecord files.
    """

    if raise_exception_if_no_pb_samples:
        assert len(
            pb_samples
        ), f"Found empty list of {sample_type_for_logging} samples to write to TFRecord files."

    def batch(list_of_items: Sequence[T], chunk_size: int) -> Iterable[Sequence[T]]:
        length_of_list = len(list_of_items)
        for idx in range(0, length_of_list, chunk_size):
            yield list_of_items[idx : min(idx + chunk_size, length_of_list)]

    uri_cls = type(uri_prefix)
    for pb_sample_batch in batch(list_of_items=pb_samples, chunk_size=chunk_size):
        with tf.io.TFRecordWriter(
            uri_cls.join(uri_prefix, f"{filename_prefix}-{str(uuid4())}.tfrecord").uri
        ) as writer:
            for sample in pb_sample_batch:
                writer.write(sample.SerializeToString())
    logger.info(
        f"Wrote {len(pb_samples)} {sample_type_for_logging} samples to {uri_prefix}"
    )
