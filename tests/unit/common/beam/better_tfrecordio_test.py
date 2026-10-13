import glob
import os
import tempfile

import apache_beam as beam
import tensorflow as tf
from absl.testing import absltest
from apache_beam.options.pipeline_options import PipelineOptions

from gigl.common.beam.better_tfrecordio import BetterWriteToTFRecord
from tests.test_assets.test_case import TestCase

_NUM_EXAMPLES = 200


def _make_example(example_id: int) -> tf.train.Example:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "id": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[example_id])
                ),
                "embedding": tf.train.Feature(
                    float_list=tf.train.FloatList(value=[example_id * 0.5, -1.0, 3.25])
                ),
                "name": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[f"node_{example_id}".encode()])
                ),
            }
        )
    )


class BetterWriteToTFRecordTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        output_dir = tempfile.TemporaryDirectory()
        self.addCleanup(output_dir.cleanup)
        self._output_dir = output_dir.name
        self._examples = [_make_example(i) for i in range(_NUM_EXAMPLES)]

    def _write(self, max_bytes_per_shard: int) -> list[str]:
        with beam.Pipeline(options=PipelineOptions(runner="DirectRunner")) as p:
            _ = (
                p
                | beam.Create(self._examples)
                | BetterWriteToTFRecord(
                    file_path_prefix=os.path.join(self._output_dir, "examples"),
                    max_bytes_per_shard=max_bytes_per_shard,
                )
            )
        return sorted(glob.glob(os.path.join(self._output_dir, "examples*.tfrecord")))

    def _read_examples(self, shard_paths: list[str]) -> list[tf.train.Example]:
        # TFRecordDataset verifies the length and data CRCs of every record, so
        # a framing error in the sink fails the read instead of yielding bytes.
        return [
            tf.train.Example.FromString(record.numpy())
            for record in tf.data.TFRecordDataset(shard_paths)
        ]

    def _assert_same_examples(self, actual: list[tf.train.Example]) -> None:
        def by_id(example: tf.train.Example) -> int:
            return example.features.feature["id"].int64_list.value[0]

        self.assertEqual(sorted(actual, key=by_id), self._examples)

    def test_round_trips_examples(self) -> None:
        shard_paths = self._write(max_bytes_per_shard=int(2e8))

        self.assertNotEmpty(shard_paths)
        self._assert_same_examples(self._read_examples(shard_paths))

    def test_max_bytes_per_shard_splits_output(self) -> None:
        max_bytes_per_shard = 1024
        shard_paths = self._write(max_bytes_per_shard=max_bytes_per_shard)

        # The 200 records serialize to ~13 KB, so a 1 KiB cap cannot hold them in one shard.
        self.assertGreater(len(shard_paths), 1)
        # A shard closes once it reaches the cap, so it overshoots by at most
        # the one record that crossed it.
        largest_record_bytes = max(
            len(example.SerializeToString()) for example in self._examples
        )
        framing_bytes = 16  # 8-byte length + 4-byte length CRC + 4-byte data CRC
        for shard_path in shard_paths:
            self.assertLessEqual(
                os.path.getsize(shard_path),
                max_bytes_per_shard + largest_record_bytes + framing_bytes,
            )
        self._assert_same_examples(self._read_examples(shard_paths))


if __name__ == "__main__":
    absltest.main()
