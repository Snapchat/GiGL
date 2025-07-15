import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Tuple, Union

import tensorflow as tf
import torch
from parameterized import param, parameterized
from torch.testing import assert_close

from gigl.common import UriFactory
from gigl.common.data.dataloaders import (
    SerializedTFRecordInfo,
    TFDatasetOptions,
    TFRecordDataLoader,
)
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict

_FEATURE_SPEC_WITH_ENTITY_KEY: FeatureSpecDict = {
    "node_id": tf.io.FixedLenFeature([], tf.int64),
    "feature_0": tf.io.FixedLenFeature([], tf.float32),
    "feature_1": tf.io.FixedLenFeature([], tf.float32),
}

_FEATURE_SPEC_WITHOUT_ENTITY_KEY: FeatureSpecDict = {
    "feature_0": tf.io.FixedLenFeature([], tf.float32),
    "feature_1": tf.io.FixedLenFeature([], tf.float32),
}


def _get_mock_node_examples() -> List[tf.train.Example]:
    """Generate mock examples for testing.

    These examples are, for now, hard-coded to match the feature spec defined in TFRecordDataLoaderTest.setUp().
    And are also hard-coded to have 100 examples.
    """
    examples: List[tf.train.Example] = []
    for i in range(100):
        examples.append(
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "node_id": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[i])
                        ),
                        "feature_0": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[i * 10])
                        ),
                        "feature_1": tf.train.Feature(
                            float_list=tf.train.FloatList(value=[i * 0.1])
                        ),
                    }
                )
            )
        )
    return examples


class TFRecordDataLoaderTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)

        examples = _get_mock_node_examples()
        with tf.io.TFRecordWriter(str(self.data_dir / "100.tfrecord")) as writer:
            for example in examples:
                writer.write(example.SerializeToString())

    def tearDown(self):
        super().tearDown()
        self.temp_dir.cleanup()

    @parameterized.expand(
        [
            param(
                "No features",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=[],
                feature_dim=0,
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=None,
            ),
            param(
                "One feature",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=["feature_0"],
                feature_dim=0,
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=torch.tensor(
                    range(100), dtype=torch.float32
                ).reshape(100, 1)
                * 10,
            ),
            param(
                "Two features",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=["feature_0", "feature_1"],
                feature_dim=0,
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=torch.concat(
                    (
                        torch.tensor(range(100), dtype=torch.float32).reshape(100, 1)
                        * 10,
                        torch.tensor(range(100), dtype=torch.float32).reshape(100, 1)
                        * 0.1,
                    ),
                    dim=1,
                ),
            ),
            param(
                "Two features, no entity key in feature schema",
                feature_spec=_FEATURE_SPEC_WITHOUT_ENTITY_KEY,
                feature_keys=["feature_0", "feature_1"],
                feature_dim=0,
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=torch.concat(
                    (
                        torch.tensor(range(100), dtype=torch.float32).reshape(100, 1)
                        * 10,
                        torch.tensor(range(100), dtype=torch.float32).reshape(100, 1)
                        * 0.1,
                    ),
                    dim=1,
                ),
            ),
        ]
    )
    def test_load_as_torch_tensors(
        self,
        _,
        feature_spec: FeatureSpecDict,
        feature_keys: List[str],
        feature_dim: int,
        expected_id_tensor: torch.Tensor,
        expected_feature_tensor: Optional[torch.Tensor],
    ):
        loader = TFRecordDataLoader(rank=0, world_size=1)
        node_ids, feature_tensor = loader.load_as_torch_tensors(
            serialized_tf_record_info=SerializedTFRecordInfo(
                tfrecord_uri_prefix=UriFactory.create_uri(self.data_dir),
                feature_spec=feature_spec,
                feature_keys=feature_keys,
                feature_dim=feature_dim,
                entity_key="node_id",
                tfrecord_uri_pattern="100.tfrecord",
            ),
            tf_dataset_options=TFDatasetOptions(deterministic=True),
        )

        assert_close(node_ids, expected_id_tensor)

        assert_close(feature_tensor, expected_feature_tensor)

    def test_build_dataset_for_uris(self):
        dataset = TFRecordDataLoader._build_dataset_for_uris(
            uris=[UriFactory.create_uri(self.data_dir / "100.tfrecord")],
            feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,  # Feature Spec is guaranteed to have entity key when this function is called
        ).unbatch()

        nodes = {r["node_id"].numpy() for r in dataset}

        self.assertEqual(nodes, set(range(100)))

    @parameterized.expand(
        [
            param(
                "just_node",
                feature_keys=[],
                feature_dim=0,
                expected_node_ids=torch.empty(0),
                expected_features=None,
                entity_key="node_id",
            ),
            param(
                "node_with_features",
                feature_keys=["foo_feature"],
                feature_dim=1,
                expected_node_ids=torch.empty(0),
                expected_features=torch.empty(0, 1),
                entity_key="node_id",
            ),
            param(
                "just_edge",
                feature_keys=[],
                feature_dim=0,
                expected_node_ids=torch.empty(2, 0),
                expected_features=None,
                entity_key=("src_node_id", "dst_node_id"),
            ),
            param(
                "edge_with_features",
                feature_keys=["foo_feature", "bar_feature"],
                feature_dim=3,
                expected_node_ids=torch.empty(2, 0),
                expected_features=torch.empty(0, 3),
                entity_key=("src_node_id", "dst_node_id"),
            ),
        ]
    )
    def test_load_empty_directory(
        self,
        _,
        feature_keys: List[str],
        feature_dim: int,
        expected_node_ids: torch.Tensor,
        expected_features: Optional[torch.Tensor],
        entity_key: Union[str, Tuple[str, str]],
    ):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        loader = TFRecordDataLoader(rank=0, world_size=1)
        node_ids, feature_ids = loader.load_as_torch_tensors(
            serialized_tf_record_info=SerializedTFRecordInfo(
                tfrecord_uri_prefix=UriFactory.create_uri(temp_dir.name),
                feature_spec={},  # Doesn't matter what this is.
                feature_keys=feature_keys,
                feature_dim=feature_dim,
                entity_key=entity_key,
                tfrecord_uri_pattern=".tfrecord",
            ),
            tf_dataset_options=TFDatasetOptions(deterministic=True),
        )

        assert_close(node_ids, expected_node_ids)
        assert_close(feature_ids, expected_features)

    @parameterized.expand(
        [
            param(
                "workers<files",
                num_workers=4,
                num_files=10,
                expected_partitions=[[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]],
            ),
            param(
                "workers>files",
                num_workers=4,
                num_files=2,
                expected_partitions=[[0], [1], [], []],
            ),
        ]
    )
    def test_partition(
        self, _, num_workers: int, num_files: int, expected_partitions: List[List[int]]
    ):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        path = Path(temp_dir.name)
        for i in range(num_files):
            with open(path / f"{i:0>2}.tfrecord", "w") as f:
                f.write("")

        for worker in range(num_workers):
            loader = TFRecordDataLoader(rank=worker, world_size=num_workers)
            uris = loader._partition_children_uris(
                UriFactory.create_uri(path), ".*tfrecord"
            )
            with self.subTest(f"worker: {worker}"):
                expected = expected_partitions[worker]
                self.assertEqual(
                    [u.uri for u in uris],
                    [str(path / f"{i:0>2}.tfrecord") for i in expected],
                )
