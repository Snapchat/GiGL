import tempfile
import unittest
from pathlib import Path
from typing import Optional, Tuple, Union

import tensorflow as tf
import torch
from parameterized import param, parameterized
from torch.testing import assert_close

from gigl.common import UriFactory
from gigl.common.data.dataloaders import (
    SerializedTFRecordInfo,
    TFDatasetOptions,
    TFRecordDataLoader,
    _get_labels_from_features,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    get_mocked_dataset_artifact_metadata,
)
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
)
from tests.test_assets.distributed.utils import assert_tensor_equality

_FEATURE_SPEC_WITH_ENTITY_KEY: FeatureSpecDict = {
    "node_id": tf.io.FixedLenFeature([], tf.int64),
    "feature_0": tf.io.FixedLenFeature([], tf.float32),
    "feature_1": tf.io.FixedLenFeature([], tf.float32),
    "label_0": tf.io.FixedLenFeature([], tf.int64),
    "label_1": tf.io.FixedLenFeature([], tf.int64),
}

_FEATURE_SPEC_WITHOUT_ENTITY_KEY: FeatureSpecDict = {
    "feature_0": tf.io.FixedLenFeature([], tf.float32),
    "feature_1": tf.io.FixedLenFeature([], tf.float32),
    "label_0": tf.io.FixedLenFeature([], tf.int64),
    "label_1": tf.io.FixedLenFeature([], tf.int64),
}


def _get_mock_node_examples() -> list[tf.train.Example]:
    """Generate mock examples for testing.

    These examples are, for now, hard-coded to match the feature spec defined in TFRecordDataLoaderTest.setUp().
    And are also hard-coded to have 100 examples.
    """
    examples: list[tf.train.Example] = []
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
                        "label_0": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[i % 2])
                        ),
                        "label_1": tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[i % 3])
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

        # Create standard examples without labels
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
                "No features, no labels",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=[],
                feature_dim=0,
                label_keys=[],
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=None,
                expected_label_tensor=None,
            ),
            param(
                "One feature, no labels",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=["feature_0"],
                feature_dim=1,
                label_keys=[],
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=torch.tensor(
                    range(100), dtype=torch.float32
                ).reshape(100, 1)
                * 10,
                expected_label_tensor=None,
            ),
            param(
                "Two features, no labels",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=["feature_0", "feature_1"],
                feature_dim=2,
                label_keys=[],
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
                expected_label_tensor=None,
            ),
            param(
                "Two features, no entity key in feature schema, no labels",
                feature_spec=_FEATURE_SPEC_WITHOUT_ENTITY_KEY,
                feature_keys=["feature_0", "feature_1"],
                feature_dim=2,
                label_keys=[],
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
                expected_label_tensor=None,
            ),
            param(
                "Two features with labels",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=["feature_0", "feature_1"],
                feature_dim=2,
                label_keys=["label_0"],
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=torch.concat(
                    (
                        torch.tensor(range(100), dtype=torch.float32).reshape(100, 1)
                        * 10,  # feature_0
                        torch.tensor(range(100), dtype=torch.float32).reshape(100, 1)
                        * 0.1,  # feature_1
                    ),
                    dim=1,
                ),
                expected_label_tensor=torch.tensor(
                    [i % 2 for i in range(100)], dtype=torch.float32
                ).reshape(100, 1),
            ),
            param(
                "One feature with labels",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=["feature_0"],
                feature_dim=1,
                label_keys=["label_0"],
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=torch.tensor(
                    range(100), dtype=torch.float32
                ).reshape(100, 1)
                * 10,  # feature_0
                expected_label_tensor=torch.tensor(
                    [i % 2 for i in range(100)], dtype=torch.float32
                ).reshape(
                    100, 1
                ),  # label_0
            ),
            param(
                "Only labels, no features",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=[],
                feature_dim=0,
                label_keys=["label_0"],
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=None,
                expected_label_tensor=torch.tensor(
                    [i % 2 for i in range(100)], dtype=torch.float32
                ).reshape(100, 1),
            ),
            param(
                "Two features with two labels",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=["feature_0", "feature_1"],
                feature_dim=2,
                label_keys=["label_0", "label_1"],
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=torch.concat(
                    (
                        torch.tensor(range(100), dtype=torch.float32).reshape(100, 1)
                        * 10,  # feature_0
                        torch.tensor(range(100), dtype=torch.float32).reshape(100, 1)
                        * 0.1,  # feature_1
                    ),
                    dim=1,
                ),
                expected_label_tensor=torch.concat(
                    (
                        torch.tensor(
                            [i % 2 for i in range(100)], dtype=torch.float32
                        ).reshape(
                            100, 1
                        ),  # label_0
                        torch.tensor(
                            [i % 3 for i in range(100)], dtype=torch.float32
                        ).reshape(
                            100, 1
                        ),  # label_1
                    ),
                    dim=1,
                ),
            ),
            param(
                "One feature with two labels",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=["feature_0"],
                feature_dim=1,
                label_keys=["label_0", "label_1"],
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=torch.tensor(
                    range(100), dtype=torch.float32
                ).reshape(100, 1)
                * 10,  # feature_0
                expected_label_tensor=torch.concat(
                    (
                        torch.tensor(
                            [i % 2 for i in range(100)], dtype=torch.float32
                        ).reshape(
                            100, 1
                        ),  # label_0
                        torch.tensor(
                            [i % 3 for i in range(100)], dtype=torch.float32
                        ).reshape(
                            100, 1
                        ),  # label_1
                    ),
                    dim=1,
                ),
            ),
            param(
                "Only two labels, no features",
                feature_spec=_FEATURE_SPEC_WITH_ENTITY_KEY,
                feature_keys=[],
                feature_dim=0,
                label_keys=["label_0", "label_1"],
                expected_id_tensor=torch.tensor(range(100)),
                expected_feature_tensor=None,
                expected_label_tensor=torch.concat(
                    (
                        torch.tensor(
                            [i % 2 for i in range(100)], dtype=torch.float32
                        ).reshape(
                            100, 1
                        ),  # label_0
                        torch.tensor(
                            [i % 3 for i in range(100)], dtype=torch.float32
                        ).reshape(
                            100, 1
                        ),  # label_1
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
        feature_keys: list[str],
        feature_dim: int,
        label_keys: list[str],
        expected_id_tensor: torch.Tensor,
        expected_feature_tensor: Optional[torch.Tensor],
        expected_label_tensor: Optional[torch.Tensor],
    ):
        """Test TFRecordDataLoader's ability to load features and optionally labels."""
        loader = TFRecordDataLoader(rank=0, world_size=1)
        node_ids, feature_tensor, label_tensor = loader.load_as_torch_tensors(
            serialized_tf_record_info=SerializedTFRecordInfo(
                tfrecord_uri_prefix=UriFactory.create_uri(self.data_dir),
                feature_spec=feature_spec,
                feature_keys=feature_keys,
                feature_dim=feature_dim,
                entity_key="node_id",
                label_keys=label_keys,
                tfrecord_uri_pattern="100.tfrecord",
            ),
            tf_dataset_options=TFDatasetOptions(deterministic=True),
        )

        # Verify entity IDs are loaded correctly
        assert_close(node_ids, expected_id_tensor)

        assert_close(feature_tensor, expected_feature_tensor)

        assert_close(label_tensor, expected_label_tensor)

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
                expected_label_tensor=None,
                entity_key="node_id",
            ),
            param(
                "node_with_features",
                feature_keys=["foo_feature"],
                feature_dim=1,
                expected_node_ids=torch.empty(0),
                expected_features=torch.empty(0, 1),
                expected_label_tensor=None,
                entity_key="node_id",
            ),
            param(
                "just_edge",
                feature_keys=[],
                feature_dim=0,
                expected_node_ids=torch.empty(2, 0),
                expected_features=None,
                expected_label_tensor=None,
                entity_key=("src_node_id", "dst_node_id"),
            ),
            param(
                "edge_with_features",
                feature_keys=["foo_feature", "bar_feature"],
                feature_dim=3,
                expected_node_ids=torch.empty(2, 0),
                expected_features=torch.empty(0, 3),
                expected_label_tensor=None,
                entity_key=("src_node_id", "dst_node_id"),
            ),
            param(
                "node_with_label_only",
                feature_keys=[],
                feature_dim=0,
                expected_node_ids=torch.empty(0),
                expected_features=None,
                expected_label_tensor=torch.empty(0, 1),  # 1 label
                entity_key="node_id",
                label_keys=["label"],
            ),
            param(
                "node_with_features_and_label",
                feature_keys=["foo_feature"],
                feature_dim=1,
                expected_node_ids=torch.empty(0),
                expected_features=torch.empty(0, 1),  # 1 feature
                expected_label_tensor=torch.empty(0, 1),  # 1 label
                entity_key="node_id",
                label_keys=["label"],
            ),
        ]
    )
    def test_load_empty_directory(
        self,
        _,
        feature_keys: list[str],
        feature_dim: int,
        expected_node_ids: torch.Tensor,
        expected_features: Optional[torch.Tensor],
        expected_label_tensor: Optional[torch.Tensor],
        entity_key: Union[str, Tuple[str, str]],
        label_keys: list[str] = [],
    ):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)

        loader = TFRecordDataLoader(rank=0, world_size=1)
        node_ids, feature_tensor, label_tensor = loader.load_as_torch_tensors(
            serialized_tf_record_info=SerializedTFRecordInfo(
                tfrecord_uri_prefix=UriFactory.create_uri(temp_dir.name),
                feature_spec={},  # Doesn't matter what this is.
                feature_keys=feature_keys,
                feature_dim=feature_dim,
                entity_key=entity_key,
                label_keys=label_keys,
            ),
            tf_dataset_options=TFDatasetOptions(deterministic=True),
        )

        assert_close(node_ids, expected_node_ids)
        assert_close(feature_tensor, expected_features)
        assert_close(label_tensor, expected_label_tensor)

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
        self, _, num_workers: int, num_files: int, expected_partitions: list[list[int]]
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

    def test_load_labels_from_pb(self):
        mocked_dataset_artifact_metadata: MockedDatasetArtifactMetadata = (
            get_mocked_dataset_artifact_metadata()[
                CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO.name
            ]
        )
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=mocked_dataset_artifact_metadata.frozen_gbml_config_uri
            )
        )
        preprocessed_metadata_pb_wrapper = (
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
        )
        condensed_node_type = (
            gbml_config_pb_wrapper.graph_metadata_pb_wrapper.homogeneous_condensed_node_type
        )
        node_metadata = preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
            condensed_node_type
        ]
        loader = TFRecordDataLoader(rank=0, world_size=1)
        _, feature_tensor, label_tensor = loader.load_as_torch_tensors(
            serialized_tf_record_info=SerializedTFRecordInfo(
                tfrecord_uri_prefix=UriFactory.create_uri(
                    node_metadata.tfrecord_uri_prefix
                ),
                feature_spec=preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_schema_map[
                    condensed_node_type
                ].feature_spec,
                feature_keys=node_metadata.feature_keys,
                feature_dim=node_metadata.feature_dim,
                entity_key=node_metadata.node_id_key,
                label_keys=node_metadata.label_keys,
                tfrecord_uri_pattern=".*\.tfrecord$",
            ),
            tf_dataset_options=TFDatasetOptions(deterministic=True),
        )
        # Ensure we have loaded data
        assert feature_tensor is not None and label_tensor is not None
        self.assertEqual(feature_tensor.size(1), node_metadata.feature_dim)
        self.assertEqual(label_tensor.size(1), len(node_metadata.label_keys))

    @parameterized.expand(
        [
            param(
                "Basic test with label_dim=1",
                feature_and_label_tensor=torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                    ]
                ),
                label_dim=1,
                expected_features=torch.tensor(
                    [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0], [9.0, 10.0, 11.0]]
                ),
                expected_labels=torch.tensor([[4.0], [8.0], [12.0]]),
            ),
            param(
                "Test with label_dim=2",
                feature_and_label_tensor=torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
                ),
                label_dim=2,
                expected_features=torch.tensor([[1.0, 2.0, 3.0], [6.0, 7.0, 8.0]]),
                expected_labels=torch.tensor([[4.0, 5.0], [9.0, 10.0]]),
            ),
            param(
                "Test with single feature column",
                feature_and_label_tensor=torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                ),
                label_dim=1,
                expected_features=torch.tensor([[1.0], [3.0], [5.0]]),
                expected_labels=torch.tensor([[2.0], [4.0], [6.0]]),
            ),
            param(
                "Test with no features and labels",
                feature_and_label_tensor=torch.tensor([[3.0], [6.0]]),
                label_dim=1,
                expected_features=None,
                expected_labels=torch.tensor([[3.0], [6.0]]),
            ),
            param(
                "Test with features and no labels",
                feature_and_label_tensor=torch.tensor([[3.0], [6.0]]),
                label_dim=0,
                expected_features=torch.tensor([[3.0], [6.0]]),
                expected_labels=None,
            ),
        ]
    )
    def test_get_labels_from_features(
        self,
        _,
        feature_and_label_tensor: torch.Tensor,
        label_dim: int,
        expected_features: Optional[torch.Tensor],
        expected_labels: Optional[torch.Tensor],
    ):
        features, labels = _get_labels_from_features(
            feature_and_label_tensor, label_dim
        )
        if expected_features is None:
            self.assertIsNone(features)
        else:
            assert (
                features is not None
            ), "Expected features was None, but expected a tensor"
            assert_tensor_equality(features, expected_features)
        if expected_labels is None:
            self.assertIsNone(labels)
        else:
            assert labels is not None, "Expected labels was None, but expected a tensor"
            assert_tensor_equality(labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
