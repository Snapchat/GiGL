import tempfile

import tensorflow as tf
import torch
from absl.testing import absltest
from torch.testing import assert_close

from gigl.common import UriFactory
from gigl.common.data.loader_types import SerializedTFRecordInfo
from gigl.common.data.parquet_loader import ParquetDataLoader
from gigl.src.mocking.lib.mock_input_for_subgraph_sampler import (
    _write_edge_parquet,
    _write_node_parquet,
)
from tests.test_assets.test_case import TestCase


class MockedParquetDataTest(TestCase):
    def test_node_parquet_loads_back(self) -> None:
        node_features = torch.arange(15, dtype=torch.float64).reshape(5, 3) * 0.5
        node_labels = torch.tensor([0, 2, 1, 2, 0])
        with tempfile.TemporaryDirectory() as directory:
            uri = UriFactory.create_uri(directory)
            _write_node_parquet(
                uri_prefix=uri,
                node_features=node_features,
                node_labels=node_labels,
                label_column="label",
            )
            loaded = ParquetDataLoader(rank=0, world_size=1).load_as_torch_tensors(
                SerializedTFRecordInfo(
                    tfrecord_uri_prefix=None,
                    parquet_uri_prefix=uri,
                    feature_keys=["f0", "f1", "f2"],
                    feature_spec={
                        "node_id": tf.io.FixedLenFeature([], tf.int64),
                        "label": tf.io.FixedLenFeature([], tf.int64),
                    },
                    feature_dim=3,
                    entity_key="node_id",
                    label_keys=["label"],
                )
            )

        assert_close(loaded.ids, torch.arange(5))
        assert loaded.features is not None
        assert_close(loaded.features, node_features.to(torch.float32))
        assert loaded.labels is not None
        assert_close(loaded.labels.reshape(-1), node_labels.to(torch.float32))

    def test_edge_parquet_loads_back(self) -> None:
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        with tempfile.TemporaryDirectory() as directory:
            uri = UriFactory.create_uri(directory)
            _write_edge_parquet(
                uri_prefix=uri, edge_index=edge_index, edge_features=edge_features
            )
            loaded = ParquetDataLoader(rank=0, world_size=1).load_as_torch_tensors(
                SerializedTFRecordInfo(
                    tfrecord_uri_prefix=None,
                    parquet_uri_prefix=uri,
                    feature_keys=["f0", "f1"],
                    feature_spec={
                        "src": tf.io.FixedLenFeature([], tf.int64),
                        "dst": tf.io.FixedLenFeature([], tf.int64),
                    },
                    feature_dim=2,
                    entity_key=("src", "dst"),
                )
            )

        assert_close(loaded.ids, edge_index)
        assert loaded.features is not None
        assert_close(loaded.features, edge_features)

    def test_edge_parquet_without_features_loads_back(self) -> None:
        edge_index = torch.tensor([[0, 1, 2], [2, 0, 1]])
        with tempfile.TemporaryDirectory() as directory:
            uri = UriFactory.create_uri(directory)
            _write_edge_parquet(
                uri_prefix=uri, edge_index=edge_index, edge_features=None
            )
            loaded = ParquetDataLoader(rank=0, world_size=1).load_as_torch_tensors(
                SerializedTFRecordInfo(
                    tfrecord_uri_prefix=None,
                    parquet_uri_prefix=uri,
                    feature_keys=[],
                    feature_spec={
                        "src": tf.io.FixedLenFeature([], tf.int64),
                        "dst": tf.io.FixedLenFeature([], tf.int64),
                    },
                    feature_dim=0,
                    entity_key=("src", "dst"),
                )
            )

        assert_close(loaded.ids, edge_index)
        self.assertIsNone(loaded.features)


if __name__ == "__main__":
    absltest.main()
