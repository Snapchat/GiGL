import tempfile
import unittest
from collections import defaultdict
from typing import Dict, List

import tensorflow as tf

from gigl.common import LocalUri
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.graph_data import Node, NodeType
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    CombinedIterableDatasets,
    LoopyIterableDataset,
    TfRecordsIterableDataset,
)
from snapchat.research.gbml import graph_schema_pb2, training_samples_schema_pb2
from tests.test_assets.graph_metadata_constants import (
    EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
)


class CombinedIterableDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._condensed_node_types = list(
            EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER.condensed_node_types
        )
        self._node_types = EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER.node_types
        self._num_records_per_file = 5
        self._num_files_per_node_type = 2
        self._files = self.__mock_tfrecord_files()

    def __mock_tfrecord_files(self):
        tfrecord_files: Dict[NodeType, List[LocalUri]] = defaultdict(list)
        for node_type_idx, condensed_node_type in enumerate(self._condensed_node_types):
            node_type = EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER.condensed_node_type_to_node_type_map[
                condensed_node_type
            ]
            for file_id in range(self._num_files_per_node_type):
                tfh = tempfile.NamedTemporaryFile(delete=False)
                with tf.io.TFRecordWriter(tfh.name) as writer:
                    for record_id in range(self._num_records_per_file):
                        node_id = (
                            (
                                node_type_idx
                                * self._num_files_per_node_type
                                * self._num_records_per_file
                            )
                            + (file_id * self._num_records_per_file)
                            + record_id
                        )
                        pb = training_samples_schema_pb2.RootedNodeNeighborhood(
                            root_node=graph_schema_pb2.Node(
                                node_id=node_id, condensed_node_type=condensed_node_type
                            )
                        )
                        writer.write(pb.SerializeToString())
                tfrecord_files[node_type].append(LocalUri(tfh.name))

        return tfrecord_files

    def test_can_load_loopy_data(self):
        def preprocess_raw_sample_fn(
            raw_data: bytes,
        ) -> Node:
            sample = training_samples_schema_pb2.RootedNodeNeighborhood()
            sample.ParseFromString(raw_data)
            samples, _ = GbmlProtosTranslator.node_from_NodePb(
                node_pb=sample.root_node,
                graph_metadata_pb_wrapper=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            )
            return samples

        loopy_datasets_map: Dict[NodeType, LoopyIterableDataset] = {}
        for condensed_node_type_str in self._files:
            tf_dataset = TfRecordsIterableDataset(
                tf_record_uris=self._files[condensed_node_type_str],
                process_raw_sample_fn=preprocess_raw_sample_fn,
            )
            loopy_dataset = LoopyIterableDataset(iterable_dataset=tf_dataset)
            loopy_datasets_map[condensed_node_type_str] = loopy_dataset

        dataset = CombinedIterableDatasets(iterable_dataset_map=loopy_datasets_map)  # type: ignore
        dataset_iter = iter(dataset)
        for _ in range(15):
            dataset_sample = next(dataset_iter)
            self.assertEqual(self._node_types, list(dataset_sample.keys()))
            self.assertEqual(
                self._node_types, [node.type for node in list(dataset_sample.values())]
            )

    def test_can_load_non_loopy_data(self):
        def preprocess_raw_sample_fn(
            raw_data: bytes,
        ) -> Node:
            sample = training_samples_schema_pb2.RootedNodeNeighborhood()
            sample.ParseFromString(raw_data)
            samples, _ = GbmlProtosTranslator.node_from_NodePb(
                node_pb=sample.root_node,
                graph_metadata_pb_wrapper=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            )
            return samples

        datasets_map: Dict[NodeType, TfRecordsIterableDataset] = {}
        for condensed_node_type_str in self._files:
            tf_dataset = TfRecordsIterableDataset(
                tf_record_uris=self._files[condensed_node_type_str],
                process_raw_sample_fn=preprocess_raw_sample_fn,
            )
            datasets_map[condensed_node_type_str] = tf_dataset

        dataset = CombinedIterableDatasets(iterable_dataset_map=datasets_map)  # type: ignore
        dataset_iter = iter(dataset)
        for _ in range(10):
            dataset_sample = next(dataset_iter)
            self.assertEqual(self._node_types, list(dataset_sample.keys()))
            self.assertEqual(
                self._node_types, [node.type for node in list(dataset_sample.values())]
            )
        with self.assertRaises(RuntimeError):
            dataset_sample = next(dataset_iter)
