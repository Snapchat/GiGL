import tempfile
import unittest
from typing import List, cast

import tensorflow as tf

from gigl.common import LocalUri
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.model import GraphBackend
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    LoopyIterableDataset,
    TfRecordsIterableDataset,
)
from snapchat.research.gbml import graph_schema_pb2, training_samples_schema_pb2
from tests.test_assets.graph_metadata_constants import (
    DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
)


class TfRecordsIterableDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._num_records_per_file = 5
        self._num_files = 2
        self._files = self.__mock_tfrecord_files()

    def __mock_tfrecord_files(self):
        tfrecord_files: List[LocalUri] = list()
        for file_id in range(self._num_files):
            tfh = tempfile.NamedTemporaryFile(delete=False)
            with tf.io.TFRecordWriter(tfh.name) as writer:
                for record_id in range(self._num_records_per_file):
                    node_id = (file_id * self._num_records_per_file) + record_id
                    pb = training_samples_schema_pb2.SupervisedNodeClassificationSample(
                        root_node=graph_schema_pb2.Node(node_id=node_id)
                    )
                    writer.write(pb.SerializeToString())
            tfrecord_files.append(LocalUri(tfh.name))

        return tfrecord_files

    def test_can_load_data(self):
        pyg_builder = GraphBuilderFactory.get_graph_builder(
            backend_name=GraphBackend.PYG
        )
        proto_translator = GbmlProtosTranslator()

        def preprocess_raw_sample_fn(
            raw_data: bytes,
        ) -> PygGraphData:
            sample = training_samples_schema_pb2.SupervisedNodeClassificationSample()
            sample.ParseFromString(raw_data)
            neighborhood = (
                sample.neighborhood
            )  # TODO (svij-sc) : Refactor to use `preprocess_node_classification_raw_sample_fn` from future PRs
            pyg_data: PygGraphData = cast(
                PygGraphData,
                proto_translator.graph_data_from_GraphPb(
                    samples=[neighborhood],
                    graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
                    builder=pyg_builder,
                ),
            )
            return pyg_data

        dataset = TfRecordsIterableDataset(
            tf_record_uris=self._files, process_raw_sample_fn=preprocess_raw_sample_fn
        )
        samples = []
        for sample in dataset:
            self.assertTrue(isinstance(sample, PygGraphData))
            samples.append(sample)

        self.assertTrue(len(samples) == self._num_records_per_file * self._num_files)

    def test_loopy_iterable_dataset(self):
        dataset = TfRecordsIterableDataset(
            tf_record_uris=[self._files[0]], process_raw_sample_fn=lambda x: x
        )

        # Make sure we have a finite, nonzero number of records.
        num_records = len(list(dataset))
        self.assertGreater(num_records, 0)

        # Check that we can loop past this number of records and the looping is cyclic.
        loopy_dataset = LoopyIterableDataset(iterable_dataset=dataset)
        loopy_dataset_iter = iter(loopy_dataset)
        loopy_dataset_entries = [
            next(loopy_dataset_iter) for _ in range(num_records + 5)
        ]
        self.assertEquals(
            loopy_dataset_entries[0], loopy_dataset_entries[0 + num_records]
        )
