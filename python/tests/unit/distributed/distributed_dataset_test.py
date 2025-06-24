import unittest
from collections import abc
from typing import Any, MutableMapping, Optional, Type, Union

import graphlearn_torch as glt
import torch
from parameterized import param, parameterized
from torch.multiprocessing import Manager
from torch.testing import assert_close

from gigl.common.data.load_torch_tensors import (
    SerializedGraphMetadata,
    SerializedTFRecordInfo,
)
from gigl.distributed import (
    DistLinkPredictionDataset,
    DistPartitioner,
    DistRangePartitioner,
    DistributedContext,
    build_dataset,
)
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    get_mocked_dataset_artifact_metadata,
)
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    TOY_GRAPH_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import DEFAULT_HOMOGENEOUS_EDGE_TYPE
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)


class _FakeSplitter:
    def __init__(
        self,
        splits: Union[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            dict[EdgeType, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        ],
    ):
        self.splits = splits
        self._supervision_edge_types = [DEFAULT_HOMOGENEOUS_EDGE_TYPE]

    def __call__(self, edge_index):
        return self.splits

    @property
    def should_convert_labels_to_edges(self):
        return False


_USER = NodeType("user")
_STORY = NodeType("story")


class DistributedDatasetTestCase(unittest.TestCase):
    def setUp(self):
        self._master_ip_address = "localhost"
        self._world_size = 1
        self._num_rpc_threads = 4

    def assert_tensor_equal(
        self,
        actual: Optional[Union[torch.Tensor, abc.Mapping[Any, torch.Tensor]]],
        expected: Optional[Union[torch.Tensor, abc.Mapping[Any, torch.Tensor]]],
    ):
        if type(actual) != type(expected):
            self.fail(f"Expected type {type(expected)} but got {type(actual)}")
        if isinstance(actual, dict) and isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual.keys():
                assert_close(actual[key], expected[key], atol=0, rtol=0)
        elif isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
            assert_close(actual, expected, atol=0, rtol=0)

    @parameterized.expand(
        [
            param(
                "Test Building Dataset for tensor-based partitioning",
                partitioner_class=DistPartitioner,
            ),
            param(
                "Test Building Dataset for range-based partitioning",
                partitioner_class=DistRangePartitioner,
            ),
        ]
    )
    def test_build_dataset(self, _, partitioner_class: Type[DistPartitioner]):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()

        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            output_dict=output_dict,
            should_load_tensors_in_parallel=True,
            master_ip_address=self._master_ip_address,
            master_port=master_port,
            partitioner_class=partitioner_class,
        )

        self.assertIsNone(dataset.train_node_ids)
        self.assertIsNone(dataset.val_node_ids)
        self.assertIsNone(dataset.test_node_ids)
        self.assertIsInstance(dataset.node_ids, torch.Tensor)

    def test_build_and_split_dataset_homogeneous(self):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()
        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            output_dict=output_dict,
            should_load_tensors_in_parallel=True,
            master_ip_address=self._master_ip_address,
            master_port=master_port,
            splitter=_FakeSplitter(
                (
                    torch.tensor([1000]),
                    torch.tensor([2000, 3000]),
                    torch.tensor([3000, 4000, 5000]),
                ),
            ),
        )

        self.assert_tensor_equal(dataset.train_node_ids, torch.tensor([1000]))
        self.assert_tensor_equal(dataset.val_node_ids, torch.tensor([2000, 3000]))
        self.assert_tensor_equal(
            dataset.test_node_ids, torch.tensor([3000, 4000, 5000])
        )
        # Check that the node ids have *all* node ids, including nodes not included in train, val, and test.
        self.assert_tensor_equal(
            dataset.node_ids,
            torch.tensor(
                [
                    1000,
                    2000,
                    3000,
                    3000,
                    4000,
                    5000,
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                ]
            ),
        )

    @parameterized.expand(
        [
            param(
                "One supervision edge type",
                splits={
                    _USER: (
                        torch.tensor([1000]),
                        torch.tensor([2000]),
                        torch.tensor([3000]),
                    )
                },
                expected_train_node_ids={_USER: torch.tensor([1000])},
                expected_val_node_ids={_USER: torch.tensor([2000])},
                expected_test_node_ids={_USER: torch.tensor([3000])},
                expected_node_ids={
                    _USER: torch.tensor(
                        [
                            1000,
                            2000,
                            3000,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                        ]
                    ),
                    _STORY: torch.tensor(
                        [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                        ]
                    ),
                },
            ),
            param(
                "One supervision edge type - different numbers of train-test-val",
                splits={
                    _USER: (
                        torch.tensor([1000]),
                        torch.tensor([2000, 3000]),
                        torch.tensor([3000, 4000, 5000]),
                    )
                },
                expected_train_node_ids={_USER: torch.tensor([1000])},
                expected_val_node_ids={_USER: torch.tensor([2000, 3000])},
                expected_test_node_ids={_USER: torch.tensor([3000, 4000, 5000])},
                expected_node_ids={
                    _USER: torch.tensor(
                        [
                            1000,
                            2000,
                            3000,
                            3000,
                            4000,
                            5000,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                        ]
                    ),
                    _STORY: torch.tensor(
                        [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                        ]
                    ),
                },
            ),
            param(
                "Two supervision edge types - two target node types",
                splits={
                    _USER: (
                        torch.tensor([1000]),
                        torch.tensor([2000]),
                        torch.tensor([3000]),
                    ),
                    _STORY: (
                        torch.tensor([4000]),
                        torch.tensor([5000]),
                        torch.tensor([6000]),
                    ),
                },
                expected_train_node_ids={
                    _USER: torch.tensor([1000]),
                    _STORY: torch.tensor([4000]),
                },
                expected_val_node_ids={
                    _USER: torch.tensor([2000]),
                    _STORY: torch.tensor([5000]),
                },
                expected_test_node_ids={
                    _USER: torch.tensor([3000]),
                    _STORY: torch.tensor([6000]),
                },
                expected_node_ids={
                    _USER: torch.tensor(
                        [
                            1000,
                            2000,
                            3000,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                        ]
                    ),
                    _STORY: torch.tensor(
                        [
                            4000,
                            5000,
                            6000,
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            15,
                            16,
                            17,
                            18,
                        ]
                    ),
                },
            ),
        ]
    )
    def test_build_and_split_dataset_heterogeneous(
        self,
        _,
        splits,
        expected_train_node_ids,
        expected_val_node_ids,
        expected_test_node_ids,
        expected_node_ids,
    ):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        manager = Manager()
        output_dict: MutableMapping[int, DistLinkPredictionDataset] = manager.dict()
        dataset = run_distributed_dataset(
            rank=0,
            world_size=self._world_size,
            mocked_dataset_info=HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            output_dict=output_dict,
            should_load_tensors_in_parallel=True,
            master_ip_address=self._master_ip_address,
            master_port=master_port,
            splitter=_FakeSplitter(splits),
        )

        self.assert_tensor_equal(dataset.train_node_ids, expected_train_node_ids)
        self.assert_tensor_equal(dataset.val_node_ids, expected_val_node_ids)
        self.assert_tensor_equal(dataset.test_node_ids, expected_test_node_ids)
        # Check that the node ids have *all* node ids, including nodes not included in train, val, and test.
        self.assert_tensor_equal(dataset.node_ids, expected_node_ids)

    # This tests that if we build a dataset with a supervision edge type which is not a message passing edge type, we still correctly load the supervision edge
    def test_build_dataset_with_unique_supervision_edge_type(self):
        master_port = glt.utils.get_free_port(self._master_ip_address)
        mocked_dataset_artifact_metadata: MockedDatasetArtifactMetadata = (
            get_mocked_dataset_artifact_metadata()[
                TOY_GRAPH_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
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
        graph_metadata_pb_wrapper = gbml_config_pb_wrapper.graph_metadata_pb_wrapper

        serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
            preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            tfrecord_uri_pattern=".*\.tfrecord(\.gz)?$",
        )

        # We augment the serialized graph metadata to make the supervision edge type different than the message passing edge type
        assert isinstance(
            serialized_graph_metadata.node_entity_info, SerializedTFRecordInfo
        )
        assert isinstance(
            serialized_graph_metadata.edge_entity_info, SerializedTFRecordInfo
        )
        assert isinstance(
            serialized_graph_metadata.positive_label_entity_info,
            SerializedTFRecordInfo,
        )
        assert isinstance(
            serialized_graph_metadata.negative_label_entity_info,
            SerializedTFRecordInfo,
        )

        node_type = NodeType("user")
        message_passing_edge_type = EdgeType(node_type, Relation("to"), node_type)
        labeled_edge_type = EdgeType(node_type, Relation("labeled"), node_type)

        augmented_serialized_graph_metadata = SerializedGraphMetadata(
            node_entity_info={node_type: serialized_graph_metadata.node_entity_info},
            edge_entity_info={
                message_passing_edge_type: serialized_graph_metadata.edge_entity_info
            },
            positive_label_entity_info={
                labeled_edge_type: serialized_graph_metadata.positive_label_entity_info
            },
            negative_label_entity_info={
                labeled_edge_type: serialized_graph_metadata.negative_label_entity_info
            },
        )

        distributed_context = DistributedContext(
            main_worker_ip_address="localhost",
            global_rank=0,
            global_world_size=1,
        )

        dataset = build_dataset(
            serialized_graph_metadata=augmented_serialized_graph_metadata,
            distributed_context=distributed_context,
            sample_edge_direction="in",
            should_load_tensors_in_parallel=True,
            _dataset_building_port=master_port,
        )

        self.assertTrue(
            isinstance(dataset.positive_edge_label, abc.Mapping)
            and labeled_edge_type in dataset.positive_edge_label
        )
        self.assertTrue(
            isinstance(dataset.negative_edge_label, abc.Mapping)
            and labeled_edge_type in dataset.negative_edge_label
        )


if __name__ == "__main__":
    unittest.main()
