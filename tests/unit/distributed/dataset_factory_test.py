from collections import abc
from typing import Any, cast
from unittest.mock import Mock, patch

from absl.testing import absltest
from parameterized import param, parameterized

from gigl.distributed.constants import (
    DEFAULT_PARTITIONER_CHUNK_COUNT,
    DEFAULT_PARTITIONER_NUM_RPC_THREADS,
)
from gigl.distributed.dataset_factory import (
    _build_dataset_process,
    build_dataset,
    build_dataset_from_task_config_uri,
)
from gigl.distributed.dist_context import DistributedContext
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE
from tests.test_assets.test_case import TestCase


# TODO(kmonte, mkolodner): Add more tests for heterogeneous datasets.
class TestDatasetFactory(TestCase):
    def setUp(self):
        # Set up any necessary context or mock data
        self._dist_context = DistributedContext(
            main_worker_ip_address="localhost", global_rank=0, global_world_size=1
        )

    @patch("gigl.distributed.dataset_factory.mp.spawn")
    @patch("gigl.distributed.dataset_factory.mp.Manager")
    def test_build_dataset_propagates_custom_partitioner_knobs(
        self, mock_manager: Mock, mock_spawn: Mock
    ) -> None:
        mock_manager.return_value.dict.return_value = {"dataset": Mock()}

        build_dataset(
            serialized_graph_metadata=Mock(),
            sample_edge_direction="in",
            distributed_context=self._dist_context,
            partitioner_chunk_count=3,
            partitioner_num_rpc_threads=7,
        )

        self.assertEqual(mock_spawn.call_args.kwargs["fn"], _build_dataset_process)
        self.assertEqual(mock_spawn.call_args.kwargs["args"][-2:], (3, 7))

    @patch("gigl.distributed.dataset_factory.mp.spawn")
    @patch("gigl.distributed.dataset_factory.mp.Manager")
    def test_build_dataset_propagates_default_partitioner_knobs(
        self, mock_manager: Mock, mock_spawn: Mock
    ) -> None:
        mock_manager.return_value.dict.return_value = {"dataset": Mock()}

        build_dataset(
            serialized_graph_metadata=Mock(),
            sample_edge_direction="in",
            distributed_context=self._dist_context,
        )

        self.assertEqual(
            mock_spawn.call_args.kwargs["args"][-2:],
            (DEFAULT_PARTITIONER_CHUNK_COUNT, DEFAULT_PARTITIONER_NUM_RPC_THREADS),
        )

    def test_build_dataset_rejects_invalid_partitioner_knobs(self) -> None:
        for parameter_name, invalid_value in (
            ("partitioner_chunk_count", True),
            ("partitioner_chunk_count", "3"),
            ("partitioner_num_rpc_threads", 0),
            ("partitioner_num_rpc_threads", 1.5),
        ):
            with self.subTest(
                parameter_name=parameter_name, invalid_value=invalid_value
            ):
                with self.assertRaises(ValueError):
                    build_dataset(
                        serialized_graph_metadata=Mock(),
                        sample_edge_direction="in",
                        distributed_context=self._dist_context,
                        **{parameter_name: cast(Any, invalid_value)},
                    )

    @patch("gigl.distributed.dataset_factory.shutdown_rpc")
    @patch("gigl.distributed.dataset_factory.barrier")
    @patch("gigl.distributed.dataset_factory._load_and_build_partitioned_dataset")
    @patch("gigl.distributed.dataset_factory.init_rpc")
    @patch("gigl.distributed.dataset_factory.init_worker_group")
    def test_build_dataset_process_wires_partitioner_knobs(
        self,
        mock_init_worker_group: Mock,
        mock_init_rpc: Mock,
        mock_load_dataset: Mock,
        mock_barrier: Mock,
        mock_shutdown_rpc: Mock,
    ) -> None:
        output_dict: dict[str, Mock] = {}
        output_dataset = Mock()
        mock_load_dataset.return_value = output_dataset

        _build_dataset_process(
            process_number_on_current_machine=0,
            output_dict=output_dict,
            serialized_graph_metadata=Mock(),
            master_ip_address="localhost",
            master_dataset_building_ports=(1234, 1235),
            node_rank=0,
            node_world_size=1,
            sample_edge_direction="in",
            should_load_tensors_in_parallel=True,
            partitioner_class=None,
            node_tf_dataset_options=Mock(),
            edge_tf_dataset_options=Mock(),
            partitioner_chunk_count=3,
            partitioner_num_rpc_threads=7,
        )

        mock_init_rpc.assert_called_once_with(
            master_addr="localhost", master_port=1234, num_rpc_threads=7
        )
        self.assertEqual(
            mock_load_dataset.call_args.kwargs["partitioner_chunk_count"], 3
        )
        self.assertIs(output_dict["dataset"], output_dataset)

    @parameterized.expand(
        [
            param("training", is_inference=False),
            param("inference", is_inference=True),
        ]
    )
    def test_build_dataset_from_task_config_uri_homogeneous(
        self, _, is_inference: bool
    ):
        # Test with a valid task config URI
        task_config_uri = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ].frozen_gbml_config_uri

        dataset = build_dataset_from_task_config_uri(
            task_config_uri,
            self._dist_context,
            is_inference=is_inference,
            _tfrecord_uri_pattern=".*data.tfrecord$",
        )

        if is_inference:
            self.assertIsNone(dataset.train_node_ids)
            self.assertIsNone(dataset.val_node_ids)
            self.assertIsNone(dataset.test_node_ids)
        else:
            # Mapping despite being "homogeneous" as ABLP uses labels as edge types.
            # Use assert isinstance instead of self.assertIsInstance to type narrow.
            assert isinstance(dataset.train_node_ids, abc.Mapping)
            self.assertTrue(
                dataset.train_node_ids.keys() == set([DEFAULT_HOMOGENEOUS_NODE_TYPE])
            )
            assert isinstance(dataset.val_node_ids, abc.Mapping)
            self.assertTrue(
                dataset.val_node_ids.keys() == set([DEFAULT_HOMOGENEOUS_NODE_TYPE])
            )
            assert isinstance(dataset.test_node_ids, abc.Mapping)
            self.assertTrue(
                dataset.val_node_ids.keys() == set([DEFAULT_HOMOGENEOUS_NODE_TYPE])
            )


if __name__ == "__main__":
    absltest.main()
