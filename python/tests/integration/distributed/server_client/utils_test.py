import os
import unittest
from unittest import mock
import torch
import torch.multiprocessing as mp
from graphlearn_torch.distributed import shutdown_rpc, init_client

from gigl.env.distributed import NUM_PROCESSES_PER_COMPUTE, NUM_PROCESSES_PER_STORAGE
from gigl.distributed.server_client.server_main import run_servers
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.common import UriFactory
from gigl.env.pipelines_config import get_resource_config
from gigl.env.distributed import GraphStoreInfo
from gigl.distributed.utils import get_graph_store_info
from gigl.distributed.server_client.utils import get_sampler_input_for_inference
from gigl.src.common.types.graph_data import NodeType
from gigl.distributed.utils import get_free_port
from tests.test_assets.distributed.utils import assert_tensor_equality
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper


def _client_process(
    client_rank: int,
    node_type: NodeType,
    expected_sampler_input: dict[int, list[torch.Tensor]]
) -> None:
    torch.distributed.init_process_group()
    cluster_info = get_graph_store_info()
    sampler_input = get_sampler_input_for_inference(
        client_rank,
        cluster_info,
        node_type,
    )
    rank_expected_sampler_input = expected_sampler_input[client_rank]
    assert len(sampler_input) == len(rank_expected_sampler_input)
    for i in range(len(sampler_input)):
        assert_tensor_equality(sampler_input[i], rank_expected_sampler_input[i])

class TestUtils(unittest.TestCase):
    def test_get_sampler_input_for_inference(self):
        # Simulating two server machine, two compute machines.
        # Each machine has one process.
        cora_supervised_info = get_mocked_dataset_artifact_metadata()[
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        task_config_uri = cora_supervised_info.frozen_gbml_config_uri
        task_config = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            gbml_config_uri=task_config_uri
        )
        resource_config_uri = get_resource_config().get_resource_config_uri
        cluster_info = GraphStoreInfo(
            num_cluster_nodes=4,
            num_storage_nodes=2,
            num_compute_nodes=2,
            num_processes_per_storage=1,
            num_processes_per_compute=1,
            cluster_master_ip="127.0.0.1",
            storage_cluster_master_ip="127.0.0.1",
            compute_cluster_master_ip="127.0.0.1",
            cluster_master_port=get_free_port(),
            storage_cluster_master_port=get_free_port(),
            compute_cluster_master_port=get_free_port(),
        )
        # Start server process
        with mock.patch.dict(
            os.environ,
            {
                "RANK": "0",
                "WORLD_SIZE": "2",
                NUM_PROCESSES_PER_STORAGE: "1",
                NUM_PROCESSES_PER_COMPUTE: "2",
            },
            clear=False,
        ):
            server_processes = mp.spawn(
                run_servers,
                args=[
                    cluster_info, # cluster_info
                    UriFactory.create_uri(task_config_uri), # task_config_uri
                    True, # is_inference
                ]
            )
            assert server_processes is not None

        with mock.patch.dict(
            os.environ,
            {
                "RANK": "1",
                "WORLD_SIZE": "2",
                NUM_PROCESSES_PER_STORAGE: "1",
                NUM_PROCESSES_PER_COMPUTE: "2",
            },
            clear=False,
        ):
            client_processes = mp.spawn(
                _client_process,
                args=[
                    task_config.graph_metadata_pb_wrapper.homogeneous_node_type, # node_type
                ]
            )
            assert client_processes is not None
