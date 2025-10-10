import unittest
import uuid
from textwrap import dedent

from parameterized import param, parameterized

from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
from gigl.env.pipelines_config import get_resource_config
from gigl.common.constants import GIGL_RELEASE_SRC_IMAGE_CPU


class NetworkingUtlsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._resource_config = get_resource_config()
        self._project = self._resource_config.project
        self._location = self._resource_config.region
        self._service_account = self._resource_config.service_account_email
        self._staging_bucket = (
            self._resource_config.temp_assets_regional_bucket_path.uri
        )
        self._vertex_ai_service = VertexAIService(
            project=self._project,
            location=self._location,
            service_account=self._service_account,
            staging_bucket=self._staging_bucket,
        )
        super().setUp()

    @parameterized.expand(
        [
            param(
                "Test with 1 compute node and 1 storage node",
                compute_nodes=1,
                storage_nodes=1,
            ),
            param(
                "Test with 2 compute nodes and 2 storage nodes",
                compute_nodes=2,
                storage_nodes=2,
            ),
        ]
    )
    def test_get_graph_store_info(self, _, storage_nodes, compute_nodes):
        job_name = f"GiGL-Integration-Test-Graph-Store-{uuid.uuid4()}"
        command = [
            "python",
            "-c",
            dedent(
                f"""
                import torch
                from gigl.distributed.utils import get_graph_store_info
                torch.distributed.init_process_group(backend="gloo")
                info = get_graph_store_info()
                assert info.num_storage_nodes == {storage_nodes}, f"Expected {storage_nodes} storage nodes, but got {{ info.num_storage_nodes }}"
                assert info.num_compute_nodes == {compute_nodes}, f"Expected {compute_nodes} compute nodes, but got {{ info.num_compute_nodes }}"
                assert info.num_cluster_nodes == {storage_nodes + compute_nodes}, f"Expected {storage_nodes + compute_nodes} cluster nodes, but got {{ info.num_cluster_nodes }}"
                assert info.cluster_master_ip is not None, f"Cluster master IP is None"
                assert info.storage_cluster_master_ip is not None, f"Storage cluster master IP is None"
                assert info.compute_cluster_master_ip is not None, f"Compute cluster master IP is None"
                assert info.cluster_master_port is not None, f"Cluster master port is None"
                assert info.storage_cluster_master_port is not None, f"Storage cluster master port is None"
                assert info.compute_cluster_master_port is not None, f"Compute cluster master port is None"
                """
            ),
        ]
        compute_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=GIGL_RELEASE_SRC_IMAGE_CPU,
            replica_count=compute_nodes,
            command=command,
            machine_type="n2-standard-8",
        )
        storage_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=GIGL_RELEASE_SRC_IMAGE_CPU,
            replica_count=storage_nodes,
            machine_type="n1-standard-4",
            command=command,
        )

        self._vertex_ai_service.launch_graph_store_job(
            compute_cluster_config, storage_cluster_config
        )
