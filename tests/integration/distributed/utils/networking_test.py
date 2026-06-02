import uuid
from textwrap import dedent

from google.cloud.aiplatform_v1.types import env_var
from parameterized import param, parameterized

from gigl.common.constants import DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.constants import GIGL_RESOURCE_CONFIG_URI_ENV_KEY
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.file_loader import FileLoader
from tests.test_assets.test_case import TestCase


class NetworkingUtlsIntegrationTest(TestCase):
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

        # get_graph_store_info() (run on the launched workers) calls
        # get_resource_config() to build the readiness URI, so the workers need a
        # resource config they can read. The test runner's resource config URI may
        # be a local path that does not exist on the worker image, so we upload the
        # in-memory resource config to the regional bucket (which the workers can
        # read from GCS) and pass that URI via GIGL_RESOURCE_CONFIG_URI.
        self._file_loader = FileLoader()
        self._remote_resource_config_uri = (
            self._resource_config.temp_assets_regional_bucket_path
            / "gigl"
            / "integration_tests"
            / "networking"
            / f"resource_config_{uuid.uuid4()}.yaml"
        )
        ProtoUtils().write_proto_to_yaml(
            proto=self._resource_config.resource_config,
            uri=self._remote_resource_config_uri,
        )
        super().setUp()

    def tearDown(self):
        self._file_loader.delete_files([self._remote_resource_config_uri])
        super().tearDown()

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
        # launch_graph_store_job propagates the compute pool's environment_variables
        # to both the compute and storage container specs, so the uploaded resource
        # config URI is visible to every worker.
        resource_config_env_vars = [
            env_var.EnvVar(
                name=GIGL_RESOURCE_CONFIG_URI_ENV_KEY,
                value=self._remote_resource_config_uri.uri,
            )
        ]
        compute_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
            replica_count=compute_nodes,
            command=command,
            machine_type="n2-standard-8",
            environment_variables=resource_config_env_vars,
        )
        storage_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
            replica_count=storage_nodes,
            machine_type="n1-standard-4",
            command=command,
        )

        self._vertex_ai_service.launch_graph_store_job(
            compute_cluster_config, storage_cluster_config
        )
