import os
import uuid

import torch
from google.cloud.aiplatform_v1.types import env_var
from parameterized import param, parameterized

from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
from gigl.common.utils.file_loader import FileLoader
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.distributed.utils import get_graph_store_info
from gigl.env.constants import (
    GIGL_CPU_DOCKER_URI_ENV_KEY,
    GIGL_RESOURCE_CONFIG_URI_ENV_KEY,
)
from gigl.env.pipelines_config import get_resource_config
from tests.test_assets.test_case import TestCase

# Short timeout so a broken fresh image fails fast instead of hanging CI until the
# outer Cloud Build timeout. launch_graph_store_job passes this through to Vertex AI
# directly (it does not apply the 24h launch_job default).
_INTEGRATION_JOB_TIMEOUT_S = 30 * 60


def _assert_graph_store_info(num_storage_nodes: int, num_compute_nodes: int) -> None:
    """Worker entrypoint for ``test_get_graph_store_info``.

    Runs on every node of the launched Vertex AI graph-store cluster (invoked via a
    thin ``python -c`` import+call; the freshly built ``src-cpu`` image contains this
    module). Initializes the cluster-wide process group, fetches the graph-store info,
    and asserts the derived topology matches the expected node counts.

    An ``AssertionError`` here exits the worker non-zero, which fails the Vertex AI
    job — surfaced back in the test by ``launch_graph_store_job``'s blocking wait.

    Args:
        num_storage_nodes: Expected number of storage nodes in the cluster.
        num_compute_nodes: Expected number of compute nodes in the cluster.
    """
    torch.distributed.init_process_group(backend="gloo")
    info = get_graph_store_info()
    assert info.num_storage_nodes == num_storage_nodes, (
        f"Expected {num_storage_nodes} storage nodes, but got {info.num_storage_nodes}"
    )
    assert info.num_compute_nodes == num_compute_nodes, (
        f"Expected {num_compute_nodes} compute nodes, but got {info.num_compute_nodes}"
    )
    assert info.num_cluster_nodes == num_storage_nodes + num_compute_nodes, (
        f"Expected {num_storage_nodes + num_compute_nodes} cluster nodes, but got {info.num_cluster_nodes}"
    )
    assert info.cluster_master_ip is not None, "Cluster master IP is None"
    assert info.storage_cluster_master_ip is not None, (
        "Storage cluster master IP is None"
    )
    assert info.compute_cluster_master_ip is not None, (
        "Compute cluster master IP is None"
    )
    assert info.cluster_master_port is not None, "Cluster master port is None"
    assert info.storage_cluster_master_port is not None, (
        "Storage cluster master port is None"
    )
    assert info.compute_cluster_master_port is not None, (
        "Compute cluster master port is None"
    )


class NetworkingUtilsIntegrationTest(TestCase):
    def setUp(self) -> None:
        # Read the fresh-source image first, before any cloud work, so a misconfigured
        # run (no GIGL_CPU_DOCKER_URI) fails fast without leaving GCS side effects
        # (unittest skips tearDown when setUp raises).
        self._src_cpu_image_uri = os.environ[GIGL_CPU_DOCKER_URI_ENV_KEY]

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

    def tearDown(self) -> None:
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
        # Thin import+call of the real worker function defined above. The freshly-built
        # ``src-cpu`` image contains this module, so it is importable on the workers.
        # storage_nodes/compute_nodes are test-controlled ints, so
        # interpolating them into the call is safe.
        command = [
            "python",
            "-c",
            f"from tests.integration.distributed.utils.networking_test import _assert_graph_store_info; "
            f"_assert_graph_store_info(num_storage_nodes={storage_nodes}, num_compute_nodes={compute_nodes})",
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
            container_uri=self._src_cpu_image_uri,
            replica_count=compute_nodes,
            command=command,
            machine_type="n2-standard-8",
            environment_variables=resource_config_env_vars,
            timeout_s=_INTEGRATION_JOB_TIMEOUT_S,
        )
        storage_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=self._src_cpu_image_uri,
            replica_count=storage_nodes,
            machine_type="n1-standard-4",
            command=command,
            timeout_s=_INTEGRATION_JOB_TIMEOUT_S,
        )

        self._vertex_ai_service.launch_graph_store_job(
            compute_cluster_config, storage_cluster_config
        )
