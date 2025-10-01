import os
import tempfile
import unittest
import uuid

import kfp
from parameterized import param, parameterized

from gigl.common import UriFactory
from gigl.common.constants import DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU
from gigl.common.services.vertex_ai import (
    COMPUTE_CLUSTER_MASTER_KEY,
    STORAGE_CLUSTER_MASTER_KEY,
    VertexAiJobConfig,
    VertexAIService,
)
from gigl.env.pipelines_config import get_resource_config


@kfp.dsl.component
def source() -> int:
    return 42


@kfp.dsl.component
def doubler(a: int) -> int:
    return a * 2


@kfp.dsl.component
def adder(a: int, b: int) -> int:
    return a + b


@kfp.dsl.component
def division_by_zero(a: int) -> float:  # This is meant to fail
    return a / 0


@kfp.dsl.pipeline(name="kfp-integration-test")
def get_pipeline() -> int:
    source_task = source()
    double_task = doubler(a=source_task.output)
    adder_task = adder(a=source_task.output, b=double_task.output)
    return adder_task.output


@kfp.dsl.pipeline(name="kfp-integration-test-that-fails")
def get_pipeline_that_fails() -> float:
    source_task = source()
    fails_task = division_by_zero(a=source_task.output)  # This is meant to fail
    return fails_task.output


class VertexAIPipelineIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.resource_config = get_resource_config()
        self.project = self.resource_config.project
        self.location = self.resource_config.region
        self.service_account = self.resource_config.service_account_email
        self.staging_bucket = self.resource_config.temp_assets_regional_bucket_path.uri
        self.vertex_ai_service = VertexAIService(
            project=self.project,
            location=self.location,
            service_account=self.service_account,
            staging_bucket=self.staging_bucket,
        )
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def _test_launch_job(self):
        job_name = f"GiGL-Integration-Test-{uuid.uuid4()}"
        container_uri = "condaforge/miniforge3:25.3.0-1"
        command = ["python", "-c", "import logging; logging.info('Hello, World!')"]

        job_config = VertexAiJobConfig(
            job_name=job_name, container_uri=container_uri, command=command
        )

        self.vertex_ai_service.launch_job(job_config)

    @parameterized.expand(
        [
            param(
                "one server, one client",
                num_servers=1,
                num_clients=1,
                expected_worker_pool_specs=[
                    {
                        "machine_type": "n1-standard-4",
                        "num_replicas": 1,
                        "image_uri": "condaforge/miniforge3:25.3.0-1",
                    },
                    {},
                    {},
                    {
                        "machine_type": "n2-standard-8",
                        "num_replicas": 1,
                        "image_uri": DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
                    },
                ],
            ),
            param(
                "one server, two clients",
                num_servers=1,
                num_clients=2,
                expected_worker_pool_specs=[
                    {
                        "machine_type": "n1-standard-4",
                        "num_replicas": 1,
                        "image_uri": "condaforge/miniforge3:25.3.0-1",
                    },
                    {},
                    {},
                    {
                        "machine_type": "n2-standard-8",
                        "num_replicas": 2,
                        "image_uri": DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
                    },
                ],
            ),
            param(
                "two servers, one client",
                num_servers=2,
                num_clients=1,
                expected_worker_pool_specs=[
                    {
                        "machine_type": "n1-standard-4",
                        "num_replicas": 1,
                        "image_uri": "condaforge/miniforge3:25.3.0-1",
                    },
                    {
                        "machine_type": "n1-standard-4",
                        "num_replicas": 1,
                        "image_uri": "condaforge/miniforge3:25.3.0-1",
                    },
                    {},
                    {
                        "machine_type": "n2-standard-8",
                        "num_replicas": 1,
                        "image_uri": DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
                    },
                ],
            ),
            param(
                "two servers, two clients",
                num_servers=2,
                num_clients=2,
                expected_worker_pool_specs=[
                    {
                        "machine_type": "n1-standard-4",
                        "num_replicas": 1,
                        "image_uri": "condaforge/miniforge3:25.3.0-1",
                    },
                    {
                        "machine_type": "n1-standard-4",
                        "num_replicas": 1,
                        "image_uri": "condaforge/miniforge3:25.3.0-1",
                    },
                    {},
                    {
                        "machine_type": "n2-standard-8",
                        "num_replicas": 2,
                        "image_uri": DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
                    },
                ],
            ),
        ]
    )
    def test_launch_graph_store_job(
        self, _, num_servers, num_clients, expected_worker_pool_specs
    ):
        env_checks = f"logging.info(f'Graph cluster master: {{os.environ[\"{STORAGE_CLUSTER_MASTER_KEY}\"]}}, compute cluster master: {{os.environ[\"{COMPUTE_CLUSTER_MASTER_KEY}\"]}}')"
        command = [
            "python",
            "-c",
            f"import os; import logging; {env_checks}",
        ]
        job_name = f"GiGL-Integration-Test-Graph-Store-{uuid.uuid4()}"
        storage_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri="condaforge/miniforge3:25.3.0-1",  # different images for storage and compute
            replica_count=num_servers,
            machine_type="n1-standard-4",  # Different machine shapes - ideally we would test with GPU too but want to save on costs
            command=command,
        )
        compute_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,  # different image for storage and compute
            replica_count=num_clients,
            command=command,
            machine_type="n2-standard-8",  # Different machine shapes - ideally we would test with GPU too but want to save on costs
        )

        job = self.vertex_ai_service.launch_graph_store_job(
            storage_cluster_config, compute_cluster_config
        )

        self.assertEqual(
            len(job.job_spec.worker_pool_specs), len(expected_worker_pool_specs)
        )
        for i, worker_pool_spec in enumerate(job.job_spec.worker_pool_specs):
            expected_worker_pool_spec = expected_worker_pool_specs[i]
            if expected_worker_pool_spec:
                self.assertEqual(
                    worker_pool_spec.machine_spec.machine_type,
                    expected_worker_pool_spec["machine_type"],
                )
                self.assertEqual(
                    worker_pool_spec.replica_count,
                    expected_worker_pool_spec["num_replicas"],
                )
                self.assertEqual(
                    worker_pool_spec.container_spec.image_uri,
                    expected_worker_pool_spec["image_uri"],
                )

    def _test_run_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_def = os.path.join(tmpdir, "pipeline.yaml")
            kfp.compiler.Compiler().compile(get_pipeline, pipeline_def)
            job = self.vertex_ai_service.run_pipeline(
                display_name="integration-test-pipeline",
                template_path=UriFactory.create_uri(pipeline_def),
                run_keyword_args={},
                experiment="gigl-integration-tests",
                labels={"gigl-integration-test": "true"},
            )
            # Wait for the run to complete, 30 minutes is probably too long but
            # we don't want this test to be flaky.
            self.vertex_ai_service.wait_for_run_completion(
                job.resource_name, timeout=60 * 30, polling_period_s=10
            )

            # Also verify that we can fetch a pipeline.
            run = self.vertex_ai_service.get_pipeline_job_from_job_name(job.name)
            self.assertEqual(run.resource_name, job.resource_name)
            self.assertEqual(run.labels["gigl-integration-test"], "true")

    def _test_run_pipeline_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_def = os.path.join(tmpdir, "pipeline_that_fails.yaml")
            kfp.compiler.Compiler().compile(get_pipeline_that_fails, pipeline_def)
            job = self.vertex_ai_service.run_pipeline(
                display_name="integration-test-pipeline-that-fails",
                template_path=UriFactory.create_uri(pipeline_def),
                run_keyword_args={},
                experiment="gigl-integration-tests",
                labels={"gigl-integration-test": "true"},
            )
            with self.assertRaises(RuntimeError):
                self.vertex_ai_service.wait_for_run_completion(
                    job.resource_name, timeout=60 * 30, polling_period_s=10
                )


if __name__ == "__main__":
    unittest.main()
