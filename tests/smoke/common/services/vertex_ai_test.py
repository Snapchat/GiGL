import os
import tempfile
import uuid

import kfp
from absl.testing import absltest
from google.cloud.aiplatform_v1.types import env_var
from parameterized import param, parameterized

from gigl.common import UriFactory
from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
from gigl.env.constants import GIGL_CPU_DOCKER_URI_ENV_KEY
from gigl.env.pipelines_config import get_resource_config
from tests.test_assets.test_case import TestCase

# Short timeout so a broken image fails fast instead of hanging CI until the outer
# Cloud Build timeout. launch_graph_store_job passes this through to Vertex AI
# directly (it does not apply the 24h launch_job default).
_SMOKE_JOB_TIMEOUT_S = 30 * 60


def _assert_machine_cpu_count(expected_cpu_count: int) -> None:
    """Worker entrypoint: assert the provisioned VM exposes the expected vCPU count.

    Invoked on a smoke-test worker via a thin ``python -c`` import+call (the freshly
    built ``src-cpu`` image contains this module). ``os.cpu_count()`` reflects the
    machine_type's vCPUs on Vertex AI's dedicated VMs. An ``AssertionError`` exits the
    worker non-zero, failing the job — surfaced back in the test by the launch's
    blocking wait.

    Args:
        expected_cpu_count: vCPU count the machine_type should provision.
    """
    num_cpus = os.cpu_count()
    assert num_cpus == expected_cpu_count, (
        f"Expected {expected_cpu_count} CPUs, but got {num_cpus}"
    )


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


@kfp.dsl.pipeline(name="kfp-smoke-test")
def get_pipeline() -> int:
    source_task = source()
    double_task = doubler(a=source_task.output)
    adder_task = adder(a=source_task.output, b=double_task.output)
    return adder_task.output


@kfp.dsl.pipeline(name="kfp-smoke-test-that-fails")
def get_pipeline_that_fails() -> float:
    source_task = source()
    fails_task = division_by_zero(a=source_task.output)  # This is meant to fail
    return fails_task.output


class VertexAIPipelineSmokeTest(TestCase):
    def setUp(self):
        # Read the fresh-source image first, before any cloud work, so a misconfigured
        # run (no GIGL_CPU_DOCKER_URI) fails fast.
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
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_launch_job(self):
        job_name = f"GiGL-Smoke-Test-{uuid.uuid4()}"
        command = [
            "python",
            "-c",
            "from tests.smoke.common.services.vertex_ai_test import _assert_machine_cpu_count; "
            "_assert_machine_cpu_count(4)",  # n1-standard-4
        ]
        job_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=self._src_cpu_image_uri,
            command=command,
            machine_type="n1-standard-4",
            environment_variables=[env_var.EnvVar(name="FOO", value="BAR")],
            timeout_s=_SMOKE_JOB_TIMEOUT_S,
        )

        job = self._vertex_ai_service.launch_job(job_config)
        self.assertIn(
            env_var.EnvVar(name="FOO", value="BAR"),
            job.job_spec.worker_pool_specs[0].container_spec.env,
        )

    @parameterized.expand(
        [
            param("one compute, one storage", num_compute=1, num_storage=1),
            param("two compute, one storage", num_compute=2, num_storage=1),
        ]
    )
    def test_launch_graph_store_job(self, _, num_compute, num_storage):
        job_name = f"GiGL-Smoke-Test-Graph-Store-{uuid.uuid4()}"
        # Deliberately different machine shapes per pool, so the CPU check verifies
        # each shape actually provisioned as requested.
        compute_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=self._src_cpu_image_uri,
            replica_count=num_compute,
            machine_type="n1-standard-4",
            command=[
                "python",
                "-c",
                "from tests.smoke.common.services.vertex_ai_test import _assert_machine_cpu_count; "
                "_assert_machine_cpu_count(4)",  # n1-standard-4
            ],
            timeout_s=_SMOKE_JOB_TIMEOUT_S,
        )
        storage_cluster_config = VertexAiJobConfig(
            job_name=job_name,
            container_uri=self._src_cpu_image_uri,
            replica_count=num_storage,
            machine_type="n2-standard-8",
            command=[
                "python",
                "-c",
                "from tests.smoke.common.services.vertex_ai_test import _assert_machine_cpu_count; "
                "_assert_machine_cpu_count(8)",  # n2-standard-8
            ],
            timeout_s=_SMOKE_JOB_TIMEOUT_S,
        )

        job = self._vertex_ai_service.launch_graph_store_job(
            compute_cluster_config, storage_cluster_config
        )

        # Built here (not in the @parameterized.expand decorator, which evaluates at
        # import time) since the image is read from the environment in setUp.
        expected_worker_pool_specs: list[dict] = [
            {
                "machine_type": "n1-standard-4",
                "num_replicas": 1,
                "image_uri": self._src_cpu_image_uri,
            },
        ]
        if num_compute > 1:
            expected_worker_pool_specs.append(
                {
                    "machine_type": "n1-standard-4",
                    "num_replicas": num_compute - 1,
                    "image_uri": self._src_cpu_image_uri,
                }
            )
        else:
            expected_worker_pool_specs.append({})
        expected_worker_pool_specs.append(
            {
                "machine_type": "n2-standard-8",
                "num_replicas": num_storage,
                "image_uri": self._src_cpu_image_uri,
            }
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

    def test_run_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_def = os.path.join(tmpdir, "pipeline.yaml")
            kfp.compiler.Compiler().compile(get_pipeline, pipeline_def)
            job = self._vertex_ai_service.run_pipeline(
                display_name="smoke-test-pipeline",
                template_path=UriFactory.create_uri(pipeline_def),
                run_keyword_args={},
                experiment="gigl-smoke-tests",
                labels={"gigl-smoke-test": "true"},
            )
            # Wait for the run to complete, 30 minutes is probably too long but
            # we don't want this test to be flaky.
            self._vertex_ai_service.wait_for_run_completion(
                job.resource_name, timeout=60 * 30, polling_period_s=10
            )

            # Also verify that we can fetch a pipeline.
            run = self._vertex_ai_service.get_pipeline_job_from_job_name(job.name)
            self.assertEqual(run.resource_name, job.resource_name)
            self.assertEqual(run.labels["gigl-smoke-test"], "true")

    def test_run_pipeline_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_def = os.path.join(tmpdir, "pipeline_that_fails.yaml")
            kfp.compiler.Compiler().compile(get_pipeline_that_fails, pipeline_def)
            job = self._vertex_ai_service.run_pipeline(
                display_name="smoke-test-pipeline-that-fails",
                template_path=UriFactory.create_uri(pipeline_def),
                run_keyword_args={},
                experiment="gigl-smoke-tests",
                labels={"gigl-smoke-test": "true"},
            )
            with self.assertRaises(RuntimeError):
                self._vertex_ai_service.wait_for_run_completion(
                    job.resource_name, timeout=60 * 30, polling_period_s=10
                )


if __name__ == "__main__":
    absltest.main()
