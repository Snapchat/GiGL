import os
import tempfile
import unittest
import uuid

import kfp

from gigl.common import UriFactory
from gigl.common.services.vertex_ai import VertexAiJobConfig, VertexAIService
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


@kfp.dsl.pipeline(name="kfp-integration-test")
def get_pipeline() -> int:
    source_task = source()
    double_task = doubler(a=source_task.output)
    adder_task = adder(a=source_task.output, b=double_task.output)
    return adder_task.output


class VertexAIPipelineIntegrationTest(unittest.TestCase):
    def test_launch_job(self):
        resource_config = get_resource_config()
        project = resource_config.project
        location = resource_config.region
        service_account = resource_config.service_account_email
        staging_bucket = resource_config.temp_assets_regional_bucket_path.uri
        job_name = f"GiGL-Intergration-Test-{uuid.uuid4()}"
        container_uri = "continuumio/miniconda3:4.12.0"
        command = ["python", "-c", "import logging; logging.info('Hello, World!')"]

        job_config = VertexAiJobConfig(
            job_name=job_name, container_uri=container_uri, command=command
        )

        vertex_ai_service = VertexAIService(
            project=project,
            location=location,
            service_account=service_account,
            staging_bucket=staging_bucket,
        )

        vertex_ai_service.launch_job(job_config)

    def test_run_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_def = os.path.join(tmpdir, "pipeline.yaml")
            kfp.compiler.Compiler().compile(get_pipeline, pipeline_def)
            resource_config = get_resource_config()
            ps = VertexAIService(
                project=resource_config.project,
                location=resource_config.region,
                service_account=resource_config.service_account_email,
                staging_bucket=resource_config.temp_assets_regional_bucket_path.uri,
            )
            job = ps.run_pipeline(
                display_name="integration-test-pipeline",
                template_path=UriFactory.create_uri(pipeline_def),
                run_keyword_args={},
                experiment="gigl-integration-tests",
            )
            # Wait for the run to complete, 30 minutes is probably too long but
            # we don't want this test to be flaky.
            ps.wait_for_run_completion(
                job.resource_name, timeout=60 * 30, polling_period_s=10
            )

            # Also verify that we can fetch a pipeline.
            run = ps.get_pipeline_job_from_job_name(job.name)
            self.assertEqual(run.resource_name, job.resource_name)


if __name__ == "__main__":
    unittest.main()
