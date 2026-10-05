from apache_beam.runners.dataflow.dataflow_runner import DataflowPipelineResult
from google.cloud import dataflow

from gigl.common.utils.dataflow import get_console_uri_from_pipeline_result
from tests.test_assets.test_case import TestCase


class DataflowUtilsTest(TestCase):
    def test_get_console_uri_from_pipeline_result(self) -> None:
        # DataflowRunner builds its result around the Dataflow API's `Job` message; this is
        # that message as the runner hands it back after submission.
        job = dataflow.Job(
            id="2026-01-01_00_00_00-123", project_id="my-project", location="us-east1"
        )
        pipeline_result = DataflowPipelineResult(job=job, runner=None)

        console_uri = get_console_uri_from_pipeline_result(
            pipeline_result=pipeline_result
        )

        self.assertEqual(
            console_uri.uri,
            "https://console.cloud.google.com/dataflow/jobs/us-east1/"
            "2026-01-01_00_00_00-123?project=my-project",
        )
