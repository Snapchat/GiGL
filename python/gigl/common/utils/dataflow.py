from apache_beam.runners.dataflow.dataflow_runner import DataflowPipelineResult

from gigl.common import HttpUri
from gigl.common.logger import Logger

logger = Logger()


def get_console_uri_from_pipeline_result(
    pipeline_result: DataflowPipelineResult,
) -> HttpUri:
    return HttpUri(
        f"https://console.cloud.google.com/dataflow/jobs/"
        f"{pipeline_result._job.location}/"
        f"{pipeline_result.job_id()}?"
        f"project={pipeline_result._job.projectId}"
    )
