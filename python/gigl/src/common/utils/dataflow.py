from typing import Any, Optional

from apache_beam.options.pipeline_options import (
    DebugOptions,
    GoogleCloudOptions,
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)

from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.env.dep_constants import GIGL_DATAFLOW_IMAGE
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants import gcs as gcs_constants
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.dataflow_job_options import CommonOptions

logger = Logger()


def get_sanitized_dataflow_job_name(name: str) -> str:
    name = name.lower()
    name = name.replace("_", "-")
    name = "".join([c for c in name if c.isalnum() or c == "-"])
    logger.info(f"Will use sanitized dataflow job name: {name}")
    return name


def init_beam_pipeline_options(
    applied_task_identifier: AppliedTaskIdentifier,
    job_name_suffix: str,
    component: Optional[GiGLComponents] = None,
    custom_worker_image_uri: Optional[str] = None,
    **kwargs: Any,
) -> PipelineOptions:
    """Can pass in any options i.e.
    init_beam_pipeline_options(num_workers=1, max_num_workers=32, ...)
    The options passed in will override default options if we define them.
    For example, you can override the job_name by passing in `job_name="something"`

    Args:
        applied_task_identifier (AppliedTaskIdentifier)
        job_name_suffix (str): Unique identifier for the dataflow job in relation to this task (applied_task_identifier)
            i.e. job_name_suffix = "inference"
    Returns:
       PipelineOptions: options you can use to generate the pipeline
    """
    job_name = get_sanitized_dataflow_job_name(
        f"gigl-{applied_task_identifier}-{job_name_suffix}"
    )

    options = PipelineOptions(**kwargs)
    common_options = options.view_as(CommonOptions)

    resource_config_uri = UriFactory.create_uri(
        uri=get_resource_config().get_resource_config_uri
    )

    common_options.resource_config_uri = get_resource_config().get_resource_config_uri

    # https://cloud.google.com/dataflow/docs/guides/build-container-image#pre-build_using_a_dockerfile
    setup_options = options.view_as(SetupOptions)
    setup_options.sdk_location = "container"
    worker_options: WorkerOptions = options.view_as(WorkerOptions)
    worker_options.sdk_container_image = custom_worker_image_uri or GIGL_DATAFLOW_IMAGE

    debug_options = options.view_as(DebugOptions)
    debug_options.experiments = debug_options.experiments or [
        "shuffle_mode=service",
        "use_runner_v2",
        "enable_stackdriver_agent_metrics",
        # Allows you to increase the size of your job graph to more than 10MB
        # Temporarily circumventing large job graphs; ideally we should try to limit this but applying band-aid for now.
        # https://cloud.google.com/knowledge/kb/dataflow-job-fails-with-error-message-for-large-job-graphs-000007130
        "upload_graph",
    ]
    standard_options = options.view_as(StandardOptions)
    standard_options.runner = (
        standard_options.runner or get_resource_config().dataflow_runner
    )

    google_cloud_options = options.view_as(GoogleCloudOptions)
    google_cloud_options.labels = google_cloud_options.labels or (
        get_resource_config().get_resource_labels_formatted_for_dataflow(
            component=component
        )
    )
    google_cloud_options.project = (
        google_cloud_options.project or get_resource_config().project
    )
    google_cloud_options.job_name = job_name
    google_cloud_options.staging_location = (
        google_cloud_options.staging_location
        or gcs_constants.get_dataflow_staging_gcs_path(
            applied_task_identifier=applied_task_identifier, job_name=job_name
        ).uri
    )
    google_cloud_options.temp_location = (
        google_cloud_options.temp_location
        or gcs_constants.get_dataflow_temp_gcs_path(
            applied_task_identifier=applied_task_identifier, job_name=job_name
        ).uri
    )
    google_cloud_options.region = (
        google_cloud_options.region or get_resource_config().region
    )

    # For context see: https://cloud.google.com/dataflow/docs/reference/service-options#python
    # This is different than how `num_workers` is leveraged by dataflow in the default `PipelineOptions` exposed by beam.
    # i.e. simply setting `num_workers` in `PipelineOptions`, the dataflow service still may downscale to 1 worker.
    # vs. setting `min_num_workers` in `dataflow_service_options` explicitly will ensure that the service will not downscale below
    # that number.
    if kwargs.get("num_workers"):
        num_workers = kwargs.get("num_workers")
        logger.info(
            f"Setting `min_num_workers` for Dataflow explicitly to {num_workers}"
        )
        dataflow_service_options = google_cloud_options.dataflow_service_options or []
        dataflow_service_options.append(f"min_num_workers={num_workers}")
        google_cloud_options.dataflow_service_options = dataflow_service_options

    google_cloud_options.service_account_email = (
        google_cloud_options.service_account_email
        or (get_resource_config().service_account_email)
    )

    return options
