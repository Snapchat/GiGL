"""Class for interacting with Vertex AI.

Below are some brief definitions of the terminology used by Vertex AI Pipelines:

Resource name: A globally unique identifier for the pipeline, follows https://google.aip.dev/122 and is of the form projects/<project-id>/locations/<location>/pipelineJobs/<job-name>
Job name: aka job_id aka PipelineJob.name the name of a pipeline run, must be unique for a given project and location
Display name: AFAICT purely cosmetic name for a pipeline, can be filtered on but does not show up in the UI
Pipeline name: The name for the pipeline supplied by the pipeline definition (pipeline.yaml).

And a walkthrough to explain how the terminology is used:
```py
@kfp.dsl.component
def source() -> int:
    return 42

@kfp.dsl.component
def doubler(a: int) -> int:
    return a * 2

@kfp.dsl.component
def adder(a: int, b: int) -> int:
    return a + b

@kfp.dsl.pipeline
def get_pipeline() -> int: # NOTE: `get_pipeline` here is  the Pipeline name
    source_task = source()
    double_task = doubler(a=source_task.output)
    adder_task = adder(a=source_task.output, b=double_task.output)
    return adder_task.output

tempdir = tempfile.TemporaryDirectory()
tf = os.path.join(tempdir.name, "pipeline.yaml")
print(f"Writing pipeline definition to {tf}")
kfp.compiler.Compiler().compile(get_pipeline, tf)
job = aip.PipelineJob(
        display_name="this_is_our_pipeline_display_name",
        template_path=tf,
        pipeline_root="gs://my-bucket/pipeline-root",
)
    job.submit(service_account="my-sa@my-project.gserviceaccount.com")
```

Which outputs the following:
Creating PipelineJob
PipelineJob created. Resource name: projects/my-project-id/locations/us-central1/pipelineJobs/get-pipeline-20250226170755
To use this PipelineJob in another session:
pipeline_job = aiplatform.PipelineJob.get('projects/my-project-id/locations/us-central1/pipelineJobs/get-pipeline-20250226170755')
View Pipeline Job:
https://console.cloud.google.com/vertex-ai/locations/us-central1/pipelines/runs/get-pipeline-20250226170755?project=my-project-id
Associating projects/my-project-id/locations/us-central1/pipelineJobs/get-pipeline-20250226170755 to Experiment: example-experiment


And `job` has some properties set as well:

```py
print(f"{job.display_name=}") # job.display_name='this_is_our_pipeline_display_name'
print(f"{job.resource_name=}") # job.resource_name='projects/my-project-id/locations/us-central1/pipelineJobs/get-pipeline-20250226170755'
print(f"{job.name=}") # job.name='get-pipeline-20250226170755' # NOTE: by default, the "job name" is the pipeline name + datetime
```
"""

import datetime
import time
from dataclasses import dataclass
from typing import Final, Optional, Union

from google.cloud import aiplatform
from google.cloud.aiplatform_v1.types import (
    ContainerSpec,
    DiskSpec,
    MachineSpec,
    WorkerPoolSpec,
    env_var,
)

from gigl.common import GcsUri, Uri
from gigl.common.logger import Logger
from gigl.src.common.constants.distributed import (
    COMPUTE_CLUSTER_MASTER_KEY,
    COMPUTE_CLUSTER_NUM_NODES_KEY,
    STORAGE_CLUSTER_MASTER_KEY,
    STORAGE_CLUSTER_NUM_NODES_KEY,
)

logger = Logger()

LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY: Final[
    str
] = "LEADER_WORKER_INTERNAL_IP_FILE_PATH"


DEFAULT_PIPELINE_TIMEOUT_S: Final[int] = 60 * 60 * 36  # 36 hours
DEFAULT_CUSTOM_JOB_TIMEOUT_S: Final[int] = 60 * 60 * 24  # 24 hours


@dataclass
class VertexAiJobConfig:
    job_name: str
    container_uri: str
    command: list[str]
    args: Optional[list[str]] = None
    environment_variables: Optional[list[dict[str, str]]] = None
    machine_type: str = "n1-standard-4"
    accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED"
    accelerator_count: int = 0
    replica_count: int = 1
    boot_disk_type: str = "pd-ssd"  # Persistent Disk SSD
    boot_disk_size_gb: int = 100  # Default disk size in GB
    labels: Optional[dict[str, str]] = None
    timeout_s: Optional[
        int
    ] = None  # Will default to DEFAULT_CUSTOM_JOB_TIMEOUT_S if not provided
    enable_web_access: bool = True


class VertexAIService:
    """
    A class representing a Vertex AI service.

    Args:
        project (str): The project ID.
        location (str): The location of the service.
        service_account (str): The service account to use for authentication.
        staging_bucket (str): The staging bucket for the service.
    """

    def __init__(
        self,
        project: str,
        location: str,
        service_account: str,
        staging_bucket: str,
    ):
        self._project = project
        self._location = location
        self._service_account = service_account
        self._staging_bucket = staging_bucket
        aiplatform.init(
            project=self._project,
            location=self._location,
            staging_bucket=self._staging_bucket,
        )

    @property
    def project(self) -> str:
        """The GCP project that is being used for this service."""
        return self._project

    def launch_job(self, job_config: VertexAiJobConfig) -> aiplatform.CustomJob:
        """
        Launch a Vertex AI CustomJob.
        See the docs for more info.
        https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.CustomJob

        Args:
            job_config (VertexAiJobConfig): The configuration for the job.

        Returns:
            The completed CustomJob.
        """
        logger.info(f"Running Vertex AI job: {job_config.job_name}")

        machine_spec = _create_machine_spec(job_config)

        # This file is used to store the leader worker's internal IP address.
        # Whenever `connect_worker_pool()` is called, the leader worker will
        # write its internal IP address to this file. The other workers will
        # read this file to get the leader worker's internal IP address.
        # See connect_worker_pool() implementation for more details.
        leader_worker_internal_ip_file_path = GcsUri.join(
            self._staging_bucket,
            job_config.job_name,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            "leader_worker_internal_ip.txt",
        )
        env_vars = [
            env_var.EnvVar(
                name=LEADER_WORKER_INTERNAL_IP_FILE_PATH_ENV_KEY,
                value=leader_worker_internal_ip_file_path.uri,
            )
        ]

        container_spec = _create_container_spec(job_config, env_vars)

        disk_spec = _create_disk_spec(job_config)

        assert (
            job_config.replica_count >= 1
        ), "Replica count can be at minumum 1, i.e. leader worker"

        leader_worker_spec = WorkerPoolSpec(
            machine_spec=machine_spec,
            container_spec=container_spec,
            disk_spec=disk_spec,
            replica_count=1,
        )

        worker_pool_specs: list[WorkerPoolSpec] = [leader_worker_spec]

        if job_config.replica_count > 1:
            worker_spec = WorkerPoolSpec(
                machine_spec=machine_spec,
                container_spec=container_spec,
                disk_spec=disk_spec,
                replica_count=job_config.replica_count - 1,
            )
            worker_pool_specs.append(worker_spec)

        logger.info(
            f"Running Custom job {job_config.job_name} with worker_pool_specs {worker_pool_specs}, in project: {self._project}/{self._location} using staging bucket: {self._staging_bucket}, and attached labels: {job_config.labels}"
        )

        if not job_config.timeout_s:
            logger.info(
                f"No timeout set for Vertex AI job, setting default timeout to {DEFAULT_CUSTOM_JOB_TIMEOUT_S/60/60} hours"
            )
            job_config.timeout_s = DEFAULT_CUSTOM_JOB_TIMEOUT_S
        else:
            logger.info(
                f"Running Vertex AI job with timeout {job_config.timeout_s} seconds"
            )

        job = aiplatform.CustomJob(
            display_name=job_config.job_name,
            worker_pool_specs=worker_pool_specs,
            project=self._project,
            location=self._location,
            labels=job_config.labels,
            staging_bucket=self._staging_bucket,
        )
        job.submit(
            service_account=self._service_account,
            timeout=job_config.timeout_s,
            enable_web_access=job_config.enable_web_access,
        )
        job.wait_for_resource_creation()
        logger.info(f"Created job: {job.resource_name}")
        # Copying https://github.com/googleapis/python-aiplatform/blob/v1.48.0/google/cloud/aiplatform/jobs.py#L207-L215
        # Since for some reason upgrading from VertexAI v1.27.1 to v1.48.0
        # caused the logs to occasionally not be printed.
        logger.info(
            f"See job logs at: https://console.cloud.google.com/ai/platform/locations/{self._location}/training/{job.name}?project={self._project}"
        )
        job.wait_for_completion()
        return job

    def launch_graph_store_job(
        self, storage_cluster: VertexAiJobConfig, compute_cluster: VertexAiJobConfig
    ) -> aiplatform.CustomJob:
        """Launch a Vertex AI Graph Store job.

        This launches one Vertex AI CustomJob with two worker pools, see
        https://cloud.google.com/vertex-ai/docs/training/distributed-training
        for more details.

        These jobs will have the follow env variables set
            - GIGL_STORAGE_CLUSTER_MASTER_KEY
            - GIGL_COMPUTE_CLUSTER_MASTER_KEY
        Whose values are the cluster ranks of the leaders for the storage and compute clusters respectively.
        For example, if if there are 2 nodes in the storage cluster, and 3 nodes in the compute cluster,
        Then,
            - GIGL_STORAGE_CLUSTER_MASTER_KEY = 0
            - GIGL_COMPUTE_CLUSTER_MASTER_KEY = 2 # e.g. the "first" worker in the computer cluster pool is the leader.

        NOTE:
            We use the job_name, timeout, and enable_web_access from the storage cluster.
            These fields, if set on the compute cluster, will be ignored.

        Args:
            storage_cluster (VertexAiJobConfig): The configuration for the storage cluster.
            compute_cluster (VertexAiJobConfig): The configuration for the compute cluster.

        Returns:
            The completed CustomJob.
        """
        storage_machine_spec = _create_machine_spec(storage_cluster)
        compute_machine_spec = _create_machine_spec(compute_cluster)
        storage_disk_spec = _create_disk_spec(storage_cluster)
        compute_disk_spec = _create_disk_spec(compute_cluster)

        env_vars: list[env_var.EnvVar] = [
            env_var.EnvVar(
                name=STORAGE_CLUSTER_MASTER_KEY,
                value="0",
            ),
            env_var.EnvVar(
                name=COMPUTE_CLUSTER_MASTER_KEY,
                value=str(storage_cluster.replica_count),
            ),
            env_var.EnvVar(
                name=STORAGE_CLUSTER_NUM_NODES_KEY,
                value=str(storage_cluster.replica_count),
            ),
            env_var.EnvVar(
                name=COMPUTE_CLUSTER_NUM_NODES_KEY,
                value=str(compute_cluster.replica_count),
            ),
        ]

        storage_container_spec = _create_container_spec(storage_cluster, env_vars)
        compute_container_spec = _create_container_spec(compute_cluster, env_vars)

        worker_pool_specs: list[Union[WorkerPoolSpec, dict]] = []

        leader_worker_spec = WorkerPoolSpec(
            machine_spec=storage_machine_spec,
            container_spec=storage_container_spec,
            disk_spec=storage_disk_spec,
            replica_count=1,
        )
        worker_pool_specs.append(leader_worker_spec)
        if storage_cluster.replica_count > 1:
            worker_spec = WorkerPoolSpec(
                machine_spec=storage_machine_spec,
                container_spec=storage_container_spec,
                disk_spec=storage_disk_spec,
                replica_count=storage_cluster.replica_count - 1,
            )
            worker_pool_specs.append(worker_spec)
        else:
            worker_pool_specs.append({})
        # For whatever reason, VAI errors out (indescriptly) if we put the computer cluster as anything other
        # than the fourth worker pool.
        # So we need to pad.
        worker_pool_specs.append({})
        worker_spec = WorkerPoolSpec(
            machine_spec=compute_machine_spec,
            container_spec=compute_container_spec,
            disk_spec=compute_disk_spec,
            replica_count=compute_cluster.replica_count,
        )
        worker_pool_specs.append(worker_spec)

        job = aiplatform.CustomJob(
            display_name=storage_cluster.job_name,
            worker_pool_specs=worker_pool_specs,
            project=self._project,
            location=self._location,
            labels=storage_cluster.labels,
            staging_bucket=self._staging_bucket,
        )
        job.submit(
            service_account=self._service_account,
            timeout=storage_cluster.timeout_s,
            enable_web_access=storage_cluster.enable_web_access,
        )
        job.wait_for_resource_creation()
        logger.info(f"Created job: {job.resource_name}")
        # Copying https://github.com/googleapis/python-aiplatform/blob/v1.48.0/google/cloud/aiplatform/jobs.py#L207-L215
        # Since for some reason upgrading from VertexAI v1.27.1 to v1.48.0
        # caused the logs to occasionally not be printed.
        logger.info(
            f"See job logs at: https://console.cloud.google.com/ai/platform/locations/{self._location}/training/{job.name}?project={self._project}"
        )
        job.wait_for_completion()
        return job

    def run_pipeline(
        self,
        display_name: str,
        template_path: Uri,
        run_keyword_args: dict[str, str],
        job_id: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        experiment: Optional[str] = None,
    ) -> aiplatform.PipelineJob:
        """
        Runs a pipeline using the Vertex AI Pipelines service.
        For more info, see the Vertex AI docs
        https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.PipelineJob#google_cloud_aiplatform_PipelineJob_submit

        Args:
            display_name (str): The display of the pipeline.
            template_path (Uri): The path to the compiled pipeline YAML.
            run_keyword_args (dict[str, str]): Runtime arguements passed to your pipeline.
            job_id (Optional[str]): The ID of the job. If not provided will be the *pipeline_name* + datetime.
                                    Note: The pipeline_name and display_name are *not* the same.
                                    Note: pipeline_name comes is defined in the `template_path` and ultimately comes from Python pipeline definition.
                                    If provided, must be unique.
            labels (Optional[dict[str, str]]): Labels to associate with the run.
            experiment (Optional[str]): The name of the experiment to associate the run with.
        Returns:
            The PipelineJob created.
        """
        job = aiplatform.PipelineJob(
            display_name=display_name,
            template_path=template_path.uri,
            parameter_values=run_keyword_args,
            job_id=job_id,
            project=self._project,
            location=self._location,
            labels=labels,
        )
        job.submit(service_account=self._service_account, experiment=experiment)
        logger.info(f"Created run: {job.resource_name}")
        if experiment:
            logger.info(
                f"Associated run {job.resource_name} with experiment: {experiment}"
            )
        if labels:
            logger.info(f"Associated run {job.resource_name} with labels: {labels}")

        return job

    def get_pipeline_job_from_job_name(self, job_name: str) -> aiplatform.PipelineJob:
        """Fetches the pipeline job with the given job name."""
        return aiplatform.PipelineJob.get(
            f"projects/{self._project}/locations/{self._location}/pipelineJobs/{job_name}"
        )

    @staticmethod
    def get_pipeline_run_url(project: str, location: str, job_name: str) -> str:
        """Returns the URL for the pipeline run."""
        return f"https://console.cloud.google.com/vertex-ai/locations/{location}/pipelines/runs/{job_name}?project={project}"

    @staticmethod
    def wait_for_run_completion(
        resource_name: str,
        timeout: float = DEFAULT_PIPELINE_TIMEOUT_S,
        polling_period_s: int = 60,
    ) -> None:
        """
        Waits for a run to complete.

        Args:
            resource_name (str): The resource name of the run.
            timeout (float): The maximum time to wait for the run to complete, in seconds. Defaults to 7200.
            polling_period_s (int): The time to wait between polling the run status, in seconds. Defaults to 60.
        Returns:
            None
        """
        start_time = time.time()
        logger.info(f"Waiting for run to complete: {resource_name}")
        run = aiplatform.PipelineJob.get(resource_name=resource_name)
        while start_time + timeout > time.time():
            # Note that accesses to `run.state` cause a network call under the hood.
            # We should be careful with accessing this too frequently, and "cache"
            # the state if we need to access it multiple times in short succession.
            state = run.state
            logger.info(
                f"Run {resource_name} in state: {state.name if state else state}"
            )
            if state == aiplatform.gapic.PipelineState.PIPELINE_STATE_SUCCEEDED:
                logger.info("Vertex AI finished with status Succeeded!")
                return
            elif state in (
                aiplatform.gapic.PipelineState.PIPELINE_STATE_FAILED,
                aiplatform.gapic.PipelineState.PIPELINE_STATE_CANCELLED,
            ):
                logger.warning(f"Vertex AI run stopped with status: {state.name}.")
                logger.warning(
                    f"See run at: {VertexAIService.get_pipeline_run_url(run.project, run.location, run.name)}"
                )
                raise RuntimeError(f"Vertex AI run stopped with status: {state.name}.")
            time.sleep(polling_period_s)

        else:
            logger.warning("Timeout reached. Stopping the run.")
            logger.warning(
                f"See run at: {VertexAIService.get_pipeline_run_url(run.project, run.location, run.name)}"
            )
            run.cancel()
            raise RuntimeError(
                f"Vertex AI run stopped with status: {run.state}. "
                f"Please check the Vertex AI page to trace down the error."
            )


def _create_machine_spec(job_config: VertexAiJobConfig) -> MachineSpec:
    """Get the machine spec for a job config."""
    machine_spec = MachineSpec(
        machine_type=job_config.machine_type,
        accelerator_type=job_config.accelerator_type,
        accelerator_count=job_config.accelerator_count,
    )
    return machine_spec


def _create_container_spec(
    job_config: VertexAiJobConfig, env_vars: list[env_var.EnvVar]
) -> ContainerSpec:
    """Get the container spec for a job config."""
    container_spec = ContainerSpec(
        image_uri=job_config.container_uri,
        command=job_config.command,
        args=job_config.args,
        env=env_vars,
    )
    return container_spec


def _create_disk_spec(job_config: VertexAiJobConfig) -> DiskSpec:
    """Get the disk spec for a job config."""
    disk_spec = DiskSpec(
        boot_disk_type=job_config.boot_disk_type,
        boot_disk_size_gb=job_config.boot_disk_size_gb,
    )
    return disk_spec
