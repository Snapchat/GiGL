import concurrent.futures
import datetime
from typing import List, Optional

import google.api_core.exceptions
import google.cloud.dataproc_v1 as dataproc_v1
from google.api_core.future.polling import POLLING_PREDICATE
from google.api_core.retry import Retry
from google.cloud.dataproc_v1.services.job_controller.pagers import ListJobsPager
from google.cloud.dataproc_v1.types import JobStatus

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.utils.retry import retry

logger = Logger()

_DATAPROC_JOB_URL_FMT = "https://console.cloud.google.com/dataproc/jobs/{job_id}"


def _log_spark_cluster(job_id: str) -> None:
    """Logs a URL for viewing Spark cluster jobs info."""
    log_url = _DATAPROC_JOB_URL_FMT.format(
        job_id=job_id,
    )
    logger.info(f"Spark cluster job info is located at: {log_url}")


class DataprocService:
    """
    A service class that provides methods to interact with Google Cloud Dataproc.

    Args:
        project_id (str): The ID of the Google Cloud project.
        region (str): The region where the Dataproc cluster is located.
    """

    def __init__(self, project_id: str, region: str) -> None:
        self.project_id = project_id
        self.region = region
        client_options = {"api_endpoint": f"{self.region}-dataproc.googleapis.com:443"}
        self.cluster_client = dataproc_v1.ClusterControllerClient(
            client_options=client_options
        )
        self.job_client = dataproc_v1.JobControllerClient(client_options=client_options)

    def does_cluster_exist(
        self,
        cluster_name: str,
    ) -> bool:
        """
        Checks if a cluster with the given name exists.

        Args:
            cluster_name (str): The name of the cluster to check.

        Returns:
            bool: True if the cluster exists, False otherwise.
        """
        request = dataproc_v1.GetClusterRequest(
            project_id=self.project_id,
            region=self.region,
            cluster_name=cluster_name,
        )
        does_cluster_exist = False
        try:
            response = self.cluster_client.get_cluster(request=request)
            assert (
                response.cluster_name == cluster_name
            ), f"Tried fetching {cluster_name}, got {response}"
            does_cluster_exist = True
        except google.api_core.exceptions.NotFound as e:
            logger.info(e)
        return does_cluster_exist

    @retry(
        exception_to_check=google.api_core.exceptions.ServiceUnavailable,  # retry on Dataproc resource exhaustion exception
        tries=6,
        delay_s=1200,  # wait 20 mins between retries
        backoff=1,
    )  # retry for 2 hours
    def create_cluster(self, cluster_spec: dict) -> None:
        """Creates a dataproc cluster

        Args:
            cluster_spec (dict): A dictionary containing the cluster specification.
                For more details, refer to the documentation at:
                https://cloud.google.com/python/docs/reference/dataproc/latest/google.cloud.dataproc_v1.types.Cluster

        Returns:
            None
        """
        logger.info(f"Creating cluster: {cluster_spec}")
        request = {
            "project_id": self.project_id,
            "region": self.region,
            "cluster": cluster_spec,
        }
        operation = self.cluster_client.create_cluster(request=request)
        result = operation.result()
        cluster_name = result.cluster_name
        logger.info(f"Cluster created successfully: {cluster_name}")
        running_job_ids = self.get_running_job_ids_on_cluster(cluster_name)
        logger.info(f"Running jobs: {running_job_ids}")
        if len(running_job_ids) > 1:
            logger.warning(
                f"Found {running_job_ids} jobs on the {cluster_name} cluster. Expected only one. Jobs: {running_job_ids}"
            )

        for job in running_job_ids:
            _log_spark_cluster(job)

    def delete_cluster(self, cluster_name: str) -> None:
        """
        Deletes a cluster with the given name.

        Args:
            cluster_name (str): The name of the cluster to delete.

        Returns:
            None
        """
        operation = self.cluster_client.delete_cluster(
            request={
                "project_id": self.project_id,
                "region": self.region,
                "cluster_name": cluster_name,
            }
        )
        result = operation.result()
        logger.info(result)

    def submit_and_wait_scala_spark_job(
        self,
        cluster_name: str,
        max_job_duration: datetime.timedelta,
        main_jar_file_uri: Uri,
        runtime_args: Optional[List[str]] = [],
        extra_jar_file_uris: Optional[List[str]] = [],
        properties: Optional[dict] = {},
        fail_if_job_already_running_on_cluster: Optional[bool] = True,
    ) -> None:
        """
        Submits a Scala Spark job to a Dataproc cluster and waits for its completion.

        Args:
            cluster_name (str): The name of the Dataproc cluster.
            max_job_duration (datetime.timedelta): The maximum duration allowed for the job to run.
            main_jar_file_uri (Uri): The URI of the main jar file for the Spark job.
            runtime_args (Optional[List[str]]: Additional runtime arguments for the Spark job. Defaults to [].
            extra_jar_file_uris (Optional[List[str]]: Additional jar file URIs for the Spark job. Defaults to [].
            fail_if_job_already_running_on_cluster (Optional[bool]): Whether to fail if there are already running jobs on the cluster. Defaults to True.

        Returns:
            None
        """
        job = {
            "placement": {"cluster_name": cluster_name},
            # arguments: https://cloud.google.com/python/docs/reference/dataproc/latest/google.cloud.dataproc_v1.types.SparkJob
            "spark_job": {
                "args": runtime_args,
                "main_jar_file_uri": main_jar_file_uri,
                "jar_file_uris": extra_jar_file_uris,
                "properties": properties,
            },
        }
        if fail_if_job_already_running_on_cluster:
            running_job_ids = self.get_running_job_ids_on_cluster(
                cluster_name=cluster_name
            )
            num_running_jobs = len(running_job_ids)
            assert (
                num_running_jobs == 0
            ), f"Found '{num_running_jobs}' running jobs for cluster '{cluster_name}'. Cannot submit a new job."

        operation = self.job_client.submit_job_as_operation(
            request={"project_id": self.project_id, "region": self.region, "job": job}
        )
        current_job_id = operation.metadata.job_id

        _POLLING = Retry(
            predicate=POLLING_PREDICATE,  # retries.if_exception_type(_OperationNotComplete)
            initial=1.0,  # seconds
            maximum=60.0,  # seconds
            multiplier=1.5,
            timeout=60 * 60 * 12,  # 12 hours
        )
        logger.info(f"Will run and wait for the following job: {job}")

        try:
            response = operation.result(
                polling=_POLLING, timeout=int(max_job_duration.total_seconds())
            )

            # Note, this logs important information about the job like placement, cluster uuid, etc. Which is being used by other scripts to
            # lookup information about running jobs. Do not remove.
            # TODO: (svij-sc), likely we want to surface this information up in a more structured way then having to parse the logs.
            logger.info(response)
            _log_spark_cluster(response.job_uuid)

        except concurrent.futures.TimeoutError:
            request = dataproc_v1.CancelJobRequest(
                project_id=self.project_id,
                region=self.region,
                job_id=current_job_id,
            )
            resp = self.job_client.cancel_job(request=request)
            raise concurrent.futures.TimeoutError(
                f"Cancelled job with id: '{current_job_id}' on cluster: '{cluster_name}' since it was running longer than max job duration: '{max_job_duration}'"
            )

    def get_submitted_job_ids(self, cluster_name: str) -> List[str]:
        """
        Retrieves the job IDs of all active jobs submitted to a specific cluster.

        Args:
            cluster_name (str): The name of the cluster.

        Returns:
            List[str]: The job IDs of all active jobs submitted to the cluster.
        """
        submitted_jobs: ListJobsPager = self.job_client.list_jobs(
            project_id=self.project_id,
            region=self.region,
            filter=f"clusterName={cluster_name} AND status.state=ACTIVE",
        )
        job_ids = [job.reference.job_id for job in submitted_jobs.jobs]
        return job_ids

    def get_running_job_ids_on_cluster(self, cluster_name: str) -> List[str]:
        """
        Retrieves the running job IDs on the specified cluster.

        Args:
            cluster_name (str): The name of the cluster.

        Returns:
            List[str]: The running job IDs on the cluster.
        """
        job_ids = self.get_submitted_job_ids(cluster_name=cluster_name)
        running_job_ids = []
        for job_id in job_ids:
            job_status = self.job_client.get_job(
                project_id=self.project_id, region=self.region, job_id=job_id
            )
            if job_status and job_status.status.state == JobStatus.State.RUNNING:
                running_job_ids.append(job_id)

        return running_job_ids
