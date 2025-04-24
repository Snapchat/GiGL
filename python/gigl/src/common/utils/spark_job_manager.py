import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

from google.cloud.dataproc_v1.types import (
    Cluster,
    ClusterConfig,
    DiskConfig,
    EndpointConfig,
    GceClusterConfig,
    InstanceGroupConfig,
    LifecycleConfig,
    NodeInitializationAction,
    SoftwareConfig,
)
from google.protobuf.duration_pb2 import Duration

from gigl.common import GcsUri, Uri
from gigl.common.logger import Logger
from gigl.common.services.dataproc import DataprocService

IDLE_TTL_DEFAULT_S = 600  # Auto delete cluster after 10 mins of idle time (i.e. no job is running on cluster)
IDLE_TTL_DEV_DEFAULT_S = 36_000  # Auto delete cluster after 10 hours of idle time (i.e. no job is running on cluster)
logger = Logger()


@dataclass
class DataprocClusterInitData:
    project: str
    region: str
    service_account: str
    temp_assets_bucket: str
    cluster_name: str
    num_workers: int
    machine_type: str
    num_local_ssds: int
    is_debug_mode: bool
    debug_cluster_owner_alias: Optional[str] = None
    init_script_uri: Optional[GcsUri] = None
    labels: Optional[Dict[str, str]] = None


class SparkJobManager:
    def __init__(self, project: str, region: str, cluster_name: str):
        self.__dataproc_service = DataprocService(project_id=project, region=region)
        self.__cluster_name = SparkJobManager.get_sanitized_dataproc_cluster_name(
            cluster_name=cluster_name
        )

    def create_dataproc_cluster(
        self, cluster_init_data: DataprocClusterInitData, use_spark35: bool = False
    ):
        init_actions = []
        metadata = {}

        if cluster_init_data.init_script_uri is not None:
            logger.info(
                f"Adding node init action to run following executable on every node: {cluster_init_data.init_script_uri}"
            )
            init_action = NodeInitializationAction(
                executable_file=cluster_init_data.init_script_uri,
                execution_timeout=Duration(seconds=300),  # 5 mins
            )
            init_actions.append(init_action)

        if cluster_init_data.debug_cluster_owner_alias is not None:
            logger.info(
                f"Trying to setup a debug cluster with cluster_owner: {cluster_init_data.debug_cluster_owner_alias}"
            )
            metadata["OWNER"] = cluster_init_data.debug_cluster_owner_alias

        idle_ttl_s = SparkJobManager.__get_cluster_idle_time(
            is_debug_mode=cluster_init_data.is_debug_mode
        )

        disk_config = DiskConfig(
            boot_disk_type="pd-standard",
            boot_disk_size_gb=500,
            num_local_ssds=cluster_init_data.num_local_ssds,
            local_ssd_interface="nvme",
        )

        master_config = InstanceGroupConfig(
            num_instances=1,
            machine_type_uri=cluster_init_data.machine_type,
            disk_config=disk_config,
        )

        worker_config = InstanceGroupConfig(
            num_instances=cluster_init_data.num_workers,
            machine_type_uri=cluster_init_data.machine_type,
            disk_config=disk_config,
        )

        gce_cluster_config: GceClusterConfig
        image_version: str
        if use_spark35:
            # https://cloud.google.com/dataproc/docs/concepts/versioning/dataproc-release-2.2
            image_version = "2.2.19-ubuntu22"
            gce_cluster_config = GceClusterConfig(
                service_account=cluster_init_data.service_account,
                service_account_scopes=[
                    "https://www.googleapis.com/auth/cloud-platform"
                ],
                internal_ip_only=False,
                metadata=metadata,
            )

        else:
            # https://cloud.google.com/dataproc/docs/concepts/versioning/dataproc-release-2.0
            image_version = "2.0.47-ubuntu18"
            gce_cluster_config = GceClusterConfig(
                service_account=cluster_init_data.service_account,
                service_account_scopes=[
                    "https://www.googleapis.com/auth/cloud-platform"
                ],
                tags=["default-allow-internal"],
                metadata=metadata,
            )

        endpoint_config = EndpointConfig(enable_http_port_access=True)
        software_config = SoftwareConfig(
            image_version=image_version,
            optional_components=["JUPYTER"],
            properties={"dataproc:dataproc.monitoring.stackdriver.enable": "true"},
        )

        lifecycle_config = LifecycleConfig(idle_delete_ttl=Duration(seconds=idle_ttl_s))

        bucket = cluster_init_data.temp_assets_bucket.replace("gs://", "")

        config = ClusterConfig(
            config_bucket=bucket,
            temp_bucket=bucket,
            master_config=master_config,
            worker_config=worker_config,
            gce_cluster_config=gce_cluster_config,
            endpoint_config=endpoint_config,
            software_config=software_config,
            lifecycle_config=lifecycle_config,
            initialization_actions=init_actions,
        )

        cluster_spec = Cluster(
            project_id=cluster_init_data.project,
            cluster_name=cluster_init_data.cluster_name,
            config=config,
            labels=cluster_init_data.labels,
        )

        if not self.__dataproc_service.does_cluster_exist(
            cluster_name=cluster_init_data.cluster_name
        ):
            logger.info(
                f"Will try to create cluster {cluster_init_data.cluster_name} with spec: {cluster_spec}"
            )
            self.__dataproc_service.create_cluster(cluster_spec=cluster_spec)
        else:
            logger.info(
                f"Cluster ({cluster_init_data.cluster_name}) already exists, skipping creation..."
            )

    def submit_and_wait_scala_spark_job(
        self,
        main_jar_file_uri: Uri,
        max_job_duration: datetime.timedelta,
        runtime_args: List[str] = [],
        extra_jar_file_uris: List[str] = [],
        use_spark35: bool = False,
    ):
        # The DataprocFileOutputCommitter feature is an enhanced version of the open source FileOutputCommitter.
        # It enables concurrent writes by Apache Spark jobs to an output location. We have seen that introducing
        # this committer can lead to a significant decrease in GCS write issues.
        # More info: https://cloud.google.com/dataproc/docs/guides/dataproc-fileoutput-committer
        # Only work with more recent versions of Dataproc images
        properties = (
            {
                "spark.hadoop.mapreduce.outputcommitter.factory.class": "org.apache.hadoop.mapreduce.lib.output.DataprocFileOutputCommitterFactory",
                "spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs": "false",
            }
            if use_spark35
            else {}
        )
        self.__dataproc_service.submit_and_wait_scala_spark_job(
            cluster_name=self.__cluster_name,
            max_job_duration=max_job_duration,
            main_jar_file_uri=main_jar_file_uri,
            runtime_args=runtime_args,
            extra_jar_file_uris=extra_jar_file_uris,
            properties=properties,
        )

    def delete_cluster(self):
        self.__dataproc_service.delete_cluster(
            cluster_name=self.__cluster_name,
        )

    # must match pattern (?:[a-z](?:[-a-z0-9]{0,49}[a-z0-9])?)
    @staticmethod
    def get_sanitized_dataproc_cluster_name(cluster_name: str):
        cluster_name = cluster_name.replace("_", "-")[:50]
        if cluster_name.endswith("-"):
            cluster_name = cluster_name[:-1] + "Z"
        return cluster_name

    @staticmethod
    def __get_cluster_idle_time(is_debug_mode):
        if is_debug_mode:
            return IDLE_TTL_DEV_DEFAULT_S
        else:
            return IDLE_TTL_DEFAULT_S
