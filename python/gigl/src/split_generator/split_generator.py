import argparse
import datetime
from typing import Optional

import gigl.common.utils.local_fs as local_fs_utils
import gigl.env.dep_constants as dep_constants
import gigl.src.common.constants.gcs as gcs_constants
from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.constants import SPARK_35_TFRECORD_JAR_GCS_PATH
from gigl.common.logger import Logger
from gigl.common.metrics.decorators import flushes_metrics, profileit
from gigl.common.utils.gcs import GcsUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.constants.metrics import TIMER_SPLIT_GENERATOR_S
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
    initialize_metrics,
)
from gigl.src.common.utils.spark_job_manager import (
    DataprocClusterInitData,
    SparkJobManager,
)

logger = Logger()

MAX_JOB_DURATION = datetime.timedelta(
    hours=4
)  # Allowed max job duration for SplitGen job -- for MAU workload


class SplitGenerator:
    """
    GiGL Component that reads localized subgraph samples produced by Subgraph Sampler, and executes logic to split the data into training, validation and test sets.
    The semantics of which nodes and edges end up in which data split depends on the particular semantics of the splitting strategy.
    """

    def __prepare_staging_paths(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        applied_task_identifier: AppliedTaskIdentifier,
    ) -> None:
        """
        Clean up paths that Split Generator would be writing to in order to avoid clobbering of data.
        These paths are inferred from the GbmlConfig and the
        :return:
        """

        gcs_utils = GcsUtils()

        split_gen_applied_task_dir = (
            gcs_constants.get_split_generator_assets_temp_gcs_path(
                applied_task_identifier=applied_task_identifier
            )
        )
        gcs_utils.delete_files_in_bucket_dir(gcs_path=split_gen_applied_task_dir)

        split_gen_output_paths = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper.get_output_paths()
        )
        for output_path in split_gen_output_paths:
            if isinstance(output_path, GcsUri):
                gcs_utils.delete_files_in_bucket_dir(gcs_path=output_path)
            elif isinstance(output_path, LocalUri):
                local_fs_utils.delete_local_directory(local_path=output_path)
            else:
                raise ValueError(
                    f"Unsupported path type: found path {output_path} of type {type(output_path)}"
                )

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_SPLIT_GENERATOR_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        cluster_name: Optional[str] = None,
        debug_cluster_owner_alias: Optional[str] = None,
        skip_cluster_delete: bool = False,
    ):
        # Default to spark 35
        use_spark35 = True
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=task_config_uri
            )
        )
        if (
            gbml_config_pb_wrapper.shared_config.should_skip_training
            and gbml_config_pb_wrapper.shared_config.should_skip_model_evaluation
        ):
            logger.info(
                "We want to skip training and evaluation, so there is no need to run split generator."
                + "Exiting."
            )
            return

        self.__prepare_staging_paths(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            applied_task_identifier=applied_task_identifier,
        )

        resource_config = get_resource_config(resource_config_uri=resource_config_uri)

        gcs_utils = GcsUtils()

        main_jar_file_uri: LocalUri = dep_constants.get_jar_file_uri(
            component=GiGLComponents.SplitGenerator,
            use_spark35=use_spark35,
        )

        main_jar_file_name = main_jar_file_uri.uri.split("/")[-1]

        sgn_jar_file_gcs_path = GcsUri.join(
            gcs_constants.get_split_generator_assets_temp_gcs_path(
                applied_task_identifier=applied_task_identifier
            ),
            main_jar_file_name,
        )

        logger.info(f"Uploading local jar file to {sgn_jar_file_gcs_path}")
        gcs_utils.upload_files_to_gcs(
            {main_jar_file_uri: sgn_jar_file_gcs_path}, parallel=False
        )

        if isinstance(resource_config_uri, LocalUri):
            resource_config_gcs_path: GcsUri = GcsUri.join(
                gcs_constants.get_applied_task_temp_gcs_path(
                    applied_task_identifier=applied_task_identifier
                ),
                "resource_config.yaml",
            )
            logger.info(
                f"Uploading Local Resource Config to: {resource_config_gcs_path}"
            )
            gcs_utils.upload_files_to_gcs(
                {resource_config_uri: resource_config_gcs_path}, parallel=False
            )
        else:
            resource_config_gcs_path = GcsUri(resource_config_uri.uri)

        if not cluster_name:
            cluster_name = f"split_{applied_task_identifier}"
        cluster_name = cluster_name.replace("_", "-")[:50]
        if cluster_name.endswith("-"):
            cluster_name = cluster_name[:-1] + "z"

        dataproc_helper = SparkJobManager(
            project=resource_config.project,
            region=resource_config.region,
            cluster_name=cluster_name,
        )

        cluster_init_data = DataprocClusterInitData(
            project=resource_config.project,
            region=resource_config.region,
            service_account=resource_config.service_account_email,
            temp_assets_bucket=str(resource_config.temp_assets_regional_bucket_path),
            cluster_name=cluster_name,
            machine_type=resource_config.split_generator_config.machine_type,
            num_workers=resource_config.split_generator_config.num_replicas,
            num_local_ssds=resource_config.split_generator_config.num_local_ssds,
            debug_cluster_owner_alias=debug_cluster_owner_alias,
            is_debug_mode=skip_cluster_delete or debug_cluster_owner_alias,  # type: ignore
            labels=resource_config.get_resource_labels(
                component=GiGLComponents.SplitGenerator
            ),
        )

        dataproc_helper.create_dataproc_cluster(
            cluster_init_data=cluster_init_data, use_spark35=use_spark35
        )

        dataproc_helper.submit_and_wait_scala_spark_job(
            main_jar_file_uri=sgn_jar_file_gcs_path.uri,
            max_job_duration=MAX_JOB_DURATION,
            runtime_args=[
                applied_task_identifier,
                task_config_uri.uri,
                resource_config_gcs_path.uri,
            ],
            extra_jar_file_uris=[SPARK_35_TFRECORD_JAR_GCS_PATH],
            use_spark35=use_spark35,
        )
        if not skip_cluster_delete:
            logger.info(
                f"skip_cluster_delete marked to {skip_cluster_delete}; will delete cluster"
            )
            dataproc_helper.delete_cluster()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to generate embeddings from a GBML model"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="Gbml config uri",
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
    )
    parser.add_argument(
        "--cluster_name",
        type=str,
        help="Optional param if you want to re-use a cluster for continous development purposes."
        + "Otherwise, a cluster name will automatically be generated based on job_name",
        required=False,
    )
    parser.add_argument(
        "--skip_cluster_delete",
        action="store_true",
        help="Provide flag to skip automatic cleanup of dataproc cluster. This way you can re-use the cluster for development purposes",
        default=False,
    )
    parser.add_argument(
        "--debug_cluster_owner_alias",
        type=str,
        help="debug_cluster_owner_alias",
        required=False,
    )
    args = parser.parse_args()

    applied_task_identifier = AppliedTaskIdentifier(args.job_name)
    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
    if not args.job_name or not args.task_config_uri or not args.resource_config_uri:
        raise RuntimeError("Missing command-line arguments")

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    split_generator = SplitGenerator()
    split_generator.run(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        cluster_name=args.cluster_name,
        debug_cluster_owner_alias=args.debug_cluster_owner_alias,
        skip_cluster_delete=args.skip_cluster_delete,
    )
