import argparse
import datetime
import os
from distutils.util import strtobool
from typing import Optional, Sequence

import gigl.env.dep_constants as dep_constants
import gigl.src.common.constants.gcs as gcs_constants
import gigl.src.common.constants.metrics as metrics_constants
from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.constants import (
    SPARK_31_TFRECORD_JAR_GCS_PATH,
    SPARK_35_TFRECORD_JAR_GCS_PATH,
)
from gigl.common.logger import Logger
from gigl.common.metrics.decorators import flushes_metrics, profileit
from gigl.common.utils import os_utils
from gigl.common.utils.gcs import GcsUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.constants.metrics import TIMER_SUBGRAPH_SAMPLER_S
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
    initialize_metrics,
)
from gigl.src.common.utils.spark_job_manager import (
    DataprocClusterInitData,
    SparkJobManager,
)
from gigl.src.subgraph_sampler.lib.ingestion_protocol import BaseIngestion

logger = Logger()

MAX_JOB_DURATION = datetime.timedelta(
    hours=5
)  # Allowed max job duration for SGS job -- for MAU workload


class SubgraphSampler:
    """
    GiGL Component that generates k-hop localized subgraphs for each node in the graph using Spark/Scala running on Dataproc.
    """

    def __prepare_staging_paths(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> None:
        # Clear paths that Subgraph Sampler would be writing to, to avoid clobbering of data.
        # Some of these paths are inferred from paths specified in the GbmlConfig.
        # Other paths are inferred from the AppliedTaskIdentifier.
        logger.info("Preparing staging paths for Subgraph Sampler...")
        paths_to_delete = (
            [
                gcs_constants.get_subgraph_sampler_root_dir(
                    applied_task_identifier=applied_task_identifier
                )
            ]
            + gbml_config_pb_wrapper.flattened_graph_metadata_pb_wrapper.get_output_paths()
        )
        file_loader = FileLoader()
        file_loader.delete_files(uris=paths_to_delete)
        logger.info("Staging paths for Subgraph Sampler prepared.")

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_SUBGRAPH_SAMPLER_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        cluster_name: Optional[str] = None,
        debug_cluster_owner_alias: Optional[str] = None,
        custom_worker_image_uri: Optional[str] = None,
        skip_cluster_delete: bool = False,
        additional_spark35_jar_file_uris: Sequence[Uri] = (),
    ):
        resource_config = get_resource_config(resource_config_uri=resource_config_uri)
        gbml_config_pb_wrapper: GbmlConfigPbWrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=task_config_uri
            )
        )
        self.__prepare_staging_paths(
            applied_task_identifier=applied_task_identifier,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )

        use_graph_db = (
            gbml_config_pb_wrapper.dataset_config.subgraph_sampler_config.HasField(
                "graph_db_config"
            )
        )
        use_spark35: bool = bool(
            strtobool(
                gbml_config_pb_wrapper.dataset_config.subgraph_sampler_config.experimental_flags.get(
                    "use_spark35_runner", "False"
                )
            )
        )
        if use_graph_db:
            # Run spark35 runner if we're using graphdb version of the subgraph sampler
            logger.info(
                "Will default to using Spark 3.5 runner for Subgraph Sampler since graph_db_config is set"
            )
            use_spark35 = True

            metrics_service = get_metrics_service_instance()
            if metrics_service is not None:
                metrics_service.add_count(
                    metric_name=metrics_constants.COUNT_SGS_USES_GRAPHDB, count=1
                )

        should_ingest_into_graph_db: bool = use_graph_db and getattr(
            gbml_config_pb_wrapper.dataset_config.subgraph_sampler_config.graph_db_config,
            "graph_db_ingestion_cls_path",
        )

        if should_ingest_into_graph_db:
            graph_db_config = (
                gbml_config_pb_wrapper.dataset_config.subgraph_sampler_config.graph_db_config
            )

            graph_db_ingestion_config_cls_str: str = (
                graph_db_config.graph_db_ingestion_cls_path
            )

            graph_db_ingestion_args = graph_db_config.graph_db_ingestion_args  # type: ignore
            graph_db_args = graph_db_config.graph_db_args
            all_graph_db_args = {**graph_db_ingestion_args, **graph_db_args}

            graph_db_ingestion_cls = os_utils.import_obj(
                obj_path=graph_db_ingestion_config_cls_str
            )

            try:
                graph_db_ingestion_config: BaseIngestion = graph_db_ingestion_cls(
                    **all_graph_db_args
                )
            except Exception as e:
                logger.error(
                    f"Could not instantiate class {graph_db_ingestion_cls} with args {graph_db_ingestion_args}"
                )
                raise e

            logger.info(
                f"Instantiated {graph_db_ingestion_cls} with args {graph_db_ingestion_args}"
            )

            logger.info("Running ingestion...")
            graph_db_ingestion_config.ingest(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                resource_config_uri=resource_config_uri,
                applied_task_identifier=applied_task_identifier,
                custom_worker_image_uri=custom_worker_image_uri,
            )

            logger.info("Ingestion complete. Cleaning up...")
            graph_db_ingestion_config.clean_up()

        # must match pattern (?:[a-z](?:[-a-z0-9]{0,49}[a-z0-9])?)
        if not cluster_name:
            cluster_name = f"sgs_{applied_task_identifier}"
        cluster_name = cluster_name.replace("_", "-")[:50]
        if cluster_name.endswith("-"):
            cluster_name = cluster_name[:-1] + "Z"

        gcs_utils = GcsUtils()

        main_jar_file_uri: LocalUri = dep_constants.get_jar_file_uri(
            component=GiGLComponents.SubgraphSampler, use_spark35=use_spark35
        )
        logger.info(f"Using main jar file: {main_jar_file_uri}")
        main_jar_file_name: str = main_jar_file_uri.uri.split("/")[-1]
        jar_file_local_dir: str = os.path.dirname(main_jar_file_uri.uri)
        logger.info(f"Jar file local dir: {jar_file_local_dir}")

        jar_file_gcs_bucket: GcsUri = gcs_constants.get_subgraph_sampler_root_dir(
            applied_task_identifier=applied_task_identifier
        )
        jars_to_upload: dict[Uri, GcsUri] = {
            main_jar_file_uri: GcsUri.join(jar_file_gcs_bucket, main_jar_file_name)
        }

        # Since Spark 3.5 and Spark 3.1 are using different versions of Scala
        # We need to pass the correct extra jar file to the Spark cluster,
        # Otherwise, we may see some errors like:
        # java.io.InvalidClassException; local class incompatible: stream classdesc serialVersionUID = -1, local class serialVersionUID = 2
        if use_spark35:
            for jar_uri in additional_spark35_jar_file_uris:
                jars_to_upload[jar_uri] = GcsUri.join(
                    jar_file_gcs_bucket, jar_uri.get_basename()
                )

        sgs_jar_file_gcs_path = GcsUri.join(
            jar_file_gcs_bucket,
            main_jar_file_name,
        )

        logger.info(f"Uploading jar files {jars_to_upload}")
        FileLoader().load_files(source_to_dest_file_uri_map=jars_to_upload)

        extra_jar_file_uris = [
            jars_to_upload[jar].uri
            for jar in jars_to_upload
            if jar != main_jar_file_uri
        ]
        if use_spark35:
            extra_jar_file_uris.append(SPARK_35_TFRECORD_JAR_GCS_PATH)
        else:
            extra_jar_file_uris.append(SPARK_31_TFRECORD_JAR_GCS_PATH)

        logger.info(f"Will add the following jars to all jobs: {extra_jar_file_uris}")

        resource_config_gcs_path: GcsUri
        task_config_gcs_path: GcsUri
        file_loader = FileLoader()
        if not isinstance(resource_config_uri, GcsUri):
            resource_config_gcs_path = GcsUri.join(
                gcs_constants.get_applied_task_temp_gcs_path(
                    applied_task_identifier=applied_task_identifier
                ),
                "resource_config.yaml",
            )
            logger.info(
                f"Uploading resource config : {resource_config_uri} to gcs: {resource_config_gcs_path}"
            )
            file_loader.load_file(
                file_uri_src=resource_config_uri, file_uri_dst=resource_config_gcs_path
            )
        else:
            resource_config_gcs_path = resource_config_uri
        if not isinstance(task_config_uri, GcsUri):
            task_config_gcs_path = GcsUri.join(
                gcs_constants.get_applied_task_temp_gcs_path(
                    applied_task_identifier=applied_task_identifier
                ),
                "task_config.yaml",
            )
            logger.info(
                f"Uploading task config : {task_config_uri} to gcs: {task_config_gcs_path}"
            )
            file_loader.load_file(
                file_uri_src=task_config_uri, file_uri_dst=task_config_gcs_path
            )
        else:
            task_config_gcs_path = task_config_uri

        logger.info(
            f"Using resource config: {resource_config_gcs_path} and task config: {task_config_gcs_path}"
        )
        dataproc_helper = SparkJobManager(
            project=resource_config.project,
            region=resource_config.region,
            cluster_name=cluster_name,
        )

        cluster_init_data = DataprocClusterInitData(
            project=resource_config.project,
            region=resource_config.region,
            service_account=resource_config.service_account_email,
            cluster_name=cluster_name,
            machine_type=resource_config.subgraph_sampler_config.machine_type,
            temp_assets_bucket=str(resource_config.temp_assets_regional_bucket_path),
            num_workers=resource_config.subgraph_sampler_config.num_replicas,
            num_local_ssds=resource_config.subgraph_sampler_config.num_local_ssds,
            debug_cluster_owner_alias=debug_cluster_owner_alias,
            is_debug_mode=skip_cluster_delete or bool(debug_cluster_owner_alias),
            labels=resource_config.get_resource_labels(
                component=GiGLComponents.SubgraphSampler
            ),
        )

        if use_spark35:
            logger.warning(
                "You are using Spark 3.5 runner for Subgraph Sampler, not all features are supported yet."
            )

        dataproc_helper.create_dataproc_cluster(
            cluster_init_data=cluster_init_data,
            use_spark35=use_spark35,
        )

        dataproc_helper.submit_and_wait_scala_spark_job(
            main_jar_file_uri=sgs_jar_file_gcs_path.uri,
            max_job_duration=MAX_JOB_DURATION,
            runtime_args=[
                task_config_gcs_path.uri,
                applied_task_identifier,
                resource_config_gcs_path.uri,
            ],
            extra_jar_file_uris=extra_jar_file_uris,
            use_spark35=use_spark35,
        )

        if not skip_cluster_delete:
            logger.info(
                f"skip_cluster_delete marked to {skip_cluster_delete}; will delete cluster"
            )
            dataproc_helper.delete_cluster()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to sample subgraphs from preprocessed graph/feature data"
        + "Using the subgraphs, generates samples that can be consumed by rest of the pipeline."
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
        required=True,
    )
    parser.add_argument(
        "--task_config_uri",
        type=str,
        help="Gbml frozen config uri",
        required=True,
    )
    parser.add_argument(
        "--resource_config_uri",
        type=str,
        help="Runtime argument for resource and env specifications of each component",
        required=True,
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
    parser.add_argument(
        "--custom_worker_image_uri",
        type=str,
        help="Docker image to use for the worker harness in dataflow jobs (optional)",
        required=False,
    )
    parser.add_argument(
        "--additional_spark35_jar_file_uris",
        action="append",
        type=str,
        required=False,
        default=[],
        help="Additional URIs to be added to the Spark cluster.",
    )

    args = parser.parse_args()

    if not args.job_name or not args.task_config_uri or not args.resource_config_uri:
        raise RuntimeError(
            f"Missing command-line arguments, expected all of [job_name, task_config_uri, resource_config_uri]. Received: {args}"
        )

    ati = AppliedTaskIdentifier(args.job_name)
    task_config_uri = UriFactory.create_uri(uri=args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(uri=args.resource_config_uri)
    applied_task_identifier = AppliedTaskIdentifier(args.job_name)
    custom_worker_image_uri = args.custom_worker_image_uri

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    sgs = SubgraphSampler()
    sgs.run(
        applied_task_identifier=ati,
        task_config_uri=task_config_uri,
        cluster_name=args.cluster_name,
        debug_cluster_owner_alias=args.debug_cluster_owner_alias,
        skip_cluster_delete=args.skip_cluster_delete,
        resource_config_uri=resource_config_uri,
        custom_worker_image_uri=custom_worker_image_uri,
        # Filter out empty strings which kfp *may* add...
        additional_spark35_jar_file_uris=[
            UriFactory.create_uri(jar)
            for jar in args.additional_spark35_jar_file_uris
            if jar
        ],
    )
