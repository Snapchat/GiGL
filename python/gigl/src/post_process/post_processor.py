import argparse
import sys
import tempfile
import traceback
from typing import Optional

from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.metrics.decorators import flushes_metrics, profileit
from gigl.common.utils import os_utils
from gigl.common.utils.gcs import GcsUtils
from gigl.src.common.constants import gcs as gcs_constants
from gigl.src.common.constants.metrics import TIMER_POST_PROCESSOR_S
from gigl.src.common.translators.model_eval_metrics_translator import (
    EvalMetricsCollectionTranslator,
)
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.model_eval_metrics import EvalMetricsCollection
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.common.utils.metrics_service_provider import (
    get_metrics_service_instance,
    initialize_metrics,
)
from gigl.src.post_process.lib.base_post_processor import BasePostProcessor
from gigl.src.post_process.utils.unenumeration import unenumerate_all_inferred_bq_assets
from snapchat.research.gbml import gbml_config_pb2

logger = Logger()


class PostProcessor:
    def __run_post_process(
        self,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
        applied_task_identifier: AppliedTaskIdentifier,
    ):
        post_processor_cls_str: str = (
            gbml_config_pb.post_processor_config.post_processor_cls_path
        )
        kwargs = gbml_config_pb.post_processor_config.post_processor_args
        kwargs["applied_task_identifier"] = applied_task_identifier

        if post_processor_cls_str == "":
            logger.warning(
                "No post processor class path provided in config, will skip post processor"
            )
        else:
            try:
                post_processor_cls = os_utils.import_obj(post_processor_cls_str)
                post_processor: BasePostProcessor = post_processor_cls(**kwargs)
                assert isinstance(post_processor, BasePostProcessor)
                logger.info(
                    f"Instantiate class {post_processor_cls_str} with kwargs: {kwargs}"
                )
            except Exception as e:
                logger.error(
                    f"Could not instantiate class {post_processor_cls_str}: {e}"
                )
                raise e

            logger.info(
                f"Running user post processor class: {post_processor.__class__}, with config: {gbml_config_pb}"
            )
            post_processor_metrics: Optional[
                EvalMetricsCollection
            ] = post_processor.run_post_process(gbml_config_pb=gbml_config_pb)
            if post_processor_metrics is not None:
                self.__write_post_processor_metrics_to_uri(
                    model_eval_metrics=post_processor_metrics,
                    gbml_config_pb=gbml_config_pb,
                )

        # Run shared logic of cleaning up of assets considered temporary
        if gbml_config_pb.shared_config.should_skip_automatic_temp_asset_cleanup:
            logger.info(
                "Will skip automatic cleanup of temporary assets as `should_skip_automatic_temp_asset_cleanup`"
                + f" was set to truthy vlue: {gbml_config_pb.shared_config.should_skip_automatic_temp_asset_cleanup}"
            )
        else:
            gcs_utils = GcsUtils()
            temp_dir_gcs_path: GcsUri = gcs_constants.get_applied_task_temp_gcs_path(
                applied_task_identifier=applied_task_identifier
            )
            logger.info(
                f"Will automatically cleanup the temporary assets directory: ${temp_dir_gcs_path}"
            )
            gcs_utils.delete_files_in_bucket_dir(gcs_path=temp_dir_gcs_path)

    def __write_post_processor_metrics_to_uri(
        self,
        model_eval_metrics: EvalMetricsCollection,
        gbml_config_pb: gbml_config_pb2.GbmlConfig,
    ):
        file_loader = FileLoader()
        tfh = tempfile.NamedTemporaryFile(delete=False)
        local_tfh_uri = LocalUri(tfh.name)
        post_processor_log_metrics_uri = UriFactory.create_uri(
            uri=gbml_config_pb.shared_config.postprocessed_metadata.post_processor_log_metrics_uri
        )
        EvalMetricsCollectionTranslator.write_kfp_metrics_to_pipeline_metric_path(
            eval_metrics=model_eval_metrics, path=local_tfh_uri
        )
        file_loader.load_file(
            file_uri_src=local_tfh_uri, file_uri_dst=post_processor_log_metrics_uri
        )
        logger.info(f"Wrote eval metrics to {post_processor_log_metrics_uri.uri}.")

    def __should_run_unenumeration(
        self, gbml_config_wrapper: GbmlConfigPbWrapper
    ) -> bool:
        """
        When using the experimental GLT backend, we should run unenumeration in the post processor.
        """
        return gbml_config_wrapper.should_use_experimental_glt_backend

    def __run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
    ):
        gbml_config_wrapper: GbmlConfigPbWrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=task_config_uri
            )
        )
        if self.__should_run_unenumeration(gbml_config_wrapper=gbml_config_wrapper):
            logger.info(f"Running unenumeration for inferred assets in post processor")
            unenumerate_all_inferred_bq_assets(
                gbml_config_pb_wrapper=gbml_config_wrapper
            )
            logger.info(
                f"Finished running unenumeration for inferred assets in post processor"
            )

        self.__run_post_process(
            gbml_config_pb=gbml_config_wrapper.gbml_config_pb,
            applied_task_identifier=applied_task_identifier,
        )

    @flushes_metrics(get_metrics_service_instance_fn=get_metrics_service_instance)
    @profileit(
        metric_name=TIMER_POST_PROCESSOR_S,
        get_metrics_service_instance_fn=get_metrics_service_instance,
    )
    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
    ):
        try:
            return self.__run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
            )

        except Exception as e:
            logger.error(
                "Post Processor failed due to a raised exception; which will follow"
            )
            logger.error(e)
            logger.error(traceback.format_exc())
            sys.exit(f"System will now exit: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Program to run user defined logic that runs after the whole pipeline. "
        + "Subsequently cleans up any temporary assets"
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
    args = parser.parse_args()

    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
    applied_task_identifier = AppliedTaskIdentifier(args.job_name)
    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    post_processor = PostProcessor()
    post_processor.run(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
    )
