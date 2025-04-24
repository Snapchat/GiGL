import argparse
from typing import Optional

import gigl.src.common.constants.gcs as gcs_constants
from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.common.utils.metrics_service_provider import initialize_metrics

# TODO: (svij) Rename Trainer to TrainerV1
from gigl.src.training.v1.trainer import Trainer as TrainerV1
from gigl.src.training.v2.glt_trainer import GLTTrainer

logger = Logger()


class Trainer:
    def __remove_existing_trainer_paths(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        applied_task_identifier: AppliedTaskIdentifier,
    ) -> None:
        """
        Clean up paths that Trainer would be writing to in order to avoid clobbering of data.
        These paths are inferred from the GbmlConfig and the AppliedTaskIdentifier.
        :return:
        """

        logger.info("Preparing staging paths for Trainer...")
        paths_to_delete = (
            [
                gcs_constants.get_trainer_asset_dir_gcs_path(
                    applied_task_identifier=applied_task_identifier
                )
            ]
            + gbml_config_pb_wrapper.trained_model_metadata_pb_wrapper.get_output_paths()
        )
        file_loader = FileLoader()
        logger.info(f"Will delete files @ the following paths: {paths_to_delete}")
        file_loader.delete_files(uris=paths_to_delete)

    def run(
        self,
        applied_task_identifier: AppliedTaskIdentifier,
        task_config_uri: Uri,
        resource_config_uri: Uri,
        cpu_docker_uri: Optional[str] = None,
        cuda_docker_uri: Optional[str] = None,
    ) -> None:
        gbml_config_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            gbml_config_uri=task_config_uri
        )
        if (
            gbml_config_wrapper.shared_config.should_skip_training
            and gbml_config_wrapper.shared_config.should_skip_model_evaluation
        ):
            logger.info("Skipping both training and evaluation. Exiting.")
            return

        if not gbml_config_wrapper.shared_config.should_skip_training:
            self.__remove_existing_trainer_paths(
                gbml_config_pb_wrapper=gbml_config_wrapper,
                applied_task_identifier=applied_task_identifier,
            )

        if gbml_config_wrapper.should_use_experimental_glt_backend:
            trainer_v2 = GLTTrainer()
            trainer_v2.run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )
        else:
            trainer_v1 = TrainerV1()
            trainer_v1.run(
                applied_task_identifier=applied_task_identifier,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                cpu_docker_uri=cpu_docker_uri,
                cuda_docker_uri=cuda_docker_uri,
            )


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
        "--cpu_docker_uri",
        type=str,
        help="User Specified or KFP compiled Docker Image for CPU training",
        required=False,
    )
    parser.add_argument(
        "--cuda_docker_uri",
        type=str,
        help="User Specified or KFP compiled Docker Image for GPU training",
        required=False,
    )

    args = parser.parse_args()

    if not args.job_name or not args.task_config_uri or not args.resource_config_uri:
        raise RuntimeError("Missing command-line arguments")

    applied_task_identifier = AppliedTaskIdentifier(args.job_name)
    task_config_uri = UriFactory.create_uri(args.task_config_uri)
    resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
    cpu_docker_uri, cuda_docker_uri = args.cpu_docker_uri, args.cuda_docker_uri

    initialize_metrics(task_config_uri=task_config_uri, service_name=args.job_name)

    trainer = Trainer()
    trainer.run(
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        cpu_docker_uri=cpu_docker_uri,
        cuda_docker_uri=cuda_docker_uri,
    )
