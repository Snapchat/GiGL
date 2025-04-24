from typing import List

from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.src.common.constants import gcs as gcs_constants
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils

logger = Logger()


class InferenceAssets:
    """
    Utility class for managing temp and permanent inferencer assets.
    """

    @staticmethod
    def get_unenumerated_embedding_table_path(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper, node_type: str
    ) -> str:
        """
        Get the unenumerated embedding table path for a given node type.
        i.e. table contains the embeddings indexed by original node id
        """
        return gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map[
            node_type
        ].embeddings_path

    @staticmethod
    def get_unenumerated_prediction_table_path(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper, node_type: str
    ) -> str:
        """
        Get the unenumerated embedding table path for a given node type.
        i.e. table contains the embeddings indexed by original node id
        """
        return gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map[
            node_type
        ].predictions_path

    @staticmethod
    def get_enumerated_embedding_table_path(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper, node_type: str
    ) -> str:
        """
        Get the enumerated embedding table path for a given node type.
        i.e. table should containe (enumerated_node_id: int) ---> embedding
        """
        unenumerated_bq_table_path = (
            InferenceAssets.get_unenumerated_embedding_table_path(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
            )
        )
        #  This is optional and may be none; we traditionally dont have any asserts here so matching that style
        #  This should probably change in the future
        if not unenumerated_bq_table_path:
            return ""
        return InferenceAssets._create_enumerated_bq_table_name(
            unenumerated_bq_table_path=unenumerated_bq_table_path
        )

    @staticmethod
    def get_enumerated_predictions_table_path(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper, node_type: str
    ) -> str:
        """
        Get the enumerated predictions table path for a given node type.
        i.e. table should containe (enumerated_node_id: int) ---> prediction
        """
        unenumerated_bq_table_path = (
            InferenceAssets.get_unenumerated_prediction_table_path(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
            )
        )
        #  This is optional and may be none; we traditionally dont have any asserts here so matching that style
        #  This should probably change in the future
        if not unenumerated_bq_table_path:
            return ""
        return InferenceAssets._create_enumerated_bq_table_name(
            unenumerated_bq_table_path=unenumerated_bq_table_path
        )

    @staticmethod
    def prepare_staging_paths(
        applied_task_identifier: AppliedTaskIdentifier,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        project: str,
    ) -> None:
        """
        Prepare staging paths for inferencer assets by clearing the paths that inferencer
        would be writing to, to avoid clobbering of data.
        """
        logger.info("Preparing staging paths for Inferencer...")

        InferenceAssets._delete_temp_gcs_files(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            applied_task_identifier=applied_task_identifier,
            project=project,
        )
        InferenceAssets._delete_bq_output_tables(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            project=project,
        )
        logger.info("Staging paths for Inferencer prepared.")

    @staticmethod
    def get_gcs_asset_write_path_prefix(
        applied_task_identifier: AppliedTaskIdentifier, bq_table_path: str
    ) -> GcsUri:
        """
        Formulated an intermediary GCS path for writing embeddings or predictions based on the bq table path
        Args:
            applied_task_identifier (AppliedTaskIdentifier): The name provided for the gigl job
            bq_table_path (str): Path to the table for embeddings or predictions output

        Returns:
            GcsUri: The path to the gcs folder based on the bq table path
        """
        formatted_gcs_path = bq_table_path.replace(".", "_").replace(":", "__")
        # TODO (mkolodner): Update code to write to gcs in permanent storage location for enabling gcs inferencer output
        # TODO (svij): Ideally this should be writing to gcs paths formulated by:
        #   gigl.src.common.constants.gcs._INFERENCER_PREFIX
        return GcsUri.join(
            gcs_constants.get_applied_task_temp_gcs_path(
                applied_task_identifier=applied_task_identifier
            ),
            f"{formatted_gcs_path}/",
        )

    @staticmethod
    def _delete_temp_gcs_files(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        applied_task_identifier: AppliedTaskIdentifier,
        project: str,
    ):
        """
        Delete temporary GCS files created by the inferencer.
        """
        logger.info("Deleting temporary GCS files...")
        gcs_utils = GcsUtils(project=project)

        active_bq_table_paths = []
        for (
            node_type
        ) in (
            gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map.keys()
        ):
            bq_table_path_unenumerated_predictions = (
                InferenceAssets.get_unenumerated_prediction_table_path(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
                )
            )
            bq_table_path_unenumerated_embeddings = (
                InferenceAssets.get_unenumerated_embedding_table_path(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
                )
            )
            if bq_table_path_unenumerated_predictions:
                active_bq_table_paths.append(bq_table_path_unenumerated_predictions)
                active_bq_table_paths.append(
                    InferenceAssets._create_enumerated_bq_table_name(
                        unenumerated_bq_table_path=bq_table_path_unenumerated_predictions
                    )
                )
            if bq_table_path_unenumerated_embeddings:
                active_bq_table_paths.append(bq_table_path_unenumerated_embeddings)
                active_bq_table_paths.append(
                    InferenceAssets._create_enumerated_bq_table_name(
                        unenumerated_bq_table_path=bq_table_path_unenumerated_embeddings
                    )
                )

        for bq_table_path in active_bq_table_paths:
            table_gcs_write_path_uri: GcsUri = (
                InferenceAssets.get_gcs_asset_write_path_prefix(
                    applied_task_identifier=applied_task_identifier,
                    bq_table_path=bq_table_path,
                )
            )
            gcs_utils.delete_files_in_bucket_dir(table_gcs_write_path_uri)

    @staticmethod
    def _delete_bq_output_tables(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        project: str,
    ):
        logger.info("Deleting BigQuery output tables...")
        bq_utils = BqUtils(project=project)
        active_bq_table_paths = []
        for (
            node_type
        ) in (
            gbml_config_pb_wrapper.shared_config.inference_metadata.node_type_to_inferencer_output_info_map.keys()
        ):
            bq_table_path_unenumerated_predictions = (
                InferenceAssets.get_unenumerated_prediction_table_path(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
                )
            )
            bq_table_path_unenumerated_embeddings = (
                InferenceAssets.get_unenumerated_embedding_table_path(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper, node_type=node_type
                )
            )
            if bq_table_path_unenumerated_predictions:
                active_bq_table_paths.append(bq_table_path_unenumerated_predictions)
                active_bq_table_paths.append(
                    InferenceAssets._create_enumerated_bq_table_name(
                        unenumerated_bq_table_path=bq_table_path_unenumerated_predictions
                    )
                )
            if bq_table_path_unenumerated_embeddings:
                active_bq_table_paths.append(bq_table_path_unenumerated_embeddings)
                active_bq_table_paths.append(
                    InferenceAssets._create_enumerated_bq_table_name(
                        unenumerated_bq_table_path=bq_table_path_unenumerated_embeddings
                    )
                )

        for bq_table_path in active_bq_table_paths:
            bq_utils.delete_bq_table_if_exist(bq_table_path=bq_table_path)

    @staticmethod
    def _create_enumerated_bq_table_name(unenumerated_bq_table_path: str) -> str:
        """
        embeddingsPath contains the unenumerated embeddings table path. This function returns the input enumerated embeddings table path.
        bq_table_path: str: The path to the enumerated embeddings table. Format should be project-id.dataset-id.table-id
        """
        bq_table_path_list: List[str] = unenumerated_bq_table_path.split(".")
        assert (
            len(bq_table_path_list) == 3
        ), f"Invalid bq_table_path: {unenumerated_bq_table_path}, expected format: project-id.dataset-id.table-id; got"
        project_id, dataset_id, table_id = bq_table_path_list
        return f"{project_id}.{dataset_id}.enumerated_{table_id}"
