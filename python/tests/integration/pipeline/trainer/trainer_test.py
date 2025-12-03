import datetime
import json
import tempfile
import unittest
from typing import Tuple

import torch

import gigl.src.common.constants.gcs as gcs_consts
from gigl.common import GcsUri, LocalUri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
)
from gigl.src.training.v1.lib.training_process import GnnTrainingProcess
from snapchat.research.gbml import gbml_config_pb2

logger = Logger()


class TrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.__gcs_utils = GcsUtils()
        self.__proto_utils = ProtoUtils()
        self.__trainer = GnnTrainingProcess()

    def tearDown(self) -> None:
        pass

    def __validate_trainer_output_exists(
        self, gbml_config_pb: gbml_config_pb2.GbmlConfig
    ):
        trained_model_metadata = gbml_config_pb.shared_config.trained_model_metadata

        # Check eval metrics exists and contains a dict.
        self.assertIsNotNone(trained_model_metadata.eval_metrics_uri)
        eval_metrics_uri = GcsUri(trained_model_metadata.eval_metrics_uri)
        self.assertTrue(self.__gcs_utils.does_gcs_file_exist(gcs_path=eval_metrics_uri))
        tfh = self.__gcs_utils.download_file_from_gcs_to_temp_file(
            gcs_path=eval_metrics_uri
        )
        content = json.load(tfh)
        self.assertTrue(isinstance(content, dict))

        # Check trained model exists.
        self.assertIsNotNone(trained_model_metadata.trained_model_uri)
        trained_model_uri = GcsUri(trained_model_metadata.trained_model_uri)
        self.assertTrue(self.__gcs_utils.does_gcs_file_exist(trained_model_uri))

        # TODO(nshah-sc): add a check for scripted model once we start persisting this.

    def __generate_and_populate_mocked_dataset_info_gbml_config_pb(
        self, mocked_dataset_info: MockedDatasetInfo
    ) -> Tuple[gbml_config_pb2.GbmlConfig, LocalUri, GcsUri]:
        task_name = mocked_dataset_info.name
        artifact_metadata = get_mocked_dataset_artifact_metadata()[task_name]
        gbml_config_pb = self.__proto_utils.read_proto_from_yaml(
            uri=artifact_metadata.frozen_gbml_config_uri,
            proto_cls=gbml_config_pb2.GbmlConfig,
        )

        # Overwrite output paths.
        trained_model_metadata = gbml_config_pb.shared_config.trained_model_metadata

        output_directory = GcsUri.join(
            gcs_consts.get_applied_task_temp_gcs_path(
                applied_task_identifier=AppliedTaskIdentifier("trainer_pipeline_test")
            ),
            task_name,
            str(datetime.datetime.now().timestamp()),
        )
        trained_model_metadata.trained_model_uri = GcsUri.join(
            output_directory, "trained_model.pt"
        ).uri
        trained_model_metadata.scripted_model_uri = GcsUri.join(
            output_directory, "scripted_model.pt"
        ).uri
        trained_model_metadata.eval_metrics_uri = GcsUri.join(
            output_directory, "eval_metrics.json"
        ).uri

        # Write out the config.
        f = tempfile.NamedTemporaryFile(delete=False)
        gbml_config_local_uri = LocalUri(f.name)
        self.__proto_utils.write_proto_to_yaml(
            proto=gbml_config_pb, uri=gbml_config_local_uri
        )

        return gbml_config_pb, gbml_config_local_uri, output_directory

    def test_supervised_node_classification_training(self):
        (
            frozen_gbml_config_pb,
            frozen_gbml_config_uri,
            output_gcs_dir,
        ) = self.__generate_and_populate_mocked_dataset_info_gbml_config_pb(
            mocked_dataset_info=CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO
        )

        self.__trainer.run(
            task_config_uri=frozen_gbml_config_uri, device=torch.device("cpu")
        )

        self.__validate_trainer_output_exists(gbml_config_pb=frozen_gbml_config_pb)

        self.__gcs_utils.delete_files_in_bucket_dir(gcs_path=output_gcs_dir)

    def test_node_anchor_based_link_prediction_training(self):
        (
            frozen_gbml_config_pb,
            frozen_gbml_config_uri,
            output_gcs_dir,
        ) = self.__generate_and_populate_mocked_dataset_info_gbml_config_pb(
            mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO
        )

        self.__trainer.run(
            task_config_uri=frozen_gbml_config_uri, device=torch.device("cpu")
        )

        self.__validate_trainer_output_exists(gbml_config_pb=frozen_gbml_config_pb)

        self.__gcs_utils.delete_files_in_bucket_dir(gcs_path=output_gcs_dir)
