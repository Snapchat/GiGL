import tempfile
import unittest

import gigl.src.common.constants.gcs as gcs_constants
from gigl.common import GcsUri, LocalUri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.common.utils.timeout import timeout
from gigl.src.config_populator.config_populator import ConfigPopulator
from snapchat.research.gbml import gbml_config_pb2
from tests.test_assets.uri_constants import DEFAULT_TEST_RESOURCE_CONFIG_URI

logger = Logger()


CONFIG_POPULATOR_PIPELINE_TIMEOUT_SECONDS = 300


class ConfigPopulatorPipelineTest(unittest.TestCase):
    """
    This test checks the completion of the ConfigPopulator step.
    """

    def __run_config_populator_pipeline(self) -> GcsUri:
        # create inner function, so args can be passed into timeout decorator
        @timeout(
            CONFIG_POPULATOR_PIPELINE_TIMEOUT_SECONDS,
            error_message="Config Populator pipeline timed out",
        )
        def __run_config_populator_pipeline_w_timeout() -> GcsUri:
            # Run config populator
            config_populator = ConfigPopulator()
            frozen_gbml_config_uri = config_populator.run(
                applied_task_identifier=self.applied_task_identifier,
                task_config_uri=self.template_gbml_config_uri,
                resource_config_uri=DEFAULT_TEST_RESOURCE_CONFIG_URI,
            )
            return frozen_gbml_config_uri

        # call run with timeout
        frozen_gbml_config_uri = __run_config_populator_pipeline_w_timeout()
        return frozen_gbml_config_uri

    def test_config_populator_pipeline_completion(self):
        frozen_gbml_config_uri = self.__run_config_populator_pipeline()
        self.assertTrue(
            self.gcs_utils.does_gcs_file_exist(gcs_path=frozen_gbml_config_uri)
        )

    def setUp(self) -> None:
        self.applied_task_identifier = AppliedTaskIdentifier(
            f"config_populator_testing_{current_formatted_datetime()}"
        )

        # Make a simple empty GbmlConfig proto
        f = tempfile.NamedTemporaryFile(delete=False)
        self.template_gbml_config_uri = LocalUri(f.name)
        task_metadata_pb = gbml_config_pb2.GbmlConfig.TaskMetadata(
            node_based_task_metadata=gbml_config_pb2.GbmlConfig.TaskMetadata.NodeBasedTaskMetadata()
        )
        gbml_config_pb = gbml_config_pb2.GbmlConfig(task_metadata=task_metadata_pb)
        proto_utils = ProtoUtils()
        proto_utils.write_proto_to_yaml(
            proto=gbml_config_pb, uri=self.template_gbml_config_uri
        )

        self.gcs_utils = GcsUtils()

    def tearDown(self) -> None:
        logger.info("Config Populator pipeline completion test tear down.")
        self.gcs_utils.delete_files_in_bucket_dir(
            gcs_path=gcs_constants.get_applied_task_temp_gcs_path(
                applied_task_identifier=self.applied_task_identifier
            )
        )

        self.gcs_utils.delete_files_in_bucket_dir(
            gcs_path=gcs_constants.get_applied_task_perm_gcs_path(
                applied_task_identifier=self.applied_task_identifier
            )
        )
        return super().tearDown()
