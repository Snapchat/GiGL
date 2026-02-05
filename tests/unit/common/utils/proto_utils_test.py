import os
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

from absl.testing import absltest

from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from snapchat.research.gbml import gbml_config_pb2
from tests.test_assets.test_case import TestCase

logger = Logger()

TEST_TASK_CONFIG = """
sharedConfig:
    isGraphDirected: true
datasetConfig:
    dataPreprocessorConfig:
        dataPreprocessorArgs:
            bq_edges_table_name: project.dataset.bq_edges_table_name_${now:%Y%m%d}
            positive_label_date_range: ${now:%Y%m%d,days-10}:${now:%Y%m%d,days-1}
"""


class ProtoUtilsTest(TestCase):
    def setUp(self):
        super().setUp()
        self.proto_utils = ProtoUtils()

        tmp_file = NamedTemporaryFile(delete=False)
        logger.info(f"Writing test config to {tmp_file.name}")
        tmp_file.write(TEST_TASK_CONFIG.encode())
        tmp_file.close()
        self.test_task_config_path = tmp_file.name

    def tearDown(self):
        super().tearDown()
        # delete the temporary file
        os.remove(self.test_task_config_path)

    def test_can_read_gbml_config_from_yaml(self):
        task_config = self.proto_utils.read_proto_from_yaml(
            uri=LocalUri(self.test_task_config_path),
            proto_cls=gbml_config_pb2.GbmlConfig,
        )

        self.assertEqual(task_config.shared_config.is_graph_directed, True)
        expected_bq_edges_table_name = (
            f"project.dataset.bq_edges_table_name_{datetime.now().strftime('%Y%m%d')}"
        )
        self.assertEqual(
            task_config.dataset_config.data_preprocessor_config.data_preprocessor_args[
                "bq_edges_table_name"
            ],
            expected_bq_edges_table_name,
        )
        expected_positive_label_date_range_start = (
            datetime.now() - timedelta(days=10)
        ).strftime("%Y%m%d")
        expected_positive_label_date_range_end = (
            datetime.now() - timedelta(days=1)
        ).strftime("%Y%m%d")
        self.assertEqual(
            task_config.dataset_config.data_preprocessor_config.data_preprocessor_args[
                "positive_label_date_range"
            ],
            f"{expected_positive_label_date_range_start}:{expected_positive_label_date_range_end}",
        )


if __name__ == "__main__":
    absltest.main()
