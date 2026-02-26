from unittest.mock import MagicMock, patch

from parameterized import param, parameterized

from gigl.src.common.utils.bq import BqUtils
from tests.test_assets.test_case import TestCase


class BqUtilsTest(TestCase):
    @parameterized.expand(
        [
            param(
                bq_table_path="bq_project.bq_dataset.bq_table",
                expected_project_id="bq_project",
                expected_dataset_id="bq_dataset",
                expected_table_name="bq_table",
            ),
            param(
                bq_table_path="bq_project:bq_dataset.bq_table",
                expected_project_id="bq_project",
                expected_dataset_id="bq_dataset",
                expected_table_name="bq_table",
            ),
        ]
    )
    def test_parse_and_format_bq_path(
        self,
        bq_table_path,
        expected_project_id,
        expected_dataset_id,
        expected_table_name,
    ):
        (
            parsed_project_id,
            parsed_dataset_id,
            parsed_table_name,
        ) = BqUtils.parse_bq_table_path(bq_table_path=bq_table_path)
        self.assertEqual(parsed_project_id, expected_project_id)
        self.assertEqual(parsed_dataset_id, expected_dataset_id)
        self.assertEqual(parsed_table_name, expected_table_name)
        reconstructed_bq_table_path = BqUtils.join_path(
            parsed_project_id, parsed_dataset_id, parsed_table_name
        )
        self.assertEqual(
            reconstructed_bq_table_path, BqUtils.format_bq_path(bq_table_path)
        )


@patch("gigl.src.common.utils.bq.bigquery.Client")
class GetLatestTableTest(TestCase):
    PREFIX = "myproject.mydataset.events_"

    def _make_bq_utils(self, mock_client_cls: MagicMock) -> BqUtils:
        return BqUtils(project="myproject")

    def test_returns_latest_table(self, mock_client_cls: MagicMock) -> None:
        bq_utils = self._make_bq_utils(mock_client_cls)
        with patch.object(
            bq_utils,
            "list_matching_tables",
            return_value=[
                "myproject.mydataset.events_20250101",
                "myproject.mydataset.events_20250103",
                "myproject.mydataset.events_20250102",
            ],
        ):
            result = bq_utils.get_latest_table(bq_table_path_prefix=self.PREFIX)
        self.assertEqual(result, "myproject.mydataset.events_20250103")

    def test_cap_date_filters_newer_tables(self, mock_client_cls: MagicMock) -> None:
        bq_utils = self._make_bq_utils(mock_client_cls)
        with patch.object(
            bq_utils,
            "list_matching_tables",
            return_value=[
                "myproject.mydataset.events_20250101",
                "myproject.mydataset.events_20250103",
                "myproject.mydataset.events_20250102",
            ],
        ):
            result = bq_utils.get_latest_table(
                bq_table_path_prefix=self.PREFIX, cap_date="20250102"
            )
        self.assertEqual(result, "myproject.mydataset.events_20250102")

    def test_no_matching_tables_raises(self, mock_client_cls: MagicMock) -> None:
        bq_utils = self._make_bq_utils(mock_client_cls)
        with patch.object(bq_utils, "list_matching_tables", return_value=[]):
            with self.assertRaises(ValueError):
                bq_utils.get_latest_table(bq_table_path_prefix=self.PREFIX)

    def test_cap_date_filters_all_raises(self, mock_client_cls: MagicMock) -> None:
        bq_utils = self._make_bq_utils(mock_client_cls)
        with patch.object(
            bq_utils,
            "list_matching_tables",
            return_value=[
                "myproject.mydataset.events_20250201",
                "myproject.mydataset.events_20250202",
            ],
        ):
            with self.assertRaises(ValueError):
                bq_utils.get_latest_table(
                    bq_table_path_prefix=self.PREFIX, cap_date="20250101"
                )

    def test_returns_latest_table_with_hourly_suffix(
        self, mock_client_cls: MagicMock
    ) -> None:
        bq_utils = self._make_bq_utils(mock_client_cls)
        with patch.object(
            bq_utils,
            "list_matching_tables",
            return_value=[
                "myproject.mydataset.events_2025010100",
                "myproject.mydataset.events_2025010112",
                "myproject.mydataset.events_2025010106",
            ],
        ):
            result = bq_utils.get_latest_table(
                bq_table_path_prefix=self.PREFIX, table_partition_suffix="YYYYMMDDHH"
            )
        self.assertEqual(result, "myproject.mydataset.events_2025010112")
