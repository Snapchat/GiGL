from typing import Sequence
from unittest.mock import MagicMock, call, patch

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


def _make_mock_table_list_items(
    table_ids: Sequence[str], project: str = "myproject", dataset: str = "mydataset"
) -> list[MagicMock]:
    """Create mock TableListItem objects for BQ client.list_tables()."""
    mock_tables: list[MagicMock] = []
    for table_id in table_ids:
        mock_table = MagicMock()
        mock_table.table_id = table_id
        mock_table.full_table_id = f"{project}:{dataset}.{table_id}"
        mock_tables.append(mock_table)
    return mock_tables


@patch("gigl.src.common.utils.bq.bigquery.Client")
class GetLatestTableTest(TestCase):
    PREFIX = "myproject.mydataset.events_"

    def _make_bq_utils(self, mock_client_cls: MagicMock) -> BqUtils:
        return BqUtils(project="myproject")

    def _set_mock_tables(
        self, mock_client_cls: MagicMock, table_ids: Sequence[str]
    ) -> None:
        mock_client_cls.return_value.list_tables.return_value = (
            _make_mock_table_list_items(table_ids)
        )

    def test_returns_latest_table(self, mock_client_cls: MagicMock) -> None:
        """Happy path: picks lexicographic max while ignoring non-prefix and wrong-length tables."""
        bq_utils = self._make_bq_utils(mock_client_cls)
        self._set_mock_tables(
            mock_client_cls,
            [
                "events_20250101",
                "events_20250103",
                "events_20250102",
                "events_backup_20250101",  # wrong suffix length → skipped
                "old_events_20250104",  # wrong prefix → skipped
            ],
        )
        result = bq_utils.get_latest_table(bq_table_path_prefix=self.PREFIX)
        self.assertEqual(result, "myproject.mydataset.events_20250103")

    def test_cap_date_filters_newer_tables(self, mock_client_cls: MagicMock) -> None:
        bq_utils = self._make_bq_utils(mock_client_cls)
        self._set_mock_tables(
            mock_client_cls,
            ["events_20250101", "events_20250103", "events_20250102"],
        )
        result = bq_utils.get_latest_table(
            bq_table_path_prefix=self.PREFIX, cap_date="20250102"
        )
        self.assertEqual(result, "myproject.mydataset.events_20250102")

    @parameterized.expand(
        [
            param(
                "empty_table_list",
                table_ids=[],
                cap_date=None,
            ),
            param(
                "cap_date_filters_all",
                table_ids=["events_20250201", "events_20250202"],
                cap_date="20250101",
            ),
        ]
    )
    def test_raises_when_no_matching_tables(
        self,
        mock_client_cls: MagicMock,
        _name: str,
        table_ids: list[str],
        cap_date: str,
    ) -> None:
        bq_utils = self._make_bq_utils(mock_client_cls)
        self._set_mock_tables(mock_client_cls, table_ids)
        with self.assertRaises(ValueError):
            bq_utils.get_latest_table(
                bq_table_path_prefix=self.PREFIX, cap_date=cap_date
            )

    def test_returns_latest_table_with_hourly_suffix(
        self, mock_client_cls: MagicMock
    ) -> None:
        bq_utils = self._make_bq_utils(mock_client_cls)
        self._set_mock_tables(
            mock_client_cls,
            ["events_2025010100", "events_2025010112", "events_2025010106"],
        )
        result = bq_utils.get_latest_table(
            bq_table_path_prefix=self.PREFIX, table_partition_suffix="YYYYMMDDHH"
        )
        self.assertEqual(result, "myproject.mydataset.events_2025010112")


@patch("gigl.src.common.utils.bq.bigquery.Client")
class BqUtilsLabelsTest(TestCase):
    INSTANCE_LABELS: dict[str, str] = {"team": "gnn", "env": "dev"}

    def test_merge_labels_defaults_used_when_no_override(
        self, mock_client_cls: MagicMock
    ) -> None:
        bq_utils = BqUtils(project="p", labels=self.INSTANCE_LABELS)
        self.assertEqual(bq_utils._merge_labels({}), {"team": "gnn", "env": "dev"})

    def test_merge_labels_merges_and_overrides(
        self, mock_client_cls: MagicMock
    ) -> None:
        bq_utils = BqUtils(project="p", labels=self.INSTANCE_LABELS)
        merged = bq_utils._merge_labels({"team": "ml", "job": "train"})
        self.assertEqual(merged, {"team": "ml", "env": "dev", "job": "train"})

    @patch("gigl.src.common.utils.bq.bigquery.QueryJobConfig")
    def test_run_query_applies_merged_labels(
        self, mock_job_config_cls: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        bq_utils = BqUtils(project="p", labels=self.INSTANCE_LABELS)
        mock_job_config = mock_job_config_cls.return_value
        bq_utils.run_query("SELECT 1", labels={"job": "train"})
        self.assertEqual(
            mock_job_config.labels, {"team": "gnn", "env": "dev", "job": "train"}
        )

    @patch("gigl.src.common.utils.bq.bigquery.QueryJobConfig")
    def test_count_rows_applies_merged_labels(
        self, mock_job_config_cls: MagicMock, mock_client_cls: MagicMock
    ) -> None:
        bq_utils = BqUtils(project="p", labels=self.INSTANCE_LABELS)
        mock_job_config = mock_job_config_cls.return_value
        mock_client_cls.return_value.query.return_value.result.return_value = [
            {"ct": 42}
        ]
        result = bq_utils.count_number_of_rows_in_bq_table("proj.ds.tbl")
        self.assertEqual(result, 42)
        self.assertEqual(
            mock_job_config.labels, {"team": "gnn", "env": "dev"}
        )
