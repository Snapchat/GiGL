from unittest.mock import MagicMock, patch

from absl.testing import absltest
from google.api_core.exceptions import NotFound

from gigl.common.beam.sharded_read import _assert_shard_key_in_table
from tests.test_assets.test_case import TestCase


@patch("gigl.common.beam.sharded_read.BqUtils")
class AssertShardKeyInTableTest(TestCase):
    def test_passes_when_shard_key_is_a_column(
        self, mock_bq_utils_cls: MagicMock
    ) -> None:
        mock_bq_utils_cls.return_value.fetch_bq_table_schema.return_value = {
            "user_id": MagicMock(),
            "timestamp": MagicMock(),
        }

        _assert_shard_key_in_table(
            table_name="my-project.my-dataset.my-table", shard_key="user_id"
        )

        mock_bq_utils_cls.assert_called_once_with()
        mock_bq_utils_cls.return_value.fetch_bq_table_schema.assert_called_once_with(
            bq_table="my-project.my-dataset.my-table"
        )

    def test_raises_when_shard_key_missing(self, mock_bq_utils_cls: MagicMock) -> None:
        mock_bq_utils_cls.return_value.fetch_bq_table_schema.return_value = {
            "user_id": MagicMock(),
        }

        with self.assertRaises(ValueError):
            _assert_shard_key_in_table(
                table_name="my-project.my-dataset.my-table", shard_key="not_a_column"
            )

    def test_propagates_not_found_for_missing_table(
        self, mock_bq_utils_cls: MagicMock
    ) -> None:
        mock_bq_utils_cls.return_value.fetch_bq_table_schema.side_effect = NotFound(
            "Table not found"
        )

        with self.assertRaises(NotFound):
            _assert_shard_key_in_table(
                table_name="my-project.my-dataset.missing-table", shard_key="user_id"
            )


if __name__ == "__main__":
    absltest.main()
