import uuid

from absl.testing import absltest
from google.api_core.exceptions import NotFound

from gigl.common.beam.sharded_read import (
    BigQueryShardedReadConfig,
    ShardedExportRead,
)
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.bq import BqUtils
from tests.test_assets.test_case import TestCase


class ShardedReadIntegrationTest(TestCase):
    """Integration tests for ShardedExportRead's shard-key validation against real BigQuery."""

    def setUp(self):
        resource_config = get_resource_config()
        test_unique_name = f"gigl_sharded_read_test_{uuid.uuid4().hex}"

        self.bq_utils = BqUtils(project=resource_config.project)
        self.bq_project = resource_config.project
        self.bq_dataset = resource_config.temp_assets_bq_dataset_name
        self.table_path = self.bq_utils.join_path(
            self.bq_project,
            self.bq_dataset,
            test_unique_name,
        )

        table_path = self.bq_utils.format_bq_path(self.table_path)
        create_query = f"""
        CREATE OR REPLACE TABLE `{table_path}` AS
        SELECT
            id as user_id,
            CONCAT('test_name_', CAST(id AS STRING)) as name
        FROM UNNEST(GENERATE_ARRAY(0, 19)) as id
        """
        self.bq_utils.run_query(query=create_query, labels={})

    def tearDown(self):
        self.bq_utils.delete_bq_table_if_exist(
            bq_table_path=self.table_path, not_found_ok=True
        )

    def test_constructs_when_shard_key_is_a_column(self):
        sharded_read_config = BigQueryShardedReadConfig(
            shard_key="user_id",
            project_id=self.bq_project,
            temp_dataset_name=self.bq_dataset,
            num_shards=2,
        )

        # Construction validates the shard key against the real table schema.
        ShardedExportRead(
            table_name=self.table_path,
            sharded_read_info=sharded_read_config,
        )

    def test_raises_when_shard_key_is_not_a_column(self):
        sharded_read_config = BigQueryShardedReadConfig(
            shard_key="not_a_column",
            project_id=self.bq_project,
            temp_dataset_name=self.bq_dataset,
            num_shards=2,
        )

        with self.assertRaises(ValueError):
            ShardedExportRead(
                table_name=self.table_path,
                sharded_read_info=sharded_read_config,
            )

    def test_raises_when_table_does_not_exist(self):
        missing_table_path = self.bq_utils.join_path(
            self.bq_project,
            self.bq_dataset,
            f"gigl_sharded_read_missing_{uuid.uuid4().hex}",
        )
        sharded_read_config = BigQueryShardedReadConfig(
            shard_key="user_id",
            project_id=self.bq_project,
            temp_dataset_name=self.bq_dataset,
            num_shards=2,
        )

        with self.assertRaises(NotFound):
            ShardedExportRead(
                table_name=missing_table_path,
                sharded_read_info=sharded_read_config,
            )


if __name__ == "__main__":
    absltest.main()
