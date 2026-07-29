import uuid

import apache_beam as beam
from absl.testing import absltest
from apache_beam.options.pipeline_options import GoogleCloudOptions, PipelineOptions
from apache_beam.testing.util import assert_that, equal_to
from google.api_core.exceptions import NotFound

from gigl.common.beam.sharded_read import (
    BigQueryShardedReadConfig,
    ShardedExportRead,
)
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.bq import BqUtils
from tests.test_assets.test_case import TestCase

NUM_ROWS = 20


class ShardedReadIntegrationTest(TestCase):
    """Integration tests for ShardedExportRead against real BigQuery."""

    def setUp(self):
        resource_config = get_resource_config()
        test_unique_name = f"gigl_sharded_read_test_{uuid.uuid4().hex}"

        self.bq_utils = BqUtils(project=resource_config.project)
        self.bq_project = resource_config.project
        self.bq_dataset = resource_config.temp_assets_bq_dataset_name
        self.temp_location = (
            f"{resource_config.temp_assets_regional_bucket_path}/"
            f"sharded_read_test/{test_unique_name}"
        )
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
        FROM UNNEST(GENERATE_ARRAY(0, {NUM_ROWS - 1})) as id
        """
        self.bq_utils.run_query(query=create_query, labels={})

    def tearDown(self):
        self.bq_utils.delete_bq_table_if_exist(
            bq_table_path=self.table_path, not_found_ok=True
        )

    def test_reads_all_rows_across_shards(self):
        sharded_read_config = BigQueryShardedReadConfig(
            shard_key="user_id",
            project_id=self.bq_project,
            temp_dataset_name=self.bq_dataset,
            num_shards=2,
        )

        # Construction validates the shard key against the real table schema.
        sharded_read = ShardedExportRead(
            table_name=self.table_path,
            sharded_read_info=sharded_read_config,
        )

        # ReadFromBigQuery's EXPORT method stages query results in GCS, so the
        # DirectRunner pipeline needs a project and temp_location to run against.
        options = PipelineOptions()
        google_cloud_options = options.view_as(GoogleCloudOptions)
        google_cloud_options.project = self.bq_project
        google_cloud_options.temp_location = self.temp_location

        # ShardedExportRead flattens every shard, so the read yields one dict per
        # row with the table's column names as keys. We assert that all NUM_ROWS
        # rows are recovered across the shards.
        expected_rows = [
            {"user_id": i, "name": f"test_name_{i}"} for i in range(NUM_ROWS)
        ]
        with beam.Pipeline(options=options) as pipeline:
            rows = pipeline | sharded_read
            assert_that(rows, equal_to(expected_rows))

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
