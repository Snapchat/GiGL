import time
import unittest
import uuid

from parameterized import param, parameterized

from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.bq import BqUtils


class BqUtilsIntegrationTest(unittest.TestCase):
    """Integration tests for BqUtils that use real BigQuery resources."""

    def setUp(self):
        """Set up test fixtures using real BQ resources from resource config."""
        resource_config = get_resource_config()
        test_unique_name = f"gigl_bq_test_{uuid.uuid4().hex}"

        self.bq_utils = BqUtils(project=resource_config.project)
        self.bq_project = resource_config.project
        self.bq_dataset = resource_config.temp_assets_bq_dataset_name

        # Create unique table names for this test run
        self.source_table_name = f"{test_unique_name}_source"
        self.dest_table_name = f"{test_unique_name}_dest"

        self.source_table_path = self.bq_utils.join_path(
            self.bq_project,
            self.bq_dataset,
            self.source_table_name,
        )
        self.dest_table_path = self.bq_utils.join_path(
            self.bq_project,
            self.bq_dataset,
            self.dest_table_name,
        )

    def tearDown(self):
        """Clean up test tables after each test."""
        self.bq_utils.delete_bq_table_if_exist(
            bq_table_path=self.source_table_path, not_found_ok=True
        )
        self.bq_utils.delete_bq_table_if_exist(
            bq_table_path=self.dest_table_path, not_found_ok=True
        )

    def _create_test_table_with_data(self, table_path: str, num_rows: int = 100):
        """Helper method to create a test table with sample data.

        Args:
            table_path: Full path to the table to create
            num_rows: Number of rows to insert
        """
        # Use a query-based approach to create the table with data
        # This is more reliable than streaming inserts for testing
        table_path = self.bq_utils.format_bq_path(table_path)

        # Generate test data using a query
        create_query = f"""
        CREATE OR REPLACE TABLE `{table_path}` AS
        SELECT
            id,
            CONCAT('test_name_', CAST(id AS STRING)) as name,
            CAST(id AS FLOAT64) * 1.5 as value
        FROM UNNEST(GENERATE_ARRAY(0, {num_rows - 1})) as id
        """

        self.bq_utils.run_query(query=create_query, labels={})

        # Small delay to ensure table is fully created and queryable
        time.sleep(1)

    def _verify_tables_have_identical_data(self, source_table: str, dest_table: str):
        """Helper method to verify two tables have identical data.

        Args:
            source_table: Source table path
            dest_table: Destination table path

        Raises:
            AssertionError: If tables have different data
        """
        source_formatted = self.bq_utils.format_bq_path(source_table)
        dest_formatted = self.bq_utils.format_bq_path(dest_table)

        # Query to check if tables are identical
        # This uses EXCEPT to find rows in source not in dest, and vice versa
        verify_query = f"""
        WITH source_not_in_dest AS (
            SELECT * FROM `{source_formatted}`
            EXCEPT DISTINCT
            SELECT * FROM `{dest_formatted}`
        ),
        dest_not_in_source AS (
            SELECT * FROM `{dest_formatted}`
            EXCEPT DISTINCT
            SELECT * FROM `{source_formatted}`
        )
        SELECT
            (SELECT COUNT(*) FROM source_not_in_dest) as rows_only_in_source,
            (SELECT COUNT(*) FROM dest_not_in_source) as rows_only_in_dest
        """

        result = self.bq_utils.run_query(query=verify_query, labels={})
        for row in result:
            rows_only_in_source = row["rows_only_in_source"]
            rows_only_in_dest = row["rows_only_in_dest"]

            self.assertEqual(
                rows_only_in_source,
                0,
                f"Found {rows_only_in_source} rows in source table that are not in destination",
            )
            self.assertEqual(
                rows_only_in_dest,
                0,
                f"Found {rows_only_in_dest} rows in destination table that are not in source",
            )

    def test_copy_table_basic(self):
        """Test basic table copy functionality."""
        # Create source table with data
        num_rows = 50
        self._create_test_table_with_data(self.source_table_path, num_rows=num_rows)

        # Verify source table exists and has data
        self.assertTrue(self.bq_utils.does_bq_table_exist(self.source_table_path))
        source_row_count = self.bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self.source_table_path, labels={}
        )
        self.assertEqual(source_row_count, num_rows)

        # Copy the table
        self.bq_utils.copy_table(
            source_table=self.source_table_path,
            destination_table=self.dest_table_path,
            overwrite=True,
        )

        # Small delay to ensure copy is complete
        time.sleep(1)

        # Verify destination table exists
        self.assertTrue(self.bq_utils.does_bq_table_exist(self.dest_table_path))

        # Verify destination has same number of rows as source
        dest_row_count = self.bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self.dest_table_path, labels={}
        )
        self.assertEqual(dest_row_count, source_row_count)

        # Verify schemas match
        source_schema = self.bq_utils.fetch_bq_table_schema(self.source_table_path)
        dest_schema = self.bq_utils.fetch_bq_table_schema(self.dest_table_path)
        self.assertEqual(set(source_schema.keys()), set(dest_schema.keys()))

        # Verify actual data is identical
        self._verify_tables_have_identical_data(
            self.source_table_path, self.dest_table_path
        )

    def test_copy_table_overwrite(self):
        """Test that copy_table can overwrite an existing destination table."""
        # Create source table with data
        num_source_rows = 30
        self._create_test_table_with_data(
            self.source_table_path, num_rows=num_source_rows
        )

        # Create destination table with different data
        num_dest_rows = 100
        self._create_test_table_with_data(self.dest_table_path, num_rows=num_dest_rows)

        # Verify both tables exist with different row counts
        self.assertTrue(self.bq_utils.does_bq_table_exist(self.source_table_path))
        self.assertTrue(self.bq_utils.does_bq_table_exist(self.dest_table_path))

        dest_row_count_before = self.bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self.dest_table_path, labels={}
        )
        self.assertEqual(dest_row_count_before, num_dest_rows)

        # Copy source to destination with overwrite=True
        self.bq_utils.copy_table(
            source_table=self.source_table_path,
            destination_table=self.dest_table_path,
            overwrite=True,
        )

        # Small delay to ensure copy is complete
        time.sleep(1)

        # Verify destination now has same row count as source
        dest_row_count_after = self.bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self.dest_table_path, labels={}
        )
        self.assertEqual(dest_row_count_after, num_source_rows)

        # Verify actual data is identical to source (not the old dest data)
        self._verify_tables_have_identical_data(
            self.source_table_path, self.dest_table_path
        )

    @parameterized.expand(
        [
            param(
                "Test with colon separator in source path",
                source_format="project:dataset.table",
                dest_format="project.dataset.table",
            ),
            param(
                "Test with colon separator in dest path",
                source_format="project.dataset.table",
                dest_format="project:dataset.table",
            ),
            param(
                "Test with both using dot separator",
                source_format="project.dataset.table",
                dest_format="project.dataset.table",
            ),
        ]
    )
    def test_copy_table_path_formats(self, _, source_format: str, dest_format: str):
        """Test that copy_table handles different path format styles correctly."""
        # Create source table with data
        num_rows = 20
        self._create_test_table_with_data(self.source_table_path, num_rows=num_rows)

        # Format paths according to the test parameters
        source_path = self._format_path(self.source_table_path, source_format)
        dest_path = self._format_path(self.dest_table_path, dest_format)

        # Copy the table using the specified path formats
        self.bq_utils.copy_table(
            source_table=source_path, destination_table=dest_path, overwrite=True
        )

        # Small delay to ensure copy is complete
        time.sleep(1)

        # Verify the copy was successful
        self.assertTrue(self.bq_utils.does_bq_table_exist(self.dest_table_path))
        dest_row_count = self.bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self.dest_table_path, labels={}
        )
        self.assertEqual(dest_row_count, num_rows)

        # Verify actual data is identical regardless of path format used
        self._verify_tables_have_identical_data(
            self.source_table_path, self.dest_table_path
        )

    def _format_path(self, table_path: str, format_style: str) -> str:
        """Helper to format table path according to format style.

        Args:
            table_path: Standard table path (project.dataset.table)
            format_style: Either 'project:dataset.table' or 'project.dataset.table'

        Returns:
            Formatted table path
        """
        parts = table_path.split(".")
        if format_style.startswith("project:"):
            return f"{parts[0]}:{parts[1]}.{parts[2]}"
        else:
            return table_path

    def test_copy_empty_table(self):
        """Test copying a table with no rows (schema only)."""
        # Create source table without data using a query
        source_table_formatted = self.bq_utils.format_bq_path(self.source_table_path)
        create_empty_query = f"""
        CREATE OR REPLACE TABLE `{source_table_formatted}` AS
        SELECT
            CAST(id AS INT64) as id,
            CONCAT('test_name_', CAST(id AS STRING)) as name,
            CAST(id AS FLOAT64) * 1.5 as value
        FROM UNNEST(GENERATE_ARRAY(0, -1)) as id
        """
        self.bq_utils.run_query(query=create_empty_query, labels={})

        # Small delay to ensure table is created
        time.sleep(1)

        # Verify source table exists but has no rows
        self.assertTrue(self.bq_utils.does_bq_table_exist(self.source_table_path))
        source_row_count = self.bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self.source_table_path, labels={}
        )
        self.assertEqual(source_row_count, 0)

        # Copy the empty table
        self.bq_utils.copy_table(
            source_table=self.source_table_path,
            destination_table=self.dest_table_path,
            overwrite=True,
        )

        # Verify destination table exists and is also empty
        self.assertTrue(self.bq_utils.does_bq_table_exist(self.dest_table_path))
        dest_row_count = self.bq_utils.count_number_of_rows_in_bq_table(
            bq_table=self.dest_table_path, labels={}
        )
        self.assertEqual(dest_row_count, 0)

        # Verify schema was copied correctly
        dest_schema = self.bq_utils.fetch_bq_table_schema(self.dest_table_path)
        self.assertEqual(len(dest_schema), 3)  # id, name, value
        self.assertIn("id", dest_schema)
        self.assertIn("name", dest_schema)
        self.assertIn("value", dest_schema)

        # Verify both tables are empty and identical (no data differences)
        self._verify_tables_have_identical_data(
            self.source_table_path, self.dest_table_path
        )


if __name__ == "__main__":
    unittest.main()
