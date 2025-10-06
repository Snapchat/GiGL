import unittest
import uuid

from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.bq import BqUtils


class TestBqUtilsRunIfExists(unittest.TestCase):
    """Integration test suite for BqUtils.run_if_exists method."""

    def setUp(self) -> None:
        """Set up test fixtures before each test method."""
        # Get the temp dataset from resource config
        resource_config = get_resource_config()
        self.temp_dataset = resource_config.temp_assets_bq_dataset_name
        self.project = resource_config.project

        # Initialize BqUtils
        self.bq_utils = BqUtils(project=self.project)

        # Create unique table names for each test to avoid conflicts
        self.test_table_suffix = str(uuid.uuid4()).replace("-", "_")
        self.test_table_name = f"bq_integration_test_{self.test_table_suffix}"
        self.test_table_path = (
            f"{self.project}.{self.temp_dataset}.{self.test_table_name}"
        )

        # Ensure the temp dataset exists
        self.bq_utils.create_bq_dataset(f"{self.project}.{self.temp_dataset}")

        # Test labels for query jobs
        self.test_labels = {
            "test_type": "integration",
            "component": "bq_utils",
            "method": "run_if_exists",
        }
        super().setUp()

    def tearDown(self) -> None:
        """Clean up after each test method."""
        # Clean up any test tables that might have been created
        self.bq_utils.delete_bq_table_if_exist(self.test_table_path, not_found_ok=True)
        super().tearDown()

    def test_run_if_exists_table_does_not_exist_creates_table(self) -> None:
        """Test that run_if_exists runs the query when table doesn't exist."""
        # Arrange: Ensure table doesn't exist
        self.assertFalse(
            self.bq_utils.does_bq_table_exist(self.test_table_path),
            f"Table {self.test_table_path} should not exist before test",
        )

        # Create query that creates a table with test data
        create_table_query = f"""
        CREATE TABLE `{self.test_table_path}` AS
        SELECT
            1 as id,
            'initial_value' as test_column,
            'created_by_run_if_exists' as source
        """

        # Act: Call run_if_exists
        result = self.bq_utils.run_if_exists(
            query=create_table_query,
            output_table=self.test_table_path,
            labels=self.test_labels,
        )

        # Assert: Query should have been executed (result is not None)
        self.assertIsNotNone(
            result, "Query should have been executed when table doesn't exist"
        )

        # Assert: Table should now exist
        self.assertTrue(
            self.bq_utils.does_bq_table_exist(self.test_table_path),
            f"Table {self.test_table_path} should exist after run_if_exists",
        )

        # Assert: Table should contain the expected data
        row_count = self.bq_utils.count_number_of_rows_in_bq_table(
            self.test_table_path, labels=self.test_labels
        )
        self.assertEqual(row_count, 1, "Table should contain exactly 1 row")

        # Verify the content of the created table
        verify_query = f"SELECT * FROM `{self.test_table_path}`"
        result_rows = self.bq_utils.run_query(verify_query, labels=self.test_labels)

        rows = list(result_rows)
        self.assertEqual(len(rows), 1, "Should have exactly one row")

        row = rows[0]
        self.assertEqual(row["id"], 1)
        self.assertEqual(row["test_column"], "initial_value")
        self.assertEqual(row["source"], "created_by_run_if_exists")

    def test_run_if_exists_table_exists_skips_query(self) -> None:
        """Test that run_if_exists skips the query when table already exists."""
        # Arrange: Create the table first with initial data
        initial_create_query = f"""
        CREATE TABLE `{self.test_table_path}` AS
        SELECT
            1 as id,
            'original_value' as test_column,
            'initial_creation' as source
        """

        self.bq_utils.run_query(initial_create_query, labels=self.test_labels)

        # Verify table exists and has initial data
        self.assertTrue(
            self.bq_utils.does_bq_table_exist(self.test_table_path),
            f"Table {self.test_table_path} should exist before test",
        )

        initial_row_count = self.bq_utils.count_number_of_rows_in_bq_table(
            self.test_table_path, labels=self.test_labels
        )
        self.assertEqual(initial_row_count, 1, "Table should initially contain 1 row")

        # Create a query that would modify the table if executed
        # This query would add a new row if executed
        modify_table_query = f"""
        INSERT INTO `{self.test_table_path}`
        SELECT
            2 as id,
            'modified_value' as test_column,
            'should_not_be_executed' as source
        """

        # Act: Call run_if_exists on existing table
        result = self.bq_utils.run_if_exists(
            query=modify_table_query,
            output_table=self.test_table_path,
            labels=self.test_labels,
        )

        # Assert: Query should NOT have been executed (result is None)
        self.assertIsNone(
            result, "Query should not have been executed when table exists"
        )

        # Assert: Table should still exist
        self.assertTrue(
            self.bq_utils.does_bq_table_exist(self.test_table_path),
            f"Table {self.test_table_path} should still exist",
        )

        # Assert: Table should still contain only the original data (not modified)
        final_row_count = self.bq_utils.count_number_of_rows_in_bq_table(
            self.test_table_path, labels=self.test_labels
        )
        self.assertEqual(final_row_count, 1, "Table should still contain only 1 row")

        # Verify the content hasn't changed
        verify_query = f"SELECT * FROM `{self.test_table_path}` ORDER BY id"
        result_rows = self.bq_utils.run_query(verify_query, labels=self.test_labels)

        rows = list(result_rows)
        self.assertEqual(len(rows), 1, "Should still have exactly one row")

        row = rows[0]
        self.assertEqual(row["id"], 1)
        self.assertEqual(row["test_column"], "original_value")
        self.assertEqual(row["source"], "initial_creation")

        # Ensure the row that would have been inserted by the skipped query is NOT there
        check_for_skipped_row_query = f"""
        SELECT COUNT(*) as count
        FROM `{self.test_table_path}`
        WHERE source = 'should_not_be_executed'
        """
        skipped_row_result = self.bq_utils.run_query(
            check_for_skipped_row_query, labels=self.test_labels
        )
        skipped_count = list(skipped_row_result)[0]["count"]
        self.assertEqual(
            skipped_count, 0, "No rows from the skipped query should exist"
        )

    def test_run_if_exists_with_complex_table_path_formats(self) -> None:
        """Test run_if_exists works with different BQ table path formats."""
        # Test with colon-separated format
        colon_table_path = (
            f"{self.project}:{self.temp_dataset}.{self.test_table_name}_colon"
        )

        # Arrange: Ensure table doesn't exist
        self.assertFalse(
            self.bq_utils.does_bq_table_exist(colon_table_path),
            f"Table {colon_table_path} should not exist before test",
        )

        # Create query
        create_table_query = f"""
        CREATE TABLE `{self.project}.{self.temp_dataset}.{self.test_table_name}_colon` AS
        SELECT 'colon_format_test' as format_type
        """

        # Act: Call run_if_exists with colon format
        result = self.bq_utils.run_if_exists(
            query=create_table_query,
            output_table=colon_table_path,
            labels=self.test_labels,
        )

        # Assert: Query should have been executed
        self.assertIsNotNone(
            result, "Query should have been executed with colon format"
        )

        # Assert: Table should exist (check with dot format)
        dot_format_path = (
            f"{self.project}.{self.temp_dataset}.{self.test_table_name}_colon"
        )
        self.assertTrue(
            self.bq_utils.does_bq_table_exist(dot_format_path),
            f"Table should exist after creation with colon format",
        )

        # Clean up
        self.bq_utils.delete_bq_table_if_exist(dot_format_path, not_found_ok=True)


if __name__ == "__main__":
    unittest.main()
