import unittest

from parameterized import param, parameterized

from gigl.src.common.utils.bq import BqUtils


class BqUtilsTest(unittest.TestCase):
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
