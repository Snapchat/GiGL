from unittest.mock import MagicMock

from gigl.src.common.utils.output_record_validation import (
    validate_bq_table_row_count,
    validate_node_output_records,
)
from tests.test_assets.test_case import TestCase


class ValidateBqTableRowCountTest(TestCase):
    """Unit tests for validate_bq_table_row_count."""

    def setUp(self) -> None:
        self._bq_utils = MagicMock()

    def test_passes_when_table_exists_and_count_matches(self) -> None:
        self._bq_utils.does_bq_table_exist.return_value = True
        self._bq_utils.count_number_of_rows_in_bq_table.return_value = 100

        validate_bq_table_row_count(
            bq_utils=self._bq_utils,
            table_path="project.dataset.table",
            expected_count=100,
            label="[paper] Embeddings",
        )

    def test_raises_when_table_does_not_exist(self) -> None:
        self._bq_utils.does_bq_table_exist.return_value = False

        with self.assertRaises(ValueError) as ctx:
            validate_bq_table_row_count(
                bq_utils=self._bq_utils,
                table_path="project.dataset.missing_table",
                expected_count=100,
                label="[paper] Embeddings",
            )

        error_message = str(ctx.exception)
        self.assertIn("does not exist", error_message)
        self.assertIn("project.dataset.missing_table", error_message)

    def test_raises_when_count_mismatches(self) -> None:
        self._bq_utils.does_bq_table_exist.return_value = True
        self._bq_utils.count_number_of_rows_in_bq_table.return_value = 50

        with self.assertRaises(ValueError) as ctx:
            validate_bq_table_row_count(
                bq_utils=self._bq_utils,
                table_path="project.dataset.table",
                expected_count=100,
                label="[paper] Embeddings",
            )

        error_message = str(ctx.exception)
        self.assertIn("mismatch", error_message)
        self.assertIn("100", error_message)
        self.assertIn("50", error_message)


class ValidateNodeOutputRecordsTest(TestCase):
    """Unit tests for validate_node_output_records."""

    def setUp(self) -> None:
        self._bq_utils = MagicMock()

    def _set_all_tables_exist(self) -> None:
        self._bq_utils.does_bq_table_exist.return_value = True

    def _set_row_counts(self, counts: dict[str, int]) -> None:
        """Configure mock to return different row counts per table path."""
        self._bq_utils.count_number_of_rows_in_bq_table.side_effect = (
            lambda table, labels={}: counts[table]
        )

    def test_single_node_type_embeddings_match(self) -> None:
        self._set_all_tables_exist()
        self._set_row_counts({"enum_table": 100, "emb_table": 100})

        validate_node_output_records(
            bq_utils=self._bq_utils,
            expected_count_tables={"paper": "enum_table"},
            embeddings_tables={"paper": "emb_table"},
        )

    def test_single_node_type_both_outputs_match(self) -> None:
        self._set_all_tables_exist()
        self._set_row_counts({"enum_table": 100, "emb_table": 100, "pred_table": 100})

        validate_node_output_records(
            bq_utils=self._bq_utils,
            expected_count_tables={"paper": "enum_table"},
            embeddings_tables={"paper": "emb_table"},
            predictions_tables={"paper": "pred_table"},
        )

    def test_only_predictions_no_embeddings(self) -> None:
        self._set_all_tables_exist()
        self._set_row_counts({"enum_table": 100, "pred_table": 100})

        validate_node_output_records(
            bq_utils=self._bq_utils,
            expected_count_tables={"paper": "enum_table"},
            predictions_tables={"paper": "pred_table"},
        )

    def test_raises_on_embeddings_count_mismatch(self) -> None:
        self._set_all_tables_exist()
        self._set_row_counts({"enum_table": 100, "emb_table": 50})

        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_table"},
                embeddings_tables={"paper": "emb_table"},
            )

        self.assertIn("[paper] Embeddings", str(ctx.exception))

    def test_raises_on_predictions_count_mismatch(self) -> None:
        self._set_all_tables_exist()
        self._set_row_counts({"enum_table": 100, "pred_table": 200})

        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_table"},
                predictions_tables={"paper": "pred_table"},
            )

        self.assertIn("[paper] Predictions", str(ctx.exception))

    def test_raises_on_missing_embeddings_table(self) -> None:
        self._bq_utils.does_bq_table_exist.side_effect = (
            lambda path: path != "emb_missing"
        )
        self._set_row_counts({"enum_table": 100})

        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_table"},
                embeddings_tables={"paper": "emb_missing"},
            )

        self.assertIn("does not exist", str(ctx.exception))
        self.assertIn("emb_missing", str(ctx.exception))

    def test_raises_on_missing_predictions_table(self) -> None:
        self._bq_utils.does_bq_table_exist.side_effect = (
            lambda path: path != "pred_missing"
        )
        self._set_row_counts({"enum_table": 100})

        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_table"},
                predictions_tables={"paper": "pred_missing"},
            )

        self.assertIn("does not exist", str(ctx.exception))

    def test_raises_when_neither_embeddings_nor_predictions_provided(self) -> None:
        self._set_all_tables_exist()
        self._set_row_counts({"enum_table": 100})

        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_table"},
            )

        self.assertIn(
            "Neither embeddings_path nor predictions_path is set", str(ctx.exception)
        )

    def test_raises_when_enumerated_table_is_empty_string(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": ""},
                embeddings_tables={"paper": "emb_table"},
            )

        self.assertIn("No enumerated_node_ids_bq_table configured", str(ctx.exception))

    def test_raises_when_enumerated_table_does_not_exist(self) -> None:
        self._bq_utils.does_bq_table_exist.return_value = False

        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_missing"},
                embeddings_tables={"paper": "emb_table"},
            )

        self.assertIn("enumerated_node_ids_bq_table does not exist", str(ctx.exception))

    def test_multiple_node_types_all_valid(self) -> None:
        self._set_all_tables_exist()
        self._set_row_counts(
            {
                "enum_author": 30,
                "emb_author": 30,
                "enum_paper": 50,
                "emb_paper": 50,
            }
        )

        validate_node_output_records(
            bq_utils=self._bq_utils,
            expected_count_tables={"author": "enum_author", "paper": "enum_paper"},
            embeddings_tables={"author": "emb_author", "paper": "emb_paper"},
        )

    def test_multiple_errors_collected(self) -> None:
        """Two independent errors are aggregated into a single ValueError."""
        self._bq_utils.does_bq_table_exist.side_effect = (
            lambda path: path != "emb_paper_missing"
        )
        self._set_row_counts(
            {
                "enum_author": 30,
                "emb_author": 20,  # Mismatch
                "enum_paper": 50,
                # emb_paper_missing does not exist
            }
        )

        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={
                    "author": "enum_author",
                    "paper": "enum_paper",
                },
                embeddings_tables={
                    "author": "emb_author",
                    "paper": "emb_paper_missing",
                },
            )

        error_message = str(ctx.exception)
        self.assertIn("2 error(s)", error_message)
        self.assertIn("[author]", error_message)
        self.assertIn("[paper]", error_message)

    def test_raises_on_unexpected_embeddings_key(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_table"},
                embeddings_tables={"unknown_type": "emb_table"},
            )

        self.assertIn("unknown_type", str(ctx.exception))

    def test_raises_on_unexpected_predictions_key(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_table"},
                predictions_tables={"typo_paper": "pred_table"},
            )

        self.assertIn("typo_paper", str(ctx.exception))

    def test_none_input_tables_treated_as_empty(self) -> None:
        """None embeddings/predictions tables behave the same as empty dicts."""
        self._set_all_tables_exist()
        self._set_row_counts({"enum_table": 100})

        with self.assertRaises(ValueError) as ctx:
            validate_node_output_records(
                bq_utils=self._bq_utils,
                expected_count_tables={"paper": "enum_table"},
                embeddings_tables=None,
                predictions_tables=None,
            )

        self.assertIn(
            "Neither embeddings_path nor predictions_path is set", str(ctx.exception)
        )
