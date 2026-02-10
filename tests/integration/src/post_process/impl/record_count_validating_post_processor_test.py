import tempfile
import time
import uuid

from absl.testing import absltest

from gigl.common import LocalUri
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.post_process.impl.record_count_validating_post_processor import (
    RecordCountValidatingPostProcessor,
)
from snapchat.research.gbml import (
    gbml_config_pb2,
    graph_schema_pb2,
    preprocessed_metadata_pb2,
)
from tests.test_assets.test_case import TestCase


class RecordCountValidatingPostProcessorTest(TestCase):
    """Integration tests for RecordCountValidatingPostProcessor against real BigQuery."""

    def setUp(self):
        resource_config = get_resource_config()
        self._test_unique_name = f"gigl_rcv_test_{uuid.uuid4().hex[:12]}"

        self._bq_utils = BqUtils(project=resource_config.project)
        self._bq_project = resource_config.project
        self._bq_dataset = resource_config.temp_assets_bq_dataset_name
        self._proto_utils = ProtoUtils()

        self._tables_to_cleanup: list[str] = []
        self._validator = RecordCountValidatingPostProcessor()

    def tearDown(self):
        for table_path in self._tables_to_cleanup:
            self._bq_utils.delete_bq_table_if_exist(
                bq_table_path=table_path, not_found_ok=True
            )

    def _make_table_path(self, suffix: str) -> str:
        table_name = f"{self._test_unique_name}_{suffix}"
        return self._bq_utils.join_path(self._bq_project, self._bq_dataset, table_name)

    def _create_test_table(self, table_path: str, num_rows: int) -> None:
        """Create a BQ table with the specified number of rows and track for cleanup."""
        self._tables_to_cleanup.append(table_path)
        formatted = self._bq_utils.format_bq_path(table_path)
        create_query = f"""
        CREATE OR REPLACE TABLE `{formatted}` AS
        SELECT
            id,
            CONCAT('node_', CAST(id AS STRING)) as node_id
        FROM UNNEST(GENERATE_ARRAY(0, {num_rows - 1})) as id
        """
        self._bq_utils.run_query(query=create_query, labels={})
        time.sleep(1)

    def _build_gbml_config_wrapper(
        self,
        node_types: list[str],
        enumerated_tables: dict[str, str],
        embeddings_tables: dict[str, str] | None = None,
        predictions_tables: dict[str, str] | None = None,
    ) -> GbmlConfigPbWrapper:
        """Build a GbmlConfigPbWrapper with the given table paths.

        Args:
            node_types: List of string node types.
            enumerated_tables: Map of node_type -> enumerated_node_ids_bq_table path.
            embeddings_tables: Map of node_type -> embeddings BQ table path.
            predictions_tables: Map of node_type -> predictions BQ table path.
        """
        embeddings_tables = embeddings_tables or {}
        predictions_tables = predictions_tables or {}

        # Build graph metadata with condensed node type and edge type mappings.
        # GbmlConfigPbWrapper requires both maps to be non-empty.
        graph_metadata = graph_schema_pb2.GraphMetadata()
        for i, nt in enumerate(node_types):
            graph_metadata.node_types.append(nt)
            graph_metadata.condensed_node_type_map[i] = nt

        # Add a dummy edge type so condensed_edge_type_map is non-empty
        dummy_edge = graph_schema_pb2.EdgeType(
            src_node_type=node_types[0],
            relation="dummy",
            dst_node_type=node_types[0],
        )
        graph_metadata.edge_types.append(dummy_edge)
        graph_metadata.condensed_edge_type_map[0].CopyFrom(dummy_edge)

        # Build preprocessed metadata
        preprocessed_metadata = preprocessed_metadata_pb2.PreprocessedMetadata()
        for i, nt in enumerate(node_types):
            node_output = (
                preprocessed_metadata.condensed_node_type_to_preprocessed_metadata[i]
            )
            node_output.enumerated_node_ids_bq_table = enumerated_tables.get(nt, "")

        # Write preprocessed metadata to a temp file
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        preprocessed_metadata_uri = LocalUri(f.name)
        self._proto_utils.write_proto_to_yaml(
            proto=preprocessed_metadata, uri=preprocessed_metadata_uri
        )

        # Build the main GbmlConfig
        gbml_config = gbml_config_pb2.GbmlConfig()
        gbml_config.graph_metadata.CopyFrom(graph_metadata)
        gbml_config.shared_config.preprocessed_metadata_uri = (
            preprocessed_metadata_uri.uri
        )

        # Populate inference metadata
        for nt in node_types:
            inference_output = gbml_config.shared_config.inference_metadata.node_type_to_inferencer_output_info_map[
                nt
            ]
            if nt in embeddings_tables:
                inference_output.embeddings_path = embeddings_tables[nt]
            if nt in predictions_tables:
                inference_output.predictions_path = predictions_tables[nt]

        return GbmlConfigPbWrapper(gbml_config_pb=gbml_config)

    def test_validation_passes_when_counts_match(self):
        """All tables exist and row counts match — validation should pass."""
        num_rows = 25
        enum_table = self._make_table_path("enum_paper")
        emb_table = self._make_table_path("emb_paper")

        self._create_test_table(enum_table, num_rows)
        self._create_test_table(emb_table, num_rows)

        wrapper = self._build_gbml_config_wrapper(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": emb_table},
        )

        # Should not raise
        self._validator._validate_record_counts(
            gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
        )

    def test_validation_fails_on_row_count_mismatch(self):
        """Embeddings table has fewer rows — should raise ValueError."""
        enum_table = self._make_table_path("enum_paper")
        emb_table = self._make_table_path("emb_paper")

        self._create_test_table(enum_table, 50)
        self._create_test_table(emb_table, 30)

        wrapper = self._build_gbml_config_wrapper(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": emb_table},
        )

        with self.assertRaises(ValueError):
            self._validator._validate_record_counts(
                gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
            )

    def test_validation_fails_on_missing_embeddings_table(self):
        """embeddings_path is set but BQ table does not exist — should raise ValueError."""
        enum_table = self._make_table_path("enum_paper")
        self._create_test_table(enum_table, 25)

        nonexistent_emb_table = self._make_table_path("emb_nonexistent")
        # Do NOT create this table — it should not exist

        wrapper = self._build_gbml_config_wrapper(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": nonexistent_emb_table},
        )

        with self.assertRaises(ValueError):
            self._validator._validate_record_counts(
                gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
            )

    def test_validation_fails_on_missing_predictions_table(self):
        """predictions_path is set but BQ table does not exist — should raise ValueError."""
        enum_table = self._make_table_path("enum_paper")
        self._create_test_table(enum_table, 25)

        nonexistent_pred_table = self._make_table_path("pred_nonexistent")

        wrapper = self._build_gbml_config_wrapper(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            predictions_tables={"paper": nonexistent_pred_table},
        )

        with self.assertRaises(ValueError):
            self._validator._validate_record_counts(
                gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
            )

    def test_validation_fails_when_no_output_paths_set(self):
        """Neither embeddings nor predictions is set — should raise ValueError."""
        enum_table = self._make_table_path("enum_paper")
        self._create_test_table(enum_table, 25)

        wrapper = self._build_gbml_config_wrapper(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            # No embeddings or predictions
        )

        with self.assertRaises(ValueError):
            self._validator._validate_record_counts(
                gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
            )

    def test_validation_with_multiple_node_types(self):
        """Heterogeneous graph with multiple node types — each validated independently."""
        num_rows_author = 30
        num_rows_paper = 50

        enum_author = self._make_table_path("enum_author")
        emb_author = self._make_table_path("emb_author")
        enum_paper = self._make_table_path("enum_paper")
        emb_paper = self._make_table_path("emb_paper")

        self._create_test_table(enum_author, num_rows_author)
        self._create_test_table(emb_author, num_rows_author)
        self._create_test_table(enum_paper, num_rows_paper)
        self._create_test_table(emb_paper, num_rows_paper)

        wrapper = self._build_gbml_config_wrapper(
            node_types=["author", "paper"],
            enumerated_tables={
                "author": enum_author,
                "paper": enum_paper,
            },
            embeddings_tables={
                "author": emb_author,
                "paper": emb_paper,
            },
        )

        # Should not raise
        self._validator._validate_record_counts(
            gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
        )

    def test_validation_partial_output_only_embeddings(self):
        """Only embeddings (no predictions) for a node type — should pass."""
        num_rows = 40
        enum_table = self._make_table_path("enum_paper")
        emb_table = self._make_table_path("emb_paper")

        self._create_test_table(enum_table, num_rows)
        self._create_test_table(emb_table, num_rows)

        wrapper = self._build_gbml_config_wrapper(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": emb_table},
            # No predictions — that's fine
        )

        # Should not raise
        self._validator._validate_record_counts(
            gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
        )

    def test_multiple_errors_collected(self):
        """One table missing AND one count mismatch — both errors appear in exception."""
        enum_author = self._make_table_path("enum_author")
        emb_author = self._make_table_path("emb_author")
        enum_paper = self._make_table_path("enum_paper")

        self._create_test_table(enum_author, 30)
        self._create_test_table(emb_author, 20)  # Mismatch: 20 != 30
        self._create_test_table(enum_paper, 50)

        nonexistent_emb_paper = self._make_table_path("emb_paper_missing")

        wrapper = self._build_gbml_config_wrapper(
            node_types=["author", "paper"],
            enumerated_tables={
                "author": enum_author,
                "paper": enum_paper,
            },
            embeddings_tables={
                "author": emb_author,
                "paper": nonexistent_emb_paper,
            },
        )

        with self.assertRaises(ValueError) as ctx:
            self._validator._validate_record_counts(
                gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
            )

        error_message = str(ctx.exception)
        self.assertIn("2 error(s)", error_message)
        self.assertIn("[author]", error_message)
        self.assertIn("[paper]", error_message)

    def test_validation_passes_with_both_embeddings_and_predictions(self):
        """Both embeddings and predictions tables exist with matching counts."""
        num_rows = 35
        enum_table = self._make_table_path("enum_paper")
        emb_table = self._make_table_path("emb_paper")
        pred_table = self._make_table_path("pred_paper")

        self._create_test_table(enum_table, num_rows)
        self._create_test_table(emb_table, num_rows)
        self._create_test_table(pred_table, num_rows)

        wrapper = self._build_gbml_config_wrapper(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": emb_table},
            predictions_tables={"paper": pred_table},
        )

        # Should not raise
        self._validator._validate_record_counts(
            gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
        )

    def test_validation_fails_on_predictions_count_mismatch(self):
        """Predictions table has more rows — should raise ValueError."""
        enum_table = self._make_table_path("enum_paper")
        pred_table = self._make_table_path("pred_paper")

        self._create_test_table(enum_table, 40)
        self._create_test_table(pred_table, 60)

        wrapper = self._build_gbml_config_wrapper(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            predictions_tables={"paper": pred_table},
        )

        with self.assertRaises(ValueError):
            self._validator._validate_record_counts(
                gbml_config_wrapper=wrapper, bq_utils=self._bq_utils
            )


if __name__ == "__main__":
    absltest.main()
