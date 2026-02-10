import os
import tempfile
import time
import uuid

from absl.testing import absltest

from gigl.common import LocalUri
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types import AppliedTaskIdentifier
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
        self._temp_files_to_cleanup: list[str] = []

    def tearDown(self):
        for table_path in self._tables_to_cleanup:
            self._bq_utils.delete_bq_table_if_exist(
                bq_table_path=table_path, not_found_ok=True
            )
        for temp_file in self._temp_files_to_cleanup:
            if os.path.exists(temp_file):
                os.remove(temp_file)

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

    def _track_temp_file(self, path: str) -> None:
        """Track a temp file for cleanup in tearDown."""
        self._temp_files_to_cleanup.append(path)

    def _build_task_config_uri(
        self,
        node_types: list[str],
        enumerated_tables: dict[str, str],
        embeddings_tables: dict[str, str] | None = None,
        predictions_tables: dict[str, str] | None = None,
    ) -> LocalUri:
        """Build a GbmlConfig proto, write it to a temp YAML file, and return the URI.

        The config is set up so that PostProcessor._run() completes without side effects:
        - GLT backend is NOT enabled (skips unenumeration).
        - post_processor_cls_path is empty (skips user-defined post-processing).
        - should_skip_automatic_temp_asset_cleanup is true (skips GCS cleanup).

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

        # We add a dummy edge type because GbmlConfigPbWrapper requires
        # condensed_edge_type_map to be non-empty in order to initialize
        # the graph_metadata_pb_wrapper (see GbmlConfigPbWrapper.__load_graph_metadata_pb_wrapper,
        # gigl/src/common/types/pb_wrappers/gbml_config.py:181-184).
        # Our validator only checks node-level tables, not edge-level tables.
        dummy_edge = graph_schema_pb2.EdgeType(
            src_node_type=node_types[0],
            relation="dummy",
            dst_node_type=node_types[0],
        )
        graph_metadata.edge_types.append(dummy_edge)
        graph_metadata.condensed_edge_type_map[0].CopyFrom(dummy_edge)

        # Build preprocessed metadata and write to temp file
        preprocessed_metadata = preprocessed_metadata_pb2.PreprocessedMetadata()
        for i, nt in enumerate(node_types):
            node_output = (
                preprocessed_metadata.condensed_node_type_to_preprocessed_metadata[i]
            )
            node_output.enumerated_node_ids_bq_table = enumerated_tables.get(nt, "")

        preprocessed_metadata_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".yaml"
        )
        preprocessed_metadata_uri = LocalUri(preprocessed_metadata_file.name)
        self._track_temp_file(preprocessed_metadata_file.name)
        self._proto_utils.write_proto_to_yaml(
            proto=preprocessed_metadata, uri=preprocessed_metadata_uri
        )

        # Build the main GbmlConfig
        gbml_config = gbml_config_pb2.GbmlConfig()
        gbml_config.graph_metadata.CopyFrom(graph_metadata)
        gbml_config.shared_config.preprocessed_metadata_uri = (
            preprocessed_metadata_uri.uri
        )
        gbml_config.shared_config.should_skip_automatic_temp_asset_cleanup = True

        # Populate inference metadata
        for nt in node_types:
            inference_output = gbml_config.shared_config.inference_metadata.node_type_to_inferencer_output_info_map[
                nt
            ]
            if nt in embeddings_tables:
                inference_output.embeddings_path = embeddings_tables[nt]
            if nt in predictions_tables:
                inference_output.predictions_path = predictions_tables[nt]

        # Write GbmlConfig to temp file
        task_config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml")
        task_config_uri = LocalUri(task_config_file.name)
        self._track_temp_file(task_config_file.name)
        self._proto_utils.write_proto_to_yaml(proto=gbml_config, uri=task_config_uri)

        return task_config_uri

    def _run_validator(self, task_config_uri: LocalUri) -> None:
        """Run the RecordCountValidatingPostProcessor via its public run() API."""
        validator = RecordCountValidatingPostProcessor()
        validator.run(
            applied_task_identifier=AppliedTaskIdentifier(self._test_unique_name),
            task_config_uri=task_config_uri,
            resource_config_uri=get_resource_config().get_resource_config_uri,
        )

    def test_validation_passes_when_counts_match(self):
        """All tables exist and row counts match — validation should pass."""
        num_rows = 25
        enum_table = self._make_table_path("enum_paper")
        emb_table = self._make_table_path("emb_paper")

        self._create_test_table(enum_table, num_rows)
        self._create_test_table(emb_table, num_rows)

        task_config_uri = self._build_task_config_uri(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": emb_table},
        )

        # Should not raise
        self._run_validator(task_config_uri)

    def test_validation_fails_on_row_count_mismatch(self):
        """Embeddings table has fewer rows — should raise SystemExit."""
        enum_table = self._make_table_path("enum_paper")
        emb_table = self._make_table_path("emb_paper")

        self._create_test_table(enum_table, 50)
        self._create_test_table(emb_table, 30)

        task_config_uri = self._build_task_config_uri(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": emb_table},
        )

        with self.assertRaises(SystemExit):
            self._run_validator(task_config_uri)

    def test_validation_fails_on_missing_embeddings_table(self):
        """embeddings_path is set but BQ table does not exist — should raise SystemExit."""
        enum_table = self._make_table_path("enum_paper")
        self._create_test_table(enum_table, 25)

        nonexistent_emb_table = self._make_table_path("emb_nonexistent")
        # Do NOT create this table — it should not exist

        task_config_uri = self._build_task_config_uri(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": nonexistent_emb_table},
        )

        with self.assertRaises(SystemExit):
            self._run_validator(task_config_uri)

    def test_validation_fails_on_missing_predictions_table(self):
        """predictions_path is set but BQ table does not exist — should raise SystemExit."""
        enum_table = self._make_table_path("enum_paper")
        self._create_test_table(enum_table, 25)

        nonexistent_pred_table = self._make_table_path("pred_nonexistent")

        task_config_uri = self._build_task_config_uri(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            predictions_tables={"paper": nonexistent_pred_table},
        )

        with self.assertRaises(SystemExit):
            self._run_validator(task_config_uri)

    def test_validation_fails_when_no_output_paths_set(self):
        """Neither embeddings nor predictions is set — should raise SystemExit."""
        enum_table = self._make_table_path("enum_paper")
        self._create_test_table(enum_table, 25)

        task_config_uri = self._build_task_config_uri(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            # No embeddings or predictions
        )

        with self.assertRaises(SystemExit):
            self._run_validator(task_config_uri)

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

        task_config_uri = self._build_task_config_uri(
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
        self._run_validator(task_config_uri)

    def test_validation_partial_output_only_embeddings(self):
        """Only embeddings (no predictions) for a node type — should pass."""
        num_rows = 40
        enum_table = self._make_table_path("enum_paper")
        emb_table = self._make_table_path("emb_paper")

        self._create_test_table(enum_table, num_rows)
        self._create_test_table(emb_table, num_rows)

        task_config_uri = self._build_task_config_uri(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": emb_table},
            # No predictions — that's fine
        )

        # Should not raise
        self._run_validator(task_config_uri)

    def test_multiple_errors_collected(self):
        """One table missing AND one count mismatch — both errors appear in exception."""
        enum_author = self._make_table_path("enum_author")
        emb_author = self._make_table_path("emb_author")
        enum_paper = self._make_table_path("enum_paper")

        self._create_test_table(enum_author, 30)
        self._create_test_table(emb_author, 20)  # Mismatch: 20 != 30
        self._create_test_table(enum_paper, 50)

        nonexistent_emb_paper = self._make_table_path("emb_paper_missing")

        task_config_uri = self._build_task_config_uri(
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

        with self.assertRaises(SystemExit) as ctx:
            self._run_validator(task_config_uri)

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

        task_config_uri = self._build_task_config_uri(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            embeddings_tables={"paper": emb_table},
            predictions_tables={"paper": pred_table},
        )

        # Should not raise
        self._run_validator(task_config_uri)

    def test_validation_fails_on_predictions_count_mismatch(self):
        """Predictions table has more rows — should raise SystemExit."""
        enum_table = self._make_table_path("enum_paper")
        pred_table = self._make_table_path("pred_paper")

        self._create_test_table(enum_table, 40)
        self._create_test_table(pred_table, 60)

        task_config_uri = self._build_task_config_uri(
            node_types=["paper"],
            enumerated_tables={"paper": enum_table},
            predictions_tables={"paper": pred_table},
        )

        with self.assertRaises(SystemExit):
            self._run_validator(task_config_uri)


if __name__ == "__main__":
    absltest.main()
