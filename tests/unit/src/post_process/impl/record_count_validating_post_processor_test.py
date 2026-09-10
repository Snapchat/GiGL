import tempfile
from unittest.mock import MagicMock, patch

from gigl.common import LocalUri
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.post_process.impl.record_count_validating_post_processor import (
    RecordCountValidatingPostProcessor,
)
from snapchat.research.gbml import (
    gbml_config_pb2,
    graph_schema_pb2,
    preprocessed_metadata_pb2,
)
from tests.test_assets.test_case import TestCase


class RecordCountValidatingPostProcessorAdapterTest(TestCase):
    """Unit tests verifying _validate_record_counts correctly derives dicts from GbmlConfigPbWrapper."""

    def setUp(self) -> None:
        self._proto_utils = ProtoUtils()
        self._temp_files: list[str] = []

    def tearDown(self) -> None:
        import os

        for f in self._temp_files:
            if os.path.exists(f):
                os.remove(f)

    def _build_gbml_config_pb(
        self,
        node_types: list[str],
        enumerated_tables: dict[str, str],
        embeddings_tables: dict[str, str] | None = None,
        predictions_tables: dict[str, str] | None = None,
    ) -> gbml_config_pb2.GbmlConfig:
        """Build a GbmlConfig proto for testing the adapter layer.

        Uses the same construction pattern as the integration tests at
        tests/integration/src/post_process/impl/record_count_validating_post_processor_test.py:70-148.
        """
        embeddings_tables = embeddings_tables or {}
        predictions_tables = predictions_tables or {}

        graph_metadata = graph_schema_pb2.GraphMetadata()
        for i, nt in enumerate(node_types):
            graph_metadata.node_types.append(nt)
            graph_metadata.condensed_node_type_map[i] = nt

        dummy_edge = graph_schema_pb2.EdgeType(
            src_node_type=node_types[0],
            relation="dummy",
            dst_node_type=node_types[0],
        )
        graph_metadata.edge_types.append(dummy_edge)
        graph_metadata.condensed_edge_type_map[0].CopyFrom(dummy_edge)

        preprocessed_metadata = preprocessed_metadata_pb2.PreprocessedMetadata()
        for i, nt in enumerate(node_types):
            node_output = (
                preprocessed_metadata.condensed_node_type_to_preprocessed_metadata[i]
            )
            node_output.enumerated_node_ids_bq_table = enumerated_tables.get(nt, "")

        preprocessed_metadata_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".yaml"
        )
        self._temp_files.append(preprocessed_metadata_file.name)
        self._proto_utils.write_proto_to_yaml(
            proto=preprocessed_metadata,
            uri=LocalUri(preprocessed_metadata_file.name),
        )

        gbml_config = gbml_config_pb2.GbmlConfig()
        gbml_config.graph_metadata.CopyFrom(graph_metadata)
        gbml_config.shared_config.preprocessed_metadata_uri = (
            preprocessed_metadata_file.name
        )
        gbml_config.shared_config.should_skip_automatic_temp_asset_cleanup = True

        for nt in node_types:
            inference_output = gbml_config.shared_config.inference_metadata.node_type_to_inferencer_output_info_map[
                nt
            ]
            if nt in embeddings_tables:
                inference_output.embeddings_path = embeddings_tables[nt]
            if nt in predictions_tables:
                inference_output.predictions_path = predictions_tables[nt]

        return gbml_config

    @patch(
        "gigl.src.post_process.impl.record_count_validating_post_processor.validate_node_output_records"
    )
    def test_single_node_type_with_embeddings(self, mock_validate: MagicMock) -> None:
        gbml_config_pb = self._build_gbml_config_pb(
            node_types=["paper"],
            enumerated_tables={"paper": "project.dataset.enum_paper"},
            embeddings_tables={"paper": "project.dataset.emb_paper"},
        )

        validator = RecordCountValidatingPostProcessor(
            applied_task_identifier=AppliedTaskIdentifier("test")
        )
        validator._validate_record_counts(
            gbml_config_wrapper=self._wrap_config(gbml_config_pb),
            bq_utils=MagicMock(),
        )

        mock_validate.assert_called_once()
        call_kwargs = mock_validate.call_args.kwargs
        self.assertEqual(
            call_kwargs["expected_count_tables"],
            {"paper": "project.dataset.enum_paper"},
        )
        self.assertEqual(
            call_kwargs["embeddings_tables"],
            {"paper": "project.dataset.emb_paper"},
        )
        self.assertEqual(call_kwargs["predictions_tables"], {})

    @patch(
        "gigl.src.post_process.impl.record_count_validating_post_processor.validate_node_output_records"
    )
    def test_single_node_type_with_both_outputs(self, mock_validate: MagicMock) -> None:
        gbml_config_pb = self._build_gbml_config_pb(
            node_types=["paper"],
            enumerated_tables={"paper": "project.dataset.enum_paper"},
            embeddings_tables={"paper": "project.dataset.emb_paper"},
            predictions_tables={"paper": "project.dataset.pred_paper"},
        )

        validator = RecordCountValidatingPostProcessor(
            applied_task_identifier=AppliedTaskIdentifier("test")
        )
        validator._validate_record_counts(
            gbml_config_wrapper=self._wrap_config(gbml_config_pb),
            bq_utils=MagicMock(),
        )

        call_kwargs = mock_validate.call_args.kwargs
        self.assertEqual(
            call_kwargs["embeddings_tables"],
            {"paper": "project.dataset.emb_paper"},
        )
        self.assertEqual(
            call_kwargs["predictions_tables"],
            {"paper": "project.dataset.pred_paper"},
        )

    @patch(
        "gigl.src.post_process.impl.record_count_validating_post_processor.validate_node_output_records"
    )
    def test_blank_embeddings_path_filtered_out(self, mock_validate: MagicMock) -> None:
        """Blank proto string defaults should not appear in embeddings_tables."""
        gbml_config_pb = self._build_gbml_config_pb(
            node_types=["paper"],
            enumerated_tables={"paper": "project.dataset.enum_paper"},
            # No embeddings — proto default is ""
        )

        validator = RecordCountValidatingPostProcessor(
            applied_task_identifier=AppliedTaskIdentifier("test")
        )
        validator._validate_record_counts(
            gbml_config_wrapper=self._wrap_config(gbml_config_pb),
            bq_utils=MagicMock(),
        )

        call_kwargs = mock_validate.call_args.kwargs
        self.assertEqual(call_kwargs["embeddings_tables"], {})
        self.assertEqual(call_kwargs["predictions_tables"], {})

    @patch(
        "gigl.src.post_process.impl.record_count_validating_post_processor.validate_node_output_records"
    )
    def test_multiple_node_types(self, mock_validate: MagicMock) -> None:
        gbml_config_pb = self._build_gbml_config_pb(
            node_types=["author", "paper"],
            enumerated_tables={
                "author": "project.dataset.enum_author",
                "paper": "project.dataset.enum_paper",
            },
            embeddings_tables={
                "author": "project.dataset.emb_author",
                "paper": "project.dataset.emb_paper",
            },
        )

        validator = RecordCountValidatingPostProcessor(
            applied_task_identifier=AppliedTaskIdentifier("test")
        )
        validator._validate_record_counts(
            gbml_config_wrapper=self._wrap_config(gbml_config_pb),
            bq_utils=MagicMock(),
        )

        call_kwargs = mock_validate.call_args.kwargs
        self.assertEqual(
            call_kwargs["expected_count_tables"],
            {
                "author": "project.dataset.enum_author",
                "paper": "project.dataset.enum_paper",
            },
        )
        self.assertEqual(
            call_kwargs["embeddings_tables"],
            {
                "author": "project.dataset.emb_author",
                "paper": "project.dataset.emb_paper",
            },
        )

    def _wrap_config(
        self, gbml_config_pb: gbml_config_pb2.GbmlConfig
    ) -> GbmlConfigPbWrapper:
        return GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
