from pathlib import Path
from typing import cast

from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from gigl.analytics.data_analyzer.config import DataAnalyzerConfig, load_analyzer_config
from tests.test_assets.test_case import TestCase

SAMPLE_CONFIG_PATH = (
    Path(__file__).parents[3]
    / "test_assets"
    / "analytics"
    / "sample_analyzer_config.yaml"
)


class DataAnalyzerConfigTest(TestCase):
    def test_load_valid_config(self) -> None:
        config = load_analyzer_config(str(SAMPLE_CONFIG_PATH))
        self.assertIsInstance(config, DataAnalyzerConfig)
        self.assertEqual(len(config.node_tables), 1)
        self.assertEqual(len(config.edge_tables), 1)
        self.assertEqual(config.node_tables[0].node_type, "user")
        self.assertEqual(config.node_tables[0].label_column, "label")
        self.assertEqual(config.edge_tables[0].edge_type, "follows")
        self.assertEqual(config.output_gcs_path, "gs://test-bucket/analysis_output/")
        self.assertEqual(config.fan_out, [15, 10, 5])

    def test_optional_fields_default_to_none_or_false(self) -> None:
        yaml_str = """
        node_tables:
          - bq_table: "p.d.t"
            node_type: "user"
            id_column: "uid"
            feature_columns: ["f1"]
        edge_tables:
          - bq_table: "p.d.e"
            edge_type: "follows"
            src_id_column: "src"
            dst_id_column: "dst"
        output_gcs_path: "gs://bucket/out/"
        """
        raw = OmegaConf.create(yaml_str)
        merged = OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), raw)
        config = cast(DataAnalyzerConfig, OmegaConf.to_object(merged))
        self.assertIsNone(config.node_tables[0].label_column)
        self.assertIsNone(config.edge_tables[0].timestamp_column)
        self.assertIsNone(config.fan_out)
        self.assertFalse(config.compute_reciprocity)
        self.assertFalse(config.compute_homophily)

    def test_missing_required_field_raises(self) -> None:
        yaml_str = """
        node_tables:
          - bq_table: "p.d.t"
            node_type: "user"
        edge_tables: []
        output_gcs_path: "gs://bucket/out/"
        """
        raw = OmegaConf.create(yaml_str)
        with self.assertRaises(MissingMandatoryValue):
            merged = OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), raw)
            OmegaConf.to_object(merged)

    def test_node_table_without_feature_columns(self) -> None:
        """Nodes with no features are legal; feature_columns defaults to []."""
        yaml_str = """
        node_tables:
          - bq_table: "p.d.t"
            node_type: "user"
            id_column: "uid"
        edge_tables:
          - bq_table: "p.d.e"
            edge_type: "follows"
            src_id_column: "src"
            dst_id_column: "dst"
        output_gcs_path: "gs://bucket/out/"
        """
        raw = OmegaConf.create(yaml_str)
        merged = OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), raw)
        config = cast(DataAnalyzerConfig, OmegaConf.to_object(merged))
        self.assertEqual(config.node_tables[0].feature_columns, [])

    def test_homogeneous_edge_backfills_src_and_dst_node_type(self) -> None:
        """Single-node-table configs auto-populate src/dst node types."""
        config = load_analyzer_config(str(SAMPLE_CONFIG_PATH))
        self.assertEqual(config.edge_tables[0].src_node_type, "user")
        self.assertEqual(config.edge_tables[0].dst_node_type, "user")


SAMPLE_HETERO_YAML = """
node_tables:
  - bq_table: "p.d.users"
    node_type: "user"
    id_column: "uid"
  - bq_table: "p.d.content"
    node_type: "content"
    id_column: "cid"
edge_tables:
  - bq_table: "p.d.viewed"
    edge_type: "viewed"
    src_id_column: "user_id"
    dst_id_column: "content_id"
    src_node_type: "user"
    dst_node_type: "content"
output_gcs_path: "gs://bucket/out/"
"""


class DataAnalyzerConfigHeterogeneousTest(TestCase):
    """Tests for heterogeneous graph support (I3) and identifier validation (I1)."""

    def _write_yaml(self, yaml_str: str) -> str:
        import tempfile

        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        fd.write(yaml_str)
        fd.close()
        return fd.name

    def test_heterogeneous_config_with_node_types_loads(self) -> None:
        path = self._write_yaml(SAMPLE_HETERO_YAML)
        config = load_analyzer_config(path)
        self.assertEqual(len(config.node_tables), 2)
        self.assertEqual(config.edge_tables[0].src_node_type, "user")
        self.assertEqual(config.edge_tables[0].dst_node_type, "content")

    def test_heterogeneous_missing_src_node_type_raises(self) -> None:
        """Regression test for I3: multi-node-table configs must declare both sides."""
        yaml_str = SAMPLE_HETERO_YAML.replace('    src_node_type: "user"\n', "")
        path = self._write_yaml(yaml_str)
        with self.assertRaises(ValueError) as ctx:
            load_analyzer_config(path)
        self.assertIn("src_node_type is required", str(ctx.exception))

    def test_heterogeneous_unknown_node_type_raises(self) -> None:
        yaml_str = SAMPLE_HETERO_YAML.replace(
            '    dst_node_type: "content"', '    dst_node_type: "movie"'
        )
        path = self._write_yaml(yaml_str)
        with self.assertRaises(ValueError) as ctx:
            load_analyzer_config(path)
        self.assertIn("is not a declared node_type", str(ctx.exception))

    def test_invalid_bq_table_reference_raises(self) -> None:
        """Regression test for I1: reject malformed table identifiers."""
        yaml_str = SAMPLE_HETERO_YAML.replace(
            'bq_table: "p.d.users"', 'bq_table: "p.d.users; DROP TABLE x"'
        )
        path = self._write_yaml(yaml_str)
        with self.assertRaises(ValueError) as ctx:
            load_analyzer_config(path)
        self.assertIn("not a valid BigQuery table reference", str(ctx.exception))

    def test_invalid_column_identifier_raises(self) -> None:
        """Regression test for I1: reject column names with backticks/quotes."""
        yaml_str = SAMPLE_HETERO_YAML.replace(
            'src_id_column: "user_id"', 'src_id_column: "user`id"'
        )
        path = self._write_yaml(yaml_str)
        with self.assertRaises(ValueError) as ctx:
            load_analyzer_config(path)
        self.assertIn("not a valid BigQuery column identifier", str(ctx.exception))

    def test_column_with_whitespace_raises(self) -> None:
        """Regression test for I1: reject column names containing whitespace."""
        yaml_str = SAMPLE_HETERO_YAML.replace(
            'dst_id_column: "content_id"', 'dst_id_column: "content id"'
        )
        path = self._write_yaml(yaml_str)
        with self.assertRaises(ValueError) as ctx:
            load_analyzer_config(path)
        self.assertIn("not a valid BigQuery column identifier", str(ctx.exception))
