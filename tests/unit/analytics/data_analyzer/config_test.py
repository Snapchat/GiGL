from pathlib import Path

from omegaconf import OmegaConf

from gigl.analytics.data_analyzer.config import (
    DataAnalyzerConfig,
    load_analyzer_config,
)
from tests.test_assets.test_case import TestCase

SAMPLE_CONFIG_PATH = (
    Path(__file__).parents[3] / "test_assets" / "analytics" / "sample_analyzer_config.yaml"
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
        config = OmegaConf.to_object(merged)
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
        with self.assertRaises(Exception):
            merged = OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), raw)
            OmegaConf.to_object(merged)
