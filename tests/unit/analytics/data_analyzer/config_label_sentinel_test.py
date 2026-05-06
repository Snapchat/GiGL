"""Schema validation tests for the NC supervision additions to NodeTableSpec.

Exercises ``label_sentinel_values``, ``split_column``, and the three new
``DataAnalyzerConfig`` flags (``compute_per_class_feature_stats``,
``compute_label_informativeness``, ``label_homophily_edge_sample_cap``).
"""

import tempfile
from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from gigl.analytics.data_analyzer.config import DataAnalyzerConfig, load_analyzer_config
from tests.test_assets.test_case import TestCase


def _write_yaml(yaml_str: str) -> str:
    """Write a YAML string to a temp file and return its absolute path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(yaml_str)
    tmp.flush()
    tmp.close()
    return tmp.name


class LabelSentinelConfigTest(TestCase):
    def test_default_label_sentinel_values_is_empty_list(self) -> None:
        yaml_str = """
        node_tables:
          - bq_table: "p.d.t"
            node_type: "user"
            id_column: "uid"
            label_column: "label"
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
        self.assertEqual(config.node_tables[0].label_sentinel_values, [])
        self.assertIsNone(config.node_tables[0].split_column)

    def test_loads_label_sentinel_values_and_split_column(self) -> None:
        path = _write_yaml(
            """
            node_tables:
              - bq_table: "p.d.t"
                node_type: "user"
                id_column: "uid"
                label_column: "node_label"
                label_sentinel_values:
                  - "-1"
                  - "unknown"
                split_column: "split"
            edge_tables:
              - bq_table: "p.d.e"
                edge_type: "follows"
                src_id_column: "src"
                dst_id_column: "dst"
            output_gcs_path: "gs://bucket/out/"
            """
        )
        try:
            config = load_analyzer_config(path)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(config.node_tables[0].label_sentinel_values, ["-1", "unknown"])
        self.assertEqual(config.node_tables[0].split_column, "split")

    def test_empty_sentinel_string_rejected(self) -> None:
        path = _write_yaml(
            """
            node_tables:
              - bq_table: "p.d.t"
                node_type: "user"
                id_column: "uid"
                label_column: "node_label"
                label_sentinel_values:
                  - ""
            edge_tables:
              - bq_table: "p.d.e"
                edge_type: "follows"
                src_id_column: "src"
                dst_id_column: "dst"
            output_gcs_path: "gs://bucket/out/"
            """
        )
        try:
            with self.assertRaises(ValueError):
                load_analyzer_config(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_sentinel_without_label_column_rejected(self) -> None:
        """Sentinels apply to label_column only — declaring them without it is a bug."""
        path = _write_yaml(
            """
            node_tables:
              - bq_table: "p.d.t"
                node_type: "user"
                id_column: "uid"
                label_sentinel_values:
                  - "-1"
            edge_tables:
              - bq_table: "p.d.e"
                edge_type: "follows"
                src_id_column: "src"
                dst_id_column: "dst"
            output_gcs_path: "gs://bucket/out/"
            """
        )
        try:
            with self.assertRaises(ValueError):
                load_analyzer_config(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_invalid_split_column_identifier_rejected(self) -> None:
        path = _write_yaml(
            """
            node_tables:
              - bq_table: "p.d.t"
                node_type: "user"
                id_column: "uid"
                label_column: "label"
                split_column: "bad column"
            edge_tables:
              - bq_table: "p.d.e"
                edge_type: "follows"
                src_id_column: "src"
                dst_id_column: "dst"
            output_gcs_path: "gs://bucket/out/"
            """
        )
        try:
            with self.assertRaises(ValueError):
                load_analyzer_config(path)
        finally:
            Path(path).unlink(missing_ok=True)


class DataAnalyzerConfigFlagDefaultsTest(TestCase):
    def test_nc_flag_defaults(self) -> None:
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
        # Per-class feature stats default on (cheap; highest-value NC signal).
        self.assertTrue(config.compute_per_class_feature_stats)
        # Label informativeness default off (expensive full-graph join).
        self.assertFalse(config.compute_label_informativeness)
        # Default sample cap is 50M edges.
        self.assertEqual(config.label_homophily_edge_sample_cap, 50_000_000)

    def test_overriding_homophily_sample_cap(self) -> None:
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
        label_homophily_edge_sample_cap: 0
        compute_label_informativeness: true
        compute_per_class_feature_stats: false
        """
        raw = OmegaConf.create(yaml_str)
        merged = OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), raw)
        config = cast(DataAnalyzerConfig, OmegaConf.to_object(merged))
        self.assertEqual(config.label_homophily_edge_sample_cap, 0)
        self.assertTrue(config.compute_label_informativeness)
        self.assertFalse(config.compute_per_class_feature_stats)
