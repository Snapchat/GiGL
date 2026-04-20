from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING, OmegaConf

from gigl.common.logger import Logger

logger = Logger()


@dataclass
class NodeTableSpec:
    """Specification for a node table in BigQuery."""

    bq_table: str = MISSING
    node_type: str = MISSING
    id_column: str = MISSING
    feature_columns: list[str] = field(default_factory=list)
    label_column: Optional[str] = None


@dataclass
class EdgeTableSpec:
    """Specification for an edge table in BigQuery."""

    bq_table: str = MISSING
    edge_type: str = MISSING
    src_id_column: str = MISSING
    dst_id_column: str = MISSING
    feature_columns: list[str] = field(default_factory=list)
    timestamp_column: Optional[str] = None


@dataclass
class DataAnalyzerConfig:
    """Configuration for the BQ Data Analyzer.

    Parsed from YAML via OmegaConf.

    Example:
        >>> config = load_analyzer_config("gs://bucket/config.yaml")
        >>> config.node_tables[0].bq_table
        'project.dataset.user_nodes'
    """

    node_tables: list[NodeTableSpec] = MISSING
    edge_tables: list[EdgeTableSpec] = MISSING
    output_gcs_path: str = MISSING
    fan_out: Optional[list[int]] = None
    compute_reciprocity: bool = False
    compute_homophily: bool = False
    compute_connected_components: bool = False
    compute_clustering: bool = False


def load_analyzer_config(config_path: str) -> DataAnalyzerConfig:
    """Load and validate a DataAnalyzerConfig from a YAML file.

    Args:
        config_path: Local file path or GCS URI to the YAML config.

    Returns:
        Validated DataAnalyzerConfig instance.

    Raises:
        omegaconf.errors.MissingMandatoryValue: If required fields are missing.
    """
    raw = OmegaConf.load(config_path)
    merged = OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), raw)
    config: DataAnalyzerConfig = OmegaConf.to_object(merged)  # type: ignore
    logger.info(
        f"Loaded analyzer config with {len(config.node_tables)} node tables "
        f"and {len(config.edge_tables)} edge tables"
    )
    return config
