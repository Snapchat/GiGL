import re
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING, OmegaConf

from gigl.common.logger import Logger

logger = Logger()

# BigQuery identifier regexes used to reject configs that would be interpolated
# directly into SQL. See https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical
# for the allowed grammar. Tables are of the form project.dataset.table;
# columns are simple unquoted identifiers.
_BQ_TABLE_REGEX = re.compile(r"^[A-Za-z0-9_.\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_$\-]+$")
_BQ_COLUMN_REGEX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_bq_table(name: str, field_label: str) -> None:
    if not _BQ_TABLE_REGEX.fullmatch(name):
        raise ValueError(
            f"{field_label}={name!r} is not a valid BigQuery table reference. "
            f"Expected project.dataset.table with no backticks, whitespace, or quotes."
        )


def _validate_bq_column(name: str, field_label: str) -> None:
    if not _BQ_COLUMN_REGEX.fullmatch(name):
        raise ValueError(
            f"{field_label}={name!r} is not a valid BigQuery column identifier. "
            f"Expected [A-Za-z_][A-Za-z0-9_]* with no backticks, whitespace, or quotes."
        )


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
    """Specification for an edge table in BigQuery.

    For heterogeneous graphs (more than one node table), src_node_type and
    dst_node_type must be set to the node_type of the matching node table.
    For homogeneous graphs (single node table) they default to that node_type.
    """

    bq_table: str = MISSING
    edge_type: str = MISSING
    src_id_column: str = MISSING
    dst_id_column: str = MISSING
    src_node_type: Optional[str] = None
    dst_node_type: Optional[str] = None
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


def _validate_and_backfill(config: DataAnalyzerConfig) -> None:
    """Run identifier validation and backfill default node-type references.

    - Every bq_table must match project.dataset.table.
    - Every id_column / src_id_column / dst_id_column / feature_column /
      label_column / timestamp_column must be a bare BQ identifier.
    - For homogeneous configs, an edge table with no src_node_type /
      dst_node_type inherits the single node table's node_type.
    - For heterogeneous configs, every edge table must explicitly declare
      src_node_type and dst_node_type, and both must resolve to a known
      node_type.
    """
    known_node_types = {nt.node_type for nt in config.node_tables}
    single_node_type: Optional[str] = (
        next(iter(known_node_types)) if len(config.node_tables) == 1 else None
    )

    for node_table in config.node_tables:
        _validate_bq_table(node_table.bq_table, "node_tables.bq_table")
        _validate_bq_column(node_table.id_column, "node_tables.id_column")
        for col in node_table.feature_columns:
            _validate_bq_column(col, "node_tables.feature_columns")
        if node_table.label_column is not None:
            _validate_bq_column(node_table.label_column, "node_tables.label_column")

    for edge_table in config.edge_tables:
        _validate_bq_table(edge_table.bq_table, "edge_tables.bq_table")
        _validate_bq_column(edge_table.src_id_column, "edge_tables.src_id_column")
        _validate_bq_column(edge_table.dst_id_column, "edge_tables.dst_id_column")
        for col in edge_table.feature_columns:
            _validate_bq_column(col, "edge_tables.feature_columns")
        if edge_table.timestamp_column is not None:
            _validate_bq_column(
                edge_table.timestamp_column, "edge_tables.timestamp_column"
            )

        if edge_table.src_node_type is None:
            if single_node_type is not None:
                edge_table.src_node_type = single_node_type
            else:
                raise ValueError(
                    f"edge_type={edge_table.edge_type}: src_node_type is required "
                    f"when there are multiple node tables"
                )
        if edge_table.dst_node_type is None:
            if single_node_type is not None:
                edge_table.dst_node_type = single_node_type
            else:
                raise ValueError(
                    f"edge_type={edge_table.edge_type}: dst_node_type is required "
                    f"when there are multiple node tables"
                )
        if edge_table.src_node_type not in known_node_types:
            raise ValueError(
                f"edge_type={edge_table.edge_type}: src_node_type="
                f"{edge_table.src_node_type!r} is not a declared node_type. "
                f"Known: {sorted(known_node_types)}"
            )
        if edge_table.dst_node_type not in known_node_types:
            raise ValueError(
                f"edge_type={edge_table.edge_type}: dst_node_type="
                f"{edge_table.dst_node_type!r} is not a declared node_type. "
                f"Known: {sorted(known_node_types)}"
            )


def load_analyzer_config(config_path: str) -> DataAnalyzerConfig:
    """Load and validate a DataAnalyzerConfig from a YAML file.

    Args:
        config_path: Local file path or GCS URI to the YAML config.

    Returns:
        Validated DataAnalyzerConfig instance with node-type references
        backfilled on edge tables.

    Raises:
        omegaconf.errors.MissingMandatoryValue: If required fields are missing.
        ValueError: If any bq_table or column name is not a valid BigQuery
            identifier, or if a heterogeneous config is missing a required
            src_node_type / dst_node_type.
    """
    raw = OmegaConf.load(config_path)
    merged = OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), raw)
    config: DataAnalyzerConfig = OmegaConf.to_object(merged)  # type: ignore
    _validate_and_backfill(config)
    logger.info(
        f"Loaded analyzer config with {len(config.node_tables)} node tables "
        f"and {len(config.edge_tables)} edge tables"
    )
    return config
