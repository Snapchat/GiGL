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
    """Specification for a node table in BigQuery.

    Node-classification supervision is activated when ``label_column`` is
    set. ``label_sentinel_values`` lets users distinguish "missing" labels
    encoded as ``-1`` / ``"unknown"`` from SQL NULL — both are excluded
    from the valid-label denominator used by class-imbalance and
    homophily computations, but are reported separately so the upstream
    bug can be traced. ``split_column`` enables split-validation checks
    (cross-split node-id leakage as a Tier 1 hard fail, plus per-split
    TFDV slicing for distribution drift).
    """

    bq_table: str = MISSING
    node_type: str = MISSING
    id_column: str = MISSING
    feature_columns: list[str] = field(default_factory=list)
    label_column: Optional[str] = None
    label_sentinel_values: list[str] = field(default_factory=list)
    split_column: Optional[str] = None


EDGE_ROLE_MESSAGE_PASSING = "message_passing"
EDGE_ROLE_SUPERVISION_POS = "supervision_pos"
EDGE_ROLE_SUPERVISION_NEG = "supervision_neg"
_VALID_EDGE_ROLES = frozenset(
    {EDGE_ROLE_MESSAGE_PASSING, EDGE_ROLE_SUPERVISION_POS, EDGE_ROLE_SUPERVISION_NEG}
)


@dataclass
class EdgeTableSpec:
    """Specification for an edge table in BigQuery.

    For heterogeneous graphs (more than one node table), src_node_type and
    dst_node_type must be set to the node_type of the matching node table.
    For homogeneous graphs (single node table) they default to that node_type.

    ``role`` marks the table's purpose for cross-table supervision analysis.
    Defaults to ``"message_passing"`` when omitted. ``node_anchor`` selects
    which side (src or dst) of the table is the anchor for the per-anchor
    cross-table stats; required on ``supervision_pos`` tables, ignored when
    no analysis applies.
    """

    bq_table: str = MISSING
    edge_type: str = MISSING
    src_id_column: str = MISSING
    dst_id_column: str = MISSING
    src_node_type: Optional[str] = None
    dst_node_type: Optional[str] = None
    feature_columns: list[str] = field(default_factory=list)
    timestamp_column: Optional[str] = None
    role: Optional[str] = None
    node_anchor: Optional[str] = None


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

    # Node-classification supervision tier flags. Activate any time a
    # NodeTableSpec.label_column is set.
    #
    # ``compute_per_class_feature_stats`` controls TFDV slicing on the
    # label column inside the feature profiler — default on because it's
    # the highest-value NC-specific feature signal and costs one extra
    # column on the existing BQ projection.
    #
    # ``compute_label_informativeness`` is the expensive (full-graph
    # mutual-information) homophily measure from Platonov et al. 2023.
    # Default off; the cheaper sampled adjusted-homophily always runs.
    #
    # ``label_homophily_edge_sample_cap`` caps the message-passing edge
    # sample used to compute adjusted homophily. ``0`` means full-graph.
    compute_per_class_feature_stats: bool = True
    compute_label_informativeness: bool = False
    label_homophily_edge_sample_cap: int = 50_000_000

    # Per-chunk feature cap for TFDV profiling. Wide projections explode
    # Beam 2.56's CombinePerKey state and trip
    # "Instruction id ... was not registered" failures on Runner v2;
    # chunking keeps every Dataflow job within the runner's
    # state-iteration budget. 350 was validated end-to-end on a 706-col /
    # ~950 M-row user table.
    max_features_per_chunk: int = 350

    # Per-config Dataflow job name prefix. Combined with a per-run
    # timestamp at the entry point to keep concurrent / repeated runs
    # from colliding on the fixed Dataflow job name. The CLI flag
    # ``--job_name_prefix`` overrides this when set; if neither is set
    # the entry point fails fast.
    job_name_prefix: Optional[str] = None


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
        if node_table.split_column is not None:
            _validate_bq_column(node_table.split_column, "node_tables.split_column")
        # Sentinel values are not SQL identifiers (they're literal label
        # values), but they're still embedded into SQL via parameterized
        # IN clauses elsewhere. Reject empty strings to fail fast on
        # likely-misconfigured YAML where a value got stripped.
        for sentinel in node_table.label_sentinel_values:
            if sentinel == "":
                raise ValueError(
                    f"node_tables.label_sentinel_values contains an empty string "
                    f"for node_type={node_table.node_type!r}; declare each "
                    "sentinel value explicitly (e.g. '-1', 'unknown')."
                )
        if node_table.label_sentinel_values and node_table.label_column is None:
            raise ValueError(
                f"node_type={node_table.node_type!r}: label_sentinel_values "
                "are declared but label_column is not set; sentinels apply "
                "to the label_column only."
            )

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

        if edge_table.role is None:
            edge_table.role = EDGE_ROLE_MESSAGE_PASSING
        elif edge_table.role not in _VALID_EDGE_ROLES:
            raise ValueError(
                f"edge_type={edge_table.edge_type}: role={edge_table.role!r} "
                f"is not valid. Expected one of {sorted(_VALID_EDGE_ROLES)}."
            )

        if edge_table.node_anchor is not None:
            if edge_table.node_anchor not in (
                edge_table.src_node_type,
                edge_table.dst_node_type,
            ):
                raise ValueError(
                    f"edge_type={edge_table.edge_type}: node_anchor="
                    f"{edge_table.node_anchor!r} must equal src_node_type="
                    f"{edge_table.src_node_type!r} or dst_node_type="
                    f"{edge_table.dst_node_type!r}."
                )
        elif edge_table.role == EDGE_ROLE_SUPERVISION_POS:
            raise ValueError(
                f"edge_type={edge_table.edge_type}: node_anchor is required "
                f"when role={EDGE_ROLE_SUPERVISION_POS!r}."
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
    logger.info(f"Loading analyzer config from {config_path}")
    raw = OmegaConf.load(config_path)
    merged = OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), raw)
    config: DataAnalyzerConfig = OmegaConf.to_object(merged)  # type: ignore
    _validate_and_backfill(config)
    logger.info(
        f"Loaded analyzer config with {len(config.node_tables)} node tables "
        f"and {len(config.edge_tables)} edge tables"
    )
    return config
