"""Pydantic result types and JSON artifact IO for the BQ Data Analyzer.

Each analyzer component (:class:`GraphStructureAnalyzer`, :class:`FeatureProfiler`)
returns a versioned Pydantic model. Components persist their results as
JSON sidecars at ``{output_gcs_path}/{component}.json`` using
:func:`write_artifact`; consumers (report generator, downstream quality
gates) rehydrate them via :func:`load_artifact`.

Envelope shape (see :class:`GraphStructureArtifact`,
:class:`FeatureProfileArtifact`)::

    {
      "schema_version": "1",
      "component": "feature_profile",
      "generated_at": "2026-04-23T20:00:00+00:00",
      "data": { ... }
    }

Additive fields bump nothing (JSON readers tolerate them); rename / remove
bumps :data:`SCHEMA_VERSION` and requires consumers to handle the new major.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from gigl.common import GcsUri
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils

logger = Logger()

SCHEMA_VERSION: Final[Literal["1"]] = "1"

_Component = Literal["graph_structure", "feature_profile"]


class TopKEntry(BaseModel):
    """One row of a top-K most-frequent-hash listing."""

    model_config = ConfigDict(extra="forbid")

    hash: int
    count: int
    fraction: float


class EmbeddingDiagnosticsResult(BaseModel):
    """Structural sanity counts for a single REPEATED FLOAT column.

    ``unique_ratio`` close to 1.0 means embeddings are well-differentiated;
    low values indicate upstream degeneracy (many rows sharing the same
    vector). ``top_k`` surfaces the most frequent exact-duplicate clusters
    via ``FARM_FINGERPRINT(TO_JSON_STRING(<col>))``.
    """

    model_config = ConfigDict(extra="forbid")

    total: int
    unique_count: int
    unique_ratio: float
    top_k: list[TopKEntry] = Field(default_factory=list)


class DegreeStats(BaseModel):
    """Degree distribution statistics for one edge type and direction.

    Computed from ``APPROX_QUANTILES(degree, 100)`` in BigQuery.
    """

    model_config = ConfigDict(extra="forbid")

    min: int
    max: int
    mean: float
    median: int
    p90: int
    p99: int
    p999: int
    percentiles: list[int]
    buckets: dict[str, int]


class PerClassDegreeStats(BaseModel):
    """Per-class degree-distribution summary for one labeled node type.

    Surfaces the silent NC-at-scale footgun where labeled positive-class
    nodes are biased toward high-degree, leading to "model just learns
    degree" behavior at inference. Computed once per (node_type, class)
    by joining the labeled node table to the message-passing edge table.

    The companion ``cold_start_count`` counts class members with degree
    <= 1 — these will fail at inductive serving regardless of training
    quality.
    """

    model_config = ConfigDict(extra="forbid")

    class_value: str
    count: int
    cold_start_count: int
    mean_degree: float
    median_degree: int
    p90_degree: int
    p99_degree: int
    max_degree: int
    buckets: dict[str, int] = Field(default_factory=dict)


class HomophilyStats(BaseModel):
    """Homophily measures for one (labeled node type, edge type) pair.

    ``edge_homophily`` is the raw fraction of message-passing edges whose
    endpoints share a label. It's the standard textbook measure but is
    not comparable across datasets with different class priors.

    ``adjusted_homophily`` corrects for class priors per Platonov et al.,
    *Characterizing Graph Datasets for Node Classification*, NeurIPS
    2023. Range is approximately [-1, 1]; values near 0 indicate
    "no signal beyond class priors", positive means homophilic, negative
    heterophilic.

    ``edge_sample_count`` records how many edges were sampled to compute
    the measures so consumers can assess statistical reliability.
    """

    model_config = ConfigDict(extra="forbid")

    edge_type: str
    edge_homophily: float
    adjusted_homophily: float
    edge_sample_count: int
    label_informativeness: Optional[float] = None


class LabelSentinelStats(BaseModel):
    """Sentinel-vs-NULL accounting for one labeled node type.

    Surfaces the *upstream-bug* case where labels are present but encode
    "missing/unknown" via sentinel values like ``-1`` or ``"unknown"``
    rather than SQL NULL. Treating those as real classes silently
    poisons training. Reported as a separate count from NULL so the
    upstream owner can be paged on the right thing.

    ``valid_label_coverage`` is the fraction of rows with a real label
    (non-NULL AND non-sentinel) and is what downstream class-imbalance /
    homophily computations use as the denominator.
    """

    model_config = ConfigDict(extra="forbid")

    total_rows: int
    null_count: int
    sentinel_counts: dict[str, int] = Field(default_factory=dict)
    valid_label_count: int
    valid_label_coverage: float


class CrossSplitOverlap(BaseModel):
    """Cross-split node-id leakage stats for one labeled node type.

    A node-id appearing in more than one split is unconditionally a
    bug — train/val/test contamination silently inflates eval metrics.
    The analyzer treats any non-zero ``overlap_node_count`` as a Tier 1
    style hard fail and raises :class:`DataQualityError`.
    """

    model_config = ConfigDict(extra="forbid")

    overlap_node_count: int
    split_value_counts: dict[str, int] = Field(default_factory=dict)


class NodeClassificationSupervisionStats(BaseModel):
    """Aggregated NC supervision-tier results for one labeled node type.

    Holds the BQ-side metrics that aren't covered by the TFDV slicing
    pass (per-class degree, homophily, split-leakage). Per-class label
    histograms and per-class feature null-rates are produced by the
    feature profiler via TFDV ``slice_functions`` and surface there.

    ``sentinel_degree_stats`` carries the same shape as ``per_class_degree``
    but for rows whose label matches a value declared in
    ``NodeTableSpec.label_sentinel_values`` (e.g. ``-1``). Surfacing the
    sentinel pool's degree distribution exposes whether "no ground-truth
    label" rows are mostly cold-start (cheap to keep as message-passing
    context) or mostly hubs (will dominate aggregation and bias the model).
    """

    model_config = ConfigDict(extra="forbid")

    node_type: str
    label_column: str
    sentinel_stats: LabelSentinelStats
    per_class_degree: list[PerClassDegreeStats] = Field(default_factory=list)
    sentinel_degree_stats: list[PerClassDegreeStats] = Field(default_factory=list)
    homophily: list[HomophilyStats] = Field(default_factory=list)
    cross_split_overlap: Optional[CrossSplitOverlap] = None


class SupervisionCrossTableStats(BaseModel):
    """Per-anchor cross-table statistics for one (driver, other) edge-table pair.

    Computed from a positive (or negative) supervision edge table — the
    *driver* — against another edge table — the *other* — that shares the
    same ``(src_node_type, dst_node_type)``. The driver defines the anchor
    population: distinct anchor IDs that appear in the driver. For each such
    anchor we count how many edges it has in ``other`` and report the
    distribution. ``overlap_pair_count`` flags ``(anchor, neighbor)`` pairs
    that appear in both tables — typically a label-leakage signal.
    """

    model_config = ConfigDict(extra="forbid")

    driver_edge_type: str
    driver_role: str
    other_edge_type: str
    other_role: str
    node_anchor: str

    driver_anchor_count: int
    driver_pair_count: int
    other_pair_count: int
    overlap_pair_count: int
    driver_anchors_with_zero_other: int

    avg_other_per_driver_anchor: float
    p50_other_per_driver_anchor: int
    p90_other_per_driver_anchor: int
    p99_other_per_driver_anchor: int
    max_other_per_driver_anchor: int


class GraphAnalysisResult(BaseModel):
    """Complete result of graph structure analysis across all tiers.

    Tier 1 fields are always populated. Tier 3/4 fields may be empty
    dicts if the corresponding checks were not applicable or not enabled.
    """

    model_config = ConfigDict(extra="forbid")

    # Tier 1: hard fails
    duplicate_node_counts: dict[str, int] = Field(default_factory=dict)
    dangling_edge_counts: dict[str, int] = Field(default_factory=dict)
    referential_integrity_violations: dict[str, int] = Field(default_factory=dict)

    # Tier 2: core metrics
    node_counts: dict[str, int] = Field(default_factory=dict)
    edge_counts: dict[str, int] = Field(default_factory=dict)
    null_rates: dict[str, dict[str, float]] = Field(default_factory=dict)
    duplicate_edge_counts: dict[str, int] = Field(default_factory=dict)
    self_loop_counts: dict[str, int] = Field(default_factory=dict)
    isolated_node_counts: dict[str, int] = Field(default_factory=dict)
    degree_stats: dict[str, DegreeStats] = Field(default_factory=dict)
    top_hubs: dict[str, list[tuple[str, int]]] = Field(default_factory=dict)
    super_hub_int16_clamp_count: dict[str, int] = Field(default_factory=dict)
    cold_start_node_counts: dict[str, int] = Field(default_factory=dict)
    feature_memory_bytes: dict[str, int] = Field(default_factory=dict)
    neighbor_explosion_estimate: dict[str, int] = Field(default_factory=dict)

    # Tier 3: label and heterogeneous
    class_imbalance: dict[str, dict[str, int]] = Field(default_factory=dict)
    label_coverage: dict[str, float] = Field(default_factory=dict)
    edge_type_distribution: dict[str, int] = Field(default_factory=dict)
    edge_type_node_coverage: dict[str, dict[str, int]] = Field(default_factory=dict)

    # Tier 4: opt-in
    reciprocity: dict[str, float] = Field(default_factory=dict)
    power_law_exponent: dict[str, float] = Field(default_factory=dict)

    # Supervision cross-table analysis (per (driver, other) edge-table pair).
    supervision_cross_table_stats: list[SupervisionCrossTableStats] = Field(
        default_factory=list
    )

    # Node-classification supervision tier (per labeled node type).
    node_classification_supervision_stats: list[
        NodeClassificationSupervisionStats
    ] = Field(default_factory=list)

    # Per-block rendered BQ SQL captured at execution time, keyed by a
    # block identifier the report JS uses to locate the corresponding
    # section header. The flat shape (a single ``dict[str, list[str]]``)
    # is intentional — the JS does dict lookups, not parsing.
    queries: dict[str, list[str]] = Field(default_factory=dict)


class FeatureProfileError(BaseModel):
    """One per-table failure or skip captured during feature profiling.

    Surfaces to the HTML report so users can see *why* a table did not
    produce a FACETS embed instead of silently missing from the result.

    ``stage`` is one of:
        * ``"schema_fetch"`` — BigQuery schema lookup raised
        * ``"empty_projection"`` — no profileable columns after projection
        * ``"dataflow"`` — the per-table Dataflow pipeline raised
        * ``"embedding_diagnostics"`` — the post-Dataflow diagnostics query raised

    For ``stage == "dataflow"`` we additionally try to capture the Dataflow
    job identifiers so the report can deep-link to the failed job's logs:

        * ``job_id`` — the Dataflow job UUID, or ``None`` if the runner
          isn't Dataflow / the result didn't expose one.
        * ``job_name`` — the human-readable job name (e.g.
          ``gigl-analyzer-svij-test-20260506-1430-profile-node-user``).
        * ``console_url`` — a link to the Dataflow console for the job, or
          ``None`` when ``job_id`` / region / project couldn't be resolved.
    """

    model_config = ConfigDict(extra="forbid")

    result_key: str
    bq_table: str
    stage: str
    message: str
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    console_url: Optional[str] = None


class FeatureProfileResult(BaseModel):
    """Result of TFDV feature profiling across all tables.

    ``facets_html_paths`` / ``stats_paths`` point at per-table GCS
    artifacts produced by the Dataflow pipelines. When a node table
    declares ``label_column`` or ``split_column`` the profiler enables
    TFDV ``slice_functions`` on those columns; the resulting per-slice
    stats live alongside the unsliced stats in the same TFRecord and the
    per-slice listing is surfaced via ``slice_columns_by_result_key`` so
    consumers know which slices to expect.

    ``embedding_diagnostics`` is keyed by result_key
    (``"node:{type}"`` / ``"edge:{type}"``) and then by embedding column
    name. ``errors`` collects per-table failures or skips so consumers
    (and the HTML report) can show why a particular table did not produce
    facets.
    """

    model_config = ConfigDict(extra="forbid")

    # List-valued so a wide table can be split into multiple per-chunk
    # Dataflow pipelines (one Facets HTML + stats TFRecord per chunk).
    # Single-chunk tables produce a list of length 1.
    facets_html_paths: dict[str, list[str]] = Field(default_factory=dict)
    stats_paths: dict[str, list[str]] = Field(default_factory=dict)
    schema_paths: dict[str, list[str]] = Field(default_factory=dict)
    anomalies: dict[str, list[str]] = Field(default_factory=dict)
    embedding_diagnostics: dict[str, dict[str, EmbeddingDiagnosticsResult]] = Field(
        default_factory=dict
    )
    errors: list[FeatureProfileError] = Field(default_factory=list)

    # Per-result-key list of column names that were used as TFDV slice
    # functions. Consumers (the HTML report, downstream gates) read this
    # to know which slice listings to render from the TFDV stats.
    slice_columns_by_result_key: dict[str, list[str]] = Field(default_factory=dict)


class GraphStructureArtifact(BaseModel):
    """Versioned envelope for a :class:`GraphAnalysisResult`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    component: Literal["graph_structure"] = "graph_structure"
    generated_at: datetime
    data: GraphAnalysisResult


class FeatureProfileArtifact(BaseModel):
    """Versioned envelope for a :class:`FeatureProfileResult`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = SCHEMA_VERSION
    component: Literal["feature_profile"] = "feature_profile"
    generated_at: datetime
    data: FeatureProfileResult


def write_artifact(
    result: Union[GraphAnalysisResult, FeatureProfileResult],
    component: _Component,
    output_gcs_path: str,
) -> str:
    """Serialize ``result`` into a versioned envelope and persist it.

    Writes ``{output_gcs_path}/{component}.json``. If ``output_gcs_path``
    starts with ``gs://`` the payload is uploaded via ``GcsUtils``; otherwise
    the parent directory is created and the file is written locally.

    Args:
        result: The component's in-memory result model.
        component: Which component is writing (``"graph_structure"`` or
            ``"feature_profile"``). Must match the ``result`` type.
        output_gcs_path: Directory URI or local path. Trailing slashes are
            stripped.

    Returns:
        The full path (GCS URI or absolute local path) that was written.

    Raises:
        TypeError: If ``result`` does not match the declared ``component``.
        ValueError: If ``component`` is not one of the known literals.
    """
    now = datetime.now(timezone.utc)
    if component == "graph_structure":
        if not isinstance(result, GraphAnalysisResult):
            raise TypeError(
                f"component='graph_structure' expects GraphAnalysisResult, "
                f"got {type(result).__name__}"
            )
        artifact: BaseModel = GraphStructureArtifact(generated_at=now, data=result)
    elif component == "feature_profile":
        if not isinstance(result, FeatureProfileResult):
            raise TypeError(
                f"component='feature_profile' expects FeatureProfileResult, "
                f"got {type(result).__name__}"
            )
        artifact = FeatureProfileArtifact(generated_at=now, data=result)
    else:
        raise ValueError(
            f"component={component!r} must be 'graph_structure' or 'feature_profile'"
        )

    payload = artifact.model_dump_json(indent=2)
    trimmed = output_gcs_path.rstrip("/")
    path = f"{trimmed}/{component}.json"
    if trimmed.startswith("gs://"):
        GcsUtils().upload_from_string(GcsUri(path), payload)
    else:
        local_path = Path(path).expanduser().resolve()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(payload)
        path = str(local_path)
    logger.info(f"Wrote {component} artifact to {path}")
    return path


def load_artifact(
    path: str, expected_component: _Component
) -> Union[GraphAnalysisResult, FeatureProfileResult]:
    """Load and validate a JSON sidecar, returning its ``.data`` payload.

    Args:
        path: GCS URI (``gs://...``) or local filesystem path to the JSON.
        expected_component: Which component's artifact is expected. A
            mismatch raises ``ValueError`` rather than silently returning
            the wrong type.

    Returns:
        The component's result model (``GraphAnalysisResult`` or
        ``FeatureProfileResult``).

    Raises:
        ValueError: If the loaded envelope's ``component`` does not match
            ``expected_component``, or its ``schema_version`` is unknown.
    """
    if path.startswith("gs://"):
        text = GcsUtils().read_from_gcs(GcsUri(path))
    else:
        text = Path(path).expanduser().resolve().read_text()

    if expected_component == "graph_structure":
        artifact_gs: GraphStructureArtifact = (
            GraphStructureArtifact.model_validate_json(text)
        )
        return artifact_gs.data
    elif expected_component == "feature_profile":
        artifact_fp: FeatureProfileArtifact = (
            FeatureProfileArtifact.model_validate_json(text)
        )
        return artifact_fp.data
    else:
        raise ValueError(
            f"expected_component={expected_component!r} must be "
            f"'graph_structure' or 'feature_profile'"
        )


__all__ = [
    "SCHEMA_VERSION",
    "TopKEntry",
    "EmbeddingDiagnosticsResult",
    "DegreeStats",
    "PerClassDegreeStats",
    "HomophilyStats",
    "LabelSentinelStats",
    "CrossSplitOverlap",
    "NodeClassificationSupervisionStats",
    "SupervisionCrossTableStats",
    "GraphAnalysisResult",
    "FeatureProfileError",
    "FeatureProfileResult",
    "GraphStructureArtifact",
    "FeatureProfileArtifact",
    "write_artifact",
    "load_artifact",
]
