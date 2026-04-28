from dataclasses import dataclass, field


@dataclass
class DegreeStats:
    """Degree distribution statistics for one edge type and direction.

    Computed from APPROX_QUANTILES(degree, 100) in BigQuery.
    """

    min: int
    max: int
    mean: float
    median: int
    p90: int
    p99: int
    p999: int
    percentiles: list[int]
    buckets: dict[str, int]  # "0-1": count, "2-10": count, etc.


@dataclass
class GraphAnalysisResult:
    """Complete result of graph structure analysis across all tiers.

    Tier 1 fields are always populated. Tier 3/4 fields may be empty
    dicts if the corresponding checks were not applicable or not enabled.
    """

    # Tier 1: hard fails
    duplicate_node_counts: dict[str, int] = field(default_factory=dict)
    dangling_edge_counts: dict[str, int] = field(default_factory=dict)
    referential_integrity_violations: dict[str, int] = field(default_factory=dict)

    # Tier 2: core metrics
    node_counts: dict[str, int] = field(default_factory=dict)
    edge_counts: dict[str, int] = field(default_factory=dict)
    null_rates: dict[str, dict[str, float]] = field(default_factory=dict)
    duplicate_edge_counts: dict[str, int] = field(default_factory=dict)
    self_loop_counts: dict[str, int] = field(default_factory=dict)
    isolated_node_counts: dict[str, int] = field(default_factory=dict)
    degree_stats: dict[str, DegreeStats] = field(default_factory=dict)
    top_hubs: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    super_hub_int16_clamp_count: dict[str, int] = field(default_factory=dict)
    cold_start_node_counts: dict[str, int] = field(default_factory=dict)
    feature_memory_bytes: dict[str, int] = field(default_factory=dict)
    neighbor_explosion_estimate: dict[str, int] = field(default_factory=dict)

    # Tier 3: label and heterogeneous
    class_imbalance: dict[str, dict[str, int]] = field(default_factory=dict)
    label_coverage: dict[str, float] = field(default_factory=dict)
    edge_type_distribution: dict[str, int] = field(default_factory=dict)
    edge_type_node_coverage: dict[str, dict[str, int]] = field(default_factory=dict)

    # Tier 4: opt-in
    reciprocity: dict[str, float] = field(default_factory=dict)
    power_law_exponent: dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureProfileResult:
    """Result of TFDV feature profiling across all tables.

    Contains GCS paths to generated artifacts.
    """

    facets_html_paths: dict[str, str] = field(default_factory=dict)
    stats_paths: dict[str, str] = field(default_factory=dict)
    schema_paths: dict[str, str] = field(default_factory=dict)
    anomalies: dict[str, list[str]] = field(default_factory=dict)
