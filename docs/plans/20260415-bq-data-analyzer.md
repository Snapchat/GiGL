# BQ Data Analyzer

## Problem

Before training a GNN on graph data in BigQuery, we need to understand data quality, feature distributions, and graph
structure. Today, TFDV statistics generation is embedded inside the DataPreprocessor pipeline and commented out
(`gigl/src/data_preprocessor/lib/transform/utils.py:300`). There is no standalone way to analyze BQ tables without
running the full preprocessing flow.

## Solution

A standalone DataAnalyzer module under `gigl/analytics/` that takes BQ table references as input and produces a single
HTML report combining TFDV feature statistics, FACETS visualizations, data quality checks, and graph structure metrics.

## Components

### 1. DataAnalyzerConfig (dataclass)

- List of node table specs: BQ table URI, feature columns, ID column, node type
- List of edge table specs: BQ table URI, src/dst ID columns, feature columns, edge type
- Output GCS path for results
- Optional: resource config URI for Dataflow sizing

### 2. FeatureProfiler (reuses existing TFDV Beam components)

For each table: builds a Beam pipeline that reads from BQ, converts to TFExamples, runs `tfdv.GenerateStatistics()`,
writes stats TFRecord and FACETS HTML.

Reuses: `IngestRawFeatures` (`utils.py:85`), `GenerateAndVisualizeStats` (`utils.py:120`), `WriteTFSchema`
(`utils.py:186`), `init_beam_pipeline_options` (`dataflow.py`). Runs on Dataflow, same infra as DataPreprocessor.

Also runs `tfdv.infer_schema()` for schema inference and `tfdv.validate_statistics()` for anomaly detection.

### 3. GraphStructureAnalyzer (BQ SQL, extends BQGraphValidator)

Two files: `queries.py` (SQL templates as string constants, same pattern as
`gigl/src/data_preprocessor/lib/enumerate/queries.py`) and `graph_structure_analyzer.py` (orchestrator that calls
`BqUtils.run_query()`).

Validation dimensions are informed by a literature review of 12 production GNN papers. See
`docs/plans/20260415-bq-data-analyzer-references.md` for full details and justifications.

#### Cost Control

| Check category                                                                                    | Can TABLESAMPLE? | Reason                                                       |
| ------------------------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------ |
| Row-independent (NULL rates, feature distributions, cardinality)                                  | Yes              | Rows are independent; sampling preserves distribution        |
| Structure-dependent (degree, referential integrity, hubs, isolated nodes, self-loops, duplicates) | No               | Sampling edges destroys connectivity, distorts degree counts |
| Expensive structure-dependent (reciprocity, homophily, connected components)                      | No               | Must run on full data; opt-in via config flags instead       |

#### Tier 1: Hard Fails (always-on, block training)

Violations here mean training will fail or produce silently corrupt results.

| Check                      | What                           | Justification                                                                                                                                                         |
| -------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dangling edges             | NULL src or dst                | Breaks graph construction. Reuses `BQGraphValidator`. (GiGL)                                                                                                          |
| Edge referential integrity | src or dst not in node table   | Breaks graph loading during `DistPartitioner`. `LEFT JOIN ... WHERE node.id IS NULL`. (GiGL enumeration, AliGraph: ID space mismatches caused silent NaN propagation) |
| Duplicate nodes            | Same ID appears multiple times | Ambiguous feature loading, corrupt aggregation. (All papers, implied)                                                                                                 |

#### Tier 2: Core Graph Understanding (always-on, essential metrics)

Must-run checks that quantify fundamental graph properties. Not blocking, but critical for sampling strategy,
architecture selection, and resource allocation.

| Check                                   | What                                                           | Justification                                                                                                                                                                                                                      |
| --------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Node/edge counts                        | `COUNT(*)` per table                                           | Basic sanity. Off-by-orders-of-magnitude = pipeline failure.                                                                                                                                                                       |
| Degree distribution (in/out separately) | min, max, mean, median, p90, p99, p99.9 via `APPROX_QUANTILES` | Single most important structural property. PinSage: 46% improvement from importance sampling on power-law graphs. BLADE: 230% embedding improvement from degree-adaptive neighborhoods. LiGNN: +3.2% AUC from 20 to 200 neighbors. |
| Degree-stratified bucket counts         | Nodes in buckets: 0-1, 2-10, 11-100, 101-1K, 1K-10K, 10K+      | Different degree ranges need different sampling budgets. BLADE's core insight: neighborhood size should follow a power-law of in-degree.                                                                                           |
| Super-hub int16 clamp count             | Nodes with degree > 32,767                                     | GiGL clamps to `torch.iinfo(torch.int16).max` in `distributed/utils/degree.py:134`. Silently truncates; affects PPR sampling probabilities.                                                                                        |
| Top-K hub nodes                         | Highest-degree nodes per edge type                             | Hub nodes dominate aggregation (PinSage). AliGraph: top 1% vertices accessed in >80% of mini-batches.                                                                                                                              |
| Isolated node count                     | Zero in-degree AND zero out-degree                             | Cannot receive message-passing signal. AliGraph: 8% of nodes were in components < 10 nodes.                                                                                                                                        |
| Cold-start node count                   | Degree 0-1 per type                                            | Candidates for graph densification. LiGNN: +0.28% AUC from adding artificial edges for cold-start members.                                                                                                                         |
| Self-loop count                         | Edges where src == dst                                         | Double-counted if pipeline adds self-loops (A+I normalization). AliGraph: self-loops helped 4/5 benchmarks but hurt fraud detection.                                                                                               |
| Duplicate edge count                    | Same (src, dst) pair per edge type                             | Inflates degree, distorts aggregation. Most GNN frameworks assume simple graphs.                                                                                                                                                   |
| NULL rates per column                   | Batched `COUNTIF(col IS NULL) / COUNT(*)`                      | Feature Propagation: GNNs tolerate up to 90% missing with ~5% accuracy drop. Phase transition at 95%.                                                                                                                              |
| Feature memory budget                   | nodes x feature_dim x dtype_size per type                      | Must fit in distributed memory. 1B nodes x 256-dim x fp16 = 512GB. (AliGraph, LiGNN, TwHIN)                                                                                                                                        |
| Neighbor explosion estimate             | Estimated subgraph size for given fan-out                      | OOM risk. At fan-out [15,10,5] with avg_degree=100, ~75K nodes per seed. (Layer-Neighbor Sampling)                                                                                                                                 |

#### Tier 3: Label and Heterogeneous Quality (always-on if applicable)

Run automatically when `label_column` is configured or multiple edge types exist.

| Check                       | What                                             | Justification                                                                                                                                                              |
| --------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Class imbalance             | Per-class counts, max/min ratio                  | Message passing amplifies imbalance 2-3x. Critical threshold at 1:5. At 1:10, minority F1 drops 30-40%. At 1:20, some classes become undetectable. (GraphSMOTE)            |
| Label coverage              | Fraction with non-NULL labels per type           | Supervision signal strength. Alert if < 5%. (GraphSMOTE)                                                                                                                   |
| Edge type distribution      | Per-type count and fraction                      | Imbalance causes sampling bias for minority types. TwHIN: high-coverage vs low-coverage relations are fundamentally different. Alert if any type < 0.1% or dominant > 90%. |
| Per-edge-type node coverage | Fraction of nodes participating in each relation | Identifies which node types have sparse coverage in which relations. (TwHIN)                                                                                               |

#### Tier 4: Advanced / Opt-in (expensive, run on full data)

Config flags to enable. No TABLESAMPLE; these run on full data or not at all.

| Check                   | Config flag                          | Justification                                                                                                                                |
| ----------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Edge reciprocity        | `compute_reciprocity: true`          | Self-join, expensive at 1B+. Directed vs undirected semantics. BLADE: in-degree != out-degree matters for dual embeddings.                   |
| Homophily ratio         | `compute_homophily: true`            | Standard GNNs degrade 30-50% on heterophilic graphs (h < 0.3). MLP outperforms GCN at h < 0.2. Needs labels + full edges. (Beyond Homophily) |
| Connected components    | `compute_connected_components: true` | Disconnected components can't share info. AliGraph: 8% in small components. Requires iterative SQL or BQ Graph GQL.                          |
| Power-law exponent      | Computed from degree stats (cheap)   | Alpha < 2 = extreme hub concentration. Predicts benefit of importance sampling. (PinSage, BLADE)                                             |
| Temporal edge freshness | `timestamp_column` in edge config    | Stale edges inject noise. LiGNN: 5.8% AUC from temporal awareness. AliGraph: 15% daily churn, 2-3% AUC degradation.                          |
| Clustering coefficient  | `compute_clustering: true`           | Over-smoothing risk with deeper models. Feature Propagation: higher clustering aids feature recovery.                                        |

#### Key Thresholds

| Metric                           | Green         | Yellow         | Red             | Source                  |
| -------------------------------- | ------------- | -------------- | --------------- | ----------------------- |
| Edge homophily                   | > 0.7         | 0.3 - 0.7      | < 0.3           | Beyond Homophily        |
| Class imbalance ratio            | < 1:5         | 1:5 - 1:10     | > 1:10          | GraphSMOTE              |
| Feature missing rate             | < 10%         | 10 - 50%       | > 90%           | Feature Propagation     |
| Missing feature phase transition | < 90%         | 90 - 95%       | > 95%           | Feature Propagation     |
| Isolated node fraction           | < 1%          | 1 - 5%         | > 5%            | AliGraph                |
| Degree p99/median                | < 50          | 50 - 100       | > 100           | PinSage                 |
| Node degree (int16 clamp)        | < 32,767      | n/a            | > 32,767        | GiGL                    |
| Neighbor explosion (per seed)    | < 50K         | 50K - 100K     | > 100K          | Layer-Neighbor Sampling |
| Cold-start fraction (degree 0-1) | < 5%          | 5 - 10%        | > 10%           | LiGNN                   |
| Edge type dominance              | No type > 80% | Any type > 90% | Any type < 0.1% | TwHIN                   |

#### Query Inventory (queries.py)

| Constant                           | Tier | BQ Query                                                         |
| ---------------------------------- | ---- | ---------------------------------------------------------------- |
| `NODE_COUNT_QUERY`                 | 2    | `SELECT COUNT(*) FROM {table}`                                   |
| `EDGE_COUNT_QUERY`                 | 2    | `SELECT COUNT(*) FROM {table}`                                   |
| `NULL_RATES_QUERY`                 | 2    | Batched `COUNTIF(col IS NULL) / COUNT(*)` per column             |
| `DUPLICATE_NODE_COUNT_QUERY`       | 1    | `GROUP BY id HAVING COUNT(*) > 1`                                |
| `DUPLICATE_EDGE_COUNT_QUERY`       | 2    | `GROUP BY (src, dst) HAVING COUNT(*) > 1`                        |
| `DANGLING_EDGES_QUERY`             | 1    | `WHERE src IS NULL OR dst IS NULL`                               |
| `EDGE_REFERENTIAL_INTEGRITY_QUERY` | 1    | `LEFT JOIN node_table WHERE node.id IS NULL`                     |
| `SELF_LOOP_COUNT_QUERY`            | 2    | `WHERE src = dst`                                                |
| `ISOLATED_NODE_COUNT_QUERY`        | 2    | `LEFT JOIN edge_table ... WHERE edge IS NULL`                    |
| `DEGREE_DISTRIBUTION_QUERY`        | 2    | `APPROX_QUANTILES` on `GROUP BY` degree counts (in/out separate) |
| `DEGREE_BUCKET_QUERY`              | 2    | Bucket counts: 0-1, 2-10, 11-100, 101-1K, 1K-10K, 10K+           |
| `TOP_K_HUBS_QUERY`                 | 2    | `ORDER BY degree DESC LIMIT k`                                   |
| `SUPER_HUB_INT16_CLAMP_QUERY`      | 2    | `HAVING COUNT(*) > 32767`                                        |
| `COLD_START_NODE_COUNT_QUERY`      | 2    | `LEFT JOIN ... HAVING COUNT(edge) <= 1`                          |
| `CLASS_IMBALANCE_QUERY`            | 3    | `GROUP BY label_column` with counts                              |
| `LABEL_COVERAGE_QUERY`             | 3    | `COUNTIF(label IS NOT NULL) / COUNT(*)`                          |
| `EDGE_TYPE_DISTRIBUTION_QUERY`     | 3    | `COUNT(*)` per edge type table                                   |
| `EDGE_TYPE_NODE_COVERAGE_QUERY`    | 3    | `COUNT(DISTINCT src)` and `COUNT(DISTINCT dst)` per edge type    |

`FEATURE_MEMORY_BUDGET`, `NEIGHBOR_EXPLOSION_ESTIMATE`, and `POWER_LAW_EXPONENT` are computed in Python from schema
metadata and degree stats, not BQ queries.

#### Result Dataclass

```python
@dataclass
class DegreeStats:
    min: int
    max: int
    mean: float
    median: int
    p90: int
    p99: int
    p999: int
    percentiles: list[int]  # 100 values from APPROX_QUANTILES
    buckets: dict[str, int]  # "0-1": count, "2-10": count, etc.

@dataclass
class GraphAnalysisResult:
    # Tier 1: hard fails
    duplicate_node_counts: dict[str, int]
    dangling_edge_counts: dict[str, int]
    referential_integrity_violations: dict[str, int]

    # Tier 2: core graph understanding
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    null_rates: dict[str, dict[str, float]]
    duplicate_edge_counts: dict[str, int]
    self_loop_counts: dict[str, int]
    isolated_node_counts: dict[str, int]
    degree_stats: dict[str, DegreeStats]       # edge_type -> stats (in + out separate)
    top_hubs: dict[str, list[tuple[str, int]]]
    super_hub_int16_clamp_count: dict[str, int]
    cold_start_node_counts: dict[str, int]
    feature_memory_bytes: dict[str, int]
    neighbor_explosion_estimate: dict[str, int]

    # Tier 3: label and heterogeneous (populated if applicable)
    class_imbalance: dict[str, dict[str, int]]
    label_coverage: dict[str, float]
    edge_type_distribution: dict[str, int]
    edge_type_node_coverage: dict[str, dict[str, int]]  # edge_type -> {src_coverage, dst_coverage}

    # Tier 4: opt-in (populated if enabled)
    reciprocity: dict[str, float]
    power_law_exponent: dict[str, float]
```

### 4. ReportGenerator (AI-owned, see `gigl/analytics/data_analyzer/PRD.md`)

Takes all raw outputs (TFDV stats, BQ query results, schemas) and generates a single self-contained HTML report. This
component is defined by the PRD and generated/maintained by AI.

## Data Flow

```
BQ Tables (node + edge)
    |
    +-- FeatureProfiler (Dataflow)
    |   +-- TFDV stats (per table)
    |   +-- Inferred schemas
    |   +-- Anomaly reports
    |
    +-- GraphStructureAnalyzer (BQ SQL)
    |   +-- Counts
    |   +-- Degree distributions
    |   +-- Quality checks
    |   +-- Hub analysis
    |
    +-- ReportGenerator
        +-- Single HTML report (GCS)
```

## Entry Point

```python
# CLI
python -m gigl.analytics.data_analyzer \
    --analyzer_config_uri gs://bucket/analyzer_config.yaml \
    --resource_config_uri gs://bucket/resource_config.yaml

# Python
from gigl.analytics.data_analyzer import DataAnalyzer
analyzer = DataAnalyzer()
analyzer.run(config=config, resource_config_uri=resource_config_uri)
```

## Config Format (YAML)

```yaml
node_tables:
  - bq_table: "project.dataset.user_nodes"
    node_type: "user"
    id_column: "user_id"
    feature_columns: ["age", "country", "embedding"]
    label_column: "label"  # optional, enables class imbalance and label coverage checks

edge_tables:
  - bq_table: "project.dataset.user_user_edges"
    edge_type: "engages"
    src_id_column: "src_user_id"
    dst_id_column: "dst_user_id"
    feature_columns: ["weight", "recency"]

output_gcs_path: "gs://bucket/analysis_output/"

# Neighbor explosion estimation (optional)
fan_out: [15, 10, 5]  # sampling fan-out per GNN layer, used to estimate subgraph size

# Opt-in expensive checks (Tier 4)
# compute_reciprocity: false
# compute_homophily: false
# compute_connected_components: false
# compute_clustering: false
```

## Reused Code

| Component                  | Source                       | Path                                                    |
| -------------------------- | ---------------------------- | ------------------------------------------------------- |
| TFDV stats generation      | `GenerateAndVisualizeStats`  | `gigl/src/data_preprocessor/lib/transform/utils.py:120` |
| Raw feature ingestion      | `IngestRawFeatures`          | `gigl/src/data_preprocessor/lib/transform/utils.py:85`  |
| Instance dict to TFExample | `InstanceDictToTFExample`    | `gigl/src/data_preprocessor/lib/transform/utils.py:42`  |
| TF schema writing          | `WriteTFSchema`              | `gigl/src/data_preprocessor/lib/transform/utils.py:186` |
| Beam pipeline options      | `init_beam_pipeline_options` | `gigl/src/common/utils/dataflow.py`                     |
| Dangling edge check        | `BQGraphValidator`           | `gigl/analytics/graph_validation/bq_graph_validator.py` |
| BQ utilities               | `BqUtils`                    | `gigl/src/common/utils/bq.py`                           |
| GCS path generation        | `gcs_constants`              | `gigl/src/common/constants/gcs.py`                      |
| BQ data references         | `BigqueryNodeDataReference`  | `gigl/src/data_preprocessor/lib/ingest/bigquery.py`     |

## New Files

```
gigl/analytics/
    __init__.py                      # existing
    graph_validation/                # existing
    data_analyzer/
        __init__.py
        PRD.md                       # AI-owned: requirements for HTML report
        data_analyzer.py             # main orchestrator + CLI entry point
        config.py                    # DataAnalyzerConfig dataclass
        feature_profiler.py          # TFDV Beam pipeline builder
        graph_structure_analyzer.py  # BQ SQL queries for graph metrics
        report_generator.py          # AI-owned: HTML report generation (built from PRD.md)
        queries.py                   # SQL query templates
```

## Tradeoff Analysis

### TFDV for feature profiling vs alternatives

**Chose:** TFDV (TensorFlow Data Validation) on Dataflow.

**Rejected:**

- **Dataplex Auto Data Quality**: Google's managed BQ-native successor to TFDV. Serverless, petabyte-scale, zero infra.
  We rejected it because TFDV code already exists in the codebase (commented out at `utils.py:300`), TFDV provides
  FACETS HTML visualizations that Dataplex does not, and TFDV gives schema inference + anomaly detection in one
  pipeline. Risk: TFDV is in slow-maintenance mode (~1 release/year, Python capped at 3.11).
- **Great Expectations**: Industry standard, BQ SQL pushdown, 300+ checks, very active. Rejected because it's a new
  dependency and doesn't provide FACETS-style visualizations. Better suited for ongoing validation contracts than
  one-time profiling.
- **whylogs**: Profiles scale with features not rows (ideal for 1B nodes), has Dataflow integration. Rejected because
  it's a new dependency and the codebase already has TFDV infrastructure.
- **ydata-profiling**: Not suitable for billion-scale. Requires pulling data into Python/pandas.

### Standalone module vs new pipeline stage

**Chose:** Standalone module in `gigl/analytics/data_analyzer/`.

**Rejected:** Making it a new GiGL pipeline stage (like DataPreprocessor).

Analysis should be runnable independently before committing to a full pipeline run. A pipeline stage requires a
GbmlConfig proto, resource config, and the full orchestration framework. The standalone module takes a simple YAML
config with BQ table references and can be run ad-hoc. This keeps the feedback loop short.

### BQ SQL for graph structure analysis vs graph libraries

**Chose:** BQ-native SQL queries using `APPROX_QUANTILES`, `APPROX_TOP_COUNT`, etc.

**Rejected:**

- **cuGraph (NVIDIA RAPIDS)**: Best performance for graph analytics with GPUs. Requires data export from BQ to GPU
  memory and GPU infrastructure we may not have.
- **NetworKit**: Parallel C++ graph analytics, handles billions of edges on CPU. Requires data export from BQ to local
  storage.
- **GraphFrames on Spark/Dataproc**: Distributed graph algorithms with BQ connector. Adds Spark cluster management
  overhead for metrics we can compute in plain SQL.
- **BQ Graph (GQL)**: Preview status, requires Enterprise edition, limited to pattern matching (no PageRank, community
  detection yet).

Data already lives in BQ. No data movement needed. BQ approximate functions handle billion-scale efficiently, and the
existing codebase uses `BqUtils` extensively. For basic structural analysis (degree distributions, counts, hubs), BQ SQL
is sufficient. If we need advanced graph algorithms (community detection, PageRank) later, we can add a GraphFrames or
cuGraph integration as a separate component.

### Single HTML report vs multiple artifacts

**Chose:** Single self-contained HTML file.

**Rejected:** Separate FACETS HTML per table + JSON data files + notebook/dashboard.

A single HTML is portable: share via GCS link, open in any browser, no serving infrastructure needed. Gives a complete
picture in one view. FACETS per-table HTML is embedded inline. For up to ~20 tables this keeps file size manageable.

### AI-owned report generator vs hand-coded templates

**Chose:** AI-owned component driven by a PRD at `gigl/analytics/data_analyzer/PRD.md`.

**Rejected:** Hand-coded Jinja2 or string template approach.

The HTML report layout will evolve as we add analysis dimensions. A PRD that AI agents can read and regenerate the code
from means the visualization stays in sync with requirements without manual template maintenance. The PRD serves as both
documentation and executable spec.

## Verification

- Unit tests: mock BQ responses, verify query generation, verify config parsing
- Integration test: run against a small test BQ table, verify HTML output is generated
- Manual: inspect the HTML report in a browser

## References

Full literature review with multiple insights per paper and insight-to-analysis mappings:
[`docs/plans/20260415-bq-data-analyzer-references.md`](20260415-bq-data-analyzer-references.md)

12 papers reviewed: PinSage (Pinterest), PinnerSage (Pinterest), BLADE (Amazon), LiGNN (LinkedIn), TwHIN (Twitter/X),
GiGL (Snap), AliGraph (Alibaba), GraphSMOTE, Beyond Homophily, Uber/Grab fraud detection, Google Maps ETA, Feature
Propagation. 100+ total insights extracted.

### Aegis Integration (Phase 2)

We intentionally **do not depend on Aegis in Phase 1** of the BQ Data Analyzer. Instead, we keep the analyzer as a
self-contained GiGL module that can run against arbitrary node/edge BQ tables given only a YAML config.

#### What Aegis provides

Aegis is Snap’s centralized data quality and anomaly detection service. For BQ / BigLake datasets that are onboarded,
Aegis can automatically compute:

- **Profile measures per feature** (e.g., `null_ratio`, `zero_ratio`, `nan_ratio`, `min`, `max`, `mean`, `median`,
  `p90`, `p99`) into per-dataset, per-day tables such as\
  `sc-dig.aegis_{tier}.aegis_{data_source_id}_numerical_YYYYMMDD` and `..._categorical_YYYYMMDD`. :llmCitationRef[1]
- **Array/struct profile tables**, **current row-count tables**, and a shared **user coverage stats** table keyed by
  `data_source_id`. :llmCitationRef[2]
- Optional anomaly detection and alerting on top of those metrics (WoW, ARIMA, UDB, CIC). :llmCitationRef[3]
  :llmCitationRef[4]

Relevant references:

- **Aegis Adhoc Data Quality Check User Guide** (how to onboard adhoc BQ datasets, config naming, CLI backfill, and
  output table schemas). :llmCitationRef[5]
- **Aegis – Data Quality Monitoring & Anomaly Detection** (Confluence overview, supported data sources, profile measure
  semantics). :llmCitationRef[6]
- **Aegis Config v2** (YAML schema for BQ/BigLake data sources and detectors, examples for training datasets).
  :llmCitationRef[7]
- **Aegis New Backend Design** (modular architecture: profile measures, anomaly detection, metadata store, alerting).
  :llmCitationRef[8]

#### Why we are not using Aegis in Phase 1

Phase 1 goals are:

- **Fast, low-friction EDA for arbitrary graph BQ tables** (often adhoc node/edge tables that change frequently).
- **Graph-specific structure analysis** (degree distributions, hubs, isolated nodes, referential integrity, homophily
  proxies, neighbor explosion, etc.), which Aegis does not natively compute today.

Coupling Phase 1 directly to Aegis would introduce several frictions:

- **Onboarding overhead per dataset**: Even in adhoc mode, Aegis requires adding a config YAML in the Flowrida repo
  (`metrics_gov/aegis/configs/adhoc/`), following strict dataset naming conventions (partitioned table prefixes like
  `*_YYYYMMDD`/`*_YYYYMMDDHH`), and running the CLI backfill for each new dataset we want to analyze. :llmCitationRef[9]
  This is exactly the per-table “add code to Flowrida just to look at my data” tax that this tool is trying to avoid.
- **Infra dependencies (BigLake / Nexus / JAM / permissions)**: Adhoc Aegis flows today often involve BigLake syncs or
  Nexus tables, service account wiring (JAMs like `aegis_uum_test`), and additional IAM leases to read the Aegis metrics
  datasets. :llmCitationRef[10] :llmCitationRef[11] That is appropriate for long-lived production datasets but heavy for
  one-off graph experiments.
- **Mismatch in focus**: Aegis is optimized for **continuous monitoring + alerting** (row counts, profile measures,
  usage coverage, detector alerts) across many prod datasets; this module is optimized for **deep, graph-specific
  inspection** (e.g., super-hub clamping risks, neighbor explosion, degree-bucket distributions, homophily signals)
  prior to committing to a GNN pipeline. We would still have to build most of the graph-structure analysis ourselves,
  even if we delegated generic null/row-count stats to Aegis.

Given the above, we treat Aegis as **highly complementary** but not a hard runtime dependency for Phase 1. Phase 1 keeps
all necessary profiling and structure checks self-contained (TFDV + BQ SQL) so that graph practitioners can run it with
only a YAML config and BQ access.

#### Phase 2: Aegis integration plan and PoC ideas

Once the BQ Data Analyzer is stable and useful on its own, we plan a Phase 2 integration with Aegis along two axes:

1. **Read existing Aegis metrics when available (PoC 1)**

   - For BQ datasets that are *already* onboarded to Aegis (e.g., UUM MDS outputs), the analyzer can optionally
     **consume Aegis profile tables** instead of recomputing basic stats:
     - `aegis_{data_source_id}_numerical_YYYYMMDD` / `..._categorical_YYYYMMDD` / `..._array_YYYYMMDD` /
       `..._current_row_count_YYYYMMDD` / `user_coverage_stats`. :llmCitationRef[12]
   - The HTML report would treat these as another input source and clearly label “metrics imported from Aegis” vs
     “metrics computed by analyzer.”
   - **PoC**: pick one existing graph-ish dataset that already has Aegis monitoring (e.g., a UUM or training dataset),
     run the analyzer twice (with and without Aegis import), and sanity-check metric parity (row counts, null ratios,
     coverage).

2. **Optionally publish graph metrics into Aegis (PoC 2)**

   - For production graph datasets, we can add a small writer that emits **graph-structure summaries** (degree buckets,
     isolated/cold-start node fractions, super-hub counts) into an Aegis-compatible metrics table or view, so they show
     up alongside standard profile measures in Aegis dashboards/alerts. :llmCitationRef[13] :llmCitationRef[14]
   - This keeps Aegis as the single pane of glass for ongoing monitoring, while the analyzer remains the “rich HTML
     report” for deep dives.
   - **PoC**: define a minimal Aegis-style schema for graph metrics, write a one-off backfill for a single dataset, and
     validate that the metrics can be queried and (eventually) surfaced via Aegis UI.

3. **Longer-term: better UX for adhoc graph datasets**

   - Longer-term, if Aegis exposes a higher-level UI/CLI for “adhoc BQ analysis” (no Flowrida PRs, just point at a
     table), the BQ Data Analyzer could be wired in as an implementation detail: Aegis would trigger the analyzer and
     host links to the generated HTML report.
   - We explicitly **defer this to future work**, pending Aegis UX/infra evolution.

In summary, **Phase 1** keeps the BQ Data Analyzer self-contained and immediately usable for graph workloads; **Phase
2** focuses on **reusing Aegis where it’s already strong (metrics storage, monitoring, anomaly detection)** and
publishing graph-specific metrics back into that ecosystem, without blocking v1 on current Aegis onboarding and UX
constraints.
