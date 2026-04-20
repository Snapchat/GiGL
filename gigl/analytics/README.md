# GiGL Analytics

Pre-training graph data validation and analysis tooling. Use this module before committing to a GNN training run to
catch data quality and structural issues that silently degrade model quality.

Two subpackages:

- [`data_analyzer/`](data_analyzer/) — end-to-end `DataAnalyzer` that runs 4 tiers of BigQuery checks and produces a
  single self-contained HTML report. **Start here.**
- [`graph_validation/`](graph_validation/) — lightweight standalone validators (currently: `BQGraphValidator` for
  dangling-edge checks). Use when you only need one check and not the full report.

## Quickstart

Three steps to a working report.

### 1. Authenticate to BigQuery

```bash
gcloud auth application-default login
```

### 2. Write a YAML config

Save as `my_analyzer_config.yaml`:

```yaml
node_tables:
  - bq_table: "your-project.your_dataset.user_nodes"
    node_type: "user"
    id_column: "user_id"
    feature_columns: ["age", "country"]
    # label_column: "label"  # optional, enables Tier 3 checks

edge_tables:
  - bq_table: "your-project.your_dataset.user_edges"
    edge_type: "follows"
    src_id_column: "src_user_id"
    dst_id_column: "dst_user_id"
    # timestamp_column: "ts"  # optional, enables temporal freshness

# Where to write the HTML report. Use a local path for quick iteration
# or a gs:// URI to upload to GCS.
output_gcs_path: "/tmp/my_analysis/"

# Optional: sizing for neighbor-explosion estimates (fan-out per GNN layer).
fan_out: [15, 10, 5]
```

### 3. Run the analyzer

```bash
uv run python -m gigl.analytics.data_analyzer \
    --analyzer_config_uri my_analyzer_config.yaml
```

When it finishes you will see:

```
[INFO] Report written to /tmp/my_analysis/report.html
```

Open the file in any browser. No server, no external dependencies, fully offline.

## What it checks

The analyzer organizes checks into four tiers. Tiers 1 and 2 always run; tier 3 auto-enables when your config supports
it; tier 4 is opt-in.

| Tier                         | When                                                                                 | What it checks                                                                                                                                                                                                                          |
| ---------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Hard fails**            | Always                                                                               | Dangling edges, referential integrity (edges to non-existent nodes), duplicate nodes. Raises `DataQualityError` — report still generated to show partial results.                                                                       |
| **2. Core metrics**          | Always                                                                               | Node/edge counts, degree distribution (in/out), degree buckets, top-K hubs, super-hub int16 clamp count, cold-start node count, self-loops, duplicate edges, NULL rates per column, feature memory budget, neighbor-explosion estimate. |
| **3. Label + heterogeneous** | Auto when `label_column` is set on any node table, or when multiple edge types exist | Class imbalance, label coverage, edge type distribution, per-edge-type node coverage.                                                                                                                                                   |
| **4. Advanced**              | Opt-in via config flags                                                              | Reciprocity, homophily, connected components, clustering coefficient. Runs on full data (no sampling).                                                                                                                                  |

Full per-check rationale with literature citations lives in the
[design doc](../../docs/plans/20260415-bq-data-analyzer.md) and
[literature review](../../docs/plans/20260415-bq-data-analyzer-references.md).

## Interpreting the report

The report color-codes every numeric finding against thresholds drawn from 18 production GNN papers. Summary:

| Metric                                                   | Green | Yellow     | Red     | What to do when yellow/red                                                                                                                    |
| -------------------------------------------------------- | ----- | ---------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| Dangling edges / referential integrity / duplicate nodes | 0     | —          | any > 0 | Fix the input tables. Training will fail or silently corrupt otherwise.                                                                       |
| Feature missing rate                                     | < 10% | 10–50%     | > 90%   | Plan an imputation strategy; at > 95% the Feature Propagation phase transition hits and GNNs stop recovering signal well.                     |
| Isolated node fraction                                   | < 1%  | 1–5%       | > 5%    | Filter isolated nodes or densify (LiGNN-style) for cold-start cohorts.                                                                        |
| Cold-start fraction (degree 0–1)                         | < 5%  | 5–10%      | > 10%   | Candidates for graph densification; also flag for special handling at serving time.                                                           |
| Super-hub int16 clamp (degree > 32,767)                  | 0     | —          | any > 0 | GiGL silently truncates super-hub degrees. Either cap the hub's edges upstream or plan to fix the clamp.                                      |
| Degree p99/median                                        | < 50  | 50–100     | > 100   | Use importance sampling (PinSage) or degree-adaptive neighborhoods (BLADE).                                                                   |
| Class imbalance ratio                                    | < 1:5 | 1:5 – 1:10 | > 1:10  | Message passing amplifies label imbalance 2-3x in representation space (GraphSMOTE). Consider resampling or GraphSMOTE-style synthetic nodes. |
| Edge homophily (Tier 4)                                  | > 0.7 | 0.3 – 0.7  | < 0.3   | Standard GCN/GAT fail at low h. Consider H2GCN-style architectures; at h < 0.2 an MLP often wins.                                             |

Full threshold table with citations:
[`docs/plans/20260415-bq-data-analyzer-references.md`](../../docs/plans/20260415-bq-data-analyzer-references.md#19-consolidated-threshold-table).

## Advanced config

Optional YAML keys beyond the minimal quickstart:

```yaml
# Enable Tier 3 class-imbalance + label-coverage checks for a node type:
node_tables:
  - bq_table: ...
    label_column: "label"

# Enable Tier 4 temporal freshness for an edge type:
edge_tables:
  - bq_table: ...
    timestamp_column: "created_at"

# Neighbor explosion estimation — the fan-out per GNN layer you plan to train with:
fan_out: [15, 10, 5]

# Opt-in Tier 4 checks. Default false; all run on full data (no sampling).
compute_reciprocity: true
compute_homophily: true
compute_connected_components: true
compute_clustering: true
```

## Python API

The CLI wraps a regular class. Call from your own code when you want programmatic access to the result dataclass:

```python
from gigl.analytics.data_analyzer import DataAnalyzer
from gigl.analytics.data_analyzer.config import load_analyzer_config

config = load_analyzer_config("my_analyzer_config.yaml")
analyzer = DataAnalyzer()
report_path = analyzer.run(config=config)
# report_path points to the written report.html
```

The underlying `GraphStructureAnalyzer` is also callable directly if you only want the raw `GraphAnalysisResult`
dataclass and no HTML:

```python
from gigl.analytics.data_analyzer.graph_structure_analyzer import GraphStructureAnalyzer

result = GraphStructureAnalyzer().analyze(config)
print(result.degree_stats)
```

## graph_validation

One-off validators for the subset of cases where the full analyzer is overkill. Today the only check is dangling-edge
detection:

```python
from gigl.analytics.graph_validation import BQGraphValidator

has_dangling = BQGraphValidator.does_edge_table_have_dangling_edges(
    edge_table="your-project.your_dataset.user_edges",
    src_node_column_name="src_user_id",
    dst_node_column_name="dst_user_id",
)
```

The `DataAnalyzer` runs this check (and many more) as part of Tier 1, so prefer the full analyzer unless you
specifically need to script a one-line gate.

## Scope and limitations

The v1 analyzer has two deliberate scope cuts:

- **FeatureProfiler is a stub.** The class is wired in but the TFDV/Dataflow pipeline that produces FACETS HTML per
  table is deferred to a follow-up PR. Calling it today logs a warning and returns an empty `FeatureProfileResult`. The
  main report is fully functional without it.
- **Tier 4 queries are not implemented.** Reciprocity, homophily, connected components, and clustering coefficient
  config flags are accepted but currently no-op. The power-law exponent (Tier 4) is computed from degree stats as an
  approximation.

## Deeper reading

- [Design doc](../../docs/plans/20260415-bq-data-analyzer.md) — architecture, 4-tier validation, cost control, tradeoff
  analysis
- [Literature review](../../docs/plans/20260415-bq-data-analyzer-references.md) — 18 production GNN papers, 100+
  findings, consolidated threshold table
- [1-pager](../../docs/plans/20260416-data-analyzer-1-pager.md) — executive summary
- [Engineering spec](../../docs/plans/20260416-data-analyzer-engineering-spec.md) — per-layer implementation contract
- [Report PRD](data_analyzer/report/PRD.md) — product intent for the HTML report
- [Report SPEC](data_analyzer/report/SPEC.md) — technical contract for regenerating the HTML/JS/CSS assets
