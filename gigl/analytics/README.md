# GiGL Analytics

Pre-training graph data validation and analysis tooling. Use this module before committing to a GNN training run to
catch data quality and structural issues that silently degrade model quality.

Two subpackages:

- [`data_analyzer/`](data_analyzer/) — end-to-end `DataAnalyzer` that runs BigQuery checks and produces a single
  self-contained HTML report. **Start here.**
- [`graph_validation/`](graph_validation/) — lightweight standalone validators (currently: `BQGraphValidator` for
  dangling-edge checks). Use when you only need one check and not the full report.

## Quickstart

**Prerequisites.** Follow the [GiGL installation guide](../../docs/user_guide/getting_started/installation.md) so that
`uv` and GiGL's Python dependencies are available. Then authenticate to BigQuery:

```bash
gcloud auth application-default login
```

**1. Write a YAML config.** Save as `my_analyzer_config.yaml`:

```yaml
node_tables:
  - bq_table: "your-project.your_dataset.user_nodes"
    node_type: "user"
    id_column: "user_id"
    feature_columns: ["age", "country"]  # optional; [] or omit if the node has no features
    # label_column: "label"              # optional; enables Tier 3 label checks

edge_tables:
  - bq_table: "your-project.your_dataset.user_edges"
    edge_type: "follows"
    src_id_column: "src_user_id"
    dst_id_column: "dst_user_id"

# Where to write the HTML report. Local path for quick iteration, or a gs:// URI.
output_gcs_path: "/tmp/my_analysis/"

# Optional: sizing for the neighbor-explosion estimate (fan-out per GNN layer).
fan_out: [15, 10, 5]
```

**2. Run the analyzer.**

```bash
uv run python -m gigl.analytics.data_analyzer \
    --analyzer_config_uri my_analyzer_config.yaml
```

**3. Open the report.** When the run completes:

```
[INFO] Report written to /tmp/my_analysis/report.html
```

Open the file in any browser. No server, no external dependencies, fully offline.

## What it checks

The analyzer organizes checks into four tiers. Tiers 1 and 2 always run; Tier 3 auto-enables when your config supports
it; Tier 4 is opt-in.

| Tier                         | When                                                                                 | What it checks                                                                                                                                                                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Hard fails**            | Always                                                                               | Dangling edges (NULL src/dst), referential integrity (edges pointing to nodes not in the node table), duplicate nodes. Raises `DataQualityError` — the report still renders to show partial results.                                                                                   |
| **2. Core metrics**          | Always                                                                               | Node/edge counts, degree distribution (in/out) with percentiles, degree buckets, top-K hubs, super-hub int16 clamp count, cold-start node count, self-loops, duplicate edges, NULL rates per column, feature memory budget estimate, neighbor-explosion estimate (requires `fan_out`). |
| **3. Label + heterogeneous** | Auto when `label_column` is set on any node table, or when multiple edge types exist | Class imbalance, label coverage, edge type distribution, per-edge-type node coverage.                                                                                                                                                                                                  |
| **4. Advanced**              | Opt-in via config flags                                                              | Power-law exponent (implemented as a degree-stats approximation). Reciprocity, homophily, connected components, clustering coefficient are **not yet implemented** — the flags are accepted but currently no-op.                                                                       |

The thresholds below come from a review of production GNN papers (PinSage, BLADE, LiGNN, TwHIN, AliGraph, GraphSMOTE,
Beyond Homophily, Feature Propagation, and others). See the inline citations in the threshold table for what each paper
contributes.

## Interpreting the report

The report color-codes every numeric finding. Summary of the most important thresholds:

| Metric                                                   | Green | Yellow     | Red     | What to do when yellow/red                                                                                                                                    |
| -------------------------------------------------------- | ----- | ---------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dangling edges / referential integrity / duplicate nodes | 0     | —          | any > 0 | Fix the input tables. Training will fail or silently corrupt otherwise.                                                                                       |
| Feature missing rate                                     | < 10% | 10–50%     | > 90%   | Plan an imputation strategy; above ~95% the Feature Propagation phase transition (Rossi et al., ICLR 2022) hits and GNNs stop recovering signal reliably.     |
| Isolated node fraction                                   | < 1%  | 1–5%       | > 5%    | Filter isolated nodes or densify (LiGNN, KDD 2024) for cold-start cohorts.                                                                                    |
| Cold-start fraction (degree 0–1)                         | < 5%  | 5–10%      | > 10%   | Candidates for graph densification; also flag for special handling at serving time.                                                                           |
| Super-hub int16 clamp (degree > 32,767)                  | 0     | —          | any > 0 | GiGL silently truncates super-hub degrees in `gigl/distributed/utils/degree.py`. Either cap the hub's edges upstream or plan to address the clamp.            |
| Degree p99 / median                                      | < 50  | 50–100     | > 100   | Use importance sampling (PinSage, KDD 2018) or degree-adaptive neighborhoods (BLADE, WSDM 2023) — degree skew is the single biggest lever in production GNNs. |
| Class imbalance ratio                                    | < 1:5 | 1:5 – 1:10 | > 1:10  | Message passing amplifies label imbalance 2–3× in representation space (GraphSMOTE, WSDM 2021). Consider resampling or GraphSMOTE-style synthetic nodes.      |
| Edge homophily (Tier 4, future)                          | > 0.7 | 0.3 – 0.7  | < 0.3   | Standard GCN/GAT fail at low h (Zhu et al., NeurIPS 2020). Consider H2GCN-style architectures; below h ≈ 0.2 a plain MLP often wins.                          |

## Advanced config

Optional YAML keys beyond the minimal quickstart:

```yaml
# Enable Tier 3 class-imbalance + label-coverage checks for a node type:
node_tables:
  - bq_table: ...
    label_column: "label"

# Neighbor explosion estimation — the fan-out per GNN layer you plan to train with:
fan_out: [15, 10, 5]

# Tier 4 opt-in flags. Default false.
# NOTE: Only `compute_reciprocity` is wired into the analyzer today and it logs a
# warning rather than computing a result. The other three flags are placeholders
# for future work (see "Scope and limitations" below).
compute_reciprocity: true
compute_homophily: true
compute_connected_components: true
compute_clustering: true

# Per-edge-type timestamp hint. NOTE: accepted by the config schema but not yet
# consumed by any Tier 4 query (temporal freshness check is planned).
edge_tables:
  - bq_table: ...
    timestamp_column: "created_at"
```

## Python API

The CLI wraps a regular class. Call from your own code when you want programmatic access to the `GraphAnalysisResult`:

```python
from gigl.analytics.data_analyzer import DataAnalyzer
from gigl.analytics.data_analyzer.config import load_analyzer_config

config = load_analyzer_config("my_analyzer_config.yaml")
analyzer = DataAnalyzer()
report_path = analyzer.run(config=config)
# report_path points to the written report.html (local path or gs:// URI)
```

The underlying `GraphStructureAnalyzer` is also callable directly if you want the raw result dataclass and no HTML:

```python
from gigl.analytics.data_analyzer.graph_structure_analyzer import GraphStructureAnalyzer

result = GraphStructureAnalyzer().analyze(config)
print(result.degree_stats)
```

See a rendered report example at
[`tests/test_assets/analytics/golden_report.html`](../../tests/test_assets/analytics/golden_report.html) to preview the
output format before authenticating to BQ.

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
specifically need a one-line gate (e.g., inside an Airflow task or a preprocessing job). This subpackage is the intended
home for additional standalone validators in the future.

## Scope and limitations

Current implementation status:

- **FeatureProfiler is a stub.** The class is wired in but the TFDV/Dataflow pipeline that would produce FACETS HTML per
  table is deferred to a follow-up PR. Calling it today logs a warning and returns an empty `FeatureProfileResult`. The
  main report is fully functional without it.
- **Tier 4 checks are partial.** Power-law exponent is computed as a degree-stats approximation. Reciprocity, homophily,
  connected components, and clustering coefficient config flags are accepted but currently no-op. The `timestamp_column`
  edge field is accepted but no temporal-freshness query runs yet.
- **Heterogeneous graphs: referential integrity caveat.** For each edge table, the referential-integrity check joins
  against `config.node_tables[0]`. On heterogeneous graphs where different edges reference different node types, the
  current implementation will under-report integrity violations — fix is tracked for a follow-up.
- **GCS upload** works via `GcsUtils.upload_from_string` when `output_gcs_path` is a `gs://` URI, and falls back to
  local filesystem write otherwise.

## Related documents

Within this module:

- [`data_analyzer/report/PRD.md`](data_analyzer/report/PRD.md) — product intent for the HTML report (AI-owned)
- [`data_analyzer/report/SPEC.md`](data_analyzer/report/SPEC.md) — technical contract for the AI-owned HTML/JS/CSS
  assets
