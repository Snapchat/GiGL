# BQ Data Analyzer: Engineering Spec

## Context

This spec defines the implementation plan for the BQ Data Analyzer described in the
[design doc](20260415-bq-data-analyzer.md) and [1-pager](20260416-data-analyzer-1-pager.md). The analyzer takes BQ table
references via YAML config and produces a single HTML report covering data quality, feature distributions, and graph
structure metrics for GNN training readiness.

Validation dimensions are informed by a
[literature review of 18 production GNN papers](20260415-bq-data-analyzer-references.md).

## Architecture

```
YAML Config --> DataAnalyzer (orchestrator)
                   |
                   +-- GraphStructureAnalyzer (BQ SQL, 25 checks across 4 tiers)
                   |       uses: BqUtils, BQGraphValidator
                   |       output: GraphAnalysisResult
                   |
                   +-- FeatureProfiler (Beam/Dataflow)
                   |       uses: GenerateAndVisualizeStats, IngestRawFeatures
                   |       output: FeatureProfileResult
                   |
                   +-- ReportGenerator (data_analyzer/report/)
                           uses: GraphAnalysisResult + FeatureProfileResult
                           output: single self-contained HTML -> GCS
```

## Implementation: Layer-by-Layer

### Layer 1: Config and Types

**Files:**

- `gigl/analytics/data_analyzer/config.py`
- `gigl/analytics/data_analyzer/types.py`

**config.py** defines three dataclasses parsed from YAML via OmegaConf (matching `gigl/common/utils/yaml_loader.py`
pattern):

```python
@dataclass
class NodeTableSpec:
    bq_table: str
    node_type: str
    id_column: str
    feature_columns: list[str]
    label_column: Optional[str] = None

@dataclass
class EdgeTableSpec:
    bq_table: str
    edge_type: str
    src_id_column: str
    dst_id_column: str
    feature_columns: list[str] = field(default_factory=list)
    timestamp_column: Optional[str] = None

@dataclass
class DataAnalyzerConfig:
    node_tables: list[NodeTableSpec]
    edge_tables: list[EdgeTableSpec]
    output_gcs_path: str
    fan_out: Optional[list[int]] = None
    compute_reciprocity: bool = False
    compute_homophily: bool = False
    compute_connected_components: bool = False
    compute_clustering: bool = False
```

Config loading uses `OmegaConf.load()` + `OmegaConf.merge(OmegaConf.structured(DataAnalyzerConfig), loaded)` +
`OmegaConf.to_object()`.

**types.py** defines result dataclasses:

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
    percentiles: list[int]
    buckets: dict[str, int]  # "0-1": count, "2-10": count, etc.

@dataclass
class GraphAnalysisResult:
    # Tier 1: hard fails
    duplicate_node_counts: dict[str, int]
    dangling_edge_counts: dict[str, int]
    referential_integrity_violations: dict[str, int]
    # Tier 2: core metrics
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    null_rates: dict[str, dict[str, float]]
    duplicate_edge_counts: dict[str, int]
    self_loop_counts: dict[str, int]
    isolated_node_counts: dict[str, int]
    degree_stats: dict[str, DegreeStats]
    top_hubs: dict[str, list[tuple[str, int]]]
    super_hub_int16_clamp_count: dict[str, int]
    cold_start_node_counts: dict[str, int]
    feature_memory_bytes: dict[str, int]
    neighbor_explosion_estimate: dict[str, int]
    # Tier 3: label and heterogeneous
    class_imbalance: dict[str, dict[str, int]]
    label_coverage: dict[str, float]
    edge_type_distribution: dict[str, int]
    edge_type_node_coverage: dict[str, dict[str, int]]
    # Tier 4: opt-in
    reciprocity: dict[str, float]
    power_law_exponent: dict[str, float]

@dataclass
class FeatureProfileResult:
    facets_html_paths: dict[str, str]   # table_name -> GCS path to FACETS HTML
    stats_paths: dict[str, str]         # table_name -> GCS path to stats TFRecord
    schema_paths: dict[str, str]        # table_name -> GCS path to schema proto
    anomalies: dict[str, list[str]]     # table_name -> list of anomaly descriptions
```

**Tests:** `tests/unit/analytics/data_analyzer/config_test.py`

- Parse valid YAML, verify all fields populated
- Parse YAML with optional fields omitted, verify defaults
- Parse invalid YAML (missing required fields), verify error raised
- Validate edge table references against node table types

______________________________________________________________________

### Layer 2: BQ Queries and GraphStructureAnalyzer

**Files:**

- `gigl/analytics/data_analyzer/queries.py`
- `gigl/analytics/data_analyzer/graph_structure_analyzer.py`

**queries.py**: 18 SQL template constants as module-level strings. Each parameterized with `.format()` for table names
and column names. Pattern matches `gigl/src/data_preprocessor/lib/enumerate/queries.py`.

Query inventory (from design doc):

| Constant                           | Tier | Purpose                                   |
| ---------------------------------- | ---- | ----------------------------------------- |
| `NODE_COUNT_QUERY`                 | 2    | `SELECT COUNT(*) FROM {table}`            |
| `EDGE_COUNT_QUERY`                 | 2    | `SELECT COUNT(*) FROM {table}`            |
| `NULL_RATES_QUERY`                 | 2    | Batched `COUNTIF` per column              |
| `DUPLICATE_NODE_COUNT_QUERY`       | 1    | `GROUP BY id HAVING COUNT(*) > 1`         |
| `DUPLICATE_EDGE_COUNT_QUERY`       | 2    | `GROUP BY (src, dst) HAVING COUNT(*) > 1` |
| `DANGLING_EDGES_QUERY`             | 1    | `WHERE src IS NULL OR dst IS NULL`        |
| `EDGE_REFERENTIAL_INTEGRITY_QUERY` | 1    | `LEFT JOIN ... WHERE IS NULL`             |
| `SELF_LOOP_COUNT_QUERY`            | 2    | `WHERE src = dst`                         |
| `ISOLATED_NODE_COUNT_QUERY`        | 2    | `LEFT JOIN ... WHERE edge IS NULL`        |
| `DEGREE_DISTRIBUTION_QUERY`        | 2    | `APPROX_QUANTILES` on degree counts       |
| `DEGREE_BUCKET_QUERY`              | 2    | Bucket counts for 6 ranges                |
| `TOP_K_HUBS_QUERY`                 | 2    | `ORDER BY degree DESC LIMIT k`            |
| `SUPER_HUB_INT16_CLAMP_QUERY`      | 2    | `HAVING COUNT(*) > 32767`                 |
| `COLD_START_NODE_COUNT_QUERY`      | 2    | Degree 0-1 count                          |
| `CLASS_IMBALANCE_QUERY`            | 3    | `GROUP BY label_column`                   |
| `LABEL_COVERAGE_QUERY`             | 3    | `COUNTIF(label IS NOT NULL)`              |
| `EDGE_TYPE_DISTRIBUTION_QUERY`     | 3    | `COUNT(*)` per type                       |
| `EDGE_TYPE_NODE_COVERAGE_QUERY`    | 3    | `COUNT(DISTINCT src/dst)` per type        |

`FEATURE_MEMORY_BUDGET`, `NEIGHBOR_EXPLOSION_ESTIMATE`, and `POWER_LAW_EXPONENT` are computed in Python from schema
metadata and degree stats.

**graph_structure_analyzer.py**:

- `GraphStructureAnalyzer.__init__(bq_project: Optional[str])` creates `BqUtils`
- `analyze(config: DataAnalyzerConfig) -> GraphAnalysisResult`
- Runs queries in parallel via `concurrent.futures.ThreadPoolExecutor` (same pattern as `data_preprocessor.py:311`)
- Tier 1 checks raise `DataQualityError` on failure
- Tier 3 checks auto-enabled when `label_column` or multiple edge types present
- Tier 4 checks gated by config flags

**Tests:** `tests/unit/analytics/data_analyzer/queries_test.py`

- For each query constant: verify `.format()` with test table/column names produces valid SQL containing expected
  clauses
- Verify `NULL_RATES_QUERY` batches multiple columns into one query
- Verify `DEGREE_DISTRIBUTION_QUERY` produces separate in/out queries

**Tests:** `tests/unit/analytics/data_analyzer/graph_structure_analyzer_test.py`

- Mock `BqUtils.run_query()` with `@patch`
- Feed canned query results (lists of dicts mimicking BQ `RowIterator`)
- Verify `GraphAnalysisResult` fields populated correctly
- Verify Tier 1 checks raise `DataQualityError` when violations found
- Verify Tier 3 checks skipped when no `label_column` configured
- Verify Tier 4 checks skipped when config flags are False
- Verify `feature_memory_bytes` computed correctly from schema metadata
- Verify `neighbor_explosion_estimate` computed from degree stats and fan_out config

______________________________________________________________________

### Layer 3: FeatureProfiler (TFDV/Dataflow)

**Files:**

- `gigl/analytics/data_analyzer/feature_profiler.py`

Standalone Beam pipeline builder. For each table in config:

1. Builds feature spec from column names and inferred types
2. Creates Beam pipeline via `init_beam_pipeline_options()`
3. Reads from BQ via `BigqueryNodeDataReference` / `BigqueryEdgeDataReference`
4. Runs `GenerateAndVisualizeStats` for TFDV stats + FACETS HTML
5. Runs `tfdv.infer_schema()` for schema inference
6. Runs `tfdv.validate_statistics()` for anomaly detection
7. Writes outputs to GCS under `{output_gcs_path}/tfdv/{table_name}/`

Returns `FeatureProfileResult` with GCS paths.

Launches one Dataflow job per table via `ThreadPoolExecutor` with a lock (same pattern as DataPreprocessor for
serialized `p.run()` calls).

**Tests:** No unit tests for the Beam pipeline itself (Dataflow execution). Tested via integration test with real BQ and
Dataflow.

**Tests:** `tests/integration/analytics/data_analyzer/feature_profiler_test.py`

- Requires cloud resources (BQ table, Dataflow, GCS bucket)
- Runs profiler against a small test BQ table
- Verifies output files exist on GCS (stats TFRecord, FACETS HTML, schema proto)

______________________________________________________________________

### Layer 4: HTML Report

**Files:**

- `gigl/analytics/data_analyzer/report/SPEC.md`
- `gigl/analytics/data_analyzer/report/report_generator.py`
- `gigl/analytics/data_analyzer/report/report.ai.html`
- `gigl/analytics/data_analyzer/report/charts.ai.js`
- `gigl/analytics/data_analyzer/report/styles.ai.css`
- `gigl/analytics/data_analyzer/report/__init__.py`

**SPEC.md**: AI-owned specification for the HTML report. Defines:

- Report sections (header, overview dashboard, data quality, feature statistics, graph structure, footer)
- Visual style (#f8f9fa background, monospace data values, sans-serif labels, 1200px max-width)
- Color coding (green #28a745, yellow #ffc107, red #dc3545) with threshold values from literature
- Collapsible sections (CSS-only)
- Self-contained constraint (no external dependencies)
- FACETS embedding via iframe/shadow DOM

AI agents read SPEC.md to generate/regenerate the `.ai.{html|js|css}` files.

**report_generator.py**: Python module:

- `generate(analysis_result: GraphAnalysisResult, profile_result: Optional[FeatureProfileResult], config: DataAnalyzerConfig) -> str`
- Loads `report.ai.html` template
- Serializes results to JSON and injects into template
- Embeds FACETS HTML strings inline (reads from GCS paths in `FeatureProfileResult`)
- Returns the complete HTML string

**report.ai.html**: Main template with placeholder slots for data injection. Contains the report structure.

**charts.ai.js**: Inline SVG chart generation for degree distribution histograms. No external charting library.

**styles.ai.css**: Embedded CSS for the report layout and color coding.

**Tests:** `tests/unit/analytics/data_analyzer/report/report_generator_test.py`

- Snapshot test:
  1. Construct `GraphAnalysisResult` with deterministic test data
  2. Call `generate()` (without TFDV results, those are optional)
  3. Compare output against golden file at `tests/test_assets/analytics/golden_report.html`
  4. If snapshot differs, test fails with a diff showing what changed
- Structural tests:
  - Verify output contains expected section headings
  - Verify Tier 1 hard-fail results appear with red indicators
  - Verify threshold color coding applied correctly (green/yellow/red)
  - Verify optional sections (Tier 3, Tier 4) omitted when data not present

______________________________________________________________________

### Layer 5: Orchestrator and CLI

**Files:**

- `gigl/analytics/data_analyzer/data_analyzer.py`
- `gigl/analytics/data_analyzer/__init__.py`

**data_analyzer.py**:

```python
class DataAnalyzer:
    def run(
        self,
        config: DataAnalyzerConfig,
        resource_config_uri: Optional[Uri] = None,
    ) -> str:  # returns GCS path to HTML report
        ...
```

Orchestration:

1. Run `GraphStructureAnalyzer.analyze(config)` and `FeatureProfiler.profile(config)` in parallel via
   `ThreadPoolExecutor`
2. If GraphStructureAnalyzer raises `DataQualityError` (Tier 1 failure), still generate report showing the failures, but
   log error
3. Pass both results to `ReportGenerator.generate()`
4. Upload HTML string to `{config.output_gcs_path}/report.html` via `GcsUtils`
5. Log the GCS path

CLI entry point (`if __name__ == "__main__"`):

```python
parser = argparse.ArgumentParser()
parser.add_argument("--analyzer_config_uri", required=True)
parser.add_argument("--resource_config_uri", required=False)
```

Matches DataPreprocessor CLI pattern.

**Tests:** No dedicated unit test for the orchestrator (it's thin glue). Tested via integration test.

______________________________________________________________________

## Testing Summary

| Test file                          | Layer | Type        | What it tests                                            |
| ---------------------------------- | ----- | ----------- | -------------------------------------------------------- |
| `config_test.py`                   | 1     | Unit        | YAML parsing, defaults, validation                       |
| `queries_test.py`                  | 2     | Unit        | SQL template correctness (string assertions)             |
| `graph_structure_analyzer_test.py` | 2     | Unit        | Mocked BQ, result population, tier gating, error raising |
| `report_generator_test.py`         | 4     | Unit        | Snapshot test, structural assertions, threshold coloring |
| `feature_profiler_test.py`         | 3     | Integration | Real BQ + Dataflow, output file existence                |
| `graph_structure_analyzer_test.py` | 2     | Integration | Real BQ, query execution, result correctness             |

All unit tests use `tests.test_assets.test_case.TestCase` as base class. BQ mocking uses
`@patch("gigl.src.common.utils.bq.bigquery.Client")` pattern from existing `bq_test.py`.

## AI-Owned File Convention

Files named `*.ai.{html|js|css}` are generated and maintained by AI agents. The `SPEC.md` in the same directory defines
the requirements. To regenerate:

1. Read `SPEC.md`
2. Generate/update the `.ai.*` files to match the spec
3. Run snapshot test to verify no regressions

This convention is new to the codebase. The `report/` directory is the first instance.

## Reused Code

| Component                    | Path                                                    |
| ---------------------------- | ------------------------------------------------------- |
| `GenerateAndVisualizeStats`  | `gigl/src/data_preprocessor/lib/transform/utils.py:120` |
| `IngestRawFeatures`          | `gigl/src/data_preprocessor/lib/transform/utils.py:85`  |
| `InstanceDictToTFExample`    | `gigl/src/data_preprocessor/lib/transform/utils.py:42`  |
| `WriteTFSchema`              | `gigl/src/data_preprocessor/lib/transform/utils.py:186` |
| `init_beam_pipeline_options` | `gigl/src/common/utils/dataflow.py`                     |
| `BQGraphValidator`           | `gigl/analytics/graph_validation/bq_graph_validator.py` |
| `BqUtils`                    | `gigl/src/common/utils/bq.py`                           |
| `gcs_constants`              | `gigl/src/common/constants/gcs.py`                      |
| `BigqueryNodeDataReference`  | `gigl/src/data_preprocessor/lib/ingest/bigquery.py`     |
| `yaml_loader`                | `gigl/common/utils/yaml_loader.py`                      |

## Verification

- `make unit_test_py PY_TEST_FILES="config_test.py"` for config parsing
- `make unit_test_py PY_TEST_FILES="queries_test.py"` for SQL templates
- `make unit_test_py PY_TEST_FILES="graph_structure_analyzer_test.py"` for mocked BQ
- `make unit_test_py PY_TEST_FILES="report_generator_test.py"` for snapshot test
- `make integration_test PY_TEST_FILES="feature_profiler_test.py"` for TFDV/Dataflow
- Manual: run analyzer against real BQ tables, open HTML report in browser
