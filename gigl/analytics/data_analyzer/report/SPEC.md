# Report Generator SPEC

## Purpose

This SPEC defines the single self-contained HTML report that the BQ Data Analyzer produces for a graph dataset. The
three `.ai.{html,js,css}` files in this directory implement the SPEC and should be regenerated from it whenever the SPEC
changes. The Python `report_generator.py` module is the only non-AI-owned component in this directory; it loads the AI
assets via `importlib.resources`, injects data from a `GraphAnalysisResult` dataclass, and writes a single HTML file to
disk.

## Constraints

- Single self-contained HTML file. No external CDN, no external JS/CSS/font dependencies, no network requests at view
  time.
- Opens in any modern browser (Chrome, Firefox, Safari, Edge) without a server.
- Max-width 1200px, centered horizontally.
- Light background (`#f8f9fa`).
- Monospace font (`ui-monospace`, `SFMono-Regular`, `Menlo`, `monospace`) for all numeric data values; sans-serif
  (`system-ui`, `-apple-system`, `"Segoe UI"`, `Roboto`, sans-serif) for labels and headings.
- Collapsible sections use `<details>` / `<summary>` (no JS required to expand/collapse).
- Color coding for status uses these exact values:
  - Green: `#28a745` (OK)
  - Yellow: `#ffc107` (warning)
  - Red: `#dc3545` (critical)
- Total report HTML should be reasonable in size (a single dataset's report with embedded FACETS iframes may be
  multi-MB; that is acceptable).

## Sections (in display order)

1. **Header** (`<header id="report-header">`) — "GiGL Data Analysis Report" title, generation timestamp, and a short
   config summary listing the analyzed node tables and edge tables.
2. **Overview Dashboard** (`<section id="overview">`) — Card grid showing total nodes, total edges, number of node
   types, number of edge types, and an overall traffic-light status indicator (green/yellow/red). The status is the
   worst severity across all detected issues.
3. **Data Quality** (`<section id="data-quality">`) — Per-table NULL rates table sorted highest-first with rows
   color-coded (NULL rate > 50% = yellow,
   > 90% = red). Duplicate node counts, duplicate edge counts, dangling edge counts, and referential integrity
   > violations. Any nonzero count in these four is rendered red.
4. **Feature Statistics** (`<section id="feature-statistics">`) — Optional. One `<details>` block per table. Each block
   embeds the corresponding FACETS HTML via `<iframe src="...">` using a **relative path** of the form
   `feature_profiler/{kind}s/{type_name}/facets.html` (derived from the result_key like `node:user`). Above the iframe,
   an "Open full-screen ↗" anchor opens the same relative path in a new tab. Relative paths mean the embed and
   full-screen link both work as long as the report folder retains the layout produced by `FeatureProfiler` (i.e.
   `report.html` and the `feature_profiler/` subdirectory live in the same directory). The absolute GCS URI from
   `facets_html_paths` is shown as a label for traceability. When `profile.errors` is non-empty, a red warning box plus
   a per-error table (table key, stage, BQ table, **Dataflow job**, message) is rendered at the top of the section so
   users can diagnose schema-fetch failures, empty projections, Dataflow crashes, and embedding-diagnostics failures
   without reading logs. For `stage == "dataflow"` errors the Dataflow job cell links to the Cloud Console URL when
   `job_id` / project / region are all known; otherwise the cell shows the job name or `—`. Section is hidden only when
   both `profile.facets_html_paths` and `profile.errors` are empty.
5. **Graph Structure** (`<section id="graph-structure">`) — Node and edge count table. Per-edge-type degree distribution
   rendered as inline SVG histogram using the `buckets` dict from `DegreeStats` (buckets `0-1`, `2-10`, `11-100`,
   `101-1K`, `1K-10K`, `10K+`). Top-20 hub table per edge type. Super-hub int16 clamp warning box (red) shown if any
   edge type reports a clamp count > 0. Each per-edge-type subsection header (`Degree distribution`, `Top-20 hubs`)
   carries a `<details class="query-disclosure">Show SQL` button rendered next to the heading; expanding it shows the
   rendered BigQuery SQL strings recorded under the matching `analysis.queries` block ID.
6. **Supervision Overlap** (`<section id="supervision-overlap">`) — One card per `SupervisionCrossTableStats` entry
   showing `{driver_edge_type} → {other_edge_type} ({other_role})`, anchored on `node_anchor`. Row labels reference the
   actual `driver_edge_type` and `other_edge_type` names directly (e.g. "Distinct anchors in `viewed_pos`", "Avg edges
   in `viewed_neg` per anchor in `viewed_pos`") rather than generic "driver" / "other" placeholders. Each card lists
   distinct anchor / pair counts on each side, per-anchor count distribution (avg / p50 / p90 / p99 / max), the count of
   anchors with zero edges on the other side, and the overlap pair count (label-leakage signal). Each card title is
   accompanied by a `Show SQL` disclosure exposing the underlying cross-table query. Section is hidden when
   `analysis.supervision_cross_table_stats` is empty.
6a. **Node Classification Supervision** (`<section id="node-classification-supervision">`) — One card per labeled node
    type. Subsections are: **Label hygiene** (sentinel / NULL / valid counts), **Per-class degree** (one row per class
    with count, cold-start fraction, mean / median / p90 / p99 / max degree, and an inline SVG sparkline histogram in a
    `Distribution` column rendered from the per-class `buckets` dict), **Homophily** (per-edge-type edge / adjusted
    homophily and sample size), and **Train / val / test split** (cross-split id leakage and per-split row counts). Each
    subsection's `<h4>` has a `Show SQL` disclosure next to it; for Homophily the disclosure aggregates queries across
    all matching edge types.
7. **Advanced** (`<section id="advanced">`) — Optional Tier 3 / Tier 4 data. Shown only if the relevant fields are
   populated. Each subsection's `<h3>` carries a `Show SQL` disclosure when corresponding queries were recorded:
   - Class imbalance (bar chart and per-class counts)
   - Label coverage (percentage per node type)
   - Edge type distribution (bar chart)
   - Reciprocity per edge type
   - Power-law exponent per edge type
8. **Footer** (`<footer id="report-footer">`) — GiGL version / commit and a list of raw artifact GCS paths.

## Key Thresholds

Thresholds used to color-code metrics. These must match the design doc (`docs/plans/20260415-bq-data-analyzer.md`)
exactly.

| Metric                           | Green         | Yellow     | Red        |
| -------------------------------- | ------------- | ---------- | ---------- |
| Edge homophily                   | > 0.7         | 0.3 - 0.7  | < 0.3      |
| Class imbalance ratio            | < 1:5         | 1:5 - 1:10 | > 1:10     |
| Feature missing rate             | < 10%         | 10 - 50%   | > 90%      |
| Isolated node fraction           | < 1%          | 1 - 5%     | > 5%       |
| Degree p99/median                | < 50          | 50 - 100   | > 100      |
| Node degree (int16 clamp)        | < 32,767      | n/a        | > 32,767   |
| Cold-start fraction (degree 0-1) | < 5%          | 5 - 10%    | > 10%      |
| Edge type dominance              | No type > 80% | Any > 90%  | Any < 0.1% |
| Overlap pair fraction            | 0             | (0, 1%)    | ≥ 1%       |
| Driver anchors with zero `other` | < 5%          | 5 - 50%    | > 50%      |

## Data Injection Contract

`report_generator.py` produces a final HTML file by performing four exact string replacements on `report.ai.html`:

| Placeholder                  | Replaced with                                                |
| ---------------------------- | ------------------------------------------------------------ |
| `/* INJECT_STYLES */`        | Raw contents of `styles.ai.css`                              |
| `/* INJECT_SCRIPTS */`       | Raw contents of `charts.ai.js`                               |
| `/* INJECT_ANALYSIS_DATA */` | JSON-serialized `GraphAnalysisResult` (`dataclasses.asdict`) |
| `/* INJECT_PROFILE_DATA */`  | JSON-serialized `FeatureProfileResult` (or `{}` if absent)   |

The JS reads these injected JSON strings from hidden script tags:

```html
<script id="analysis-data" type="application/json">/* INJECT_ANALYSIS_DATA */</script>
<script id="profile-data"  type="application/json">/* INJECT_PROFILE_DATA */</script>
```

On page load the JS:

1. Parses both JSON blobs.
2. Populates each section by generating DOM nodes (never `innerHTML` with untrusted strings; always `textContent`).
3. Renders the degree distribution as an inline SVG bar chart.
4. Applies color coding (`status-green`, `status-yellow`, `status-red`) based on the thresholds above.
5. Hides `#feature-statistics` if the profile data is empty / `{}`.
6. Hides `#advanced` if no Tier 3 or Tier 4 data is present.
7. Renders per-block `Show SQL` disclosures from the `analysis.queries` map (see contract below). The disclosure is
   omitted when no queries were recorded for the matching block ID.

### `analysis.queries` Contract

`GraphAnalysisResult.queries` is a flat `dict[str, list[str]]` populated at execution time by the analyzer. Keys are
block IDs that the report renderer uses to locate which `<details class="query-disclosure">` to attach near a header.
Multiple SQL strings under one key are rendered as separate `<pre class="sql">` blocks. Block ID conventions:

| Section              | Pattern                                          |
| -------------------- | ------------------------------------------------ |
| Data quality         | `data_quality:<metric>:<scope>`                  |
| Graph structure      | `graph_structure:<metric>:<edge_or_node_type>`   |
| NC supervision       | `nc_supervision:<metric>:<node_type>[:<edge>]`   |
| Supervision overlap  | `supervision_overlap:<driver>:<other>:<roles>`   |
| Advanced             | `advanced:<metric>:<scope>`                      |

Renderer behavior:

- **Per-block headers** (Degree distribution per edge type, Top hubs per edge type, NC supervision sub-blocks,
  Supervision overlap card title, Advanced sub-blocks) render one `Show SQL` disclosure per block ID using
  `renderBlockHeader` / `renderQueryDisclosure`.
- **Aggregated section headers** (NULL rates, Integrity checks, Counts, Homophily within an NC supervision card)
  render one disclosure that pulls every block ID matching a prefix using `renderQueryDisclosureByPrefix`.
- Missing keys (or an empty `analysis.queries`) cause the disclosure to be skipped silently — old artifacts that
  predate this field still render correctly without buttons.

### JS Element Contract

The JS queries these DOM IDs. The HTML template must provide them:

- `#report-header`
- `#overview`
- `#data-quality`
- `#feature-statistics`
- `#graph-structure`
- `#supervision-overlap`
- `#advanced`
- `#report-footer`
- `#analysis-data` (hidden JSON script tag)
- `#profile-data` (hidden JSON script tag)

## Regeneration Instructions

To regenerate `report.ai.html`, `charts.ai.js`, and `styles.ai.css`:

1. Read this `SPEC.md` in full.
2. Implement the sections, element IDs, thresholds, and data injection contract exactly as specified.
3. Keep the HTML template minimal (all content is rendered by JS).
4. Keep the JS as a single IIFE with no external dependencies; use DOM helpers, not templating libraries.
5. Use the exact color hex values specified in "Constraints".
6. Update the snapshot test golden file at `tests/test_assets/analytics/golden_report.html` after regenerating.
