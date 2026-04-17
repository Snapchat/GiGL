# Report Generator SPEC

## Purpose

This SPEC defines the single self-contained HTML report that the BQ Data Analyzer
produces for a graph dataset. The three `.ai.{html,js,css}` files in this
directory implement the SPEC and should be regenerated from it whenever the SPEC
changes. The Python `report_generator.py` module is the only non-AI-owned
component in this directory; it loads the AI assets via `importlib.resources`,
injects data from a `GraphAnalysisResult` dataclass, and writes a single HTML
file to disk.

## Constraints

- Single self-contained HTML file. No external CDN, no external JS/CSS/font
  dependencies, no network requests at view time.
- Opens in any modern browser (Chrome, Firefox, Safari, Edge) without a server.
- Max-width 1200px, centered horizontally.
- Light background (`#f8f9fa`).
- Monospace font (`ui-monospace`, `SFMono-Regular`, `Menlo`, `monospace`) for all
  numeric data values; sans-serif (`system-ui`, `-apple-system`,
  `"Segoe UI"`, `Roboto`, sans-serif) for labels and headings.
- Collapsible sections use `<details>` / `<summary>` (no JS required to
  expand/collapse).
- Color coding for status uses these exact values:
  - Green: `#28a745` (OK)
  - Yellow: `#ffc107` (warning)
  - Red: `#dc3545` (critical)
- Total report HTML should be reasonable in size (a single dataset's report
  with embedded FACETS iframes may be multi-MB; that is acceptable).

## Sections (in display order)

1. **Header** (`<header id="report-header">`) — "GiGL Data Analysis Report"
   title, generation timestamp, and a short config summary listing the analyzed
   node tables and edge tables.
2. **Overview Dashboard** (`<section id="overview">`) — Card grid showing total
   nodes, total edges, number of node types, number of edge types, and an
   overall traffic-light status indicator (green/yellow/red). The status is the
   worst severity across all detected issues.
3. **Data Quality** (`<section id="data-quality">`) — Per-table NULL rates table
   sorted highest-first with rows color-coded (NULL rate > 50% = yellow,
   > 90% = red). Duplicate node counts, duplicate edge counts, dangling edge
   counts, and referential integrity violations. Any nonzero count in these
   four is rendered red.
4. **Feature Statistics** (`<section id="feature-statistics">`) — Optional. One
   subsection per table with the corresponding FACETS HTML embedded inside an
   `<iframe srcdoc="...">` to isolate styles. Entire section is hidden if no
   profile data is provided.
5. **Graph Structure** (`<section id="graph-structure">`) — Node and edge count
   table. Per-edge-type degree distribution rendered as inline SVG histogram
   using the `buckets` dict from `DegreeStats` (buckets `0-1`, `2-10`, `11-100`,
   `101-1K`, `1K-10K`, `10K+`). Top-20 hub table per edge type. Super-hub int16
   clamp warning box (red) shown if any edge type reports a clamp count > 0.
6. **Advanced** (`<section id="advanced">`) — Optional Tier 3 / Tier 4 data.
   Shown only if the relevant fields are populated:
   - Class imbalance (bar chart and per-class counts)
   - Label coverage (percentage per node type)
   - Edge type distribution (bar chart)
   - Reciprocity per edge type
   - Power-law exponent per edge type
7. **Footer** (`<footer id="report-footer">`) — GiGL version / commit, list of
   raw artifact GCS paths, and a condensed list of literature references (the
   18 papers from `docs/plans/20260415-bq-data-analyzer-references.md`).

## Key Thresholds

Thresholds used to color-code metrics. These must match the design doc
(`docs/plans/20260415-bq-data-analyzer.md`) exactly.

| Metric                             | Green          | Yellow       | Red          |
|------------------------------------|----------------|--------------|--------------|
| Edge homophily                     | > 0.7          | 0.3 - 0.7    | < 0.3        |
| Class imbalance ratio              | < 1:5          | 1:5 - 1:10   | > 1:10       |
| Feature missing rate               | < 10%          | 10 - 50%     | > 90%        |
| Isolated node fraction             | < 1%           | 1 - 5%       | > 5%         |
| Degree p99/median                  | < 50           | 50 - 100     | > 100        |
| Node degree (int16 clamp)          | < 32,767       | n/a          | > 32,767     |
| Cold-start fraction (degree 0-1)   | < 5%           | 5 - 10%      | > 10%        |
| Edge type dominance                | No type > 80%  | Any > 90%    | Any < 0.1%   |

## Data Injection Contract

`report_generator.py` produces a final HTML file by performing four exact
string replacements on `report.ai.html`:

| Placeholder                   | Replaced with                                                  |
|-------------------------------|----------------------------------------------------------------|
| `/* INJECT_STYLES */`         | Raw contents of `styles.ai.css`                                |
| `/* INJECT_SCRIPTS */`        | Raw contents of `charts.ai.js`                                 |
| `/* INJECT_ANALYSIS_DATA */`  | JSON-serialized `GraphAnalysisResult` (`dataclasses.asdict`)   |
| `/* INJECT_PROFILE_DATA */`   | JSON-serialized `FeatureProfileResult` (or `{}` if absent)     |

The JS reads these injected JSON strings from hidden script tags:

```html
<script id="analysis-data" type="application/json">/* INJECT_ANALYSIS_DATA */</script>
<script id="profile-data"  type="application/json">/* INJECT_PROFILE_DATA */</script>
```

On page load the JS:

1. Parses both JSON blobs.
2. Populates each section by generating DOM nodes (never `innerHTML` with
   untrusted strings; always `textContent`).
3. Renders the degree distribution as an inline SVG bar chart.
4. Applies color coding (`status-green`, `status-yellow`, `status-red`) based
   on the thresholds above.
5. Hides `#feature-statistics` if the profile data is empty / `{}`.
6. Hides `#advanced` if no Tier 3 or Tier 4 data is present.

### JS Element Contract

The JS queries these DOM IDs. The HTML template must provide them:

- `#report-header`
- `#overview`
- `#data-quality`
- `#feature-statistics`
- `#graph-structure`
- `#advanced`
- `#report-footer`
- `#analysis-data` (hidden JSON script tag)
- `#profile-data`  (hidden JSON script tag)

## Regeneration Instructions

To regenerate `report.ai.html`, `charts.ai.js`, and `styles.ai.css`:

1. Read this `SPEC.md` in full.
2. Implement the sections, element IDs, thresholds, and data injection contract
   exactly as specified.
3. Keep the HTML template minimal (all content is rendered by JS).
4. Keep the JS as a single IIFE with no external dependencies; use DOM helpers,
   not templating libraries.
5. Use the exact color hex values specified in "Constraints".
6. Update the snapshot test golden file at
   `tests/test_assets/analytics/golden_report.html` after regenerating.
