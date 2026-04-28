# PRD: BQ Data Analyzer HTML Report

## Status

**AI-owned.** An AI agent reads this PRD together with the sibling `SPEC.md` and regenerates `report.ai.html`,
`charts.ai.js`, and `styles.ai.css` when the product intent or technical contract changes. This PRD describes *why* and
*what*; `SPEC.md` describes *how*.

## Problem

Before training a GNN on graph data in BigQuery, engineers need a fast way to see whether the data is healthy enough to
train on. Today they find out only after a Dataflow job crashes or a trainer produces a poor model, which costs days and
thousands of dollars per iteration.

A review of 18 production GNN papers ([reference doc](../../../docs/plans/20260415-bq-data-analyzer-references.md))
found that graph-specific data properties drive 30-230% model quality differences. None of these are caught by standard
tabular data quality tools. We need a report that surfaces these graph-specific issues in a form engineers can act on in
minutes, not days.

## Users

| Persona                                  | Primary need                                                              | Frequency                  |
| ---------------------------------------- | ------------------------------------------------------------------------- | -------------------------- |
| **GNN engineer running an applied task** | Decide whether a new BQ dataset is trainable, and if not, what to fix     | Per new dataset or refresh |
| **Applied task reviewer / tech lead**    | Sanity-check a teammate's dataset choices before approving a training run | Per PR                     |
| **On-call engineer**                     | Triage why a training run degraded vs last week                           | Per incident               |

Out of scope: data scientists doing generic exploratory data analysis, product managers, non-technical stakeholders.

## User Stories

1. **As a GNN engineer**, I point the analyzer at a new BQ node/edge table pair and open the resulting HTML report.
   Within 30 seconds of scrolling I know whether the dataset has any training-blocking issues (dangling edges,
   referential integrity, duplicates).
2. **As a GNN engineer**, I inspect the degree distribution histogram for each edge type and decide whether my planned
   fan-out is realistic or will cause neighbor explosion.
3. **As a reviewer**, I share the GCS link to the report in a PR comment. My teammate opens it in a browser without
   installing anything.
4. **As an on-call engineer**, I run the analyzer on today's data and last week's data and diff the two reports to see
   what changed.
5. **As any of the above**, I expand the collapsed sections I do not care about so the overview stays scannable.

## Goals

1. **Zero-setup viewing.** The report opens in any modern browser with no server, no CDN, no authentication beyond the
   GCS link. Works offline once downloaded.
2. **Action-oriented.** Every numeric finding is color-coded against a literature-derived threshold (green/yellow/red)
   so the reader knows what to do about it.
3. **Traceable.** Every color-coded threshold and every check cites the paper or codebase location that justifies it, so
   readers can verify claims.
4. **Portable.** A single `.html` file that can be shared in chat, stored indefinitely in GCS, and archived alongside
   the training run it describes.
5. **Graph-native.** Surfaces metrics that matter for GNNs specifically (degree distribution, super-hub int16 clamp,
   cold-start fraction, homophily, neighbor explosion), not just generic tabular stats.
6. **AI-regenerable.** The three `.ai.*` assets can be regenerated deterministically from this PRD plus `SPEC.md`
   without human intervention on the HTML/JS/CSS.

## Non-Goals

- **Not a real-time monitoring dashboard.** Aegis covers that
  ([Phase 2](../../../docs/plans/20260415-bq-data-analyzer.md#aegis-integration-phase-2)). This report is a
  point-in-time snapshot.
- **Not a BI tool.** No filtering, drill-down, or ad-hoc querying. The report is a rendered artifact, not an interactive
  app.
- **Not cross-dataset comparison.** Diffing reports is a user workflow (open two tabs), not a report feature.
- **Not a model evaluation report.** This is about training data, not trained model performance.
- **Not accessible (WCAG AA) in v1.** We document this gap and will address it if the report is used by users who need
  it.

## Functional Requirements

Each requirement maps to a section of `SPEC.md` where the implementation contract lives.

**FR-1: Overview at a glance.** The first screen (above the fold) shows total nodes, total edges, node/edge type counts,
and a single green/yellow/red status light summarizing the worst issue found. Rationale: engineers decide "do I need to
look deeper" in the first 5 seconds.

**FR-2: Hard-fail visibility.** Dangling edges, referential integrity violations, and duplicate nodes render red
regardless of magnitude. These block training entirely. The report shows them prominently even if count is exactly one.
Rationale: [GiGL](../../../docs/plans/20260415-bq-data-analyzer-references.md#6-gigl),
[AliGraph (7.1)](../../../docs/plans/20260415-bq-data-analyzer-references.md#7-aligraph) — silent NaN propagation from
referential integrity violations is a production-documented failure mode.

**FR-3: Degree distribution per edge type.** Inline SVG histogram using the six literature-aligned buckets: `0-1`,
`2-10`, `11-100`, `101-1K`, `1K-10K`, `10K+`. Separate in-degree and out-degree. Rationale:
[BLADE](../../../docs/plans/20260415-bq-data-analyzer-references.md#3-blade) showed 230% embedding improvement from
degree-adaptive neighborhoods; the reader needs to see which buckets dominate.

**FR-4: Super-hub warning.** A red call-out appears when any node exceeds the GiGL int16 degree clamp (32,767). Include
the count and the affected edge type. Rationale:
[GiGL (6.2)](../../../docs/plans/20260415-bq-data-analyzer-references.md#6-gigl) — the clamp is silent in production and
corrupts PPR sampling probabilities. Users have no other way to discover this.

**FR-5: Cold-start visibility.** Show the count and fraction of degree-0-1 nodes per type. Color-code the fraction
against the 5% / 10% threshold. Rationale:
[LiGNN (4.1)](../../../docs/plans/20260415-bq-data-analyzer-references.md#4-lignn) — +0.28% AUC from cold-start
densification; the reader decides whether densification is worth investigating.

**FR-6: Optional Tier 3 visibility.** Class imbalance, label coverage, edge type distribution, and per-edge-type node
coverage are shown only when the input data supports them. Rationale: a report full of "not applicable" sections is
noise.

**FR-7: Embedded FACETS.** When feature profiling is available, the FACETS HTML output is embedded inline via
`<iframe srcdoc="...">` so that the TFDV-generated styles do not leak into the main report. Rationale: FACETS is an
industry-standard visualization; engineers already know how to read it.

**FR-8: Collapsible sections.** Every section below the overview is independently collapsible via native
`<details>`/`<summary>` with sensible defaults (hard fails always open; advanced sections closed by default). Rationale:
the report is comprehensive by design, but any one reading needs only the sections relevant to their question.

**FR-9: Literature citations in footer.** The footer lists the 18 source papers used to set thresholds, with inline
references wherever a threshold is color-coded. Rationale: "cite sources" is an explicit user preference, and traceable
thresholds are more defensible than magic numbers.

**FR-10: Raw artifact links.** The footer lists GCS paths to the raw outputs (TFDV stats `.tfrecord`, FACETS `.html` per
table, schema `.pbtxt`) so the reader can dig deeper with other tools.

## Non-Functional Requirements

| Requirement                                      | Target                                                                                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **Load time** (opening the HTML from local disk) | Under 3 seconds for a report with up to 20 tables                                                                        |
| **File size**                                    | Under 1 MB baseline; up to ~10 MB when FACETS iframes are embedded                                                       |
| **Browser support**                              | Latest Chrome, Firefox, Safari, Edge. No IE.                                                                             |
| **Dependencies**                                 | Zero external — no CDN, no Google Fonts, no JS framework. All CSS/JS inlined.                                            |
| **Portability**                                  | Viewing the report over a GCS `gs://` link works without re-download. Saving to disk works.                              |
| **Determinism**                                  | Same input data + same analyzer version produces byte-identical HTML (enables snapshot testing).                         |
| **Security**                                     | All data injected via `textContent`, never `innerHTML`. FACETS embeds are isolated in iframes. No remote resource loads. |
| **Accessibility**                                | Best-effort only in v1: semantic HTML, reasonable color contrast. Full WCAG AA is a non-goal.                            |

## Success Metrics

How we know this PRD was successfully implemented:

1. **Snapshot test stays green.** The golden file at `tests/test_assets/analytics/golden_report.html` matches the
   generated output for a known input. Any intentional change to the report requires a reviewed update to the golden
   file.
2. **Report opens standalone.** Downloading the HTML file and opening it offline produces the same rendering as opening
   it from GCS.
3. **All threshold values match the design doc.** A reviewer can open `SPEC.md`, the `20260415-bq-data-analyzer.md`
   design doc, and the rendered report and confirm all three agree on green/yellow/red cutoffs.
4. **Regeneration works end-to-end.** An AI agent, given only this PRD and `SPEC.md`, regenerates `report.ai.html`,
   `charts.ai.js`, and `styles.ai.css` such that the snapshot test still passes.

## Open Questions

1. **Should the report surface the power-law exponent estimate by default?** We compute it from degree stats (cheap),
   but
   [Demystifying (17.1)](../../../docs/plans/20260415-bq-data-analyzer-references.md#17-demystifying-common-beliefs-in-graph-ml)
   cautions against relying on derived metrics that summarize away the full distribution. Current answer: show it only
   in the Advanced section with a caveat.
2. **Should FACETS embeds be lazy-loaded?** A 20-table report with FACETS per table can be ~10 MB. Lazy loading (iframe
   `loading="lazy"`) would speed first paint but complicates the "single self-contained HTML" goal. Current answer:
   eager load; revisit if reports routinely exceed 10 MB.
3. **Should we support dark mode?** Not in v1. The color-coded thresholds (red/yellow/green) assume a light background;
   a dark theme would need separate color values.

## References

- **Technical spec:** [`SPEC.md`](SPEC.md) in this directory — the contract for regenerating the `.ai.*` files.
- **Design doc:** [`docs/plans/20260415-bq-data-analyzer.md`](../../../docs/plans/20260415-bq-data-analyzer.md) —
  architecture, 4-tier validation, cost control, tradeoff analysis.
- **Literature review:**
  [`docs/plans/20260415-bq-data-analyzer-references.md`](../../../docs/plans/20260415-bq-data-analyzer-references.md) —
  18 papers, 100+ findings with source citations, consolidated threshold table.
- **1-pager:** [`docs/plans/20260416-data-analyzer-1-pager.md`](../../../docs/plans/20260416-data-analyzer-1-pager.md) —
  executive summary for peer engineers.
- **Engineering spec:**
  [`docs/plans/20260416-data-analyzer-engineering-spec.md`](../../../docs/plans/20260416-data-analyzer-engineering-spec.md)
  — per-layer implementation plan.
