# BQ Data Analyzer for GiGL

## Problem

We train GNNs on billion-node graphs stored in BigQuery. Today there is no way to analyze this data before committing to
a full pipeline run. TFDV statistics generation exists in the codebase but is commented out and tightly coupled to the
DataPreprocessor. Engineers discover data issues (dangling edges, extreme degree skew, missing features, class
imbalance) only after training fails or produces poor results.

A review of 18 production GNN papers (PinSage, LiGNN, TwHIN, GiGL, AliGraph, BLADE, and others) shows that
graph-specific data properties directly determine model quality. Degree distribution alone accounts for 46-230%
performance differences depending on sampling strategy (PinSage, BLADE). Class imbalance is amplified 2-3x by message
passing (GraphSMOTE). Missing features are tolerable up to 90% but hit a phase transition at 95% (Feature Propagation,
ICLR 2022). None of these are caught by standard tabular data quality tools.

## Solution

A standalone `DataAnalyzer` module under `gigl/analytics/` that takes a YAML config pointing at BQ tables and produces a
single HTML report covering data quality, feature distributions (via TFDV/FACETS), and graph structure metrics (via BQ
SQL).

**Alternatives considered:**

| Option                  | Verdict             | Reason                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Aegis from day 1        | Deferred to Phase 2 | Aegis requires per-dataset onboarding (Flowrida config, BigLake sync, JAM permissions). It does not compute graph-specific metrics (degree distributions, hubs, referential integrity, homophily). Good for continuous monitoring of production datasets, not for ad-hoc graph EDA. Phase 2 will read Aegis metrics when available and publish graph metrics back into Aegis. |
| Dataplex Auto DQ        | Rejected            | Google's BQ-native successor to TFDV. Serverless and zero-infra, but lacks FACETS visualizations and schema inference.                                                                                                                                                                                                                                                        |
| Great Expectations      | Rejected            | Industry standard with BQ SQL pushdown, but new dependency and no FACETS. Better for ongoing contracts than one-time profiling.                                                                                                                                                                                                                                               |
| New GiGL pipeline stage | Rejected            | Requires GbmlConfig proto and full orchestration. Analysis should run independently with a simple YAML config.                                                                                                                                                                                                                                                                |

## Architecture

```
YAML Config (BQ tables + columns)
    |
    +-- FeatureProfiler (Dataflow)        reuses existing TFDV Beam components
    |   +-- TFDV stats + FACETS HTML      (GenerateAndVisualizeStats, IngestRawFeatures)
    |   +-- Schema inference + anomalies
    |
    +-- GraphStructureAnalyzer (BQ SQL)   extends existing BQGraphValidator
    |   +-- 25 validation checks across 4 tiers
    |   +-- All structure checks run on full data (no TABLESAMPLE)
    |
    +-- ReportGenerator (AI-owned)
        +-- Single self-contained HTML report -> GCS
```

**Validation tiers:**

- **Tier 1 (Hard fails):** Dangling edges, referential integrity, duplicate nodes. Block training.
- **Tier 2 (Core metrics):** Degree distribution, hubs, isolated nodes, cold-start count, NULL rates, memory budget,
  neighbor explosion estimate. Always-on.
- **Tier 3 (Label/heterogeneous):** Class imbalance, label coverage, edge type distribution. Auto-enabled when
  applicable.
- **Tier 4 (Opt-in):** Reciprocity, homophily, connected components, temporal freshness. Config flags, full data only.

## What the Literature Says Peers Should Know

From 18 papers across Pinterest, LinkedIn, Twitter/X, Snap, Alibaba, Amazon, Google, Uber, Grab, and Meta:

| Finding                                             | Impact                                           | Source                     |
| --------------------------------------------------- | ------------------------------------------------ | -------------------------- |
| Degree distribution determines sampling strategy    | 46-230% quality difference                       | PinSage, BLADE             |
| Cold-start nodes (degree 0-1) need densification    | +0.28% AUC at LinkedIn                           | LiGNN                      |
| GiGL clamps degrees to int16 max (32,767)           | Silent precision loss for super-hubs             | GiGL codebase              |
| Class imbalance amplified 2-3x by message passing   | Minority F1 drops 30-40% at 1:10 ratio           | GraphSMOTE                 |
| Stale edges degrade quality 2-8% AUC                | Temporal freshness is first-class                | LiGNN, AliGraph, Uber/Grab |
| Standard GNNs fail on heterophilic graphs (h < 0.3) | 30-50% accuracy drop; MLP wins at h < 0.2        | Beyond Homophily           |
| Feature missing rate tolerable up to 90%            | Phase transition at 95%; 5% accuracy drop at 90% | Feature Propagation        |

## Phasing

- **Phase 1 (this work):** Self-contained analyzer. YAML config, TFDV + BQ SQL, single HTML report. No external
  dependencies beyond BQ and Dataflow.
- **Phase 2:** Aegis integration. Read existing Aegis profile metrics when available. Publish graph-structure metrics
  back into Aegis for continuous monitoring.

______________________________________________________________________

## Appendix

### A. Full Documentation

| Document                                                     | Contents                                                                                          |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------- |
| [Design doc](20260415-bq-data-analyzer.md)                   | Components, config format, query inventory, result dataclasses, tradeoff analysis, reused code    |
| [Literature review](20260415-bq-data-analyzer-references.md) | 18 papers, 100+ findings with source citations, common themes table, consolidated threshold table |
| `gigl/analytics/data_analyzer/PRD.md`                        | HTML report specification (AI-owned)                                                              |

### B. Key Thresholds (from literature)

| Metric                  | Green | Yellow     | Red    |
| ----------------------- | ----- | ---------- | ------ |
| Edge homophily          | > 0.7 | 0.3 - 0.7  | < 0.3  |
| Class imbalance         | < 1:5 | 1:5 - 1:10 | > 1:10 |
| Feature missing         | < 10% | 10 - 50%   | > 90%  |
| Cold-start fraction     | < 5%  | 5 - 10%    | > 10%  |
| Degree p99/median       | < 50  | 50 - 100   | > 100  |
| Neighbor explosion/seed | < 50K | 50K - 100K | > 100K |

### C. Aegis Phase 2 Details

See the [design doc, Aegis Integration section](20260415-bq-data-analyzer.md#aegis-integration-phase-2) for the full
Phase 2 plan including two PoC proposals: (1) consuming existing Aegis profile tables, (2) publishing graph metrics into
Aegis-compatible tables.
