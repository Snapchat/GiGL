# SPEC: `gigl.analytics`

Per-symbol contracts for **agent-owned** public symbols in `gigl.analytics`. Each section below is the contract for one
agent-owned symbol; the matching tests under `tests/unit/analytics/` pin its behavior, and the implementation in
`_*_impl_ai.py` is regenerable from this spec.

Other public symbols in `gigl.analytics` are human-owned and not spec'd here — their docstrings in source are
authoritative.

## Table of Contents

Agent-owned (spec'd below):

- [`summary`](#summary) — Per-seed fanout summary for a sampled HeteroData batch.
- `HeteroDataSummary` — Frozen dataclass returned by `summary`; see the `summary` section.

Human-owned (see docstrings in source):

- `log_startup_diagnostics` (`inspect.py`) — Logs sampler-RAM estimate + local partition counts at training startup.

______________________________________________________________________

## `summary(data: HeteroData) -> HeteroDataSummary`

### Purpose

Per-seed fanout summary for sampled `HeteroData` batches from a GiGL's neighbor loader implementations. Use it in
progress logs to spot empty hops, degenerate fanouts, or seed-side mis-attribution at a glance.

```python
from gigl.analytics.inspect import summary

logger.info(f"fanout: {summary(batch)}")
# → "seeds=128 hop1(min=3 med=10 avg=12.5 max=25) hop2(min=12 med=80 avg=91.2 max=240)"
```

Use with batches produced by `DistNeighborLoader` and `DistABLPLoader` (`gigl/distributed/`); both attach the required
sampler metadata.

### Interface

```python
summary(data: HeteroData) -> HeteroDataSummary
```

`HeteroDataSummary` — `@dataclass(frozen=True)`:

- `seeds: int`
- `per_hop: list[HeteroDataSummary.HopStats]` — one per hop, ordered 1..K.
- `__str__` renders `"seeds=N hop1(min=X med=Y avg=Z max=W) hop2(...) ..."` with `avg` formatted to one decimal place.

`HeteroDataSummary.HopStats` — nested `@dataclass(frozen=True)` with `min: int`, `med: int`, `avg: float`, `max: int`:
per-seed neighbor count distribution at that hop.

### Input requirements

The batch must carry sampler-set metadata, or `summary` raises `ValueError`:

- `data[seed_type].batch_size > 0` on exactly one node type.
- `data.num_sampled_nodes: dict[NodeType, Tensor]` at the HeteroData root — entry for the seed type, length `K + 1`.
- `data.num_sampled_edges: dict[EdgeType, Tensor]` at the HeteroData root — entry for every edge type in
  `data.edge_types`, length `K`.

### Error contract

`summary` raises `ValueError` (no other exception types, no fallback) when:

- Zero or multiple node types have `batch_size > 0`.
- `data.num_sampled_nodes` is missing, not a dict, or lacks the seed type.
- `data.num_sampled_edges` is missing, not a dict, or lacks any edge type in `data.edge_types`.
- `num_sampled_nodes[seed_type]` implies fewer than 1 hop.

### Non-goals

- Homogeneous `Data` objects — users can call `summary` by converting to HeteroData first.

### Verification

`tests/unit/analytics/inspect_test.py` pins the behavior with hand-rolled HeteroData fixtures (per-seed walk tables in
the docstrings), `ValueError` guardrails for each contract violation, canonical `__str__` assertions, and end-to-end
runs against `DistNeighborLoader` and `DistABLPLoader`. Implementations must pass that suite.
