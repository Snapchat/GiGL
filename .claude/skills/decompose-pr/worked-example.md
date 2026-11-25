# Decompose-PR — Worked Example

A complete decomposition for one realistic GiGL feature branch, grounding the rules in `SKILL.md` against a concrete
diff. Use this file when seeing the procedure on a real branch is more useful than re-reading the rules in the abstract.
Step references below (Step 0, Step 2, Step 3, Step 6) point at the same Steps in `SKILL.md` in this directory.

**Input:** branch `feat/add-weighted-loss`, ~760 LOC of production code (~940 LOC including tests) across 18 files.
Contains: proto change (~30 LOC), two resource-config YAMLs (~40 LOC) read by the trainer, new `WeightedLossModule` (220
LOC production, 3 files; plus ~90 LOC of tests), trainer wiring (160 LOC, 4 files), `DistNeighborSampler.sample()`
refactor (190 LOC production, 1 file; plus ~90 LOC of tests), unrelated `gigl/common/uri.py` docstring cleanup (~15
LOC).

**Atomic unit (Step 0):** The trainer-wiring PR must include the proto field (`WeightedLossConfig`), the new
`WeightedLossModule`, AND the resource configs that reference the new schema — the trainer reads all three. The sampler
refactor is independent (default `loss_config=None` preserves behavior). The module is independently useful (orphan-code
OK). Configs ride with the trainer wiring per the bundle-with-consumer rule (Step 2).

**Candidate PRs — all open against `main`. LOC counts are production-only; tests ship with their code but don't count
toward the budget.**

| #   | Title                                                                | Files                                                                                               | LOC (prod) | Wide/Deep    | Cherry-picks | Why it's its own PR                                                                                                                      |
| --- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------- | ------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| A   | Clean up stale docstrings in `gigl/common/uri.py`                    | 1                                                                                                   | 15         | — (trivial)  | —            | Unrelated cleanup.                                                                                                                       |
| B   | Add `WeightedLossConfig` proto message                               | proto + generated `_pb2.py*`                                                                        | ~30 (+gen) | Wide-shallow | —            | Protos always own PR.                                                                                                                    |
| C   | Add `WeightedLossModule` (+ unit tests)                              | `gigl/nn/weighted_loss*.py`, `gigl/nn/__init__.py`, `tests/unit/nn/weighted_loss_test.py`           | 220        | Deep         | B            | New utility; needs the proto symbols. Tests ship in same PR (don't count toward budget).                                                 |
| D   | Thread optional `loss_config` through `DistNeighborSampler.sample()` | `gigl/distributed/dist_neighbor_sampler.py`, `tests/unit/distributed/dist_neighbor_sampler_test.py` | 190        | Deep         | B            | Refactor (no behavior change when `loss_config=None`); needs proto symbols.                                                              |
| F   | Wire `WeightedLossConfig` into V2 GLT trainer (+ resource configs)   | 4 files in `gigl/src/training/`, two YAMLs in `deployment/configs/`                                 | 200        | Deep         | B, C, D      | Integration PR. Reads the new proto field, uses module + sampler, and the resource configs come along per the bundle-with-consumer rule. |

The **Cherry-picks** column lists which predecessor PRs each branch logically depends on. F's branch construction
cherry-picks each ancestor PR's **own commits exactly once** in topological order — never `main..decomp/C` or
`main..decomp/D` as full ranges, since those already contain B and would replay it after we apply B directly:

```bash
git checkout -b decomp/F main
git cherry-pick <B's OWN commits>      # proto symbols
git cherry-pick <C's OWN commits>      # WeightedLossModule (F imports it). C's branch also contains B's commits;
                                       # we don't re-include them here — B was already applied above.
git cherry-pick <D's OWN commits>      # sampler refactor. Same caveat: D's branch contains B's commits; not re-included.
git cherry-pick <F's OWN commits>      # trainer wiring + the two resource configs
git push -u origin decomp/F
gh pr create --base main --title "Wire WeightedLossConfig into V2 GLT trainer"
```

**Dependency tree and review order (Step 6 output):**

```
Dependency tree (all PRs open against main):

#A — Clean up stale docstrings in gigl/common/uri.py        (root, independent)
#B — Add WeightedLossConfig proto message                   (root)
  ├─► #C — Add WeightedLossModule (+tests)                  (cherry-picks B)
  ├─► #D — Thread optional loss_config through sampler      (cherry-picks B)
  └─► #F — Wire WeightedLossConfig into trainer (+configs)  (cherry-picks B, C, D)

Wave-based review schedule:
1. #A — independent of everything. Marked ready-for-review immediately.
2. #B — root of the feature tree. Marked ready-for-review immediately (in parallel with #A).
3. #C, #D — both depend on #B; both opened as drafts initially. After #B lands, rebase #C and #D and mark them
   ready-for-review (in parallel — they're independent of each other).
4. #F — depends on #C and #D; opened as a draft initially. After #C and #D both land, rebase #F and mark it
   ready-for-review.

All five PRs exist as drafts from the moment Step 7 runs, so CI runs on every branch from the start. Conversion to
ready-for-review happens wave-by-wave as predecessors merge.
```

**What the PR descriptions do NOT say:**

- No `"Depends on #B"` in C, D, or F. Each describes its own change.
- No `"This is the final integration PR for WeightedLossConfig"` in F. F's description explains the wiring concretely:
  "Reads `WeightedLossConfig` from the gbml config; instantiates `WeightedLossModule`; passes the config to
  `DistNeighborSampler.sample()` via the new `loss_config` arg. Resource configs under `deployment/configs/` are updated
  in this PR because they reference the new schema and the trainer reads them."
- No `"Consumers land in follow-up PRs"` in B. Instead: "Adds the proto field. No consumers in this PR — proto changes
  ship independently of consumers in this codebase. Generated via `make compile_protos`; no hand edits to `*_pb2.py*`."

**Why drafts:** before #B merges, #C's branch contains B's proto change plus C's module — that's what lets #C build and
run CI standalone. But reviewers never see that pre-rebase form: #C is a draft until #B merges. When #B lands and #C is
rebased onto `origin/main`, B's commits drop out and #C's diff is just the module. The wave-based draft model exists
precisely to keep transient diff bloat off the reviewer's plate.

If during PR construction you discover that, say, the sampler refactor (D) is actually behaviorally entangled with the
trainer wiring (F), Step 3's escape-hatch hierarchy applies: prefer orphan code or a feature flag; only
declare-atomic-and-don't-split as a last resort.
