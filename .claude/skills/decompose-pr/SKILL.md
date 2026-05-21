______________________________________________________________________

## description: Split a feature branch into self-contained, main-based PRs. Invoke when a branch is too large to review effectively (≈300+ LOC, mixes protos/configs/logic, spans unrelated concerns), the user says "decompose / split / break up this PR / this branch is too big", or before requesting code review on a large branch. argument-hint: "[branch-name | (empty for current branch)]"

# Decompose PR

Split a large feature branch into a series of small, self-contained PRs whose **git base** is `main` — never another
PR's branch — so reviewers can evaluate each one independently and the queue keeps moving. When a PR has a logical
dependency on another, its branch contains a **cherry-picked copy** of the predecessor's code so the PR builds and is
reviewable on its own; what stays main-based is the git merge-base, not the diff content. The output is a **dependency
tree / DAG of PRs**, plus a suggested review order — not a linear stack.

**Core principle:** One purpose per PR. Each PR independently reviewable. Every PR's git base is `main`. Decomposition
is the author's job, not the reviewer's.

**Announce at start:** "I'm using the decompose-pr skill to split this branch into reviewable PRs."

## Instructions

When this skill is invoked with `$ARGUMENTS`, execute the following sections in order.

`$ARGUMENTS` selects the branch to decompose:

- empty — operate on the current branch's diff vs `main`
- `<branch-name>` — operate on that branch's diff vs `main`

______________________________________________________________________

### 1. When to use, when not to

**Use this skill when** any of these hold for the branch:

- Diff is ≥300 LOC, or touches ≥10 files.
- Touches more than one category from: protos, resource/task configs, library code, trainer/inferencer wiring, tests,
  docs.
- Contains both refactor and feature work.
- A reviewer has said "this is too much" or you anticipate they will.
- You're about to request code review and the diff feels large to you.

**Do NOT decompose when** the branch is a single atomic change — e.g. a one-file bug fix with its test, a one-line
config flip, a rename plus its callers. Splitting an atomic change creates artificial PRs that are individually broken
or meaningless. Honest is better than artificial.

Reference: Google's small CL guidance — https://google.github.io/eng-practices/review/developer/small-cls.html. ≈100 LOC
is the ideal target for a single PR; 200 LOC in one file is fine, 200 LOC across 50 files is not.

______________________________________________________________________

### 2. Step 0: Identify the atomic unit

**Before splitting anything, name the smallest set of changes that must land together for the runtime to stay green.**

Read the branch. Ask: if I removed change X from this branch, would the test suite still pass? Would the binaries still
start? Would the existing callers still type-check?

Anything outside the atomic unit is fair game to split. Anything inside the atomic unit may still be splittable, but
only via the escape hatches in Step 3 (orphan code, feature flag, deliberate duplication).

Record the atomic unit in a one-line note before continuing — e.g. _"Atomic unit: trainer must import
`WeightedLossModule` and the proto must define `WeightedLossConfig` before the trainer wiring PR can land; everything
else is splittable."_

This step exists because Claude under deadline pressure will otherwise decompose into PRs that individually break
`main`. Naming the atomic unit up front prevents that failure mode.

______________________________________________________________________

### 3. Step 1: Inventory the branch

Run, and read the output:

```bash
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
git log --oneline main..$BRANCH
git diff --stat main...$BRANCH
git diff --name-only main...$BRANCH | sort
```

Group every changed file into one of these categories:

| Category                               | Examples                                                                                            | LOC counted toward                     |
| -------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Protos / schema                        | `proto/snapchat/research/gbml/*.proto`, generated `*_pb2.py*`                                       | Protos PR                              |
| Resource / task configs                | `deployment/configs/*.yaml`, `examples/**/*.yaml`                                                   | Configs PR                             |
| Pure refactor (no behavior change)     | Renamed symbols, restructured class internals, signature changes that preserve callers via defaults | Refactor PR                            |
| New utility / infrastructure           | New module under `gigl/`, no caller yet on `main`                                                   | Orphan-code PR                         |
| Behavioral change ("the feature")      | Code that flips a code path, adds a new pipeline stage, changes outputs                             | Feature PR                             |
| Tests for existing code                | `tests/unit/**` that exercise code already on `main`                                                | Tests-ahead PR or fold into related PR |
| Docs-only / docstring                  | `*.md`, docstring edits, comments                                                                   | Docs PR                                |
| Unrelated cleanup ("while I was here") | Whatever doesn't fit above                                                                          | Always its own PR — see Red Flags      |

If a file straddles categories (e.g. one file has a refactor and a new behavior), note it — Step 4 covers the
`git restore --source ... -p` escape for splitting hunks within a file.

______________________________________________________________________

### 4. Step 2: Classify and slice

Walk each category in this order, applying the matching rule. Build a candidate list of PRs as you go.

**Protos / schema → always its own PR. Lands first.**\
Generated `*_pb2.py*` files ship in the same PR as the `.proto` change. Reviewers for protos are often different people
than reviewers for the consuming code (data team, schema owners, API stability). Cost of separation is near-zero; cost
of bundling is that proto review blocks all consumer review.

**Resource / task configs → own PR. The rule is about review concern, not runtime coupling.**\
Runtime configs (under `deployment/configs/`) are reviewed for ops/runtime impact — memory, GPUs, regions, quotas. Task
configs are reviewed for pipeline correctness. Either way, configs and code go separate so the runtime-concern review
doesn't get buried inside a feature-review.

**Exception — example / demonstration configs** (under `examples/`): these are documentation, not deployment. They show
users how a new option is used. Updates to example configs can ride with the PR that introduces the option being
demonstrated, because the example IS the consumer's documentation. This exception does NOT extend to
`deployment/configs/` or other runtime configs.

**"Trivially coupled"** does NOT mean "the config is inert without the consumer." Inert configs still ship as their own
(usually small, wide-shallow) PR. "Trivially coupled" means a one-line config flip that IS the only behavior change in
the branch (e.g. `enable_feature_x: false → true` and nothing else in the diff).

**Pure refactor → separate from feature PR.**\
A refactor PR must not change behavior (Google rule). Verifiable by: existing tests pass unchanged AND the test suite is
the same. If you can't honestly say that, you have a feature change masquerading as a refactor — surface it. The
refactor PR is reviewed for "did the semantics actually stay the same"; the feature PR is reviewed for "is the new
behavior right." Different review questions, different PRs.

**New utility / infrastructure → ship as orphan code OK.**\
A new module with no caller on `main` is fine as long as the module itself is tested (unit tests in the same PR). The
reviewer evaluates the module on its own merits — does it do what its docstring says, are the tests meaningful. Don't
bundle the caller into the same PR just because "it would look weird with no consumer." That's what the PR description's
"why" sentence is for.

**Behavioral change ("the feature") → smallest possible PR that flips behavior.**\
This is usually the integration PR — it imports the new utility, reads the new config field, calls the new module.
Should be small if the prerequisite PRs did their jobs. If risky or interacting with concurrent work, ship behind a
feature flag default-off.

**Tests for existing code → own PR if they precede a refactor; otherwise stay with the code they test.**\
Tests-ahead-of-refactor is a known good pattern: ship the tests for the existing behavior first, then refactor with
confidence. Tests for new code MUST ship in the same PR as the new code (Google rule, reinforced by Red Flag below).

**Docs-only → own PR. Never ride along with a feature.**\
"While I was here I fixed some docstrings" is a Red Flag. The docstrings get their own one-minute PR. Bundling them into
a feature PR pollutes the diff and tempts the reviewer to either skip the substantive code or skip the docs.

**Unrelated cleanup → its own PR, every time.**\
See Red Flags. No "while I was here" merges.

For each candidate PR, classify it as **wide** (many files, simple changes) or **deep** (few files, complex changes).
Pick one per non-trivial PR. **A PR that is both wide AND deep is too large** — find a split. A small PR (\<100 LOC, \<5
files) can be neither; that's fine. The rule binds at scale.

______________________________________________________________________

### 5. Step 3: Order via a dependency tree (not a stack)

**Every PR's git base is `main`.** Not another PR's branch. But each PR's branch may — and should — contain code from
its logical predecessors, so the PR builds, type-checks, and is reviewable on its own.

This is the key distinction the skill enforces:

- **GitHub-stacking** (forbidden): PR_B's GitHub base is PR_A's branch. Merging PR_A is a prerequisite to merging PR_B.
  Reviewing PR_B is conditioned on PR_A's review state. The chain forces serial merge order.
- **Main-based with logical dependency** (correct): PR_B's GitHub base is `main`. PR_B's branch contains a cherry-picked
  copy of PR_A's commits plus PR_B's own commits. PR_A and PR_B are independent merge candidates from Git's perspective.
  Reviewers review them in dependency order, but neither blocks the other in GitHub.

Build a dependent PR like this:

```bash
git checkout -b decomp/B main
git cherry-pick <PR_A's commits>       # dependency code, so PR_B builds standalone
git cherry-pick <PR_B's own commits>   # this PR's content
git push -u origin decomp/B
gh pr create --base main --title "..." # base is main, NOT decomp/A
```

When PR_A merges, rebase PR_B onto the updated `main`. Git detects that PR_A's commits are already in main and drops
them from PR_B's branch — PR_B's diff against `main` now shows only PR_B's own code:

```bash
git checkout decomp/B
git fetch origin
git rebase origin/main
git push --force-with-lease
```

**The output of decomposition is a dependency DAG of PRs**, not a linear stack. Roots can be reviewed in parallel.
Interior nodes wait on their parents. Diamonds — one PR with two predecessor parents that share a common ancestor — are
fine if the dependencies are real; each leaf's branch just cherry-picks all its ancestors' commits.

```
PR_A (proto)            ──┐
                          ├─► PR_C (module,  cherry-picks A)   ──┐
PR_B (docs cleanup)       │                                       ├─► PR_F (integration, cherry-picks A+C+D)
                          └─► PR_D (refactor, cherry-picks A)   ──┘

PR_E (configs)            ── independent
```

**Smells** (these are the real over-decomposition signals — note that diamonds are NOT a smell under this model):

- **Deep chains (>3 levels).** Leaves carry many ancestors' commits and rebase churn compounds. Fold middle nodes
  together.
- **High in-degree (>2 dependencies on one PR).** Means the dependent branch is huge (it cherry-picks all ancestors) and
  the decomposition has split the wrong things. Fold ancestors into the dependent, or fold sibling ancestors together.
- **Many roots (>4 truly independent PRs).** May be over-decomposed; the branch was probably a collection of unrelated
  work, and some leaves may merit being their own branch entirely.

**Escape-hatch hierarchy** when even the dependency-tree decomposition doesn't make a PR small enough to stand alone:

1. **Orphan code in earlier PR** (preferred). Ship the new utility / module / proto field without any caller. Tests
   cover the new code on its own. Later PRs cherry-pick it in and wire it up.
2. **Feature flag default-off**. Ship the new behavior wired up but inert. A later PR — or a config change — flips the
   flag.
3. **Small deliberate duplication.** Rare. Duplicate a short helper into two PRs; reconcile in a follow-up. Document
   with `# TODO(<gh-user>): de-dupe in follow-up`.
4. **Declare atomic and don't split.** Last resort. Better an honest big PR than a dishonest split that breaks `main`
   between merges.

**Conflicts during decomposition are a signal, not an obstacle.** If cherry-picking PR_A's commits into PR_B's branch
produces conflicts with PR_B's own changes, the two PRs aren't independently understandable — the same hunks pull in two
directions. Re-evaluate the split before forcing it through.

______________________________________________________________________

### 6. Step 4: Build the PRs

**Default workflow — clean per-commit splits:**

```bash
git checkout main
git pull
git checkout -b decomp/<name>
git cherry-pick <commit-sha-or-range>
# verify the build / tests / type-check still pass
make type_check
make unit_test_py PY_TEST_FILES="<relevant_test.py>"
git push -u origin decomp/<name>
```

**Interleaved same-file changes — `cherry-pick` fails or merges nonsense:**

When the feature branch has commits that touched the same file for multiple logical concerns, `cherry-pick` will
misattribute hunks. Use `git restore --source` to pick hunks interactively:

```bash
git checkout main
git pull
git checkout -b decomp/<name>
# pull the file from the feature branch, but only the hunks for THIS PR:
git restore --source=<feature-branch> -p path/to/file.py
# repeat per file
# stage and commit only what you picked
git add -p
git commit -m "..."
```

`git restore --source=<branch> -p` opens the same interactive hunk picker as `git add -p`, but sourcing hunks from
another branch instead of the worktree.

**Alternative: soft-reset and re-stage.** When the feature branch is itself a single big commit, fork once and rebuild
commits from scratch:

```bash
git checkout main
git pull
git checkout -b decomp/<name>
git checkout <feature-branch> -- .   # bring all changes into the worktree
git reset main                        # but uncommit, so everything is unstaged
git add -p path/to/files-for-this-PR  # interactively stage
git commit -m "..."
```

**Renames:** Do the rename in its own wide-shallow PR first, or keep rename+content-edit atomic in one PR if separating
them would obscure review. Never try to split a rename from its content edits across two PRs — Git's rename detection
will produce confusing diffs.

**Generated files (e.g. `*_pb2.py*`):** Ship in the same PR as the source change (`.proto` file). Call out in the
description: `Regenerated via make compile_protos; no hand edits to generated files.` Reviewers skim the generated diff
and focus on the source.

______________________________________________________________________

### 7. Step 5: Write each PR description

Each PR uses the project's `pull_request_template.md`. Fill it in **self-contained** — a reviewer must be able to
evaluate the PR without reading any other PR.

**Forbidden — these are all forward references:**

- `"This will be done in PR #4"`
- `"Builds on top of #123"`
- `"Depends on #N"`
- `"Companion PR: #M"`
- `"Part 2 of 5"`
- `"Final integration PR for feature X"`
- `"Consumers land in follow-up PRs"`

The reviewer should not need to know that other PRs exist to evaluate this one. PR-number cross-links couple review
attention across PRs — exactly what main-based decomposition is supposed to prevent.

**Required substitute — `TODO(<gh-user>)` comments in the code:**

```python
# Good — names intent only
# TODO(kmontemayor): wire up in a follow-up.
# TODO(kmontemayor): wire into GltTrainer.
# TODO(kmontemayor): see INFRA-1234.

# Bad — describes the follow-up's scope, mechanism, or benefit
# TODO(kmontemayor): wire DistLoaderMetricsCollector into GltTrainer behind an
#     opt-in flag so users can enable per-batch sampling/load timing.
# TODO(kmontemayor): see PR #4567.
# TODO(kmontemayor): this is the first half of the metrics feature.
```

The TODO names the author and expresses **intent only**, in one line. It **never** names a PR number, describes the
follow-up's scope, names the mechanism, or sells the benefit. The bad form encodes a mini-spec that goes stale the
moment the implementation drifts; the good form is durable. If the follow-up has a tracked ticket, the comment may
reference the ticket — tickets are stable; PR numbers are not.

**Self-contained PR description anatomy** (mapped onto GiGL's template):

```markdown
**Scope of work done**

<One paragraph: what this PR does and WHY it's an isolated unit.
Do NOT mention other PRs. If the PR adds orphan code with no caller,
say so explicitly: "This PR introduces FooModule with unit tests.
No callers in this PR — FooModule is independently useful and
reviewable on its own merits.">

Where is the documentation for this feature?: <link or N/A>

Did you add automated tests or write a test plan?

<Concrete test plan: what you ran, what passed. Bullet list of `make`
commands or specific test files. For pure-refactor PRs, explicitly
note "existing tests pass unchanged.">

***Updated Changelog.md?*** YES / NO

***Ready for code review?:*** YES / NO
```

If the PR is the integration PR (the one that flips behavior), the "Scope of work" paragraph explains the behavior
change directly — it doesn't say "this completes the X feature." A reviewer who only sees this PR should understand
what's changing and why.

______________________________________________________________________

### 8. Step 6: Present the dependency tree and review order

After sketching the PRs (or after opening them), present the user with the **dependency tree** plus a **suggested review
order** that respects it. Make three things explicit:

1. All PRs are open against `main` (git base) — none are GitHub-stacked.
2. Sibling PRs in the tree (same level, same parents) can be reviewed in parallel.
3. As earlier PRs merge, dependent ones rebase to drop the cherry-picked predecessor commits from their diffs.

Format:

```
Dependency tree (all PRs open against main):

#<A> — <title>  (root, independent)
#<B> — <title>  (root)
  ├─► #<C> — <title>  (cherry-picks B)
  └─► #<D> — <title>  (cherry-picks B)
        └─► #<F> — <title>  (cherry-picks B, C, D)
#<E> — <title>  (root, independent)

Suggested review order:
1. #<A>, #<B>, #<E> — roots. Review in any order; can be in parallel.
2. #<C>, #<D> — depend on #<B>. Review after #<B> lands. #<C> and #<D> are independent — review in parallel.
3. #<F> — depends on #<C> and #<D>. Review last.

After each merge, rebase the open dependents on origin/main before review — Git drops the merged commits from their
diffs.
```

For a flat decomposition with no dependencies, render the tree as a flat list — the format still shows the reviewer that
everything is independent. For a single-chain decomposition (A → B → C), still draw the chain explicitly so the reviewer
sees the structure.

______________________________________________________________________

### 9. Quick Reference

| Change category                                       | Strategy                                                        | Ordering                                                                       |
| ----------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Proto / schema                                        | Own PR, includes generated files                                | First                                                                          |
| Resource / task configs (under `deployment/configs/`) | Own PR (separate from code)                                     | Before consumer lands if inert (orphan-config), or after consumer is on `main` |
| `examples/` configs demonstrating a new option        | May ride with the PR that introduces the option (documentation) | With the consumer                                                              |
| Pure refactor (no behavior change)                    | Own PR; existing tests pass unchanged                           | Before the feature it enables                                                  |
| New utility / module (no caller)                      | Own PR, orphan-code OK, ships with unit tests                   | Before the integration PR                                                      |
| Feature integration (flips behavior)                  | Smallest PR that flips it; feature-flag default-off when risky  | Last in the chain                                                              |
| Tests for existing code                               | Own PR if it precedes a refactor; otherwise with the code       | Before a refactor it protects                                                  |
| Docs-only / docstring                                 | Own PR                                                          | Anytime; doesn't block anything                                                |
| Unrelated cleanup                                     | Own PR, never bundled                                           | Anytime; doesn't block anything                                                |

| Question                                        | Answer                                                                                                                                               |
| ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Can two PRs be git-based on each other?         | No. Every PR's git base is `main`; GitHub-stacking is forbidden.                                                                                     |
| What if PR_B needs PR_A's code to build?        | Cherry-pick PR_A's commits into PR_B's branch. PR_B is still git-based on `main`. The duplicated code drops out when PR_B rebases after PR_A merges. |
| What does the decomposition output look like?   | A dependency DAG of PRs plus a suggested review order. Roots can be reviewed in parallel; dependents wait.                                           |
| Can I write "Depends on #N" in the description? | No. That's a forward reference.                                                                                                                      |
| Can I write `# TODO(me): follow-up` in code?    | Yes, if it names intent only — no PR numbers, no follow-up scope.                                                                                    |
| Can a PR be both wide and deep?                 | Not if it's large. Split it.                                                                                                                         |
| What if the changes truly can't be split?       | Declare atomic and ship as one PR. Honest > artificial.                                                                                              |

______________________________________________________________________

### 10. Common Mistakes

| Rationalization                                                             | Reality                                                                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Stack them (git-base PR_B on PR_A) so each is independently reviewable."   | GitHub-stacking forces serial merge order and ties review state across the chain. Instead, **cherry-pick PR_A's commits into PR_B's branch while keeping PR_B's git base at `main`**. PR_B builds standalone; the duplicated code drops out on rebase after PR_A merges. Both PRs are independent merge candidates in GitHub. |
| "PR_B can't be opened until PR_A merges, because PR_B needs PR_A's code."   | No. PR_B's branch cherry-picks PR_A's commits in addition to PR_B's own. PR_B is openable immediately, with `--base main`. When PR_A merges, rebase PR_B and the duplicated code drops out of the diff.                                                                                                                       |
| "A diamond dependency is a smell."                                          | Under GitHub-stacking, yes. Under main-based PRs with cherry-picked predecessor code, a diamond is just a DAG — the joining PR cherry-picks both parents' commits. The actual smells are deep chains (>3 levels), high in-degree (>2 dependencies on one PR), and many independent roots (>4).                                |
| "Depends on #N is just metadata, not a forward reference."                  | It is a forward reference. The reviewer reads `#N` and now their evaluation of this PR is conditioned on another PR. That's exactly what main-based decomposition prevents.                                                                                                                                                   |
| "Backfill PR numbers in descriptions — it's cheap."                         | The rule isn't about timing. It's about coupling reviewer attention. Backfilled or not, the cross-link is still there.                                                                                                                                                                                                        |
| "Companion PR" annotation is friendly context.                              | It tells the reviewer "you need to read another PR to understand this one." The PR's own description should provide that context.                                                                                                                                                                                             |
| "Final integration PR for the X feature."                                   | Same problem — frames this PR as a chapter in a story the reviewer must follow. The PR description should describe THIS change, not the saga.                                                                                                                                                                                 |
| "The proto change is one line, ride along with the feature."                | Proto reviewers are usually different people reviewing for different concerns (schema stability, API compatibility). Cost of separation is near-zero; cost of bundling is that proto review blocks consumer review.                                                                                                           |
| "Config without a consumer in isolation is confusing."                      | That's what the PR description's "why" sentence is for. Reviewer-confusion isn't fixed by bundling — it's fixed by writing a clearer description.                                                                                                                                                                             |
| "The config field is inert without the consumer, so they belong together."  | The rule is about review concern, not runtime coupling. Inert configs still ship as their own (small, wide-shallow) PR. Bundle ≠ atomic. Exception: `examples/` configs that demonstrate a new option may ride with the consumer that introduces the option — they're documentation.                                          |
| "The TODO needs full context so future-me understands what to do."          | The TODO needs INTENT, not scope. Future-you reads the surrounding code for context. A scope-describing TODO becomes outdated the moment the implementation drifts.                                                                                                                                                           |
| "It's all one logical commit so I'll just split by file."                   | Splitting at file boundaries produces PRs that are individually broken at HEAD. Split by *purpose*, not by file location.                                                                                                                                                                                                     |
| "Tests are big, I'll send them in a follow-up."                             | Direct violation of tests-with-code. Tests ship in the same PR as the new code.                                                                                                                                                                                                                                               |
| "TODO(author) note IS a downstream reference if it says 'follow-up'."       | "Follow-up" is intent, not a reference. The line crosses when the TODO names a PR number or describes the follow-up's scope.                                                                                                                                                                                                  |
| "Wide-and-deep is fine if the PR is overall small."                         | If it's small the rule doesn't bind. If it's not small, pick one.                                                                                                                                                                                                                                                             |
| "I'll stack just this once because rebasing is annoying."                   | Stacking shifts the rebase cost from the author to every reviewer in the chain. The rule exists exactly to prevent that.                                                                                                                                                                                                      |
| "Splitting will create too many PRs and overwhelm reviewers."               | Reviewers prefer five 100-LOC PRs over one 500-LOC PR. Microsoft's 15M-PR analysis confirms throughput improves with smaller PRs even at higher count.                                                                                                                                                                        |
| "While I was here, I fixed some docstrings — they fit in the feature PR."   | They don't. Docstring fixes are a 60-second separate PR. Bundling them pollutes the feature diff.                                                                                                                                                                                                                             |
| "Decomposition makes review easy — we can skip the formal review subagent." | No. Decomposition makes review *fast*, not *optional*. Each PR still gets reviewed; the skill earns its keep by making each review trivial.                                                                                                                                                                                   |

______________________________________________________________________

### 11. Red Flags

**Never:**

- GitHub-stack a PR (set its `--base` to another open PR's branch). Every PR's base is `main`; predecessor code lives in
  the dependent branch via `git cherry-pick`.
- Write `"Depends on #N"`, `"Companion PR"`, `"Part N of M"`, `"Final integration PR"`, or any other inter-PR reference
  in a PR description.
- Bundle unrelated cleanup ("while I was here") into a feature PR.
- Ship code without its tests; ship tests after their code.
- Bundle protos with the code that consumes them.
- Bundle resource/task configs with the feature code.
- Make a PR that is both wide AND deep when it's also large.
- Use this skill's "small PRs" framing to justify skipping code review on the resulting PRs.
- Forget to name the atomic unit before splitting.

**Always:**

- Name the atomic unit (Step 0) before slicing.
- Base every PR on `main`.
- Ship tests with the code they test.
- Rebase later PRs after earlier ones land on `main`.
- Tell the user the review order explicitly (Step 6) with a one-line "why this order" per PR.
- Use `# TODO(<gh-user>): intent` instead of forward references when later work is needed.
- Verify the build / type-check / relevant tests pass on each PR branch before requesting review.
- Declare a change atomic and don't split when splitting would create artificial PRs.

______________________________________________________________________

### 12. Worked example

**Input:** branch `feat/add-weighted-loss`, 940 LOC across 18 files. Contains: proto change, two resource-config YAMLs,
new `WeightedLossModule` (220 LOC, 3 files), trainer wiring (160 LOC, 4 files), `DistNeighborSampler.sample()` refactor
(190 LOC, 1 file), tests (180 LOC), unrelated `gigl/common/uri.py` docstring cleanup (~15 LOC).

**Atomic unit (Step 0):** Trainer must import `WeightedLossModule` AND the proto must define `WeightedLossConfig` for
the trainer-wiring PR to land. Sampler refactor is independent (default `loss_config=None` preserves behavior). Module
is independently useful (orphan-code OK). Configs are inert until the trainer reads them.

**Candidate PRs — all open against `main`:**

| #   | Title                                                                | Files                                                                                               | LOC                  | Wide/Deep    | Cherry-picks | Why it's its own PR                                                         |
| --- | -------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------- | ------------ | ------------ | --------------------------------------------------------------------------- |
| A   | Clean up stale docstrings in `gigl/common/uri.py`                    | 1                                                                                                   | 15                   | — (trivial)  | —            | Unrelated cleanup.                                                          |
| B   | Add `WeightedLossConfig` proto message                               | proto + generated `_pb2.py*`                                                                        | ~30 hand + generated | Wide-shallow | —            | Protos always own PR.                                                       |
| C   | Add `WeightedLossModule` and unit tests                              | `gigl/nn/weighted_loss*.py`, `gigl/nn/__init__.py`, `tests/unit/nn/weighted_loss_test.py`           | 310                  | Deep         | B            | New utility; needs the proto symbols. Tests in same PR.                     |
| D   | Thread optional `loss_config` through `DistNeighborSampler.sample()` | `gigl/distributed/dist_neighbor_sampler.py`, `tests/unit/distributed/dist_neighbor_sampler_test.py` | 250                  | Deep         | B            | Refactor (no behavior change when `loss_config=None`); needs proto symbols. |
| E   | Update resource configs for weighted-loss runs                       | two YAMLs under `deployment/configs/`                                                               | 40                   | Wide-shallow | B            | Config change separated from code. References the new schema in B.          |
| F   | Wire `WeightedLossConfig` into V2 GLT trainer                        | 4 files in `gigl/src/training/`                                                                     | 160                  | Deep         | B, C, D, E   | The integration PR. Flips behavior. Smallest possible.                      |

The **Cherry-picks** column lists which predecessor PRs' commits each branch must cherry-pick in addition to its own
content. F's branch construction looks like:

```bash
git checkout -b decomp/F main
git cherry-pick <B's commits>          # proto symbols
git cherry-pick <C's commits>          # WeightedLossModule (F imports it)
git cherry-pick <D's commits>          # sampler refactor (F passes loss_config to sample())
git cherry-pick <E's commits>          # resource configs (F reads them)
git cherry-pick <F's own commits>      # trainer wiring
git push -u origin decomp/F
gh pr create --base main --title "Wire WeightedLossConfig into V2 GLT trainer"
```

**Dependency tree and review order (Step 6 output):**

```
Dependency tree (all PRs open against main):

#A — Clean up stale docstrings in gigl/common/uri.py        (root, independent)
#B — Add WeightedLossConfig proto message                   (root)
  ├─► #C — Add WeightedLossModule and unit tests            (cherry-picks B)
  ├─► #D — Thread optional loss_config through sampler      (cherry-picks B)
  └─► #E — Update resource configs                          (cherry-picks B)
        └─► #F — Wire WeightedLossConfig into trainer       (cherry-picks B, C, D, E)

Suggested review order:
1. #A — independent of everything. Review anytime.
2. #B — root of the feature tree. Review after #A or in parallel.
3. #C, #D, #E — all depend on #B but are independent of each other. Review in parallel after #B lands.
4. #F — depends on #C, #D, and #E. Review last; rebase after the three land to clean the diff.

After each PR merges, rebase the open dependents on origin/main before review.
```

**What the PR descriptions do NOT say:**

- No `"Depends on #B"` in C, D, E, or F. Each describes its own change.
- No `"This is the final integration PR for WeightedLossConfig"` in F. F's description explains the wiring concretely:
  "Reads `WeightedLossConfig` from the gbml config; instantiates `WeightedLossModule`; passes the config to
  `DistNeighborSampler.sample()` via the new `loss_config` arg."
- No `"Consumers land in follow-up PRs"` in B. Instead: "Adds the proto field. No consumers in this PR — proto changes
  ship independently of consumers in this codebase. Generated via `make compile_protos`; no hand edits to `*_pb2.py*`."

**On the transient diff bloat:** before #B merges, #C's diff against `main` shows B's proto change plus C's module. That
is expected — the reviewer should review #B first (smaller, simpler), and after #B merges, rebasing #C drops B's commits
so #C's diff is just the module. The reviewer never sees more than one PR at a time anyway, in dependency order.

If during PR construction you discover that, say, the sampler refactor (D) is actually behaviorally entangled with the
trainer wiring (F), Step 3's escape-hatch hierarchy applies: prefer orphan code or a feature flag; only
declare-atomic-and-don't-split as a last resort.
