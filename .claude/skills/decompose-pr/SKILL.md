______________________________________________________________________

## description: Split a feature branch into self-contained, main-based PRs. Invoke **only when the user explicitly asks** to decompose / split / break up a branch or PR. Never auto-trigger on diff size, file count, or perceived complexity. argument-hint: "[branch-name | (empty for current branch)] [--execute]"

# Decompose PR

Split a large feature branch into a series of small, self-contained PRs whose **git base** is `main` — never another
PR's branch — so reviewers can evaluate each one independently and the queue keeps moving. When a PR has a logical
dependency on another, its branch contains a **cherry-picked copy** of the predecessor's code so the PR builds and is
reviewable on its own; what stays main-based is the git merge-base, not the diff content. The output is a **dependency
tree / DAG of PRs**, plus a suggested review order — not a linear stack.

**Core principle:** One purpose per PR. Each PR independently reviewable. Every PR's git base is `main`. Decomposition
is the author's job, not the reviewer's.

**Default behavior: plan only.** The skill produces the decomposition (dependency tree + per-PR breakdown) and stops at
Step 6. Branches are not created, commits are not cherry-picked, nothing is pushed, no PRs are opened. The user reads
the plan, workshops it (merge two PRs, split one, reorder, swap categories), and only when satisfied invokes again with
`--execute` — or simply tells Claude "go ahead and create them" — to do the git work (Step 7). The breakdown often needs
refinement; running the git commands too early creates branches that need to be thrown away.

**Announce at start:** "I'm using the decompose-pr skill to plan a decomposition of this branch."

## Instructions

When this skill is invoked with `$ARGUMENTS`, execute Steps 0–6 in order, then **stop**. Only run Step 7 if `--execute`
was passed or the user has explicitly told you to proceed.

`$ARGUMENTS` parsing:

- First positional arg (or empty) — branch to decompose. Defaults to the current branch's diff vs `main`.
- `--execute` flag — after producing the plan, also run the git commands in Step 7 (with per-PR confirmation for
  `gh pr create` since PRs are externally visible). Without this flag, Step 7 is skipped entirely.

______________________________________________________________________

### 1. Step 0: Identify the atomic unit

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

### 2. Step 1: Inventory the branch

Run, and read the output:

```bash
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
git log --oneline main..$BRANCH
git diff --stat main...$BRANCH
git diff --name-only main...$BRANCH | sort
```

**On LOC as a sizing heuristic:** LOC is a rough guide, not a hard rule. A single complex algorithm of 600 LOC is fine
as one atomic PR — splitting it artificially makes review harder, not easier. The signals that actually matter are
"reviewer cannot hold this in their head" and "this PR has more than one purpose," not a line count.

**Tests do not count toward LOC budget.** When sizing a PR, count production LOC only. Tests still ship in the same PR
as the code they test (Step 2), but they don't push the PR over a budget line. A 100-LOC production change with 250 LOC
of tests is a 100-LOC PR.

Group every changed file into one of these categories:

| Category                               | Examples                                                                                            | LOC counted toward                        |
| -------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| Protos / schema                        | `proto/snapchat/research/gbml/*.proto`, generated `*_pb2.py*`                                       | Protos PR                                 |
| Configs (new options)                  | `deployment/configs/*.yaml`, `examples/**/*.yaml`, task configs — when introduced with new code     | Bundled with the consumer that reads them |
| Pure refactor (no behavior change)     | Renamed symbols, restructured class internals, signature changes that preserve callers via defaults | Refactor PR                               |
| New utility / infrastructure           | New module under `gigl/`, no caller yet on `main`                                                   | Orphan-code PR                            |
| Behavioral change ("the feature")      | Code that flips a code path, adds a new pipeline stage, changes outputs                             | Feature PR (bundled with its new configs) |
| Tests for existing code                | `tests/unit/**` that exercise code already on `main`                                                | Tests-ahead PR or fold into related PR    |
| Docs-only / docstring                  | `*.md`, unrelated docstring cleanups, comments unrelated to a feature                               | Docs PR                                   |
| Unrelated cleanup ("while I was here") | Whatever doesn't fit above                                                                          | Always its own PR — see Red Flags         |

If a file straddles categories (e.g. one file has a refactor and a new behavior), note it — Step 4 covers the
`git restore --source ... -p` escape for splitting hunks within a file.

______________________________________________________________________

### 3. Step 2: Classify and slice

Walk each category in this order, applying the matching rule. Build a candidate list of PRs as you go.

**Protos / schema → always its own PR. Lands first.**\
Generated `*_pb2.py*` files ship in the same PR as the `.proto` change. Reviewers for protos are often different people
than reviewers for the consuming code (data team, schema owners, API stability). Cost of separation is near-zero; cost
of bundling is that proto review blocks all consumer review.

**Configs → bundle with the consumer that reads them.**\
When a branch introduces a new config option AND the code that reads / pipes / consumes it, all of that is **one PR**.
The config field is inert without the consumer; the consumer is the natural review surface for evaluating "does this new
option do what it claims." Splitting them apart creates a config-only PR that the reviewer cannot meaningfully evaluate
("what is this for?") and a consumer PR whose feature-level review has to handwave the config.

This applies to all config locations: `deployment/configs/`, task configs, example configs in `examples/`. The rule is
the same.

The only time a config change is its own PR is when **the config change is the only change** — e.g. an ops-tuning diff
that adjusts memory limits, regions, or quotas with no code change. In that case the PR is trivially its own PR by
virtue of being the only thing in the branch; no decomposition decision is needed.

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

### 4. Step 3: Order via a dependency tree (not a stack)

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

### 5. Step 4: Build the PRs

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

### 6. Step 5: Write each PR description

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

### 7. Step 6: Present the dependency tree and review order

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

### 8. Step 7: Execute (only if `--execute` or user confirmation)

**Skip this section entirely if `--execute` was not passed and the user has not explicitly told you to proceed.** The
Step 6 plan is the deliverable; the user reviews and refines it before any git work happens.

When the user confirms, run the git commands per the plan. For each PR in dependency-tree order (roots first, then
children):

```bash
git checkout main
git pull
git checkout -b decomp/<descriptive-name>

# For a dependent PR, cherry-pick predecessor commits FIRST, then this PR's own commits:
git cherry-pick <predecessor commit ranges>
git cherry-pick <this PR's commit ranges>

# Verify the branch builds, type-checks, and tests pass
make type_check
make unit_test_py PY_TEST_FILES="<relevant_test.py>"

# Push the branch
git push -u origin decomp/<descriptive-name>
```

**Confirm with the user before each `gh pr create`.** PRs are externally visible; the user should approve titles and
descriptions before they go up. Use the format:

```
Ready to open PR for branch `decomp/<name>`:
  Title: <title>
  Base:  main
  Body:  <paste the Step 5 description sketch>

Proceed? (yes / edit / skip)
```

Wait for explicit confirmation per PR. Don't batch — each `gh pr create` is its own confirmation. Skipping one PR
shouldn't block the rest; mark it deferred and move on. After all confirmed PRs are open, restate the dependency tree
and review order from Step 6 with the actual PR numbers filled in — that's the final hand-off to the user.

______________________________________________________________________

### 9. Quick Reference

| Change category                                  | Strategy                                                       | Ordering                        |
| ------------------------------------------------ | -------------------------------------------------------------- | ------------------------------- |
| Proto / schema                                   | Own PR, includes generated files                               | First                           |
| New configs (added with their consumer)          | Bundle with the consumer that reads them                       | With the consumer               |
| Config-only changes (ops tuning, no code change) | The PR is trivially its own — config IS the change             | Anytime; doesn't block anything |
| Pure refactor (no behavior change)               | Own PR; existing tests pass unchanged                          | Before the feature it enables   |
| New utility / module (no caller)                 | Own PR, orphan-code OK, ships with unit tests                  | Before the integration PR       |
| Feature integration (flips behavior)             | Smallest PR that flips it; feature-flag default-off when risky | Last in the chain               |
| Tests for existing code                          | Own PR if it precedes a refactor; otherwise with the code      | Before a refactor it protects   |
| Docs-only / docstring                            | Own PR                                                         | Anytime; doesn't block anything |
| Unrelated cleanup                                | Own PR, never bundled                                          | Anytime; doesn't block anything |

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
| "Configs should always be in their own PR for ops review."                  | Old rule, no longer applies. New configs ride with the consumer that reads them — a config field without a consumer is incoherent to review. The only config-only PR is one where the config change is the only change in the branch (e.g. an ops-tuning diff with no code change).                                           |
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

The **Cherry-picks** column lists which predecessor PRs' commits each branch must cherry-pick in addition to its own
content. F's branch construction looks like:

```bash
git checkout -b decomp/F main
git cherry-pick <B's commits>          # proto symbols
git cherry-pick <C's commits>          # WeightedLossModule (F imports it)
git cherry-pick <D's commits>          # sampler refactor (F passes loss_config to sample())
git cherry-pick <F's own commits>      # trainer wiring + the two resource configs
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

Suggested review order:
1. #A — independent of everything. Review anytime.
2. #B — root of the feature tree. Review after #A or in parallel.
3. #C, #D — both depend on #B but are independent of each other. Review in parallel after #B lands.
4. #F — depends on #C and #D (and B transitively). Review last; rebase after the two land to clean the diff.

After each PR merges, rebase the open dependents on origin/main before review.
```

**What the PR descriptions do NOT say:**

- No `"Depends on #B"` in C, D, or F. Each describes its own change.
- No `"This is the final integration PR for WeightedLossConfig"` in F. F's description explains the wiring concretely:
  "Reads `WeightedLossConfig` from the gbml config; instantiates `WeightedLossModule`; passes the config to
  `DistNeighborSampler.sample()` via the new `loss_config` arg. Resource configs under `deployment/configs/` are updated
  in this PR because they reference the new schema and the trainer reads them."
- No `"Consumers land in follow-up PRs"` in B. Instead: "Adds the proto field. No consumers in this PR — proto changes
  ship independently of consumers in this codebase. Generated via `make compile_protos`; no hand edits to `*_pb2.py*`."

**On the transient diff bloat:** before #B merges, #C's diff against `main` shows B's proto change plus C's module. That
is expected — the reviewer should review #B first (smaller, simpler), and after #B merges, rebasing #C drops B's commits
so #C's diff is just the module. The reviewer never sees more than one PR at a time anyway, in dependency order.

If during PR construction you discover that, say, the sampler refactor (D) is actually behaviorally entangled with the
trainer wiring (F), Step 3's escape-hatch hierarchy applies: prefer orphan code or a feature flag; only
declare-atomic-and-don't-split as a last resort.
