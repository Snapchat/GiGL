______________________________________________________________________

## description: Use when the user explicitly asks to decompose, split, or break up a branch or PR. Never auto-trigger based on diff size, file count, or perceived complexity. Plan-only by default; `--execute` or explicit user go-ahead required for any git or PR action. argument-hint: "[branch-name | (empty for current branch)] [--execute]"

# Decompose PR

Split a large feature branch into a DAG of PRs that are **reviewed one wave at a time** so each PR's diff is small and
self-contained at review time. Every PR's git base is `main` — never another PR's branch — so reviewers can evaluate it
independently. Dependent PR branches are built locally via **cherry-pick** so they build and run CI standalone, but the
PR is opened as a **draft** and only marked ready-for-review after its predecessors merge — at which point the rebased
branch's diff against `main` shows only the PR's own code. Output is a dependency DAG plus a wave-based review schedule,
not a linear stack. One purpose per PR; decomposition is the author's job, not the reviewer's.

**Default: plan only.** The skill produces the decomposition and stops at Step 6 — no branches, no commits, no PRs. The
user reads the plan, workshops it, and only when satisfied invokes again with `--execute` or tells Claude to proceed
(Step 7). Running git commands too early creates branches that get thrown away.

**Default: wave-based draft opening.** Under `--execute`, all PRs are opened as drafts (`gh pr create --draft`) so CI
runs against every branch immediately. Only the **root wave** — PRs with no unmerged predecessors — is converted to
ready-for-review (`gh pr ready`). When a root PR merges into `main`, its dependents get rebased (predecessor commits
drop out) and converted to ready-for-review. Repeat per wave.

**Announce at start:** "I'm using the decompose-pr skill to plan a decomposition of this branch."

## Instructions

Execute Steps 0–6 in order, then **stop**. Run Step 7 only if `--execute` was passed or the user has explicitly told you
to proceed.

`$ARGUMENTS`: first positional is the branch to decompose (default: current branch's diff vs `main`). `--execute`
enables Step 7 with per-PR confirmation for `gh pr create`.

______________________________________________________________________

### Step 0: Identify the atomic unit

**Before splitting anything, name the smallest set of changes that must land together for the runtime to stay green.**

Read the branch. Ask: if I removed change X from this branch, would the test suite still pass? Would the binaries still
start? Would the existing callers still type-check?

Anything outside the atomic unit is fair game to split. Anything inside the atomic unit may still be splittable, but
only via the escape hatches in Step 3 (orphan code, feature flag, deliberate duplication).

Record the atomic unit in a one-line note before continuing — e.g. _"Atomic unit: trainer must import
`WeightedLossModule` and the proto must define `WeightedLossConfig` before the trainer wiring PR can land; everything
else is splittable."_

______________________________________________________________________

### Step 1: Inventory the branch

Run, and read the output:

```bash
BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
git log --oneline main..$BRANCH
git diff --stat main...$BRANCH
git diff --name-only main...$BRANCH | sort
```

**LOC is a rough guide, not a hard rule.** The real signals are "reviewer can't hold this in their head" and "this PR
has more than one purpose." A 600-LOC single algorithm is fine as one PR; splitting it artificially makes review harder.

**Tests don't count toward the LOC budget.** Count production LOC only. Tests ship in the same PR as the code they test
(Step 2) — a 100-LOC production change with 250 LOC of tests is a 100-LOC PR.

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

### Step 2: Classify and slice

Walk each category in this order, applying the matching rule. Build a candidate list of PRs as you go.

**Protos / schema → always its own PR. Lands first.**\
Generated `*_pb2.py*` files ship in the same PR as the `.proto` change. Reviewers for protos are often different people
than reviewers for the consuming code (data team, schema owners, API stability). Cost of separation is near-zero; cost
of bundling is that proto review blocks all consumer review.

**Configs → bundle with the consumer that reads them.**\
When a branch introduces a new config option AND the code that consumes it, all of that is **one PR**. The config field
is inert without the consumer; the consumer is the natural review surface for "does this option do what it claims." This
applies to all config locations: `deployment/configs/`, task configs, `examples/`. The only exception is config-only
changes (e.g. ops-tuning diffs adjusting memory limits or quotas) — those are trivially their own PR by being the only
thing in the branch.

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

### Step 3: Order via a dependency tree (not a stack)

The key distinction:

- **GitHub-stacking** (forbidden): PR_B's GitHub base is PR_A's branch. Merging PR_A is a prerequisite to merging PR_B.
  Reviewing PR_B is conditioned on PR_A's review state. The chain forces serial merge order.
- **Main-based with logical dependency** (correct): PR_B's GitHub base is `main`. PR_B's branch contains a cherry-picked
  copy of PR_A's commits plus PR_B's own commits. PR_A and PR_B are independent merge candidates from Git's perspective.
  Reviewers review them in dependency order, but neither blocks the other in GitHub. PR_B is opened as a **draft**
  initially — CI runs against the cherry-picked-predecessor form, but reviewers are NOT pinged until PR_A merges and
  PR_B's branch is rebased to drop A's commits.

Build a dependent PR like this:

```bash
git checkout -b decomp/B main
git cherry-pick <PR_A's OWN commits>   # dependency code, so PR_B builds standalone
git cherry-pick <PR_B's OWN commits>   # this PR's content
git push -u origin decomp/B
gh pr create --base main --title "..." # base is main, NOT decomp/A
```

**Always cherry-pick each PR's *own* commits — not full branch ranges.** "PR_A's own commits" means the commits authored
for PR_A's topic, not the range `main..decomp/A` (which already includes any of A's own ancestors that A also
cherry-picked in). Cherry-picking by full range double-applies shared ancestors and produces conflicts or empty-commit
noise.

When PR_A merges, rebase PR_B onto the updated `main`. Git detects that PR_A's commits are already in main and drops
them from PR_B's branch — PR_B's diff against `main` now shows only PR_B's own code:

```bash
git checkout decomp/B
git fetch origin
git rebase origin/main
git push --force-with-lease
```

Roots can be reviewed in parallel; interior nodes wait on their parents. Diamonds — one PR with two predecessor parents
that share a common ancestor — are fine if the dependencies are real. Each leaf's branch cherry-picks **each ancestor
PR's own commits exactly once, in topological order**; a shared ancestor gets cherry-picked once total into the
descendant, never once per parent.

```
PR_A (proto)            ──┐
                          ├─► PR_C (module,  cherry-picks A)   ──┐
PR_B (docs cleanup)       │                                       ├─► PR_F (integration, cherry-picks A+C+D)
                          └─► PR_D (refactor, cherry-picks A)   ──┘

PR_E (configs)            ── independent
```

**Sanity-check patterns** (each can be the right shape for genuinely large work — pause and ask the question, fold only
if the answer says fold; note that diamonds are NOT on this list under this model):

- **Deep chains (>3 levels).** A chain may genuinely be the right shape for sequential work — proto → utility → refactor
  → integration is four levels and reasonable. But each added level multiplies rebase churn for leaves. Ask: is each
  middle node a distinct purpose, or could two adjacent middle nodes be the same PR? Fold only if adjacent nodes share a
  purpose.
- **High in-degree (>2 dependencies on one PR).** Integration PRs that wire together several independent components
  legitimately have many parents — that's what "integration" means. Ask: are the parents actually independent of each
  other, or did the split pull apart things that belong together? Fold only if the parents are entangled. Do not fold
  parents into the dependent just to drive the in-degree down.
- **Many roots (>4 independent PRs).** A branch that did several truly unrelated things will produce many roots — that's
  correct decomposition, not over-decomposition. Ask: was this one feature or several? If several, the roots may merit
  being their own branches/PR series rather than a single decomposition output. Folding unrelated roots together to
  reduce root count re-creates the original mega-PR.

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

### Step 4: Build the PRs

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

### Step 5: Write each PR description

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

### Step 6: Present the dependency tree and review order

After sketching the PRs (or after opening them as drafts), present the user with the **dependency tree** plus a
**wave-based review schedule** that respects it. Make three things explicit:

1. All PRs are open against `main` (git base) — none are GitHub-stacked.
2. Sibling PRs in the tree (same level, same parents) can be marked ready-for-review in parallel once their parents
   merge.
3. As earlier PRs merge, dependents rebase to drop the cherry-picked predecessor commits, then convert from draft to
   ready-for-review.

Format:

```
Dependency tree (all PRs open against main):

#<A> — <title>  (root, independent)
#<B> — <title>  (root)
  ├─► #<C> — <title>  (cherry-picks B)
  └─► #<D> — <title>  (cherry-picks B)
        └─► #<F> — <title>  (cherry-picks B, C, D)
#<E> — <title>  (root, independent)

Wave-based review schedule:
1. #<A>, #<B>, #<E> — roots. Marked ready-for-review immediately (in any order; can be in parallel).
2. #<C>, #<D> — depend on #<B>. Marked ready-for-review after #<B> lands. #<C> and #<D> are independent — convert in
   parallel.
3. #<F> — depends on #<C> and #<D>. Marked ready-for-review last, after both #<C> and #<D> land.

When wave N's parents merge into `main`, rebase wave N's branches on `origin/main` (cherry-picked predecessor commits
drop out), then convert each from draft to ready-for-review (`gh pr ready <number>`).
```

For a flat decomposition with no dependencies, render the tree as a flat list — the format still shows the reviewer that
everything is independent. For a single-chain decomposition (A → B → C), still draw the chain explicitly so the reviewer
sees the structure.

______________________________________________________________________

### Step 7: Execute (only if `--execute` or user confirmation)

**Skip this section entirely if `--execute` was not passed and the user has not explicitly told you to proceed.** The
Step 6 plan is the deliverable; the user reviews and refines it before any git work happens.

**Preflight — working tree must be clean.** Before any checkout, verify there are no uncommitted local changes. A
`git checkout` carries tracked modifications across branches and they can end up staged into a decomposition branch
without anyone noticing:

```bash
git status --short
# If output is non-empty, STOP. Ask the user to stash, commit, or discard the local changes
# before proceeding with --execute. Do NOT auto-stash; the user should decide.
```

Once the working tree is clean, build **every** planned branch in dependency-tree order (roots first, then children).
Each branch must build standalone — that's what the cherry-picks are for. The branches all get pushed; the PRs all get
opened as drafts; only the root wave is converted to ready-for-review in this same pass.

```bash
git checkout main
git pull
git checkout -b decomp/<descriptive-name>

# For a dependent PR, cherry-pick each predecessor PR's OWN commits (not full branch ranges)
# in topological order — shared ancestors get cherry-picked exactly once. See Step 3.
git cherry-pick <predecessor 1's OWN commits>
git cherry-pick <predecessor 2's OWN commits>
# ... each ancestor's own commits, exactly once, in topological order
git cherry-pick <this PR's OWN commits>

# Verify the branch builds, type-checks, and tests pass
make type_check
make unit_test_py PY_TEST_FILES="<relevant_test.py>"

# Push the branch
git push -u origin decomp/<descriptive-name>
```

**Confirm with the user before each `gh pr create --draft`.** PRs are externally visible; the user should approve titles
and descriptions before they go up. Use the format:

```
Ready to open DRAFT PR for branch `decomp/<name>`:
  Title:  <title>
  Base:   main
  Status: draft (CI runs, no reviewers pinged yet)
  Body:   <paste the Step 5 description sketch>

Proceed? (yes / edit / skip)
```

Wait for explicit confirmation per PR. Don't batch — each `gh pr create --draft` is its own confirmation.

**After all drafts are open, mark the root wave ready-for-review.** Roots are the PRs with no unmerged predecessors —
they're ready to be reviewed immediately because their diffs against `main` are already clean. Confirm with the user
before each conversion:

```
Ready to mark #<N> as ready-for-review (`gh pr ready <N>`)?
Reviewers will be pinged. Proceed? (yes / skip)
```

Only convert the root wave in this pass. Dependent waves stay as drafts; they get converted later, after their
predecessors merge.

**Skipping a PR defers its entire subtree.** Descendants contain cherry-picked copies of the skipped (unreviewed)
predecessor's code, so they cannot merge either. Independent siblings are unaffected. Tell the user explicitly:
"Skipping #<N> defers #<M> and #<K> (its dependents) too."

After the root wave is ready-for-review and all dependent PRs are open as drafts, hand off to the user:

> Roots are ready for review (#<A>, #<B>, #<E>). Dependent PRs (#<C>, #<D>, #<F>) are open as drafts so CI is running.
> Once each root merges into `main`, rebase its dependents on `origin/main` and run `gh pr ready <number>` to advance
> them to ready-for-review. Independent subtrees can advance in parallel.

Restate the dependency tree and wave-based review schedule from Step 6 with the actual PR numbers filled in.

______________________________________________________________________

### Quick Reference

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

| Question                                        | Answer                                                                                                                                                                                                                                                   |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Can two PRs be git-based on each other?         | No. Every PR's git base is `main`; GitHub-stacking is forbidden.                                                                                                                                                                                         |
| What if PR_B needs PR_A's code to build?        | Cherry-pick PR_A's commits into PR_B's branch so PR_B builds and CI runs standalone. Open PR_B as a **draft** (`gh pr create --draft`) with `--base main`. Convert PR_B from draft to ready-for-review only after PR_A merges and PR_B has been rebased. |
| When does a draft PR become ready for review?   | When all of its predecessors in the dependency tree have merged into `main` and the branch has been rebased so the diff against `main` shows only the PR's own code.                                                                                     |
| What does the decomposition output look like?   | A dependency DAG of PRs plus a wave-based review schedule. Roots are marked ready-for-review immediately; dependents stay as drafts until their predecessors merge.                                                                                      |
| Can I write "Depends on #N" in the description? | No. That's a forward reference.                                                                                                                                                                                                                          |
| Can I write `# TODO(me): follow-up` in code?    | Yes, if it names intent only — no PR numbers, no follow-up scope.                                                                                                                                                                                        |
| Can a PR be both wide and deep?                 | Not if it's large. Split it.                                                                                                                                                                                                                             |
| What if the changes truly can't be split?       | Declare atomic and ship as one PR. Honest > artificial.                                                                                                                                                                                                  |

______________________________________________________________________

### Common Mistakes

`common-mistakes.md` next to this file pairs each common decomposition rationalization with the rule it breaks.

______________________________________________________________________

### Red Flags

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

### Worked example

`worked-example.md` next to this file walks one realistic feature branch through a full 4-PR decomposition — candidate
PR table, cherry-pick commands, dependency tree, suggested review order, and notes on what the PR descriptions
deliberately do not say.
