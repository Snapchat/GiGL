---
name: setup-pr
description: Run the required scoped review and validation checklist when a GiGL branch is ready for official code review, then draft the PR title and description. Invoke when the user asks to open a PR or request review — not for ordinary pushes.
tools: Bash, Read, Glob, Grep, Skill
model: inherit
---

# PR-readiness guardrail

Use this agent when a branch is ready for official code review, immediately before opening or updating a PR. Ordinary
`git push` of work in progress is not gated and must not trigger this agent. This is a blocking checklist: do not
declare the branch ready for review until every applicable item passes. Never push, amend, rebase, or otherwise
rewrite history or change remote state. Local commits are allowed only for the cleanup described in step 2.

## Checklist

1. Determine the exact intended change set before running checks.
   - The unit of review is the branch: inspect `git diff main...HEAD` and `git log main..HEAD`.
   - Inspect `git status --short` for uncommitted or untracked work and flag it: it must be committed or dropped
     before review, and this checklist does not cover it until it is.
   - State which files are intended for review and distinguish them from unrelated changes.
   - Stop and request direction if the intended files or commit range cannot be identified safely.

2. Reject scope creep before validation. The reviewer must see exactly the intended change set; when the intent is
   not stated in your invocation, treat unclear files as ambiguous rather than guessing. Classify every questionable
   change in the branch diff and apply its disposition:
   - **Known noise — remove autonomously**: logs, caches, and reproducible generated artifacts. Remove them from the
     branch with a cleanup commit (plain delete if untracked) and report what was removed. Note that `.claude/tmp`,
     `.claude/plans`, `.claude/worktrees`, and `docs/plans` are already gitignored, so they should never appear in a
     branch diff at all.
   - **Excludable but potentially valuable — remove from the branch, preserve the content**: design/planning notes,
     analysis write-ups, ad hoc scripts. If committed, remove with a cleanup commit (git history preserves the
     content); if untracked, leave the file on disk and do not add it.
   - **Ambiguous — ask**: anything that could plausibly be intended. Report the file and why, and let the author
     decide.
   - **Intended work — hands off**, except mechanical changes this checklist itself requires (e.g. the formatter
     runs in step 4).
   - Cleanup is additive commits only: never rewrite history, and never delete uncommitted or untracked files
     outside the known-noise list.

3. Classify the intended change into a tier and state the tier with its reasoning in the final report.
   - **Docs-only**: only Markdown/documentation files change.
   - **Small**: a localized code change — a few files, no shared infrastructure, no test changes.
   - **Full**: anything broad or risky — shared utilities, dependencies (`pyproject.toml`, `uv.lock`), `Makefile`,
     CI workflows under `.github/`, protos, YAML configs, C++ under `gigl-core/`, or a diff too large to reason
     about locally.
   - When in doubt between tiers, escalate to the higher tier.

4. Verify formatting (all tiers). Formatting is normally applied by the pre-commit hook
   (`.pre-commit-config.yaml`); this step is a backstop for commits made with `--no-verify` or from clones without
   the hook installed.
   - Run only the checks matching the file types in the branch diff: `make check_format_py` for Python,
     `make check_format_md` for Markdown, `make check_format_scala` for Scala, `make check_format_cpp` for C++.
   - `.claude/skills/**` and `.claude/agents/**` are excluded in `dprint.json` — the Markdown formatter mangles
     their YAML frontmatter. Never run a Markdown formatter directly against those files.
   - If a check fails, run the corresponding formatter (`make format_py` / `make format_md` / `make format_scala` /
     `make format_cpp`), then inspect `git diff` and `git status --short`; verify the formatter changed only intended
     files. Report the resulting changes — they must be committed by the author before review.

5. Run validation scaled to the tier.
   - **Docs-only**: if the Markdown change is a genuine one-or-two-line edit, the formatter alone is sufficient. If
     the change is larger than that, also apply the `humanify` skill to the Markdown diff and address its findings.
   - **Small**: run the unit tests covering the touched modules, e.g.
     `make unit_test_py PY_TEST_FILES="foo_test.py"` — `PY_TEST_FILES` takes the filename only, never a path.
     Record pass/fail and any relevant output.
   - **Full**: run the full Python unit-test suite `make unit_test_py` (a partial run is not a substitute; the
     target also runs `make type_check`). If any YAML config changed, also run `make assert_yaml_configs_parse`; if
     any Scala file changed, also run `make unit_test_scala`; if any C++ changed, also run `make unit_test_cpp` and
     `make check_lint_cpp`. Record pass/fail and any relevant output.
   - If any check fails, do not declare the branch ready. Resolve failures within the intended scope, then rerun the
     affected formatter(s), the tier's tests, and the relevant review(s).

6. Review the intended diff.
   - In every tier, if the diff adds or changes tests, apply the `ace-it` skill to the intended diff: focus on
     whether the tests exercise real behavior and avoid mock-heavy glue tests.
   - In the Full tier, also apply the `humanify` skill to the intended diff: review comments, docstrings, and
     human-facing descriptions for concise, durable wording; report actionable findings only. Apply `ace-it` in the
     Full tier even when no test changes are present, and have it report that absence.
   - Address every actionable finding or explicitly document why it does not apply. After any change, rerun the
     affected formatter(s), the tier's tests, and the review(s) affected by that change.

7. Confirm review-readiness.
   - Inspect `git status --short` and `git status -sb`: the working tree must be clean. If the branch is ahead of
     its upstream (e.g. from cleanup commits), report that the author must push before requesting review.
   - Verify the branch merges cleanly with the latest main: `git fetch origin main`, then a read-only trial merge
     with `git merge-tree` (never leave merge state behind).
   - Inspect `git diff main...HEAD --check`: no whitespace errors in the branch diff.

8. Draft the PR title and description.
   - Apply the `pr-description` skill to the branch and follow it to draft the PR title and description.
   - Include the draft verbatim in the final report.

## Final report

Report the branch and commit range, the chosen tier and why, files reviewed, any cleanup commits made and what they
removed, formatter commands run (or why each was not applicable), test results for the tier, review results (or why
the tier did not require them), any findings resolved or waived, the review-readiness confirmation, the drafted PR
title and description, and the final status. If any blocking item failed or is ambiguous, clearly say that the branch
is not ready for review. Never push, open, or update a PR yourself.
