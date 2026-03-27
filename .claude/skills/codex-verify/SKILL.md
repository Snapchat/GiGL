______________________________________________________________________

## description: Invoke Codex to do review, save findings to .claude/tmp/codex-verify/ argument-hint: "\[unstaged | staged | feature <desc> | followup <review-file> | plan <plan-file-or-desc>\]"

# Codex Verify

Invoke Codex CLI to perform a structured code or plan review, capture the results, and save them to
`.claude/tmp/codex-verify/`.

## Instructions

When this skill is invoked with `$ARGUMENTS`, execute the following sections in order.

______________________________________________________________________

### 1. Parse scope

Determine the review scope from `$ARGUMENTS`:

| Input                    | Scope    | Slug                       | Description                                                                           |
| ------------------------ | -------- | -------------------------- | ------------------------------------------------------------------------------------- |
| (empty) or `unstaged`    | unstaged | `unstaged`                 | Review uncommitted working-tree changes                                               |
| `staged`                 | staged   | `staged`                   | Review staged (cached) changes only                                                   |
| `feature <desc>`         | feature  | `feature-<slugified-desc>` | Review the repo in the context of a specific feature or concern described by `<desc>` |
| `followup <review-file>` | followup | `followup`                 | Re-verify issues from a previous review file                                          |
| `plan <file-or-desc>`    | plan     | `plan-<slugified-desc>`    | Review an implementation plan for correctness, completeness, and feasibility          |

For **feature** and **plan** scopes: slugify the description (lowercase, spaces/punctuation to hyphens, max 40 chars).
Example: `feature embedding resume logic` -> slug `feature-embedding-resume-logic`.

For **followup** scope: the `<review-file>` must be a path to an existing review markdown file (e.g.
`.claude/tmp/codex-verify/20260221-full/review.md`). Read it — you'll need its content for the prompt. If the file
doesn't exist, tell the user and stop.

For **plan** scope: the argument can be either a path to an existing plan file (e.g. `docs/plans/my_plan.md`) or a short
description. If it looks like a file path and exists, read it — you'll include its content in the prompt. If it's a
description, slugify it for the slug. If a file path is given, derive the slug from the filename (e.g.
`docs/plans/ty_type_checker.md` -> slug `plan-ty-type-checker`).

Store the scope name and slug for later steps.

______________________________________________________________________

### 2. Setup workspace

```bash
mkdir -p .claude/tmp/codex-verify
WORK_DIR=".claude/tmp/codex-verify/$(date +%Y%m%d-%H%M%S)-<slug>"
mkdir -p "$WORK_DIR"
```

All files live inside `$WORK_DIR`:

- `PROMPT_FILE="$WORK_DIR/prompt.md"`
- `RESULT_FILE="$WORK_DIR/result.md"`
- `RUN_SCRIPT="$WORK_DIR/run.sh"`
- `REVIEW_FILE="$WORK_DIR/review.md"`

______________________________________________________________________

### 3. Write codex prompt

Write the following prompt to `$PROMPT_FILE`. The prompt has three parts: preamble, scope-specific instructions, and
output format.

**Part A — Preamble** (same for all scopes):

```
You are performing a review on the GiGL repository (Gigantic Graph Learning) — an open-source library for training and inference of Graph Neural Networks at billion-scale.

IMPORTANT: Read these files for context before starting your review:
1. CLAUDE.md — project overview, architecture, coding standards, and conventions

Working directory: <CWD>

REVIEW STANDARDS:
- Be extremely critical and thorough. Do NOT give the benefit of the doubt — if something looks wrong or suspicious, flag it.
- Every issue MUST cite a specific file and line number as evidence. No vague or unsubstantiated claims.
- Read every relevant file fully. Do not skim or summarize — look for subtle bugs, edge cases, and logic errors.
- Check for correctness first, then style. Prioritize issues that would cause runtime failures, data corruption, or security vulnerabilities.
- If you are uncertain whether something is an issue, flag it as Medium with a note explaining your uncertainty rather than omitting it.
- Do not pad findings with trivial style nits to appear thorough. Only report issues that matter.
```

**Part B — Scope-specific instructions**:

For review-based scopes (`unstaged`, `staged`, `feature`), `codex exec review` handles diff discovery natively, so Part
B is **omitted** from the prompt. The prompt contains only Part A (preamble) + Part C (output format). However, append
these focus hints to the preamble:

For **unstaged** / **staged**: append to preamble:

```
FOCUS: Review the changed code for bugs, security issues, correctness, and adherence to project conventions.
For each modified file, also read the full file to understand context — do not review diffs in isolation.
```

For **feature** (with `<desc>` being the user's feature description): append to preamble:

```
FOCUS: Review changes through the lens of: <desc>
For each modified file, also read the full file to understand context — do not review diffs in isolation.
Check for correctness, edge cases, error handling, integration with the rest of the codebase, and any gaps.
```

For **followup** (with `<previous-review-content>` being the content of the previous review file):

```
SCOPE: Follow-up verification of previous review.

A previous code review found the issues listed below. Your job is to check each open issue against the CURRENT code and determine if it has been fixed.

PREVIOUS REVIEW ISSUES:
---
<previous-review-content>
---

For each issue in the previous review:
1. Read the file and line referenced
2. Determine if the issue still exists in the current code
3. Mark it as "fixed" or "still-open" with explanation

Also check if the fixes introduced any NEW issues.
```

For **plan** — if a plan file was provided, include its content as `<plan-content>`. If only a description was given,
use it as `<plan-description>`:

When a plan file is provided:

```
SCOPE: Plan review — <plan-file-path>

You are reviewing an implementation plan BEFORE code is written. Your goal is to catch design issues, missing edge cases, and feasibility problems early — before any implementation effort is wasted.

THE PLAN:
---
<plan-content>
---

Review this plan against the actual codebase for:
1. **Feasibility**: Do the files, classes, and functions referenced in the plan actually exist? Are the assumptions about the codebase accurate?
2. **Completeness**: Are there missing steps, unhandled edge cases, or components the plan doesn't account for?
3. **Correctness**: Will the proposed approach actually work? Are there architectural conflicts or ordering issues?
4. **Impact analysis**: What existing code will break or need updating that the plan doesn't mention? Check imports, tests, configs, and downstream consumers.
5. **Alternatives**: Are there simpler approaches the plan overlooks? Does existing code already solve part of the problem?
6. **Risk areas**: Which parts of the plan are most likely to cause issues during implementation?

Read the files referenced in the plan to verify assumptions. Also check CLAUDE.md for project conventions the plan should follow.
```

When only a description is provided:

```
SCOPE: Plan review — <plan-description>

You are reviewing a proposed approach BEFORE code is written. The user wants to: <plan-description>

Explore the codebase to understand the current state, then assess:
1. **Current state**: What relevant code already exists? What patterns are established?
2. **Feasibility**: Is this approach viable given the current architecture?
3. **Risks**: What are the likely pitfalls or complications?
4. **Recommendations**: Suggest a concrete implementation approach with specific files and changes needed.
5. **Missing pieces**: What does the user need to think about that they might not have considered?

Read CLAUDE.md for project conventions and explore relevant source files.
```

**Part C — Output format** (same for all scopes except plan — see below):

```
FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:

# Code Review — <scope-name>

**Date**: <today YYYY-MM-DD>
**Scope**: <scope description>
**Reviewer**: Codex CLI

## Summary
One paragraph describing what was reviewed and overall assessment.

## Issues

Number each issue sequentially. Use this exact format:

### <N>. <Short title>
- **Status**: `open-bug` | `open-hardening` | `open-style`
- **Priority**: Critical | High | Medium | Low
- **Location**: `file/path.py:line`
- **Description**: What is wrong.
- **Fix**: How to fix it (include code snippet if helpful).

If this is a followup review, also include:

## Fixed Since Last Review
For each issue from the previous review that is now fixed:
### <original-N>. <Short title>
- **Status**: `fixed`
- **Verified**: How you confirmed the fix.

## Still Open
For issues not yet fixed, copy them with updated line numbers if code shifted.

## Statistics
- Total issues: N
- By severity: Critical: N, High: N, Medium: N, Low: N
- By category: bug: N, hardening: N, style: N

## Verdict
- **Clean**: No issues found.
- **Approve with nits**: Only Medium/Low issues.
- **Needs changes**: Any Critical or High issues (list which ones block).
```

**For plan scope**, use this output format instead of the above:

```
FORMAT YOUR OUTPUT EXACTLY AS FOLLOWS:

# Plan Review — <plan-name-or-description>

**Date**: <today YYYY-MM-DD>
**Scope**: Plan review
**Reviewer**: Codex CLI

## Summary
One paragraph overall assessment of the plan's quality and readiness for implementation.

## Feasibility Check

### Verified Assumptions
List assumptions in the plan that you confirmed by reading the code:
- ✅ <assumption> — verified in `file/path.py:line`

### Invalid Assumptions
List assumptions that are WRONG or outdated:
- ❌ <assumption> — actual state: <what you found>

## Issues

Number each issue sequentially. Use this exact format:

### <N>. <Short title>
- **Severity**: Critical | High | Medium | Low
- **Category**: `missing-step` | `wrong-assumption` | `edge-case` | `ordering` | `breaking-change` | `convention-violation` | `alternative-approach`
- **Location**: `file/path.py:line` (if referencing existing code) or "Plan step N" (if referencing the plan)
- **Description**: What is wrong or missing.
- **Recommendation**: How to fix the plan.

## Impact Analysis
List files/modules that will be affected but aren't mentioned in the plan:
- `file/path.py` — reason it's affected

## Recommendations
Bullet list of suggested improvements to the plan before implementation begins.

## Statistics
- Total issues: N
- By severity: Critical: N, High: N, Medium: N, Low: N
- Verified assumptions: N, Invalid assumptions: N

## Verdict
- **Ready to implement**: Plan is solid, no blocking issues.
- **Needs revision**: Has Critical/High issues that should be addressed in the plan before coding.
- **Needs more investigation**: Key unknowns remain — suggest specific areas to explore.
```

______________________________________________________________________

### 4. Launch codex in background

Set timeout based on scope:

- `unstaged`, `staged`: 300 seconds
- `feature`, `followup`, `plan`: 450 seconds

Write `$RUN_SCRIPT` (a bash script) to `$WORK_DIR/run.sh`.

Choose the right codex subcommand based on scope:

- **`unstaged`**: Use `codex exec review --uncommitted` — native diff-aware review.
- **`staged`**: Use `codex exec review --uncommitted` — native diff-aware review (staged changes are included).
- **`feature`**: Use `codex exec review --base main` — reviews all changes on the current branch vs main.
- **`followup`, `plan`**: Use `codex exec` — custom prompt, not git-change-based.

For **review-based scopes** (unstaged, staged, feature), the prompt file contains only the preamble + output format
(Part A + Part C). The scope-specific diff instructions (Part B) are NOT needed — `codex exec review` handles diff
discovery natively. Pass the prompt as custom review instructions via stdin.

```bash
#!/bin/bash
cd <CWD>
# For unstaged/staged:
codex exec review --uncommitted --ephemeral -o "<RESULT_FILE>" - < "<PROMPT_FILE>"
# For feature:
codex exec review --base main --ephemeral -o "<RESULT_FILE>" - < "<PROMPT_FILE>"
# For followup/plan:
codex exec -s read-only --ephemeral -o "<RESULT_FILE>" - < "<PROMPT_FILE>"
```

Then run it as a **single background Bash command** using `run_in_background: true` with timeout set to the scope
timeout plus 30s buffer:

```bash
bash $WORK_DIR/run.sh
```

Tell the user: "Codex review running in background. Will report when done..."

______________________________________________________________________

### 5. Wait for completion

Use `TaskOutput` with `block: true` and timeout matching the scope timeout plus 30s to wait for the background task.

When it returns:

- **Success** (exit code 0) and `$RESULT_FILE` exists and is non-empty: proceed to step 6.
- **Failure** (non-zero exit code): read the task output for error details, report the error to the user, and stop.
  Common issues:
  - "command not found" → codex CLI not installed or not in PATH
  - Invalid flags → check `codex exec --help` for correct syntax
  - Permission errors → sandbox restrictions
- **Timeout**: tell the user the review timed out and stop.

______________________________________________________________________

### 6. Save and present

Read `$RESULT_FILE`. Copy its content to `$REVIEW_FILE`.

Present to the user:

1. **Where the review was saved**: the `$REVIEW_FILE` path
2. **Quick summary**: extract the Statistics and Verdict sections
3. **Critical/High issues**: list any Critical or High priority/severity issues with their number, title, and location
4. **Next steps**: suggest actions based on the verdict:
   - For code review scopes:
     - If "Clean" or "Approve with nits": "No blocking issues. Consider fixing the nits at your convenience."
     - If "Needs changes": "Fix the Critical/High issues above, then re-verify
       with:\\n`/codex-verify followup <REVIEW_FILE>`"
   - For plan scope:
     - If "Ready to implement": "Plan looks solid. Proceed with implementation."
     - If "Needs revision": "Address the Critical/High issues in the plan before implementing, then re-verify
       with:\\n`/codex-verify plan <plan-file>`"
     - If "Needs more investigation": "Key unknowns remain. Explore the listed areas before finalizing the plan."

Also mention: "Full review: `$REVIEW_FILE` | Codex workspace: `$WORK_DIR`"

______________________________________________________________________

### 7. Suggest follow-up

For **code review** scopes (unstaged, staged, feature, followup) — if there are Critical or High issues, end with:

```
To fix and re-verify:
1. Ask me to fix specific issues: "fix issues 3, 8, 15 from the review"
2. Re-verify: /codex-verify followup <REVIEW_FILE>
```

If this IS a followup review, also show the delta:

- How many issues were fixed vs still open
- Any new issues introduced

For **plan** scope — tailor the follow-up to plan revision:

If verdict is "Needs revision", end with:

```
To revise and re-verify the plan:
1. Ask me to update the plan: "address issues 2, 5, 7 from the plan review"
2. Re-verify: /codex-verify plan <plan-file>
```

If verdict is "Ready to implement", end with:

```
Plan looks good! You can proceed with implementation.
```
