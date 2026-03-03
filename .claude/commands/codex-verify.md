______________________________________________________________________

## description: Delegate code review to Codex CLI, save findings to /tmp/codex-verify/ argument-hint: "\[full | unstaged | staged | feature <desc> | followup <review-file>\]"

# Codex Verify

Delegate a structured code review to Codex CLI, capture the results, and save them to `/tmp/codex-verify/`.

## Instructions

When this skill is invoked with `$ARGUMENTS`, execute the following sections in order.

______________________________________________________________________

### 1. Parse scope

Determine the review scope from `$ARGUMENTS`:

| Input                    | Scope    | Slug                       | Description                                                                           |
| ------------------------ | -------- | -------------------------- | ------------------------------------------------------------------------------------- |
| (empty) or `full`        | full     | `full`                     | Full repo review of all source files                                                  |
| `unstaged`               | unstaged | `unstaged`                 | Review uncommitted working-tree changes                                               |
| `staged`                 | staged   | `staged`                   | Review staged (cached) changes only                                                   |
| `feature <desc>`         | feature  | `feature-<slugified-desc>` | Review the repo in the context of a specific feature or concern described by `<desc>` |
| `followup <review-file>` | followup | `followup`                 | Re-verify issues from a previous review file                                          |

For **feature** scope: slugify the description (lowercase, spaces/punctuation to hyphens, max 40 chars). Example:
`feature embedding resume logic` -> slug `feature-embedding-resume-logic`.

For **followup** scope: the `<review-file>` must be a path to an existing review markdown file (e.g.
`/tmp/codex-verify/20260221-full/review.md`). Read it — you'll need its content for the prompt. If the file doesn't
exist, tell the user and stop.

Store the scope name and slug for later steps.

______________________________________________________________________

### 2. Setup workspace

```bash
mkdir -p /tmp/codex-verify
WORK_DIR="/tmp/codex-verify/$(date +%Y%m%d-%H%M%S)-<slug>"
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
You are performing a code review on the GiGL repository (Gigantic Graph Learning) — an open-source library for training and inference of Graph Neural Networks at billion-scale.

IMPORTANT: Read these files for context before starting your review:
1. CLAUDE.md — project overview, architecture, coding standards, and conventions

Working directory: <CWD>
```

**Part B — Scope-specific instructions**:

For **full**:

```
SCOPE: Full repository review.

Review all source files in the repository. Focus on:
- gigl/ — core library (pipeline components, distributed training, neural network modules, common utilities)
- scripts/ — utility scripts
- containers/ — Dockerfiles
- deployment/ — deployment configs and resource configs
- proto/ — Protobuf definitions
- examples/ — example configs and usage
- tests/ — unit, integration, and e2e tests
- scala/ and scala_spark35/ — Scala/Spark code (legacy)
- tools/ — tooling

Run `find . -name '*.py' -not -path './.venv/*' -not -path './build/*' -not -path './__pycache__/*' -not -path './gigl.egg-info/*' -not -path './snapchat/*' -not -path './typings/*'` to discover Python files. If there are more than 100 files, warn in the output that coverage may be incomplete and prioritize gigl/ and tests/.
Also discover Scala files: run `find scala/ scala_spark35/ -name '*.scala'`.
Also review config files: run `find examples/ deployment/ -name '*.yaml'` and check for consistency with code and proto definitions.
Read each relevant file and review it.
Cross-check interactions between pipeline components (ConfigPopulator → DataPreprocessor → Trainer → Inferencer → PostProcessor), proto definitions, and distributed training infrastructure.
```

For **unstaged**:

```
SCOPE: Uncommitted working-tree changes.

Review all uncommitted changes — unstaged, staged, and untracked:
1. Run `git diff` to see unstaged modifications to tracked files.
2. Run `git diff --cached` to see staged changes.
3. Run `git ls-files --others --exclude-standard` to list untracked files, then read each one.
Focus on whether the changes introduce bugs, security issues, or break existing patterns.
```

For **staged**:

```
SCOPE: Staged changes only.

Review only staged changes. Run `git diff --cached` to see what's staged.
Focus on whether the changes are correct, complete, and ready to commit.
```

For **feature** (with `<desc>` being the user's feature description):

```
SCOPE: Feature-focused review — <desc>

Review the repository specifically through the lens of: <desc>
Identify files relevant to this feature/concern, read them, and review for:
- Correctness of the feature implementation
- Edge cases and error handling
- Integration with the rest of the codebase
- Any gaps or missing pieces
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

**Part C — Output format** (same for all scopes):

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

______________________________________________________________________

### 4. Launch codex in background

Set timeout based on scope:

- `full`: 600 seconds
- `unstaged`, `staged`: 300 seconds
- `feature`, `followup`: 450 seconds

Write `$RUN_SCRIPT` (a bash script) to `$WORK_DIR/run.sh`.

Use `codex exec` (not `codex exec review`) for all scopes. This gives us `-o` for clean output capture and allows a
custom prompt for structured output format. The prompt itself tells codex what to review (git diff, full files, etc.).

```bash
#!/bin/bash
cd <CWD>
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
3. **Critical/High issues**: list any Critical or High priority issues with their number, title, and location
4. **Next steps**: suggest actions based on the verdict:
   - If "Clean" or "Approve with nits": "No blocking issues. Consider fixing the nits at your convenience."
   - If "Needs changes": "Fix the Critical/High issues above, then re-verify
     with:\\n`/codex-verify followup <REVIEW_FILE>`"

Also mention: "Full review: `$REVIEW_FILE` | Codex workspace: `$WORK_DIR`"

______________________________________________________________________

### 7. Suggest follow-up

If there are Critical or High issues, end with:

```
To fix and re-verify:
1. Ask me to fix specific issues: "fix issues 3, 8, 15 from the review"
2. Re-verify: /codex-verify followup <REVIEW_FILE>
```

If this IS a followup review, also show the delta:

- How many issues were fixed vs still open
- Any new issues introduced
