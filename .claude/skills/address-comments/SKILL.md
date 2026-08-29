______________________________________________________________________

## description: Fetch unresolved PR review comments, triage them, and propose an implementation plan. Invoke with a PR URL or number. argument-hint: "<pr-url-or-number>"

# Address Comments

Read GitHub PR review comments, triage each one (address / discuss / skip), propose concrete implementation plans
grounded in the current code, and implement after user approval.

**Requires**: `gh pr view`, `gh api graphql` Recommended allowlist entries for `.claude/settings.local.json`:

```
"Bash(gh pr view:*)"
"Bash(gh api:*)"
```

## Instructions

When this skill is invoked with `$ARGUMENTS`, execute the following sections in order.

______________________________________________________________________

### 1. Parse arguments and verify access

Extract `OWNER`, `REPO`, and `PR_NUMBER` from `$ARGUMENTS`:

| Input          | Example                                     | How to parse                                    |
| -------------- | ------------------------------------------- | ----------------------------------------------- |
| Full URL       | `https://github.com/Snapchat/GiGL/pull/567` | Regex: `github\.com/([^/]+)/([^/]+)/pull/(\d+)` |
| `owner/repo#N` | `Snapchat/GiGL#567`                         | Split on `/` and `#`                            |
| Bare number    | `567`                                       | Resolve repo below                              |

For **bare numbers**, resolve the upstream repo (not the current fork) via:

```bash
gh pr view {N} --json headRepository --jq '.headRepository.owner.login + "/" + .headRepository.name'
```

Run this as the **upfront access check**. If the command fails (permission denied, not authenticated, PR not found),
tell the user and stop.

Then **verify the current branch** matches the PR head:

```bash
PR_HEAD=$(gh pr view {PR_NUMBER} --repo {OWNER}/{REPO} --json headRefName --jq '.headRefName')
CURRENT_BRANCH=$(git branch --show-current)
```

If `$PR_HEAD != $CURRENT_BRANCH`, warn the user:

> Current branch `{CURRENT_BRANCH}` does not match PR head `{PR_HEAD}`. Run `gh pr checkout {PR_NUMBER}` first, then
> re-invoke this skill.

Do **not** proceed to code reads or implementation on a mismatched branch.

______________________________________________________________________

### 2. Fetch PR context and review threads

Run a single GraphQL query to fetch PR metadata, all review threads, and top-level reviews:

```bash
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        title
        body
        headRefName
        baseRefName
        headRefOid
        reviews(first: 50) {
          pageInfo { hasNextPage endCursor }
          nodes { body state author { login } }
        }
        reviewThreads(first: 100) {
          pageInfo { hasNextPage endCursor }
          nodes {
            isResolved
            isOutdated
            line
            originalLine
            startLine
            path
            diffSide
            subjectType
            comments(first: 20) {
              pageInfo { hasNextPage endCursor }
              nodes { body author { login } url }
            }
          }
        }
      }
    }
  }
' -F owner="{OWNER}" -F repo="{REPO}" -F pr="{PR_NUMBER}"
```

If any connection's `pageInfo.hasNextPage` is `true`, paginate with `after: "{endCursor}"` until all data is fetched.

Store the full JSON response for use in subsequent steps.

______________________________________________________________________

### 3. Filter to unresolved items

From the fetched data, separate into two groups:

**Unresolved inline threads**: `reviewThreads` nodes where `isResolved == false`.

**Top-level review bodies**: `reviews` nodes with non-empty `body` and `state` in `["COMMENTED", "CHANGES_REQUESTED"]`.

Discard:

- Resolved threads (`isResolved == true`)
- Bot comments (author login matching common bot patterns: `github-actions`, `dependabot`, etc.)
- Reviews with `state == "APPROVED"` and empty body
- Reviews with `state == "DISMISSED"`

If nothing remains unresolved, tell the user: "No unresolved review comments found on PR #{N}." and stop.

Report counts: "Found X unresolved inline threads and Y top-level reviews."

______________________________________________________________________

### 4. Read referenced code

For each **unresolved inline thread**:

1. **Check if `path` exists** on the current branch. If not, flag the thread as `discuss` with a note: "File `{path}` no
   longer exists on this branch (deleted or renamed)."

2. **Read the full file** at `path` using the Read tool. This follows the same principle as `codex-verify` — do not
   review comments in isolation from their surrounding code.

3. **Identify the commented region** using `line` (preferred) or `originalLine` (fallback). For multi-line ranges, use
   `startLine`..`line`.

4. **Flag edge cases**:

   - `diffSide == "LEFT"`: comment is on removed code. Note: "This comment references deleted/old code."
   - `isOutdated == true`: code has changed since the comment. Note: "Code has changed since this comment was posted —
     verify the approach still applies."
   - Neither `line` nor `originalLine` available: present the comment body with a warning that the exact location could
     not be determined.

**Top-level review bodies** (no `path`/`line`) skip this step — they are handled separately in Step 6.

______________________________________________________________________

### 5. Triage each comment

For each unresolved thread (inline or top-level), classify into one of three categories:

| Category  | Criteria                                                                                    | Action                                                              |
| --------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `address` | Reviewer requests a code change, refactor, test, naming change, etc.                        | Propose concrete implementation with files, functions, line numbers |
| `discuss` | Architectural question, ambiguous request, or concern needing human judgment                | Flag for user decision with context                                 |
| `skip`    | CI/CD bot comment, informational note, or suggestion that contradicts CLAUDE.md conventions | Skip with reasoning                                                 |

For `address` items:

- Read CLAUDE.md conventions to ensure the proposed approach aligns with project standards.
- If the comment touches a shared API or utility, also read adjacent call sites and tests.
- Produce a concrete plan: which file(s) to edit, what to change, and why.

For `skip` items:

- Explain clearly why the comment is being skipped (e.g., "This contradicts the project convention of X per CLAUDE.md",
  or "This is a CI bot notification, not a review request").

For `discuss` items:

- Summarize the concern and what options the user has.

______________________________________________________________________

### 6. Present the plan

Output the triage results in this format, with **separate sections** for top-level reviews and inline comments:

```
## PR Comment Review — #{N}: {title}

**Branch**: {head} -> {base}
**Unresolved threads**: X inline, Y top-level
**After triage**: A to address, B to discuss, C to skip
```

If there are top-level reviews:

```
---

### Top-Level Reviews

#### Review by @{reviewer} ({state})
> {review body}

**Triage**: address | discuss | skip
**Reasoning**: {approach or skip reasoning}
```

For each inline comment:

```
---

### Inline Comments

#### Comment {N}: {file}:{line} -- @{reviewer}
> {first comment body}
> > {reply body, if any}

{warnings if outdated/deleted/LEFT-side}

**Triage**: address | discuss | skip
**Approach**: {concrete implementation plan or skip reasoning}
```

End with a summary table and prompt:

```
---

### Summary

| # | Location | Reviewer | Triage | Summary |
|---|----------|----------|--------|---------|
| 1 | path:42  | user     | address | Rename X to Y |
| 2 | (top-level) | user  | skip   | CI bot noise |

---

Reply with which items to proceed with (e.g. "address all", "address 1,3,5", "skip 2").
```

**This is a hard stop.** Do NOT proceed to implementation until the user replies.

______________________________________________________________________

### 7. Implement approved changes

After the user confirms which items to address, implement them one at a time. For each change:

1. Read the full file being modified.
2. Make the edit using the Edit tool.
3. Run the appropriate verification per `.claude/formatting.md`:
   - **Python files**: `make format_py`, `make type_check`, and `make unit_test_py PY_TEST_FILES="relevant_test.py"` if
     relevant tests exist.
   - **Scala files**: `make format_scala`
   - **Markdown files**: `make format_md`
4. Report what was changed.

After all changes are applied:

1. Run `make check_format` as a final gate.
2. Show a summary of all changes made.
3. Suggest next steps: run full relevant tests, commit, push.
