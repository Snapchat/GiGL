---
name: pr-description
description: Use when writing or revising a pull request description — before creating or updating a PR, or when asked to summarize a branch for review.
---

# PR Description

Write for a reviewer deciding why the change matters and what outcome to expect. Explain the product or operational intent first; use the diff and commits to support it, not to narrate every edit.

## Gather evidence

1. Identify the current branch and determine the comparison base. Prefer an explicit base branch, PR metadata, merge-base, or repository convention. Do not assume a base branch when the evidence is ambiguous.
2. Inspect the base diff and relevant commits. Separate:
   - **Intent and outcome:** the user or system problem addressed and the resulting behavior.
   - **High-level approach:** the few design choices that accomplish the outcome. Filenames, regenerated artifacts, fixtures, and mechanical edits are not approach.
   - **Implementation detail:** file-by-file or mechanical changes that normally belong in the diff, not the description.
3. Inspect available tests, CI results, experiment outputs, and launched-job links. Report only evidence that is present and attributable to this change.
4. Identify externally visible changes. Call out material API, configuration, CLI, behavior, compatibility, or operational-interface changes, each with one illustrative example. If the change breaks existing configs, callers, or jobs, list every affected field or interface; otherwise the example stands alone.

If the intent, base, or verification evidence cannot be established from the branch and supplied context, ask only for that missing information. If evidence shows a relevant verification or performance job ran but its URL is missing, ask for the URL. Do not infer a rationale from implementation alone or invent results, links, baselines, measurements, or validation.

## Draft the description

Before drafting, read `../humanify/SKILL.md` and apply its prose rules — with one inversion: the PR description IS the change story, so "replaced X with Y", dates, and measured deltas belong here.

Use this structure, omitting sections that are empty or irrelevant:

```markdown
## Summary

Explain why this change is needed and the outcome in a short paragraph or a few bullets.

## Approach

For a substantive change, give at most a few design-level bullets explaining how the outcome is achieved. Omit for trivial changes.

## User-facing changes

Describe material API, config, CLI, behavioral, or compatibility changes. Show only essential illustrative examples.

## Verification

State only checks, tests, CI, or jobs actually run and their observed result. Do not count tests added, fixtures updated, or schemas regenerated as verification unless the corresponding check ran.

## Performance / launched jobs

Link each relevant launched job. Summarize supported performance characteristics compactly.

If two jobs ran the same workload with only this change differing, show the directly computable deltas. Otherwise report each job on its own. Do not estimate missing data or claim the change caused a result.

| Job | Workload / configuration | Relevant characteristic | Result / status |
| --- | --- | --- | --- |
| [Job name](URL) | Important setup only | e.g. throughput, latency, cost, scale | Measured value or running status |
```

Use the performance table only when a relevant job was launched and a link is available. Preserve uncertainty: label a job as running, failed, or not yet evaluated when that is all the evidence supports. Do not turn the PR into a changelog, repeat the diff, or include frivolous implementation history.

## Revise existing text

Keep supported facts, then remove laundry lists and low-value details. Reorder the description so rationale and outcome precede approach, and make user-facing changes and verification easy to scan. Keep the final wording high-level, concrete, and proportional to the change.
