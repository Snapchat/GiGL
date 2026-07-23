---
name: humanify
description: Use when writing or editing code comments, docstrings, or PR/commit descriptions.
---

# Humanify

## Overview

Write for a hurried teammate reading the diff, not a spec reviewer. The code already shows *what* it does — comment the
*why*, concretely, then stop. A comment that merely restates the line next to it rots the moment the code changes;
delete it, or fix the name. Every comment is a promise to keep it true: write fewer and better ones — trimming prose,
never structure.

## Length earns its place when the danger needs it

A warning or workaround keeps the sentences its danger needs — and its bug/spec link. Don't amputate the *why* or the
link to save lines.

## Write the standing state, not the change

A comment is read cold a year from now by someone who never saw this PR. Write every clause as a fact about what the
code IS and the constraint to respect now — rationale in the present tense, phrased as a standing property of the
current choice.

Comparisons, numbers, and rejected alternatives belong here when they explain the live choice — keep them, just state
them as what's true rather than what an experiment found:

```
# t2d: Tau vCPUs are full physical cores (no SMT), so t2d is ~37% cheaper than n2d-highmem
# at this scale. c3 is not cheaper here — it OOMs on this workload.
```

The commit and PR own the change story. Cut anything that only resolves if you saw the PR:

- the thing the code moved *from* — "replaced", "switched from", "the old n2d".
- the event that decided it — "a sweep on 20260711 measured…", "execution_date=…".
- who asked — "per reviewer request", "unchanged per request".
- links to dated experiment write-ups (`docs/plans/…-2026-07-19.md`).

Litmus: read each clause as a stranger a year out. "the old n2d" fails (old versus what?); "per request" fails (whose?);
"~37% cheaper than n2d-highmem here" passes — keep that one.

## Trim prose, not structure

Brevity is about prose, not layout. Keep scannable structure — bulleted `Args`/`Returns`/`Raises`, tables, per-field
lists — even when it costs lines; a reader finding the one field they need fast is the whole point. Two entries that
read alike (e.g. two returned tensors with the same shape) are parallel structure, not redundant prose — keep them as
separate bullets. Follow the project's docstring convention: "write fewer and better" governs redundant prose and
what-restatement, not required structure.

## Enumerations become lists

Prose that names three or more parallel things — flags, knobs, fields, options, steps — hides them: a reader scanning
for one has to parse a whole sentence. Break the set into a list, one item per line; keep the surrounding sentences for
the *why*.

Trigger: you're about to write a parenthetical or comma-run of items — e.g.
`(job_name, task_config_uri, resource_config_uri, cpu_docker_uri, cuda_docker_uri)`. That comma-run is the signal to
make a list.

Before — five knobs buried in a parenthetical:

```
...forwarding the launch knobs (job_name, task_config_uri, resource_config_uri,
cpu_docker_uri, cuda_docker_uri) to the trainer entrypoint...
```

After — the knobs as a list:

```
The trainer entrypoint takes these launch knobs:
  - `--job_name`
  - `--task_config_uri`
  - `--resource_config_uri`
  - `--cpu_docker_uri`     (CPU image; optional)
  - `--cuda_docker_uri`    (GPU image; optional)
```
