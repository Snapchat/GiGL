______________________________________________________________________

## name: humanify description: Use when writing or editing code comments, docstrings, or PR/commit descriptions.

# Humanify

## Overview

Write for a hurried teammate reading the diff, not a spec reviewer. The code already shows *what* it does — comment the
*why*, concretely, then stop. Every comment is a promise to keep it true, so write fewer and better ones — trimming
prose, never structure.

## Comment the why, not the what

The code is the *what*. A comment earns its place only by speaking at a **different level than the code** — higher
(intent, rationale) or lower (a precise caveat). A comment at the *same* level as the line next to it just restates it:
cut it.

Why this, not just "be concise": a *what*-comment mirrors the code, so it rots the moment the code changes — and a
comment that contradicts the code is worse than none (code and comments stay in sync only ~20% of the time; mismatched
comments track with bugs). A *why*-comment describes intent, which barely moves.

## First pass — should this comment exist at all?

Decide before you shorten:

- Restates the code → **delete it** (`# increment i` adds nothing).
- A clearer name would remove the need → **rename instead of commenting**.
- Commented-out code, changelog/byline/banner noise → **delete** (the VCS remembers).

## The contract — what a kept comment looks like

In order, nothing else:

1. A plain one-line lead — what it is or what to do.
2. A concrete example — the real command, path, or value.
3. The one-sentence *why* — the reason a reader would otherwise get wrong.

## When a longer comment earns its length

Some comments must be long — don't amputate these; give them the sentences the danger needs, plus a link:

- Rationale and **rejected alternatives** ("not X because…") — unrecoverable from code.
- **Warnings**: not thread-safe, O(n²), mutates input, ordering matters.
- A **workaround / unidiomatic line** — say why it's needed and link the bug/spec, so nobody "fixes" it into a break.

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

The commit message and PR own the change story, so those details stay out of the comment. Cut anything that only
resolves if you saw the PR:

- the thing the code moved *from* — "replaced", "switched from", "the old n2d".
- the event that decided it — "a sweep on 20260711 measured…", "execution_date=…".
- who asked — "per reviewer request", "unchanged per request".
- links to dated experiment write-ups (`docs/plans/…-2026-07-19.md`).

Litmus: read each clause as a stranger a year out. "the old n2d" fails (old versus what?); "per request" fails (whose?);
"~37% cheaper than n2d-highmem here" passes — keep that one.

## Trim prose, not structure

Brevity is about prose, not layout. Keep scannable structure — bulleted `Args`/`Returns`/`Raises`, tables, per-field
lists — even when it costs lines; a reader finding the one field they need fast is the whole point of writing for a
hurried teammate. Two entries that read alike (e.g. two returned tensors with the same shape) are parallel structure,
not redundant prose — keep them as separate bullets; parallel form scans faster than a merged sentence. Follow the
project's docstring convention (e.g. Google-style `Args`/`Returns`/`Raises`): "write fewer and better" governs redundant
prose and what-restatement, not required structure.

## Before / after (real)

Before — an essay that buries the point:

```
Invoke with -m rather than `python gigl/src/training/trainer.py`: in the image
gigl is an editable install whose finder points at a build-time path that
doesn't exist at runtime, so a plain file invocation can't import gigl.common.
-m puts the package root on sys.path so gigl resolves to the source baked into
the image — matching how every other pipeline entrypoint is launched. A file
invocation would import a half-broken gigl and fail deep in training with a
confusing traceback...
```

After — lead, example, one why:

```
Launch the trainer as a module, not by file path:
    python -m gigl.src.training.trainer --job_name=my_job \
      --task_config_uri=... --resource_config_uri=...

Run as -m (not python <path>) so `import gigl` resolves to the copy installed
in the image, not the working directory.
```

## Quick reference

| Instead of                                                | Write                                         |
| --------------------------------------------------------- | --------------------------------------------- |
| a comment that restates the code                          | delete it — or a clearer name                 |
| a paragraph justifying one line                           | one sentence, or nothing                      |
| how sys.path / finders / probes work                      | "run as -m so imports resolve"                |
| "basically", "in general", hedges                         | the claim, stated plainly                     |
| "t2d replaced n2d after the 20260711 sweep measured -37%" | "t2d is ~37% cheaper than n2d-highmem here"   |
| "max replicas unchanged per reviewer request"             | the live constraint: "don't rescale replicas" |
| collapsing bulleted Args/Returns into prose to save lines | keep the bullets — structure *is* readability |

## Litmus test

Read it aloud. Sounds like a person explaining to a colleague → ship it. Documentation of documentation → cut it,
rename, or delete.
