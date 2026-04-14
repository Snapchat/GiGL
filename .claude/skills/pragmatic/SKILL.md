______________________________________________________________________

## description: Apply KISS and pragmatic simplification to code, diffs, or plans. Invoke when the user says kiss, keep it simple, be pragmatic, don't overcomplicate, simplify this, reduce abstraction, avoid overengineering, or asks for the smallest correct change. argument-hint: "[file-path | plan | (empty for unstaged diff)]"

# Pragmatic

Scan code, diffs, or plans for over-engineering and produce concrete simplification recommendations.

## Instructions

When this skill is invoked with `$ARGUMENTS`, execute the following sections in order.

______________________________________________________________________

### 1. Parse scope

Determine the review mode from `$ARGUMENTS`:

| Input                      | Mode   | Target                                                   |
| -------------------------- | ------ | -------------------------------------------------------- |
| (empty)                    | `diff` | Unstaged changes (`git diff`)                            |
| A path to an existing file | `file` | That specific file                                       |
| `plan`                     | `plan` | The current conversation's plan or most recent plan file |

Rules:

- If `$ARGUMENTS` is a file path and the file exists, use **file** mode.
- If `$ARGUMENTS` is the word `plan`, use **plan** mode. If a plan file path follows (e.g. `plan docs/my_plan.md`), read
  that file.
- Otherwise, use **diff** mode and treat any extra words as a focus hint (e.g. "the sampler" narrows review to
  sampler-related changes).

______________________________________________________________________

### 2. Read the target

Gather the code to review based on mode:

- **diff**: Run `git diff` (unstaged). If empty, try `git diff --cached` (staged). If both empty, tell the user there
  are no changes to review and stop.
- **file**: Read the file with the Read tool.
- **plan**: Read the plan file, or extract the plan from conversation context.

______________________________________________________________________

### 3. Scan for anti-patterns

Review the target code against these specific checks. For each violation found, record the file path, line number, which
check it violates, and what the simpler alternative is.

**Checks to apply:**

1. **Re-implemented utility** — Code that reimplements something already available in the codebase. Common ones to watch
   for:
   - `gigl.common.utils.retry.retry` — decorator with backoff, deadline, and exception filtering. Do not hand-roll retry
     loops with `time.sleep()`.
   - `gigl.src.common.utils.timeout.timeout` — decorator for process-level timeouts. Do not hand-roll signal/threading
     timeout logic.
   - `gigl.common.logger.Logger` — GCP-aware logger. Do not use `logging.getLogger()` directly.
   - `Uri.get_basename()`, `Uri.join()`, `Uri / other` — URI path operations. Do not parse URIs with manual
     `.split("/")` calls.
   - `gigl.common.collections.frozen_dict.FrozenDict` — immutable dict. Do not hand-roll frozen dict wrappers.
2. **Unnecessary abstraction** — A new class, factory, builder, registry, or strategy object that has only one
   implementation or caller.
3. **One-off tiny type** — A dataclass or wrapper with 1-2 fields that could be a tuple, dict, or existing type.
4. **Flag-heavy function** — A function accumulating boolean flags, optional callbacks, or modal branching that should
   be split into separate entry points.
5. **Inlineable helper** — A private function called once or twice that is only a few lines long and does not materially
   improve readability or testability.
6. **Premature generalization** — Abstraction justified only by hypothetical future use ("we might need this later")
   with no second caller today.

**Correctness guardrails — do NOT flag these as over-engineering:**

- Validation, error handling, and guards that protect against invalid state.
- Tests, explicit exceptions, or invariants required for correctness.
- Stable public APIs or shared abstractions with multiple active callers.
- Reuse that is already paying for itself (multiple real callers, not hypothetical).

______________________________________________________________________

### 4. Present findings

Format output as a numbered list of simplification opportunities. Only include items where you found a concrete
violation — do not pad with generic advice.

```
## Simplification opportunities

### 1. <title>
- **Location**: `file/path.py:line`
- **Pattern**: <which check from step 3 it violates>
- **Current**: <what the code does now>
- **Simpler**: <the concrete simpler alternative>

### 2. ...
```

If no issues are found, say so: "No over-engineering found. The code looks appropriately simple."

After the list:

- If reviewing a **plan**, call out where the plan is over-generalized and suggest narrower implementation steps.
- If reviewing **code**, offer to apply the simplifications: "Want me to apply any of these? e.g. 'fix 1, 3, 5'"

______________________________________________________________________

### 5. GiGL-specific patterns to watch for

These are real patterns from this codebase. Use them as reference when scanning.

Use `Uri.get_basename()` instead of manual string splitting:

```python
# Avoid — reimplements Uri.get_basename()
main_jar_file_name = main_jar_file_uri.uri.split("/")[-1]

# Prefer
main_jar_file_name = main_jar_file_uri.get_basename()
```

Use `@retry()` instead of hand-rolled retry loops:

```python
# Avoid — reimplements gigl.common.utils.retry.retry
for attempt in range(max_retries):
    try:
        result = flaky_api_call()
        break
    except ApiError:
        time.sleep(2 ** attempt)

# Prefer
from gigl.common.utils.retry import retry

@retry(exception_to_check=ApiError, tries=5, delay_s=1, backoff=2)
def flaky_api_call() -> Result:
    ...
```

Use `Logger()` instead of `logging.getLogger()`:

```python
# Avoid — bypasses GCP-aware logging
import logging
logger = logging.getLogger(__name__)

# Prefer
from gigl.common.logger import Logger
logger = Logger()
```

Prefer separate entry points over flag-heavy functions:

```python
# Avoid
def build_dataset(mode: str, *, use_remote: bool, use_cache: bool) -> DistDataset:
    if use_remote:
        ...
    elif use_cache:
        ...
    else:
        ...

# Prefer
def build_local_dataset() -> DistDataset:
    ...

def build_remote_dataset() -> DistDataset:
    ...
```

Prefer lightweight types over one-off dataclasses:

```python
# Avoid — one-off type used in a single function
@dataclass(frozen=True)
class BatchBounds:
    start_index: int
    end_index: int

# Prefer
batch_start, batch_end = 0, 128
```
