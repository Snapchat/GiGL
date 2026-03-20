______________________________________________________________________

## description: Apply KISS and pragmatic simplification to the current task, plan, or code. Invoke when the user says kiss, keep it simple, be pragmatic, don't overcomplicate, simplify this, reduce abstraction, avoid overengineering, or asks for the smallest correct change. argument-hint: "[optional focus]"

# Pragmatic

Use the K.I.S.S. guidance from `CLAUDE.md` to simplify the current task, plan, or code without dropping correctness.

## Instructions

When this skill is invoked with `$ARGUMENTS`, execute the following sections in order.

______________________________________________________________________

### 1. Anchor on the concrete need

Identify the current concrete use case before proposing changes.

- Optimize for today's requirement, not hypothetical future reuse.
- If `$ARGUMENTS` names a specific area, use it as the simplification target.
- If the request is broad, infer the narrowest immediate objective from the conversation and code.

______________________________________________________________________

### 2. Inspect existing code before inventing structure

Look for existing utilities, patterns, and nearby implementations before proposing new abstractions.

- Reuse or lightly refactor existing code when it already solves most of the problem.
- Do not introduce new layers just because they look architecturally neat.
- Prefer local edits over cross-cutting reorganization unless the current shape is clearly blocking correctness.

______________________________________________________________________

### 3. Prefer the smallest direct implementation

Bias toward the simplest implementation that cleanly solves the current problem.

- Avoid abstraction for hypothetical extensibility or configuration.
- Keep linear workflows easy to follow from top to bottom.
- Inline tiny one-off helpers when extraction does not materially improve readability, testability, or error handling.
- Prefer explicit parameter passing over stateful helper objects when the workflow is local and sequential.

______________________________________________________________________

### 4. Keep data and control flow lightweight

Reduce indirection in both data shapes and function structure.

- Prefer tuples, dicts, or existing objects over tiny one-off dataclasses, wrappers, or config classes.
- Avoid one-off builders, factories, registries, or strategy objects unless they remove real duplication or support
  multiple active implementations.
- If a function is accumulating mode flags, optional callbacks, or branching setup paths, prefer separate simple entry
  points over one generic function.

______________________________________________________________________

### 5. Preserve correctness while simplifying

Do not simplify away behavior that is required for safe and correct code.

- Keep validation, error handling, and required invariants intact.
- Do not remove tests, guards, or explicit exceptions that protect against invalid state.
- Do not churn stable public APIs or established shared abstractions unless they are part of the actual problem.
- Defer generalization until a second real use case exists, but do not delete real reuse that is already paying for
  itself.

______________________________________________________________________

### 6. Produce a pragmatic answer

When responding to the user, structure the output around the simplest viable path.

- State the recommended simple approach first.
- Name the specific complexity to avoid and why it is unnecessary for the current use case.
- If editing code, make the smallest coherent change set that solves the problem.
- If reviewing a plan, call out where the plan is over-generalized and replace it with a narrower implementation path.

______________________________________________________________________

### 7. Apply these default heuristics

Use these checks when deciding whether to simplify:

- A helper called once or twice and only a few lines long usually should be inlined.
- A short-lived internal type with two small fields usually should be a tuple, dict, or existing object.
- A function with several boolean flags usually should become separate entry points by use case.
- A new abstraction justified only by "we might need this later" usually should not be added.
- If the codebase already has a utility that solves the problem, use it instead of re-implementing it.

______________________________________________________________________

### 8. Keep these examples in mind

Prefer inlining tiny one-off helpers:

```python
# Avoid
def _normalized_node_id(node_id: str) -> str:
    return node_id.strip().lower()


def load_node(node_id: str) -> Node:
    return node_store[_normalized_node_id(node_id)]


# Prefer
def load_node(node_id: str) -> Node:
    normalized_node_id = node_id.strip().lower()
    return node_store[normalized_node_id]
```

Prefer lightweight internal data over tiny one-off types:

```python
# Avoid
@dataclass(frozen=True)
class BatchBounds:
    start_index: int
    end_index: int


batch_bounds = BatchBounds(start_index=0, end_index=128)


# Prefer
batch_bounds: tuple[int, int] = (0, 128)
start_index, end_index = batch_bounds
```

Prefer separate simple entry points over one generic function with flags:

```python
# Avoid
def write_output(records: list[Record], *, validate: bool, compress: bool, upload: bool) -> None:
    ...


# Prefer
def write_local_output(records: list[Record]) -> None:
    ...


def write_uploaded_output(records: list[Record]) -> None:
    ...
```
