# GiGL Tensor Shape Contracts

This context defines the vocabulary for GiGL's developer-facing tensor shape contracts at high-value graph learning
boundaries.

## Language

**Shape Contract**: A runtime-checkable declaration of a tensor's dtype and named dimensions at a GiGL boundary.
_Avoid_: tensor validation, shape check

**Contract Boundary**: A loader, sampler, or model interface where a Shape Contract is declared to detect malformed
tensors before they reach downstream computation. _Avoid_: internal tensor operation, blanket annotation

**Runtime Contract Checking**: Execution of Shape Contracts during automated tests or an explicit debugging session. It
is disabled in normal production execution. _Avoid_: production validation, always-on checking

## Annotation Conventions

- Reuse a named axis across parameters and returns when their sizes must match. For example,
  `Float[Tensor, "queries candidates"]` and `Int[Tensor, " candidates"]` bind the candidate counts together.
- Prefix a shape containing one named axis with a space, such as `Int[Tensor, " anchors"]`. Jaxtyping ignores leading
  whitespace. It prevents Python lint and type tools from treating the otherwise valid identifier `anchors` as a
  forward type reference. This is intentional and applies consistently to single-axis contracts such as `nodes`,
  `candidates`, and `positives`; multi-axis strings are not valid identifiers and do not need it.
- Prefix an axis name with an underscore when values in a container may legitimately differ in that dimension. For
  example, node types may use `Float[Tensor, "_nodes embedding_dim"]` because each type has its own node count.
- Use `Float[Tensor, "..."]` only when PyTorch broadcasting makes tensor rank intentionally variable.

Authors do not need to know numeric dimension sizes. A named axis such as `nodes` binds whatever size arrives at runtime.
Use a descriptive anonymous axis such as `_nodes` when its size must not bind to the same-looking axis elsewhere, or `_`
when even the axis meaning is unknown. If rank is also unknown, use `...` only when variable rank is part of the
supported API. Otherwise, inspect the producer and consumers and establish the boundary's actual contract before
annotating it. A Shape Contract declares supported behavior; it does not infer behavior from the tensor it receives.

Shape Contracts cover stable loader and sampler tensors, exported model `forward` and `decode` methods, and loss
interfaces. Dynamic PyG and TorchRec keyed containers and low-level message-passing operations remain described by their
native types because their tensor shapes depend on graph structure rather than a single stable API contract.

## Runtime Enforcement

GiGL's unit, integration, and end-to-end test launchers call `install_runtime_typechecking()` before test discovery. The
import hook instruments annotated functions and dataclass constructors in `_SHAPE_CONTRACT_MODULES` when those modules
are subsequently imported. On each executed call, Jaxtyping and Beartype check tensor type, dtype, rank, fixed axes,
and equality between repeated named axes. Arguments are checked before the function body; return values are checked
after the body returns.

A violation raises `jaxtyping.TypeCheckError` synchronously. If a test does not catch it, the test is reported as an
error and the test command exits nonzero. Only executed calls are checked. A function that a test never calls receives
no runtime coverage.

Runtime checking is not enabled by importing GiGL. It is disabled in production and in direct `pytest` or `unittest`
commands that bypass GiGL's test launchers. For an explicit debugging session, call `install_runtime_typechecking()`
before importing the GiGL modules under investigation.
