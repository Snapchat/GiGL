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
