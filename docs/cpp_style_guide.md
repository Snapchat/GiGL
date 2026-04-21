# C++ Style Guide

GiGL enforces C++ style automatically via two tools:

- **clang-format** (`.clang-format`) — code formatting
- **clang-tidy** (`.clang-tidy`) — static analysis and lint

All clang-tidy warnings are treated as errors.

## Running the Tools

```bash
make format_cpp  # Format all C++ files in-place
make check_lint_cpp  # Run clang-tidy static analysis
```

______________________________________________________________________

## Formatting (`.clang-format`)

The style is based on LLVM with the following notable deviations:

### Line length

```
ColumnLimit: 120
```

120 columns rather than the LLVM default of 80. ML and graph code tends to have longer identifiers and nested template
types; 120 gives enough room without forcing awkward wraps.

### Indentation and braces

```
IndentWidth: 4
BreakBeforeBraces: Attach   # K&R / "same-line" style
UseTab: Never
IndentCaseLabels: true      # case labels indented inside switch
NamespaceIndentation: None  # namespace bodies not indented
```

### Pointer and reference alignment

```
PointerAlignment: Left
```

Pointers bind to the type, not the name: `int* x`, not `int *x`.

### Parameter and argument wrapping

```
BinPackArguments: false
BinPackParameters: false
```

When a function call or declaration doesn't fit on one line, every argument/parameter gets its own line. Mixed
"bin-packing" (some on one line, some wrapped) is not allowed.

### Templates

```
AlwaysBreakTemplateDeclarations: true
```

`template <...>` always appears on its own line, keeping the declaration signature visually separate from the template
header.

### Include ordering

Includes are sorted and split into three priority groups:

| Priority | Pattern                              | Group                                 |
| -------- | ------------------------------------ | ------------------------------------- |
| 1        | `.*`                                 | Local project headers (first)         |
| 2        | `^"(llvm\|llvm-c\|clang\|clang-c)/"` | LLVM/Clang internal headers           |
| 3        | `^(<\|"(gtest\|isl\|json)/)`         | System and third-party headers (last) |

### Raw string formatting

Raw string literals with the `pb` delimiter (e.g. `R"pb(...)pb"`) are formatted as TextProto using Google style,
matching the protobuf idiom used throughout the codebase.

______________________________________________________________________

## Static Analysis (`.clang-tidy`)

### Check philosophy

A broad set of check families is enabled to catch bugs, enforce modern C++ idioms, and maintain readability. All
warnings are errors — there is no "warning-only" category.

Enabled families:

| Family                      | What it covers                                                                                                                         |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `boost-use-to-string`       | Prefer `std::to_string` over `boost::lexical_cast` for numeric conversions                                                             |
| `bugprone-*`                | Common programming mistakes: dangling handles, suspicious string construction, assert side effects, etc.                               |
| `cert-*`                    | CERT secure coding rules for error handling (`err34-c`), floating-point loops (`flp30-c`), and RNG seeding (`msc32-c`, `msc50/51-cpp`) |
| `clang-diagnostic-*`        | Compiler diagnostic warnings surfaced as lint checks (e.g. `-Wall`, `-Wextra` violations)                                              |
| `cppcoreguidelines-*`       | C++ Core Guidelines: no raw `malloc`, no union member access, no object slicing, safe downcasts                                        |
| `google-*`                  | Google C++ style: explicit constructors, no global names in headers, safe `memset` usage                                               |
| `hicpp-exception-baseclass` | All thrown exceptions must derive from `std::exception`                                                                                |
| `misc-*`                    | Miscellaneous: header-only definitions, suspicious enum usage, throw-by-value/catch-by-reference, etc.                                 |
| `modernize-*`               | Modernize to C++11/14/17: `nullptr`, range-based for, `make_unique`, `using` aliases, etc.                                             |
| `performance-*`             | Unnecessary copies, inefficient string ops, missed `emplace`, type promotions in math functions                                        |
| `readability-*`             | Naming conventions, braces around statements, boolean simplification, function size limits                                             |

### Disabled checks

Some checks in the above families are disabled where they produce excessive noise or conflict with common patterns in
this codebase:

| Disabled check                                        | Reason                                                                                                                                                                                                                                                 |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bugprone-easily-swappable-parameters`                | Tensor and sampler APIs legitimately have many adjacent same-typed parameters                                                                                                                                                                          |
| `bugprone-implicit-widening-of-multiplication-result` | Crashes clang-tidy 15 on a construct in `ATen/core/dynamic_type.h` (upstream LLVM bug). Re-enable when upgrading past clang-tidy 15.                                                                                                                   |
| `bugprone-narrowing-conversions`                      | Too noisy in ML code mixing `int`/`int64_t`/`size_t` for tensor dimensions                                                                                                                                                                             |
| `misc-confusable-identifiers`                         | Performs an O(n²) comparison of all identifiers in scope to detect Unicode homoglyphs. PyTorch headers introduce thousands of identifiers, making this check account for ~70% of total lint time. All identifiers in this codebase are standard ASCII. |
| `misc-const-correctness`                              | Produces false positives with pybind11 types whose mutation happens through `operator[]` (which is non-const). The check incorrectly suggests `const` on variables that are mutated.                                                                   |
| `misc-no-recursion`                                   | Recursive graph algorithms are intentional                                                                                                                                                                                                             |
| `modernize-avoid-c-arrays`                            | C arrays are needed for pybind11 and C-interop code                                                                                                                                                                                                    |
| `modernize-use-trailing-return-type`                  | Trailing return types (`auto f() -> T`) are only useful when the return type depends on template params. Requiring them everywhere is non-standard and reduces readability.                                                                            |
| `readability-avoid-const-params-in-decls`             | Incorrectly fires on `const T&` parameters in multi-line declarations (clang-tidy 15 bug). The check is meant for top-level const on by-value params, which is a separate, valid concern.                                                              |
| `readability-container-contains`                      | `.contains()` requires C++20; the codebase builds with C++17                                                                                                                                                                                           |
| `readability-identifier-length`                       | Short loop variables (`i`, `j`, `k`) are idiomatic                                                                                                                                                                                                     |
| `readability-function-cognitive-complexity`           | Algorithmic code often requires nesting that is inherent to the problem structure. Enforcing an arbitrary complexity ceiling discourages clarity and encourages artificial decomposition.                                                              |
| `readability-magic-numbers`                           | Literal constants are common in ML code (e.g. feature dimensions)                                                                                                                                                                                      |

### Naming conventions

Enforced via `readability-identifier-naming`:

| Identifier kind                                           | Convention                  | Example           |
| --------------------------------------------------------- | --------------------------- | ----------------- |
| Classes, enums, unions                                    | `CamelCase`                 | `DistDataset`     |
| Type template parameters                                  | `CamelCase`                 | `NodeType`        |
| Functions, methods                                        | `camelBack`                 | `sampleNeighbors` |
| Variables, parameters, members                            | `camelBack`                 | `numNodes`        |
| Private/protected members                                 | `camelBack` with `_` prefix | `_nodeFeatures`   |
| Constants (`constexpr`, `const` globals, class constants) | `CamelCase` with `k` prefix | `kMaxBatchSize`   |

### Key option tuning

| Option                                                     | Value             | Effect                                                                                                                                                                                |
| ---------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WarningsAsErrors`                                         | `*`               | Every check failure is a hard error in CI                                                                                                                                             |
| `HeaderFilterRegex`                                        | `.*/gigl/csrc/.*` | Scopes checks to our own headers. Using `.*` causes clang-tidy to report warnings from every PyTorch/pybind11 header it parses, flooding output with thousands of third-party issues. |
| `FormatStyle`                                              | `none`            | clang-tidy does not auto-reformat; use clang-format separately                                                                                                                        |
| `bugprone-string-constructor.LargeLengthThreshold`         | `8388608` (8 MB)  | Strings larger than 8 MB from a length argument are flagged                                                                                                                           |
| `modernize-loop-convert.NamingStyle`                       | `CamelCase`       | Auto-generated loop variable names use CamelCase                                                                                                                                      |
| `readability-function-size.LineThreshold`                  | `1000`            | Functions over 1000 lines are flagged                                                                                                                                                 |
| `readability-braces-around-statements.ShortStatementLines` | `0`               | Braces required for all control-flow bodies, even single-line                                                                                                                         |

______________________________________________________________________

## pybind11 Extension Modules

Extension modules live under `gigl/csrc/`.

### Naming convention

| File                       | Purpose                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| `python_<name>.cpp`        | pybind11 bindings — contains the `PYBIND11_MODULE` definition    |
| `<name>.cpp` / `<name>.cu` | Implementation — function and class definitions                  |
| `<name>.h`                 | Declarations (function signatures, class definitions, constants) |

Example: to add a `my_op` extension under `gigl/csrc/sampling/`:

```
gigl/csrc/sampling/python_my_op.cpp   ← pybind11 bindings
gigl/csrc/sampling/my_op.cpp          ← implementation
```

The compiled `.so` is installed to the same directory and importable as `gigl.csrc.sampling.my_op`.
