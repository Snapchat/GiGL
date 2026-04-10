# C++ Style Guide

GiGL enforces C++ style automatically via two tools:

- **clang-format** (`.clang-format`) — code formatting
- **clang-tidy** (`.clang-tidy`) — static analysis and lint

Both run as part of CI. All clang-tidy warnings are treated as errors.

## Running the Tools

```bash
# Format all C++ files in-place
clang-format -i $(find gigl/csrc -name '*.cpp' -o -name '*.h')

# Run static analysis
clang-tidy gigl/csrc/**/*.cpp
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

| Family                           | What it covers                                                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `boost-use-to-string`            | Prefer `std::to_string` over `boost::lexical_cast` for numeric conversions                                                             |
| `bugprone-*`                     | Common programming mistakes: dangling handles, suspicious string construction, assert side effects, etc.                               |
| `cert-*` (selected)              | CERT secure coding rules for error handling (`err34-c`), floating-point loops (`flp30-c`), and RNG seeding (`msc32-c`, `msc50/51-cpp`) |
| `clang-analyzer-*`               | Clang static analyzer: memory safety, null dereferences, use-after-free, etc.                                                          |
| `clang-diagnostic-*`             | Compiler diagnostic warnings surfaced as lint checks                                                                                   |
| `cppcoreguidelines-*` (selected) | C++ Core Guidelines: no raw `malloc`, no union member access, no object slicing, safe downcasts                                        |
| `google-*` (selected)            | Google C++ style: explicit constructors, no global names in headers, safe `memset` usage                                               |
| `hicpp-exception-baseclass`      | All thrown exceptions must derive from `std::exception`                                                                                |
| `misc-*`                         | Miscellaneous: header-only definitions, suspicious enum usage, throw-by-value/catch-by-reference, etc.                                 |
| `modernize-*`                    | Modernize to C++11/14/17: `nullptr`, range-based for, `make_unique`, `using` aliases, etc.                                             |
| `performance-*`                  | Unnecessary copies, inefficient string ops, missed `emplace`, type promotions in math functions                                        |
| `readability-*`                  | Naming conventions, braces around statements, boolean simplification, function size limits                                             |

### Disabled checks

Some checks in the above families are disabled where they produce excessive noise or conflict with common patterns in
this codebase:

| Disabled check                            | Reason                                                                              |
| ----------------------------------------- | ----------------------------------------------------------------------------------- |
| `bugprone-easily-swappable-parameters`    | Tensor and sampler APIs legitimately have many adjacent same-typed parameters       |
| `bugprone-narrowing-conversions`          | Too noisy in ML code mixing `int`/`int64_t`/`size_t` for tensor dimensions          |
| `clang-analyzer-alpha*`                   | Alpha checks are experimental and unstable                                          |
| `clang-analyzer-cplusplus.NewDeleteLeaks` | Ownership is managed via smart pointers; raw-new leaks are already caught elsewhere |
| `misc-no-recursion`                       | Recursive graph algorithms are intentional                                          |
| `modernize-avoid-c-arrays`                | C arrays are needed for pybind11 and C-interop code                                 |
| `readability-container-contains`          | `.contains()` requires C++20; the codebase builds with C++17                        |
| `readability-identifier-length`           | Short loop variables (`i`, `j`, `k`) are idiomatic                                  |
| `readability-magic-numbers`               | Literal constants are common in ML code (e.g. feature dimensions)                   |

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

| Option                                                     | Value            | Effect                                                         |
| ---------------------------------------------------------- | ---------------- | -------------------------------------------------------------- |
| `WarningsAsErrors`                                         | `*`              | Every check failure is a hard error in CI                      |
| `HeaderFilterRegex`                                        | `.*`             | Checks apply to all headers, not just implementation files     |
| `FormatStyle`                                              | `none`           | clang-tidy does not auto-reformat; use clang-format separately |
| `bugprone-string-constructor.LargeLengthThreshold`         | `8388608` (8 MB) | Strings larger than 8 MB from a length argument are flagged    |
| `modernize-loop-convert.NamingStyle`                       | `CamelCase`      | Auto-generated loop variable names use CamelCase               |
| `readability-function-size.LineThreshold`                  | `1000`           | Functions over 1000 lines are flagged                          |
| `readability-braces-around-statements.ShortStatementLines` | `2`              | Single-line bodies up to 2 lines may omit braces               |
