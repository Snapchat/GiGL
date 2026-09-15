# Pre-Submit Checklist

Do not suppress errors with workarounds like `# type: ignore`:

1. `make type_check` — runs **ty** static type checker (config in `pyproject.toml` under `[tool.ty]`)
2. `make unit_test_py PY_TEST_FILES="relevant_test.py"`
3. `make integration_test PY_TEST_FILES="relevant_test.py"` (if cross-component behavior changed)
4. `make check_format` (or `make format` to auto-fix)

# Formatting Details

- **ruff check**: Removes unused imports (`F401`) and sorts imports (`I`). Excludes `*_pb2.py*` and ignores `F401` in
  `__init__.py`.
- **ruff format**: Code formatter (line length 88, black-compatible). Excludes `*_pb2.py*`.
- **dprint**: Markdown formatter (wrap 120, tables built in; also formats toml/dockerfile/cmake). Configured in
  `dprint.json`.

**Note:** pre-commit formats the staged Python and Markdown files (ruff + dprint) plus whitespace/EOF fixes. It does not
cover Scala or C++, and it only sees staged files, so still run the formatter(s) for what you edited (e.g.
`make format_scala`) and `make check_format` before submitting.
