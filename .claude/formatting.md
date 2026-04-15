# Pre-Submit Checklist

Do not suppress errors with workarounds like `# type: ignore`:

1. `make type_check`
2. `make unit_test_py PY_TEST_FILES="relevant_test.py"`
3. `make integration_test PY_TEST_FILES="relevant_test.py"` (if cross-component behavior changed)
4. `make check_format` (or `make format` to auto-fix)

# Formatting Details

- **ruff check**: Removes unused imports (`F401`) and sorts imports (`I`). Excludes `*_pb2.py*` and ignores `F401` in
  `__init__.py`.
- **ruff format**: Code formatter (line length 88, black-compatible). Excludes `*_pb2.py*`.
- **mdformat**: Markdown formatter (wrap 120, tables extension).

**Note:** `make format` is NOT a pre-commit hook — pre-commit only runs whitespace and EOF fixes. Always run
`make format` (or `make check_format`) manually before submitting.
