# Pre-Submit Checklist

Do not suppress errors with workarounds like `# type: ignore`:

1. `make type_check` — runs **ty** static type checker (config in `pyproject.toml` under `[tool.ty]`)
2. `make unit_test_py PY_TEST_FILES="relevant_test.py"`
3. `make integration_test PY_TEST_FILES="relevant_test.py"` (if cross-component behavior changed)
4. `make check_format` (or `make format` to auto-fix)

# Formatting Details

- **autoflake**: Removes unused imports. Excludes `*_pb2.py*` and `__init__.py`.
- **isort**: Sorts imports (black profile).
- **black**: Code formatter (line length 88). Excludes `*_pb2.py*`.
- **mdformat**: Markdown formatter (wrap 120, tables extension).

**Note:** `make format` is NOT a pre-commit hook — pre-commit only runs whitespace and EOF fixes. Always run
`make format` (or `make check_format`) manually before submitting.
