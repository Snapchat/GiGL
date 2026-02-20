# General Development Guidance

## Git

Use one branch for each feature, feel free to create multiple branches (and PRs) if your task has multiple independent
features.

Each PR should be small.

Branch names should be of the form: `{USER}/snake-case-feature`. Use the following to determine {USER}:

1. The `GIGL_ALERT_EMAILS` environment variable, do not use the full email just the user name.
2. The whoami cli command

## Workflow

We have several formatting tools, see `./formatting.md` for more details. Only use the formatter(s) for the files you
edited. For instance, if you only edited python files, only use `make format_py`.

Similarly, if you only edited `foo.py`, only run tests for `foo_test.py` with
`make unit_test_py PY_TEST_FILES="foo_test.py"`
