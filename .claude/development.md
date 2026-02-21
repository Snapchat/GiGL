# General Development Guidance

## Git

Only create new branches if you are on `main` or explicitly asked to.

Branch names should be of the form: `{USER}/snake-case-feature`. Use the following to determine {USER}:

1. The `GIGL_ALERT_EMAILS` environment variable, do not use the full email just the user name.
2. The whoami cli command

If you are asked to break up your work into PRs, then each PR should be as small and self contained as possible,
following general best development practices.

## Workflow

We have several formatting tools, see `./formatting.md` for more details. Only use the formatter(s) for the files you
edited. For instance, if you only edited python files, only use `make format_py`.

Similarly, if you only edited `foo.py`, only run tests for `tests/unit/common/foo_test.py` with
`make unit_test_py PY_TEST_FILES="foo_test.py"`
