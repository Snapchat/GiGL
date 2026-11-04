"""Prints the e2e test matrix for a PR comment, as a ``matrix=<JSON>`` line for ``$GITHUB_OUTPUT``.

The comment body comes from the ``COMMENT_BODY`` environment variable, never from a command line,
so comment text is never interpolated into a shell. The default minor (``.python-version``) always
runs every pipeline (``run_all_e2e_tests``). ``/e2e_test all`` (exactly one space, any case,
not followed by more word characters, so `/e2e_test allow` does not match) adds
every other supported minor with the in-memory pipelines only (``run_glt_e2e_tests``).

Examples, with versions 3.11.14,3.12.14,3.13.15 and default 3.11:
    $ COMMENT_BODY='/e2e_test' python .github/scripts/e2e_matrix.py
    matrix=[{"python":"3.11","target":"run_all_e2e_tests"}]
    $ COMMENT_BODY='/all_test' python .github/scripts/e2e_matrix.py
    matrix=[{"python":"3.11","target":"run_all_e2e_tests"}]
    $ COMMENT_BODY='/e2e_test all' python .github/scripts/e2e_matrix.py
    matrix=[{"python":"3.11","target":"run_all_e2e_tests"},{"python":"3.12","target":"run_glt_e2e_tests"},{"python":"3.13","target":"run_glt_e2e_tests"}]
"""

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Optional

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parents[1]

# Loaded by path because `.github/scripts` is not a package.
_spec = importlib.util.spec_from_file_location(
    "supported_python_versions", _SCRIPTS_DIR / "supported_python_versions.py"
)
assert _spec is not None and _spec.loader is not None
_versions = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_versions)


def matrix(repo_root: Path, comment_body: str) -> list[dict[str, str]]:
    """Returns the ``{"python", "target"}`` legs to run for ``comment_body``."""
    versions = _versions.read_supported_versions(repo_root)
    default_minor = _versions.read_default_minor(repo_root, versions)
    legs = [{"python": default_minor, "target": "run_all_e2e_tests"}]
    if re.search(r"/e2e_test all\b", comment_body, re.IGNORECASE):
        legs += [
            {"python": _versions.minor_of(version), "target": "run_glt_e2e_tests"}
            for version in versions
            if _versions.minor_of(version) != default_minor
        ]
    return legs


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)
    legs = matrix(args.repo_root, os.environ.get("COMMENT_BODY", ""))
    print("matrix=" + json.dumps(legs, separators=(",", ":")))


if __name__ == "__main__":
    main()
