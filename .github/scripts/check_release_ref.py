"""Fails unless the workflow was dispatched on the tag ``v<version>`` of the checked-out ``pyproject.toml``.

Reads ``GITHUB_REF_TYPE`` and ``GITHUB_REF_NAME`` from the environment. Stdlib only.

Example:
    $ GITHUB_REF_TYPE=tag GITHUB_REF_NAME=v0.4.0 python .github/scripts/check_release_ref.py
"""

import os
import sys
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def main() -> int:
    ref_type = os.environ["GITHUB_REF_TYPE"]
    ref_name = os.environ["GITHUB_REF_NAME"]
    if ref_type != "tag":
        print(
            f"dispatched on {ref_type} {ref_name}; select the release tag vX.Y.Z as the ref",
            file=sys.stderr,
        )
        return 1
    version = tomllib.loads(_PYPROJECT.read_text())["project"]["version"]
    if f"v{version}" != ref_name:
        print(
            f"pyproject.toml version {version} != tag {ref_name}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
