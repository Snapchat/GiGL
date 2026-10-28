"""Prints the Python versions GiGL supports, as ``key=value`` lines for ``$GITHUB_OUTPUT``.

The list is ``SUPPORTED_PYTHON_VERSIONS`` in ``gigl/dep_vars.env``; the default minor is the one in
``.python-version``. Stdlib only, so it runs on a bare runner interpreter.

Example:
    $ python .github/scripts/supported_python_versions.py
    versions=["3.11.14","3.12.14","3.13.15"]
    minors=["3.11","3.12","3.13"]
    default_minor=3.11
    nondefault_minors=["3.12","3.13"]
    builds=[{"python":"3.11.14","tag":"py311"},{"python":"3.12.14","tag":"py312"},{"python":"3.13.15","tag":"py313"}]
    minor_count=3
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KEY = "SUPPORTED_PYTHON_VERSIONS"
_PATCH_VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def minor_of(version: str) -> str:
    """``3.11.14`` -> ``3.11``."""
    return ".".join(version.split(".")[:2])


def read_supported_versions(repo_root: Path) -> list[str]:
    """Returns the validated ``X.Y.Z`` entries of ``SUPPORTED_PYTHON_VERSIONS``."""
    values = {}
    for line in (repo_root / "gigl" / "dep_vars.env").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    if _KEY not in values:
        raise ValueError(f"{_KEY} is not set in gigl/dep_vars.env")
    versions = values[_KEY].split(",")
    for version in versions:
        if not _PATCH_VERSION.match(version):
            raise ValueError(f"{_KEY} entry {version!r} is not of the form X.Y.Z")
    minors = [minor_of(version) for version in versions]
    if len(set(minors)) != len(minors):
        raise ValueError(f"{_KEY} lists a minor more than once: {versions}")
    return versions


def read_default_minor(repo_root: Path, versions: list[str]) -> str:
    """Returns the minor of ``.python-version``, which must be one of ``versions``' minors."""
    default = (repo_root / ".python-version").read_text().strip()
    default_minor = minor_of(default)
    if default_minor not in [minor_of(version) for version in versions]:
        raise ValueError(f".python-version {default} is not in {_KEY}: {versions}")
    return default_minor


def outputs(repo_root: Path) -> dict[str, str]:
    """Returns the ``GITHUB_OUTPUT`` entries for the repo at ``repo_root``."""
    versions = read_supported_versions(repo_root)
    default_minor = read_default_minor(repo_root, versions)
    minors = [minor_of(version) for version in versions]
    builds = [
        {"python": version, "tag": "py" + minor_of(version).replace(".", "")}
        for version in versions
    ]

    def dump(value: object) -> str:
        return json.dumps(value, separators=(",", ":"))

    return {
        "versions": dump(versions),
        "minors": dump(minors),
        "default_minor": default_minor,
        "nondefault_minors": dump([m for m in minors if m != default_minor]),
        "builds": dump(builds),
        "minor_count": str(len(minors)),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)
    for key, value in outputs(args.repo_root).items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
