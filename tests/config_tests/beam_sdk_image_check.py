"""Fails if the Dataflow base image's Beam SDK reference drifts from the repo's pins.

The ``BEAM_SDK_IMAGE`` default in ``containers/Dockerfile.dataflow.base`` supplies the worker
boot harness, and it has to agree with two other files:
  - its tag, e.g. ``2.56.0``, equals the ``apache-beam`` pin in ``pyproject.toml``;
  - the Python minor in its name, e.g. ``beam_python3.11_sdk``, equals ``.python-version``.
"""

import re
import tomllib
from pathlib import Path
from typing import Final

# We're in GiGL/tests/config_tests, so we need to go up two levels to find the repo root
REPO_ROOT: Final = Path(__file__).parent.parent.parent
BEAM_SDK_IMAGE_PATTERN: Final = re.compile(
    r"^ARG BEAM_SDK_IMAGE=apache/beam_python(?P<minor>\d+\.\d+)_sdk:(?P<tag>[^@\s]+)@sha256:",
    re.MULTILINE,
)
APACHE_BEAM_PIN_PATTERN: Final = re.compile(
    r"^apache-beam(?:\[[^\]]*\])?==(?P<version>.+)$"
)

if __name__ == "__main__":
    dockerfile = (REPO_ROOT / "containers" / "Dockerfile.dataflow.base").read_text()
    image = BEAM_SDK_IMAGE_PATTERN.search(dockerfile)
    assert image, (
        "No `ARG BEAM_SDK_IMAGE=apache/beam_pythonX.Y_sdk:<tag>@sha256:...` default found"
    )

    transform_deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())[
        "project"
    ]["optional-dependencies"]["transform"]
    pins = [
        match["version"]
        for dep in transform_deps
        if (match := APACHE_BEAM_PIN_PATTERN.match(dep))
    ]
    assert len(pins) == 1, (
        f"Expected one `apache-beam==` pin in pyproject.toml, got {pins}"
    )

    python_minor = ".".join(
        (REPO_ROOT / ".python-version").read_text().strip().split(".")[:2]
    )
    assert image["tag"] == pins[0], (
        f"BEAM_SDK_IMAGE tag {image['tag']} != apache-beam pin {pins[0]} in pyproject.toml"
    )
    assert image["minor"] == python_minor, (
        f"BEAM_SDK_IMAGE Python minor {image['minor']} != .python-version minor {python_minor}"
    )
    print(f"BEAM_SDK_IMAGE matches: apache-beam {pins[0]}, Python {python_minor}")
