import re
import tomllib
from pathlib import Path

# We're in GiGL/tests/config_tests, so we need to go up two levels to find GiGL/gigl/dep_vars.env
REPO_ROOT = Path(__file__).parent.parent.parent
DEP_VARS_FILE_PATH = Path.joinpath(REPO_ROOT, "gigl", "dep_vars.env")


def _minor(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def check_supported_python_versions(repo_root: Path = REPO_ROOT) -> None:
    """`SUPPORTED_PYTHON_VERSIONS` must agree with the classifiers, `requires-python` and `.python-version`."""
    values = dict(
        line.split("=", 1)
        for line in (repo_root / "gigl" / "dep_vars.env").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    versions = values["SUPPORTED_PYTHON_VERSIONS"].split(",")
    minors = sorted({_minor(version) for version in versions})
    minor_strings = [
        f"{major}.{minor}" for major, minor in minors
    ]  # numerically sorted
    # 3.11.14,3.12.14,3.13.15 -> ">=3.11,<3.14"
    expected_requires = f">={minor_strings[0]},<{minors[-1][0]}.{minors[-1][1] + 1}"

    for pyproject_path in ("pyproject.toml", "gigl-core/pyproject.toml"):
        project = tomllib.loads((repo_root / pyproject_path).read_text())["project"]
        classifier_minors = sorted(
            (
                match.group(1)
                for classifier in project["classifiers"]
                if (
                    match := re.fullmatch(
                        r"Programming Language :: Python :: (3\.\d+)", classifier
                    )
                )
            ),
            key=_minor,
        )
        assert classifier_minors == minor_strings, (
            f"{pyproject_path} classifiers {classifier_minors} != SUPPORTED_PYTHON_VERSIONS minors {minor_strings}"
        )
        assert project["requires-python"] == expected_requires, (
            f"{pyproject_path} requires-python {project['requires-python']} != {expected_requires}"
        )

    default = (repo_root / ".python-version").read_text().strip()
    assert default in versions, (
        f".python-version {default} is not in SUPPORTED_PYTHON_VERSIONS {versions}"
    )


if __name__ == "__main__":
    assert DEP_VARS_FILE_PATH.exists(), (
        f"File `gigl/dep_vars.env` not found at: {DEP_VARS_FILE_PATH}"
    )
    with open(file=DEP_VARS_FILE_PATH, mode="r") as f:
        # Ensure we only have comments, empty lines, or lines with variable definitions
        for line in f.readlines():
            if line.startswith("#") or not line.strip():  # Is line a comment or empty?
                continue
            if (
                "=" not in line or ":=" in line
            ):  # := dictates runtime evaluation of the variable; = is static
                raise ValueError(
                    f"Invalid line found in `gigl/dep_vars.env`: {line}. Expected format: var=value"
                )
    check_supported_python_versions()
