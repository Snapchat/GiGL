import importlib.util
import re
import tomllib
from pathlib import Path

# We're in GiGL/tests/config_tests, so we need to go up two levels to find GiGL/gigl/dep_vars.env
REPO_ROOT = Path(__file__).parent.parent.parent
DEP_VARS_FILE_PATH = Path.joinpath(REPO_ROOT, "gigl", "dep_vars.env")


def _load_supported_python_versions_script():
    script = REPO_ROOT / ".github" / "scripts" / "supported_python_versions.py"
    spec = importlib.util.spec_from_file_location("supported_python_versions", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_supported_python_versions(repo_root: Path = REPO_ROOT) -> None:
    """`SUPPORTED_PYTHON_VERSIONS` must agree with the classifiers and `requires-python`.

    Parsing the list and the `.python-version` rule come from `.github/scripts/supported_python_versions.py`.
    """
    script = _load_supported_python_versions_script()
    versions = script.read_supported_versions(repo_root)
    script.read_default_minor(repo_root, versions)
    minor_strings = [script.minor_of(version) for version in versions]
    # 3.11.14,3.12.14,3.13.15 -> ">=3.11,<3.14"
    last_major, last_minor = minor_strings[-1].split(".")
    expected_requires = f">={minor_strings[0]},<{last_major}.{int(last_minor) + 1}"

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
            key=lambda minor: tuple(int(part) for part in minor.split(".")),
        )
        assert classifier_minors == minor_strings, (
            f"{pyproject_path} classifiers {classifier_minors} != SUPPORTED_PYTHON_VERSIONS minors {minor_strings}"
        )
        assert project["requires-python"] == expected_requires, (
            f"{pyproject_path} requires-python {project['requires-python']} != {expected_requires}"
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
