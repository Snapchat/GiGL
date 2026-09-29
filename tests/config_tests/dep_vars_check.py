import tomllib
from pathlib import Path

import yaml

# We're in GiGL/tests/config_tests, so we need to go up two levels to find GiGL/gigl/dep_vars.env
REPO_ROOT = Path(__file__).parent.parent.parent
DEP_VARS_FILE_PATH = Path.joinpath(REPO_ROOT, "gigl", "dep_vars.env")


def check_ci_python_matrices() -> None:
    """CI matrices must list every classifier minor, or every one but the .python-version minor."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    supported = [
        c.split(" :: ")[-1]
        for c in pyproject["project"]["classifiers"]
        if c.startswith("Programming Language :: Python :: 3.")
    ]
    default = ".".join((REPO_ROOT / ".python-version").read_text().split(".")[:2])
    non_default = [v for v in supported if v != default]
    jobs = {}
    for workflow in ["on-pr-comment.yml", "on-pr-merge.yml"]:
        path = REPO_ROOT / ".github" / "workflows" / workflow
        jobs.update(yaml.safe_load(path.read_text())["jobs"])
    for job_name, expected in [
        ("unit-test-python", supported),
        ("ci-unit-test-python", supported),
        ("integration-e2e-test-nondefault-python", non_default),
        ("ci-integration-e2e-test-nondefault-python", non_default),
    ]:
        matrix = jobs[job_name]["strategy"]["matrix"]
        actual = matrix.get("python") or [leg["python"] for leg in matrix["include"]]
        assert actual == expected, f"{job_name} matrix is {actual}, expected {expected}"


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
    check_ci_python_matrices()
