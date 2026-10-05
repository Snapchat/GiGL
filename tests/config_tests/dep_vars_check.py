import json
import re
import tomllib
from pathlib import Path

import yaml

# We're in GiGL/tests/config_tests, so we need to go up two levels to find GiGL/gigl/dep_vars.env
REPO_ROOT = Path(__file__).parent.parent.parent
DEP_VARS_FILE_PATH = Path.joinpath(REPO_ROOT, "gigl", "dep_vars.env")


def check_ci_python_matrices() -> None:
    """CI matrices must list every classifier minor, every one but the .python-version minor, or only it."""
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
        ("ci-unit-test-python-matrix", supported),
        ("ci-integration-e2e-test-nondefault-python-matrix", non_default),
    ]:
        matrix = jobs[job_name]["strategy"]["matrix"]
        actual = matrix.get("python") or [leg["python"] for leg in matrix["include"]]
        assert actual == expected, f"{job_name} matrix is {actual}, expected {expected}"
    # integration-e2e-test picks its legs from the comment: `/e2e_test all` runs every minor,
    # `/e2e_test` only the default one.
    include = jobs["integration-e2e-test"]["strategy"]["matrix"]["include"]
    assert include.startswith(
        "${{ fromJSON(contains(github.event.comment.body, '/e2e_test all') &&"
    ), include
    all_legs, default_legs = [
        json.loads(legs) for legs in re.findall(r"'(\[.*?\])'", include)
    ]
    assert [leg["python"] for leg in all_legs] == supported, (
        f"`/e2e_test all` runs {all_legs}, expected {supported}"
    )
    assert [leg["python"] for leg in default_legs] == [default], (
        f"`/e2e_test` runs {default_legs}, expected {[default]}"
    )
    # The default minor runs every pipeline; the other minors run only the in-memory (GLT) ones.
    for leg in all_legs + default_legs:
        expected_target = (
            "run_all_e2e_tests" if leg["python"] == default else "run_glt_e2e_tests"
        )
        assert leg["target"] == expected_target, leg


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
