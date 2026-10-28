import re
import tomllib
from pathlib import Path

import yaml

# We're in GiGL/tests/config_tests, so we need to go up two levels to find GiGL/gigl/dep_vars.env
REPO_ROOT = Path(__file__).parent.parent.parent
DEP_VARS_FILE_PATH = Path.joinpath(REPO_ROOT, "gigl", "dep_vars.env")


def _minor(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def _read_dep_vars(repo_root: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1)
        for line in (repo_root / "gigl" / "dep_vars.env").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )


def check_supported_python_versions(repo_root: Path = REPO_ROOT) -> None:
    """`SUPPORTED_PYTHON_VERSIONS` must agree with the classifiers, `requires-python` and `.python-version`."""
    values = _read_dep_vars(repo_root)
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


# Key family -> the suffix its value must end with after the `-py<XY>` tag.
_PER_MINOR_KEY_FAMILIES = {
    "DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG": "",
    "DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG": "",
    "DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG": "",
    "DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG": "",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA": "",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU": "",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU": "",
    "DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH": ".yaml",
}


def check_per_minor_keys(repo_root: Path = REPO_ROOT) -> None:
    """Every supported minor has each per-minor key, ending in its `-py<XY>` tag, and no other minor has keys.

    Also checks that the Cloud Build `_BUILDER_IMAGE` default is the `.python-version` minor's builder.
    """
    values = _read_dep_vars(repo_root)
    tags = [
        "py" + "".join(version.split(".")[:2])
        for version in values["SUPPORTED_PYTHON_VERSIONS"].split(",")
    ]
    expected_keys = set()
    for tag in tags:
        for family, extension in _PER_MINOR_KEY_FAMILIES.items():
            key = f"{family}_{tag}"
            expected_keys.add(key)
            assert key in values, f"gigl/dep_vars.env is missing {key}"
            assert values[key].endswith(f"-{tag}{extension}"), (
                f"{key}={values[key]} does not end in -{tag}{extension}"
            )
    stray_keys = {key for key in values if re.search(r"_py\d+$", key)} - expected_keys
    assert not stray_keys, (
        f"gigl/dep_vars.env has per-minor keys for minors not in SUPPORTED_PYTHON_VERSIONS: {sorted(stray_keys)}"
    )

    default_tag = "py" + "".join(
        (repo_root / ".python-version").read_text().strip().split(".")[:2]
    )
    cloud_build = (
        repo_root / ".github" / "cloud_builder" / "run_command_on_active_checkout.yaml"
    ).read_text()
    match = re.search(r'^\s*_BUILDER_IMAGE: "(.*)"$', cloud_build, re.MULTILINE)
    assert match, "run_command_on_active_checkout.yaml has no _BUILDER_IMAGE default"
    expected_builder = values[
        f"DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG_{default_tag}"
    ]
    assert match.group(1) == expected_builder, (
        f"Cloud Build _BUILDER_IMAGE {match.group(1)} != {expected_builder}"
    )


def _needs(job: dict) -> list[str]:
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else needs


def _run_text(job: dict) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _check_matrix(job_name: str, job: dict, key: str, expected: str, need: str) -> None:
    matrix = job["strategy"]["matrix"][key]
    assert matrix == expected, (
        f"{job_name} matrix {key} is {matrix}, expected {expected}"
    )
    assert need in _needs(job), f"{job_name} needs {job.get('needs')}, expected {need}"


def check_ci_python_matrices() -> None:
    """CI matrices must read the supported minors from the `python-versions` / `e2e-plan` jobs."""
    workflows = REPO_ROOT / ".github" / "workflows"
    comment = yaml.safe_load((workflows / "on-pr-comment.yml").read_text())["jobs"]
    merge = yaml.safe_load((workflows / "on-pr-merge.yml").read_text())["jobs"]

    assert "supported_python_versions.py" in _run_text(comment["python-versions"])
    assert "e2e_matrix.py" in _run_text(comment["e2e-plan"])
    minors = "${{ fromJSON(needs.python-versions.outputs.minors) }}"
    _check_matrix(
        "unit-test-python",
        comment["unit-test-python"],
        "python",
        minors,
        "python-versions",
    )
    _check_matrix(
        "integration-e2e-test",
        comment["integration-e2e-test"],
        "include",
        "${{ fromJSON(needs.e2e-plan.outputs.matrix) }}",
        "e2e-plan",
    )

    assert "supported_python_versions.py" in _run_text(merge["python-versions"])
    _check_matrix(
        "ci-unit-test-python-matrix",
        merge["ci-unit-test-python-matrix"],
        "python",
        minors,
        "python-versions",
    )
    _check_matrix(
        "ci-integration-e2e-test-nondefault-python-matrix",
        merge["ci-integration-e2e-test-nondefault-python-matrix"],
        "python",
        "${{ fromJSON(needs.python-versions.outputs.nondefault_minors) }}",
        "python-versions",
    )
    for gate, matrix_job in [
        ("ci-unit-test-python", "ci-unit-test-python-matrix"),
        (
            "ci-integration-e2e-test-nondefault-python",
            "ci-integration-e2e-test-nondefault-python-matrix",
        ),
    ]:
        assert matrix_job in _needs(merge[gate]), f"{gate} must need {matrix_job}"
        assert "always()" in merge[gate]["if"], f"{gate} `if` must use always()"


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
    check_per_minor_keys()

    check_ci_python_matrices()
