import re
from pathlib import Path

# We're in GiGL/tests/config_tests, so we need to go up two levels to find GiGL/gigl/dep_vars.env
_REPO_ROOT = Path(__file__).parent.parent.parent
DEP_VARS_FILE_PATH = Path.joinpath(_REPO_ROOT, "gigl", "dep_vars.env")
GLT_PIN_CMAKE_PATH = Path.joinpath(_REPO_ROOT, "gigl-core", "GLT_PIN.cmake")

# Matches `set(GIGL_GLT_COMMIT "<sha>")`, tolerating arbitrary internal whitespace.
_GIGL_GLT_COMMIT_RE = re.compile(
    r"""^\s*set\s*\(\s*GIGL_GLT_COMMIT\s+"(?P<sha>[0-9a-f]{40})"\s*\)""",
    re.MULTILINE,
)


def parse_dep_vars(dep_vars_text: str) -> dict[str, str]:
    """Parses `gigl/dep_vars.env` into a mapping, validating its static key=value format.

    The file is sourced by make, bash, python, and sbt without any additional parsing, so only
    comments, blank lines, and static ``key=value`` assignments are permitted.

    Args:
        dep_vars_text: Full contents of `gigl/dep_vars.env`.

    Returns:
        Mapping of variable name to value.

    Raises:
        ValueError: If a line is neither a comment, blank, nor a static `key=value` assignment.
    """
    dep_vars: dict[str, str] = {}
    for line in dep_vars_text.splitlines():
        if line.startswith("#") or not line.strip():  # Is line a comment or empty?
            continue
        if (
            "=" not in line or ":=" in line
        ):  # := dictates runtime evaluation of the variable; = is static
            raise ValueError(
                f"Invalid line found in `gigl/dep_vars.env`: {line}. Expected format: var=value"
            )
        key, _, value = line.partition("=")
        dep_vars[key.strip()] = value.strip()
    return dep_vars


def parse_gigl_glt_commit(glt_pin_cmake_text: str) -> str:
    """Extracts GIGL_GLT_COMMIT from `gigl-core/GLT_PIN.cmake`.

    Args:
        glt_pin_cmake_text: Full contents of `gigl-core/GLT_PIN.cmake`.

    Returns:
        The pinned 40-character GLT commit SHA.

    Raises:
        ValueError: If no `set(GIGL_GLT_COMMIT "<40-hex-sha>")` line is present.
    """
    match = _GIGL_GLT_COMMIT_RE.search(glt_pin_cmake_text)
    if match is None:
        raise ValueError(
            "Could not find `set(GIGL_GLT_COMMIT \"<40-hex-sha>\")` in "
            f"{GLT_PIN_CMAKE_PATH}. The GLT pin must stay machine-readable so it can be checked "
            "against GLT_COMMIT_SHA in gigl/dep_vars.env."
        )
    return match.group("sha")


if __name__ == "__main__":
    assert DEP_VARS_FILE_PATH.exists(), (
        f"File `gigl/dep_vars.env` not found at: {DEP_VARS_FILE_PATH}"
    )
    assert GLT_PIN_CMAKE_PATH.exists(), (
        f"File `gigl-core/GLT_PIN.cmake` not found at: {GLT_PIN_CMAKE_PATH}"
    )

    dep_vars = parse_dep_vars(DEP_VARS_FILE_PATH.read_text())
    installer_glt_commit = dep_vars["GLT_COMMIT_SHA"]
    cpp_build_glt_commit = parse_gigl_glt_commit(GLT_PIN_CMAKE_PATH.read_text())

    # gigl-core compiles against GLT's C++ headers, and reads the private counters of
    # graphlearn_torch::ShmQueueMeta out of a shared-memory segment created by the installed GLT
    # wheel. If the two pins diverge, the compiled field offsets stop matching the running layout and
    # queue-size readings are silently wrong -- there is no compile error and no import error. This
    # assertion is the guard, and it runs in `make precondition_tests`, before anything is installed.
    if installer_glt_commit != cpp_build_glt_commit:
        raise ValueError(
            "GLT commit pins disagree.\n"
            f"  gigl/dep_vars.env        GLT_COMMIT_SHA  = {installer_glt_commit}\n"
            f"  gigl-core/GLT_PIN.cmake  GIGL_GLT_COMMIT = {cpp_build_glt_commit}\n"
            "These must be identical: dep_vars.env drives the GLT wheel build in "
            "gigl/scripts/install_glt.sh, while GLT_PIN.cmake drives the headers gigl-core compiles "
            "against. See the bump instructions at the top of gigl-core/GLT_PIN.cmake -- note that "
            "GIGL_GLT_SHM_QUEUE_H_SHA256 must be recomputed too."
        )
