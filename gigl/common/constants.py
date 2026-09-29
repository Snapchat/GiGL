import sys
from pathlib import Path
from typing import Final

# TODO: (svij) https://github.com/Snapchat/GiGL/issues/125
# common -> gigl -> python (or root dir in Docker container)
GIGL_ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent.parent

PATH_GIGL_PKG_INIT_FILE: Final[Path] = Path.joinpath(
    GIGL_ROOT_DIR, "gigl", "__init__.py"
)
PATH_BASE_IMAGES_VARIABLE_FILE: Final[Path] = Path.joinpath(
    GIGL_ROOT_DIR, "gigl", "dep_vars.env"
).absolute()


def parse_makefile_vars(makefile_path: Path) -> dict[str, str]:
    """
    Parse variables from a Makefile-like file.

    Args:
        makefile_path (Path): The path to the Makefile-like file.

    Returns:
        dict[str, str]: A dictionary containing key-value pairs of variables defined in the file.
    """
    vars_dict: dict[str, str] = {}
    with open(makefile_path, "r") as f:
        for line in f.readlines():
            if line.strip().startswith("#") or not line.strip():
                continue
            if "=" in line:
                key, value = line.split("=")
                vars_dict[key.strip()] = value.strip()
    return vars_dict


def add_python_suffix(
    stem: str,
    python_version: tuple[int, int] = (sys.version_info.major, sys.version_info.minor),
) -> str:
    """
    Name the published artifact for one Python minor from its dep_vars.env stem.

    GiGL publishes every image and KFP pipeline once per supported Python minor, so the
    stems in dep_vars.env name nothing on their own.

    Args:
        stem (str): An image ref or ``.yaml`` pipeline path from dep_vars.env.
        python_version (tuple[int, int]): Python (major, minor). Defaults to the running interpreter.

    Returns:
        str: ``stem`` with ``-py{major}{minor}`` appended, placed before ``.yaml`` for a pipeline path.
            e.g. ``src-cpu:0.3.1`` -> ``src-cpu:0.3.1-py311`` and
            ``gigl-pipeline-0.3.1.yaml`` -> ``gigl-pipeline-0.3.1-py311.yaml``.
    """
    suffix = f"-py{python_version[0]}{python_version[1]}"
    if stem.endswith(".yaml"):
        return stem.removesuffix(".yaml") + suffix + ".yaml"
    return stem + suffix


_make_file_vars: dict[str, str] = parse_makefile_vars(PATH_BASE_IMAGES_VARIABLE_FILE)

DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG: Final[str] = add_python_suffix(
    _make_file_vars["DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG"]
)
DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG: Final[str] = add_python_suffix(
    _make_file_vars["DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG"]
)
DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG: Final[str] = add_python_suffix(
    _make_file_vars["DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG"]
)
SPARK_35_TFRECORD_JAR_GCS_PATH: Final[str] = _make_file_vars[
    "SPARK_35_TFRECORD_JAR_GCS_PATH"
]
SPARK_31_TFRECORD_JAR_GCS_PATH: Final[str] = _make_file_vars[
    "SPARK_31_TFRECORD_JAR_GCS_PATH"
]


# Ensure that the local path is a fully resolved local path
SPARK_35_TFRECORD_JAR_LOCAL_PATH: Final[str] = str(
    Path.joinpath(GIGL_ROOT_DIR, _make_file_vars["SPARK_35_TFRECORD_JAR_LOCAL_PATH"])
)
SPARK_31_TFRECORD_JAR_LOCAL_PATH: Final[str] = str(
    Path.joinpath(GIGL_ROOT_DIR, _make_file_vars["SPARK_31_TFRECORD_JAR_LOCAL_PATH"])
)


# === The src Docker image paths that were released as part of releasing this version of GiGL ===
DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA: Final[str] = add_python_suffix(
    _make_file_vars["DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA"]
)
DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU: Final[str] = add_python_suffix(
    _make_file_vars["DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU"]
)
DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU: Final[str] = add_python_suffix(
    _make_file_vars["DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU"]
)
DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE: Final[str] = add_python_suffix(
    _make_file_vars["DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE"]
)
DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH: Final[str] = add_python_suffix(
    _make_file_vars["DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH"]
)
# ===============================================================================================
