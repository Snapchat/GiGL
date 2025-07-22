from pathlib import Path
from typing import Final

# TODO: (svij) https://github.com/Snapchat/GiGL/issues/125
# common -> gigl -> python (or root dir in Docker container)
python_or_gigl_dir: Final[Path] = Path(__file__).resolve().parent.parent.parent
GIGL_ROOT_DIR: Final[Path] = (
    python_or_gigl_dir
    if (python_or_gigl_dir / "examples").exists()
    else python_or_gigl_dir.parent  # common -> gigl -> python -> root
)
PYTHON_ROOT_DIR: Final[Path] = python_or_gigl_dir

PATH_GIGL_PKG_INIT_FILE: Final[Path] = Path.joinpath(
    GIGL_ROOT_DIR, "python", "gigl", "__init__.py"
)
PATH_BASE_IMAGES_VARIABLE_FILE: Final[Path] = Path.joinpath(
    GIGL_ROOT_DIR, "dep_vars.env"
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


_make_file_vars: dict[str, str] = parse_makefile_vars(PATH_BASE_IMAGES_VARIABLE_FILE)

DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG: Final[str] = _make_file_vars[
    "DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG"
]
DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG: Final[str] = _make_file_vars[
    "DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG"
]
DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG: Final[str] = _make_file_vars[
    "DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG"
]
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
DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA: Final[str] = _make_file_vars[
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA"
]
DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU: Final[str] = _make_file_vars[
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU"
]
DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU: Final[str] = _make_file_vars[
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU"
]
DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE: Final[str] = _make_file_vars[
    "DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE"
]
# ===============================================================================================
