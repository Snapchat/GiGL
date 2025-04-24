import importlib.resources as pkg_resources
import os
from pathlib import Path

from gigl.common import LocalUri
from gigl.src.common.types import AppliedTaskIdentifier


def get_gigl_root_directory() -> LocalUri:
    """Returns gigl source root folder."""
    with pkg_resources.path("gigl", "__init__.py") as path:
        root_directory = path.parent
    return LocalUri(root_directory)


def get_python_project_root_path() -> LocalUri:
    """Returns project root folder."""
    file_path = Path(__file__)
    path = LocalUri(file_path.parents[4])
    return path


def get_project_root_directory() -> LocalUri:
    """Returns the parent directory of GiGL"""
    file_path = Path(__file__)
    path = LocalUri(file_path.parents[5])
    return path


def get_gbml_assets_tmp_path() -> LocalUri:
    """Returns the local path for the GBML assets."""
    return LocalUri.join(os.sep, "tmp", "gbml_assets")


def get_gbml_task_local_tmp_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> LocalUri:
    """Returns the local tmp path for the specified job name"""
    data_local_path = LocalUri.join(get_gbml_assets_tmp_path(), applied_task_identifier)
    Path(data_local_path.uri).mkdir(parents=True, exist_ok=True)
    return data_local_path


def get_gbml_logs_folder_local_tmp_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> LocalUri:
    """Returns the local logs/ folder for the specified job name"""
    logs_local_path = LocalUri.join(
        get_gbml_task_local_tmp_path(applied_task_identifier=applied_task_identifier),
        "logs",
    )
    Path(logs_local_path.uri).mkdir(parents=True, exist_ok=True)
    return logs_local_path


def get_gbml_log_file_prefix_tmp_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> LocalUri:
    """Returns the local log file prefix for the specified job name"""
    log_file_local_path = LocalUri.join(
        get_gbml_logs_folder_local_tmp_path(applied_task_identifier), "logfile"
    )
    return log_file_local_path


def get_inference_embeddings_local_tmp_dir_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> LocalUri:
    """Returns the local directory used by the Inferencer to store embeddings."""
    inference_embeddings_local_tmp_dir = LocalUri.join(
        get_gbml_task_local_tmp_path(applied_task_identifier=applied_task_identifier),
        "inference",
    )
    Path(inference_embeddings_local_tmp_dir.uri).mkdir(parents=True, exist_ok=True)
    return inference_embeddings_local_tmp_dir


def get_preprocess_local_staging_dir_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> LocalUri:
    preprocess_local_staging_path_dir = LocalUri.join(
        get_gbml_task_local_tmp_path(applied_task_identifier=applied_task_identifier),
        "staging",
    )
    Path(preprocess_local_staging_path_dir.uri).mkdir(parents=True, exist_ok=True)
    return preprocess_local_staging_path_dir


def get_inference_embeddings_local_tmp_file_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> LocalUri:
    """Returns the local file prefix for the output embeddings of the specified job name."""
    return LocalUri.join(
        get_inference_embeddings_local_tmp_dir_path(applied_task_identifier), "emb"
    )


def get_inference_predictions_local_tmp_file_prefix(
    applied_task_identifier: AppliedTaskIdentifier,
) -> LocalUri:
    """Returns the local file prefix for the output predictions of the specified job name."""
    return LocalUri.join(
        get_inference_embeddings_local_tmp_dir_path(applied_task_identifier), "pred"
    )


def get_train_val_info_local_tmp_file_path(
    applied_task_identifier: AppliedTaskIdentifier,
) -> LocalUri:
    """Returns the local file path for the train_val_nodes_info.csv of the specified job name."""
    return LocalUri.join(
        get_inference_embeddings_local_tmp_dir_path(applied_task_identifier),
        "train_val_nodes_info.csv",
    )


def get_path_to_manifest_file() -> LocalUri:
    """Returns the local path to the MANIFEST.in file."""
    return LocalUri.join(get_gigl_root_directory(), "python", "MANIFEST.in")
