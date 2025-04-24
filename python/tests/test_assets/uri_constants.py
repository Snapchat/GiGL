# TODO: (Open Source) Marked for Refactor
import os

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import GcsUri, UriFactory
from gigl.env import dep_constants

# TODO: (svij) This bucket needs to be updated
TEST_DATA_GCS_BUCKET = GcsUri(f"gs://{dep_constants.GIGL_TEST_BUCKET_NAME}/")
EXAMPLE_TASK_ASSETS_GCS_PATH = GcsUri.join(TEST_DATA_GCS_BUCKET, "example_task_assets")

DEFAULT_TEST_RESOURCE_CONFIG_URI = UriFactory.create_uri(
    os.path.join(
        local_fs_constants.get_project_root_directory(),
        "scala",
        "common",
        "src",
        "test",
        "assets",
        "resource_config.yaml",
    )
)

DEFAULT_NABLP_TASK_CONFIG_URI = UriFactory.create_uri(
    os.path.join(
        local_fs_constants.get_gigl_root_directory(),
        "src",
        "mocking",
        "configs",
        "e2e_node_anchor_based_link_prediction_template_gbml_config.yaml",
    )
)


def get_example_task_static_assets_gcs_dir(task_name: str) -> GcsUri:
    return GcsUri.join(EXAMPLE_TASK_ASSETS_GCS_PATH, task_name)
