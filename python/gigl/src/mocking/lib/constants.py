import gigl.env.dep_constants as dep_constants
from gigl.common import GcsUri, LocalUri
from gigl.src.common.constants.local_fs import get_gigl_root_directory
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType
from gigl.src.common.utils.bq import BqUtils

MOCK_DATA_GCS_BUCKET = GcsUri(f"gs://{dep_constants.GIGL_PUBLIC_BUCKET_NAME}/")
MOCK_DATA_BQ_DATASET_NAME = dep_constants.GIGL_PUBLIC_DATASET_NAME
EXAMPLE_TASK_ASSETS_GCS_PATH = GcsUri.join(MOCK_DATA_GCS_BUCKET, "mocked_assets")
MOCKED_DATASET_ARTIFACT_METADATA_LOCAL_PATH = LocalUri.join(
    get_gigl_root_directory(),
    "src",
    "mocking",
    "lib",
    "mocked_dataset_artifact_metadata.json",
)


def update_gcs_uri_with_test_assets_and_version(uri_str: str, version: str) -> str:
    """
    Replaces the bucket and path of a GCS URI with the test assets bucket and path.

    Example:
        input gs://some_bucket_name/<task_identifier>/data_preprocess/preprocessed_metadata.yaml
        output gs://{MOCK_DATA_GCS_BUCKET}/mocked_assets/<version>/<task_identifier>/data_preprocess/preprocessed_metadata.yaml
    """

    uri_tokens = uri_str.split("/")
    replaced_uri = (
        f"{EXAMPLE_TASK_ASSETS_GCS_PATH}/{version}/{'/'.join(uri_tokens[3:])}"
    )
    return replaced_uri


def update_bq_table_with_test_assets_and_version(bq_table: str, version: str) -> str:
    table_name = bq_table.split(".")[-1]
    replaced_table_name = f"{table_name}_{version}"
    replaced_bq_table = f"{MOCK_DATA_BQ_DATASET_NAME}.{replaced_table_name}"
    return replaced_bq_table


# BQ table paths for node / edge data
def get_example_task_nodes_bq_table_path(
    task_name: str, version: str, node_type: NodeType
) -> str:
    table_path = BqUtils.join_path(
        MOCK_DATA_BQ_DATASET_NAME, f"{task_name}_{str(node_type)}_nodes_{version}"
    )
    return table_path


def get_example_task_edges_bq_table_path(
    task_name: str,
    version: str,
    edge_type: EdgeType,
    edge_usage_type: EdgeUsageType,
) -> str:
    table_path = BqUtils.join_path(
        MOCK_DATA_BQ_DATASET_NAME,
        f"{task_name}_{str(edge_type)}_edges_{str(edge_usage_type)}_{version}",
    )
    return table_path


def get_example_task_static_assets_gcs_dir(task_name: str, version: str) -> GcsUri:
    return GcsUri.join(EXAMPLE_TASK_ASSETS_GCS_PATH, f"{version}/", f"{task_name}/")


# Preprocessed tfrecord paths for node / edge data


def get_example_task_preprocess_gcs_prefix(task_name: str, version: str) -> GcsUri:
    return GcsUri.join(
        get_example_task_static_assets_gcs_dir(task_name=task_name, version=version),
        "data_preprocess",
    )


def get_example_task_frozen_gbml_config_gcs_path(
    task_name: str, version: str
) -> GcsUri:
    return GcsUri.join(
        get_example_task_static_assets_gcs_dir(task_name=task_name, version=version),
        "frozen_gbml_config.yaml",
    )


def get_example_task_node_features_gcs_dir(
    task_name: str, version: str, node_type: NodeType
) -> GcsUri:
    return GcsUri.join(
        get_example_task_preprocess_gcs_prefix(task_name=task_name, version=version),
        "node_features_dir",
        node_type,
        "features/",
    )


def get_example_task_node_features_schema_gcs_path(
    task_name: str, version: str, node_type: NodeType
) -> GcsUri:
    return GcsUri.join(
        get_example_task_preprocess_gcs_prefix(task_name=task_name, version=version),
        "node_features_dir",
        node_type,
        "schema.pbtxt",
    )


def get_example_task_edge_features_gcs_dir(
    task_name: str,
    version: str,
    edge_type: EdgeType,
    edge_usage_type: EdgeUsageType,
) -> GcsUri:
    parent_uri = GcsUri.join(
        get_example_task_preprocess_gcs_prefix(task_name=task_name, version=version),
        "edge_features_dir",
        str(edge_type),
    )
    return GcsUri.join(parent_uri, f"{str(edge_usage_type)}_edges", "features/")


def get_example_task_edge_features_schema_gcs_path(
    task_name: str,
    version: str,
    edge_type: EdgeType,
    edge_usage_type: EdgeUsageType,
) -> GcsUri:
    parent_uri = GcsUri.join(
        get_example_task_preprocess_gcs_prefix(task_name=task_name, version=version),
        "edge_features_dir",
        str(edge_type),
    )
    return GcsUri.join(
        parent_uri,
        f"{str(edge_usage_type)}_edges",
        "schema.pbtxt",
    )
