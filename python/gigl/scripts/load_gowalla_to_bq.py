"""
Script to load the Gowalla bipartite graph dataset to BigQuery.

The Gowalla dataset is a bipartite graph of users and items (locations).
This script downloads the data from the neural graph collaborative filtering repository
and loads it to BigQuery as an edge table.

Data format:
Each line represents a user and the items they have interacted with:
<user_id> <item_id_1> <item_id_2> ... <item_id_n>

Example usage:
    python -m gigl.scripts.load_gowalla_to_bq \
        --project <gcp_project_id> \
        --dataset <dataset_name> \
        --table <table_name> \
        --data_file_url https://raw.githubusercontent.com/xiangwang1223/neural_graph_collaborative_filtering/master/Data/gowalla/test.txt
"""

import argparse
import json
import os
import tempfile
from typing import Final, Iterator

import google.cloud.bigquery as bigquery
import requests

from gigl.common import LocalUri, UriFactory
from gigl.common.logger import Logger
from gigl.src.common.utils.bq import BqUtils

logger = Logger()

# Default URLs for the gowalla dataset files
DEFAULT_TRAIN_URL: Final[str] = (
    "https://raw.githubusercontent.com/xiangwang1223/neural_graph_collaborative_filtering/master/Data/gowalla/train.txt"
)
DEFAULT_TEST_URL: Final[str] = (
    "https://raw.githubusercontent.com/xiangwang1223/neural_graph_collaborative_filtering/master/Data/gowalla/test.txt"
)

# Default column names for the edge table
DEFAULT_SRC_COLUMN: Final[str] = "from_user_id"
DEFAULT_DST_COLUMN: Final[str] = "to_item_id"


def download_file(url: str, local_path: str) -> None:
    """
    Download a file from a URL to a local path.

    Args:
        url (str): URL to download from.
        local_path (str): Local path to save the file.
    """
    logger.info(f"Downloading data from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Downloaded data to {local_path}")


def parse_gowalla_edges(file_path: str) -> Iterator[dict[str, int]]:
    """
    Parse the Gowalla edge list format and yield edge dictionaries.

    Each line in the file represents a user and the items they have interacted with:
    <user_id> <item_id_1> <item_id_2> ... <item_id_n>

    This function yields one edge (user -> item) per interaction.

    Args:
        file_path (str): Path to the file containing edge data.

    Yields:
        dict[str, int]: Dictionary with 'src' (user) and 'dst' (item) keys.
    """
    logger.info(f"Parsing edges from {file_path}")

    edge_count = 0
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                logger.warning(
                    f"Line {line_num} has insufficient data (expected at least 2 columns): {line}"
                )
                continue

            user_id = int(parts[0])
            item_ids = [int(item_id) for item_id in parts[1:]]

            for item_id in item_ids:
                yield {DEFAULT_SRC_COLUMN: user_id, DEFAULT_DST_COLUMN: item_id}
                edge_count += 1

    logger.info(f"Parsed {edge_count} edges from {file_path}")


def convert_edges_to_jsonl(
    input_file_path: str, output_file_path: str
) -> tuple[int, int]:
    """
    Convert the Gowalla edge list to JSONL format for BigQuery loading.

    Args:
        input_file_path (str): Path to the input edge list file.
        output_file_path (str): Path to write the JSONL output.

    Returns:
        tuple[int, int]: Number of edges and number of unique users processed.
    """
    logger.info(f"Converting edges to JSONL format: {output_file_path}")

    edge_count = 0
    unique_users = set()

    with open(output_file_path, "w") as out_f:
        for edge_dict in parse_gowalla_edges(input_file_path):
            json.dump(edge_dict, out_f)
            out_f.write("\n")
            edge_count += 1
            unique_users.add(edge_dict[DEFAULT_SRC_COLUMN])

    logger.info(
        f"Converted {edge_count} edges from {len(unique_users)} unique users to JSONL"
    )
    return edge_count, len(unique_users)


def load_gowalla_to_bigquery(
    project: str,
    dataset: str,
    table: str,
    data_file_url: str,
    src_column: str = DEFAULT_SRC_COLUMN,
    dst_column: str = DEFAULT_DST_COLUMN,
    recreate_table: bool = True,
) -> None:
    """
    Load the Gowalla dataset to BigQuery.

    Args:
        project (str): GCP project ID.
        dataset (str): BigQuery dataset name.
        table (str): BigQuery table name.
        data_file_url (str): URL to download the Gowalla data from.
        src_column (str): Name of the source column. Defaults to 'src'.
        dst_column (str): Name of the destination column. Defaults to 'dst'.
        recreate_table (bool): Whether to recreate the table if it exists. Defaults to True.
    """
    bq_utils = BqUtils(project=project)
    bq_path = BqUtils.join_path(project, dataset, table)

    logger.info(f"Loading Gowalla data to BigQuery table: {bq_path}")

    # Create the dataset if it doesn't exist
    dataset_path = BqUtils.join_path(project, dataset)
    bq_utils.create_bq_dataset(dataset_id=dataset_path, exists_ok=True)

    # Define the schema for the edge table
    schema = [
        bigquery.SchemaField(src_column, "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField(dst_column, "INTEGER", mode="REQUIRED"),
    ]

    # Recreate the table if requested
    if recreate_table:
        logger.info(f"Recreating table {bq_path}")
        bq_utils.create_or_empty_bq_table(bq_path=bq_path, schema=schema)

    # Download the data to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp_data:
        download_file(data_file_url, tmp_data.name)
        tmp_data_path = tmp_data.name

    # Convert to JSONL format
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as tmp_jsonl:
        tmp_jsonl_path = tmp_jsonl.name

    try:
        edge_count, user_count = convert_edges_to_jsonl(tmp_data_path, tmp_jsonl_path)

        logger.info(
            f"Loading {edge_count} edges from {user_count} users to BigQuery table {bq_path}"
        )

        # Load the JSONL file to BigQuery
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema=schema,
        )

        bq_utils.load_file_to_bq(
            source_path=UriFactory.create_uri(tmp_jsonl_path),
            bq_path=bq_path,
            job_config=job_config,
            retry=True,
        )

        logger.info(f"Successfully loaded Gowalla data to {bq_path}")
        logger.info(f"Total edges: {edge_count}")
        logger.info(f"Total unique users: {user_count}")

        # Verify the load
        actual_row_count = bq_utils.count_number_of_rows_in_bq_table(
            bq_table=bq_path, labels={}
        )
        logger.info(f"Verified row count in BigQuery: {actual_row_count}")

    finally:
        # Clean up temporary files
        try:
            os.unlink(tmp_data_path)
            logger.info(f"Cleaned up temporary file: {tmp_data_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {tmp_data_path}: {e}")

        try:
            os.unlink(tmp_jsonl_path)
            logger.info(f"Cleaned up temporary file: {tmp_jsonl_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file {tmp_jsonl_path}: {e}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Load Gowalla bipartite graph dataset to BigQuery"
    )

    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="GCP project ID",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="BigQuery dataset name",
    )

    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="BigQuery table name for edges",
    )

    parser.add_argument(
        "--data_file_url",
        type=str,
        default=DEFAULT_TRAIN_URL,
        help=f"URL to download the Gowalla edge data from. Defaults to train.txt: {DEFAULT_TRAIN_URL}",
    )

    parser.add_argument(
        "--src_column",
        type=str,
        default=DEFAULT_SRC_COLUMN,
        help=f"Name of the source node column. Defaults to '{DEFAULT_SRC_COLUMN}'",
    )

    parser.add_argument(
        "--dst_column",
        type=str,
        default=DEFAULT_DST_COLUMN,
        help=f"Name of the destination node column. Defaults to '{DEFAULT_DST_COLUMN}'",
    )

    parser.add_argument(
        "--no_recreate_table",
        action="store_true",
        help="Do not recreate the table if it exists (append mode)",
    )

    args = parser.parse_args()

    load_gowalla_to_bigquery(
        project=args.project,
        dataset=args.dataset,
        table=args.table,
        data_file_url=args.data_file_url,
        src_column=args.src_column,
        dst_column=args.dst_column,
        recreate_table=not args.no_recreate_table,
    )

if __name__ == "__main__":
    main()
