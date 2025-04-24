import datetime as dt
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.time import DEFAULT_DATE_FORMAT
from gigl.src.common.utils.bq import BqUtils
from gigl.src.inference.v1.lib.inference_output_schema import (
    DEFAULT_EMBEDDING_FIELD,
    DEFAULT_NODE_ID_FIELD,
)

COSINE_SIM_FIELD = "_cosine"


def get_table_paths_via_timedelta(
    bq_utils: BqUtils, reference_table: str, lookback_days: int
) -> Tuple[str, str]:
    """
    Args:
        bq_utils (BqUtils)
        reference_table (str): example: project.gbml_embeddings.embeddings_gigl_2024_01_01
        lookback_days (int): search within this many days and get the latest available table
    """
    reference_table_formatted = bq_utils.format_bq_path(bq_path=reference_table)
    reference_table_split = reference_table_formatted.split(".")
    table_id = reference_table_split[2]
    bq_dataset_path = ".".join(reference_table_split[:2])
    table_match_string = table_id.replace(
        table_id.split("_")[-2], ".*"
    )  # replacing the date in name with regex
    reference_date_str = reference_table_split[2].split("_")[-2]
    reference_date = datetime.strptime(reference_date_str, DEFAULT_DATE_FORMAT)
    look_back_date = reference_date - dt.timedelta(days=lookback_days)
    lookback_date_str = look_back_date.strftime(DEFAULT_DATE_FORMAT)
    table_names_latest_to_oldest = bq_utils.get_table_names_within_date_range(
        bq_dataset_path=bq_dataset_path,
        table_match_string=table_match_string,
        start_date=lookback_date_str,
        end_date=reference_date_str,
    )
    if len(table_names_latest_to_oldest) == 0:
        raise ValueError(
            f"latest tables list {table_names_latest_to_oldest} is empty. Please check if the reference_table is correct or increase the search date."
        )
    latest_table = table_names_latest_to_oldest[0]
    return reference_table, latest_table


def calculate_cosine_sim_between_embedding_tables(
    bq_utils: BqUtils, table_1: str, table_2: str, n: int
) -> pd.DataFrame:
    """
    Return: a pd.Dataframe with columns: {DEFAULT_NODE_ID_FIELD, _emb_1, _emb_2, COSINE_SIM_FIELD}
    NOTE: Currently, the query below takes 17min for n=100M. If in future we wish to increase n
    to avoid the issue: `results that exceed the BQ query limit`, we can comment out the last lines.
    For, now we don't do so as we don't need to evaluate cosine similarity for more than 100M embeddings.
    Hence, there is no need to store an extra table in BQ.
    """

    cosine_sim_query = f"""
    WITH joined_table AS(
        WITH reference_table AS
            (SELECT
                {DEFAULT_NODE_ID_FIELD}, {DEFAULT_EMBEDDING_FIELD} as _emb_1
            FROM
                `{table_1}`
            LIMIT {n})
        SELECT
            reference_table.{DEFAULT_NODE_ID_FIELD}, _emb_1, _emb_2
        FROM
            reference_table
        JOIN
            (SELECT {DEFAULT_NODE_ID_FIELD}, {DEFAULT_EMBEDDING_FIELD} as _emb_2
            FROM `{table_2}`) AS table_2
        ON
            reference_table.{DEFAULT_NODE_ID_FIELD} = table_2.{DEFAULT_NODE_ID_FIELD})
    SELECT
    joined_table.{DEFAULT_NODE_ID_FIELD} AS {DEFAULT_NODE_ID_FIELD},
    ( 1 -
      ML.DISTANCE(
             joined_table._emb_1,
             joined_table._emb_2,
             "COSINE"
       )
   ) AS {COSINE_SIM_FIELD}
    FROM
    joined_table
    """
    results = bq_utils.run_query(
        query=cosine_sim_query,
        labels=get_resource_config().get_resource_labels(),
    )
    # to enable returning query results that exceed the BQ query limit
    # reference:https://cloud.google.com/bigquery/docs/writing-results#large-results
    # results = bq_utils.run_query(
    #     query=cosine_sim_query,
    #     labels=get_resource_config().get_resource_labels(),
    #     allow_large_results=True,
    #     destination=_reference_table_id,
    #     write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    # )
    return results.to_dataframe()


def calculate_cosine_similarity_stats(
    cosine_sim_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates statistics of cosine similarity
    Args: pd.DataFrame: with columns: {DEFAULT_NODE_ID_FIELD, _emb_1, _emb_2, COSINE_SIM_FIELD}
    Returns:
        pd.DataFrame: with columns: {count, mean, std, min, 1%, 5%, 25%, 50%, 75%, 95%, 99%, max, dtype}
    """
    print(cosine_sim_df.head())
    return cosine_sim_df[COSINE_SIM_FIELD].describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )


def assert_cosine_similarity_stats(
    cosine_similarity_stats: pd.DataFrame, expected_cosine_similarity: Dict[str, float]
) -> None:
    for stat, expected_val in expected_cosine_similarity.items():
        if cosine_similarity_stats[stat] < expected_val:
            raise ValueError(
                f"cosine similarity {stat} is {cosine_similarity_stats[stat]}, which is less than expected value {expected_val}"
            )
