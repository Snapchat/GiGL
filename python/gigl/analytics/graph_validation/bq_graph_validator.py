from typing import Dict, Optional

from google.cloud.bigquery.table import RowIterator

from gigl.common.logger import Logger
from gigl.src.common.utils.bq import BqUtils

logger = Logger()


class BQGraphValidator:
    @staticmethod
    def does_edge_table_have_dangling_edges(
        edge_table: str,
        src_node_column_name: str,
        dst_node_column_name: str,
        query_labels: Dict[str, str] = {},
        bq_gcp_project: Optional[str] = None,
    ) -> bool:
        """
        Validate that the edge table does not contain any dangling edges.
        Meaining that an edge exists where either src_node and/or dst_node is null

        Args:
            edge_table (str): The edge table to validate
            src_node_column_name (str): The column name in the table that contains the source node ids
            dst_node_column_name (str): The column name in the table that contains the destination node ids
            query_labels (Dict[str, str], optional): Cloud Provider Labels to add to the Query. Defaults to {}.
            bq_gcp_project (Optional[str], optional): The GCP project to run the query in. If None the BQ
                client will usse the default project inferred from the environment. Defaults to None.

        Returns:
            bool: True if the edge table has no dangling edges, False otherwise
        """

        logger.info(
            f"Validating that the edge table {edge_table} with src_node_column_name="
            + f"{src_node_column_name} and dst_node_column_name={dst_node_column_name} "
            + "has no dangling edges"
        )
        query: str = f"""
            SELECT
                COUNT(*)
            FROM
                `{edge_table}`
            WHERE
                {src_node_column_name} IS NULL
                OR {dst_node_column_name} IS NULL
        """

        bq_utils = BqUtils(project=bq_gcp_project)

        result: RowIterator = bq_utils.run_query(query=query, labels=query_labels)
        count: int = list(result)[0][0]

        return count != 0
