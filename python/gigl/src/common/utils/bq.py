import datetime
import itertools
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union

import google.api_core.retry
import google.cloud.bigquery as bigquery
from google.api_core.exceptions import NotFound
from google.cloud.bigquery._helpers import _record_field_to_json
from google.cloud.bigquery.job import _AsyncJob
from google.cloud.bigquery.table import RowIterator

from gigl.common import GcsUri, LocalUri, Uri
from gigl.common.logger import Logger
from gigl.common.utils.retry import retry
from gigl.src.common.constants.time import DEFAULT_DATE_FORMAT
from gigl.src.common.utils.time import convert_days_to_ms, current_datetime

logger = Logger()


def _load_file_to_bq(
    source: Uri,
    bq_path: str,
    client: bigquery.Client,
    job_config: bigquery.LoadJobConfig,
) -> _AsyncJob:
    try:
        if isinstance(source, GcsUri):
            # if gcs, need to use load_table_from_uri
            load_job = client.load_table_from_uri(
                source.uri, bq_path, job_config=job_config
            )
        elif isinstance(source, LocalUri):
            # if local, need to use load_table_from_file
            with open(source.uri, "rb") as source_path_obj:
                # API request -- starts the job
                load_job = client.load_table_from_file(
                    source_path_obj, bq_path, job_config=job_config
                )
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")
        logger.info(f"Loading {source} to {bq_path}")
        return load_job.result()  # Waits for job to complete.
    except Exception as e:
        logger.exception(f"Could not load file to BQ. {repr(e)}")
        raise e


@retry()
def _load_file_to_bq_with_retry(
    source_path: Uri,
    bq_path: str,
    client: bigquery.Client,
    job_config: bigquery.LoadJobConfig,
) -> _AsyncJob:
    return _load_file_to_bq(source_path, bq_path, client, job_config)


class BqUtils:
    def __init__(self, project: Optional[str] = None) -> None:
        logger.info(f"BqUtils initialized with project: {project}")
        self.__bq_client = bigquery.Client(project=project)

    def create_bq_dataset(self, dataset_id, exists_ok=True) -> None:
        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"
        try:
            self.__bq_client.create_dataset(dataset, exists_ok=exists_ok)  # API request
            logger.info(f"Created dataset {dataset_id}")
        except Exception as e:
            logger.exception(f"Could not create dataset. {repr(e)}")

    def get_dataset_name_from_table(self, bq_path: str) -> str:
        dataset_id = ".".join(bq_path.split(".")[:-1])
        return dataset_id

    def create_or_empty_bq_table(
        self, bq_path: str, schema: Optional[List[bigquery.SchemaField]] = None
    ) -> None:
        bq_path = self.format_bq_path(bq_path)
        split_bq_path = bq_path.split(".")
        if len(split_bq_path) == 2:
            dataset_name = split_bq_path[0]
        elif len(split_bq_path) == 3:
            dataset_name = split_bq_path[1]
        else:
            raise Exception(f"Could not parse BQ table path: {bq_path}")
        self.__bq_client.create_dataset(
            dataset=dataset_name, exists_ok=True
        )  # No-Op if dataset exists
        self.__bq_client.delete_table(
            table=bq_path, not_found_ok=True
        )  # Deletes if table exists
        table: Union[str, bigquery.Table] = (
            bigquery.Table(table_ref=bq_path, schema=schema) if schema else bq_path
        )
        self.__bq_client.create_table(
            table=table, exists_ok=False
        )  # Recreate the table

    def count_number_of_rows_in_bq_table(
        self,
        bq_table: str,
        labels: Dict[str, str] = {},
    ) -> int:
        bq_table = bq_table.replace(":", ".")
        ROW_COUNTING_QUERY = f"""
        SELECT count(1) AS ct FROM `{bq_table}`
        """
        result = self.run_query(query=ROW_COUNTING_QUERY, labels=labels)
        for row in result:
            n_rows = row["ct"]
        return n_rows

    def count_number_of_columns_in_bq_table(
        self,
        bq_table: str,
    ) -> int:
        schema = self.fetch_bq_table_schema(bq_table=bq_table)
        return len(schema.keys())

    def run_query(
        self,
        query,
        labels: Dict[str, str],
        **job_config_args,
    ) -> RowIterator:
        logger.info(f"Running query: {query}")
        job_config = bigquery.QueryJobConfig(**job_config_args)
        job_config.labels = labels
        # Start the query, passing in the extra configuration.
        try:
            query_job = self.__bq_client.query(
                query, location="US", job_config=job_config
            )  # API request - starts the query

            # Waits for the query to finish and returns the result row iterator object.
            result = query_job.result()
            return result
        except Exception as e:
            logger.exception(f"Could not run query: {e}")
            raise e

    @staticmethod
    def format_bq_path(bq_path: str, format_for_table_reference: bool = False) -> str:
        """Formats BQ paths.

        Args:
            bq_path (str): expected to be one of:
                "<project>.<dataset>.<table>" or "<project>:<dataset>.<table>"
                "<project>.<dataset>" or "<project>:<dataset>"
                "<dataset>.<table>"
            format_for_table_reference (bool, optional): If project, dataset, and
            table are all specified; add the `:` seperator between project and dataset.
            Useful for when "table_reference" is required instead of path i.e. for
            using BigQuery IO operator for beam pipelines.
            Defaults to False.

        Returns:
            str: Formatted bq path
        """

        bq_path = bq_path.replace(":", ".")
        count_bq_path_parts = bq_path.count(".")
        assert (
            count_bq_path_parts > 0 and count_bq_path_parts < 3
        ), "BQ path expected to contain project + dataset and/or table."
        split_path = bq_path.split(".")
        project = split_path[0]
        dataset_and_table_name = ".".join(split_path[1:])
        return_path = ".".join([project, dataset_and_table_name])
        if format_for_table_reference and count_bq_path_parts == 2:
            return_path = return_path.replace(".", ":", 1)

        return return_path

    @staticmethod
    def join_path(path: str, *paths) -> str:
        joined_path = ".".join([path, *paths])
        assert joined_path.count(".") <= 2, f"Invalid BQ path: {joined_path}"
        return BqUtils.format_bq_path(joined_path)

    @staticmethod
    def parse_bq_table_path(bq_table_path: str) -> Tuple[str, str, str]:
        """
        Parses a joined bq table path into its project, dataset, and table names
        Args:
            bq_table_path (str): Joined bq table path of format `project.dataset.table`
        Returns:
            bq_project_id (str): Parsed BQ Project ID
            bq_dataset_id (str): Parsed Dataset ID
            bq_table_name (str): Parsed Table Name
        """
        split_bq_table_path = BqUtils.format_bq_path(bq_table_path).split(".")
        assert (
            len(split_bq_table_path) == 3
        ), "bqtable_path should be in the format project.dataset.table"
        bq_project_id, bq_dataset_id, bq_table_name = split_bq_table_path

        return bq_project_id, bq_dataset_id, bq_table_name

    def update_bq_dataset_retention(
        self,
        bq_dataset_path: str,
        retention_in_days: int,
        apply_retroactively: Optional[bool] = False,
    ) -> None:
        """
        Update default retention for a whole BQ dataset.
        This applies only to new tables unless apply_retroactively=True.

        :param bq_dataset_path: The BigQuery dataset path in the format `project_id.dataset_id`.
        :param retention_in_days: The number of days to retain data in BigQuery tables.
        :param apply_retroactively: If True, applies this retention policy retroactively to all existing tables in the dataset.
        """
        bq_dataset_path = BqUtils.format_bq_path(bq_dataset_path)
        dataset = self.__bq_client.get_dataset(bq_dataset_path)
        retention_in_ms = convert_days_to_ms(retention_in_days)

        dataset.default_table_expiration_ms = retention_in_ms
        try:
            self.__bq_client.update_dataset(dataset, ["default_table_expiration_ms"])
            logger.info(
                f"Updated dataset {bq_dataset_path} with default expiration in {retention_in_days} days."
            )
        except Exception as e:
            logger.exception(e)

        if apply_retroactively:
            for table_item in self.__bq_client.list_tables(dataset):
                table_id = table_item.full_table_id
                self.update_bq_table_retention(table_id, retention_in_days)

    def update_bq_table_retention(
        self, bq_table_path: str, retention_in_days: int
    ) -> None:
        """
        Update retention of a single BQ table.
        :param bq_table_path:
        :param retention_in_days:
        :param client:
        :return:
        """

        bq_table_path = BqUtils.format_bq_path(bq_table_path)
        table = bigquery.Table(bq_table_path)
        expiration_dt = current_datetime() + datetime.timedelta(days=retention_in_days)
        table.expires = expiration_dt
        try:
            self.__bq_client.update_table(table, ["expires"])
            logger.info(
                f"Updated table {bq_table_path} to expire in {retention_in_days} days."
            )
        except Exception as e:
            logger.exception(e)

    def does_bq_table_exist(self, bq_table_path: str) -> bool:
        exists = False
        try:
            bq_table_path = BqUtils.format_bq_path(bq_table_path)
            self.__bq_client.get_table(bq_table_path)  # Make an API request.
            exists = True
            logger.info(f"Table {bq_table_path} exists.")
        except NotFound:
            logger.info(f"Table {bq_table_path} not found.")
        except Exception as e:
            logger.info(f"Could not evaluate table existence. {repr(e)}")
        return exists

    def list_matching_tables(
        self, bq_dataset_path: str, table_match_string: str
    ) -> List[str]:
        bq_dataset_path = BqUtils.format_bq_path(bq_dataset_path)
        tables = self.__bq_client.list_tables(bq_dataset_path)
        matching_tables = list()
        for table_list_item in tables:
            if table_match_string in table_list_item.table_id:
                formatted_table_path = BqUtils.format_bq_path(
                    table_list_item.full_table_id
                )
                matching_tables.append(formatted_table_path)
        return matching_tables

    def delete_matching_tables(
        self, bq_dataset_path: str, table_match_string: str
    ) -> None:
        try:
            bq_dataset_path = BqUtils.format_bq_path(bq_dataset_path)
            tables = self.__bq_client.list_tables(bq_dataset_path)
            for table_list_item in tables:
                if table_match_string in table_list_item.table_id:
                    formatted_table_path = BqUtils.format_bq_path(
                        table_list_item.full_table_id
                    )
                    self.__bq_client.delete_table(formatted_table_path)
                    logger.info(f"Deleted table {formatted_table_path}.")
        except Exception as e:
            logger.exception("Error in deleting tables." + repr(e))

    def get_table_names_within_date_range(
        self,
        bq_dataset_path: str,
        table_match_string: str,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """
        start_date and end_date are in the format of 'YYYYMMDD'
        table_match_string is a regex string to match table names
        """
        _start_date = datetime.datetime.strptime(start_date, DEFAULT_DATE_FORMAT).date()
        _end_date = datetime.datetime.strptime(end_date, DEFAULT_DATE_FORMAT).date()
        filtered_tables_by_name = list()
        filtered_tables_by_date = list()
        bq_dataset_path = BqUtils.format_bq_path(bq_dataset_path)
        all_tables = self.__bq_client.list_tables(bq_dataset_path)
        for table in all_tables:
            if re.search(table_match_string, table.table_id):
                filtered_tables_by_name.append(table)
        sorted_tables_by_date = sorted(
            filtered_tables_by_name, key=lambda x: x.created, reverse=True
        )
        for table in sorted_tables_by_date:
            if _start_date <= table.created.date() <= _end_date:
                filtered_tables_by_date.append(
                    ".".join([bq_dataset_path, table.table_id])
                )
        return filtered_tables_by_date

    def delete_bq_table_if_exist(
        self, bq_table_path: str, not_found_ok: bool = True
    ) -> None:
        """bq_table_path = 'your-project.your_dataset.your_table'"""
        bq_table_path = BqUtils.format_bq_path(bq_table_path)
        try:
            self.__bq_client.delete_table(bq_table_path, not_found_ok=not_found_ok)
            logger.info(f"Table deleted '{bq_table_path}'")
        except Exception as e:
            logger.exception(f"Failed to delete table '{bq_table_path}' due to \n {e}")

    def fetch_bq_table_schema(self, bq_table: str) -> Dict[str, bigquery.SchemaField]:
        """
        Create a dictionary representation for SchemaFields from BigQuery table.
        """

        bq_table = bq_table.replace(":", ".")
        bq_schema = self.__bq_client.get_table(bq_table).schema
        schema_dict = {field.name: field for field in bq_schema}
        return schema_dict

    def load_file_to_bq(
        self,
        source_path: Uri,
        bq_path: str,
        job_config: bigquery.LoadJobConfig,
        retry: bool = False,
    ) -> _AsyncJob:
        """
        Uploads a single file to biqquery.

        Args:
            source_path (Uri): The source file to upload.
            bq_path (str): The BigQuery table path to upload to.
            job_config (bigquery.LoadJobConfig): The job configuration for the upload.
            retry (bool, optional): Whether to retry the upload if it fails. Defaults to False.
        Returns: The job object for the upload.
        """
        if retry:
            result = _load_file_to_bq_with_retry(
                source_path, bq_path, self.__bq_client, job_config
            )
        else:
            result = _load_file_to_bq(
                source_path, bq_path, self.__bq_client, job_config
            )
        return result

    def load_rows_to_bq(
        self,
        bq_path: str,
        schema: List[bigquery.SchemaField],
        rows: Iterable[Tuple],
    ) -> None:
        first_item = next(iter(rows), None)
        if first_item is None:
            logger.warning(f"No rows to insert into {bq_path}.")
            return

        _BQ_INSERT_REQUEST_LIMIT_BYTES = (
            10_000_000 - 1_000_000
        )  # 10MB is the limit; we leave 1MB of buffer
        _RETRY_BACKOFF = google.api_core.retry.Retry(
            # retries if and only if the 'reason' is 'backendError' or 'rateLimitExceeded'
            predicate=bigquery.DEFAULT_RETRY._predicate,
            initial=1.0,  # 1 second
            maximum=60.0 * 5,  # 5 minutes
            multiplier=1.5,
            timeout=60 * 30,  # 30 mins
        )

        table = bigquery.Table(table_ref=bq_path, schema=schema)
        batch_rows = []
        estimated_batch_rows_size_bytes = 0

        for row in itertools.chain([first_item], rows):
            json_row: dict = _record_field_to_json(schema, row)
            batch_rows.append(json_row)
            estimated_batch_rows_size_bytes += len(str(json_row))
            if estimated_batch_rows_size_bytes > _BQ_INSERT_REQUEST_LIMIT_BYTES:
                self.__bq_client.insert_rows_json(
                    table=table,
                    json_rows=batch_rows,
                    retry=_RETRY_BACKOFF,
                )
                batch_rows = []
                estimated_batch_rows_size_bytes = 0

        if len(batch_rows) > 0:
            self.__bq_client.insert_rows_json(
                table=table,
                json_rows=batch_rows,
                retry=_RETRY_BACKOFF,
            )

    def check_columns_exist_in_table(
        self, bq_table: str, columns: Iterable[str]
    ) -> None:
        schema = self.fetch_bq_table_schema(bq_table=bq_table)
        all_fields = set(schema.keys())
        missing_fields = [field for field in columns if field not in all_fields]
        if missing_fields:
            raise ValueError(f"Fields {missing_fields} missing from table {bq_table}.")
        else:
            logger.info(f"All requisite fields found in table {bq_table}")

    def export_to_gcs(
        self,
        bq_table_path: str,
        destination_gcs_uri: GcsUri,
        destination_format: str = "NEWLINE_DELIMITED_JSON",
    ) -> None:
        """
        Export a BigQuery table to Google Cloud Storage.

        Args:
            bq_table_path (str): The full BigQuery table path to export.
            destination_gcs_uri (str): The destination GCS URI where the table will be exported.
                If the gcs uri has * in it, the table will be exported to multiple shards.
            destination_format (str, optional): The format of the exported data. Defaults to 'NEWLINE_DELIMITED_JSON'.
                'CSV', 'AVRO', 'PARQUET' also available.
        """
        try:
            job_config = bigquery.job.ExtractJobConfig()
            job_config.destination_format = destination_format

            extract_job = self.__bq_client.extract_table(
                source=bigquery.TableReference.from_string(bq_table_path),
                destination_uris=destination_gcs_uri.uri,
                job_config=job_config,
            )

            logger.info(
                f"Exporting `{bq_table_path}` to {destination_gcs_uri} with format '{destination_format}'..."
            )
            extract_job.result()  # Waits for job to complete.
            logger.info(
                f"Exported `{bq_table_path}` to {destination_gcs_uri} successfully."
            )
        except Exception as e:
            logger.exception(f"Failed to export table to GCS.")
            raise e
