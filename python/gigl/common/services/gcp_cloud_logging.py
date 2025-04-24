from datetime import datetime
from typing import Iterable, Iterator

from google.cloud.logging_v2.services.logging_service_v2 import (
    LoggingServiceV2Client,
    pagers,
)
from google.cloud.logging_v2.types import ListLogEntriesRequest, log_entry

from gigl.common.logger import Logger

logger = Logger()

CLOUD_LOGGING_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class LogEntries:
    def __init__(self, list_log_entry_pager: pagers.ListLogEntriesPager) -> None:
        self.__list_log_entry_pager_iterator: Iterator[log_entry.LogEntry] = iter(
            list_log_entry_pager
        )

    def __iter__(self):
        return self

    def __next__(self) -> str:
        entry: log_entry.LogEntry = next(self.__list_log_entry_pager_iterator)
        return entry.text_payload


class GCPCloudLoggingService:
    def __init__(self) -> None:
        self.__client = LoggingServiceV2Client()

    def get_logs_iterator_from_k8_container(
        self,
        project_id: str,
        cluster_name: str,
        pod_name: str,
        datetime_start: datetime,
        datetime_end: datetime,
        query_filter: str,
    ) -> Iterable[str]:
        """Get logs for a kubernetes pod from GCP Logging.

        Args:
            project_id (str):
            cluster_name (str):
            pod_name (str):
            datetime_start (datetime):
            datetime_end (datetime):
            query_filter (str): A query filter to filter the logs.
            For example, if you want to get logs that contain dataflow job uris you can
            use the following query filter: https://console.cloud.google.com/dataflow/jobs/

        Returns:
            Iterable[str]: The log messages that match the query filter. Warning this will keep
            iterating, ensure you provide some restrictive query filter unless you want to download
            all the logs from the pod.
        """

        if not (query_filter.startswith('"') and query_filter.endswith('"')):
            query_filter = f'"{query_filter}"'

        resource_names = [f"projects/{project_id}"]
        order_by_clause = "timestamp desc"
        log_query_filter = f"""
        resource.type="k8s_container"
        resource.labels.cluster_name:"{cluster_name}"
        resource.labels.pod_name:"{pod_name}"
        timestamp >= "{datetime_start.strftime(CLOUD_LOGGING_DATETIME_FORMAT)}"
        timestamp <= "{datetime_end.strftime(CLOUD_LOGGING_DATETIME_FORMAT)}"
        {query_filter}
        """

        logger.info(f"Querying gcp logs with filter: {log_query_filter}")

        request = ListLogEntriesRequest(
            resource_names=resource_names,
            order_by=order_by_clause,
            filter=log_query_filter,
        )
        result: pagers.ListLogEntriesPager = self.__client.list_log_entries(
            request=request
        )
        return LogEntries(result)
