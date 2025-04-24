from typing import Optional, Protocol

from gigl.common import Uri
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper


class BaseIngestion(Protocol):
    """
    Users should implement this protocol for their ingestion into GraphDB.
    """

    def ingest(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        resource_config_uri: Uri,
        applied_task_identifier: AppliedTaskIdentifier,
        custom_worker_image_uri: Optional[
            str
        ] = None,  # TODO: (abatra2-sc): Can we make ingestion more generic? i.e not require dataflow image
    ) -> None:
        """
        This function runs the ingestion process. Should perform the operations needed to ingest all data into GraphDB
        in preperation for running subgraph sampler queries.
        """
        ...

    def clean_up(self) -> None:
        """
        This function runs after the ingestion process. It can be used to perform any operation needed such as
        closing connections, cleaning up temporary files, etc.
        """
        ...
