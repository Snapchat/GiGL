# This file can probably be gigl-generic utilities.
# We include a few graph-related IterableDatasets backed by GCS and BigQuery

from typing import Any, Iterator, List, Mapping, Optional, TypedDict

import numpy as np
import orjson
import pyarrow.parquet as pq
import torch
from google.cloud.bigquery_storage import BigQueryReadClient
from google.cloud.bigquery_storage_v1.types import DataFormat, ReadSession
from torch.utils.data._utils.worker import WorkerInfo

from gigl.common.types.uri.gcs_uri import GcsUri
from gigl.common.types.uri.uri_factory import UriFactory
from gigl.common.utils.torch_training import get_rank, get_world_size
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.training.v1.lib.data_loaders.utils import (
    get_data_split_for_current_worker,
)

SRC_FIELD = "src"
DST_FIELD = "dst"
CONDENSED_EDGE_TYPE_FIELD = "condensed_edge_type"


class HeterogeneousGraphEdgeDict(TypedDict):
    src: str
    dst: str
    condensed_edge_type: str


class GcsIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        file_uris: List[GcsUri],
        seed: int = 42,
    ) -> None:
        """
        Args:
            file_uris (List[UriType]): Holds all the uris for the dataset.
            Note: for now only uris supported are ones that `tf.data.TFRecordDataset`
            can load from default; i.e .GcsUri and LocalUri.
            We permute the file list based on a seed as a means of "shuffling" the data
            on a file-level (rather than sample-level, as would be possible in cases
            where the data fits in memory.
        """
        assert isinstance(file_uris, list)
        self._file_uris: np.ndarray = np.random.RandomState(seed).permutation(
            np.array([uri.uri for uri in file_uris])
        )
        self._file_loader: Optional[FileLoader] = None

    def _iterator_init(self):
        # Initialize it here to avoid client pickling issues for multiprocessing.
        if not self._file_loader:
            self._file_loader = FileLoader()

        # Need to first split the work based on worker information
        current_worker_file_uris_to_process = get_data_split_for_current_worker(
            self._file_uris
        )

        return current_worker_file_uris_to_process

    def __iter__(self) -> Iterator[Any]:
        raise NotImplemented


class GcsJSONLIterableDataset(GcsIterableDataset):
    def __init__(
        self,
        file_uris: List[GcsUri],
        seed: int = 42,
    ) -> None:
        """
        Args:
            file_uris (List[UriType]): Holds all the uris for the dataset.
            Note: for now only uris supported are ones that `tf.data.TFRecordDataset`
            can load from default; i.e .GcsUri and LocalUri.
            We permute the file list based on a seed as a means of "shuffling" the data
            on a file-level (rather than sample-level, as would be possible in cases
            where the data fits in memory.
        """
        super().__init__(file_uris=file_uris, seed=seed)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        current_worker_file_uris_to_process = self._iterator_init()
        assert self._file_loader is not None, "File loader not initialized"

        for file in current_worker_file_uris_to_process:
            tfh = self._file_loader.load_to_temp_file(
                file_uri_src=UriFactory.create_uri(file), delete=True
            )
            with open(tfh.name, "rb") as f:
                # Read the file and yield each line
                for line in f:
                    data = orjson.loads(line)
                    yield data


class GcsParquetIterableDataset(GcsIterableDataset):
    def __init__(
        self, file_uris: List[GcsUri], seed: int = 42, batch_size: Optional[int] = None
    ) -> None:
        """
        Args:
            file_uris (List[UriType]): Holds all the uris for the dataset.
            Note: for now only uris supported are ones that `tf.data.TFRecordDataset`
            can load from default; i.e .GcsUri and LocalUri.
            We permute the file list based on a seed as a means of "shuffling" the data
            on a file-level (rather than sample-level, as would be possible in cases
            where the data fits in memory.
        """
        self._iter_batches_kwargs = {"batch_size": batch_size} if batch_size else {}
        super().__init__(file_uris=file_uris, seed=seed)

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        # Need to first split the work based on worker information
        current_worker_file_uris_to_process = self._iterator_init()
        assert self._file_loader is not None, "File loader not initialized"

        for file in current_worker_file_uris_to_process:
            tfh = self._file_loader.load_to_temp_file(
                file_uri_src=UriFactory.create_uri(file), delete=True
            )
            parquet_file = pq.ParquetFile(tfh.name)

            for batch in parquet_file.iter_batches(**self._iter_batches_kwargs):
                df = batch.to_pandas(
                    split_blocks=True, self_destruct=True
                )  # Fast, memory-friendly
                for row in df.itertuples(index=False, name=None):
                    yield dict(zip(df.columns, row))


class BigQueryIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        table: str,  # Format: "project.dataset.table"
        random_column: str,
        project: Optional[str] = None,
        selected_fields=None,
    ):
        """
        Enables reading from a BigQuery table in a sharded manner.
        This is done by using a random column to split the data into bins
        based on the number of workers in the global dataloading process id.

        The dataset is read in a sharded manner, where each worker reads a specific
        range of rows designated by conditions on the random column.
        The random column is used to ensure that the data is evenly distributed
        across the workers.

        Args:
            table (str): BigQuery table in the format "project.dataset.table"
            random_column (str): Column name used for random sampling.  Used to ensure sharded reading of data.
            project (Optional[str]): Project ID if not included in the table string
            selected_fields (Optional[List[str]]): List of fields to select from the table
        """

        self.project = f"projects/{project}" if project else None
        self.table = table
        self.selected_fields = selected_fields or []
        if self.selected_fields and (random_column not in self.selected_fields):
            self.selected_fields.append(random_column)
        self.random_column = random_column

    def _create_read_session(
        self, client: BigQueryReadClient, row_restriction: str = ""
    ):
        project, dataset, table = self.table.split(".")
        table_path = f"projects/{project}/datasets/{dataset}/tables/{table}"

        read_options = ReadSession.TableReadOptions(
            selected_fields=self.selected_fields,
            row_restriction=row_restriction,
        )

        session = ReadSession(
            table=table_path,
            data_format=DataFormat.ARROW,
            read_options=read_options,
        )

        return client.create_read_session(
            parent=self.project,
            read_session=session,
            max_stream_count=1,
        )

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        client = BigQueryReadClient()

        worker_info: Optional[WorkerInfo] = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        global_worker_id = (get_rank() * num_workers) + worker_id
        global_num_workers = num_workers * get_world_size()

        bin_width = 1.0 / global_num_workers
        bin_start, bin_end = (
            global_worker_id * bin_width,
            (global_worker_id + 1) * bin_width,
        )
        row_restriction = f"row_id BETWEEN {bin_start} AND {bin_end}"

        session = self._create_read_session(
            client=client, row_restriction=row_restriction
        )
        stream = session.streams[0].name
        reader = client.read_rows(stream)
        rows = reader.rows(session)

        for row in rows:
            yield {key: value.as_py() for key, value in row.items()}


class GcsJSONLHeterogeneousGraphIterableDataset(GcsJSONLIterableDataset):
    def __init__(
        self,
        file_uris: List[GcsUri],
        src_field: str = SRC_FIELD,
        dst_field: str = DST_FIELD,
        condensed_edge_type_field: str = CONDENSED_EDGE_TYPE_FIELD,
        seed: int = 42,
    ) -> None:
        self._src_field = src_field
        self._dst_field = dst_field
        self._condensed_edge_type_field = condensed_edge_type_field
        super().__init__(file_uris=file_uris, seed=seed)

    def __iter__(self) -> Iterator[HeterogeneousGraphEdgeDict]:
        for data in super().__iter__():
            # Convert the data to a filtered dictionary with just essential keys.
            yield HeterogeneousGraphEdgeDict(
                src=data[self._src_field],
                dst=data[self._dst_field],
                condensed_edge_type=data[self._condensed_edge_type_field],
            )


class GcsParquetHeterogeneousGraphIterableDataset(GcsParquetIterableDataset):
    def __init__(
        self,
        file_uris: List[GcsUri],
        src_field: str = SRC_FIELD,
        dst_field: str = DST_FIELD,
        condensed_edge_type_field: str = CONDENSED_EDGE_TYPE_FIELD,
        seed: int = 42,
    ) -> None:
        self._src_field = src_field
        self._dst_field = dst_field
        self._condensed_edge_type_field = condensed_edge_type_field
        super().__init__(file_uris=file_uris, seed=seed)

    def __iter__(self) -> Iterator[HeterogeneousGraphEdgeDict]:
        for data in super().__iter__():
            yield HeterogeneousGraphEdgeDict(
                src=data[self._src_field],
                dst=data[self._dst_field],
                condensed_edge_type=data[self._condensed_edge_type_field],
            )


class BigQueryHeterogeneousGraphIterableDataset(BigQueryIterableDataset):
    def __init__(
        self,
        table: str,
        random_column: str,
        project: Optional[str] = None,
        src_field: str = SRC_FIELD,
        dst_field: str = DST_FIELD,
        condensed_edge_type_field: str = CONDENSED_EDGE_TYPE_FIELD,
        **kwargs,
    ) -> None:
        self._src_field = src_field
        self._dst_field = dst_field
        self._condensed_edge_type_field = condensed_edge_type_field
        super().__init__(
            table=table,
            project=project,
            random_column=random_column,
            selected_fields=[src_field, dst_field, condensed_edge_type_field],
            **kwargs,
        )

    def __iter__(self) -> Iterator[HeterogeneousGraphEdgeDict]:
        for row in super().__iter__():
            # Convert the data to a filtered dictionary with just essential keys.
            yield HeterogeneousGraphEdgeDict(
                src=row[self._src_field],
                dst=row[self._dst_field],
                condensed_edge_type=row[self._condensed_edge_type_field],
            )
