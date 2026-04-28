"""Shared TFDV / Beam PTransforms usable by the data preprocessor and analytics.

Exposes:
  * ``GenerateAndVisualizeStats`` - Runs ``tfdv.GenerateStatistics`` over a
    ``PCollection[pa.RecordBatch]`` and writes both a Facets HTML
    visualization and a TFDV stats TFRecord.
  * ``BqTableToRecordBatch`` - Reads the given columns from a BigQuery table
    and emits ``PCollection[pa.RecordBatch]`` suitable for TFDV. Schema is
    inferred from row values; no pre-declared TFDV schema is required.
"""

from typing import Iterable, Optional

import apache_beam as beam
import pyarrow as pa
import tensorflow_data_validation as tfdv
from apache_beam.pvalue import PBegin, PCollection
from apache_beam.transforms.window import GlobalWindow
from apache_beam.utils.windowed_value import WindowedValue
from tensorflow_metadata.proto.v0 import statistics_pb2

from gigl.common import Uri

_DEFAULT_BQ_READ_BATCH_SIZE = 1000


class GenerateAndVisualizeStats(beam.PTransform):
    """Generate TFDV statistics and a Facets HTML visualization from a record
    batch ``PCollection``.

    Writes two side-effect outputs:
      * A single-shard Facets HTML file at ``facets_report_uri``.
      * A TFRecord of ``DatasetFeatureStatisticsList`` at ``stats_output_uri``.

    Args:
        facets_report_uri: URI for the Facets HTML visualization (typically
            a ``GcsUri``; local ``LocalUri`` is also accepted for tests).
        stats_output_uri: URI (file prefix) for the TFDV stats TFRecord.
    """

    def __init__(self, facets_report_uri: Uri, stats_output_uri: Uri):
        self.facets_report_uri = facets_report_uri
        self.stats_output_uri = stats_output_uri

    def expand(
        self, features: PCollection[pa.RecordBatch]
    ) -> PCollection[statistics_pb2.DatasetFeatureStatisticsList]:
        stats = features | "Generate TFDV statistics" >> tfdv.GenerateStatistics()

        _ = (
            stats
            | "Generate stats visualization"
            >> beam.Map(tfdv.utils.display_util.get_statistics_html)
            | "Write stats Facets report HTML"
            >> beam.io.WriteToText(
                self.facets_report_uri.uri, num_shards=1, shard_name_template=""
            )
        )

        _ = (
            stats
            | "Write TFDV stats output TFRecord"
            >> tfdv.WriteStatisticsToTFRecord(self.stats_output_uri.uri)
        )

        return stats


class _RowsToRecordBatchDoFn(beam.DoFn):
    """Buffer incoming row dicts and emit ``pa.RecordBatch`` batches.

    Each output column is encoded as an Arrow list-typed column
    (``list<T>``) with NULLs mapped to Arrow nulls, matching TFDV's
    expectation that each feature column be a ``(Large)List<primitive|struct>``
    (or null). See ``tfdv.utils.stats_util.get_feature_type_from_arrow_type``.
    """

    def __init__(self, batch_size: int, feature_columns: list[str]):
        self._batch_size = batch_size
        self._feature_columns = feature_columns
        self._buffer: list[dict] = []

    def start_bundle(self) -> None:
        self._buffer = []

    def process(self, element: dict) -> Iterable[pa.RecordBatch]:
        self._buffer.append(element)
        if len(self._buffer) >= self._batch_size:
            yield self._drain()

    def finish_bundle(self) -> Iterable[WindowedValue]:
        if self._buffer:
            yield WindowedValue(
                value=self._drain(),
                timestamp=0,
                windows=(GlobalWindow(),),
            )

    def _drain(self) -> pa.RecordBatch:
        buffered = self._buffer
        self._buffer = []
        column_values: dict[str, list] = {col: [] for col in self._feature_columns}
        for row in buffered:
            for col in self._feature_columns:
                value = row[col]
                column_values[col].append(None if value is None else [value])
        return pa.RecordBatch.from_pydict(
            {col: pa.array(values) for col, values in column_values.items()}
        )


class BqTableToRecordBatch(beam.PTransform):
    """Read selected columns from a BigQuery table and emit Arrow record batches.

    The output is a ``PCollection[pa.RecordBatch]`` whose columns are Arrow
    list-typed (``list<T>``), which is the shape TFDV expects. Schema is
    inferred from row values; rows with NULL values are represented as Arrow
    nulls (missing features).

    Args:
        bq_table: Fully qualified ``project.dataset.table`` reference.
        feature_columns: Columns to select; also the columns exposed to TFDV.
        batch_size: Rows per emitted ``RecordBatch``. Defaults to 1000.
        bq_project: Optional GCP project to bill the read against. Defaults to
            the project inferred by ``beam.io.ReadFromBigQuery``.
    """

    def __init__(
        self,
        bq_table: str,
        feature_columns: list[str],
        batch_size: int = _DEFAULT_BQ_READ_BATCH_SIZE,
        bq_project: Optional[str] = None,
    ):
        if not feature_columns:
            raise ValueError(
                f"BqTableToRecordBatch requires at least one feature column "
                f"for table {bq_table!r}"
            )
        self.bq_table = bq_table
        self.feature_columns = feature_columns
        self.batch_size = batch_size
        self.bq_project = bq_project

    def expand(self, pbegin: PBegin) -> PCollection[pa.RecordBatch]:
        if not isinstance(pbegin, PBegin):
            raise TypeError(
                f"Input to {BqTableToRecordBatch.__name__} transform must be "
                f"a PBegin but found {pbegin})"
            )
        column_list = ", ".join(f"`{c}`" for c in self.feature_columns)
        query = f"SELECT {column_list} FROM `{self.bq_table}`"
        read_kwargs: dict = {
            "query": query,
            "use_standard_sql": True,
        }
        if self.bq_project is not None:
            read_kwargs["project"] = self.bq_project
        return (
            pbegin
            | "Read feature rows from BQ" >> beam.io.ReadFromBigQuery(**read_kwargs)
            | "Buffer rows and emit record batches"
            >> beam.ParDo(
                _RowsToRecordBatchDoFn(
                    batch_size=self.batch_size,
                    feature_columns=self.feature_columns,
                )
            )
        )
