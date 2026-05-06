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
from apache_beam.io.gcp.bigquery import BigQueryQueryPriority
from apache_beam.io.gcp.internal.clients.bigquery import DatasetReference
from apache_beam.pvalue import PBegin, PCollection
from apache_beam.transforms.window import GlobalWindow
from apache_beam.utils.windowed_value import WindowedValue
from tensorflow_metadata.proto.v0 import statistics_pb2

from gigl.common import Uri
from gigl.common.beam.sharded_read import BigQueryShardedReadConfig

_DEFAULT_BQ_READ_BATCH_SIZE = 1000

# Frozen at module load so unit tests that patch ``beam.io.ReadFromBigQuery``
# wholesale don't accidentally mask the ``Method`` enum value.
_BQ_READ_METHOD_EXPORT = beam.io.ReadFromBigQuery.Method.EXPORT


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
        stats_options: Optional ``tfdv.StatsOptions`` to configure
            slicing, schema-based hints, etc. When ``None``, TFDV uses
            its defaults (no slicing). Callers that need per-class /
            per-split TFDV stats wire ``slice_functions`` here.
    """

    def __init__(
        self,
        facets_report_uri: Uri,
        stats_output_uri: Uri,
        stats_options: Optional[tfdv.StatsOptions] = None,
    ):
        self.facets_report_uri = facets_report_uri
        self.stats_output_uri = stats_output_uri
        self.stats_options = stats_options

    def expand(
        self, features: PCollection[pa.RecordBatch]
    ) -> PCollection[statistics_pb2.DatasetFeatureStatisticsList]:
        if self.stats_options is not None:
            stats = features | "Generate TFDV statistics" >> tfdv.GenerateStatistics(
                options=self.stats_options
            )
        else:
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

    ``projection`` is a list of ``(column_name, sql_expression)`` pairs. Each
    pair renders as ``{sql_expression} AS \\`{column_name}\\``` in the
    SELECT. For plain scalar columns the pair is ``("age", "\\`age\\`")``;
    for derived columns (e.g. array hygiene companions) the expression is a
    full SQL fragment. The ``column_name`` is the identifier used downstream
    in the record batch and in TFDV stats.

    When ``sharded_read_config`` is provided, the read is split into
    ``num_shards`` parallel BQ queries with
    ``WHERE ABS(MOD(FARM_FINGERPRINT(CAST(shard_key AS STRING)), N)) = i``
    and ``Flatten``-ed back together. This mirrors
    :class:`gigl.common.beam.sharded_read.ShardedExportRead` and avoids the
    "single giant export" pattern that hangs Dataflow's ``SplitWithSizing``
    on very large tables (oversized status update payloads, slow GCS Avro
    reads). Without it, the read goes through a single
    ``beam.io.ReadFromBigQuery``.

    Args:
        bq_table: Fully qualified ``project.dataset.table`` reference.
        projection: ``(column_name, sql_expression)`` pairs to SELECT.
        batch_size: Rows per emitted ``RecordBatch``. Defaults to 1000.
        bq_project: Optional GCP project to bill the read against. Defaults to
            the project inferred by ``beam.io.ReadFromBigQuery``.
        sharded_read_config: Optional sharded read config. When set, fans the
            read into ``num_shards`` parallel ``EXPORT``-method reads keyed
            on ``shard_key``.
    """

    def __init__(
        self,
        bq_table: str,
        projection: list[tuple[str, str]],
        batch_size: int = _DEFAULT_BQ_READ_BATCH_SIZE,
        bq_project: Optional[str] = None,
        sharded_read_config: Optional[BigQueryShardedReadConfig] = None,
    ):
        if not projection:
            raise ValueError(
                f"BqTableToRecordBatch requires at least one projected column "
                f"for table {bq_table!r}"
            )
        if sharded_read_config is not None and sharded_read_config.num_shards <= 0:
            raise ValueError(
                f"sharded_read_config.num_shards must be > 0, got "
                f"{sharded_read_config.num_shards}"
            )
        self.bq_table = bq_table
        self.projection = projection
        self.batch_size = batch_size
        self.bq_project = bq_project
        self.sharded_read_config = sharded_read_config

    def expand(self, pbegin: PBegin) -> PCollection[pa.RecordBatch]:
        if not isinstance(pbegin, PBegin):
            raise TypeError(
                f"Input to {BqTableToRecordBatch.__name__} transform must be "
                f"a PBegin but found {pbegin})"
            )
        column_list = ", ".join(f"{expr} AS `{name}`" for name, expr in self.projection)
        column_names = [name for name, _ in self.projection]

        if self.sharded_read_config is not None:
            rows = self._sharded_read(pbegin, column_list)
        else:
            rows = self._single_read(pbegin, column_list)

        return rows | "Buffer rows and emit record batches" >> beam.ParDo(
            _RowsToRecordBatchDoFn(
                batch_size=self.batch_size,
                feature_columns=column_names,
            )
        )

    def _single_read(self, pbegin: PBegin, column_list: str) -> PCollection[dict]:
        query = f"SELECT {column_list} FROM `{self.bq_table}`"
        read_kwargs: dict = {
            "query": query,
            "use_standard_sql": True,
        }
        if self.bq_project is not None:
            read_kwargs["project"] = self.bq_project
        return pbegin | "Read feature rows from BQ" >> beam.io.ReadFromBigQuery(
            **read_kwargs
        )

    def _sharded_read(self, pbegin: PBegin, column_list: str) -> PCollection[dict]:
        # ABS(MOD(FARM_FINGERPRINT(...), N)) = i, mirroring ShardedExportRead.
        # MOD is taken before ABS because ABS errors on the largest negative
        # INT64; doing it in this order keeps every shard index in
        # [0, num_shards-1].
        config = self.sharded_read_config
        assert config is not None  # for mypy; guarded by the caller branch.
        temp_dataset = DatasetReference(
            projectId=config.project_id, datasetId=config.temp_dataset_name
        )
        per_shard: list[PCollection[dict]] = []
        for i in range(config.num_shards):
            query = (
                f"SELECT {column_list} FROM `{self.bq_table}` "
                f"WHERE ABS(MOD(FARM_FINGERPRINT(CAST({config.shard_key} AS STRING)), "
                f"{config.num_shards})) = {i}"
            )
            read_kwargs: dict = {
                "query": query,
                "use_standard_sql": True,
                "method": _BQ_READ_METHOD_EXPORT,
                "query_priority": BigQueryQueryPriority.INTERACTIVE,
                "temp_dataset": temp_dataset,
            }
            if self.bq_project is not None:
                read_kwargs["project"] = self.bq_project
            per_shard.append(
                pbegin
                | f"Read feature rows from BQ shard {i}/{config.num_shards}"
                >> beam.io.ReadFromBigQuery(**read_kwargs)
            )
        return per_shard | "Flatten BQ shards" >> beam.Flatten()
