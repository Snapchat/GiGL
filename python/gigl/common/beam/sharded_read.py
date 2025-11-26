from dataclasses import dataclass

import apache_beam as beam
from apache_beam.io.gcp.bigquery import BigQueryQueryPriority
from apache_beam.io.gcp.internal.clients.bigquery import DatasetReference
from apache_beam.pvalue import PBegin
from gigl.common.logger import Logger
from google.cloud import bigquery

logger = Logger()


@dataclass(frozen=True)
class BigQueryShardedReadConfig:
    # The key in the table that we will use to split the data into shards. This should be used if we are operating on
    # very large tables, in which case we want to only read smaller slices of the table at a time to avoid oversized status update
    # payloads.
    shard_key: str
    # The project ID to use for temporary datasets when running sharded reads.
    project_id: str
    # The temporary bigquery dataset name to use when running sharded reads.
    temp_dataset_name: str
    # The number of shards to split the data into. If not provided, the table will be shareded with a default
    # value of 20 shards.
    # TODO (mkolodner-sc): Instead of using this default, infer this value based on number of rows in table
    num_shards: int = 20


def _assert_shard_key_in_table(table_name: str, shard_key: str) -> None:
    """
    Validate that the shard key is a valid column in the BigQuery table.
    """
    client = bigquery.Client()
    table_ref = bigquery.TableReference.from_string(table_name)

    table = client.get_table(table_ref)
    column_names = [field.name for field in table.schema]

    if shard_key not in column_names:
        raise ValueError(
            f"Shard key '{shard_key}' is not a valid column in table '{table_name}'. "
            f"Available columns: {column_names}"
        )


class ShardedExportRead(beam.PTransform):
    def __init__(
        self,
        table_name: str,
        sharded_read_info: BigQueryShardedReadConfig,
        **kwargs,
    ):
        super().__init__()
        self._table_name: str = table_name
        self._num_shards: int = sharded_read_info.num_shards
        if self._num_shards <= 0:
            raise ValueError(
                f"Number of shards specified must be greater than 0, got {self._num_shards}"
            )
        self._shard_key: str = sharded_read_info.shard_key
        self._temp_dataset_reference: DatasetReference = DatasetReference(
            projectId=sharded_read_info.project_id,
            datasetId=sharded_read_info.temp_dataset_name,
        )
        self._kwargs = kwargs
        logger.info(
            f"Got ShardedExportRead arguments table_name={table_name}, sharded_read_info={sharded_read_info}, kwargs={kwargs}"
        )

        _assert_shard_key_in_table(self._table_name, self._shard_key)

    def expand(self, pbegin: PBegin):
        pcollection_list = []
        for i in range(self._num_shards):
            # We use farm_fingerprint as a determinstic hashing function which will allow us to shard
            # on keys of any type (i.e. strings, integers, etc.) We take the MOD on the returned INT64 value first
            # with the number of shards and then take the ABS value to ensure it is in range [0, num_shards-1].
            # We do it in this order since ABS can error on the largest negative INT64 number, which has no positive equivalent.
            query = (
                f"SELECT * FROM `{self._table_name}` "
                f"WHERE ABS(MOD(FARM_FINGERPRINT(CAST({self._shard_key} AS STRING)), {self._num_shards})) = {i}"
            )

            pcollection_list.append(
                pbegin
                | f"Read slice {i}/{self._num_shards}"
                >> beam.io.ReadFromBigQuery(
                    query=query,
                    use_standard_sql=True,
                    method=beam.io.ReadFromBigQuery.Method.EXPORT,
                    query_priority=BigQueryQueryPriority.INTERACTIVE,
                    temp_dataset=self._temp_dataset_reference,
                    **self._kwargs,
                )
            )

        return pcollection_list | "Flatten shards" >> beam.Flatten()
