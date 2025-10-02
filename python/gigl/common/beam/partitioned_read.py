from dataclasses import dataclass, field

import apache_beam as beam
from apache_beam.io.gcp.bigquery import BigQueryQueryPriority
from apache_beam.io.gcp.internal.clients.bigquery import DatasetReference
from apache_beam.pvalue import PBegin

from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config

logger = Logger()


@dataclass(frozen=True)
class BigQueryPartitionedReadInfo:
    # The key in the table that we will use to split the data into partitions. This should be used if we are operating on
    # very large tables, in which case we want to only read smaller slices of the table at a time to avoid oversized status update
    # payloads.
    partition_key: str
    # The number of partitions to split the data into. If not provided, the table will be partitioned with a default
    # value of 20 partitions.
    # TODO (mkolodner-sc): Instead of using this default, infer this value based on number of rows in table
    num_partitions: int = 20
    # The project ID to use for temporary datasets when running partitioned reads. If not provided, will default
    # to use the project specified in the resource config environment variables.
    project_id: str = field(default_factory=lambda: get_resource_config().project)
    # The temporary bigquery dataset name to use when running partitioned reads. If not provided, will default
    # to use the temp assets dataset name specified in the resource config environment variables.
    temp_dataset_name: str = field(
        default_factory=lambda: get_resource_config().temp_assets_bq_dataset_name
    )


class PartitionedExportRead(beam.PTransform):
    def __init__(
        self,
        table_name: str,
        partitioned_read_info: BigQueryPartitionedReadInfo,
        **kwargs,
    ):
        super().__init__()
        self._table_name: str = table_name
        self._num_partitions: int = partitioned_read_info.num_partitions
        if self._num_partitions <= 0:
            raise ValueError(
                f"Number of partitions specified must be greater than 0, got {self._num_partitions}"
            )
        self._partition_key: str = partitioned_read_info.partition_key
        self._temp_dataset_reference: DatasetReference = DatasetReference(
            projectId=partitioned_read_info.project_id,
            datasetId=partitioned_read_info.temp_dataset_name,
        )
        self._kwargs = kwargs

    def expand(self, pbegin: PBegin):
        pcollection_list = []
        for i in range(self._num_partitions):
            # We use farm_fingerprint as a determinstic hashing function which will allow us to partition
            # on keys of any type (i.e. strings, integers, etc.) We take the MOD on the returned INT64 value first
            # with the number of partitions and then take the ABS value to ensure it is in range [0, num_partitions-1].
            # We do it in this order since ABS can error on the largest negative INT64 number, which has no positive equivalent.
            query = (
                f"SELECT * FROM `{self._table_name}` "
                f"WHERE ABS(MOD(FARM_FINGERPRINT(CAST({self._partition_key} AS STRING)), {self._num_partitions}) = {i})"
            )

            pcollection_list.append(
                pbegin
                | f"Read slice {i}/{self._num_partitions}"
                >> beam.io.ReadFromBigQuery(
                    query=query,
                    use_standard_sql=True,
                    method=beam.io.ReadFromBigQuery.Method.EXPORT,
                    query_priority=BigQueryQueryPriority.INTERACTIVE,
                    temp_dataset=self._temp_dataset_reference,
                    **self._kwargs,
                )
            )

        return pcollection_list | "Flatten slices" >> beam.Flatten()
