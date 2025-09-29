from typing import Optional, cast

import apache_beam as beam
from apache_beam.io.gcp.bigquery import BigQueryQueryPriority
from apache_beam.io.gcp.internal.clients.bigquery import DatasetReference
from apache_beam.pvalue import PBegin

from gigl.env.pipelines_config import get_resource_config
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import InstanceDictPTransform


class _SlicedExportRead(beam.PTransform):
    def __init__(self, table_name: str, partition_key: str, **kwargs):
        super().__init__()
        self._table_name = table_name
        # TODO (mkolodner-sc): Infer this value based on number of rows
        self._num_slices = 20
        self._partition_key = partition_key
        resource_config = get_resource_config()
        self._temp_dataset_reference = DatasetReference(
            projectId=resource_config.project,
            datasetId=resource_config.temp_assets_bq_dataset_name,
        )
        self._kwargs = kwargs

    def expand(self, pbegin: PBegin):
        pcollection_list = []
        for i in range(self._num_slices):
            query = (
                f"SELECT * FROM `{self._table_name}` "
                f"WHERE MOD(ABS(FARM_FINGERPRINT(CAST({self._partition_key} AS STRING))), {self._num_slices}) = {i}"
            )

            pcollection_list.append(
                pbegin
                | f"Read slice {i}/{self._num_slices}"
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


def _get_bigquery_ptransform(
    table_name: str, partition_key: Optional[str], *args, **kwargs
) -> InstanceDictPTransform:
    table_name = table_name.replace(".", ":", 1)  # sanitize table name
    if partition_key is not None:
        return cast(
            InstanceDictPTransform,
            _SlicedExportRead(
                table_name=table_name, partition_key=partition_key, **kwargs
            ),
        )
    else:
        return cast(
            InstanceDictPTransform,
            beam.io.ReadFromBigQuery(
                table=table_name,
                method=beam.io.ReadFromBigQuery.Method.EXPORT,  # type: ignore
                *args,
                **kwargs,
            ),
        )


# Below type ignores are due to mypy star expansion issues: https://github.com/python/mypy/issues/6799


class BigqueryNodeDataReference(NodeDataReference):
    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        return _get_bigquery_ptransform(table_name=self.reference_uri, partition_key=self.partition_key, *args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        return f"BigqueryNodeDataReference(node_type={self.node_type}, identifier={self.identifier}, reference_uri={self.reference_uri}, partition_key={self.partition_key})"


class BigqueryEdgeDataReference(EdgeDataReference):
    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        return _get_bigquery_ptransform(table_name=self.reference_uri, partition_key=self.partition_key, *args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        return f"BigqueryEdgeDataReference(edge_type={self.edge_type}, src_identifier={self.src_identifier}, dst_identifier={self.dst_identifier}, reference_uri={self.reference_uri}, partition_key={self.partition_key})"
