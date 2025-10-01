from dataclasses import dataclass
from typing import Optional, cast

import apache_beam as beam

from gigl.common.beam.partitioned_read import (
    BigQueryPartitionedReadInfo,
    PartitionedExportRead,
)
from gigl.common.logger import Logger
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import InstanceDictPTransform

logger = Logger()


def _get_bigquery_ptransform(
    table_name: str,
    partitioned_read_info: Optional[BigQueryPartitionedReadInfo] = None,
    *args,
    **kwargs,
) -> InstanceDictPTransform:
    if partitioned_read_info is not None:
        # For partitioned reads, use the table name as-is (with dots) for SQL queries
        return cast(
            InstanceDictPTransform,
            PartitionedExportRead(
                table_name=table_name,
                partitioned_read_info=partitioned_read_info,
                **kwargs,
            ),
        )
    else:
        # For regular reads, sanitize table name (convert first dot to colon)
        table_name = table_name.replace(".", ":", 1)  # sanitize table name
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
@dataclass(frozen=True)
class BigqueryNodeDataReference(NodeDataReference):
    partitioned_read_info: Optional[BigQueryPartitionedReadInfo] = None

    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        return _get_bigquery_ptransform(table_name=self.reference_uri, partitioned_read_info=self.partitioned_read_info, *args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        return f"BigqueryNodeDataReference(node_type={self.node_type}, identifier={self.identifier}, reference_uri={self.reference_uri}, partitioned_read_info={self.partitioned_read_info})"


@dataclass(frozen=True)
class BigqueryEdgeDataReference(EdgeDataReference):
    partitioned_read_info: Optional[BigQueryPartitionedReadInfo] = None

    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        return _get_bigquery_ptransform(table_name=self.reference_uri, partitioned_read_info=self.partitioned_read_info, *args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        return f"BigqueryEdgeDataReference(edge_type={self.edge_type}, src_identifier={self.src_identifier}, dst_identifier={self.dst_identifier}, reference_uri={self.reference_uri}, partitioned_read_info={self.partitioned_read_info})"
