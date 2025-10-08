from dataclasses import dataclass
from typing import Optional, cast

import apache_beam as beam

from gigl.common.beam.sharded_read import BigQueryShardedReadConfig, ShardedExportRead
from gigl.common.logger import Logger
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import InstanceDictPTransform

logger = Logger()


def _get_bigquery_ptransform(
    table_name: str,
    sharded_read_config: Optional[BigQueryShardedReadConfig] = None,
    *args,
    **kwargs,
) -> InstanceDictPTransform:
    if sharded_read_config is not None:
        # For sharded reads, use the table name as-is (with dots) for SQL queries
        return cast(
            InstanceDictPTransform,
            ShardedExportRead(
                table_name=table_name,
                sharded_read_info=sharded_read_config,
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
    """
    Data reference for running enumeration on node data in BigQuery.
    We provide the ability to perform sharded reads using the sharded_read_config field, where the input table
    is split into smaller shards and each shard is read separately.
    This is useful for large tables that would otherwise cause oversized status update payloads, leading to job failures.
    The sharded_read_config field is optional and if not provided, the table will not be sharded
    and will be read in one ReadFromBigQuery call. General guidance is to use 10-30 shards for large tables, but may need
    tuning depending on the table size.

    Args:
        reference_uri (str): BigQuery table URI for the node data.
        node_type (NodeType): Node type for the current reference
        identifier (Optional[str]): Identifier for the node. This field is overridden by the identifier
            from the corresponding node data preprocessing spec.
        sharded_read_config (Optional[BigQueryShardedReadConfig]): Configuration for performing sharded reads for the node table.
            If not provided, the table will not be sharded and will be read in one ReadFromBigQuery call.
    """

    sharded_read_config: Optional[BigQueryShardedReadConfig] = None

    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        return _get_bigquery_ptransform(table_name=self.reference_uri, sharded_read_config=self.sharded_read_config, *args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        return f"BigqueryNodeDataReference(node_type={self.node_type}, identifier={self.identifier}, reference_uri={self.reference_uri}, sharded_read_config={self.sharded_read_config})"


@dataclass(frozen=True)
class BigqueryEdgeDataReference(EdgeDataReference):
    """
    Data reference for running enumeration on edge data in BigQuery.
    We provide the ability to perform sharded reads using the sharded_read_config field, where the input table
    is split into smaller shards and each shard is read separately.
    This is useful for large tables that would otherwise cause oversized status update payloads, leading to job failures.
    The sharded_read_config field is optional and if not provided, the table will not be sharded
    and will be read in one ReadFromBigQuery call. General guidance is to use 10-30 shards for large tables, but may need
    tuning depending on the table size.
    Args:
        reference_uri (str): BigQuery table URI for the edge data.
        edge_type (EdgeType): Edge type for the current reference
        edge_usage_type (EdgeUsageType): Edge usage type for the current reference. Defaults to EdgeUsageType.MAIN.
        src_identifier (Optional[str]): Identifier for the source node. This field is overridden by the src identifier
            from the corresponding edge data preprocessing spec.
        dst_identifier (Optional[str]): Identifier for the destination node. This field is overridden by the dst identifier
            from the corresponding edge data preprocessing spec.
        sharded_read_config (Optional[BigQueryShardedReadConfig]): Configuration for performing sharded reads for the edge table.
            If not provided, the table will not be sharded and will be read in one ReadFromBigQuery call.

    """

    sharded_read_config: Optional[BigQueryShardedReadConfig] = None

    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        return _get_bigquery_ptransform(table_name=self.reference_uri, sharded_read_config=self.sharded_read_config, *args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        return f"BigqueryEdgeDataReference(edge_type={self.edge_type}, src_identifier={self.src_identifier}, dst_identifier={self.dst_identifier}, reference_uri={self.reference_uri}, sharded_read_config={self.sharded_read_config})"
