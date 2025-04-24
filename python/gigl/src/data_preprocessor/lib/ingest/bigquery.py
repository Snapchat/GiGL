from typing import cast

import apache_beam as beam

from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import InstanceDictPTransform


def _get_bigquery_ptransform(
    table_name: str, *args, **kwargs
) -> InstanceDictPTransform:
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


class BigqueryNodeDataReference(NodeDataReference):
    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        return _get_bigquery_ptransform(table_name=self.reference_uri, *args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        return f"BigqueryNodeDataReference(node_type={self.node_type}, identifier={self.identifier}, reference_uri={self.reference_uri})"


class BigqueryEdgeDataReference(EdgeDataReference):
    def yield_instance_dict_ptransform(self, *args, **kwargs) -> InstanceDictPTransform:
        return _get_bigquery_ptransform(table_name=self.reference_uri, *args, **kwargs)  # type: ignore

    def __repr__(self) -> str:
        return f"BigqueryEdgeDataReference(edge_type={self.edge_type}, src_identifier={self.src_identifier}, dst_identifier={self.dst_identifier}, reference_uri={self.reference_uri})"
