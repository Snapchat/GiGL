from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

import google.cloud.bigquery as bigquery

DEFAULT_NODE_ID_FIELD = "node_id"
DEFAULT_EMBEDDING_FIELD = "emb"
DEFAULT_PREDICTION_FIELD = "pred"


class InferenceOutputBigqueryTableSchema(NamedTuple):
    """Thin container for inference output asset metadata
    which enables us to build and produce schemas to be fed into
    beam.io.WriteToBigQuery.  Enables us to track the node
    identifier, which assists during de-enumeration.
    """

    schema: Optional[Dict[str, List[Dict[str, str]]]] = None
    node_field: Optional[str] = None


class InferenceOutputBigqueryTableSchemaBuilder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._fields: Dict[str, bigquery.SchemaField] = dict()
        self._node_field: Optional[str] = None

    def add_field(
        self, name: str, field_type: str, mode: str
    ) -> InferenceOutputBigqueryTableSchemaBuilder:
        self._fields[name] = bigquery.SchemaField(
            name=name, field_type=field_type, mode=mode
        )
        return self

    def register_node_field(
        self, name: str
    ) -> InferenceOutputBigqueryTableSchemaBuilder:
        assert name in self._fields, f"Could not find field {name} in output fields."
        self._node_field = name
        return self

    def _build_schema_property(self) -> Dict[str, List[Dict[str, str]]]:
        schema_fields = [
            {"name": field.name, "type": field.field_type, "mode": field.mode}
            for field in self._fields.values()
        ]
        table_schema = {"fields": schema_fields}
        return table_schema

    def build(self) -> InferenceOutputBigqueryTableSchema:
        assert (
            self._node_field is not None
        ), "Node field must be defined before building."
        assert self._fields is not None, "_fields must be defined before building."

        schema = InferenceOutputBigqueryTableSchema(
            schema=self._build_schema_property(), node_field=self._node_field
        )
        self.reset()
        return schema


def _build_default_table_schema(
    field: str, should_run_unenumeration: bool = False
) -> InferenceOutputBigqueryTableSchema:
    builder = InferenceOutputBigqueryTableSchemaBuilder()

    if should_run_unenumeration:
        builder.add_field(
            name=DEFAULT_NODE_ID_FIELD, field_type="STRING", mode="REQUIRED"
        )
    else:
        builder.add_field(
            name=DEFAULT_NODE_ID_FIELD, field_type="INTEGER", mode="REQUIRED"
        )

    if field == DEFAULT_EMBEDDING_FIELD:
        builder.add_field(
            name=DEFAULT_EMBEDDING_FIELD, field_type="FLOAT", mode="REPEATED"
        )
    elif field == DEFAULT_PREDICTION_FIELD:
        builder.add_field(
            name=DEFAULT_PREDICTION_FIELD, field_type="INTEGER", mode="REQUIRED"
        )
    else:
        raise ValueError(
            f"Expected field to be one of {DEFAULT_EMBEDDING_FIELD, DEFAULT_PREDICTION_FIELD}, got {field}"
        )

    builder.register_node_field(name=DEFAULT_NODE_ID_FIELD)
    schema = builder.build()
    return schema


DEFAULT_EMBEDDINGS_TABLE_SCHEMA = _build_default_table_schema(
    field=DEFAULT_EMBEDDING_FIELD, should_run_unenumeration=False
)
DEFAULT_PREDICTIONS_TABLE_SCHEMA = _build_default_table_schema(
    field=DEFAULT_PREDICTION_FIELD, should_run_unenumeration=False
)

UNENUMERATED_EMBEDDINGS_TABLE_SCHEMA = _build_default_table_schema(
    field=DEFAULT_EMBEDDING_FIELD, should_run_unenumeration=True
)
UNENUMERATED_PREDICTIONS_TABLE_SCHEMA = _build_default_table_schema(
    field=DEFAULT_PREDICTION_FIELD, should_run_unenumeration=True
)
