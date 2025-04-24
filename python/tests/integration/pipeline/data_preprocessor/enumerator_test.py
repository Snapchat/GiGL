import unittest
from typing import Any, Dict, List, Tuple, Union

import google.cloud.bigquery as bigquery
import pandas as pd

import gigl.src.data_preprocessor.lib.enumerate.queries as enumeration_queries
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.constants.time import NODASH_DATETIME_FORMAT
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType, Relation
from gigl.src.common.utils.bq import BqUtils
from gigl.src.common.utils.time import current_formatted_datetime
from gigl.src.data_preprocessor.lib.enumerate.utils import (
    Enumerator,
    EnumeratorEdgeTypeMetadata,
    EnumeratorNodeTypeMetadata,
)
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)

logger = Logger()

# TODO: (svij-sc) Clean up the graph definition here; maybe using some changes
# Likely using something like: `python/tests/test_assets/dataset_mocking/visualization_test/visualize.py`
# Sample yaml that defines the graph: python/tests/test_assets/dataset_mocking/visualization_test/graph_config.yaml
_PERSON_NODE_TYPE = NodeType("person")
_MESSAGES_RELATION = Relation("messages")
_MESSAGES_EDGE_TYPE = EdgeType(
    src_node_type=_PERSON_NODE_TYPE,
    relation=_MESSAGES_RELATION,
    dst_node_type=_PERSON_NODE_TYPE,
)
# Define the nodes in the graph
_PERSON_NODES = ["Alice", "Bob", "Charlie"]
# Define the edges in the graph
_MESSAGE_EDGES = [("Alice", "Bob"), ("Bob", "Charlie")]
_POSITIVE_EDGES = [("Alice", "Charlie")]
_NEGATIVE_EDGES = [("Alice", "Alice"), ("Bob", "Bob"), ("Charlie", "Charlie")]


_PERSON_NODE_IDENTIFIER_FIELD = "person"
# Define node features rows for each node
_PERSON_NODE_FEATURE_FLOAT_FIELDS = ["height", "age", "weight"]
_PERSON_NODE_FEATURE_RECORDS: List[Dict[str, Any]] = [
    {
        _PERSON_NODE_IDENTIFIER_FIELD: node,
        "height": float(i),
        "age": float(i),
        "weight": float(i),
    }
    for i, node in enumerate(_PERSON_NODES)
]


_MESSAGES_EDGE_SRC_IDENTIFIER_FIELD = "from_preson"
_MESSAGES_EDGE_DST_IDENTIFIER_FIELD = "to_person"

# Define feature rows for each edge
_MESSAGE_EDGE_FEATURE_INT_FIELDS = ["is_friends_with"]
_MESSAGE_EDGE_FEATURE_RECORDS = [
    {
        _MESSAGES_EDGE_SRC_IDENTIFIER_FIELD: src,
        _MESSAGES_EDGE_DST_IDENTIFIER_FIELD: dst,
        "is_friends_with": 1,
    }
    for (src, dst) in _MESSAGE_EDGES
]
_POSITIVE_EDGE_FEATURE_INT_FIELDS = ["is_friends_with", "messages_every_day"]
_POSITIVE_EDGE_FEATURE_RECORDS = [
    {
        _MESSAGES_EDGE_SRC_IDENTIFIER_FIELD: src,
        _MESSAGES_EDGE_DST_IDENTIFIER_FIELD: dst,
        "is_friends_with": 1,
        "messages_every_day": 1,
    }
    for (src, dst) in _POSITIVE_EDGES
]
_NEGATIVE_EDGE_FEATURE_INT_FIELDS: List[str] = []
_NEGATIVE_EDGE_FEATURE_RECORDS = [
    {
        _MESSAGES_EDGE_SRC_IDENTIFIER_FIELD: src,
        _MESSAGES_EDGE_DST_IDENTIFIER_FIELD: dst,
    }
    for (src, dst) in _NEGATIVE_EDGES
]


# TODO: (svij-sc) Cleanup this test
class EnumeratorTest(unittest.TestCase):
    def __upload_records_to_bq(
        self,
        data_reference: Union[BigqueryEdgeDataReference, BigqueryNodeDataReference],
        records: List[Dict[str, Any]],
    ):
        self.__bq_utils.create_or_empty_bq_table(bq_path=data_reference.reference_uri)
        columns: List[str] = []
        schema: List[bigquery.SchemaField] = []
        for record in records[0].items():
            field_name, field_value = record
            columns.append(field_name)
            if isinstance(field_value, int):
                schema.append(
                    bigquery.SchemaField(field_name, bigquery.enums.SqlTypeNames.INT64)
                )
            elif isinstance(field_value, float):
                schema.append(
                    bigquery.SchemaField(
                        field_name, bigquery.enums.SqlTypeNames.FLOAT64
                    )
                )
            elif isinstance(field_value, str):
                schema.append(
                    bigquery.SchemaField(field_name, bigquery.enums.SqlTypeNames.STRING)
                )
            else:
                raise ValueError(
                    f"Unsupported type {type(field_value)} for field {field_name}"
                )

        df = pd.DataFrame(
            records,
            columns=columns,
        )
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE",
        )
        job = self.__client.load_table_from_dataframe(
            dataframe=df,
            destination=data_reference.reference_uri,
            job_config=job_config,
        )  # Make an API request.
        job.result()  # Wait for the job to complete.

    def setUp(self) -> None:
        self.__client = bigquery.Client()
        self.__bq_utils = BqUtils()
        current_timestamp = current_formatted_datetime(fmt=NODASH_DATETIME_FORMAT)
        self.__applied_task_identifier = f"enumerator_test_{current_timestamp}"

        project = get_resource_config().project
        temp_bq_dataset = get_resource_config().temp_assets_bq_dataset_name

        self.__input_nodes_data_reference = BigqueryNodeDataReference(
            reference_uri=BqUtils.join_path(
                project,
                temp_bq_dataset,
                f"node_features_{self.__applied_task_identifier}",
            ),
            node_type=_PERSON_NODE_TYPE,
            identifier=_PERSON_NODE_IDENTIFIER_FIELD,
        )

        self.__input_main_edges_data_reference = BigqueryEdgeDataReference(
            reference_uri=BqUtils.join_path(
                project,
                temp_bq_dataset,
                f"main_edge_features_{self.__applied_task_identifier}",
            ),
            edge_type=_MESSAGES_EDGE_TYPE,
            edge_usage_type=EdgeUsageType.MAIN,
            src_identifier=_MESSAGES_EDGE_SRC_IDENTIFIER_FIELD,
            dst_identifier=_MESSAGES_EDGE_DST_IDENTIFIER_FIELD,
        )
        self.__input_positive_edges_data_reference = BigqueryEdgeDataReference(
            reference_uri=BqUtils.join_path(
                project,
                temp_bq_dataset,
                f"positive_edge_features_{self.__applied_task_identifier}",
            ),
            edge_type=_MESSAGES_EDGE_TYPE,
            edge_usage_type=EdgeUsageType.POSITIVE,
            src_identifier=_MESSAGES_EDGE_SRC_IDENTIFIER_FIELD,
            dst_identifier=_MESSAGES_EDGE_DST_IDENTIFIER_FIELD,
        )
        self.__input_negative_edges_data_reference = BigqueryEdgeDataReference(
            reference_uri=BqUtils.join_path(
                project,
                temp_bq_dataset,
                f"negative_edge_features_{self.__applied_task_identifier}",
            ),
            edge_type=_MESSAGES_EDGE_TYPE,
            edge_usage_type=EdgeUsageType.NEGATIVE,
            src_identifier=_MESSAGES_EDGE_SRC_IDENTIFIER_FIELD,
            dst_identifier=_MESSAGES_EDGE_DST_IDENTIFIER_FIELD,
        )

        # Create the node preprocessing specs
        self.node_data_references = [
            self.__input_nodes_data_reference,
        ]
        # Create the edge preprocessing specs
        self.edge_data_references = [
            self.__input_main_edges_data_reference,
            self.__input_positive_edges_data_reference,
            self.__input_negative_edges_data_reference,
        ]

        self.__bq_tables_to_cleanup_on_teardown = [
            self.__input_nodes_data_reference.reference_uri,
            self.__input_main_edges_data_reference.reference_uri,
            self.__input_positive_edges_data_reference.reference_uri,
            self.__input_negative_edges_data_reference.reference_uri,
        ]

        self.__upload_records_to_bq(
            data_reference=self.__input_nodes_data_reference,
            records=_PERSON_NODE_FEATURE_RECORDS,
        )
        self.__upload_records_to_bq(
            data_reference=self.__input_main_edges_data_reference,
            records=_MESSAGE_EDGE_FEATURE_RECORDS,
        )
        self.__upload_records_to_bq(
            data_reference=self.__input_positive_edges_data_reference,
            records=_POSITIVE_EDGE_FEATURE_RECORDS,
        )
        self.__upload_records_to_bq(
            data_reference=self.__input_negative_edges_data_reference,
            records=_NEGATIVE_EDGE_FEATURE_RECORDS,
        )

    def __assert_bq_table_schema_contains_all_fields(
        self, table_name: str, expected_fields: List[str] = []
    ):
        logger.info(
            f"Asserting {table_name} has the following expected fields: {expected_fields}"
        )
        schema = self.__bq_utils.fetch_bq_table_schema(
            bq_table=table_name,
        )
        for field in expected_fields:
            self.assertIn(field, schema)

    def fetch_enumerated_node_map_and_assert_correctness(
        self, map_enum_node_type_metadata: Dict[NodeType, EnumeratorNodeTypeMetadata]
    ) -> Dict[int, str]:
        int_to_orig_node_id_map: Dict[int, str] = {}

        person_enumerated_node_type_metadata = map_enum_node_type_metadata[
            _PERSON_NODE_TYPE
        ]
        self.assertIsNotNone(person_enumerated_node_type_metadata)

        self.__assert_bq_table_schema_contains_all_fields(
            table_name=person_enumerated_node_type_metadata.bq_unique_node_ids_enumerated_table_name,
            expected_fields=[
                enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
                enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            ],
        )

        result = self.__bq_utils.run_query(
            query=(
                f"SELECT {enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD}, "
                f"{enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD} FROM "
                f"`{person_enumerated_node_type_metadata.bq_unique_node_ids_enumerated_table_name}`"
            ),
            labels=get_resource_config().get_resource_labels(),
        )
        num_rows = 0
        for row in result:
            num_rows += 1
            node_int_id = row[enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD]
            node_original_id = row[enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD]
            int_to_orig_node_id_map[node_int_id] = node_original_id
            self.assertLess(node_int_id, len(_PERSON_NODES))
        self.assertEqual(num_rows, len(_PERSON_NODES))

        return int_to_orig_node_id_map

    def assert_enumerated_node_features_correctness(
        self,
        int_to_orig_node_id_map: Dict[int, str],
        map_enum_node_type_metadata: Dict[NodeType, EnumeratorNodeTypeMetadata],
    ):
        person_enumerated_node_type_metadata = map_enum_node_type_metadata[
            _PERSON_NODE_TYPE
        ]

        node_id_field = _PERSON_NODE_IDENTIFIER_FIELD
        expected_node_id_fields = _PERSON_NODE_FEATURE_FLOAT_FIELDS

        self.__assert_bq_table_schema_contains_all_fields(
            table_name=person_enumerated_node_type_metadata.enumerated_node_data_reference.reference_uri,
            expected_fields=expected_node_id_fields + [node_id_field],
        )

        # Check that all the rows have unique ids and feature values.
        result = self.__bq_utils.run_query(
            query=f"""SELECT {node_id_field}, {', '.join(expected_node_id_fields)}
            FROM `{person_enumerated_node_type_metadata.enumerated_node_data_reference.reference_uri}`
            """,
            labels=get_resource_config().get_resource_labels(),
        )

        # Create a set composed of hashes of each of the original node features rows
        expected_row_hash_set = set(
            [
                hash(
                    tuple(  # List is not hashable, so we convert to tuple
                        [person_node_feature_record[node_id_field]]
                        + [
                            person_node_feature_record[field]
                            for field in expected_node_id_fields
                        ]
                    )
                )
                for person_node_feature_record in _PERSON_NODE_FEATURE_RECORDS
            ]
        )
        # Create a set composed of hashes of each of the enumerated node features rows
        row_hash_set = set(
            [
                hash(
                    tuple(  # List is not hashable, so we convert to tuple
                        [int_to_orig_node_id_map[row[node_id_field]]]
                        + [row[field] for field in expected_node_id_fields]
                    )
                )
                for row in result
            ]
        )

        self.assertEqual(
            row_hash_set,
            expected_row_hash_set,
            f"Expected {expected_row_hash_set}, got {row_hash_set}",
        )

    def assert_enumerated_edge_features_correctness(
        self,
        int_to_orig_node_id_map: Dict[int, str],
        map_enum_edge_type_metadata: Dict[
            Tuple[EdgeType, EdgeUsageType], EnumeratorEdgeTypeMetadata
        ],
    ):
        main_enumerated_edge_type_metadata = map_enum_edge_type_metadata[
            (_MESSAGES_EDGE_TYPE, EdgeUsageType.MAIN)
        ]
        positive_enumerated_edge_type_metadata = map_enum_edge_type_metadata[
            (_MESSAGES_EDGE_TYPE, EdgeUsageType.POSITIVE)
        ]
        negative_enumerated_edge_type_metadata = map_enum_edge_type_metadata[
            (_MESSAGES_EDGE_TYPE, EdgeUsageType.NEGATIVE)
        ]

        self.assertIsNotNone(main_enumerated_edge_type_metadata)
        self.assertIsNotNone(positive_enumerated_edge_type_metadata)
        self.assertIsNotNone(negative_enumerated_edge_type_metadata)

        # Check that the schema of the enumerated edge tablescontains all the expected fields.

        self.__assert_bq_table_schema_contains_all_fields(
            table_name=main_enumerated_edge_type_metadata.enumerated_edge_data_reference.reference_uri,
            expected_fields=_MESSAGE_EDGE_FEATURE_INT_FIELDS
            + [
                _MESSAGES_EDGE_SRC_IDENTIFIER_FIELD,
                _MESSAGES_EDGE_DST_IDENTIFIER_FIELD,
            ],
        )
        self.__assert_bq_table_schema_contains_all_fields(
            table_name=positive_enumerated_edge_type_metadata.enumerated_edge_data_reference.reference_uri,
            expected_fields=_POSITIVE_EDGE_FEATURE_INT_FIELDS
            + [
                _MESSAGES_EDGE_SRC_IDENTIFIER_FIELD,
                _MESSAGES_EDGE_DST_IDENTIFIER_FIELD,
            ],
        )
        self.__assert_bq_table_schema_contains_all_fields(
            table_name=negative_enumerated_edge_type_metadata.enumerated_edge_data_reference.reference_uri,
            expected_fields=[
                _MESSAGES_EDGE_SRC_IDENTIFIER_FIELD,
                _MESSAGES_EDGE_DST_IDENTIFIER_FIELD,
            ],
        )

        # Check that all the rows have unique ids and feature values.
        def __assert_enumerated_table_rows_match_original_rows(
            table_name: str,
            expected_edge_feature_fields: List[str],
            original_edge_feature_records: List[Dict[str, Any]],
        ):
            result = list(
                self.__bq_utils.run_query(
                    query=f"""SELECT {_MESSAGES_EDGE_SRC_IDENTIFIER_FIELD},
                    {_MESSAGES_EDGE_DST_IDENTIFIER_FIELD},
                    {', '.join(expected_edge_feature_fields)}
                    FROM `{table_name}`""",
                    labels=get_resource_config().get_resource_labels(),
                )
            )

            # Create a set composed of hashes of each of the original node features rows
            expected_row_hash_set = set(
                [
                    hash(
                        tuple(  # List is not hashable, so we convert to tuple
                            [
                                record[_MESSAGES_EDGE_SRC_IDENTIFIER_FIELD],
                                record[_MESSAGES_EDGE_DST_IDENTIFIER_FIELD],
                            ]
                            + [record[field] for field in expected_edge_feature_fields]
                        )
                    )
                    for record in original_edge_feature_records
                ]
            )
            # Create a set composed of hashes of each of the enumerated node features rows
            row_hash_set = set(
                [
                    hash(
                        tuple(  # List is not hashable, so we convert to tuple
                            [
                                int_to_orig_node_id_map[
                                    row[_MESSAGES_EDGE_SRC_IDENTIFIER_FIELD]
                                ],
                                int_to_orig_node_id_map[
                                    row[_MESSAGES_EDGE_DST_IDENTIFIER_FIELD]
                                ],
                            ]
                            + [row[field] for field in expected_edge_feature_fields]
                        )
                    )
                    for row in result
                ]
            )

            self.assertEqual(
                row_hash_set,
                expected_row_hash_set,
                f"Expected {expected_row_hash_set}, got {row_hash_set}",
            )

        __assert_enumerated_table_rows_match_original_rows(
            table_name=main_enumerated_edge_type_metadata.enumerated_edge_data_reference.reference_uri,
            expected_edge_feature_fields=_MESSAGE_EDGE_FEATURE_INT_FIELDS,
            original_edge_feature_records=_MESSAGE_EDGE_FEATURE_RECORDS,
        )
        __assert_enumerated_table_rows_match_original_rows(
            table_name=positive_enumerated_edge_type_metadata.enumerated_edge_data_reference.reference_uri,
            expected_edge_feature_fields=_POSITIVE_EDGE_FEATURE_INT_FIELDS,
            original_edge_feature_records=_POSITIVE_EDGE_FEATURE_RECORDS,
        )
        __assert_enumerated_table_rows_match_original_rows(
            table_name=negative_enumerated_edge_type_metadata.enumerated_edge_data_reference.reference_uri,
            expected_edge_feature_fields=[],
            original_edge_feature_records=_NEGATIVE_EDGE_FEATURE_RECORDS,
        )

    def test_for_correctness(self):
        enumerator = Enumerator()
        list_enumerator_node_type_metadata: List[EnumeratorNodeTypeMetadata]
        list_enumerator_edge_type_metadata: List[EnumeratorEdgeTypeMetadata]
        (
            list_enumerator_node_type_metadata,
            list_enumerator_edge_type_metadata,
        ) = enumerator.run(
            applied_task_identifier=AppliedTaskIdentifier(
                self.__applied_task_identifier
            ),
            node_data_references=self.node_data_references,
            edge_data_references=self.edge_data_references,
            gcp_project=get_resource_config().project,
        )

        for node_metadata in list_enumerator_node_type_metadata:
            self.__bq_tables_to_cleanup_on_teardown.append(
                node_metadata.enumerated_node_data_reference.reference_uri
            )
            self.__bq_tables_to_cleanup_on_teardown.append(
                node_metadata.bq_unique_node_ids_enumerated_table_name
            )
        for edge_metadata in list_enumerator_edge_type_metadata:
            self.__bq_tables_to_cleanup_on_teardown.append(
                edge_metadata.enumerated_edge_data_reference.reference_uri
            )

        map_enum_node_type_metadata: Dict[NodeType, EnumeratorNodeTypeMetadata] = {
            node_type_metadata.enumerated_node_data_reference.node_type: node_type_metadata
            for node_type_metadata in list_enumerator_node_type_metadata
        }
        map_enum_edge_type_metadata: Dict[
            Tuple[EdgeType, EdgeUsageType], EnumeratorEdgeTypeMetadata
        ] = {
            (
                edge_type_metadata.enumerated_edge_data_reference.edge_type,
                edge_type_metadata.enumerated_edge_data_reference.edge_usage_type,
            ): edge_type_metadata
            for edge_type_metadata in list_enumerator_edge_type_metadata
        }
        int_to_orig_node_id_map: Dict[
            int, str
        ] = self.fetch_enumerated_node_map_and_assert_correctness(
            map_enum_node_type_metadata=map_enum_node_type_metadata
        )
        self.assert_enumerated_node_features_correctness(
            int_to_orig_node_id_map=int_to_orig_node_id_map,
            map_enum_node_type_metadata=map_enum_node_type_metadata,
        )
        self.assert_enumerated_edge_features_correctness(
            int_to_orig_node_id_map=int_to_orig_node_id_map,
            map_enum_edge_type_metadata=map_enum_edge_type_metadata,
        )

    def tearDown(self) -> None:
        for table_name in self.__bq_tables_to_cleanup_on_teardown:
            self.__bq_utils.delete_bq_table_if_exist(bq_table_path=table_name)
