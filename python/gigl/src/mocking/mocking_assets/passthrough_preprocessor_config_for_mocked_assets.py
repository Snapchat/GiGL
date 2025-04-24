from typing import Any, Dict, List

import gigl.src.mocking.lib.constants as test_tasks_constants
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType, Relation
from gigl.src.data_preprocessor.lib.data_preprocessor_config import (
    DataPreprocessorConfig,
    build_ingestion_feature_spec_fn,
    build_passthrough_transform_preprocessing_fn,
)
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)
from gigl.src.data_preprocessor.lib.ingest.reference import (
    EdgeDataReference,
    NodeDataReference,
)
from gigl.src.data_preprocessor.lib.types import (
    EdgeDataPreprocessingSpec,
    EdgeOutputIdentifier,
    NodeDataPreprocessingSpec,
    NodeOutputIdentifier,
)
from gigl.src.mocking.dataset_asset_mocking_suite import mocked_datasets
from gigl.src.mocking.lib.feature_handling import get_feature_field_name
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata


class PassthroughPreprocessorConfigForMockedAssets(DataPreprocessorConfig):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.__mocked_dataset_name = kwargs.get("mocked_dataset_name", None)
        assert self.__mocked_dataset_name is not None, "mocked_dataset_name is required"

        self.__mocked_dataset: MockedDatasetInfo = mocked_datasets[
            self.__mocked_dataset_name
        ]
        self.__mocked_dataset_artifact_metadata = (
            get_mocked_dataset_artifact_metadata()[self.__mocked_dataset.name]
        )

    def get_nodes_preprocessing_spec(
        self,
    ) -> Dict[NodeDataReference, NodeDataPreprocessingSpec]:
        node_data_ref_to_preprocessing_specs: Dict[
            NodeDataReference, NodeDataPreprocessingSpec
        ] = dict()

        for node_type in self.__mocked_dataset.node_types:
            nodes_bq_table_name: str = (
                test_tasks_constants.get_example_task_nodes_bq_table_path(
                    task_name=self.__mocked_dataset.name,
                    version=self.__mocked_dataset_artifact_metadata.version,
                    node_type=node_type,
                )
            )

            node_data_ref = BigqueryNodeDataReference(
                reference_uri=nodes_bq_table_name,
                node_type=node_type,
            )
            node_feature_fields: List[str] = [
                get_feature_field_name(n=i)
                for i in range(self.__mocked_dataset.num_node_features[node_type])
            ]
            fixed_int_fields = [
                self.__mocked_dataset.node_id_column_name,
            ]
            node_labels_outputs = []

            should_use_node_labels_for_this_node_type = (
                self.__mocked_dataset.node_labels is not None
                and node_type in self.__mocked_dataset.node_labels
            )

            if should_use_node_labels_for_this_node_type:
                fixed_int_fields.append(self.__mocked_dataset.node_label_column_name)
                node_labels_outputs = [self.__mocked_dataset.node_label_column_name]

            feature_spec_fn = build_ingestion_feature_spec_fn(
                fixed_int_fields=fixed_int_fields,
                fixed_float_fields=node_feature_fields,
            )

            preprocessing_fn = build_passthrough_transform_preprocessing_fn()
            node_output_id = NodeOutputIdentifier(
                self.__mocked_dataset.node_id_column_name
            )
            node_features_outputs = node_feature_fields

            node_data_ref_to_preprocessing_specs[
                node_data_ref
            ] = NodeDataPreprocessingSpec(
                identifier_output=node_output_id,
                features_outputs=node_features_outputs,
                labels_outputs=node_labels_outputs,
                feature_spec_fn=feature_spec_fn,
                preprocessing_fn=preprocessing_fn,
            )

        return node_data_ref_to_preprocessing_specs

    def get_edges_preprocessing_spec(
        self,
    ) -> Dict[EdgeDataReference, EdgeDataPreprocessingSpec]:
        edge_data_ref_to_preprocessing_specs: Dict[
            EdgeDataReference, EdgeDataPreprocessingSpec
        ] = dict()
        for edge_type in self.__mocked_dataset.edge_types:
            main_edges_bq_table_name: str = (
                test_tasks_constants.get_example_task_edges_bq_table_path(
                    task_name=self.__mocked_dataset.name,
                    version=self.__mocked_dataset_artifact_metadata.version,
                    edge_type=edge_type,
                    edge_usage_type=EdgeUsageType.MAIN,
                )
            )

            edge_type = EdgeType(
                src_node_type=NodeType(edge_type.src_node_type),
                relation=Relation(edge_type.relation),
                dst_node_type=NodeType(edge_type.dst_node_type),
            )

            main_edge_data_ref = BigqueryEdgeDataReference(
                reference_uri=main_edges_bq_table_name,
                edge_type=edge_type,
                edge_usage_type=EdgeUsageType.MAIN,
            )

            default_edge_feature_fields: List[str] = [
                get_feature_field_name(n=i)
                for i in range(self.__mocked_dataset.num_edge_features[edge_type])
            ]

            edge_output_id = EdgeOutputIdentifier(
                src_node=NodeOutputIdentifier(
                    self.__mocked_dataset.edge_src_column_name
                ),
                dst_node=NodeOutputIdentifier(
                    self.__mocked_dataset.edge_dst_column_name
                ),
            )

            preprocessing_fn = build_passthrough_transform_preprocessing_fn()

            edge_data_ref_to_preprocessing_specs[
                main_edge_data_ref
            ] = EdgeDataPreprocessingSpec(
                identifier_output=edge_output_id,
                features_outputs=default_edge_feature_fields,
                feature_spec_fn=build_ingestion_feature_spec_fn(
                    fixed_int_fields=[
                        self.__mocked_dataset.edge_src_column_name,
                        self.__mocked_dataset.edge_dst_column_name,
                    ],
                    fixed_float_fields=default_edge_feature_fields,
                ),
                preprocessing_fn=preprocessing_fn,
            )

            should_use_user_defined_labels_for_this_edge_type = (
                self.__mocked_dataset.user_defined_edge_index is not None
                and edge_type in self.__mocked_dataset.user_defined_edge_index
            )
            if should_use_user_defined_labels_for_this_edge_type:
                assert self.__mocked_dataset.user_defined_edge_index is not None
                for edge_usage_type in self.__mocked_dataset.user_defined_edge_index[
                    edge_type
                ].keys():
                    user_defined_edges_bq_table_name: str = (
                        test_tasks_constants.get_example_task_edges_bq_table_path(
                            task_name=self.__mocked_dataset.name,
                            version=self.__mocked_dataset_artifact_metadata.version,
                            edge_type=edge_type,
                            edge_usage_type=edge_usage_type,
                        )
                    )
                    user_defined_edges_data_ref = BigqueryEdgeDataReference(
                        reference_uri=user_defined_edges_bq_table_name,
                        edge_type=edge_type,
                        edge_usage_type=edge_usage_type,
                    )
                    user_defined_edges_feature_fields: List[str] = [
                        get_feature_field_name(n=i)
                        for i in range(
                            self.__mocked_dataset.num_user_def_edge_features[edge_type][
                                edge_usage_type
                            ]
                        )
                    ]
                    edge_data_ref_to_preprocessing_specs[
                        user_defined_edges_data_ref
                    ] = EdgeDataPreprocessingSpec(
                        identifier_output=edge_output_id,
                        features_outputs=user_defined_edges_feature_fields,
                        feature_spec_fn=build_ingestion_feature_spec_fn(
                            fixed_int_fields=[
                                self.__mocked_dataset.edge_src_column_name,
                                self.__mocked_dataset.edge_dst_column_name,
                            ],
                            fixed_float_fields=user_defined_edges_feature_fields,
                        ),
                        preprocessing_fn=preprocessing_fn,
                    )

        return edge_data_ref_to_preprocessing_specs
