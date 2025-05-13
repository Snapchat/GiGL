import unittest
from collections import abc
from typing import Dict, List, Optional, Union

from parameterized import param, parameterized

from gigl.common.data.dataloaders import SerializedTFRecordInfo
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import (
    MockedDatasetArtifactMetadata,
    get_mocked_dataset_artifact_metadata,
)
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)


class TranslatorTestCase(unittest.TestCase):
    def setUp(self):
        self._name_to_mocked_dataset_map: Dict[
            str, MockedDatasetArtifactMetadata
        ] = get_mocked_dataset_artifact_metadata()

    def _assert_data_type_correctness(
        self,
        entity_info: Optional[
            Union[
                SerializedTFRecordInfo,
                Dict[NodeType, SerializedTFRecordInfo],
                Dict[EdgeType, SerializedTFRecordInfo],
                Dict[NodeType, Optional[SerializedTFRecordInfo]],
                Dict[EdgeType, Optional[SerializedTFRecordInfo]],
            ]
        ],
        is_heterogeneous: bool,
        expected_entity_types: Union[List[EdgeType], List[NodeType]],
    ):
        """
        Checks that each item in the provided serialized graph metadata is correctly typed and, if heterogeneous, that edge types and node types are as expected.
        Args:
            entity_info: Optional[
                Union[
                    SerializedTFRecordInfo,
                    Dict[NodeType, SerializedTFRecordInfo],
                    Dict[EdgeType, SerializedTFRecordInfo],
                    Dict[NodeType, Optional[SerializedTFRecordInfo]],
                    Dict[EdgeType, Optional[SerializedTFRecordInfo]],
                ]
            ]: Entity information of which type correctness is being checked for. If heterogeneous, this is expected to be a dictionary.
            is_heterogeneous (bool): Whether the provided input data is heterogeneous or homogeneous
            expected_entity_types(Union[List[EdgeType], List[NodeType]]): Expected node or edge type which we are checking against for heterogeneous inputs
        """
        self.assertIsNotNone(entity_info)
        if is_heterogeneous:
            assert isinstance(entity_info, abc.Mapping)
            self.assertTrue(sorted(entity_info.keys()) == sorted(expected_entity_types))
        else:
            self.assertNotIsInstance(entity_info, abc.Mapping)

    @parameterized.expand(
        [
            param(
                "Test Dataset Metadata Translator with homogeneous CORA NABLP dataset",
                mocked_dataset_info=CORA_NODE_ANCHOR_MOCKED_DATASET_INFO,
            ),
            param(
                "Test Dataset Metadata Translator with homogeneous CORA UDL NABLP dataset",
                mocked_dataset_info=CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
            ),
            param(
                "Test Dataset Metadata Translator with heterogeneous DBLP dataset",
                mocked_dataset_info=DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            ),
        ]
    )
    def test_translator_correctness(self, _, mocked_dataset_info: MockedDatasetInfo):
        mocked_dataset_artifact_metadata = self._name_to_mocked_dataset_map[
            mocked_dataset_info.name
        ]
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=mocked_dataset_artifact_metadata.frozen_gbml_config_uri
            )
        )

        preprocessed_metadata_pb_wrapper = (
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper
        )
        graph_metadata_pb_wrapper = gbml_config_pb_wrapper.graph_metadata_pb_wrapper

        serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
            preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
        )

        ## Node Entity Info Correctness

        self._assert_data_type_correctness(
            serialized_graph_metadata.node_entity_info,
            is_heterogeneous=graph_metadata_pb_wrapper.is_heterogeneous,
            expected_entity_types=graph_metadata_pb_wrapper.node_types,
        )

        if isinstance(serialized_graph_metadata.node_entity_info, abc.Mapping):
            serialized_node_info_iterable = list(
                serialized_graph_metadata.node_entity_info.values()
            )
        else:
            serialized_node_info_iterable = [serialized_graph_metadata.node_entity_info]

        self.assertEqual(
            len(graph_metadata_pb_wrapper.node_types),
            len(serialized_node_info_iterable),
        )

        for node_type, seralized_node_info in zip(
            graph_metadata_pb_wrapper.node_types, serialized_node_info_iterable
        ):
            condensed_node_type = (
                graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                    node_type
                ]
            )

            node_id_key = preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
                condensed_node_type
            ].node_id_key

            target_node_feature_spec = preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_schema_map[
                condensed_node_type
            ].feature_spec

            self.assertEqual(seralized_node_info.entity_key, node_id_key)

            self.assertEqual(
                seralized_node_info.feature_dim,
                preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_dim_map[
                    condensed_node_type
                ],
            )
            self.assertEqual(
                seralized_node_info.tfrecord_uri_prefix.uri,
                preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata[
                    condensed_node_type
                ].tfrecord_uri_prefix,
            )
            self.assertEqual(
                seralized_node_info.feature_keys,
                preprocessed_metadata_pb_wrapper.condensed_node_type_to_feature_keys_map[
                    condensed_node_type
                ],
            )
            self.assertEqual(
                seralized_node_info.feature_spec,
                target_node_feature_spec,
            )

        ## Edge Entity Info Correctness

        self._assert_data_type_correctness(
            serialized_graph_metadata.edge_entity_info,
            is_heterogeneous=graph_metadata_pb_wrapper.is_heterogeneous,
            expected_entity_types=graph_metadata_pb_wrapper.edge_types,
        )

        if isinstance(serialized_graph_metadata.edge_entity_info, abc.Mapping):
            serialized_edge_info_iterable = list(
                serialized_graph_metadata.edge_entity_info.values()
            )
        else:
            serialized_edge_info_iterable = [serialized_graph_metadata.edge_entity_info]

        self.assertEqual(
            len(graph_metadata_pb_wrapper.edge_types),
            len(serialized_edge_info_iterable),
        )

        for (
            edge_type,
            seralized_edge_info,
        ) in zip(graph_metadata_pb_wrapper.edge_types, serialized_edge_info_iterable):
            condensed_edge_type = (
                graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[
                    edge_type
                ]
            )
            edge_info = preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[
                condensed_edge_type
            ]

            target_edge_feature_spec = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_schema_map[
                condensed_edge_type
            ].feature_spec

            self.assertEqual(
                seralized_edge_info.entity_key,
                (edge_info.src_node_id_key, edge_info.dst_node_id_key),
            )

            self.assertEqual(
                seralized_edge_info.feature_dim,
                preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_dim_map[
                    condensed_edge_type
                ],
            )

            self.assertEqual(
                seralized_edge_info.tfrecord_uri_prefix.uri,
                edge_info.main_edge_info.tfrecord_uri_prefix,
            )
            self.assertEqual(
                seralized_edge_info.feature_keys,
                preprocessed_metadata_pb_wrapper.condensed_edge_type_to_feature_keys_map[
                    condensed_edge_type
                ],
            )
            self.assertEqual(
                seralized_edge_info.feature_spec,
                target_edge_feature_spec,
            )

        ## Positive Label Entity Info Correctness

        has_positive_labels = all(
            [
                preprocessed_metadata_pb_wrapper.has_pos_edge_features(
                    condensed_edge_type=condensed_edge_type
                )
                for condensed_edge_type in graph_metadata_pb_wrapper.condensed_edge_types
            ]
        )
        if has_positive_labels:
            self._assert_data_type_correctness(
                serialized_graph_metadata.positive_label_entity_info,
                is_heterogeneous=graph_metadata_pb_wrapper.is_heterogeneous,
                expected_entity_types=graph_metadata_pb_wrapper.edge_types,
            )
            if isinstance(
                serialized_graph_metadata.positive_label_entity_info, abc.Mapping
            ):
                serialized_positive_label_info_iterable = list(
                    serialized_graph_metadata.positive_label_entity_info.values()
                )
            else:
                serialized_positive_label_info_iterable = [
                    serialized_graph_metadata.positive_label_entity_info
                ]

            self.assertEqual(
                len(graph_metadata_pb_wrapper.edge_types),
                len(serialized_positive_label_info_iterable),
            )

            for edge_type, seralized_positive_label_info in zip(
                graph_metadata_pb_wrapper.edge_types,
                serialized_positive_label_info_iterable,
            ):
                if preprocessed_metadata_pb_wrapper.has_pos_edge_features(
                    condensed_edge_type=condensed_edge_type
                ):
                    assert (
                        seralized_positive_label_info is not None
                    )  # We use assert instead of self.assertIsNotNone since this allows type narrowing with mypy

                    edge_info = preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[
                        condensed_edge_type
                    ]

                    target_pos_edge_feature_spec = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_pos_edge_feature_schema_map[
                        condensed_edge_type
                    ].feature_spec

                    self.assertEqual(
                        seralized_positive_label_info.entity_key,
                        (edge_info.src_node_id_key, edge_info.dst_node_id_key),
                    )

                    self.assertEqual(
                        seralized_positive_label_info.feature_dim,
                        preprocessed_metadata_pb_wrapper.condensed_edge_type_to_pos_edge_feature_dim_map[
                            condensed_edge_type
                        ],
                    )

                    self.assertEqual(
                        seralized_positive_label_info.tfrecord_uri_prefix.uri,
                        edge_info.positive_edge_info.tfrecord_uri_prefix,
                    )
                    self.assertEqual(
                        seralized_positive_label_info.feature_keys,
                        preprocessed_metadata_pb_wrapper.condensed_edge_type_to_pos_edge_feature_keys_map[
                            condensed_edge_type
                        ],
                    )
                    self.assertEqual(
                        seralized_positive_label_info.feature_spec,
                        target_pos_edge_feature_spec,
                    )
                else:
                    self.assertIsNone(seralized_positive_label_info)
        else:
            self.assertIsNone(serialized_graph_metadata.positive_label_entity_info)

        ## Negative Label Entity Info Correctness

        has_negative_labels = all(
            [
                preprocessed_metadata_pb_wrapper.has_hard_neg_edge_features(
                    condensed_edge_type=condensed_edge_type
                )
                for condensed_edge_type in graph_metadata_pb_wrapper.condensed_edge_types
            ]
        )
        if has_negative_labels:
            self._assert_data_type_correctness(
                serialized_graph_metadata.negative_label_entity_info,
                is_heterogeneous=graph_metadata_pb_wrapper.is_heterogeneous,
                expected_entity_types=graph_metadata_pb_wrapper.edge_types,
            )
            if isinstance(
                serialized_graph_metadata.negative_label_entity_info, abc.Mapping
            ):
                serialized_negative_label_info_iterable = list(
                    serialized_graph_metadata.negative_label_entity_info.values()
                )
            else:
                serialized_negative_label_info_iterable = [
                    serialized_graph_metadata.negative_label_entity_info
                ]

            self.assertEqual(
                len(graph_metadata_pb_wrapper.edge_types),
                len(serialized_negative_label_info_iterable),
            )

            for edge_type, serialized_negative_label_info in zip(
                graph_metadata_pb_wrapper.edge_types,
                serialized_negative_label_info_iterable,
            ):
                if preprocessed_metadata_pb_wrapper.has_hard_neg_edge_features(
                    condensed_edge_type=condensed_edge_type
                ):
                    assert (
                        serialized_negative_label_info is not None
                    )  # We use assert instead of self.assertIsNotNone since this allows type narrowing with mypy

                    edge_info = preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata[
                        condensed_edge_type
                    ]

                    target_hard_neg_edge_feature_spec = preprocessed_metadata_pb_wrapper.condensed_edge_type_to_hard_neg_edge_feature_schema_map[
                        condensed_edge_type
                    ].feature_spec

                    self.assertEqual(
                        serialized_negative_label_info.entity_key,
                        (edge_info.src_node_id_key, edge_info.dst_node_id_key),
                    )

                    self.assertEqual(
                        serialized_negative_label_info.feature_dim,
                        preprocessed_metadata_pb_wrapper.condensed_edge_type_to_hard_neg_edge_feature_dim_map[
                            condensed_edge_type
                        ],
                    )

                    self.assertEqual(
                        serialized_negative_label_info.tfrecord_uri_prefix.uri,
                        edge_info.negative_edge_info.tfrecord_uri_prefix,
                    )
                    self.assertEqual(
                        serialized_negative_label_info.feature_keys,
                        preprocessed_metadata_pb_wrapper.condensed_edge_type_to_hard_neg_edge_feature_keys_map[
                            condensed_edge_type
                        ],
                    )
                    self.assertEqual(
                        serialized_negative_label_info.feature_spec,
                        target_hard_neg_edge_feature_spec,
                    )
                else:
                    self.assertIsNone(serialized_negative_label_info)
        else:
            self.assertIsNone(serialized_graph_metadata.negative_label_entity_info)
