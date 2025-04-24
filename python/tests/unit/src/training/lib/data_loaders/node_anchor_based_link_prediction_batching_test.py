import unittest
from typing import cast

import numpy as np

from gigl.src.common.constants.graph_metadata import (
    DEFAULT_CONDENSED_EDGE_TYPE,
    DEFAULT_CONDENSED_NODE_TYPE,
)
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.translators.training_samples_protos_translator import (
    NodeAnchorBasedLinkPredictionSample,
    TrainingSamplesProtosTranslator,
)
from gigl.src.common.types.graph_data import NodeId
from gigl.src.common.types.model import GraphBackend
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.common.utils.data.feature_serialization import FeatureSerializationUtils
from gigl.src.training.v1.lib.data_loaders.node_anchor_based_link_prediction_data_loader import (
    NodeAnchorBasedLinkPredictionBatch,
)
from snapchat.research.gbml import (
    graph_schema_pb2,
    preprocessed_metadata_pb2,
    training_samples_schema_pb2,
)
from tests.test_assets.graph_metadata_constants import (
    DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
    DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
    EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES,
    EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES,
    EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
    EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
)


class NodeAnchorBasedLinkPredictionBatchingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.heterogeneous_condensed_node_type_zero = (
            EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[0]
        )
        self.heterogeneous_condensed_node_type_one = (
            EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[1]
        )
        self.heterogeneous_condensed_node_type_two = (
            EXAMPLE_HETEROGENEOUS_CONDENSED_NODE_TYPES[2]
        )

        self.heterogeneous_condensed_edge_type_zero = (
            EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[0]
        )
        self.heterogeneous_condensed_edge_type_one = (
            EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[1]
        )
        self.heterogeneous_condensed_edge_type_two = (
            EXAMPLE_HETEROGENEOUS_CONDENSED_EDGE_TYPES[2]
        )

        dummy_node_feature_bytes = FeatureSerializationUtils.serialize_node_features(
            features=np.array([0])
        )
        dummy_edge_feature_bytes = FeatureSerializationUtils.serialize_edge_features(
            features=np.array([0, 1])
        )
        dummy_user_defined_label_edge_feature_bytes = (
            FeatureSerializationUtils.serialize_edge_features(
                features=np.array([0, 1, 2])
            )
        )

        # Create a "triangle" sample rooted at 0 with pos edge 0->1 and hard neg edge 0->3
        # The neighborhood contains a 3-clique (triangle) of edges: 0 -> 1, 0 -> 2, 1 -> 2.
        self.node_0 = graph_schema_pb2.Node(
            node_id=0,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_1 = graph_schema_pb2.Node(
            node_id=1,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_2 = graph_schema_pb2.Node(
            node_id=2,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_3 = graph_schema_pb2.Node(
            node_id=3,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
        self.edge_01 = graph_schema_pb2.Edge(
            src_node_id=0,
            dst_node_id=1,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.edge_02 = graph_schema_pb2.Edge(
            src_node_id=0,
            dst_node_id=2,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.edge_12 = graph_schema_pb2.Edge(
            src_node_id=1,
            dst_node_id=2,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.hard_neg_edge_03 = graph_schema_pb2.Edge(
            src_node_id=0,
            dst_node_id=3,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.triangle_neighborhood = graph_schema_pb2.Graph(
            nodes=[self.node_0, self.node_1, self.node_2, self.node_3],
            edges=[self.edge_01, self.edge_02, self.edge_12],
        )
        self.triangle_pb = (
            training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample(
                root_node=self.node_0,
                pos_edges=[self.edge_01],
                hard_neg_edges=[self.hard_neg_edge_03],
                neighborhood=self.triangle_neighborhood,
            )
        )

        # Create a "line" sample rooted at 3 with pos edge 3->4 and hard neg edge 3->0.
        # The neighborhood contains a 2-clique (line/edge) of edges: 3 -> 4.
        self.node_4 = graph_schema_pb2.Node(
            node_id=4,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
        self.edge_34 = graph_schema_pb2.Edge(
            src_node_id=3,
            dst_node_id=4,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.hard_neg_edge_30 = graph_schema_pb2.Edge(
            src_node_id=3,
            dst_node_id=0,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.line_neighborhood = graph_schema_pb2.Graph(
            nodes=[self.node_3, self.node_4, self.node_0], edges=[self.edge_34]
        )

        self.line_pb = training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample(
            root_node=self.node_3,
            pos_edges=[self.edge_34],
            hard_neg_edges=[self.hard_neg_edge_30],
            neighborhood=self.line_neighborhood,
        )

        # Create a "chain" sample rooted at 2 with pos edge 2->3 and hard neg edge 2->4.
        # The neighborhood contains a "chain" of edges: 1->2, 2->3
        self.edge_23 = graph_schema_pb2.Edge(
            src_node_id=2,
            dst_node_id=3,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.hard_neg_edge_24 = graph_schema_pb2.Edge(
            src_node_id=2,
            dst_node_id=4,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.chain_neighborhood = graph_schema_pb2.Graph(
            nodes=[self.node_1, self.node_2, self.node_3, self.node_4],
            edges=[self.edge_12, self.edge_23],
        )

        self.chain_pb = training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample(
            root_node=self.node_2,
            pos_edges=[self.edge_23],
            hard_neg_edges=[self.hard_neg_edge_24],
            neighborhood=self.chain_neighborhood,
        )

        # Create a sample rooted at 5 with feature rich user-defined label edges of
        # pos edge 5 -> 6 and hard neg edge 5 -> 7.
        # The neighborhood contains 5 -> 6 and 6 -> 7.
        self.node_5 = graph_schema_pb2.Node(
            node_id=5,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_6 = graph_schema_pb2.Node(
            node_id=6,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_7 = graph_schema_pb2.Node(
            node_id=7,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
        self.edge_56 = graph_schema_pb2.Edge(
            src_node_id=5,
            dst_node_id=6,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
            feature_values=dummy_edge_feature_bytes,
        )
        self.edge_67 = graph_schema_pb2.Edge(
            src_node_id=6,
            dst_node_id=7,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
            feature_values=dummy_edge_feature_bytes,
        )
        self.user_defined_pos_edge_56 = graph_schema_pb2.Edge(
            src_node_id=5,
            dst_node_id=6,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
            feature_values=dummy_user_defined_label_edge_feature_bytes,
        )
        self.user_defined_neg_edge_57 = graph_schema_pb2.Edge(
            src_node_id=5,
            dst_node_id=7,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
            feature_values=dummy_user_defined_label_edge_feature_bytes,
        )
        self.user_defined_labels_neighborhood = graph_schema_pb2.Graph(
            nodes=[self.node_5, self.node_6, self.node_7],
            edges=[self.edge_56, self.edge_67],
        )
        self.user_defined_labels_pb = (
            training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample(
                root_node=self.node_5,
                pos_edges=[self.user_defined_pos_edge_56],
                hard_neg_edges=[self.user_defined_neg_edge_57],
                neighborhood=self.user_defined_labels_neighborhood,
            )
        )
        self.user_defined_labels_preprocessed_metadata_pb = preprocessed_metadata_pb2.PreprocessedMetadata(
            condensed_node_type_to_preprocessed_metadata={
                DEFAULT_CONDENSED_NODE_TYPE: preprocessed_metadata_pb2.PreprocessedMetadata.NodeMetadataOutput(
                    feature_dim=1,
                )
            },
            condensed_edge_type_to_preprocessed_metadata={
                DEFAULT_CONDENSED_EDGE_TYPE: preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataOutput(
                    main_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(
                        feature_dim=2,
                    ),
                    positive_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(
                        feature_dim=3,
                    ),
                    negative_edge_info=preprocessed_metadata_pb2.PreprocessedMetadata.EdgeMetadataInfo(
                        feature_dim=3,
                    ),
                )
            },
        )
        self.user_defined_labels_preprocessed_metadata_pb_wrapper = PreprocessedMetadataPbWrapper(
            preprocessed_metadata_pb=self.user_defined_labels_preprocessed_metadata_pb
        )

        # Create a "triangle" sample with multiple edge and node types rooted at 8
        # with pos edge 8->9 and 8->10 and hard neg edge 8->11 and 8->12 where there are
        # two types of positive edges and two types of hard neg edges
        # The neighborhood contains a 3-clique (triangle) of edges: 8 -> 9, 8 -> 10, 9 -> 10.
        # Node 8 is CNT 0, Node 9 is CNT 1, Node 10 is CNT 2, Node 11 is CNT 1, Node 12 is CNT 2
        # Edge 8 -> 9 is CET 0, Edge 8 -> 10 is CET 1, Edge 9 -> 10 is CET 2, Edge 8 -> 11 is CET 0, Edge 8 -> 12 is CET 1
        self.node_8 = graph_schema_pb2.Node(
            node_id=8,
            condensed_node_type=self.heterogeneous_condensed_node_type_zero,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_9 = graph_schema_pb2.Node(
            node_id=9,
            condensed_node_type=self.heterogeneous_condensed_node_type_one,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_10 = graph_schema_pb2.Node(
            node_id=10,
            condensed_node_type=self.heterogeneous_condensed_node_type_two,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_11 = graph_schema_pb2.Node(
            node_id=11,
            condensed_node_type=self.heterogeneous_condensed_node_type_one,
            feature_values=dummy_node_feature_bytes,
        )
        self.node_12 = graph_schema_pb2.Node(
            node_id=12,
            condensed_node_type=self.heterogeneous_condensed_node_type_two,
            feature_values=dummy_node_feature_bytes,
        )
        self.edge_89 = graph_schema_pb2.Edge(
            src_node_id=8,
            dst_node_id=9,
            condensed_edge_type=self.heterogeneous_condensed_edge_type_zero,
        )
        self.edge_810 = graph_schema_pb2.Edge(
            src_node_id=8,
            dst_node_id=10,
            condensed_edge_type=self.heterogeneous_condensed_edge_type_one,
        )
        self.edge_910 = graph_schema_pb2.Edge(
            src_node_id=9,
            dst_node_id=10,
            condensed_edge_type=self.heterogeneous_condensed_edge_type_two,
        )
        self.hard_neg_edge_811 = graph_schema_pb2.Edge(
            src_node_id=8,
            dst_node_id=11,
            condensed_edge_type=self.heterogeneous_condensed_edge_type_zero,
        )
        self.hard_neg_edge_812 = graph_schema_pb2.Edge(
            src_node_id=8,
            dst_node_id=12,
            condensed_edge_type=self.heterogeneous_condensed_edge_type_one,
        )
        self.edge_typed_triangle_neighborhood = graph_schema_pb2.Graph(
            nodes=[self.node_8, self.node_9, self.node_10, self.node_11, self.node_12],
            edges=[self.edge_89, self.edge_810, self.edge_910],
        )
        self.edge_typed_triangle_pb = (
            training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample(
                root_node=self.node_8,
                pos_edges=[self.edge_89, self.edge_810],
                hard_neg_edges=[self.hard_neg_edge_811, self.hard_neg_edge_812],
                neighborhood=self.edge_typed_triangle_neighborhood,
            )
        )

        self.builder = GraphBuilderFactory.get_graph_builder(
            backend_name=GraphBackend.PYG
        )

    def test_translated_homogeneous_sample_from_training_sample_pb(self):
        # Build a translated sample via TrainingSamplesProtosTranslators
        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_NodeAnchorBasedLinkPredictionSamplePb(
            samples=[self.triangle_pb],
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            builder=self.builder,
        )
        triangle_translated_sample_pb: NodeAnchorBasedLinkPredictionSample = (
            translated_sample_pbs[0]
        )

        # Check the root node is as specified during construction.
        self.assertEqual(
            triangle_translated_sample_pb.root_node.id, self.node_0.node_id
        )

        # Check the positive edge is as specified during construction.
        self.assertEqual(
            len(
                triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].pos_nodes
            ),
            1,
        )
        translated_pos_node = (
            triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                DEFAULT_CONDENSED_EDGE_TYPE
            ].pos_nodes[0]
        )
        self.assertEqual(translated_pos_node, self.edge_01.dst_node_id)

        # Check the hard negative edge is as specified during construction.
        self.assertEqual(
            len(
                triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].hard_neg_nodes
            ),
            1,
        )
        translated_hard_neg_node = (
            triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                DEFAULT_CONDENSED_EDGE_TYPE
            ].hard_neg_nodes[0]
        )
        self.assertEqual(translated_hard_neg_node, self.hard_neg_edge_03.dst_node_id)

        # Check the subgraph has the correct number of nodes and edges.
        translated_subgraph = triangle_translated_sample_pb.subgraph
        translated_subgraph = cast(PygGraphData, translated_subgraph)
        self.assertEqual(
            translated_subgraph.num_nodes, len(self.triangle_neighborhood.nodes)
        )
        self.assertEqual(
            translated_subgraph.num_edges, len(self.triangle_neighborhood.edges)
        )

    def test_can_collate_homogeneous_correctly_without_edge_overlap(self):
        """
        We try to collate the triangle and the line samples.  The two do not overlap on any edges.
        We want to validate that after collating these samples, the resulting graph has the correct # nodes and edges.
        We also validate the supervision edges are same according to the input.
        :return:
        """

        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_NodeAnchorBasedLinkPredictionSamplePb(
            samples=[self.triangle_pb, self.line_pb],
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            builder=self.builder,
        )

        batch = NodeAnchorBasedLinkPredictionBatch.collate_pyg_node_anchor_based_link_prediction_minibatch(
            builder=self.builder,
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            samples=translated_sample_pbs,
        )

        # Ensure the batch has 5 nodes (from the added 0,1,2,3,4), and 4 edges (0->1, 1->2, 0->2 and 3->4)
        self.assertEqual(batch.graph.num_nodes, 5)
        self.assertEqual(batch.graph.num_edges, 4)

        # Ensure a correct map of src nodes to dst nodes is maintained for positive edges.
        for pos_edge in [self.edge_01, self.edge_34]:
            self.assertIn(
                pos_edge.src_node_id,
                batch.pos_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].root_node_to_target_node_id,
            )
            self.assertIn(
                pos_edge.dst_node_id,
                batch.pos_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].root_node_to_target_node_id[NodeId(pos_edge.src_node_id)],
            )

        # Ensure a correct map of src nodes to dst nodes is maintained for hard negative edges.
        for hard_neg_edge in [self.hard_neg_edge_03, self.hard_neg_edge_30]:
            self.assertIn(
                hard_neg_edge.src_node_id,
                batch.hard_neg_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].root_node_to_target_node_id,
            )
            self.assertIn(
                hard_neg_edge.dst_node_id,
                batch.hard_neg_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].root_node_to_target_node_id[NodeId(hard_neg_edge.src_node_id)],
            )

    def test_can_collate_homogeneous_correctly_with_edge_overlap(self):
        """
        We try to collate the triangle and the chain samples.  The two share an overlap on global edge 1->2.
        We want to validate that after collating these samples, the resulting graph does not duplicate the edge.
        We also validate the supervision edges are same according to the input.
        :return:
        """

        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_NodeAnchorBasedLinkPredictionSamplePb(
            samples=[self.triangle_pb, self.chain_pb],
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            builder=self.builder,
        )

        batch = NodeAnchorBasedLinkPredictionBatch.collate_pyg_node_anchor_based_link_prediction_minibatch(
            builder=self.builder,
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            samples=translated_sample_pbs,
        )

        # Ensure the batch has 5 nodes (from the added 0,1,2,3,4),
        # and 4 edges (from the added 0->1, 1->2, 0->2 and 2->3).
        self.assertEqual(batch.graph.num_nodes, 5)
        self.assertEqual(batch.graph.num_edges, 4)

        # Ensure a correct map of src nodes to dst nodes is maintained for positive edges.
        for pos_edge in [self.edge_01, self.edge_23]:
            self.assertIn(
                pos_edge.src_node_id,
                batch.pos_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].root_node_to_target_node_id,
            )
            self.assertIn(
                pos_edge.dst_node_id,
                batch.pos_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].root_node_to_target_node_id[NodeId(pos_edge.src_node_id)],
            )

        # Ensure a correct map of src nodes to dst nodes is maintained for hard negative edges.
        for hard_neg_edge in [self.hard_neg_edge_03, self.hard_neg_edge_24]:
            self.assertIn(
                hard_neg_edge.src_node_id,
                batch.hard_neg_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].root_node_to_target_node_id,
            )
            self.assertIn(
                hard_neg_edge.dst_node_id,
                batch.hard_neg_supervision_edge_data[
                    DEFAULT_CONDENSED_EDGE_TYPE
                ].root_node_to_target_node_id[NodeId(hard_neg_edge.src_node_id)],
            )

    def test_can_load_homogeneous_edge_features(self):
        """
        We try to collate the user defined label sample.
        We want to validate that after collating the sample, the resulting
        NodeAnchorBasedLinkPredictionSample has correct edge features for
        message passing edges, user-defined positive edges, and user-defined hard negative edges.
        :return:
        """

        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_NodeAnchorBasedLinkPredictionSamplePb(
            samples=[self.user_defined_labels_pb],
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=self.user_defined_labels_preprocessed_metadata_pb_wrapper,
            builder=self.builder,
        )

        batch = NodeAnchorBasedLinkPredictionBatch.collate_pyg_node_anchor_based_link_prediction_minibatch(
            builder=self.builder,
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=self.user_defined_labels_preprocessed_metadata_pb_wrapper,
            samples=translated_sample_pbs,
        )

        # Ensure the batch has 3 nodes (5, 6, 7),
        # and 2 message passing edges (5 -> 6, 6 -> 7),
        # 1 positive edges (5 -> 6), and 1 hard negative edges (6 -> 7).
        pos_nodes = batch.pos_supervision_edge_data[
            DEFAULT_CONDENSED_EDGE_TYPE
        ].root_node_to_target_node_id
        hard_neg_nodes = batch.hard_neg_supervision_edge_data[
            DEFAULT_CONDENSED_EDGE_TYPE
        ].root_node_to_target_node_id
        pos_edges = batch.pos_supervision_edge_data[
            DEFAULT_CONDENSED_EDGE_TYPE
        ].label_edge_features
        hard_neg_edges = batch.hard_neg_supervision_edge_data[
            DEFAULT_CONDENSED_EDGE_TYPE
        ].label_edge_features
        self.assertEqual(batch.graph.num_nodes, 3)
        self.assertEqual(batch.graph.num_edges, 2)
        self.assertEqual(len(pos_nodes), 1)
        self.assertEqual(len(hard_neg_nodes), 1)

        # Ensure the message passing edges has correct edge features
        self.assertEqual(batch.graph.edge_attr.shape[1], 2)

        # Ensure the user-defined label edges has correct edge features
        self.assertEqual(pos_edges is not None, True)
        self.assertEqual(hard_neg_edges is not None, True)
        for pos_node in pos_nodes:
            self.assertEqual(pos_edges[pos_node].shape[0], 1)  # type: ignore
            self.assertEqual(pos_edges[pos_node].shape[1], 3)  # type: ignore
        for hard_neg_node in hard_neg_nodes:
            self.assertEqual(hard_neg_edges[hard_neg_node].shape[0], 1)  # type: ignore
            self.assertEqual(hard_neg_edges[hard_neg_node].shape[1], 3)  # type: ignore

    def test_translated_heterogeneous_sample_from_training_sample_pb(self):
        """
        We try to load heterogeneous edges and nodes from sample.
        We want to validate that the resulting NodeAnchorBasedLinkPredictionSample has
        correct information for its positive and hard negative edges.
        :return:
        """
        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_NodeAnchorBasedLinkPredictionSamplePb(
            samples=[self.edge_typed_triangle_pb],
            graph_metadata_pb_wrapper=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            builder=self.builder,
        )
        triangle_translated_sample_pb: NodeAnchorBasedLinkPredictionSample = (
            translated_sample_pbs[0]
        )
        self.assertIn(
            self.heterogeneous_condensed_edge_type_zero,
            triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data,
        )
        self.assertIn(
            self.heterogeneous_condensed_edge_type_one,
            triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data,
        )
        # Check the positive edge is as specified during construction.
        self.assertEqual(
            len(
                triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                    self.heterogeneous_condensed_edge_type_zero
                ].pos_nodes
            ),
            1,
        )
        translated_pos_node = (
            triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_zero
            ].pos_nodes[0]
        )
        self.assertEqual(translated_pos_node, self.edge_89.dst_node_id)

        self.assertEqual(
            len(
                triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                    self.heterogeneous_condensed_edge_type_one
                ].pos_nodes
            ),
            1,
        )
        translated_pos_node = (
            triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_one
            ].pos_nodes[0]
        )
        self.assertEqual(translated_pos_node, self.edge_810.dst_node_id)

        # Check the hard negative edge is as specified during construction.
        self.assertEqual(
            len(
                triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                    self.heterogeneous_condensed_edge_type_zero
                ].hard_neg_nodes
            ),
            1,
        )
        translated_hard_neg_node = (
            triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_zero
            ].hard_neg_nodes[0]
        )
        self.assertEqual(translated_hard_neg_node, self.hard_neg_edge_811.dst_node_id)

        self.assertEqual(
            len(
                triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                    self.heterogeneous_condensed_edge_type_one
                ].hard_neg_nodes
            ),
            1,
        )
        translated_hard_neg_node = (
            triangle_translated_sample_pb.condensed_edge_type_to_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_one
            ].hard_neg_nodes[0]
        )
        self.assertEqual(translated_hard_neg_node, self.hard_neg_edge_812.dst_node_id)

    def can_load_heterogeneous_correctly(self):
        """
        We try to load a heterogeneous triangle sample. We want to validate that the supervision edges and nodes aer
        as expected.
        :return:
        """

        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_NodeAnchorBasedLinkPredictionSamplePb(
            samples=[self.edge_typed_triangle_pb],
            graph_metadata_pb_wrapper=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            builder=self.builder,
        )

        batch = NodeAnchorBasedLinkPredictionBatch.collate_pyg_node_anchor_based_link_prediction_minibatch(
            builder=self.builder,
            graph_metadata_pb_wrapper=EXAMPLE_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=EXAMPLE_HETEROGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            samples=translated_sample_pbs,
        )

        self.assertEqual(batch.graph.num_nodes, 5)
        self.assertEqual(batch.graph.num_edges, 5)

        # Ensure a correct map of src nodes to dst nodes is maintained for positive edges for CET 1.
        self.assertIn(
            self.edge_89.src_node_id,
            batch.pos_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_zero
            ].root_node_to_target_node_id,
        )
        self.assertIn(
            self.edge_89.dst_node_id,
            batch.pos_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_zero
            ].root_node_to_target_node_id[NodeId(self.edge_89.src_node_id)],
        )
        # Ensure a correct map of src nodes to dst nodes is maintained for positive edges for CET 2.
        self.assertIn(
            self.edge_810.src_node_id,
            batch.pos_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_one
            ].root_node_to_target_node_id,
        )
        self.assertIn(
            self.edge_810.dst_node_id,
            batch.pos_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_one
            ].root_node_to_target_node_id[NodeId(self.edge_810.src_node_id)],
        )

        # Ensure a correct map of src nodes to dst nodes is maintained for hard negative edges for CET 1.
        self.assertIn(
            self.hard_neg_edge_811.src_node_id,
            batch.hard_neg_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_zero
            ].root_node_to_target_node_id,
        )
        self.assertIn(
            self.hard_neg_edge_811.dst_node_id,
            batch.hard_neg_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_zero
            ].root_node_to_target_node_id[NodeId(self.hard_neg_edge_811.src_node_id)],
        )

        # Ensure a correct map of src nodes to dst nodes is maintained for hard negative edges for CET 2.
        self.assertIn(
            self.hard_neg_edge_812.src_node_id,
            batch.hard_neg_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_one
            ].root_node_to_target_node_id,
        )
        self.assertIn(
            self.hard_neg_edge_812.dst_node_id,
            batch.hard_neg_supervision_edge_data[
                self.heterogeneous_condensed_edge_type_one
            ].root_node_to_target_node_id[NodeId(self.hard_neg_edge_812.src_node_id)],
        )
