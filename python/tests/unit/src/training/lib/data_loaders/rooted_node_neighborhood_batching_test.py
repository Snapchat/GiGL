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
    RootedNodeNeighborhoodSample,
    TrainingSamplesProtosTranslator,
)
from gigl.src.common.types.model import GraphBackend
from gigl.src.common.utils.data.feature_serialization import FeatureSerializationUtils
from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)
from snapchat.research.gbml import graph_schema_pb2, training_samples_schema_pb2
from tests.test_assets.graph_metadata_constants import (
    DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
    DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
)


class RootedNodeNeighborhoodBatchingTest(unittest.TestCase):
    # TODO: Extend this test to heterogeneous graph data.
    def setUp(self) -> None:
        dummy_node_feature_bytes = FeatureSerializationUtils.serialize_node_features(
            features=np.array([0])
        )

        # Create a "triangle" sample rooted at 0.
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
        self.triangle_neighborhood = graph_schema_pb2.Graph(
            nodes=[self.node_0, self.node_1, self.node_2],
            edges=[self.edge_01, self.edge_02, self.edge_12],
        )
        self.triangle_pb = training_samples_schema_pb2.RootedNodeNeighborhood(
            root_node=self.node_0, neighborhood=self.triangle_neighborhood
        )

        # Create a "line" sample rooted at 3.
        # The neighborhood contains a 2-clique (line/edge) of edges: 3 -> 4.
        self.node_3 = graph_schema_pb2.Node(
            node_id=3,
            condensed_node_type=DEFAULT_CONDENSED_NODE_TYPE,
            feature_values=dummy_node_feature_bytes,
        )
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
        self.line_neighborhood = graph_schema_pb2.Graph(
            nodes=[self.node_3, self.node_4], edges=[self.edge_34]
        )

        self.line_pb = training_samples_schema_pb2.RootedNodeNeighborhood(
            root_node=self.node_3, neighborhood=self.line_neighborhood
        )

        # Create a "chain" sample rooted at 2.
        # The neighborhood contains a "chain" of edges: 1->2, 2->3
        self.edge_23 = graph_schema_pb2.Edge(
            src_node_id=2,
            dst_node_id=3,
            condensed_edge_type=DEFAULT_CONDENSED_EDGE_TYPE,
        )
        self.chain_neighborhood = graph_schema_pb2.Graph(
            nodes=[self.node_1, self.node_2, self.node_3],
            edges=[self.edge_12, self.edge_23],
        )

        self.chain_pb = training_samples_schema_pb2.RootedNodeNeighborhood(
            root_node=self.node_2, neighborhood=self.chain_neighborhood
        )

        self.builder = GraphBuilderFactory.get_graph_builder(
            backend_name=GraphBackend.PYG
        )

    def test_translated_sample_from_training_sample_pb(self):
        # Build a translated sample via TrainingSamplesProtosTranslators
        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_RootedNodeNeighborhoodPb(
            samples=[self.triangle_pb],
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            builder=self.builder,
        )
        triangle_translated_sample_pb: RootedNodeNeighborhoodSample = (
            translated_sample_pbs[0]
        )

        # Check the root node is as specified during construction.
        self.assertEqual(
            triangle_translated_sample_pb.root_node.id, self.node_0.node_id
        )

        # Check the subgraph has the correct number of nodes and edges.
        translated_subgraph = triangle_translated_sample_pb.subgraph
        translated_subgraph = cast(PygGraphData, translated_subgraph)
        self.assertEqual(
            translated_subgraph.num_nodes, len(self.triangle_neighborhood.nodes)
        )
        self.assertEqual(
            translated_subgraph.num_edges, len(self.triangle_neighborhood.edges)
        )

    def test_can_collate_correctly_without_edge_overlap(self):
        """
        We try to collate the triangle and the line samples.  The two do not overlap on any edges.
        We want to validate that after collating these samples, the resulting graph has the correct # nodes and edges.
        We also validate the supervision edges are same according to the input.
        :return:
        """

        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_RootedNodeNeighborhoodPb(
            samples=[self.triangle_pb, self.line_pb],
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            builder=self.builder,
        )
        dataloaded_samples = [
            {
                DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER.condensed_node_type_to_node_type_map[
                    DEFAULT_CONDENSED_NODE_TYPE
                ]: translanted_sample_pb
            }
            for translanted_sample_pb in translated_sample_pbs
        ]
        batch = RootedNodeNeighborhoodBatch.collate_pyg_rooted_node_neighborhood_minibatch(
            builder=self.builder,
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            samples=dataloaded_samples,
        )

        # Ensure the batch has 5 nodes (from the added 0,1,2,3,4), and 4 edges (0->1, 1->2, 0->2 and 3->4)
        self.assertEqual(batch.graph.num_nodes, 5)
        self.assertEqual(batch.graph.num_edges, 4)

    def test_can_collate_correctly_with_edge_overlap(self):
        """
        We try to collate the triangle and the chain samples.  The two share an overlap on global edge 1->2.
        We want to validate that after collating these samples, the resulting graph does not duplicate the edge.
        We also validate the supervision edges are same according to the input.
        :return:
        """

        translated_sample_pbs = TrainingSamplesProtosTranslator.training_samples_from_RootedNodeNeighborhoodPb(
            samples=[self.triangle_pb, self.chain_pb],
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            builder=self.builder,
        )
        dataloaded_samples = [
            {
                DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER.condensed_node_type_to_node_type_map[
                    DEFAULT_CONDENSED_NODE_TYPE
                ]: translanted_sample_pb
            }
            for translanted_sample_pb in translated_sample_pbs
        ]

        batch = RootedNodeNeighborhoodBatch.collate_pyg_rooted_node_neighborhood_minibatch(
            builder=self.builder,
            graph_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            preprocessed_metadata_pb_wrapper=DEFAULT_HOMOGENEOUS_PREPROCESSED_METADATA_PB_WRAPPER,
            samples=dataloaded_samples,
        )

        # Ensure the batch has 4 nodes (from the added 0,1,2,3),
        # and 4 edges (from the added 0->1, 1->2, 0->2 and 2->3).
        self.assertEqual(batch.graph.num_nodes, 4)
        self.assertEqual(batch.graph.num_edges, 4)
