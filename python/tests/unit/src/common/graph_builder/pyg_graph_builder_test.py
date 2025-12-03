import unittest

import torch

from gigl.common.collections.frozen_dict import FrozenDict
from gigl.common.logger import Logger
from gigl.src.common.graph_builder.pyg_graph_builder import PygGraphBuilder
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.types.graph_data import Edge, Node, NodeId, NodeType, Relation

logger = Logger()


class PygGraphBuilderTest(unittest.TestCase):
    def test_can_create_accurate_graph_representation(self):
        pyg_graph_builder = PygGraphBuilder()

        # We build a graph:
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 2, type: 1)
        # Node (id: 2, type: 1) --Relation (type: 1)--> Node (id: 3, type: 2)
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 3, type: 2)
        node_1 = Node(type=NodeType("1"), id=NodeId(1))
        node_2 = Node(type=NodeType("1"), id=NodeId(2))
        node_3 = Node(type=NodeType("2"), id=NodeId(3))
        edge_1 = Edge.from_nodes(
            src_node=node_1, dst_node=node_2, relation=Relation("1")
        )
        edge_2 = Edge.from_nodes(
            src_node=node_1, dst_node=node_3, relation=Relation("1")
        )
        edge_3 = Edge.from_nodes(
            src_node=node_2, dst_node=node_3, relation=Relation("1")
        )
        pyg_graph_builder.add_node(node=node_1, feature_values=torch.tensor([1, 1]))
        pyg_graph_builder.add_node(node=node_2, feature_values=torch.tensor([2, 2]))
        pyg_graph_builder.add_node(node=node_3, feature_values=torch.tensor([3, 3]))

        pyg_graph_builder.add_edge(edge=edge_1, feature_values=torch.tensor([1]))
        pyg_graph_builder.add_edge(edge=edge_2, feature_values=torch.tensor([1]))
        pyg_graph_builder.add_edge(edge=edge_3, feature_values=torch.tensor([1]))

        graph_data_from_builder = pyg_graph_builder.build()

        expected_graph_data = PygGraphData()
        expected_graph_data["1"].x = torch.tensor([[1, 1], [2, 2]])
        expected_graph_data["2"].x = torch.tensor([[3, 3]])
        expected_graph_data["1", "1", "1"].edge_attr = torch.tensor([[1]])
        expected_graph_data["1", "1", "2"].edge_attr = torch.tensor([[1], [1]])
        expected_graph_data["1", "1", "1"].edge_index = torch.LongTensor([[0], [1]])
        expected_graph_data["1", "1", "2"].edge_index = torch.LongTensor(
            [[0, 1], [0, 0]]
        )
        expected_graph_data.global_node_to_subgraph_node_mapping = FrozenDict(
            {
                Node(type=NodeType("1"), id=NodeId(1)): Node(
                    type=NodeType("1"), id=NodeId(0)
                ),
                Node(type=NodeType("1"), id=NodeId(2)): Node(
                    type=NodeType("1"), id=NodeId(1)
                ),
                Node(type=NodeType("2"), id=NodeId(3)): Node(
                    type=NodeType("2"), id=NodeId(0)
                ),
            }
        )
        self.assertEquals(graph_data_from_builder, expected_graph_data)

    def test_can_create_with_with_no_edge_and_node_features(self):
        pyg_graph_builder = PygGraphBuilder()
        # We build a graph:
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 2, type: 1)
        # Node (id: 2, type: 1) --Relation (type: 1)--> Node (id: 3, type: 2)
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 3, type: 2)
        node_1 = Node(type=NodeType("1"), id=NodeId(1))
        node_2 = Node(type=NodeType("1"), id=NodeId(2))
        node_3 = Node(type=NodeType("2"), id=NodeId(3))
        edge_1 = Edge.from_nodes(
            src_node=node_1, dst_node=node_2, relation=Relation("1")
        )
        edge_2 = Edge.from_nodes(
            src_node=node_1, dst_node=node_3, relation=Relation("1")
        )
        edge_3 = Edge.from_nodes(
            src_node=node_2, dst_node=node_3, relation=Relation("1")
        )
        pyg_graph_builder.add_node(
            node=node_1
        )  # No features provided for any nodes or edges
        pyg_graph_builder.add_node(node=node_2)
        pyg_graph_builder.add_node(node=node_3)

        pyg_graph_builder.add_edge(edge=edge_1)
        pyg_graph_builder.add_edge(edge=edge_2)
        pyg_graph_builder.add_edge(edge=edge_3)

        graph_data_from_builder = pyg_graph_builder.build()

        expected_graph_data = PygGraphData()
        expected_graph_data["1", "1", "1"].edge_index = torch.LongTensor([[0], [1]])
        expected_graph_data["1", "1", "2"].edge_index = torch.LongTensor(
            [[0, 1], [0, 0]]
        )
        expected_graph_data.global_node_to_subgraph_node_mapping = FrozenDict(
            {
                Node(type=NodeType("1"), id=NodeId(1)): Node(
                    type=NodeType("1"), id=NodeId(0)
                ),
                Node(type=NodeType("1"), id=NodeId(2)): Node(
                    type=NodeType("1"), id=NodeId(1)
                ),
                Node(type=NodeType("2"), id=NodeId(3)): Node(
                    type=NodeType("2"), id=NodeId(0)
                ),
            }
        )
        # This needs to default to 1 as pyg defaults this to 1 if no features are provided
        # This is a restriction of PyG, that is it expectes node features of atleast size 1
        expected_graph_data["1"].x = torch.ones(2, 1)
        expected_graph_data["2"].x = torch.ones(1, 1)
        self.assertEquals(graph_data_from_builder, expected_graph_data)

    def test_can_create_with_preexisting_data_objects_filtering_existing_nodes_and_edges(
        self,
    ):
        pyg_graph_builder = PygGraphBuilder()

        # graph_data_1 == graph_data_2
        graph_data_1 = PygGraphData()
        graph_data_1["1"].x = torch.tensor([[1, 1], [2, 2]])
        graph_data_1["2"].x = torch.tensor([[3, 3]])
        graph_data_1["1", "1", "1"].edge_attr = torch.tensor([[1]])
        graph_data_1["1", "1", "2"].edge_attr = torch.tensor([[1], [1]])
        graph_data_1["1", "1", "1"].edge_index = torch.LongTensor([[0], [1]])
        graph_data_1["1", "1", "2"].edge_index = torch.LongTensor([[0, 1], [0, 0]])

        graph_data_2 = PygGraphData()
        graph_data_2["1"].x = torch.tensor([[1, 1], [2, 2]])
        graph_data_2["2"].x = torch.tensor([[3, 3]])
        graph_data_2["1", "1", "1"].edge_attr = torch.tensor([[1]])
        graph_data_2["1", "1", "2"].edge_attr = torch.tensor([[1], [1]])
        graph_data_2["1", "1", "1"].edge_index = torch.LongTensor([[0], [1]])
        graph_data_2["1", "1", "2"].edge_index = torch.LongTensor([[0, 1], [0, 0]])

        pyg_graph_builder.add_graph_data(graph_data_1)
        pyg_graph_builder.add_graph_data(graph_data_2)
        graph_data_from_builder = pyg_graph_builder.build()

        graph_data_1.global_node_to_subgraph_node_mapping = FrozenDict(
            {
                Node(type=NodeType("1"), id=NodeId(0)): Node(
                    type=NodeType("1"), id=NodeId(0)
                ),
                Node(type=NodeType("1"), id=NodeId(1)): Node(
                    type=NodeType("1"), id=NodeId(1)
                ),
                Node(type=NodeType("2"), id=NodeId(0)): Node(
                    type=NodeType("2"), id=NodeId(0)
                ),
            }
        )

        self.assertEquals(graph_data_from_builder, graph_data_1)

        # Ensure works when there are no edge features either
        graph_data_1["1", "1", "1"].edge_attr = None
        graph_data_1["1", "1", "2"].edge_attr = None
        graph_data_2["1", "1", "1"].edge_attr = None
        graph_data_2["1", "1", "2"].edge_attr = None
        pyg_graph_builder.add_graph_data(graph_data_1)
        pyg_graph_builder.add_graph_data(graph_data_2)
        graph_data_without_edges_from_builder = pyg_graph_builder.build()
        self.assertEquals(graph_data_without_edges_from_builder, graph_data_1)

    def test_add_subgraph_mapped_graph_data(self):
        pyg_graph_builder = PygGraphBuilder()

        # # We build a graph:
        # Node (id: 10, type: 1) --Relation (type: 1)--> Node (id: 20, type: 1)
        # Node (id: 20, type: 1) --Relation (type: 1)--> Node (id: 30, type: 2)
        # Node (id: 10, type: 1) --Relation (type: 1)--> Node (id: 30, type: 2)
        node_1 = Node(type=NodeType("1"), id=NodeId(10))
        node_2 = Node(type=NodeType("1"), id=NodeId(20))
        node_3 = Node(type=NodeType("2"), id=NodeId(30))
        edge_1 = Edge.from_nodes(
            src_node=node_1, dst_node=node_2, relation=Relation("1")
        )
        edge_2 = Edge.from_nodes(
            src_node=node_1, dst_node=node_3, relation=Relation("1")
        )
        edge_3 = Edge.from_nodes(
            src_node=node_2, dst_node=node_3, relation=Relation("1")
        )
        pyg_graph_builder.add_node(node=node_1, feature_values=torch.tensor([1, 1]))
        pyg_graph_builder.add_node(node=node_2, feature_values=torch.tensor([2, 2]))
        pyg_graph_builder.add_node(node=node_3, feature_values=torch.tensor([3, 3]))

        pyg_graph_builder.add_edge(edge=edge_1, feature_values=torch.tensor([1]))
        pyg_graph_builder.add_edge(edge=edge_2, feature_values=torch.tensor([1]))
        pyg_graph_builder.add_edge(edge=edge_3, feature_values=torch.tensor([1]))
        graph_data_1 = pyg_graph_builder.build()

        # We build a graph:
        # Node (id: 10, type: 1) --Relation (type: 1)--> Node (id: 20, type: 1)
        # Node (id: 20, type: 1) --Relation (type: 1)--> Node (id: 40, type: 2)
        # Node (id: 10, type: 1) --Relation (type: 1)--> Node (id: 40, type: 2)
        # Node (id: 20, type: 1) --Relation (type: 2)--> Node (id: 40, type: 2)
        node_1 = Node(type=NodeType("1"), id=NodeId(10))
        node_2 = Node(type=NodeType("1"), id=NodeId(20))
        node_3 = Node(type=NodeType("2"), id=NodeId(40))
        edge_1 = Edge.from_nodes(
            src_node=node_1, dst_node=node_2, relation=Relation("1")
        )
        edge_2 = Edge.from_nodes(
            src_node=node_1, dst_node=node_3, relation=Relation("1")
        )
        edge_3 = Edge.from_nodes(
            src_node=node_2, dst_node=node_3, relation=Relation("1")
        )
        edge_4 = Edge.from_nodes(
            src_node=node_2, dst_node=node_3, relation=Relation("2")
        )

        pyg_graph_builder.add_node(node=node_1, feature_values=torch.tensor([1, 1]))
        pyg_graph_builder.add_node(node=node_2, feature_values=torch.tensor([2, 2]))
        pyg_graph_builder.add_node(node=node_3, feature_values=torch.tensor([4, 4]))

        pyg_graph_builder.add_edge(edge=edge_1, feature_values=torch.tensor([1]))
        pyg_graph_builder.add_edge(edge=edge_2, feature_values=torch.tensor([1]))
        pyg_graph_builder.add_edge(edge=edge_3, feature_values=torch.tensor([1]))
        pyg_graph_builder.add_edge(edge=edge_4, feature_values=torch.tensor([1000]))

        graph_data_2 = pyg_graph_builder.build()

        pyg_graph_builder.add_graph_data(graph_data_1)
        pyg_graph_builder.add_graph_data(graph_data_2)
        graph_data_from_builder = pyg_graph_builder.build()

        # Expected graph:
        # Node (id: 10, type: 1) --Relation (type: 1)--> Node (id: 20, type: 1)
        # Node (id: 10, type: 1) --Relation (type: 1)--> Node (id: 30, type: 2)
        # Node (id: 10, type: 1) --Relation (type: 1)--> Node (id: 40, type: 2)
        # Node (id: 20, type: 1) --Relation (type: 1)--> Node (id: 30, type: 2)
        # Node (id: 20, type: 1) --Relation (type: 1)--> Node (id: 40, type: 2)
        # Node (id: 20, type: 1) --Relation (type: 2)--> Node (id: 40, type: 2)
        # Note we normalize the node / edge ids below to start at index 0 for each node type

        expected_graph_data = PygGraphData()
        expected_graph_data["1"].x = torch.tensor([[1, 1], [2, 2]])
        expected_graph_data["2"].x = torch.tensor([[3, 3], [4, 4]])
        expected_graph_data["1", "1", "1"].edge_attr = torch.tensor([[1]])
        expected_graph_data["1", "1", "2"].edge_attr = torch.tensor(
            [[1], [1], [1], [1]]
        )
        expected_graph_data["1", "2", "2"].edge_attr = torch.tensor([[1000]])

        expected_graph_data["1", "1", "1"].edge_index = torch.LongTensor([[0], [1]])
        expected_graph_data["1", "1", "2"].edge_index = torch.LongTensor(
            [[0, 1, 0, 1], [0, 0, 1, 1]]
        )
        expected_graph_data["1", "2", "2"].edge_index = torch.LongTensor([[1], [1]])

        # Our expected graph does not have this since it is constructed outside the builder
        graph_data_from_builder.global_node_to_subgraph_node_mapping = FrozenDict({})
        self.assertEquals(graph_data_from_builder, expected_graph_data)

    def test_feature_enforcement_policies(self):
        pyg_graph_builder = PygGraphBuilder()
        node_1 = Node(type=NodeType("1"), id=NodeId(10))
        node_2 = Node(type=NodeType("1"), id=NodeId(20))
        edge_1 = Edge.from_nodes(
            src_node=node_1, dst_node=node_2, relation=Relation("1")
        )
        pyg_graph_builder.add_node(node=node_1, feature_values=torch.tensor([1, 1]))
        adds_edge_with_unregistered_node = lambda: pyg_graph_builder.add_edge(
            edge=edge_1
        )
        self.assertRaises(TypeError, adds_edge_with_unregistered_node)

        pyg_graph_builder = PygGraphBuilder()
        node_1 = Node(type=NodeType("1"), id=NodeId(10))
        node_2 = Node(type=NodeType("1"), id=NodeId(20))
        pyg_graph_builder.add_node(node=node_1, feature_values=torch.tensor([1, 1]))
        adds_node_with_no_feature = lambda: pyg_graph_builder.add_node(node=node_2)
        self.assertRaises(TypeError, adds_node_with_no_feature)

        pyg_graph_builder = PygGraphBuilder()
        node_1 = Node(type=NodeType("1"), id=NodeId(10))
        node_2 = Node(type=NodeType("1"), id=NodeId(20))
        pyg_graph_builder.add_node(node=node_1)
        add_node_with_feature = lambda: pyg_graph_builder.add_node(
            node=node_2, feature_values=torch.tensor([1, 1])
        )
        self.assertRaises(TypeError, add_node_with_feature)

        pyg_graph_builder = PygGraphBuilder()
        node_1 = Node(type=NodeType("1"), id=NodeId(10))
        node_2 = Node(type=NodeType("1"), id=NodeId(20))
        edge_1 = Edge.from_nodes(
            src_node=node_1, dst_node=node_2, relation=Relation("1")
        )
        edge_2 = Edge.from_nodes(
            src_node=node_2, dst_node=node_1, relation=Relation("1")
        )
        pyg_graph_builder.add_node(node=node_1)
        pyg_graph_builder.add_node(node=node_2)
        pyg_graph_builder.add_edge(edge=edge_1, feature_values=torch.tensor([1]))
        add_edge_with_no_feature = lambda: pyg_graph_builder.add_edge(edge=edge_2)
        self.assertRaises(TypeError, add_edge_with_no_feature)

        pyg_graph_builder = PygGraphBuilder()
        node_1 = Node(type=NodeType("1"), id=NodeId(10))
        node_2 = Node(type=NodeType("1"), id=NodeId(20))
        edge_1 = Edge.from_nodes(
            src_node=node_1, dst_node=node_2, relation=Relation("1")
        )
        edge_2 = Edge.from_nodes(
            src_node=node_2, dst_node=node_1, relation=Relation("1")
        )
        pyg_graph_builder.add_node(node=node_1)
        pyg_graph_builder.add_node(node=node_2)
        pyg_graph_builder.add_edge(edge=edge_2)
        add_edge_with_feature = lambda: pyg_graph_builder.add_edge(
            edge=edge_1, feature_values=torch.tensor([1])
        )
        self.assertRaises(TypeError, add_edge_with_feature)
