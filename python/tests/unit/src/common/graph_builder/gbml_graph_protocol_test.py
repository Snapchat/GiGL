import unittest

from gigl.common.logger import Logger
from gigl.src.common.graph_builder.gbml_graph_protocol import GbmlGraphDataProtocol
from gigl.src.common.graph_builder.pyg_graph_builder import PygGraphBuilder
from gigl.src.common.types.graph_data import Edge, Node, NodeId, NodeType, Relation

logger = Logger()


class GbmlGraphProtocolTest(unittest.TestCase):
    def setUp(self):
        # graph -> used for `are_same_graph`:
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 2, type: 1)
        # Node (id: 2, type: 1) --Relation (type: 1)--> Node (id: 3, type: 2)
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 3, type: 2)
        self.node_1 = Node(type=NodeType("1"), id=NodeId(1))
        self.node_2 = Node(type=NodeType("1"), id=NodeId(2))
        self.node_3 = Node(type=NodeType("2"), id=NodeId(3))
        self.edge_1 = Edge.from_nodes(
            src_node=self.node_1, dst_node=self.node_2, relation=Relation("1")
        )
        self.edge_2 = Edge.from_nodes(
            src_node=self.node_1, dst_node=self.node_3, relation=Relation("1")
        )
        self.edge_3 = Edge.from_nodes(
            src_node=self.node_2, dst_node=self.node_3, relation=Relation("1")
        )
        # graph -> used for `are_disjoint`:
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 2, type: 1)
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 3, type: 1)
        # Node (id: 2, type: 1) --Relation (type: 1)--> Node (id: 3, type: 1)
        # Node (id: 4, type: 1) --Relation (type: 1)--> Node (id: 5, type: 1)
        self.node_11 = Node(type=NodeType("1"), id=NodeId(1))
        self.node_22 = Node(type=NodeType("1"), id=NodeId(2))
        self.node_33 = Node(type=NodeType("1"), id=NodeId(3))
        self.node_44 = Node(type=NodeType("1"), id=NodeId(4))
        self.node_55 = Node(type=NodeType("1"), id=NodeId(5))
        self.edge_11 = Edge.from_nodes(
            src_node=self.node_11, dst_node=self.node_22, relation=Relation("1")
        )
        self.edge_22 = Edge.from_nodes(
            src_node=self.node_11, dst_node=self.node_33, relation=Relation("1")
        )
        self.edge_33 = Edge.from_nodes(
            src_node=self.node_22, dst_node=self.node_33, relation=Relation("1")
        )

        self.edge_44 = Edge.from_nodes(
            src_node=self.node_44, dst_node=self.node_55, relation=Relation("1")
        )

    def test_are_same_graph_data_pyg_builder(self):
        pyg_graph_builder = PygGraphBuilder()
        # We build a graph:
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 2, type: 1)
        # Node (id: 2, type: 1) --Relation (type: 1)--> Node (id: 3, type: 2)
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 3, type: 2)

        # No features provided for any nodes or edges
        pyg_graph_builder.add_node(node=self.node_1)
        pyg_graph_builder.add_node(node=self.node_2)
        pyg_graph_builder.add_node(node=self.node_3)

        pyg_graph_builder.add_edge(edge=self.edge_1)
        pyg_graph_builder.add_edge(edge=self.edge_2)
        pyg_graph_builder.add_edge(edge=self.edge_3)

        graph_data_from_builder_1 = pyg_graph_builder.build()

        pyg_graph_builder.add_node(node=self.node_3)
        pyg_graph_builder.add_node(node=self.node_2)
        pyg_graph_builder.add_node(node=self.node_1)

        pyg_graph_builder.add_edge(edge=self.edge_2)
        pyg_graph_builder.add_edge(edge=self.edge_3)
        pyg_graph_builder.add_edge(edge=self.edge_1)

        graph_data_from_builder_2 = pyg_graph_builder.build()

        # Since the graphs are build in different order, their internal representations might not be strictly
        # equal; but they should be similar when mapped to a global namespace
        self.assertNotEqual(graph_data_from_builder_1, graph_data_from_builder_2)
        self.assertTrue(
            GbmlGraphDataProtocol.are_same_graph(
                graph_data_from_builder_1, graph_data_from_builder_2
            )
        )

    def test_are_disjoint(self):
        pyg_graph_builder = PygGraphBuilder()

        # We build a graph `graph_data_from_builder_1`:
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 2, type: 1)
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 3, type: 1)
        # Node (id: 2, type: 1) --Relation (type: 1)--> Node (id: 3, type: 1)

        # No features provided for any nodes or edges
        pyg_graph_builder.add_node(node=self.node_11)
        pyg_graph_builder.add_node(node=self.node_22)
        pyg_graph_builder.add_node(node=self.node_33)

        pyg_graph_builder.add_edge(edge=self.edge_11)
        pyg_graph_builder.add_edge(edge=self.edge_22)
        pyg_graph_builder.add_edge(edge=self.edge_33)

        graph_data_from_builder_1 = pyg_graph_builder.build()

        # We build a secondary graph `graph_data_from_builder_2`:
        # Node (id: 1, type: 1) --Relation (type: 1)--> Node (id: 2, type: 1)
        # and check that it *is not* disjoint from `graph_data_from_builder_1`.
        pyg_graph_builder.add_node(node=self.node_11)
        pyg_graph_builder.add_node(node=self.node_22)
        pyg_graph_builder.add_edge(edge=self.edge_11)
        graph_data_from_builder_2 = pyg_graph_builder.build()

        self.assertFalse(
            GbmlGraphDataProtocol.are_disjoint(
                a=graph_data_from_builder_1, b=graph_data_from_builder_2
            )
        )

        # We build a tertiary graph `graph_data_from_builder_3`:
        # Node (id: 4, type: 1) --Relation (type: 1)--> Node (id: 5, type: 1)
        # and check that it *is* disjoint from `graph_data_from_builder_1`.

        pyg_graph_builder.add_node(node=self.node_44)
        pyg_graph_builder.add_node(node=self.node_55)

        pyg_graph_builder.add_edge(edge=self.edge_44)
        graph_data_from_builder_3 = pyg_graph_builder.build()

        self.assertTrue(
            GbmlGraphDataProtocol.are_disjoint(
                a=graph_data_from_builder_1, b=graph_data_from_builder_3
            )
        )
