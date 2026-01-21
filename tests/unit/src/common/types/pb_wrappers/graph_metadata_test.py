import unittest

from parameterized import param, parameterized

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from snapchat.research.gbml import graph_schema_pb2

_NODE_TYPE_USER: str = "user"
_NODE_TYPE_ITEM: str = "item"
_DEFAULT_RELATION = "to"

_EDGE_TYPE_USER_TO_USER_PB: graph_schema_pb2.EdgeType = graph_schema_pb2.EdgeType(
    src_node_type=_NODE_TYPE_USER,
    relation=_DEFAULT_RELATION,
    dst_node_type=_NODE_TYPE_USER,
)
_EDGE_TYPE_USER_TO_ITEM_PB: graph_schema_pb2.EdgeType = graph_schema_pb2.EdgeType(
    src_node_type=_NODE_TYPE_USER,
    relation=_DEFAULT_RELATION,
    dst_node_type=_NODE_TYPE_ITEM,
)

_HOMOGENEOUS_GRAPH_METADATA_PB = graph_schema_pb2.GraphMetadata(
    node_types=[_NODE_TYPE_USER],
    edge_types=[_EDGE_TYPE_USER_TO_USER_PB],
    condensed_node_type_map={0: _NODE_TYPE_USER},
    condensed_edge_type_map={0: _EDGE_TYPE_USER_TO_USER_PB},
)
_HETEROGENEOUS_GRAPH_METADATA_PB = graph_schema_pb2.GraphMetadata(
    node_types=[_NODE_TYPE_USER, _NODE_TYPE_ITEM],
    edge_types=[_EDGE_TYPE_USER_TO_USER_PB, _EDGE_TYPE_USER_TO_ITEM_PB],
    condensed_node_type_map={0: _NODE_TYPE_USER, 1: _NODE_TYPE_ITEM},
    condensed_edge_type_map={
        0: _EDGE_TYPE_USER_TO_USER_PB,
        1: _EDGE_TYPE_USER_TO_ITEM_PB,
    },
)

_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER = GraphMetadataPbWrapper(
    graph_metadata_pb=_HOMOGENEOUS_GRAPH_METADATA_PB
)
_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER = GraphMetadataPbWrapper(
    graph_metadata_pb=_HETEROGENEOUS_GRAPH_METADATA_PB
)


class GraphMetadataUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                graph_metadata_pb_wrapper=_HOMOGENEOUS_GRAPH_METADATA_PB_WRAPPER,
                expected_node_type=_NODE_TYPE_USER,
                expected_edge_type=EdgeType(
                    src_node_type=NodeType(_NODE_TYPE_USER),
                    relation=Relation(_DEFAULT_RELATION),
                    dst_node_type=NodeType(_NODE_TYPE_USER),
                ),
                expected_condensed_node_type=0,
                expected_condensed_edge_type=0,
            ),
        ]
    )
    def test_homogeneous_property_correctness(
        self,
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        expected_node_type: str,
        expected_edge_type: EdgeType,
        expected_condensed_node_type: int,
        expected_condensed_edge_type: int,
    ):
        """
        Tests for success of homogeneous node, edge, condensed node, and condensed edge types with a homogeneous graph
        """
        self.assertEqual(
            graph_metadata_pb_wrapper.homogeneous_node_type, expected_node_type
        )
        self.assertEqual(
            graph_metadata_pb_wrapper.homogeneous_edge_type, expected_edge_type
        )
        self.assertEqual(
            graph_metadata_pb_wrapper.homogeneous_condensed_node_type,
            expected_condensed_node_type,
        )
        self.assertEqual(
            graph_metadata_pb_wrapper.homogeneous_condensed_edge_type,
            expected_condensed_edge_type,
        )

    @parameterized.expand(
        [
            param(
                graph_metadata_pb_wrapper=_HETEROGENEOUS_GRAPH_METADATA_PB_WRAPPER,
            ),
        ]
    )
    def test_homogeneous_property_failure(
        self, graph_metadata_pb_wrapper: GraphMetadataPbWrapper
    ):
        """
        Tests for failure of homogeneous node, edge, condensed node, and condensed edge types with a heterogeneous graph
        """
        with self.assertRaises(ValueError):
            graph_metadata_pb_wrapper.homogeneous_node_type
        with self.assertRaises(ValueError):
            graph_metadata_pb_wrapper.homogeneous_edge_type
        with self.assertRaises(ValueError):
            graph_metadata_pb_wrapper.homogeneous_condensed_node_type
        with self.assertRaises(ValueError):
            graph_metadata_pb_wrapper.homogeneous_condensed_edge_type
