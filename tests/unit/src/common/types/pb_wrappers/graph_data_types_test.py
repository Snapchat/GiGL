import pickle
import unittest
from typing import Union

from gigl.src.common.types.pb_wrappers.graph_data_types import (
    _HASH_CACHE_KEY,
    EdgePbWrapper,
    GraphPbWrapper,
    NodePbWrapper,
)
from snapchat.research.gbml import graph_schema_pb2


class GraphDataTypesTest(unittest.TestCase):
    def test_node_pb_wrapper_equality(self):
        """
        Node pbs can be equal field-wise, but native pb __eq__ does not reflect their equality.
        NodePbWrapper equality checks should correct for this.
        :return:
        """

        a_pb = graph_schema_pb2.Node(node_id=1, condensed_node_type=0)
        b_pb = graph_schema_pb2.Node(node_id=1)
        # These two nodes have the same condensed node type, but one is explicit and the other implicit.
        self.assertEqual(a_pb.condensed_node_type, b_pb.condensed_node_type)
        # Despite having the same field values, the nodes are not pb-equal.
        self.assertNotEqual(a_pb, b_pb)

        a_wrap = NodePbWrapper(pb=a_pb)
        b_wrap = NodePbWrapper(pb=b_pb)

        # They should be equal according to the NodePbWrapper equality check.
        self.assertEqual(a_wrap, b_wrap)

    def test_edge_pb_wrapper_equality(self):
        """
        Edge pbs can be equal field-wise, but native pb __eq__ does not reflect their equality.
        EdgePbWrapper equality checks should correct for this.
        :return:
        """

        a_pb = graph_schema_pb2.Edge(
            src_node_id=1, dst_node_id=2, condensed_edge_type=0
        )
        b_pb = graph_schema_pb2.Edge(src_node_id=1, dst_node_id=2)

        # These two edge pbs have the same condensed edge type, but one is explicit and the other implicit.
        self.assertEqual(a_pb.condensed_edge_type, b_pb.condensed_edge_type)
        # Despite having the same field values, the edges are not pb-equal
        self.assertNotEqual(a_pb, b_pb)

        a_wrap = EdgePbWrapper(pb=a_pb)
        b_wrap = EdgePbWrapper(pb=b_pb)

        # They should be equal according to the EdgePbWrapper equality check.
        self.assertEqual(a_wrap, b_wrap)

    def test_graph_pb_wrapper_equality(self):
        """
        Two graph pbs can have the same nodes/edges in different orders, but native pb __eq__
        does not reflect their equality.
        GraphPbWrapper equality checks should correct for this.
        :return:
        """

        n1_pb = graph_schema_pb2.Node(node_id=1)
        n2_pb = graph_schema_pb2.Node(node_id=2)
        e12_pb = graph_schema_pb2.Edge(src_node_id=1, dst_node_id=2)
        e21_pb = graph_schema_pb2.Edge(dst_node_id=2, src_node_id=1)

        # These pbs represent equivalent graphs (same nodes, same edges), but are not pb-equal.
        g1_pb = graph_schema_pb2.Graph(nodes=[n1_pb, n2_pb], edges=[e12_pb, e21_pb])
        g2_pb = graph_schema_pb2.Graph(nodes=[n2_pb, n1_pb], edges=[e21_pb, e12_pb])
        self.assertNotEqual(g1_pb, g2_pb)

        # They should be equal according to the GraphPbWrapper equality check.
        g1_wrap = GraphPbWrapper(pb=g1_pb)
        g2_wrap = GraphPbWrapper(pb=g2_pb)
        self.assertEqual(g1_wrap, g2_wrap)

    def _test_cached_hash_helper(
        self, pb_wrapper: Union[NodePbWrapper, EdgePbWrapper, GraphPbWrapper]
    ):
        # Check that the hash is indeed cached after the first call.
        is_hash_cached_before = getattr(pb_wrapper, _HASH_CACHE_KEY, None) is not None
        _ = hash(pb_wrapper)
        is_hash_cached_after = getattr(pb_wrapper, _HASH_CACHE_KEY, None) is not None
        self.assertFalse(is_hash_cached_before)
        self.assertTrue(is_hash_cached_after)

        # Check that pickling does not also persist the cached hash.
        pb_wrapper_2 = pickle.loads(pickle.dumps(pb_wrapper))
        is_hash_cached_after_pickle = (
            getattr(pb_wrapper_2, _HASH_CACHE_KEY, None) is not None
        )
        self.assertFalse(is_hash_cached_after_pickle)

        # Sanity check that the two objects are still deemed equal due to their pb properties.
        are_pbs_equal = pb_wrapper == pb_wrapper_2
        self.assertTrue(are_pbs_equal)

    def test_node_pb_wrapper_caches_hash(self):
        pb_wrapper = NodePbWrapper(pb=graph_schema_pb2.Node(node_id=1))
        self._test_cached_hash_helper(pb_wrapper=pb_wrapper)

    def test_edge_pb_wrapper_caches_hash(self):
        pb_wrapper = EdgePbWrapper(
            pb=graph_schema_pb2.Edge(src_node_id=1, dst_node_id=2)
        )
        self._test_cached_hash_helper(pb_wrapper=pb_wrapper)

    def test_graph_pb_wrapper_caches_hash(self):
        n1_pb = graph_schema_pb2.Node(node_id=1)
        e12_pb = graph_schema_pb2.Edge(src_node_id=1, dst_node_id=2)
        pb_wrapper = GraphPbWrapper(
            pb=graph_schema_pb2.Graph(nodes=[n1_pb], edges=[e12_pb])
        )
        self._test_cached_hash_helper(pb_wrapper=pb_wrapper)
