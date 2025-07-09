import unittest
from typing import Union

from parameterized import param, parameterized

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.utils.parse_fanout import parse_fanout


class ParseFanoutTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Correctly parses a list fanout",
                input_fanout="[10, 15, 20]",
                expected_fanout=[10, 15, 20],
            ),
            param(
                "Correctly parses a dictionary fanout",
                input_fanout='{"user-to-user": [10, 10], "user-to-item": [20, 20]}',
                expected_fanout={
                    EdgeType(NodeType("user"), Relation("to"), NodeType("user")): [
                        10,
                        10,
                    ],
                    EdgeType(NodeType("user"), Relation("to"), NodeType("item")): [
                        20,
                        20,
                    ],
                },
            ),
        ]
    )
    def test_parse_fanout_success(
        self,
        _,
        input_fanout: str,
        expected_fanout: Union[list[int], dict[EdgeType, list[int]]],
    ):
        output_fanout = parse_fanout(fanout_str=input_fanout)
        self.assertEqual(output_fanout, expected_fanout)

    @parameterized.expand(
        [
            param(
                "Fails when list json is not well formed",
                input_fanout="[10, 15, 20)",
            ),
            param(
                "Fails when dict json is not well formed",
                input_fanout='{"user-to-user": [10, 10], "user-to-item"; [20, 20]}',
            ),
            param(
                "Fails when neither list nor dict is provided from parsed json",
                input_fanout="5",
            ),
            param(
                "Fails when list contains non-integers",
                input_fanout='["10", 15, 20]',
            ),
            param(
                "Fails when dict fanout contains non-integers",
                input_fanout='{"user-to-user": ["10", 10], "user-to-item": [20, 20]}',
            ),
            param(
                "Fails when edge type is not correctly provided",
                input_fanout='{"user_to_user": [10, 10], "user-to-item": [20, 20]}',
            ),
            param(
                "Fails when edge types contain different number of hops",
                input_fanout='{"user_to_user": [10, 10], "user-to-item": [20, 20, 30]}',
            ),
        ]
    )
    def test_parse_fanout_failure(
        self,
        _,
        input_fanout: str,
    ):
        with self.assertRaises(ValueError):
            parse_fanout(fanout_str=input_fanout)
