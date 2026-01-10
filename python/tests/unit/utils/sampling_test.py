import unittest

from parameterized import param, parameterized

from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.utils.sampling import parse_fanout


class SamplingTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Correctly parses a list fanout",
                input_fanout="[10, 15, 20]",
                expected_fanout=[10, 15, 20],
            ),
            param(
                "Correctly parses a dictionary fanout",
                input_fanout='{("user", "to", "user"): [10, 10],("user", "to", "item"): [20, 20]}',
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
        expected_fanout: list[int] | dict[EdgeType, list[int]],
    ):
        output_fanout = parse_fanout(fanout_str=input_fanout)
        self.assertEqual(output_fanout, expected_fanout)

    @parameterized.expand(
        [
            param(
                "Fails when list json is not well formed",
                input_fanout="[10, 15, 20)",
                expected_error_type=SyntaxError,
            ),
            param(
                "Fails when dict json is not well formed, missing last quote for `user-to-user` destination node type",
                input_fanout='{("user", "to", "user: [10, 10], ("user", "to", "item"): [20, 20]}',
                expected_error_type=SyntaxError,
            ),
            param(
                "Fails when neither list nor dict is provided from parsed json",
                input_fanout="5",
                expected_error_type=ValueError,
            ),
            param(
                "Fails when list contains non-integers, string",
                input_fanout='["10", 15, 20]',
                expected_error_type=ValueError,
            ),
            param(
                "Fails when list contains non-integers, float",
                input_fanout="[10.0, 15.0, 20.0]",
                expected_error_type=ValueError,
            ),
            param(
                "Fails when dict fanout contains non-integers",
                input_fanout='{"user-to-user": ["10", 10], "user-to-item": [20, 20]}',
                expected_error_type=ValueError,
            ),
            param(
                "Fails when edge type is not a tuple",
                input_fanout='{"123": [10, 10], "456": [20, 20]}',
                expected_error_type=ValueError,
            ),
            param(
                "Fails when edge type is not a tuple of length 3",
                input_fanout='{("user"): [10, 10], ("item"): [20, 20]}',
                expected_error_type=ValueError,
            ),
            param(
                "Fails when edge type fields are not strings",
                input_fanout='{("user", "to", 5): [10, 10], ("user", 5, "item"): [20, 20]}',
                expected_error_type=ValueError,
            ),
        ]
    )
    def test_parse_fanout_failure(
        self, _, input_fanout: str, expected_error_type: type[BaseException]
    ):
        with self.assertRaises(expected_error_type):
            parse_fanout(fanout_str=input_fanout)
