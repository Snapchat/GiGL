import json
import re

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation

logger = Logger()

_EDGE_TYPE_REGEX_PATTERN = r"([^-]+)-([^-]+)-([^-]+)"


def _parse_edge_type(edge_type_str: str) -> EdgeType:
    match = re.match(_EDGE_TYPE_REGEX_PATTERN, edge_type_str)
    if match:
        result = match.groups()
    else:
        raise ValueError(
            f"Failed to parse edge type: {edge_type_str}. Please ensure edge types are provided in format such as 'SRC-RELATION-DST', \
            where SRC is the source node type string, RELATION is the relation string, and DST is the destination node type string."
        )
    return EdgeType(
        src_node_type=NodeType(result[0]),
        relation=Relation(result[1]),
        dst_node_type=NodeType(result[2]),
    )

def _validate_parsed_fanout(parsed_fanout: list[int]) -> None:
    if not isinstance(parsed_fanout, list):
        raise ValueError(f"Parsed fanout expected to be a list, got {parsed_fanout} of type {type(parsed_fanout)}")
    for item in parsed_fanout:
        if not isinstance(item, int):
            raise ValueError(f"Fanout must contain integers, got {item} of type {type(item)}")

def parse_fanout(fanout_str: str) -> dict[str, list[int]]:
    try:
        parsed_fanout: dict[str, list[int]] = json.loads(fanout_str)
    except json.decoder.JSONDecodeError:
        raise ValueError(f"Failed to parse provided fanout string: {fanout_str}. Please ensure the provided string is well-formed as a json.")
    fanout = {}
    for edge_type_str, parsed_fanout in parsed_fanout.items():
        _validate_parsed_fanout(parsed_fanout=parsed_fanout)
        edge_type = _parse_edge_type(edge_type_str)
        fanout[edge_type] = parsed_fanout
    fanout_len = next(iter(fanout.values()))
    for
    logger.info(f"Parsed fanout from args: {fanout}")
    return fanout


if __name__ == "__main__":
    fanout_str = '{"c-to-d": [10, 20, 30], "a-to-b": [10, 11, 12], "e-to-f": [5, 5, 5]}'
    fanout = parse_fanout(fanout_str=fanout_str)
