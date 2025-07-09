import json
import re
from typing import Any, Union

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation

logger = Logger()

_EDGE_TYPE_REGEX_PATTERN = r"([^-]+)-([^-]+)-([^-]+)"


def _parse_edge_type(edge_type_str: str) -> EdgeType:
    """
    Parses the edge type from a provided edge type string. The edge type must be must be of form 'SRC-RELATION-DST',
    where SRC is the source node type string, RELATION is the relation string, and DST is the destination node type string.

    Args:
        edge_type_str (str): The edge type string to be parsed as an EdgeType
    Returns:
        EdgeType: The parsed edge type
    """
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


def _validate_parsed_fanout(parsed_fanout: Any) -> None:
    """
    Validates that the parsed fanout from the json is correctly specified as a list of integers.

    Args:
        parsed_fanout (Any): Fanout which is expected to be a list of integers
    """
    if not isinstance(parsed_fanout, list):
        raise ValueError(
            f"Parsed fanout expected to be a list, got {parsed_fanout} of type {type(parsed_fanout)}"
        )
    for item in parsed_fanout:
        if not isinstance(item, int):
            raise ValueError(
                f"Fanout must contain integers, got {item} of type {type(item)}"
            )


def parse_fanout(fanout_str: str) -> Union[list[int], dict[EdgeType, list[int]]]:
    """
    Parses fanout from a json string. The fanout string provided must be a well-formed and must be provided as a list[int] or as a
    dict[str, list[int]]. In the case of the dictionary specification, the keys must be of form 'SRC-RELATION-DST',
    where SRC is the source node type string, RELATION is the relation string, and DST is the destination node type string.

    For example, to parse a list[int], one could provide a fanout_str such as
        [10, 15, 20]

    To parse a dict[EdgeType, list[int]], one could provide a fanout_str such as
        {"user-to-user": [10, 10], "user-to-item": [20, 20]}

    Args:
        fanout_str (str): Json string to be parsed into fanout
    Returns:
        Union[list[int], dict[EdgeType, list[int]]]: Either a list of fanout per hop of a dictionary of edge types to their respective fanouts per hop
    """
    try:
        loaded_fanout = json.loads(fanout_str)
    except json.decoder.JSONDecodeError:
        raise ValueError(
            f"Failed to parse provided fanout string: {fanout_str}. Please ensure the provided string is well-formed as a json."
        )
    if isinstance(loaded_fanout, list):
        _validate_parsed_fanout(parsed_fanout=loaded_fanout)
        logger.info(f"Parsed list fanout from args: {loaded_fanout}")
        return loaded_fanout
    elif isinstance(loaded_fanout, dict):
        fanout: dict[EdgeType, list[int]] = {}
        for edge_type_str, parsed_fanout in loaded_fanout.items():
            _validate_parsed_fanout(parsed_fanout=parsed_fanout)
            edge_type = _parse_edge_type(edge_type_str)
            fanout[edge_type] = parsed_fanout
        fanout_len = len(next(iter(fanout.values())))
        for edge_type, fanout_list in fanout.items():
            if len(fanout_list) != fanout_len:
                raise ValueError(
                    f"Found a fanout length {fanout_list} for edge type {edge_type} which is different from earlier fanout length {fanout_len}. \
                    Please ensure all fanouts have the same number of hops."
                )
        logger.info(f"Parsed dictionary fanout from args: {fanout}")
        return fanout
    else:
        raise ValueError(
            f"Fanout must be parsed as either a dictionary or a list, got {loaded_fanout} of type {type(loaded_fanout)}"
        )
