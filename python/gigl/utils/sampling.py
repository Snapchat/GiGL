import ast
from typing import Any, Union

from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType

logger = Logger()


def _validate_parsed_edge_type(parsed_edge_type: Any) -> None:
    """
    Validates that the parsed edge type is correctly a tuple[str, str, str], denoting an edge type.
    Args:
        parsed_edge_type (Any): Edge type which is expected to be a tuple[str, str, str], corresponding to the source node type, relation, and destination node type, respectively.
    Raises:
        ValueError: if not a tuple
        ValueError: if tuple has a length which is not equal to 3
        ValueError: if not all elements of the tuple are strings
    """
    if not isinstance(parsed_edge_type, tuple) or len(parsed_edge_type) != 3:
        raise ValueError(
            f"Parsed edge type expected to be a tuple[str, str, str], got {parsed_edge_type}"
        )
    if not all([isinstance(edge_type, str) for edge_type in parsed_edge_type]):
        raise ValueError(
            f"Edge type must a tuple[str, str, str] integers, got {parsed_edge_type}"
        )


def _validate_parsed_hops(parsed_fanout: Any) -> None:
    """
    Validates that the parsed fanout is correctly specified as a list of integers.

    Args:
        parsed_fanout (Any): Fanout which is expected to be a list of integers
    Raises:
        ValueError: if not a list
        ValueError: if not all elements of the list are ints
    """
    if not isinstance(parsed_fanout, list):
        raise ValueError(
            f"Parsed fanout expected to be a list, got {parsed_fanout} of type {type(parsed_fanout)}"
        )
    if not all([isinstance(hop, int) for hop in parsed_fanout]):
        raise ValueError(f"Fanout must contain integers, got {parsed_fanout}")


def parse_fanout(fanout_str: str) -> Union[list[int], dict[EdgeType, list[int]]]:
    """
    Parses fanout from a string. The fanout string should be equivalent to a str(list[int]) or a
    str(dict[tuple[str, str, str], list[int]]), where each item in the tuple corresponds to the source node type, relation, and destination node type, respectively.

    For example, to parse a list[int], one could provide a fanout_str such as
        '[10, 15, 20]'

    To parse a dict[EdgeType, list[int]], one could provide a fanout_str such as
        '{("user", "to", "user"): [10, 10], ("user", "to", "item"): [20, 20]}'

    Args:
        fanout_str (str): Provided string to be parsed into fanout
    Returns:
        Union[list[int], dict[EdgeType, list[int]]]: Either a list of fanout per hop of a dictionary of edge types to their respective fanouts per hop
    """

    loaded_fanout = ast.literal_eval(fanout_str)
    if isinstance(loaded_fanout, list):
        _validate_parsed_hops(parsed_fanout=loaded_fanout)
        logger.info(f"Parsed list fanout from args: {loaded_fanout}")
        return loaded_fanout
    elif isinstance(loaded_fanout, dict):
        fanout: dict[EdgeType, list[int]] = {}
        for parsed_edge_type, parsed_fanout in loaded_fanout.items():
            _validate_parsed_edge_type(parsed_edge_type=parsed_edge_type)
            _validate_parsed_hops(parsed_fanout=parsed_fanout)
            edge_type = EdgeType(
                src_node_type=parsed_edge_type[0],
                relation=parsed_edge_type[1],
                dst_node_type=parsed_edge_type[2],
            )
            fanout[edge_type] = parsed_fanout
        return fanout
    else:
        raise ValueError(
            f"Fanout must be parsed as either a dictionary or a list, got {loaded_fanout} of type {type(loaded_fanout)}"
        )
