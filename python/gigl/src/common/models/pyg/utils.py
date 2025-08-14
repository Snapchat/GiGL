import inspect
from typing import Any, Dict, Iterable, Tuple

import torch_geometric

# List of arguments that can be passed to the base class of a PyG message passing layer.
MESSAGE_PASSING_BASE_CLS_ARGS = list(
    inspect.signature(torch_geometric.nn.conv.MessagePassing).parameters.keys()
)


def filter_dict(
    input_dict: Dict[str, Any], keys_to_keep: Iterable[str] = []
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Filters out certain items from an input directory based on keys to keep.

    Args:
        input_dict: Input dictionary.
        keys_to_keep: Iterable of keys to keep from the input dictionary (all others will be discarded).

    Returns:
        remaining_kwargs: Dictionary containing the remaining keyword arguments.
        discarded_kwargs: Dictionary containing the discarded keyword arguments.
    """

    remaining_kwargs = {
        key: value for key, value in input_dict.items() if key in keys_to_keep
    }
    discarded_kwargs = {
        key: value for key, value in input_dict.items() if key not in keys_to_keep
    }
    return remaining_kwargs, discarded_kwargs
