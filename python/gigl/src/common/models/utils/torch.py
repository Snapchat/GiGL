from typing import Dict, List

import torch

from gigl.src.common.types.graph_data import NodeType


def to_hetero_feat(
    h: torch.Tensor, type_indices: torch.LongTensor, types: List[str]
) -> Dict[NodeType, torch.Tensor]:
    """
    Convert homogeneous graph features into heterogeneous graph feature dict.

    Args:
        h (torch.Tensor): feature tensor for a homogeneous graph
        type_indices (torch.LongTensor): indicates the type of each row in h, corresponding to `types`
        types (list): indicates the possible types

    Returns
        Dict[str, torch.Tensor]: dictionary mapping each type to a tensor of corresponding rows in the heterogeneous graph

    """

    h_dict = {}
    for type_idx, element_type in enumerate(types):
        h_dict[NodeType(element_type)] = h[torch.where(type_indices == type_idx)]
    return h_dict
