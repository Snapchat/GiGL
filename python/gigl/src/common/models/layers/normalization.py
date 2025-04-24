from typing import Dict, Union

import torch
from torch.nn import functional as F

from gigl.src.common.types.graph_data import NodeType


def l2_normalize_embeddings(
    node_typed_embeddings: Union[torch.Tensor, Dict[NodeType, torch.Tensor]]
) -> Union[torch.Tensor, Dict[NodeType, torch.Tensor]]:
    if isinstance(node_typed_embeddings, dict):
        for node_type in node_typed_embeddings:
            node_typed_embeddings[node_type] = F.normalize(
                node_typed_embeddings[node_type], p=2, dim=-1
            )
    elif isinstance(node_typed_embeddings, torch.Tensor):
        node_typed_embeddings = F.normalize(node_typed_embeddings, p=2, dim=-1)
    else:
        raise ValueError(
            f"Expected type torch.Tensor or Dict[NodeType, torch.Tensor], got type {type(node_typed_embeddings)}"
        )
    return node_typed_embeddings
