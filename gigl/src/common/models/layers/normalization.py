from typing import Union, cast

import torch
from torch.nn import functional as F

from gigl.src.common.types.graph_data import NodeType


def l2_normalize_embeddings(
    node_typed_embeddings: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
    if isinstance(node_typed_embeddings, dict):
        node_typed_embeddings_dict = cast(
            "dict[NodeType, torch.Tensor]", node_typed_embeddings
        )  # ty#2374 workaround
        for node_type in node_typed_embeddings_dict:
            node_typed_embeddings_dict[node_type] = F.normalize(
                node_typed_embeddings_dict[node_type],
                p=2,
                dim=-1,
            )
        return node_typed_embeddings_dict
    elif isinstance(node_typed_embeddings, torch.Tensor):
        return F.normalize(node_typed_embeddings, p=2, dim=-1)
    else:
        raise ValueError(
            f"Expected type torch.Tensor or dict[NodeType, torch.Tensor], got type {type(node_typed_embeddings)}"
        )
