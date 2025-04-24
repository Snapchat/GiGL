from collections import abc
from typing import Dict, Optional, TypeVar, Union

import torch
from graphlearn_torch.partition import PartitionBook, RangePartitionBook

_KeyType = TypeVar("_KeyType")  # Generic Key Type


def share_memory(
    entity: Optional[
        Union[
            torch.Tensor,
            PartitionBook,
            Dict[_KeyType, torch.Tensor],
            Dict[_KeyType, PartitionBook],
        ]
    ],
) -> None:
    """
    Based on GraphLearn-for-PyTorch's `share_memory` implementation, with additional support for handling empty tensors with share_memory.
        https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/utils/tensor.py#L88

    Calling `share_memory_()` on an empty tensor may cause processes to hang, although the root cause of this is currently unknown. As a result,
    we opt to not move empty tensors to shared memory if they are provided.

    Args:
        entity (Optional[Union[torch.Tensor, Dict[_KeyType, torch.Tensor]]]):
            Homogeneous or heterogeneous entity of tensors which is being moved to shared memory
    """

    if entity is None:
        return None
    elif isinstance(entity, abc.Mapping):
        for entity_tensor in entity.values():
            share_memory(entity_tensor)
    elif isinstance(entity, RangePartitionBook):
        share_memory(entity.partition_bounds)
    else:
        # If the tensor has a dimension which is 0, it is an empty tensor. As a result, we don't move this
        # to shared_memory, since share_memory_() is unsafe on empty tensors, which may cause processes to hang.
        if 0 in entity.shape:
            return None
        entity.share_memory_()
