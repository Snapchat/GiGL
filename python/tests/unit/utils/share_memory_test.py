import unittest
from collections import abc
from typing import Dict, Optional, Union

import torch
from graphlearn_torch.partition import RangePartitionBook
from parameterized import param, parameterized

from gigl.src.common.types.graph_data import NodeType
from gigl.utils.share_memory import share_memory


class ShareMemoryTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test share_memory when provided entity is None",
                entity=None,
            ),
            param(
                "Test share_memory when provided entity is homogeneous",
                entity=torch.ones(10),
            ),
            param(
                "Test share_memory when provided entity is heterogeneous",
                entity={
                    NodeType("user"): torch.ones(10),
                    NodeType("item"): torch.ones(20) * 2,
                },
            ),
            param(
                "Test share_memory with range partition book",
                entity=RangePartitionBook(
                    partition_ranges=[(0, 3), (3, 5)], partition_idx=0
                ),
            ),
        ]
    )
    def test_share_memory(
        self,
        _,
        entity: Optional[
            Union[torch.Tensor, RangePartitionBook, Dict[NodeType, torch.Tensor]]
        ],
    ):
        share_memory(entity=entity)
        if isinstance(entity, torch.Tensor):
            self.assertTrue(entity.is_shared())
        elif isinstance(entity, RangePartitionBook):
            # If we have a range partition book, we don't move the partition bounds to shared memory, as the shape of this tensor
            # is very small (being at equal in length to the number of machines) and GLT doesn't natively provide support for
            # serializing a range partition book.
            self.assertFalse(entity.partition_bounds.is_shared())
        elif isinstance(entity, abc.Mapping):
            for entity_tensor in entity.values():
                self.assertTrue(entity_tensor.is_shared())

    def test_share_empty_memory(self):
        # If tensors are empty, they should not be moved to shared_memory, as this may lead to transient failures, which may cause processes to hang.

        # 1D Empty Tensor
        empty_1d_tensor = torch.empty(0)
        share_memory(empty_1d_tensor)

        self.assertFalse(empty_1d_tensor.is_shared())

        # 2D Empty Tensor
        empty_2d_tensor = torch.empty((5, 0))
        share_memory(empty_2d_tensor)
        self.assertFalse(empty_2d_tensor.is_shared())
