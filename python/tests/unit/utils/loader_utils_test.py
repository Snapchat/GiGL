import unittest

import torch
from parameterized import param, parameterized

from gigl.distributed.utils.loader import shard_nodes_by_process
from tests.test_assets.distributed.utils import assert_tensor_equality

_TEST_TENSOR = torch.Tensor([1, 3, 5, 7, 9, 11, 13, 15, 17])


class ShareMemoryTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test shard_nodes_by_process on 0 rank",
                input_tensor=_TEST_TENSOR,
                local_process_rank=0,
                local_world_size=2,
                expected_sharded_tensor=torch.Tensor([1, 3, 5, 7]),
            ),
            param(
                "Test shard_nodes_by_process on 1 rank",
                input_tensor=_TEST_TENSOR,
                local_process_rank=1,
                local_world_size=2,
                expected_sharded_tensor=torch.Tensor([9, 11, 13, 15, 17]),
            ),
        ]
    )
    def test_shard_nodes_by_process(
        self,
        _,
        input_tensor: torch.Tensor,
        local_process_rank: int,
        local_world_size: int,
        expected_sharded_tensor: torch.Tensor,
    ):
        sharded_tensor = shard_nodes_by_process(
            input_nodes=input_tensor,
            local_process_rank=local_process_rank,
            local_world_size=local_world_size,
        )
        assert_tensor_equality(sharded_tensor, expected_sharded_tensor)
