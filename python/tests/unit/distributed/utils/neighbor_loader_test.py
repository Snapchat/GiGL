import unittest

import torch
from parameterized import param, parameterized
from torch_geometric.typing import EdgeType

from gigl.distributed.utils.neighbor_loader import (
    shard_nodes_by_process,
    zero_label_edge_fanout,
)
from tests.test_assets.distributed.utils import assert_tensor_equality

_U2I_EDGE_TYPE = ("user", "to", "item")
_I2U_EDGE_TYPE = ("item", "to", "user")
_LABELED_EDGE_TYPE = ("user", "to_gigl_positive", "item")


class LoaderUtilsTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test shard_nodes_by_process on 0 rank",
                local_process_rank=0,
                local_process_world_size=2,
                expected_sharded_tensor=torch.tensor([1, 3, 5, 7]),
            ),
            param(
                "Test shard_nodes_by_process on 1 rank",
                local_process_rank=1,
                local_process_world_size=2,
                expected_sharded_tensor=torch.tensor([9, 11, 13, 15, 17]),
            ),
        ]
    )
    def test_shard_nodes_by_process(
        self,
        _,
        local_process_rank: int,
        local_process_world_size: int,
        expected_sharded_tensor: torch.Tensor,
    ):
        sharded_tensor = shard_nodes_by_process(
            input_nodes=torch.tensor([1, 3, 5, 7, 9, 11, 13, 15, 17]),
            local_process_rank=local_process_rank,
            local_process_world_size=local_process_world_size,
        )
        assert_tensor_equality(sharded_tensor, expected_sharded_tensor)

    @parameterized.expand(
        [
            param(
                "Test set_labeled_edge_type on num_neighbors dict with labeled edge type",
                num_neighbors={_U2I_EDGE_TYPE: [2, 7], _I2U_EDGE_TYPE: [3, 4]},
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                    _LABELED_EDGE_TYPE: [0, 0],
                },
            ),
            param(
                "Test set_labeled_edge_type on num_neighbors dict with no labeled edge type",
                num_neighbors={_U2I_EDGE_TYPE: [2, 7], _I2U_EDGE_TYPE: [3, 4]},
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                },
            ),
        ]
    )
    def test_zero_label_edge_fanout(
        self,
        _,
        num_neighbors: dict[EdgeType, list[int]],
        expected_num_neighbors: dict[EdgeType, list[int]],
    ):
        num_neighbors = zero_label_edge_fanout(num_neighbors=num_neighbors)
        self.assertEqual(num_neighbors, expected_num_neighbors)
