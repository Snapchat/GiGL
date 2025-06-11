import unittest

import torch
from parameterized import param, parameterized
from torch_geometric.data import HeteroData

from gigl.distributed.utils.loader import (
    remove_labeled_edge_types,
    shard_nodes_by_process,
)
from tests.test_assets.distributed.utils import assert_tensor_equality

_TEST_TENSOR = torch.Tensor([1, 3, 5, 7, 9, 11, 13, 15, 17])


class ShareMemoryTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test shard_nodes_by_process on 0 rank",
                input_tensor=_TEST_TENSOR,
                local_process_rank=0,
                local_process_world_size=2,
                expected_sharded_tensor=torch.Tensor([1, 3, 5, 7]),
            ),
            param(
                "Test shard_nodes_by_process on 1 rank",
                input_tensor=_TEST_TENSOR,
                local_process_rank=1,
                local_process_world_size=2,
                expected_sharded_tensor=torch.Tensor([9, 11, 13, 15, 17]),
            ),
        ]
    )
    def test_shard_nodes_by_process(
        self,
        _,
        input_tensor: torch.Tensor,
        local_process_rank: int,
        local_process_world_size: int,
        expected_sharded_tensor: torch.Tensor,
    ):
        sharded_tensor = shard_nodes_by_process(
            input_nodes=input_tensor,
            local_process_rank=local_process_rank,
            local_process_world_size=local_process_world_size,
        )
        assert_tensor_equality(sharded_tensor, expected_sharded_tensor)

    def test_remove_labeled_edge_types(self):
        u2s_edge_type = ("user", "to", "story")
        s2u_edge_type = ("story", "to", "user")
        labeled_edge_type = ("user", "to_gigl_positive", "story")
        data = HeteroData()
        data[u2s_edge_type].edge_index = torch.Tensor([[0, 0], [1, 1]])
        data[s2u_edge_type].edge_index = torch.Tensor([[2, 2], [3, 3]])
        data[labeled_edge_type].edge_index = torch.Tensor([[4, 4], [5, 5]])
        data.num_sampled_edges = {
            u2s_edge_type: [2, 2],
            s2u_edge_type: [3, 1],
            labeled_edge_type: [0, 0],
        }
        data = remove_labeled_edge_types(data)
        for edge_type in (u2s_edge_type, s2u_edge_type):
            self.assertIn(edge_type, data.num_sampled_edges)
            self.assertIn(edge_type, data._edge_store_dict)
            self.assertIn(edge_type, data.edge_types)
        self.assertNotIn(labeled_edge_type, data.num_sampled_edges)
        self.assertNotIn(labeled_edge_type, data._edge_store_dict)
        self.assertNotIn(labeled_edge_type, data.edge_types)
