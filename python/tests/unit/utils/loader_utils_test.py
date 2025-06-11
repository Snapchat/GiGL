import unittest

import torch
from parameterized import param, parameterized
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType

from gigl.distributed.utils.loader import (
    remove_labeled_edge_types,
    set_labeled_edge_type_fanout,
    shard_nodes_by_process,
)
from tests.test_assets.distributed.utils import assert_tensor_equality

_SHARDING_TEST_TENSOR = torch.Tensor([1, 3, 5, 7, 9, 11, 13, 15, 17])
_U2I_EDGE_TYPE = ("user", "to", "item")
_I2U_EDGE_TYPE = ("item", "to", "user")
_LABELED_EDGE_TYPE = ("user", "to_gigl_positive", "item")


class LoaderUtilsTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "Test shard_nodes_by_process on 0 rank",
                input_tensor=_SHARDING_TEST_TENSOR,
                local_process_rank=0,
                local_process_world_size=2,
                expected_sharded_tensor=torch.Tensor([1, 3, 5, 7]),
            ),
            param(
                "Test shard_nodes_by_process on 1 rank",
                input_tensor=_SHARDING_TEST_TENSOR,
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
        data = HeteroData()
        data[_U2I_EDGE_TYPE].edge_index = torch.Tensor([[0, 0], [1, 1]])
        data[_I2U_EDGE_TYPE].edge_index = torch.Tensor([[2, 2], [3, 3]])
        data[_LABELED_EDGE_TYPE].edge_index = torch.Tensor([[4, 4], [5, 5]])
        data.num_sampled_edges = {
            _U2I_EDGE_TYPE: [2, 2],
            _I2U_EDGE_TYPE: [3, 1],
            _LABELED_EDGE_TYPE: [0, 0],
        }
        data = remove_labeled_edge_types(data)
        for edge_type in (_U2I_EDGE_TYPE, _I2U_EDGE_TYPE):
            self.assertIn(edge_type, data.num_sampled_edges)
            self.assertIn(edge_type, data._edge_store_dict)
            self.assertIn(edge_type, data.edge_types)
        self.assertNotIn(_LABELED_EDGE_TYPE, data.num_sampled_edges)
        self.assertNotIn(_LABELED_EDGE_TYPE, data._edge_store_dict)
        self.assertNotIn(_LABELED_EDGE_TYPE, data.edge_types)

    @parameterized.expand(
        [
            param(
                "Test set_labeled_edge_type on num_neighbors dict with labeled edge type",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE, _LABELED_EDGE_TYPE],
                num_neighbors={_U2I_EDGE_TYPE: [2, 7], _I2U_EDGE_TYPE: [3, 4]},
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                    _LABELED_EDGE_TYPE: [0, 0],
                },
            ),
            param(
                "Test set_labeled_edge_type on num_neighbors dict with no labeled edge type",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE],
                num_neighbors={_U2I_EDGE_TYPE: [2, 7], _I2U_EDGE_TYPE: [3, 4]},
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                },
            ),
        ]
    )
    def test_set_labeled_edge_type_fanout(
        self,
        _,
        edge_types: list[EdgeType],
        num_neighbors: dict[EdgeType, list[int]],
        expected_num_neighbors: dict[EdgeType, list[int]],
    ):
        num_neighbors = set_labeled_edge_type_fanout(
            edge_types=edge_types, num_neighbors=num_neighbors
        )
        self.assertEqual(num_neighbors, expected_num_neighbors)
