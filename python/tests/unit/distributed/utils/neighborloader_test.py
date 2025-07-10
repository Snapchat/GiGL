import unittest
from typing import Optional, Union

import torch
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import EdgeType

from gigl.distributed.utils.neighborloader import (
    labeled_to_homogeneous,
    patch_fanout_for_sampling,
    shard_nodes_by_process,
    strip_label_edges,
)
from gigl.types.graph import message_passing_to_positive_label
from tests.test_assets.distributed.utils import assert_tensor_equality

_U2I_EDGE_TYPE = ("user", "to", "item")
_I2U_EDGE_TYPE = ("item", "to", "user")
_LABELED_EDGE_TYPE = message_passing_to_positive_label(_U2I_EDGE_TYPE)


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
                "Test patch_fanout_for_sampling on num_neighbors dict with labeled edge type",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE, _LABELED_EDGE_TYPE],
                num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                },
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7],
                    _I2U_EDGE_TYPE: [3, 4],
                    _LABELED_EDGE_TYPE: [0, 0],
                },
            ),
            param(
                "Test patch_fanout_for_sampling on num_neighbors list",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE, _LABELED_EDGE_TYPE],
                num_neighbors=[1, 3],
                expected_num_neighbors={
                    _U2I_EDGE_TYPE: [1, 3],
                    _I2U_EDGE_TYPE: [1, 3],
                    _LABELED_EDGE_TYPE: [0, 0],
                },
            ),
            param(
                "Test patch_fanout_for_sampling on homogeneous dataset",
                edge_types=None,
                num_neighbors=[1, 3],
                expected_num_neighbors=[1, 3],
            ),
        ]
    )
    def test_patch_neighbors_with_zero_fanout(
        self,
        _,
        edge_types: Optional[list[EdgeType]],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
        expected_num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
    ):
        num_neighbors = patch_fanout_for_sampling(edge_types, num_neighbors)
        self.assertEqual(num_neighbors, expected_num_neighbors)

    @parameterized.expand(
        [
            param(
                "Test failure when homogeneous dataset has heterogeneous num_neighbors",
                edge_types=None,
                num_neighbors={_U2I_EDGE_TYPE: [2, 7]},
            ),
            param(
                "Test failure when heterogeneous dataset and num_neighbors has different number of hops per edge type",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE],
                num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7, 10],
                    _I2U_EDGE_TYPE: [3, 4],
                },
            ),
            param(
                "Test failure for heterogeneous dataset and num_neighbors when there is a missing edge type in num_neighbors from the dataset",
                edge_types=[_U2I_EDGE_TYPE, _I2U_EDGE_TYPE],
                num_neighbors={_U2I_EDGE_TYPE: [2, 7]},
            ),
            param(
                "Test failure for heterogeneous dataset and num_neighbors when there is an extra edge type in num_neighbors from the dataset",
                edge_types=[_I2U_EDGE_TYPE],
                num_neighbors={
                    _U2I_EDGE_TYPE: [2, 7, 10],
                    _I2U_EDGE_TYPE: [3, 4],
                },
            ),
        ]
    )
    def test_patch_neighbors_failure(
        self,
        _,
        edge_types: Optional[list[EdgeType]],
        num_neighbors: Union[list[int], dict[EdgeType, list[int]]],
    ):
        with self.assertRaises(ValueError):
            num_neighbors = patch_fanout_for_sampling(edge_types, num_neighbors)

    def test_pyg_to_homogeneous(self):
        data = HeteroData()
        data[_U2I_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 0]])
        data[_I2U_EDGE_TYPE].edge_index = torch.tensor([[1, 0], [0, 1]])
        data["user"].x = torch.tensor([[1.0], [2.0]])
        data["item"].x = torch.tensor([[3.0], [4.0]])
        data.num_sampled_edges = {
            _U2I_EDGE_TYPE: torch.tensor([1, 1]),
            _I2U_EDGE_TYPE: torch.tensor([2, 2]),
        }
        data.num_sampled_nodes = {
            "user": torch.tensor([1, 1]),
            "item": torch.tensor([2, 2]),
        }

        data.batch = torch.tensor([0, 1])

        homogeneous_data = labeled_to_homogeneous(_U2I_EDGE_TYPE, data)
        self.assertIsInstance(homogeneous_data, Data)
        self.assertTrue(hasattr(homogeneous_data, "edge_index"))
        self.assertTrue(hasattr(homogeneous_data, "batch"))
        self.assertTrue(hasattr(homogeneous_data, "batch_size"))
        assert_tensor_equality(homogeneous_data.num_sampled_nodes, torch.tensor([1, 1]))
        assert_tensor_equality(homogeneous_data.num_sampled_edges, torch.tensor([1, 1]))
        assert_tensor_equality(homogeneous_data.batch, torch.tensor([0, 1]))
        self.assertEqual(homogeneous_data.batch_size, 2)

    def test_strip_label_edges(self):
        data = HeteroData()
        data[_U2I_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 0]])
        data[_I2U_EDGE_TYPE].edge_index = torch.tensor([[1, 0], [0, 1]])
        data[_LABELED_EDGE_TYPE].edge_index = torch.tensor([[0, 1], [1, 0]])
        data["user"].x = torch.tensor([[1.0], [2.0]])
        data["item"].x = torch.tensor([[3.0], [4.0]])
        data.num_sampled_edges = {
            _U2I_EDGE_TYPE: torch.tensor([1, 1]),
            _I2U_EDGE_TYPE: torch.tensor([2, 2]),
            _LABELED_EDGE_TYPE: torch.tensor([1, 1]),
        }

        stripped_data = strip_label_edges(data)
        self.assertIsInstance(stripped_data, HeteroData)
        self.assertFalse(_LABELED_EDGE_TYPE in stripped_data.edge_types)
        self.assertTrue(_U2I_EDGE_TYPE in stripped_data.edge_types)
        self.assertTrue(_I2U_EDGE_TYPE in stripped_data.edge_types)

        self.assertFalse(_LABELED_EDGE_TYPE in stripped_data.num_sampled_edges)
        self.assertTrue(_U2I_EDGE_TYPE in stripped_data.num_sampled_edges)
        self.assertTrue(_I2U_EDGE_TYPE in stripped_data.num_sampled_edges)
