import torch
from torch_geometric.data import Data, HeteroData

from gigl.distributed import _collate_dispatch as cd
from tests.test_assets.test_case import TestCase


class TestAssembleHomogeneous(TestCase):
    def test_builds_data_with_edge_index_and_node(self) -> None:
        components = {
            "node": torch.tensor([10, 11, 12]),
            "edge_index": torch.tensor([[0, 1], [1, 2]]),
            "edge": None,
            "x": None,
            "edge_attr": None,
            "batch": torch.tensor([10, 11]),
            "num_sampled_nodes": torch.tensor([2, 1]),
            "num_sampled_edges": torch.tensor([2]),
        }
        data = cd.assemble_homogeneous(components)
        self.assertIsInstance(data, Data)
        torch.testing.assert_close(data.node, components["node"])
        torch.testing.assert_close(data.edge_index, components["edge_index"])
        self.assertEqual(data.batch_size, 2)
        torch.testing.assert_close(
            data.num_sampled_nodes, components["num_sampled_nodes"]
        )


class _FakeHeteroResult:
    def __init__(self) -> None:
        self.node = {"u": torch.tensor([1, 2]), "i": torch.tensor([3])}
        self.edge_index = {("u", "to", "i"): torch.tensor([[0, 1], [0, 0]])}
        self.edge = {}
        self.x = {}
        self.edge_attr = {}
        self.batch = {"u": torch.tensor([1, 2])}
        self.num_sampled_nodes = {"u": torch.tensor([2, 0]), "i": torch.tensor([1, 0])}
        self.num_sampled_edges = {("u", "to", "i"): torch.tensor([2])}


class TestAssembleHeterogeneous(TestCase):
    def test_builds_heterodata(self) -> None:
        data = cd.assemble_heterogeneous(_FakeHeteroResult())
        self.assertIsInstance(data, HeteroData)
        torch.testing.assert_close(data["u"].node, torch.tensor([1, 2]))
        torch.testing.assert_close(
            data["u", "to", "i"].edge_index, torch.tensor([[0, 1], [0, 0]])
        )
        self.assertEqual(data["u"].batch_size, 2)
        torch.testing.assert_close(data.num_sampled_nodes["u"], torch.tensor([2, 0]))
