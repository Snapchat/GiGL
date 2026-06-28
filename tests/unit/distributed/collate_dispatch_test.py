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


class TestCollateCppHomogeneous(TestCase):
    def test_homogeneous_end_to_end(self) -> None:
        msg = {
            "ids": torch.tensor([10, 11, 12]),
            "rows": torch.tensor([0, 1]),
            "cols": torch.tensor([1, 2]),
            "num_sampled_nodes": torch.tensor([2, 1]),
            "num_sampled_edges": torch.tensor([2]),
            "batch": torch.tensor([10, 11]),
        }
        data = cd.collate_cpp_homogeneous(
            msg, batch_size=2, has_batch=True, to_device=torch.device("cpu")
        )
        # GLT homogeneous reverses: edge_index = stack([cols, rows]) (dist_loader.py:446).
        torch.testing.assert_close(
            data.edge_index, torch.stack([msg["cols"], msg["rows"]])
        )
        torch.testing.assert_close(data.node, msg["ids"])
        self.assertEqual(data.batch_size, 2)


class TestCollateCppHeterogeneous(TestCase):
    def test_heterogeneous_end_to_end(self) -> None:
        msg = {
            "u.ids": torch.tensor([100, 101]),
            "i.ids": torch.tensor([200, 201, 202]),
            "u.num_sampled_nodes": torch.tensor([2]),
            "i.num_sampled_nodes": torch.tensor([3]),
            "u__to__i.rows": torch.tensor([0, 1]),
            "u__to__i.cols": torch.tensor([0, 2]),
            "u__to__i.num_sampled_edges": torch.tensor([2]),
        }
        data = cd.collate_cpp_heterogeneous(
            msg=msg,
            node_types=["u", "i"],
            edge_type_str_to_rev={"u__to__i": ("i", "rev_to", "u")},
            reversed_edge_types=[("i", "rev_to", "u")],
            input_type="u",
            has_batch=False,
            batch_size=0,
            to_device=torch.device("cpu"),
        )
        torch.testing.assert_close(
            data["i", "rev_to", "u"].edge_index,
            torch.stack([msg["u__to__i.cols"], msg["u__to__i.rows"]]),
        )
        torch.testing.assert_close(data["u"].node, msg["u.ids"])
