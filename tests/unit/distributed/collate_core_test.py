import torch
from tests.test_assets.test_case import TestCase


class TestCollateCoreBindings(TestCase):
    def test_collate_homogeneous_stacks_edge_index(self) -> None:
        from gigl_core import collate_core

        ids = torch.tensor([10, 11, 12])
        rows = torch.tensor([0, 1])
        cols = torch.tensor([1, 2])
        out = collate_core.collate_homogeneous(
            ids=ids, rows=rows, cols=cols, eids=None, nfeats=None,
            efeats=None, batch=None, num_sampled_nodes=None, num_sampled_edges=None,
        )
        torch.testing.assert_close(out["node"], ids)
        torch.testing.assert_close(out["edge_index"], torch.stack([rows, cols]))
        self.assertIsNone(out["x"])
        self.assertIsNone(out["num_sampled_nodes"])

    def test_collate_heterogeneous_returns_struct(self) -> None:
        from gigl_core import collate_core

        msg = {
            "u.ids": torch.tensor([100, 101]),
            "i.ids": torch.tensor([200, 201, 202]),
            "u.num_sampled_nodes": torch.tensor([2]),
            "i.num_sampled_nodes": torch.tensor([3]),
            "u__to__i.rows": torch.tensor([0, 1]),
            "u__to__i.cols": torch.tensor([0, 2]),
            "u__to__i.num_sampled_edges": torch.tensor([2]),
        }
        res = collate_core.collate_heterogeneous(
            msg=msg,
            node_types=["u", "i"],
            edge_type_str_to_rev={"u__to__i": ("i", "rev_to", "u")},
            reversed_edge_types=[("i", "rev_to", "u")],
            input_type="u",
            has_batch=False,
            batch_size=0,
        )
        sampled = ("i", "rev_to", "u")
        torch.testing.assert_close(
            res.edge_index[sampled], torch.stack([msg["u__to__i.cols"], msg["u__to__i.rows"]])
        )
        torch.testing.assert_close(res.num_sampled_nodes["u"], torch.tensor([2, 0]))
        torch.testing.assert_close(res.node["i"], msg["i.ids"])
