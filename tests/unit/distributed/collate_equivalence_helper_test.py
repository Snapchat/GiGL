import torch
from absl.testing import absltest
from torch_geometric.data import Data, HeteroData

from tests.test_assets.distributed.collate_equivalence import (
    assert_collated_equal,
    assert_label_dict_equal,
)
from tests.test_assets.test_case import TestCase


class CollateEquivalenceHelperTest(TestCase):
    def test_label_dict_equal_passes_for_identical_incl_empty(self) -> None:
        a = {0: torch.tensor([1, 2]), 1: torch.tensor([], dtype=torch.long)}
        b = {0: torch.tensor([1, 2]), 1: torch.tensor([], dtype=torch.long)}
        # Should not raise: identical keys, identical values, empty tensor present.
        assert_label_dict_equal(a, b)

    def test_label_dict_equal_raises_on_missing_empty_key(self) -> None:
        # `b` dropped the empty-tensor anchor key 1 — the exact bug the helper guards.
        a = {0: torch.tensor([1, 2]), 1: torch.tensor([], dtype=torch.long)}
        b = {0: torch.tensor([1, 2])}
        with self.assertRaises(AssertionError):
            assert_label_dict_equal(a, b)

    def test_label_dict_equal_raises_on_value_mismatch(self) -> None:
        a = {0: torch.tensor([1, 2])}
        b = {0: torch.tensor([1, 3])}
        with self.assertRaises(Exception):
            assert_label_dict_equal(a, b)

    def test_label_dict_equal_raises_on_dropped_duplicate(self) -> None:
        # The oracle's torch.nonzero output preserves duplicate multiplicity when
        # a padded label row repeats a global id; an impl that de-duplicates is a
        # real bug. Exact comparison must catch the missing repeat.
        a = {0: torch.tensor([1, 1, 2])}
        b = {0: torch.tensor([1, 2])}
        with self.assertRaises(Exception):
            assert_label_dict_equal(a, b)

    def test_label_dict_equal_raises_on_reordered_values(self) -> None:
        # Values are contractually ascending (torch.nonzero order); a permuted
        # tensor with the same multiset must still FAIL under exact comparison.
        a = {0: torch.tensor([1, 2, 3])}
        b = {0: torch.tensor([3, 2, 1])}
        with self.assertRaises(Exception):
            assert_label_dict_equal(a, b)

    def _make_homogeneous(self) -> Data:
        data = Data()
        data.node = torch.tensor([10, 11, 12])
        data.x = torch.tensor([[1.0], [2.0], [3.0]])
        data.edge_index = torch.tensor([[0, 1], [1, 2]])
        data.edge_attr = torch.tensor([[0.5], [0.7]])
        data.num_sampled_nodes = torch.tensor([1, 2])
        data.num_sampled_edges = torch.tensor([2])
        data.batch = torch.tensor([10])
        data.batch_size = 1
        data.y_positive = {0: torch.tensor([1]), 1: torch.tensor([], dtype=torch.long)}
        return data

    def test_collated_equal_passes_for_identical_homogeneous(self) -> None:
        assert_collated_equal(self._make_homogeneous(), self._make_homogeneous())

    def test_collated_equal_raises_on_node_mismatch(self) -> None:
        a = self._make_homogeneous()
        b = self._make_homogeneous()
        b.node = torch.tensor([10, 11, 99])
        with self.assertRaises(Exception):
            assert_collated_equal(a, b)

    def test_collated_equal_raises_on_type_mismatch(self) -> None:
        a = self._make_homogeneous()
        b = HeteroData()
        with self.assertRaises(AssertionError):
            assert_collated_equal(a, b)


if __name__ == "__main__":
    absltest.main()
