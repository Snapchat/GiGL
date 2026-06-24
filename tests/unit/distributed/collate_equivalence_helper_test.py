import os
from collections.abc import Iterator

import torch
from absl.testing import absltest
from torch_geometric.data import Data, HeteroData

from gigl.distributed.utils.neighborloader import COLLATE_IMPL_ENV_VAR
from tests.test_assets.distributed.collate_equivalence import (
    assert_collated_equal,
    assert_impls_equivalent,
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

    def _make_heterogeneous(self) -> HeteroData:
        data = HeteroData()
        data["a"].node = torch.tensor([10])
        data["a"].batch = torch.tensor([10])
        data["a"].batch_size = 1
        data["b"].node = torch.tensor([11, 12, 13, 14])
        edge_type = ("a", "to", "b")
        data[edge_type].edge_index = torch.tensor([[0, 0], [0, 1]])
        data.num_sampled_nodes = {"a": torch.tensor([1]), "b": torch.tensor([0, 4])}
        data.num_sampled_edges = {edge_type: torch.tensor([2])}
        data.y_positive = {edge_type: {0: torch.tensor([2, 3])}}
        return data

    def test_collated_equal_passes_for_identical_heterogeneous(self) -> None:
        assert_collated_equal(self._make_heterogeneous(), self._make_heterogeneous())

    def test_collated_equal_raises_on_hetero_node_type_mismatch(self) -> None:
        a = self._make_heterogeneous()
        b = self._make_heterogeneous()
        b["b"].node = torch.tensor([11, 12, 13, 99])
        with self.assertRaises(Exception):
            assert_collated_equal(a, b)

    def test_collated_equal_raises_on_hetero_label_edge_type_mismatch(self) -> None:
        a = self._make_heterogeneous()
        b = self._make_heterogeneous()
        b.y_positive = {("a", "to", "b"): {0: torch.tensor([2])}}  # dropped index 3
        with self.assertRaises(Exception):
            assert_collated_equal(a, b)

    def test_assert_impls_equivalent_passes_for_homogeneous(self) -> None:
        # Fake loader that always returns the same homogeneous batch regardless of impl.
        expected = self._make_homogeneous()

        def _homo_loader_factory() -> Iterator[Data]:
            yield self._make_homogeneous()

        assert_impls_equivalent(_homo_loader_factory, impls=("python", "vectorized"))
        # Verify the expected batch is unchanged (sanity check on factory isolation).
        assert_collated_equal(self._make_homogeneous(), expected)

    def test_assert_impls_equivalent_passes_for_heterogeneous(self) -> None:
        # Fake loader that always returns the same heterogeneous batch regardless of impl.
        def _hetero_loader_factory() -> Iterator[HeteroData]:
            yield self._make_heterogeneous()

        assert_impls_equivalent(_hetero_loader_factory, impls=("python", "vectorized"))

    def test_assert_impls_equivalent_raises_on_mismatch(self) -> None:
        # Fake loader that returns different node tensors depending on the active impl.
        # When GIGL_COLLATE_IMPL is "vectorized" it mutates one node value to expose a
        # disagreement that assert_impls_equivalent must catch.
        def _mismatched_loader_factory() -> Iterator[Data]:
            data = self._make_homogeneous()
            if os.environ.get(COLLATE_IMPL_ENV_VAR) == "vectorized":
                data.node = torch.tensor(
                    [10, 11, 99]
                )  # differs from python's [10,11,12]
            yield data

        with self.assertRaises(Exception):
            assert_impls_equivalent(
                _mismatched_loader_factory, impls=("python", "vectorized")
            )


if __name__ == "__main__":
    absltest.main()
