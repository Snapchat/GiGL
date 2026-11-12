import time
from types import SimpleNamespace

import torch
from absl.testing import absltest
from graphlearn_torch.utils import default_id_select

from gigl.distributed.base_sampler import BaseDistNeighborSampler
from gigl.distributed.utils.fused_feature import FusedDistFeature
from gigl.distributed.utils.partition_routing import group_by_partition
from tests.test_assets.test_case import TestCase


def _masked_routing(
    ids: torch.Tensor, partition_ids: torch.Tensor, num_partitions: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """The per-partition mask routing that GLT does, for every partition."""
    order = torch.arange(ids.size(0), dtype=torch.long)
    ids_by_partition = []
    index_by_partition = []
    for pidx in range(num_partitions):
        mask = partition_ids == pidx
        ids_by_partition.append(torch.masked_select(ids, mask))
        index_by_partition.append(torch.masked_select(order, mask))
    return ids_by_partition, index_by_partition


class GroupByPartitionTest(TestCase):
    def test_matches_the_mask_routing_exactly(self) -> None:
        for seed in range(200):
            generator = torch.Generator().manual_seed(seed)
            num_partitions = int(torch.randint(1, 65, (1,), generator=generator))
            num_ids = int(torch.randint(0, 400, (1,), generator=generator))
            # A small id range makes duplicate ids likely.
            ids = torch.randint(0, 50, (num_ids,), generator=generator)
            partition_book = torch.randint(
                0, num_partitions, (50,), generator=generator
            )
            if seed % 3 == 0:
                # Leaves the upper partitions empty.
                partition_book = partition_book % max(1, num_partitions // 2)
            partition_ids = partition_book[ids]

            expected_ids, expected_index = _masked_routing(
                ids, partition_ids, num_partitions
            )
            actual_ids, actual_index = group_by_partition(
                ids, partition_ids, num_partitions
            )

            self.assertEqual(len(actual_ids), num_partitions)
            self.assertEqual(len(actual_index), num_partitions)
            for pidx in range(num_partitions):
                self.assert_tensor_equality(actual_ids[pidx], expected_ids[pidx])
                self.assert_tensor_equality(actual_index[pidx], expected_index[pidx])
                self.assertEqual(actual_ids[pidx].dtype, expected_ids[pidx].dtype)
                self.assertEqual(actual_index[pidx].dtype, expected_index[pidx].dtype)
                self.assertEqual(
                    actual_ids[pidx].untyped_storage().nbytes(),
                    expected_ids[pidx].untyped_storage().nbytes(),
                )

    def test_positions_map_back_to_the_requested_ids(self) -> None:
        ids = torch.tensor([9, 4, 9, 7, 1])
        partition_ids = torch.tensor([2, 0, 2, 1, 0])

        ids_by_partition, index_by_partition = group_by_partition(ids, partition_ids, 4)

        self.assert_tensor_equality(ids_by_partition[0], torch.tensor([4, 1]))
        self.assert_tensor_equality(ids_by_partition[1], torch.tensor([7]))
        self.assert_tensor_equality(ids_by_partition[2], torch.tensor([9, 9]))
        self.assertEqual(ids_by_partition[3].numel(), 0)
        self.assert_tensor_equality(index_by_partition[0], torch.tensor([1, 4]))
        self.assert_tensor_equality(index_by_partition[2], torch.tensor([0, 2]))

    def test_other_partition_id_dtypes_route_the_same(self) -> None:
        ids = torch.tensor([5, 6, 7, 8, 9, 10])
        for dtype in (
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        ):
            partition_ids = torch.tensor([1, 0, 1, 2, 0, 2], dtype=dtype)
            expected_ids, expected_index = _masked_routing(ids, partition_ids, 3)
            actual_ids, actual_index = group_by_partition(ids, partition_ids, 3)
            for pidx in range(3):
                self.assert_tensor_equality(actual_ids[pidx], expected_ids[pidx])
                self.assert_tensor_equality(actual_index[pidx], expected_index[pidx])

    def test_unsigned_partition_ids_above_the_signed_range_are_dropped(self) -> None:
        ids = torch.tensor([10, 11, 12, 13])
        partition_ids = torch.tensor([1, 2**64 - 1, 0, 2**63], dtype=torch.uint64)
        expected_ids, expected_index = _masked_routing(ids, partition_ids, 2)

        actual_ids, actual_index = group_by_partition(ids, partition_ids, 2)

        self.assert_tensor_equality(actual_ids[0], torch.tensor([12]))
        self.assert_tensor_equality(actual_ids[1], torch.tensor([10]))
        for pidx in range(2):
            self.assert_tensor_equality(actual_ids[pidx], expected_ids[pidx])
            self.assert_tensor_equality(actual_index[pidx], expected_index[pidx])

    def test_partitions_of_unequal_sizes_pass_the_shape_checks(self) -> None:
        ids = torch.tensor([9, 4, 9, 7, 1])
        partition_ids = torch.tensor([2, 0, 2, 1, 0], dtype=torch.uint8)

        ids_by_partition, index_by_partition = group_by_partition(ids, partition_ids, 4)

        self.assertEqual([c.numel() for c in ids_by_partition], [2, 1, 2, 0])
        self.assertEqual([c.numel() for c in index_by_partition], [2, 1, 2, 0])

    def test_out_of_range_ids_are_dropped_without_work_proportional_to_their_size(
        self,
    ) -> None:
        ids = torch.tensor([10, 11, 12, 13])
        partition_ids = torch.tensor([0, 10**9, 1, 1])

        start = time.monotonic()
        ids_by_partition, index_by_partition = group_by_partition(ids, partition_ids, 2)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 5)
        self.assertEqual(len(ids_by_partition), 2)
        self.assertEqual(len(index_by_partition), 2)
        self.assert_tensor_equality(ids_by_partition[0], torch.tensor([10]))
        self.assert_tensor_equality(ids_by_partition[1], torch.tensor([12, 13]))
        self.assert_tensor_equality(index_by_partition[1], torch.tensor([2, 3]))

    def test_negative_partition_ids_are_dropped_like_the_mask(self) -> None:
        ids = torch.tensor([10, 11, 12, 13])
        partition_ids = torch.tensor([0, -1, 1, -(10**9)])
        expected_ids, expected_index = _masked_routing(ids, partition_ids, 2)

        actual_ids, actual_index = group_by_partition(ids, partition_ids, 2)

        self.assert_tensor_equality(actual_ids[0], torch.tensor([10]))
        self.assert_tensor_equality(actual_ids[1], torch.tensor([12]))
        for pidx in range(2):
            self.assert_tensor_equality(actual_ids[pidx], expected_ids[pidx])
            self.assert_tensor_equality(actual_index[pidx], expected_index[pidx])

    def test_other_id_dtypes_are_kept(self) -> None:
        partition_ids = torch.tensor([1, 0, 1, 2, 0, 2])
        for dtype in (torch.int32, torch.int64):
            ids = torch.tensor([5, 6, 7, 8, 9, 10], dtype=dtype)
            expected_ids, expected_index = _masked_routing(ids, partition_ids, 3)
            actual_ids, actual_index = group_by_partition(ids, partition_ids, 3)
            for pidx in range(3):
                self.assertEqual(actual_ids[pidx].dtype, dtype)
                self.assert_tensor_equality(actual_ids[pidx], expected_ids[pidx])
                self.assert_tensor_equality(actual_index[pidx], expected_index[pidx])


def _random_routing_inputs(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    generator = torch.Generator().manual_seed(seed)
    num_partitions = int(torch.randint(1, 65, (1,), generator=generator))
    local_partition = int(torch.randint(0, num_partitions, (1,), generator=generator))
    ids = torch.randint(0, 50, (int(torch.randint(0, 300, (1,), generator=generator)),))
    partition_book = torch.randint(0, num_partitions, (50,), generator=generator)
    return ids, partition_book[ids], num_partitions, local_partition


class CallerRoutingTest(TestCase):
    def test_hop_routing_matches_the_mask_routing_in_partition_order(self) -> None:
        for seed in range(100):
            srcs, partition_ids, num_partitions, local_partition = (
                _random_routing_inputs(seed)
            )
            orders = torch.arange(srcs.size(0), dtype=torch.long)
            sampler = BaseDistNeighborSampler.__new__(BaseDistNeighborSampler)
            sampler.data = SimpleNamespace(
                num_partitions=num_partitions,
                partition_idx=local_partition,
                id_select=default_id_select,
            )
            expected_ids, expected_index = _masked_routing(
                srcs, partition_ids, num_partitions
            )

            routed = list(sampler._route_hop(srcs, orders, partition_ids, None))

            expected_partitions = [
                (local_partition + i) % num_partitions
                for i in range(num_partitions)
                if expected_ids[(local_partition + i) % num_partitions].numel() > 0
            ]
            self.assertEqual([r[0] for r in routed], expected_partitions)
            for pidx, ids, index in routed:
                self.assert_tensor_equality(ids, expected_ids[pidx])
                self.assert_tensor_equality(index, expected_index[pidx])

    def test_hop_routing_keeps_a_custom_id_select(self) -> None:
        srcs = torch.tensor([3, 4, 5, 6])
        partition_ids = torch.tensor([0, 1, 0, 1])
        orders = torch.arange(4)
        selections = []

        def counting_id_select(srcs, p_mask, node_pb=None):
            selections.append(p_mask)
            return torch.masked_select(srcs, p_mask)

        sampler = BaseDistNeighborSampler.__new__(BaseDistNeighborSampler)
        sampler.data = SimpleNamespace(
            num_partitions=2, partition_idx=0, id_select=counting_id_select
        )

        routed = list(sampler._route_hop(srcs, orders, partition_ids, None))

        self.assertEqual(len(selections), 2)
        self.assertEqual([r[0] for r in routed], [0, 1])
        self.assert_tensor_equality(routed[1][1], torch.tensor([4, 6]))

    def test_hop_and_remote_feature_routing_accept_partitions_of_unequal_sizes(
        self,
    ) -> None:
        srcs = torch.tensor([9, 4, 9, 7, 1])
        partition_ids = torch.tensor([2, 0, 2, 1, 0], dtype=torch.uint8)
        sampler = BaseDistNeighborSampler.__new__(BaseDistNeighborSampler)
        sampler.data = SimpleNamespace(
            num_partitions=4, partition_idx=0, id_select=default_id_select
        )
        feature = FusedDistFeature.__new__(FusedDistFeature)
        feature.num_partitions = 4
        feature.partition_idx = 0

        hops = list(sampler._route_hop(srcs, torch.arange(5), partition_ids, None))
        remote = list(feature._route_remote(srcs, partition_ids))

        self.assertEqual([h[0] for h in hops], [0, 1, 2])
        self.assertEqual([r[0] for r in remote], [1, 2])
        self.assertEqual([r[1].numel() for r in remote], [1, 2])

    def test_remote_feature_routing_matches_the_mask_routing_of_other_partitions(
        self,
    ) -> None:
        for seed in range(100):
            ids, partition_ids, num_partitions, local_partition = (
                _random_routing_inputs(seed)
            )
            feature = FusedDistFeature.__new__(FusedDistFeature)
            feature.num_partitions = num_partitions
            feature.partition_idx = local_partition
            expected_ids, expected_index = _masked_routing(
                ids, partition_ids, num_partitions
            )

            routed = list(feature._route_remote(ids, partition_ids))

            expected_partitions = [
                pidx
                for pidx in range(num_partitions)
                if pidx != local_partition and expected_ids[pidx].numel() > 0
            ]
            self.assertEqual([r[0] for r in routed], expected_partitions)
            for pidx, remote_ids, remote_index in routed:
                self.assert_tensor_equality(remote_ids, expected_ids[pidx])
                self.assert_tensor_equality(remote_index, expected_index[pidx])


if __name__ == "__main__":
    absltest.main()
