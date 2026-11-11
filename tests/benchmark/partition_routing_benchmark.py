"""Compares per-partition mask routing with grouped routing.

Run: ``PYTHONPATH=. python tests/benchmark/partition_routing_benchmark.py [--num_ids 210000] [--num_partitions 60]``
"""

import argparse
import time

import torch

from gigl.distributed.utils.partition_routing import group_by_partition


def _mask_routing(
    ids: torch.Tensor, partition_ids: torch.Tensor, num_partitions: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    order = torch.arange(ids.size(0), dtype=torch.long)
    routed = []
    for pidx in range(num_partitions):
        mask = partition_ids == pidx
        routed.append(
            (torch.masked_select(ids, mask), torch.masked_select(order, mask))
        )
    return routed


def _grouped_routing(
    ids: torch.Tensor, partition_ids: torch.Tensor, num_partitions: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ids_by_partition, index_by_partition = group_by_partition(
        ids, partition_ids, num_partitions
    )
    return list(zip(ids_by_partition, index_by_partition))


def _milliseconds_per_call(function, ids, partition_ids, num_partitions, reps) -> float:
    function(ids, partition_ids, num_partitions)
    start = time.perf_counter()
    for _ in range(reps):
        function(ids, partition_ids, num_partitions)
    return (time.perf_counter() - start) / reps * 1e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_ids", type=int, default=210_000)
    parser.add_argument("--num_partitions", type=int, default=60)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(0)
    ids = torch.randint(0, 960_000_000, (args.num_ids,))
    partition_ids = torch.randint(0, args.num_partitions, (args.num_ids,))
    for (old_ids, old_index), (new_ids, new_index) in zip(
        _mask_routing(ids, partition_ids, args.num_partitions),
        _grouped_routing(ids, partition_ids, args.num_partitions),
    ):
        assert torch.equal(old_ids, new_ids) and torch.equal(old_index, new_index)

    for threads in (torch.get_num_threads(), 1):
        torch.set_num_threads(threads)
        old_ms = _milliseconds_per_call(
            _mask_routing, ids, partition_ids, args.num_partitions, args.reps
        )
        new_ms = _milliseconds_per_call(
            _grouped_routing, ids, partition_ids, args.num_partitions, args.reps
        )
        print(
            f"ids={args.num_ids} partitions={args.num_partitions} torch_threads={threads}: "
            f"mask loop {old_ms:.2f} ms, grouped {new_ms:.2f} ms, speedup x{old_ms / new_ms:.1f}"
        )


if __name__ == "__main__":
    main()
