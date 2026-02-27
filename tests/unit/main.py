import sys
from typing import Final

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.common.utils.test_utils import parse_args, run_tests

# Slow test modules that must be spread across shards. Position in the tuple
# determines the shard: ``index % total_shards``.  **Append-only** — never
# reorder existing entries, or every module's shard assignment will shift.
#
# Durations measured 2026-02-27 (unsharded CI run, 61.7 min total):
UNIT_TEST_SHARD_PINNED_MODULES: Final[tuple[str, ...]] = (
    "tests.unit.distributed.dist_ablp_neighborloader_test",  # 24.7 min (40.0%)
    "tests.unit.distributed.distributed_dataset_test",  # 10.7 min (17.4%)
    "tests.unit.distributed.distributed_neighborloader_test",  # 9.6 min (15.5%)
    "tests.unit.distributed.distributed_partitioner_test",  # 6.5 min (10.5%)
    "tests.unit.distributed.utils.networking_test",  # 2.7 min  (4.4%)
)


def run(
    pattern: str = "*_test.py",
    shard_index: int = 0,
    total_shards: int = 0,
) -> bool:
    return run_tests(
        start_dir=LocalUri.join(
            local_fs_constants.get_project_root_directory(), "tests", "unit"
        ),
        pattern=pattern,
        use_sequential_execution=True,
        shard_index=shard_index,
        total_shards=total_shards,
        pinned_modules=UNIT_TEST_SHARD_PINNED_MODULES,
    )


if __name__ == "__main__":
    test_args = parse_args()
    was_successful: bool = run(
        pattern=test_args.test_file_pattern,
        shard_index=test_args.shard_index,
        total_shards=test_args.total_shards,
    )
    sys.exit(not was_successful)
