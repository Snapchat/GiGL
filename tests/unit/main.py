import sys

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.common.utils.test_utils import parse_args, run_tests


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
    )


if __name__ == "__main__":
    test_args = parse_args()
    was_successful: bool = run(
        pattern=test_args.test_file_pattern,
        shard_index=test_args.shard_index,
        total_shards=test_args.total_shards,
    )
    sys.exit(not was_successful)
