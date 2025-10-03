import sys

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.common.utils.test_utils import parse_args, run_tests


def run(pattern: str = "*_test.py") -> bool:
    return run_tests(
        start_dir=LocalUri.join(
            local_fs_constants.get_python_project_root_path(), "tests", "unit"
        ),
        pattern=pattern,
        use_sequential_execution=True,
    )


if __name__ == "__main__":
    was_successful: bool = run(pattern=parse_args().test_file_pattern)
    sys.exit(not was_successful)
