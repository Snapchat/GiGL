import argparse
import hashlib
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterator, Tuple
from unittest import TestCase

from gigl.common import LocalUri
from gigl.common.logger import Logger

logger = Logger()


@dataclass(frozen=True)
class TestArgs:
    """Container for CLI arguments to Python tests.

    Attributes:
        test_file_pattern: Glob pattern for filtering which test files to run.
            See doc comment in `parse_args` for more details.
        shard_index: Zero-based index of the current shard.
        total_shards: Total number of shards. 0 means no sharding.
    """

    test_file_pattern: str
    shard_index: int = 0
    total_shards: int = 0


def parse_args() -> TestArgs:
    """Parses test-exclusive CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-tf",
        "--test_file_pattern",
        default="*_test.py",
        help="""
        Glob pattern for filtering which test files to run. By default runs *all* files ("*_test.py").
        Only *one* regex is supported at a time.
        Only the file *name* is checked, if a file *path* is provided then nothing will be matched.
        (Unless your file name has "/" in it, which is very unlikely.)
        Examples:
        ```
            -tf="frozen_dict_test.py"
            -tf="pyg*_test.py"
        ```
        """,
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Zero-based index of the current shard (used with --total_shards).",
    )
    parser.add_argument(
        "--total_shards",
        type=int,
        default=0,
        help="Total number of shards. 0 or 1 means no sharding (run all tests).",
    )
    args, _ = parser.parse_known_args()
    test_args = TestArgs(
        test_file_pattern=args.test_file_pattern,
        shard_index=args.shard_index,
        total_shards=args.total_shards,
    )
    logger.info(f"Test args: {test_args}")
    return test_args


def _filter_tests_by_shard(
    suite: unittest.TestSuite, shard_index: int, total_shards: int
) -> unittest.TestSuite:
    """Filters a test suite to only include tests belonging to the given shard.

    Sharding is done at the file (module) level so that setUpClass/tearDownClass
    are not split across shards. Each top-level test group's module name is
    SHA-256 hashed and assigned to a shard via ``hash % total_shards``.

    Args:
        suite: The full test suite discovered by unittest.
        shard_index: Zero-based index of the current shard.
        total_shards: Total number of shards. If <= 1, the suite is returned
            unchanged.

    Returns:
        A new TestSuite containing only the tests assigned to this shard.
    """
    if total_shards <= 1:
        return suite

    filtered = unittest.TestSuite()
    for test_group in suite:
        module_name = _get_test_group_module_name(test_group)
        hash_value = int(hashlib.sha256(module_name.encode()).hexdigest(), 16)
        if hash_value % total_shards == shard_index:
            filtered.addTest(test_group)
    return filtered


def _get_test_group_module_name(test_group: unittest.TestSuite | TestCase) -> str:
    """Extracts the module name from a test group for shard assignment.

    Args:
        test_group: A test suite or individual test case.

    Returns:
        The module name string used for hashing.
    """
    if isinstance(test_group, unittest.TestSuite):
        # Recurse into nested suites to find the first actual test case
        for item in test_group:
            return _get_test_group_module_name(item)
    # TestCase instance — use its module
    return type(test_group).__module__


def _run_individual_test(test: TestCase) -> Tuple[bool, int]:
    # If we don't have any test cases, we skip running the test.
    # This reduces some noise in the logs.
    if test.countTestCases() == 0:
        logger.warning(
            f"Test {test} has no test cases to run. Skipping execution of this test."
        )
        return (True, 0)
    runner = unittest.TextTestRunner(verbosity=2)
    result: unittest.TestResult = runner.run(test=test)

    return (result.wasSuccessful(), test.countTestCases())


def run_tests(
    start_dir: LocalUri,
    pattern: str,
    use_sequential_execution: bool = False,
    shard_index: int = 0,
    total_shards: int = 0,
) -> bool:
    """Discovers and runs tests, optionally filtering by shard.

    Args:
        start_dir: Local directory for running tests.
        pattern: File text pattern for running tests.
        use_sequential_execution: Whether sequential execution should be used.
        shard_index: Zero-based index of the current shard.
        total_shards: Total number of shards. 0 or 1 means no sharding.

    Returns:
        Whether all tests passed successfully.
    """
    start = time.perf_counter()

    loader = unittest.TestLoader()
    # Find all tests in "tests/unit" signified by name of the file ending in the provided pattern
    suite: unittest.TestSuite = loader.discover(
        start_dir=start_dir.uri,
        pattern=pattern,
    )

    total_discovered: int = suite.countTestCases()
    suite = _filter_tests_by_shard(suite, shard_index, total_shards)

    if total_shards > 1:
        logger.info(
            f"Shard {shard_index}/{total_shards}: running {suite.countTestCases()}/{total_discovered} test cases"
        )

    was_successful: bool
    total_num_test_cases: int = 0

    if use_sequential_execution:
        runner = unittest.TextTestRunner(verbosity=2)
        was_successful = runner.run(suite).wasSuccessful()
        total_num_test_cases = suite.countTestCases()
    else:
        with ProcessPoolExecutor() as executor:
            was_successful_iter: Iterator[Tuple[bool, int]] = executor.map(
                _run_individual_test, suite._tests
            )
        was_successful = True
        for was_successful_batch, num_test_cases_ran in was_successful_iter:
            was_successful = was_successful and was_successful_batch
            total_num_test_cases += num_test_cases_ran

    logger.info(f"Ran {total_num_test_cases}/{suite.countTestCases()} test cases")
    finish = time.perf_counter()
    logger.info(f"It took {finish-start: .2f} second(s) to run tests")
    return was_successful
