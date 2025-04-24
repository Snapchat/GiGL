import argparse
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Iterator, Tuple

from gigl.common import LocalUri
from gigl.common.logger import Logger

logger = Logger()


@dataclass(frozen=True)
class TestArgs:
    """Container for CLI arguements to Python tests.

    Attributes:
        test_file_pattern (str): Glob pattern for filtering which test files to run.
            See doc comment in `parse_args` for more details.
    """

    test_file_pattern: str


def parse_args() -> TestArgs:
    """Parses test-exclusive CLI arguements."""
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
    args, _ = parser.parse_known_args()
    test_args = TestArgs(test_file_pattern=args.test_file_pattern)
    logger.info(f"Test args: {test_args}")
    return test_args


def _run_individual_test(test: unittest.TestCase) -> Tuple[bool, int]:
    runner = unittest.TextTestRunner(verbosity=2)
    result: unittest.TestResult = runner.run(test=test)

    return (result.wasSuccessful(), test.countTestCases())


def run_tests(
    start_dir: LocalUri, pattern: str, use_sequential_execution: bool = False
) -> bool:
    """
    Args:
        start_dir (LocalUri): Local Directory for running tests
        pattern (str): file text pattern for running tests
        use_sequential_execution (bool): Whether sequential exection should be used
    Return:
        bool: Whether all tests passed successfully
    """
    start = time.perf_counter()

    loader = unittest.TestLoader()
    # Find all tests in "tests/unit" signified by name of the file ending in the provided pattern
    suite: unittest.TestSuite = loader.discover(
        start_dir=start_dir.uri,
        pattern=pattern,
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
