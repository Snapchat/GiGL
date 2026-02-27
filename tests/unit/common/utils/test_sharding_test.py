import unittest

from gigl.common.utils.test_utils import _filter_tests_by_shard
from tests.test_assets.test_case import TestCase


def _make_test_suite_with_modules(module_names: list[str]) -> unittest.TestSuite:
    """Creates a test suite where each top-level group simulates a different module.

    Each module gets a dynamically created TestCase subclass with one test method,
    mirroring the structure produced by ``unittest.TestLoader.discover()``.

    Args:
        module_names: List of module name strings to simulate.

    Returns:
        A TestSuite containing one nested TestSuite per module name.
    """
    outer_suite = unittest.TestSuite()
    for module_name in module_names:
        # Dynamically create a TestCase class with a unique module
        test_class = type(
            f"TestFor_{module_name.replace('.', '_')}",
            (unittest.TestCase,),
            {
                "test_placeholder": lambda self: None,
                "__module__": module_name,
            },
        )
        inner_suite = unittest.TestSuite([test_class("test_placeholder")])
        outer_suite.addTest(inner_suite)
    return outer_suite


class FilterTestsByShardTest(TestCase):
    """Tests for the _filter_tests_by_shard function."""

    MODULES: list[str] = [
        "tests.unit.module_a_test",
        "tests.unit.module_b_test",
        "tests.unit.module_c_test",
        "tests.unit.module_d_test",
        "tests.unit.module_e_test",
        "tests.unit.module_f_test",
        "tests.unit.module_g_test",
        "tests.unit.module_h_test",
    ]

    def test_no_sharding_when_total_shards_is_zero(self) -> None:
        suite = _make_test_suite_with_modules(self.MODULES)
        result = _filter_tests_by_shard(suite, shard_index=0, total_shards=0)
        self.assertEqual(result.countTestCases(), suite.countTestCases())

    def test_no_sharding_when_total_shards_is_one(self) -> None:
        suite = _make_test_suite_with_modules(self.MODULES)
        result = _filter_tests_by_shard(suite, shard_index=0, total_shards=1)
        self.assertEqual(result.countTestCases(), suite.countTestCases())

    def test_all_tests_covered_across_shards(self) -> None:
        """Union of all shards must equal the full suite."""
        total_shards = 4
        all_test_counts: list[int] = []
        for shard_index in range(total_shards):
            suite = _make_test_suite_with_modules(self.MODULES)
            result = _filter_tests_by_shard(suite, shard_index, total_shards)
            all_test_counts.append(result.countTestCases())

        self.assertEqual(
            sum(all_test_counts),
            len(self.MODULES),
            f"Total tests across shards ({sum(all_test_counts)}) != total modules ({len(self.MODULES)})",
        )

    def test_no_overlap_between_shards(self) -> None:
        """Each module must appear in exactly one shard."""
        total_shards = 4
        seen_modules: list[str] = []
        for shard_index in range(total_shards):
            suite = _make_test_suite_with_modules(self.MODULES)
            result = _filter_tests_by_shard(suite, shard_index, total_shards)
            for test_group in result:
                assert isinstance(test_group, unittest.TestSuite)
                for test_case in test_group:
                    module = type(test_case).__module__
                    self.assertNotIn(
                        module,
                        seen_modules,
                        f"Module {module} appeared in multiple shards",
                    )
                    seen_modules.append(module)

    def test_deterministic_assignment(self) -> None:
        """Running the same shard twice must produce identical results."""
        total_shards = 3
        shard_index = 1
        suite1 = _make_test_suite_with_modules(self.MODULES)
        result1 = _filter_tests_by_shard(suite1, shard_index, total_shards)
        modules1 = [
            type(tc).__module__
            for tg in result1
            if isinstance(tg, unittest.TestSuite)
            for tc in tg
        ]

        suite2 = _make_test_suite_with_modules(self.MODULES)
        result2 = _filter_tests_by_shard(suite2, shard_index, total_shards)
        modules2 = [
            type(tc).__module__
            for tg in result2
            if isinstance(tg, unittest.TestSuite)
            for tc in tg
        ]

        self.assertEqual(modules1, modules2)

    def test_each_shard_gets_at_least_one_test_when_enough_modules(self) -> None:
        """With enough modules, each shard should get at least one test."""
        total_shards = 3
        for shard_index in range(total_shards):
            suite = _make_test_suite_with_modules(self.MODULES)
            result = _filter_tests_by_shard(suite, shard_index, total_shards)
            self.assertGreater(
                result.countTestCases(),
                0,
                f"Shard {shard_index} got no tests",
            )
