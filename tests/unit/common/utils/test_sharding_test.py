import hashlib
import unittest

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.common.utils.test_utils import _filter_tests_by_shard, _get_shard_for_module
from tests.integration.main import INTEGRATION_TEST_SHARD_PINNED_MODULES
from tests.test_assets.test_case import TestCase
from tests.unit.main import UNIT_TEST_SHARD_PINNED_MODULES


def _extract_module_names(suite: unittest.TestSuite) -> list[str]:
    """Extracts module names from a filtered test suite, preserving order.

    Assumes the suite has the two-level nesting produced by
    ``_make_test_suite_with_modules``: outer suite → inner TestSuite per
    module → individual TestCase(s).

    Args:
        suite: A filtered test suite.

    Returns:
        Ordered list of module name strings found in the suite.
    """
    return [
        type(test_case).__module__
        for test_group in suite
        if isinstance(test_group, unittest.TestSuite)
        for test_case in test_group
    ]


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
        seen_modules: set[str] = set()
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
                    seen_modules.add(module)

    def test_deterministic_assignment(self) -> None:
        """Running the same shard twice must produce identical results."""
        total_shards = 3
        shard_index = 1
        suite1 = _make_test_suite_with_modules(self.MODULES)
        result1 = _filter_tests_by_shard(suite1, shard_index, total_shards)
        modules1 = _extract_module_names(result1)

        suite2 = _make_test_suite_with_modules(self.MODULES)
        result2 = _filter_tests_by_shard(suite2, shard_index, total_shards)
        modules2 = _extract_module_names(result2)

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


class ShardPinningTest(TestCase):
    """Tests for manual shard pinning via pinned_modules."""

    PINNED: tuple[str, ...] = (
        "tests.unit.distributed.dist_ablp_neighborloader_test",
        "tests.unit.distributed.distributed_dataset_test",
        "tests.unit.distributed.distributed_neighborloader_test",
        "tests.unit.distributed.distributed_partitioner_test",
        "tests.unit.distributed.utils.networking_test",
    )

    UNPINNED: list[str] = [
        "tests.unit.module_a_test",
        "tests.unit.module_b_test",
        "tests.unit.module_c_test",
        "tests.unit.module_d_test",
        "tests.unit.module_e_test",
    ]

    def test_pinned_modules_assigned_by_position(self) -> None:
        """Pinned module at index i is assigned to shard i % total_shards."""
        total_shards = 4
        for index, module_name in enumerate(self.PINNED):
            expected_shard = index % total_shards
            actual_shard = _get_shard_for_module(module_name, total_shards, self.PINNED)
            self.assertEqual(
                actual_shard,
                expected_shard,
                f"Pinned module {module_name} (index {index}) expected shard "
                f"{expected_shard}, got {actual_shard}",
            )

    def test_pinned_modules_use_all_shards_with_four_shards(self) -> None:
        """With 5 pinned modules and 4 shards, every shard gets at least one."""
        total_shards = 4
        assigned_shards = {
            _get_shard_for_module(m, total_shards, self.PINNED) for m in self.PINNED
        }
        self.assertEqual(
            len(assigned_shards),
            min(len(self.PINNED), total_shards),
            f"Expected {min(len(self.PINNED), total_shards)} distinct shards, "
            f"got {assigned_shards}",
        )

    def test_full_coverage_no_overlap_with_pinned_and_unpinned(self) -> None:
        """All modules appear exactly once across all shards."""
        total_shards = 4
        all_modules = list(self.PINNED) + self.UNPINNED

        seen_modules: set[str] = set()
        for shard_index in range(total_shards):
            fresh_suite = _make_test_suite_with_modules(all_modules)
            result = _filter_tests_by_shard(
                fresh_suite, shard_index, total_shards, pinned_modules=self.PINNED
            )
            for test_group in result:
                assert isinstance(test_group, unittest.TestSuite)
                for test_case in test_group:
                    module = type(test_case).__module__
                    self.assertNotIn(
                        module,
                        seen_modules,
                        f"Module {module} appeared in multiple shards",
                    )
                    seen_modules.add(module)

        self.assertEqual(
            seen_modules,
            set(all_modules),
            "Not all modules were covered across shards",
        )

    def test_pinning_across_various_total_shards(self) -> None:
        """Pinned modules land on expected shards for several shard counts."""
        for total_shards in (2, 3, 4, 5, 8):
            for index, module_name in enumerate(self.PINNED):
                expected_shard = index % total_shards
                actual_shard = _get_shard_for_module(
                    module_name, total_shards, self.PINNED
                )
                self.assertEqual(
                    actual_shard,
                    expected_shard,
                    f"total_shards={total_shards}: pinned module {module_name} "
                    f"(index {index}) expected shard {expected_shard}, got {actual_shard}",
                )

    def test_unpinned_modules_use_hash(self) -> None:
        """Unpinned modules still use SHA-256 hashing, unaffected by pinned list."""
        total_shards = 4
        for module_name in self.UNPINNED:
            expected = (
                int(hashlib.sha256(module_name.encode()).hexdigest(), 16) % total_shards
            )
            actual = _get_shard_for_module(module_name, total_shards, self.PINNED)
            self.assertEqual(
                actual,
                expected,
                f"Unpinned module {module_name} should use hash-based assignment",
            )

    def test_real_unit_test_pinned_modules_cover_all_shards(self) -> None:
        """The actual UNIT_TEST_SHARD_PINNED_MODULES cover every shard with 4 shards."""
        total_shards = 4
        assigned_shards = {
            _get_shard_for_module(m, total_shards, UNIT_TEST_SHARD_PINNED_MODULES)
            for m in UNIT_TEST_SHARD_PINNED_MODULES
        }
        self.assertEqual(
            assigned_shards,
            set(range(total_shards)),
            f"Expected all shards 0..{total_shards - 1} covered, got {assigned_shards}",
        )

    def test_real_integration_test_pinned_modules_cover_all_shards(self) -> None:
        """The actual INTEGRATION_TEST_SHARD_PINNED_MODULES cover every shard with 4 shards."""
        total_shards = 4
        assigned_shards = {
            _get_shard_for_module(
                m, total_shards, INTEGRATION_TEST_SHARD_PINNED_MODULES
            )
            for m in INTEGRATION_TEST_SHARD_PINNED_MODULES
        }
        self.assertEqual(
            assigned_shards,
            set(range(total_shards)),
            f"Expected all shards 0..{total_shards - 1} covered, got {assigned_shards}",
        )


def _collect_test_ids(suite: unittest.TestSuite) -> set[str]:
    """Recursively collects all individual test case IDs from a suite.

    Args:
        suite: A (possibly nested) test suite.

    Returns:
        Set of fully-qualified test IDs (e.g. ``module.Class.test_method``).
    """
    ids: set[str] = set()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            ids.update(_collect_test_ids(item))
        else:
            ids.add(item.id())
    return ids


class RealDiscoveryShardingTest(TestCase):
    """Discovers real unit tests and verifies sharding preserves them all."""

    TOTAL_SHARDS: int = 4

    @classmethod
    def setUpClass(cls) -> None:
        start_dir = LocalUri.join(
            local_fs_constants.get_project_root_directory(), "tests", "unit"
        )
        cls.start_dir = start_dir
        full_suite = unittest.TestLoader().discover(
            start_dir=start_dir.uri, pattern="*_test.py"
        )
        cls.unsharded_test_ids = _collect_test_ids(full_suite)

    def test_sharded_tests_equal_unsharded(self) -> None:
        """Union of test IDs across all shards equals the full unsharded set."""
        sharded_test_ids: set[str] = set()
        for shard_index in range(self.TOTAL_SHARDS):
            suite = unittest.TestLoader().discover(
                start_dir=self.start_dir.uri, pattern="*_test.py"
            )
            filtered = _filter_tests_by_shard(
                suite,
                shard_index,
                self.TOTAL_SHARDS,
                UNIT_TEST_SHARD_PINNED_MODULES,
            )
            shard_ids = _collect_test_ids(filtered)
            overlap = sharded_test_ids & shard_ids
            self.assertEqual(
                overlap,
                set(),
                f"Shard {shard_index} overlaps with previous shards: {overlap}",
            )
            sharded_test_ids.update(shard_ids)

        self.assertEqual(
            sharded_test_ids,
            self.unsharded_test_ids,
            f"Test ID mismatch.\n"
            f"  Only in sharded: {sharded_test_ids - self.unsharded_test_ids}\n"
            f"  Only in unsharded: {self.unsharded_test_ids - sharded_test_ids}",
        )
