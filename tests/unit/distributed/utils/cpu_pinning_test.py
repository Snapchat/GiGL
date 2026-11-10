import os
import subprocess
import sys
import textwrap

from gigl.distributed.utils.cpu_pinning import split_cpus_for_process
from tests.test_assets.test_case import TestCase


def _topology(num_cores: int, num_allowed_cpus: int):
    """Cores of two hyperthreads each (cpu ``2k`` and ``2k + 1``); the first CPUs are allowed."""
    allowed = frozenset(range(num_allowed_cpus))
    groups = [
        frozenset({2 * core, 2 * core + 1}) & allowed for core in range(num_cores)
    ]
    return allowed, [group for group in groups if group]


def _all_sets(allowed, groups, num_ranks, workers):
    sets = {}
    for rank in range(num_ranks):
        sets[(rank, None)] = split_cpus_for_process(
            allowed, groups, rank, num_ranks, None, workers
        )
        for index in range(workers):
            sets[(rank, index)] = split_cpus_for_process(
                allowed, groups, rank, num_ranks, index, workers
            )
    return sets


class SplitCpusForProcessTest(TestCase):
    def test_four_ranks_two_workers_on_46_cpus(self) -> None:
        allowed, groups = _topology(num_cores=24, num_allowed_cpus=46)
        sets = _all_sets(allowed, groups, 4, 2)
        sizes = {key: len(value) // 2 for key, value in sets.items()}
        self.assertEqual(
            sizes,
            {
                (0, None): 1, (0, 0): 3, (0, 1): 2,
                (1, None): 1, (1, 0): 3, (1, 1): 2,
                (2, None): 1, (2, 0): 3, (2, 1): 2,
                (3, None): 1, (3, 0): 2, (3, 1): 2,
            },
        )  # fmt: skip
        self.assertEqual(sets[(0, None)], frozenset({0, 1}))

    def test_sets_are_disjoint_within_allowed_and_keep_siblings_together(self) -> None:
        allowed, groups = _topology(num_cores=24, num_allowed_cpus=46)
        sets = _all_sets(allowed, groups, 4, 2)
        seen: set[int] = set()
        for cpus in sets.values():
            self.assertTrue(cpus)
            self.assertFalse(seen & cpus)
            seen |= cpus
            self.assertLessEqual(cpus, allowed)
            for group in groups:
                if group & cpus:
                    self.assertLessEqual(group, cpus)
        self.assertEqual(seen, allowed)

    def test_even_split_gives_ranks_a_smaller_share(self) -> None:
        allowed, groups = _topology(num_cores=24, num_allowed_cpus=48)
        sets = _all_sets(allowed, groups, 4, 2)
        self.assertEqual(len(sets[(2, None)]), 2)
        self.assertEqual(len(sets[(2, 0)]), 2 * 3)
        self.assertEqual(len(sets[(2, 1)]), 2 * 2)

    def test_fewer_cores_than_processes_returns_none(self) -> None:
        allowed, groups = _topology(num_cores=11, num_allowed_cpus=22)
        self.assertIsNone(split_cpus_for_process(allowed, groups, 0, 4, None, 2))

    def test_split_is_stable_for_the_same_inputs(self) -> None:
        allowed, groups = _topology(num_cores=24, num_allowed_cpus=46)
        first = _all_sets(allowed, groups, 4, 2)
        second = _all_sets(allowed, list(reversed(groups)), 4, 2)
        self.assertEqual(first, second)


class ApplyCpuPinningTest(TestCase):
    def test_disjoint_pinning_sets_the_affinity_of_a_subprocess_only(self) -> None:
        allowed = os.sched_getaffinity(0)
        if len(allowed) < 4:
            self.skipTest("needs at least 4 allowed CPUs")
        script = textwrap.dedent(
            """
            import os
            from gigl.distributed.utils.cpu_pinning import apply_cpu_pinning

            before = os.sched_getaffinity(0)
            chosen = apply_cpu_pinning(
                "disjoint", local_rank=0, worker_index=None,
                num_local_ranks=1, workers_per_rank=1,
            )
            assert chosen is not None
            assert os.sched_getaffinity(0) == set(chosen), (os.sched_getaffinity(0), chosen)
            assert set(chosen) < before
            """
        )
        env = {key: value for key, value in os.environ.items()}
        env["PYTHONPATH"] = os.getcwd()
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(os.sched_getaffinity(0), allowed)

    def test_worker_pins_to_its_own_share_from_the_rank_context(self) -> None:
        allowed = os.sched_getaffinity(0)
        if len(allowed) < 6:
            self.skipTest("needs at least 6 allowed CPUs")
        script = textwrap.dedent(
            """
            import os
            from gigl.distributed.utils.cpu_pinning import apply_cpu_pinning, apply_worker_cpu_pinning

            rank_cpus = apply_cpu_pinning(
                "disjoint", local_rank=0, worker_index=None,
                num_local_ranks=1, workers_per_rank=2,
            )
            worker_cpus = apply_worker_cpu_pinning(1)
            assert rank_cpus is not None and worker_cpus is not None
            assert not set(rank_cpus) & set(worker_cpus)
            assert os.sched_getaffinity(0) == set(worker_cpus)
            """
        )
        env = dict(
            os.environ, PYTHONPATH=os.getcwd(), GIGL_SAMPLING_CPU_PINNING="disjoint"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_none_mode_leaves_the_affinity_alone(self) -> None:
        script = textwrap.dedent(
            """
            import os
            from gigl.distributed.utils.cpu_pinning import apply_cpu_pinning

            before = os.sched_getaffinity(0)
            assert apply_cpu_pinning("none", 0, None, 1, 1) is None
            assert os.sched_getaffinity(0) == before
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=dict(os.environ, PYTHONPATH=os.getcwd()),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
