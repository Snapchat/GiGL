import tempfile
from pathlib import Path
from typing import Optional
from unittest import mock

from absl.testing import absltest
from parameterized import param, parameterized

from gigl.utils import host_memory
from tests.test_assets.test_case import TestCase

_GIB = 2**30


class CgroupResolutionTest(TestCase):
    """The limit must be read from the process's own cgroup, not the hierarchy root.

    Reading ``/sys/fs/cgroup/memory.max`` directly reports "unlimited", or nothing, and falls back to
    host memory.
    """

    def setUp(self) -> None:
        self._root = tempfile.TemporaryDirectory()
        self.addCleanup(self._root.cleanup)
        self.root = Path(self._root.name)

    def _write_v2(self, relative: str, limit: Optional[str], current: str) -> None:
        directory = self.root / relative.strip("/")
        directory.mkdir(parents=True, exist_ok=True)
        if limit is not None:
            (directory / "memory.max").write_text(f"{limit}\n")
        (directory / "memory.current").write_text(f"{current}\n")

    def _patch(
        self,
        cgroup_path: str,
        filesystem: str = "cgroup2",
        mount_root: str = "/",
    ):
        return (
            mock.patch.object(
                host_memory, "_cgroup_paths", return_value=_expand(cgroup_path)
            ),
            mock.patch.object(
                host_memory,
                "_cgroup_mounts",
                return_value=[(mount_root, str(self.root), filesystem)],
            ),
        )

    def test_reads_the_limit_from_a_nested_v2_cgroup(self):
        self._write_v2(
            "/kubepods/pod123/container", limit=str(64 * _GIB), current=str(10 * _GIB)
        )
        paths, mounts = self._patch("/kubepods/pod123/container")

        with paths, mounts:
            result = host_memory.cgroup_limit_and_usage()

        self.assertEqual(result, (64 * _GIB, 10 * _GIB))

    def test_walks_up_to_a_limit_set_on_an_ancestor(self):
        """A container is often limited at the pod, not at its own leaf."""
        self._write_v2("/kubepods/pod123", limit=str(32 * _GIB), current=str(4 * _GIB))
        self._write_v2("/kubepods/pod123/container", limit=None, current=str(1 * _GIB))
        paths, mounts = self._patch("/kubepods/pod123/container")

        with paths, mounts:
            result = host_memory.cgroup_limit_and_usage()

        self.assertEqual(result, (32 * _GIB, 4 * _GIB))

    @parameterized.expand(
        [
            param("v2 literal max", limit="max"),
            # cgroup v1 spells unlimited as a huge sentinel rather than a word.
            param("v1 sentinel", limit=str(1 << 63)),
        ]
    )
    def test_unlimited_reads_as_no_limit(self, _, limit: str):
        self._write_v2("/leaf", limit=limit, current=str(1 * _GIB))
        paths, mounts = self._patch("/leaf")

        with paths, mounts:
            self.assertIsNone(host_memory.cgroup_limit_and_usage())

    def test_missing_files_read_as_no_limit(self):
        paths, mounts = self._patch("/absent")

        with paths, mounts:
            self.assertIsNone(host_memory.cgroup_limit_and_usage())

    def test_reads_a_v1_controller_directory(self):
        directory = self.root / "memory" / "some" / "scope"
        directory.mkdir(parents=True)
        (directory / "memory.limit_in_bytes").write_text(f"{16 * _GIB}\n")
        (directory / "memory.usage_in_bytes").write_text(f"{3 * _GIB}\n")
        paths, mounts = self._patch("/some/scope", filesystem="cgroup")

        with paths, mounts:
            result = host_memory.cgroup_limit_and_usage()

        self.assertEqual(result, (16 * _GIB, 3 * _GIB))

    def test_the_tightest_ancestor_wins_not_the_first_found(self):
        """A leaf with room under a parent without it is limited by the parent, so returning the
        first finite limit found while walking up would report headroom that does not exist."""
        self._write_v2("/pod", limit=str(8 * _GIB), current=str(7 * _GIB))  # 1 GiB left
        self._write_v2(
            "/pod/container", limit=str(64 * _GIB), current=str(1 * _GIB)
        )  # 63 GiB left
        paths, mounts = self._patch("/pod/container")

        with paths, mounts:
            result = host_memory.cgroup_limit_and_usage()

        assert result is not None
        limit, current = result
        self.assertEqual((limit, current), (8 * _GIB, 7 * _GIB))
        self.assertEqual(limit - current, 1 * _GIB)

    def test_a_mount_rooted_below_the_hierarchy_root_is_translated(self):
        """A container often bind-mounts its own subtree, so the cgroup path is relative to
        mountinfo's root; joining it to the mount point untranslated finds no directory."""
        self._write_v2("/leaf", limit=str(4 * _GIB), current=str(1 * _GIB))
        paths, mounts = self._patch(
            "/kubepods/pod123/leaf", mount_root="/kubepods/pod123"
        )

        with paths, mounts:
            result = host_memory.cgroup_limit_and_usage()

        self.assertEqual(result, (4 * _GIB, 1 * _GIB))

    def test_a_path_outside_the_mount_root_is_skipped(self):
        self._write_v2("/leaf", limit=str(4 * _GIB), current=str(1 * _GIB))
        paths, mounts = self._patch("/elsewhere/leaf", mount_root="/kubepods")

        with paths, mounts:
            self.assertIsNone(host_memory.cgroup_limit_and_usage())

    @parameterized.expand(
        [
            param("root", path="/", mount_root="/", expected="/"),
            param("plain", path="/a/b", mount_root="/", expected="/a/b"),
            param("exact match", path="/a/b", mount_root="/a/b", expected="/"),
            param("below root", path="/a/b/c", mount_root="/a/b", expected="/c"),
            param("outside root", path="/x/y", mount_root="/a", expected=None),
            # A prefix that is not a path component must not match.
            param(
                "prefix but not component", path="/ab/c", mount_root="/a", expected=None
            ),
        ]
    )
    def test_relative_to_mount_root(
        self, _, path: str, mount_root: str, expected: Optional[str]
    ):
        self.assertEqual(
            host_memory._relative_to_mount_root(path, mount_root), expected
        )

    def test_cgroup_paths_come_from_proc_and_end_at_the_root(self):
        with mock.patch(
            "builtins.open",
            mock.mock_open(read_data="0::/kubepods/pod123/container\n"),
        ):
            paths = host_memory._cgroup_paths()

        self.assertEqual(paths[0], "/kubepods/pod123/container")
        self.assertEqual(paths[-1], "/")


class AvailableMemoryTest(TestCase):
    def test_cgroup_headroom_wins_when_it_is_the_tighter_limit(self):
        """The point of the whole module: meminfo inside a container reports the HOST."""
        with (
            mock.patch.object(
                host_memory.psutil,
                "virtual_memory",
                return_value=mock.Mock(available=200 * _GIB, total=256 * _GIB),
            ),
            mock.patch.object(
                host_memory,
                "cgroup_limit_and_usage",
                return_value=(64 * _GIB, 60 * _GIB),
            ),
        ):
            self.assertEqual(host_memory.available_memory_bytes(), 4 * _GIB)

    def test_os_availability_wins_when_the_host_is_under_pressure(self):
        with (
            mock.patch.object(
                host_memory.psutil,
                "virtual_memory",
                return_value=mock.Mock(available=2 * _GIB, total=256 * _GIB),
            ),
            mock.patch.object(
                host_memory,
                "cgroup_limit_and_usage",
                return_value=(64 * _GIB, 10 * _GIB),
            ),
        ):
            self.assertEqual(host_memory.available_memory_bytes(), 2 * _GIB)

    def test_a_cgroup_over_its_limit_reports_no_headroom_rather_than_negative(self):
        with (
            mock.patch.object(
                host_memory.psutil,
                "virtual_memory",
                return_value=mock.Mock(available=100 * _GIB, total=256 * _GIB),
            ),
            mock.patch.object(
                host_memory,
                "cgroup_limit_and_usage",
                return_value=(64 * _GIB, 70 * _GIB),
            ),
        ):
            self.assertEqual(host_memory.available_memory_bytes(), 0)

    def test_describe_memory_names_both_views(self):
        text = host_memory.describe_memory()

        self.assertIn("meminfo", text)
        self.assertIn("cgroup", text)
        self.assertIn("effective available", text)

    def test_available_memory_is_within_bounds_on_this_host(self):
        """No mocks: on any host the answer must fall between zero and total memory.

        Zero is a legitimate answer for a cgroup at its limit, and the upper bound is TOTAL rather
        than a second reading of AVAILABLE, which moves between calls.
        """
        available = host_memory.available_memory_bytes()

        self.assertGreaterEqual(available, 0)
        self.assertLessEqual(available, host_memory.psutil.virtual_memory().total)


class MemoryBreakdownTest(CgroupResolutionTest):
    """``memory.current`` alone cannot say whether a peak is survivable.

    A process near its limit lives when the bytes are dirty page cache and dies when they are
    anonymous, and the two are indistinguishable on a usage graph.
    """

    def _write_stat(self, relative: str, lines: str) -> None:
        directory = self.root / relative.strip("/")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "memory.stat").write_text(lines)

    def test_reads_the_v2_field_names(self):
        self._write_v2("/leaf", limit=str(100 * _GIB), current=str(90 * _GIB))
        self._write_stat(
            "/leaf",
            "anon 48318382080\nfile 45097156608\nshmem 1073741824\n"
            "file_dirty 32212254720\nfile_writeback 1073741824\nslab 12345\n",
        )
        paths, mounts = self._patch("/leaf")

        with paths, mounts:
            breakdown = host_memory.cgroup_memory_breakdown()

        self.assertEqual(breakdown["anon"], 45 * _GIB)
        self.assertEqual(breakdown["file_dirty"], 30 * _GIB)
        self.assertEqual(breakdown["shmem"], _GIB)
        self.assertNotIn("slab", breakdown, "only the load-bearing fields are reported")

    def _write_v1(self, relative: str, limit: int, usage: int, stat: str) -> None:
        """A v1 memory-controller directory, the way `_tightest_cgroup` resolves one."""
        directory = self.root / "memory" / relative.strip("/")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "memory.limit_in_bytes").write_text(f"{limit}\n")
        (directory / "memory.usage_in_bytes").write_text(f"{usage}\n")
        (directory / "memory.stat").write_text(stat)

    def test_v1_prefers_hierarchical_totals_over_local_fields(self):
        """A v1 ancestor's usage includes descendants while its bare rss/cache do not, so the
        breakdown must use total_*. The local and total_* values here disagree deliberately."""
        self._write_v1(
            "/pod",
            limit=100 * _GIB,
            usage=90 * _GIB,
            stat=(
                f"rss {2 * _GIB}\ncache {1 * _GIB}\ndirty {_GIB // 4}\n"
                f"total_rss {80 * _GIB}\ntotal_cache {10 * _GIB}\ntotal_dirty {5 * _GIB}\n"
            ),
        )
        paths, mounts = self._patch("/pod", filesystem="cgroup")

        with paths, mounts:
            breakdown = host_memory.cgroup_memory_breakdown()

        self.assertEqual(breakdown["anon"], 80 * _GIB, "must be total_rss, not rss")
        self.assertEqual(breakdown["file"], 10 * _GIB)
        self.assertEqual(breakdown["file_dirty"], 5 * _GIB)

    def test_v1_falls_back_to_local_fields_when_totals_are_absent(self):
        self._write_v1(
            "/leaf",
            limit=100 * _GIB,
            usage=10 * _GIB,
            stat=f"rss {8 * _GIB}\ncache {1 * _GIB}\ndirty {_GIB // 2}\n",
        )
        paths, mounts = self._patch("/leaf", filesystem="cgroup")

        with paths, mounts:
            breakdown = host_memory.cgroup_memory_breakdown()

        self.assertEqual(breakdown["anon"], 8 * _GIB)
        self.assertEqual(breakdown["file"], _GIB)
        self.assertEqual(breakdown["file_dirty"], _GIB // 2)

    def test_v2_ignores_v1_field_names(self):
        """A v2 stat carrying stray v1-style names must not be misread as hierarchical."""
        self._write_v2("/leaf", limit=str(100 * _GIB), current=str(10 * _GIB))
        self._write_stat("/leaf", f"anon {6 * _GIB}\nrss {1 * _GIB}\n")
        paths, mounts = self._patch("/leaf")

        with paths, mounts:
            breakdown = host_memory.cgroup_memory_breakdown()

        self.assertEqual(breakdown["anon"], 6 * _GIB)

    def test_the_breakdown_comes_from_the_cgroup_the_limit_came_from(self):
        """A breakdown from a different level would not add up to the reported usage."""
        # The leaf has the loose limit; the ancestor is the binding one, and they disagree.
        self._write_v2("/pod/leaf", limit=str(100 * _GIB), current=str(1 * _GIB))
        self._write_stat("/pod/leaf", "anon 1073741824\n")
        self._write_v2("/pod", limit=str(50 * _GIB), current=str(49 * _GIB))
        self._write_stat("/pod", "anon 52613349376\n")
        paths, mounts = self._patch("/pod/leaf")

        with paths, mounts:
            breakdown = host_memory.cgroup_memory_breakdown()

        self.assertEqual(
            breakdown["anon"],
            49 * _GIB,
            "should read /pod's stat -- the BINDING cgroup, the one the limit came from -- not the "
            "leaf's, whose 1 GiB would understate the position by 48 GiB",
        )

    def test_no_stat_file_reads_as_no_breakdown(self):
        self._write_v2("/leaf", limit=str(100 * _GIB), current=str(10 * _GIB))
        paths, mounts = self._patch("/leaf")

        with paths, mounts:
            self.assertEqual(host_memory.cgroup_memory_breakdown(), {})

    def test_log_stage_memory_names_the_stage_and_both_categories(self):
        self._write_v2("/leaf", limit=str(100 * _GIB), current=str(90 * _GIB))
        self._write_stat(
            "/leaf", "anon 10737418240\nfile 85899345920\nfile_dirty 64424509440\n"
        )
        paths, mounts = self._patch("/leaf")

        with paths, mounts, mock.patch.object(host_memory.logger, "info") as info:
            host_memory.log_stage_memory("built CSR")

        line = info.call_args[0][0]
        self.assertIn("after built CSR", line)
        self.assertIn("90.0/100.0 GiB (90%)", line)
        self.assertIn("anon 10.0", line)
        self.assertIn("dirty 60.0", line)

    def test_log_stage_memory_says_so_when_there_is_no_limit(self):
        paths, mounts = self._patch("/absent")

        with paths, mounts, mock.patch.object(host_memory.logger, "info") as info:
            host_memory.log_stage_memory("loaded edges")

        self.assertIn("cgroup=unlimited", info.call_args[0][0])


def _expand(path: str) -> list[str]:
    """The candidate list _cgroup_paths would produce for one path: itself, its ancestors, root."""
    parts = [part for part in path.split("/") if part]
    candidates = []
    while True:
        candidates.append("/" + "/".join(parts))
        if not parts:
            break
        parts.pop()
    return candidates


if __name__ == "__main__":
    absltest.main()
