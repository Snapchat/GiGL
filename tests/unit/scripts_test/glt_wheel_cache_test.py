"""make unit_test_py PY_TEST_FILES="glt_wheel_cache_test.py"

Cache-key, patch-ordering and publish semantics of ``gigl/scripts/glt_wheel_cache.py``, plus the
hit/miss/evict orchestration in ``post_install.install_glt_wheel`` driven with real (tiny) build,
install and verify callables instead of mocks.
"""

import subprocess
import tempfile
from pathlib import Path

from absl.testing import absltest

from gigl.scripts.glt_wheel_cache import (
    compute_cache_key,
    find_cached_wheel,
    publish_wheel,
    read_patches,
)
from gigl.scripts.post_install import install_glt_wheel
from tests.test_assets.test_case import TestCase

_BASE_INPUTS = dict(
    commit="88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9",
    patch_contents=[b"a", b"b"],
    variant="cuda7.5",
    torch_version="2.8.0+cu128",
    python_tag="cp311",
)


class ComputeCacheKeyTest(TestCase):
    def test_key_is_deterministic_and_formatted(self) -> None:
        key = compute_cache_key(**_BASE_INPUTS)
        self.assertEqual(key, compute_cache_key(**_BASE_INPUTS))
        self.assertRegex(
            key, r"^glt-88ff111a-p[0-9a-f]{12}-cuda7\.5-torch2\.8\.0\+cu128-cp311$"
        )

    def test_changing_any_single_input_changes_the_key(self) -> None:
        base = compute_cache_key(**_BASE_INPUTS)
        changes = dict(
            commit="99ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9",
            patch_contents=[b"a", b"c"],
            variant="cpu",
            torch_version="2.9.0",
            python_tag="cp312",
        )
        for name, value in changes.items():
            with self.subTest(changed=name):
                self.assertNotEqual(
                    base, compute_cache_key(**{**_BASE_INPUTS, name: value})
                )

    def test_patch_order_matters(self) -> None:
        self.assertNotEqual(
            compute_cache_key(**{**_BASE_INPUTS, "patch_contents": [b"a", b"b"]}),
            compute_cache_key(**{**_BASE_INPUTS, "patch_contents": [b"b", b"a"]}),
        )


class ReadPatchesTest(TestCase):
    def test_sorted_by_filename_regardless_of_creation_order(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            for name in ["0002-b.patch", "0001-a.patch", "0003-c.patch"]:
                (Path(d) / name).write_bytes(name.encode())
            (Path(d) / "README.md").write_bytes(b"ignored")
            self.assertEqual(
                read_patches(Path(d)),
                [b"0001-a.patch", b"0002-b.patch", b"0003-c.patch"],
            )


class CacheEntryTest(TestCase):
    def test_find_cached_wheel_returns_none_for_empty_or_missing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(find_cached_wheel(Path(d) / "missing"))
            (Path(d) / "empty").mkdir()
            self.assertIsNone(find_cached_wheel(Path(d) / "empty"))

    def test_publish_wheel_replaces_existing_entry_and_leaves_no_temp_dirs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            entry = root / "cache" / "key"
            entry.mkdir(parents=True)
            (entry / "stale.whl").write_bytes(b"stale")
            wheel = root / "new.whl"
            wheel.write_bytes(b"new")

            published = publish_wheel(wheel, entry)

            self.assertEqual(published, entry / "new.whl")
            self.assertEqual(find_cached_wheel(entry), published)
            self.assertEqual([p.name for p in entry.iterdir()], ["new.whl"])
            self.assertEqual([p.name for p in (root / "cache").iterdir()], ["key"])


class InstallGltWheelTest(TestCase):
    """Drives the orchestration with callables that record calls and fail on demand."""

    def setUp(self) -> None:
        super().setUp()
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.entry = Path(self._dir.name) / "cache" / "key"
        self.events: list[str] = []
        self.verify_failures = 0

    def _build(self, output_dir: Path) -> None:
        self.events.append("build")
        (output_dir / "glt-0.whl").write_bytes(b"built")

    def _install(self, wheel: Path) -> None:
        self.events.append(f"install:{wheel.read_bytes().decode()}")
        if wheel.read_bytes() == b"corrupt":
            raise subprocess.CalledProcessError(1, "uv pip install")

    def _verify(self) -> None:
        self.events.append("verify")
        if self.verify_failures:
            self.verify_failures -= 1
            raise subprocess.CalledProcessError(1, "verify")

    def _run(self) -> None:
        install_glt_wheel(self.entry, self._build, self._install, self._verify)

    def test_miss_builds_publishes_installs_and_verifies(self) -> None:
        self._run()
        self.assertEqual(self.events, ["build", "install:built", "verify"])
        self.assertIsNotNone(find_cached_wheel(self.entry))

    def test_hit_skips_the_build(self) -> None:
        self._run()
        self.events.clear()
        self._run()
        self.assertEqual(self.events, ["install:built", "verify"])

    def test_corrupt_cached_wheel_is_evicted_and_rebuilt(self) -> None:
        self.entry.mkdir(parents=True)
        (self.entry / "glt-0.whl").write_bytes(b"corrupt")
        self._run()
        self.assertEqual(
            self.events, ["install:corrupt", "build", "install:built", "verify"]
        )
        self.assertEqual((self.entry / "glt-0.whl").read_bytes(), b"built")

    def test_verify_failure_on_hit_evicts_and_rebuilds(self) -> None:
        self._run()
        self.events.clear()
        self.verify_failures = 1
        self._run()
        self.assertEqual(
            self.events,
            ["install:built", "verify", "build", "install:built", "verify"],
        )

    def test_verify_failure_after_fresh_build_is_fatal(self) -> None:
        self.verify_failures = 1
        with self.assertRaises(subprocess.CalledProcessError):
            self._run()


if __name__ == "__main__":
    absltest.main()
