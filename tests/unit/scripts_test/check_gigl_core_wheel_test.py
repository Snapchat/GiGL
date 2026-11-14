"""``.github/scripts/check_gigl_core_wheel.py`` checks of a built gigl-core wheel."""

import importlib.util
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Optional

from absl.testing import absltest

from tests.test_assets.test_case import TestCase

_SCRIPT = Path(__file__).parents[3] / ".github" / "scripts" / "check_gigl_core_wheel.py"
_spec = importlib.util.spec_from_file_location("check_gigl_core_wheel", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check_gigl_core_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_gigl_core_wheel)

_CUDA_DYNAMIC = " 0x0000000000000001 (NEEDED)  Shared library: [libtorch_cuda.so]\n"
_CPU_DYNAMIC = " 0x0000000000000001 (NEEDED)  Shared library: [libtorch_cpu.so]\n"


def _write_wheel(directory: str, extension: Optional[Path] = None) -> zipfile.ZipFile:
    """Writes a wheel holding ``gigl_core/x.so`` (a copy of ``extension``) when given, else no extension."""
    path = Path(directory) / "gigl_core-0.0.0-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("gigl_core-0.0.0.dist-info/METADATA", "Name: gigl-core\n")
        if extension is not None:
            wheel.write(extension, "gigl_core/x.so")
    return zipfile.ZipFile(path)


def _build_extension(directory: str, links_cuda: bool) -> Path:
    """Compiles an empty ``x.so``; with ``links_cuda`` it lists a stub ``libtorch_cuda.so`` as NEEDED."""
    source = Path(directory) / "x.c"
    source.write_text("int f(void) { return 0; }\n")
    cuda_stub = Path(directory) / "libtorch_cuda.so"
    subprocess.run(
        [
            "cc",
            "-shared",
            "-fPIC",
            "-Wl,-soname,libtorch_cuda.so",
            "-o",
            str(cuda_stub),
            str(source),
        ],
        check=True,
    )
    extension = Path(directory) / "x.so"
    link = (
        ["-L" + directory, "-Wl,--no-as-needed", "-ltorch_cuda"] if links_cuda else []
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-o", str(extension), str(source), *link], check=True
    )
    return extension


class CheckGiglCoreWheelTest(TestCase):
    def test_links_libtorch_cuda(self) -> None:
        self.assertTrue(check_gigl_core_wheel.links_libtorch_cuda(_CUDA_DYNAMIC))
        self.assertFalse(check_gigl_core_wheel.links_libtorch_cuda(_CPU_DYNAMIC))

    def test_abi_tag(self) -> None:
        name = "gigl_core-0.0.0-cp312-cp312-linux_x86_64.whl"
        check_gigl_core_wheel.check_abi_tag(name, "3.12")
        with self.assertRaises(AssertionError):
            check_gigl_core_wheel.check_abi_tag(name, "3.13")

    @unittest.skipUnless(
        shutil.which("cc") and shutil.which("readelf"), "needs cc and readelf"
    )
    def test_torch_variant_reads_the_needed_libraries_of_a_real_extension(self) -> None:
        for links_cuda, variant, other_variant in (
            (False, "cpu", "cu128"),
            (True, "cu128", "cpu"),
        ):
            with tempfile.TemporaryDirectory() as directory:
                wheel = _write_wheel(directory, _build_extension(directory, links_cuda))
                check_gigl_core_wheel.check_torch_variant(wheel, variant)
                with self.assertRaises(AssertionError):
                    check_gigl_core_wheel.check_torch_variant(wheel, other_variant)

    def test_wheel_without_extension_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = _write_wheel(directory)
            with self.assertRaises(AssertionError):
                check_gigl_core_wheel.check_torch_variant(wheel, "cpu")

    def test_find_wheel_requires_exactly_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                check_gigl_core_wheel.find_wheel(directory)
            _write_wheel(directory)
            self.assertEqual(
                check_gigl_core_wheel.find_wheel(f"{directory}/*.whl").parent,
                Path(directory),
            )


if __name__ == "__main__":
    absltest.main()
