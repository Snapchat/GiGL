"""``.github/scripts/check_gigl_core_wheel.py`` checks of a built gigl-core wheel."""

import importlib.util
import tempfile
import zipfile
from pathlib import Path

from absl.testing import absltest

from tests.test_assets.test_case import TestCase

_SCRIPT = Path(__file__).parents[3] / ".github" / "scripts" / "check_gigl_core_wheel.py"
_spec = importlib.util.spec_from_file_location("check_gigl_core_wheel", _SCRIPT)
assert _spec is not None and _spec.loader is not None
check_gigl_core_wheel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_gigl_core_wheel)

_CUDA_DYNAMIC = " 0x0000000000000001 (NEEDED)  Shared library: [libtorch_cuda.so]\n"
_CPU_DYNAMIC = " 0x0000000000000001 (NEEDED)  Shared library: [libtorch_cpu.so]\n"


def _write_wheel(directory: str, requires_python: str) -> zipfile.ZipFile:
    path = Path(directory) / "gigl_core-0.0.0-cp312-cp312-linux_x86_64.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(
            "gigl_core-0.0.0.dist-info/METADATA",
            f"Name: gigl-core\nRequires-Python: {requires_python}\n",
        )
    return zipfile.ZipFile(path)


class CheckGiglCoreWheelTest(TestCase):
    def test_links_libtorch_cuda(self) -> None:
        self.assertTrue(check_gigl_core_wheel.links_libtorch_cuda(_CUDA_DYNAMIC))
        self.assertFalse(check_gigl_core_wheel.links_libtorch_cuda(_CPU_DYNAMIC))

    def test_abi_tag(self) -> None:
        name = "gigl_core-0.0.0-cp312-cp312-linux_x86_64.whl"
        check_gigl_core_wheel.check_abi_tag(name, "3.12")
        with self.assertRaises(AssertionError):
            check_gigl_core_wheel.check_abi_tag(name, "3.13")

    def test_requires_python_compares_as_a_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            # The wheel writes the specifiers in sorted order; pyproject writes them the other way round.
            wheel = _write_wheel(directory, "<3.14,>=3.11")
            check_gigl_core_wheel.check_requires_python(wheel, ">=3.11,<3.14")
            with self.assertRaises(AssertionError):
                check_gigl_core_wheel.check_requires_python(wheel, ">=3.11,<3.15")

    def test_wheel_without_extension_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wheel = _write_wheel(directory, ">=3.11")
            with self.assertRaises(AssertionError):
                check_gigl_core_wheel.check_torch_variant(wheel, "cpu")

    def test_find_wheel_requires_exactly_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                check_gigl_core_wheel.find_wheel(directory)
            _write_wheel(directory, ">=3.11")
            self.assertEqual(
                check_gigl_core_wheel.find_wheel(f"{directory}/*.whl").parent,
                Path(directory),
            )


if __name__ == "__main__":
    absltest.main()
