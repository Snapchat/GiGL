"""Unit tests for the argument handling and host checks of scripts.smoke_test_image."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from scripts import smoke_test_image
from tests.test_assets.test_case import TestCase


class SmokeTestImageArgsTest(TestCase):
    def test_exactly_one_of_imports_and_image_type_required(self) -> None:
        for extra in [[], ["--image-type", "cpu", "--imports", "torch"]]:
            with self.subTest(extra), self.assertRaises(SystemExit):
                smoke_test_image.parse_args(["--python", "3.11", *extra])

    def test_imports_without_image_type(self) -> None:
        args = smoke_test_image.parse_args(["--python", "3.11", "--imports", "torch"])
        self.assertEqual(args.imports, ["torch"])
        self.assertIsNone(args.image_type)

    def test_preset_imports(self) -> None:
        for image_type, expected in [
            ("cpu", ["torch", "graphlearn_torch"]),
            ("cuda", ["torch", "graphlearn_torch"]),
            ("builder", ["torch", "graphlearn_torch"]),
            ("dataflow", ["torch", "apache_beam"]),
        ]:
            self.assertEqual(
                smoke_test_image.IMAGE_TYPE_PRESETS[image_type].imports,
                expected,
                image_type,
            )

    def test_only_cuda_preset_requires_driver_path_and_cuda_build(self) -> None:
        for image_type, preset in smoke_test_image.IMAGE_TYPE_PRESETS.items():
            self.assertEqual(preset.require_nvidia_driver_path, image_type == "cuda")
            self.assertEqual(preset.require_cuda_build, image_type == "cuda")

    def test_only_dataflow_preset_requires_boot_env(self) -> None:
        for image_type, preset in smoke_test_image.IMAGE_TYPE_PRESETS.items():
            self.assertEqual(preset.boot_env, image_type == "dataflow")


class SmokeTestImageChecksTest(TestCase):
    def test_parse_version_compares_numerically(self) -> None:
        self.assertLess(
            smoke_test_image.parse_version("2.9"),
            smoke_test_image.parse_version("2.39"),
        )

    def test_nvidia_driver_path_accepts_both_mount_dirs_in_any_order(self) -> None:
        for value in [
            "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
            "/usr/local/cuda/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib",
        ]:
            with (
                self.subTest(value),
                mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": value}),
            ):
                smoke_test_image.check_nvidia_driver_path()

    def test_nvidia_driver_path_rejects_cuda_only_and_empty_paths(self) -> None:
        for value in ["/usr/local/cuda/lib64", "", "/opt/x/usr/local/nvidia/lib64x"]:
            with (
                self.subTest(value),
                mock.patch.dict(os.environ, {"LD_LIBRARY_PATH": value}),
            ):
                self.assertRaises(
                    AssertionError, smoke_test_image.check_nvidia_driver_path
                )


class SmokeTestImageTorchVariantTest(TestCase):
    @staticmethod
    def _fake_torch(version: str, cuda: str | None) -> mock._patch_dict:
        torch = SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda))
        return mock.patch.dict(
            sys.modules, {"torch": torch, "torch.version": torch.version}
        )

    def test_matching_variants_pass(self) -> None:
        with self._fake_torch("2.8.0+cu128", "12.8"):
            smoke_test_image.check_torch_variant("cu128")
        with self._fake_torch("2.8.0+cpu", None):
            smoke_test_image.check_torch_variant("cpu")

    def test_wrong_variant_fails(self) -> None:
        with self._fake_torch("2.8.0+cpu", None):
            with self.assertRaises(AssertionError):
                smoke_test_image.check_torch_variant("cu128")
        with self._fake_torch("2.8.0+cu128", "12.8"):
            with self.assertRaises(AssertionError):
                smoke_test_image.check_torch_variant("cpu")


if __name__ == "__main__":
    absltest.main()
