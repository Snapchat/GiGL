"""Unit tests for the argument handling and host checks of scripts.smoke_test_image."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest

from scripts import smoke_test_image
from tests.test_assets.test_case import TestCase


class SmokeTestImageArgsTest(TestCase):
    def test_imports_required_without_image_type(self) -> None:
        with self.assertRaises(SystemExit):
            smoke_test_image.parse_args(["--python", "3.11"])

    def test_explicit_imports_without_image_type_are_unchanged(self) -> None:
        args = smoke_test_image.parse_args(["--python", "3.11", "--imports", "torch"])
        self.assertEqual(args.imports, ["torch"])
        self.assertIsNone(args.venv_prefix)
        self.assertIsNone(args.min_glibc)
        self.assertFalse(args.require_nvidia_driver_path)

    def test_cpu_preset(self) -> None:
        args = smoke_test_image.parse_args(["--python", "3.11", "--image-type", "cpu"])
        self.assertEqual(args.imports, ["torch", "graphlearn_torch"])
        self.assertEqual(args.venv_prefix, "/gigl_deps/.venv")
        self.assertEqual(args.min_glibc, "2.39")
        self.assertFalse(args.require_nvidia_driver_path)

    def test_cuda_preset_adds_torchrec_only_with_cuda_flag(self) -> None:
        no_gpu = smoke_test_image.parse_args(
            ["--python", "3.11", "--image-type", "cuda"]
        )
        self.assertEqual(no_gpu.imports, ["torch", "graphlearn_torch"])
        self.assertTrue(no_gpu.require_nvidia_driver_path)
        gpu = smoke_test_image.parse_args(
            ["--python", "3.11", "--image-type", "cuda", "--cuda"]
        )
        self.assertEqual(gpu.imports, ["torch", "graphlearn_torch", "torchrec"])

    def test_only_cuda_preset_requires_cuda_build(self) -> None:
        for image_type, expected in [
            ("cuda", True),
            ("cpu", False),
            ("builder", False),
            ("dataflow", False),
        ]:
            args = smoke_test_image.parse_args(
                ["--python", "3.11", "--image-type", image_type]
            )
            self.assertEqual(args.require_cuda_build, expected, image_type)

    def test_dataflow_preset(self) -> None:
        args = smoke_test_image.parse_args(
            ["--python", "3.11", "--image-type", "dataflow"]
        )
        self.assertEqual(args.imports, ["torch", "apache_beam"])
        self.assertTrue(args.boot_env)

    def test_explicit_flags_add_to_and_override_preset(self) -> None:
        args = smoke_test_image.parse_args(
            [
                "--python",
                "3.11",
                "--image-type",
                "builder",
                "--imports",
                "torch,gigl_core",
                "--venv-prefix",
                "/other/.venv",
                "--min-glibc",
                "2.35",
            ]
        )
        self.assertEqual(args.imports, ["torch", "graphlearn_torch", "gigl_core"])
        self.assertEqual(args.venv_prefix, "/other/.venv")
        self.assertEqual(args.min_glibc, "2.35")


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
