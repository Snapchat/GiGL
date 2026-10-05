import sys

from absl.testing import absltest

from gigl.common import constants
from gigl.common.constants import PATH_BASE_IMAGES_VARIABLE_FILE, parse_makefile_vars
from tests.test_assets.test_case import TestCase

# Constants read from a dep_vars.env key that exists once per supported Python minor.
_PER_MINOR_CONSTANTS = [
    "DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG",
    "DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG",
    "DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU",
    "DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH",
]
_SUPPORTED_KEY_SUFFIXES = ["PY311", "PY312", "PY313"]


class PerMinorConstantsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._dep_vars = parse_makefile_vars(PATH_BASE_IMAGES_VARIABLE_FILE)

    def test_constants_use_running_python(self) -> None:
        key_suffix = f"PY{sys.version_info.major}{sys.version_info.minor}"
        for name in _PER_MINOR_CONSTANTS:
            with self.subTest(name):
                self.assertEqual(
                    getattr(constants, name), self._dep_vars[f"{name}_{key_suffix}"]
                )

    def test_every_supported_minor_is_listed(self) -> None:
        for name in _PER_MINOR_CONSTANTS + [
            "DOCKER_LATEST_BUILDER_IMAGE_NAME_WITH_TAG"
        ]:
            for key_suffix in _SUPPORTED_KEY_SUFFIXES:
                with self.subTest(f"{name}_{key_suffix}"):
                    self.assertIn(f"{name}_{key_suffix}", self._dep_vars)


if __name__ == "__main__":
    absltest.main()
