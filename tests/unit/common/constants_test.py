import sys

from absl.testing import absltest

from gigl.common import constants
from gigl.common.constants import (
    PATH_BASE_IMAGES_VARIABLE_FILE,
    add_python_suffix,
    parse_makefile_vars,
)
from tests.test_assets.test_case import TestCase

_IMAGE_CONSTANTS = [
    "DOCKER_LATEST_BASE_CUDA_IMAGE_NAME_WITH_TAG",
    "DOCKER_LATEST_BASE_CPU_IMAGE_NAME_WITH_TAG",
    "DOCKER_LATEST_BASE_DATAFLOW_IMAGE_NAME_WITH_TAG",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_CUDA",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU",
    "DEFAULT_GIGL_RELEASE_SRC_IMAGE_DATAFLOW_CPU",
    "DEFAULT_GIGL_RELEASE_DEV_WORKBENCH_IMAGE",
]


class AddPythonSuffixTest(TestCase):
    def test_add_python_suffix(self) -> None:
        self.assertEqual(
            add_python_suffix("registry/src-cpu:0.3.1", (3, 13)),
            "registry/src-cpu:0.3.1-py313",
        )
        self.assertEqual(
            add_python_suffix(
                "gs://bucket/releases/pipelines/gigl-pipeline-0.3.1.yaml", (3, 12)
            ),
            "gs://bucket/releases/pipelines/gigl-pipeline-0.3.1-py312.yaml",
        )

    def test_constants_use_running_python(self) -> None:
        stems = parse_makefile_vars(PATH_BASE_IMAGES_VARIABLE_FILE)
        suffix = f"-py{sys.version_info.major}{sys.version_info.minor}"
        for name in _IMAGE_CONSTANTS:
            with self.subTest(name):
                self.assertEqual(getattr(constants, name), stems[name] + suffix)
        pipeline_stem = stems["DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH"]
        self.assertEqual(
            constants.DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH,
            pipeline_stem.removesuffix(".yaml") + suffix + ".yaml",
        )


if __name__ == "__main__":
    absltest.main()
