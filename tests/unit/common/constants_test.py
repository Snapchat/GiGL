import sys

from absl.testing import absltest

from gigl.common.constants import (
    DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH,
    DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU,
    add_python_suffix,
)
from tests.test_assets.test_case import TestCase


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
        suffix = f"-py{sys.version_info.major}{sys.version_info.minor}"
        self.assertTrue(DEFAULT_GIGL_RELEASE_SRC_IMAGE_CPU.endswith(suffix))
        self.assertTrue(
            DEFAULT_GIGL_RELEASE_KFP_PIPELINE_PATH.endswith(f"{suffix}.yaml")
        )


if __name__ == "__main__":
    absltest.main()
