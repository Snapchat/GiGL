import os
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from unittest import mock
from uuid import uuid4

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from gigl.common.constants import GIGL_ROOT_DIR
from gigl.common.logger import Logger
from gigl.common.types.uri.uri_factory import UriFactory
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.file_loader import FileLoader

logger = Logger()


@dataclass(frozen=True)
class _NoteBookTestConfig:
    name: str
    notebook_path: str
    env_overrides: dict[str, str]
    timeout: int = 60 * 60


def _run_notebook(config: _NoteBookTestConfig) -> None:
    """Run a Jupyter notebook and return the executed notebook"""
    with mock.patch.dict(os.environ, config.env_overrides):
        with open(config.notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=config.timeout, kernel_name="python")
        ep.preprocess(
            nb,
            resources={"metadata": {"path": os.path.dirname(config.notebook_path)}},
        )


class TestExampleNotebooks(unittest.TestCase):
    def setUp(self):
        super().setUp()
        resource_config_uri = os.environ.get(
            "GIGL_TEST_DEFAULT_RESOURCE_CONFIG",
            str(GIGL_ROOT_DIR / "deployment/configs/e2e_glt_resource_config.yaml"),
        )
        logger.info(f"Using resource config URI: {resource_config_uri}")
        gcs_uri = (
            get_resource_config().temp_assets_regional_bucket_path
            / uuid4().hex
            / "resource_config.yaml"
        )
        logger.info(f"Using GCS URI: {gcs_uri}")
        fileloader = FileLoader()
        fileloader.load_file(UriFactory.create_uri(resource_config_uri), gcs_uri)
        self._notebooks = [
            _NoteBookTestConfig(
                name="cora",
                notebook_path=str(
                    GIGL_ROOT_DIR / "examples/link_prediction/cora.ipynb"
                ),
                env_overrides={
                    "GIGL_TEST_DEFAULT_RESOURCE_CONFIG": str(gcs_uri),
                },
            ),
            # _NoteBookTestConfig(
            #     name="toy_example",
            #     notebook_path=str(
            #         GIGL_ROOT_DIR
            #         / "examples/toy_visual_example/toy_example_walkthrough.ipynb"
            #     ),
            #     env_overrides={
            #         "GIGL_TEST_DEFAULT_RESOURCE_CONFIG": resource_config_uri,
            #     },
            # ),
        ]

    def test_notebooks(self):
        futures: dict[str, Future[None]] = {}
        with ThreadPoolExecutor() as executor:
            for notebook in self._notebooks:
                futures[notebook.name] = executor.submit(_run_notebook, notebook)

        for name, future in futures.items():
            with self.subTest(name=name):
                try:
                    future.result()
                except Exception as e:
                    self.fail(f"Notebook {name} failed with exception: {e}")

    # @mock.patch.dict(
    #     os.environ,
    #     {
    #         # "TASK_CONFIG_URI": str(
    #         #     GIGL_ROOT_DIR
    #         #     / "examples/link_prediction/configs/e2e_cora_udl_glt_task_config.yaml"
    #         # ),
    #         "GIGL_TEST_DEFAULT_RESOURCE_CONFIG": os.environ.get(
    #             "GIGL_TEST_DEFAULT_RESOURCE_CONFIG",
    #             str(GIGL_ROOT_DIR / "deployment/configs/e2e_glt_resource_config.yaml"),
    #         ),
    #     },
    # )
    # def test_cora_notebook(self):
    #     cora_notebook = str(GIGL_ROOT_DIR / "examples/link_prediction/cora.ipynb")
    #     with open(cora_notebook) as f:
    #         nb = nbformat.read(f, as_version=4)
    #     ep = ExecutePreprocessor(timeout=600, kernel_name="python")
    #     ep.preprocess(
    #         nb,
    #         resources={
    #             "metadata": {"path": str(GIGL_ROOT_DIR / "examples/link_prediction/")}
    #         },
    #     )


if __name__ == "__main__":
    unittest.main()
