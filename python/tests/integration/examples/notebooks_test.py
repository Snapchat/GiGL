import os
import unittest
from unittest import mock

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from gigl.common.constants import GIGL_ROOT_DIR


class TestExampleNotebooks(unittest.TestCase):
    @mock.patch.dict(
        os.environ,
        {
            # "TASK_CONFIG_URI": str(
            #     GIGL_ROOT_DIR
            #     / "examples/distributed/configs/e2e_cora_udl_glt_task_config.yaml"
            # ),
            "RESOURCE_CONFIG_URI": str(
                GIGL_ROOT_DIR / "deployment/configs/e2e_glt_resource_config.yaml"
            ),
        },
    )
    def test_cora_notebook(self):
        cora_notebook = str(GIGL_ROOT_DIR / "examples/distributed/cora.ipynb")
        with open(cora_notebook) as f:
            nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(
            timeout=600,
        )
        ep.preprocess(
            nb,
            resources={
                "metadata": {"path": str(GIGL_ROOT_DIR / "examples/distributed/")}
            },
        )


if __name__ == "__main__":
    unittest.main()
