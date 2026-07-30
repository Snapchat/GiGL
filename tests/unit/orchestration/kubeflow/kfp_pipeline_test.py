from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from absl.testing import absltest
from kfp.compiler import Compiler

from gigl.common.types.resource_config import CommonPipelineComponentConfigs
from gigl.orchestration.kubeflow.kfp_pipeline import generate_pipeline
from tests.test_assets.test_case import TestCase


class KfpPipelineTest(TestCase):
    def test_validator_outputs_are_pipeline_inputs_for_downstream_tasks(self) -> None:
        pipeline = generate_pipeline(
            CommonPipelineComponentConfigs(
                cuda_container_image="cuda-image",
                cpu_container_image="cpu-image",
                dataflow_container_image="dataflow-image",
            )
        )
        with TemporaryDirectory() as temp_directory:
            output_path = Path(temp_directory) / "pipeline.yaml"
            Compiler().compile(pipeline, str(output_path))
            pipeline_text = output_path.read_text()
            pipeline_spec = yaml.safe_load(pipeline_text)

        validator_task = pipeline_spec["components"]["comp-exit-handler-1"]["dag"][
            "tasks"
        ]["kfp-validation-check"]
        self.assertEqual(validator_task["cachingOptions"], {})
        self.assertNotIn("check-glt-backend", pipeline_text)
        self.assertIn(
            "outputParameterKey: resolved_task_config_uri\n"
            "                  producerTask: kfp-validation-check",
            pipeline_text,
        )
        self.assertIn(
            "outputParameterKey: resolved_resource_config_uri\n"
            "                  producerTask: kfp-validation-check",
            pipeline_text,
        )
        self.assertIn(
            "outputParameterKey: should_use_glt_backend",
            pipeline_text,
        )


if __name__ == "__main__":
    absltest.main()
