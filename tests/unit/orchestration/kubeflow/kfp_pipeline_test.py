from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from absl.testing import absltest
from kfp.compiler import Compiler

from gigl.common.types.resource_config import CommonPipelineComponentConfigs
from gigl.orchestration.kubeflow.kfp_pipeline import ExitTask, generate_pipeline
from tests.test_assets.test_case import TestCase

_COMMON_CONFIGS = CommonPipelineComponentConfigs(
    cuda_container_image="cuda-image",
    cpu_container_image="cpu-image",
    dataflow_container_image="dataflow-image",
)


def _compile(**generate_pipeline_kwargs) -> dict:
    """Compile the GiGL pipeline and return the pipeline spec as a dict."""
    pipeline = generate_pipeline(_COMMON_CONFIGS, **generate_pipeline_kwargs)
    with TemporaryDirectory() as temp_directory:
        output_path = Path(temp_directory) / "pipeline.yaml"
        Compiler().compile(pipeline, str(output_path))
        return yaml.safe_load(output_path.read_text())


def _exit_task(pipeline_spec: dict) -> dict:
    """Return the root DAG task that runs after the exit handler group completes."""
    root_tasks = pipeline_spec["root"]["dag"]["tasks"]
    return next(
        task
        for task in root_tasks.values()
        if task.get("triggerPolicy", {}).get("strategy")
        == "ALL_UPSTREAM_TASKS_COMPLETED"
    )


def _referenced_output_parameters(pipeline_spec: dict) -> set[tuple[str, str]]:
    """Return output parameters consumed by tasks in the compiled pipeline."""
    return {
        (
            parameter["taskOutputParameter"]["outputParameterKey"],
            parameter["taskOutputParameter"]["producerTask"],
        )
        for component in pipeline_spec["components"].values()
        for task in component.get("dag", {}).get("tasks", {}).values()
        for parameter in task.get("inputs", {}).get("parameters", {}).values()
        if "taskOutputParameter" in parameter
    }


class KfpPipelineTest(TestCase):
    def test_email_exit_task_is_the_default(self) -> None:
        exit_task = _exit_task(_compile())

        self.assertIn("notification-email", exit_task["componentRef"]["name"])

    def test_cleanup_exit_task_replaces_the_email_op(self) -> None:
        pipeline_spec = _compile(exit_task=ExitTask.CLEANUP)

        exit_task = _exit_task(pipeline_spec)
        component_name = exit_task["componentRef"]["name"]
        self.assertIn("cleanup-child-jobs-op", component_name)
        executor_name = component_name.replace("comp-", "exec-", 1)
        executor = pipeline_spec["deploymentSpec"]["executors"][executor_name]
        self.assertEqual(executor["container"]["image"], "cpu-image")
        self.assertEqual(
            set(exit_task["inputs"]["parameters"]),
            {"job_name", "resource_config_uri"},
        )

    def test_validator_outputs_are_pipeline_inputs_for_downstream_tasks(self) -> None:
        pipeline_spec = _compile()

        validator_task = pipeline_spec["components"]["comp-exit-handler-1"]["dag"][
            "tasks"
        ]["kfp-validation-check"]
        self.assertEqual(validator_task["cachingOptions"], {"enableCache": True})

        referenced_outputs = _referenced_output_parameters(pipeline_spec)
        for output_parameter_key in (
            "composed_task_config_snapshot_uri",
            "composed_resource_config_snapshot_uri",
            "should_use_glt_backend",
        ):
            self.assertIn(
                (output_parameter_key, "kfp-validation-check"), referenced_outputs
            )


if __name__ == "__main__":
    absltest.main()
