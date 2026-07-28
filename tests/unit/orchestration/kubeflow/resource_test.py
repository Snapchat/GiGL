import os
from unittest.mock import MagicMock, call, patch

from absl.testing import absltest

from gigl.orchestration.kubeflow.utils.resource import (
    add_task_resource_requirements,
)
from tests.test_assets.test_case import TestCase


def _make_task() -> MagicMock:
    """A stand-in PipelineTask exposing ``container_spec`` and ``set_env_variable``."""
    return MagicMock()


def _make_configs() -> MagicMock:
    configs = MagicMock()
    configs.cpu_container_image = "gcr.io/example/cpu-image:tag"
    return configs


class AddTaskResourceRequirementsEnvTest(TestCase):
    """``add_task_resource_requirements`` forwards loader-selection env vars.

    The pipeline is compiled in the submitter's environment, so a selector set
    there (e.g. ``GIGL_COLLATE_IMPL``) must be copied onto the component task;
    otherwise it never reaches the remote component pod or its worker pool.
    """

    def test_forwards_both_when_set(self) -> None:
        task = _make_task()
        with patch.dict(
            os.environ,
            {"GIGL_COLLATE_IMPL": "vectorized", "GIGL_ABLP_LABEL_FORMAT": "edge_list"},
            clear=True,
        ):
            add_task_resource_requirements(task, _make_configs())
        task.set_env_variable.assert_has_calls(
            [
                call(name="GIGL_COLLATE_IMPL", value="vectorized"),
                call(name="GIGL_ABLP_LABEL_FORMAT", value="edge_list"),
            ],
            any_order=True,
        )
        self.assertEqual(task.set_env_variable.call_count, 2)

    def test_forwards_only_the_set_var(self) -> None:
        task = _make_task()
        with patch.dict(
            os.environ, {"GIGL_ABLP_LABEL_FORMAT": "edge_list"}, clear=True
        ):
            add_task_resource_requirements(task, _make_configs())
        task.set_env_variable.assert_called_once_with(
            name="GIGL_ABLP_LABEL_FORMAT", value="edge_list"
        )

    def test_no_forwarding_when_unset(self) -> None:
        task = _make_task()
        with patch.dict(os.environ, {}, clear=True):
            add_task_resource_requirements(task, _make_configs())
        task.set_env_variable.assert_not_called()

    def test_empty_value_treated_as_unset(self) -> None:
        task = _make_task()
        with patch.dict(
            os.environ,
            {"GIGL_COLLATE_IMPL": "", "GIGL_ABLP_LABEL_FORMAT": "dict"},
            clear=True,
        ):
            add_task_resource_requirements(task, _make_configs())
        task.set_env_variable.assert_called_once_with(
            name="GIGL_ABLP_LABEL_FORMAT", value="dict"
        )

    def test_image_still_set(self) -> None:
        task = _make_task()
        configs = _make_configs()
        with patch.dict(os.environ, {}, clear=True):
            add_task_resource_requirements(task, configs)
        self.assertEqual(task.container_spec.image, configs.cpu_container_image)


if __name__ == "__main__":
    absltest.main()
