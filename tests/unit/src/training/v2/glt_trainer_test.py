"""Tests for ``gigl.src.training.v2.glt_trainer`` dispatch wiring.

Covers the CustomResourceConfig branch added alongside the existing
VertexAiResourceConfig / VertexAiGraphStoreConfig branches. VAI dispatch
is exercised by existing integration flows; these tests only assert that
CustomResourceConfig reaches ``launch_custom`` with the expected kwargs.
"""

from typing import Final
from unittest.mock import patch

from absl.testing import absltest

from gigl.common import Uri
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.training.v2.glt_trainer import GLTTrainer
from snapchat.research.gbml import gbml_config_pb2, gigl_resource_config_pb2
from tests.test_assets.test_case import TestCase

_PROCESS_COMMAND: Final[str] = "python -m gigl.src.training.v2.glt_trainer"


def _build_resource_config_with_custom_trainer() -> (
    gigl_resource_config_pb2.GiglResourceConfig
):
    shared = gigl_resource_config_pb2.SharedResourceConfig(
        resource_labels={
            "env": "test",
            "cost_resource_group_tag": "unittest_COMPONENT",
            "cost_resource_group": "gigl_test",
        },
        common_compute_config=(
            gigl_resource_config_pb2.SharedResourceConfig.CommonComputeConfig(
                project="test-project",
                region="us-central1",
                temp_assets_bucket="gs://test-temp-bucket",
                temp_regional_assets_bucket="gs://test-temp-regional-bucket",
                perm_assets_bucket="gs://test-perm-bucket",
                temp_assets_bq_dataset_name="test_temp_dataset",
                embedding_bq_dataset_name="test_embeddings_dataset",
                gcp_service_account_email=(
                    "test-sa@test-project.iam.gserviceaccount.com"
                ),
                dataflow_runner="DataflowRunner",
            )
        ),
    )
    resource_config = gigl_resource_config_pb2.GiglResourceConfig(
        shared_resource_config=shared,
    )
    resource_config.trainer_resource_config.custom_trainer_config.CopyFrom(
        gigl_resource_config_pb2.CustomResourceConfig(
            launcher_fn="my_project.launchers.ray.launch",
            launcher_args={"cluster": "dev", "num_workers": "4"},
        )
    )
    return resource_config


def _build_gbml_config_with_trainer_command() -> gbml_config_pb2.GbmlConfig:
    return gbml_config_pb2.GbmlConfig(
        trainer_config=gbml_config_pb2.GbmlConfig.TrainerConfig(
            command=_PROCESS_COMMAND,
            trainer_args={"lr": "0.01", "epochs": "5"},
        ),
    )


class TestGLTTrainerCustomDispatch(TestCase):
    """Asserts CustomResourceConfig routes to ``launch_custom``."""

    @patch("gigl.src.training.v2.glt_trainer.launch_custom")
    @patch(
        "gigl.src.training.v2.glt_trainer.GbmlConfigPbWrapper"
        ".get_gbml_config_pb_wrapper_from_uri"
    )
    @patch("gigl.src.training.v2.glt_trainer.get_resource_config")
    def test_custom_resource_config_dispatches_to_launch_custom(
        self,
        mock_get_resource_config,
        mock_get_gbml,
        mock_launch_custom,
    ):
        resource_config = _build_resource_config_with_custom_trainer()
        mock_get_resource_config.return_value = GiglResourceConfigWrapper(
            resource_config=resource_config
        )
        mock_get_gbml.return_value = GbmlConfigPbWrapper(
            gbml_config_pb=_build_gbml_config_with_trainer_command()
        )

        task_uri = Uri("gs://bucket/task.yaml")
        resource_uri = Uri("gs://bucket/resource.yaml")
        GLTTrainer().run(
            applied_task_identifier=AppliedTaskIdentifier("job-77"),
            task_config_uri=task_uri,
            resource_config_uri=resource_uri,
            cpu_docker_uri="gcr.io/p/cpu:tag",
            cuda_docker_uri="gcr.io/p/cuda:tag",
        )

        mock_launch_custom.assert_called_once()
        call_kwargs = mock_launch_custom.call_args.kwargs
        self.assertEqual(call_kwargs["component"], GiGLComponents.Trainer)
        self.assertEqual(call_kwargs["applied_task_identifier"], "job-77")
        self.assertEqual(call_kwargs["task_config_uri"], task_uri)
        self.assertEqual(call_kwargs["resource_config_uri"], resource_uri)
        self.assertEqual(call_kwargs["process_command"], _PROCESS_COMMAND)
        # trainer_args is a proto ScalarMap; compare by equality to plain dict.
        self.assertEqual(
            dict(call_kwargs["process_runtime_args"]),
            {"lr": "0.01", "epochs": "5"},
        )
        self.assertEqual(call_kwargs["cpu_docker_uri"], "gcr.io/p/cpu:tag")
        self.assertEqual(call_kwargs["cuda_docker_uri"], "gcr.io/p/cuda:tag")
        self.assertFalse(call_kwargs["is_dry_run"])

        # The forwarded CustomResourceConfig matches what we put in the
        # resource config.
        forwarded = call_kwargs["custom_resource_config"]
        self.assertEqual(forwarded.launcher_fn, "my_project.launchers.ray.launch")
        self.assertEqual(
            dict(forwarded.launcher_args),
            {"cluster": "dev", "num_workers": "4"},
        )


if __name__ == "__main__":
    absltest.main()
