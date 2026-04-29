import tempfile
from pathlib import Path
from unittest.mock import ANY, patch

import yaml
from absl.testing import absltest

from gigl.common import GcsUri, LocalUri
from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.kfp_orchestrator import KfpOrchestrator
from tests.test_assets.test_case import TestCase

logger = Logger()


class KfpOrchestratorTest(TestCase):
    @patch("gigl.orchestration.kubeflow.kfp_orchestrator.FileLoader")
    def test_compile_uploads_compiled_yaml(self, MockFileLoader):
        mock_file_loader = MockFileLoader.return_value
        mock_file_loader.load_file.return_value = None

        dst_compiled_pipeline_path = GcsUri(
            "gs://SOME NON EXISTING BUCKET/ NON EXISTING FILE"
        )
        KfpOrchestrator.compile(
            cuda_container_image="SOME NONEXISTENT IMAGE 1",
            cpu_container_image="SOME NONEXISTENT IMAGE 2",
            dataflow_container_image="SOME NONEXISTENT IMAGE 3",
            dst_compiled_pipeline_path=dst_compiled_pipeline_path,
        )
        mock_file_loader.load_file.assert_called_once_with(
            file_uri_src=ANY, file_uri_dst=dst_compiled_pipeline_path
        )

    def test_compile_bakes_env_vars_into_every_gigl_owned_executor(self):
        """env_vars passed to compile() should appear on every GiGL-owned executor's container env.

        The managed VertexNotificationEmailOp exit handler is the documented
        carve-out and must not receive the env vars.
        """
        env_vars = {
            "FOO": "bar",
            "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            dst = LocalUri(str(Path(tmp_dir) / "pipeline.yaml"))
            KfpOrchestrator.compile(
                cuda_container_image="SOME NONEXISTENT IMAGE 1",
                cpu_container_image="SOME NONEXISTENT IMAGE 2",
                dataflow_container_image="SOME NONEXISTENT IMAGE 3",
                dst_compiled_pipeline_path=dst,
                env_vars=env_vars,
            )

            with open(dst.uri, "r") as f:
                compiled = yaml.safe_load(f)

        executors = compiled["deploymentSpec"]["executors"]
        self.assertGreater(len(executors), 0, "Expected at least one executor in IR.")

        gigl_owned_with_env: list[str] = []
        notification_executors_without_env: list[str] = []
        for executor_id, executor_spec in executors.items():
            container = executor_spec.get("container", {})
            env_list = container.get("env", [])
            env_dict = {entry["name"]: entry["value"] for entry in env_list}
            is_notification = "notification-email" in executor_id.lower()
            if is_notification:
                # The managed notification op must not receive our env vars.
                for name in env_vars:
                    self.assertNotIn(
                        name,
                        env_dict,
                        f"Env var {name} unexpectedly applied to managed "
                        f"notification executor {executor_id}.",
                    )
                notification_executors_without_env.append(executor_id)
            else:
                for name, value in env_vars.items():
                    self.assertEqual(
                        env_dict.get(name),
                        value,
                        f"Executor {executor_id} missing env var {name}={value}; "
                        f"actual env: {env_dict}.",
                    )
                gigl_owned_with_env.append(executor_id)

        self.assertGreater(
            len(gigl_owned_with_env),
            0,
            "Expected at least one GiGL-owned executor to receive env vars.",
        )

    def test_compile_without_env_vars_does_not_inject_env(self):
        """When env_vars is omitted, no GiGL-owned executor should pick up phantom env entries from this code path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dst = LocalUri(str(Path(tmp_dir) / "pipeline.yaml"))
            KfpOrchestrator.compile(
                cuda_container_image="SOME NONEXISTENT IMAGE 1",
                cpu_container_image="SOME NONEXISTENT IMAGE 2",
                dataflow_container_image="SOME NONEXISTENT IMAGE 3",
                dst_compiled_pipeline_path=dst,
            )

            with open(dst.uri, "r") as f:
                compiled = yaml.safe_load(f)

        # Only assert the default (unset) case adds no FOO key — we don't make
        # claims about other env entries that KFP itself may inject.
        for executor_spec in compiled["deploymentSpec"]["executors"].values():
            env_list = executor_spec.get("container", {}).get("env", [])
            env_names = {entry["name"] for entry in env_list}
            self.assertNotIn("FOO", env_names)


if __name__ == "__main__":
    absltest.main()
