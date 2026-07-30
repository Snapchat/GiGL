from pathlib import Path

from absl.testing import absltest

from tests.test_assets.test_case import TestCase

_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPONENTS_ROOT = _REPO_ROOT / "gigl" / "orchestration" / "kubeflow" / "components"


def _read_component_spec(component_name: str) -> str:
    return (_COMPONENTS_ROOT / component_name / "component.yaml").read_text()


class ComponentSpecTest(TestCase):
    def test_config_validator_resolves_both_configs(self) -> None:
        component_spec = _read_component_spec("config_validator")

        self.assertIn("name: bootstrap_resource_config_hash", component_spec)
        self.assertIn("name: resolved_task_config_uri", component_spec)
        self.assertIn("name: resolved_resource_config_uri", component_spec)
        self.assertIn("name: should_use_glt_backend", component_spec)
        self.assertIn(
            "{outputPath: resolved_task_config_uri}",
            component_spec,
        )
        self.assertIn(
            "{outputPath: resolved_resource_config_uri}",
            component_spec,
        )
        self.assertIn(
            "{outputPath: should_use_glt_backend}",
            component_spec,
        )

    def test_data_preprocessor_accepts_source_image_uri_flags(self) -> None:
        component_spec = _read_component_spec("data_preprocessor")

        self.assertIn("name: cpu_docker_uri", component_spec)
        self.assertIn("name: cuda_docker_uri", component_spec)
        self.assertIn("--cpu_docker_uri, {inputValue: cpu_docker_uri}", component_spec)
        self.assertIn(
            "--cuda_docker_uri, {inputValue: cuda_docker_uri}", component_spec
        )

    def test_post_processor_accepts_source_image_uri_flags(self) -> None:
        component_spec = _read_component_spec("post_processor")

        self.assertIn("name: cpu_docker_uri", component_spec)
        self.assertIn("name: cuda_docker_uri", component_spec)
        self.assertIn("--cpu_docker_uri, {inputValue: cpu_docker_uri}", component_spec)
        self.assertIn(
            "--cuda_docker_uri, {inputValue: cuda_docker_uri}", component_spec
        )


if __name__ == "__main__":
    absltest.main()
