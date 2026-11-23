from gigl.common import LocalUri
from gigl.common.utils.yaml_loader import load_resolved_yaml
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.spark_job_manager import (
    build_split_generator_cluster_name,
    build_subgraph_sampler_cluster_name,
)
from gigl.src.validation_check.libs.name_checks import (
    check_if_kfp_pipeline_job_name_valid,
)
from tests.e2e_tests.e2e_test import E2ETestsSpec, build_e2e_job_name
from tests.test_assets.test_case import TestCase

_E2E_TESTS_YAML_PATH = "tests/e2e_tests/e2e_tests.yaml"


def _load_job_names_and_task_configs() -> dict[str, str]:
    """Returns the full e2e job name for each test, mapped to the test's task config URI."""
    spec = load_resolved_yaml(LocalUri(_E2E_TESTS_YAML_PATH), E2ETestsSpec)
    return {
        build_e2e_job_name(name, test.name_suffix): test.task_config_uri
        for name, test in spec.tests.items()
    }


class E2EJobNamesTest(TestCase):
    def test_job_names_differ_between_calls(self) -> None:
        self.assertNotEqual(
            build_e2e_job_name("my_test", "_on_20260101_000000"),
            build_e2e_job_name("my_test", "_on_20260101_000000"),
        )

    def test_all_job_names_are_valid_kfp_job_names(self) -> None:
        job_names = _load_job_names_and_task_configs()
        self.assertGreater(len(job_names), 0)
        for job_name in job_names:
            with self.subTest(job_name=job_name):
                check_if_kfp_pipeline_job_name_valid(job_name)

    def test_spark_cluster_names_are_not_truncated(self) -> None:
        # Tests that do not use the GLT backend run the Spark subgraph sampler and split generator.
        tabularized_job_names = []
        for job_name, task_config_uri in _load_job_names_and_task_configs().items():
            wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=LocalUri(task_config_uri)
            )
            if not wrapper.should_use_glt_backend:
                tabularized_job_names.append(job_name)
        self.assertGreater(len(tabularized_job_names), 0)

        for job_name in tabularized_job_names:
            dashed_job_name = job_name.replace("_", "-")
            with self.subTest(job_name=job_name):
                self.assertEqual(
                    build_subgraph_sampler_cluster_name(job_name),
                    f"sgs-{dashed_job_name}",
                )
                self.assertEqual(
                    build_split_generator_cluster_name(job_name),
                    f"split-{dashed_job_name}",
                )
