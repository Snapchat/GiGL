"""
This script is used to run end-to-end (e2e) tests for the GiGL project.

Usage:
    python -m python.tests.e2e_tests.e2e_test \
        --compiled_pipeline_path <path_to_compiled_pipeline.yaml> \
        --test_spec_uri <path_to_test_specs.yaml>

Arguments:
    The following arguments are required:
        --compiled_pipeline_path: The KFP pipeline definition to use.
        --test_spec_uri: URI to a YAML file containing test specifications.
    The following arguments are optional:
        --test_names: [Optional] Test name to run from the test spec file. The value can be repeated for running multiple tests.
            If not provided, all tests in the spec files will be run.
            Example: --test_names=cora_glt_udl_test_on --test_names=cora_nalp_test_on

The test_spec_uri is a yaml file of the following format:
    ```yaml
    tests:
      "name_of_the_test":
        name_suffix: "_on_${now:}"
        task_config_uri: "path/to/task_config.yaml"
        resource_config_uri: "path/to/resource_config.yaml"
        uses_in_memory_sampling: "True"
        start_at: "config_populator" # Optional, default is config_populator.
        stop_after: "post_processor" # Optional, defaults to None
        wait_for_completion: "True" # Optional, default is True.
        run_labels: # Labels to associate with the pipeline run. Default is:
          gigl_commit: "${git_hash:}" # Resolves to current git hash for the active working directory
          gigl_version: "${__version__}" # Resolves to current GiGL version
    ```

Examples:
    # Run all tests from combined config
    python -m python.tests.e2e_tests.e2e_test \
        --compiled_pipeline_path=/tmp/gigl/my_pipeline.yaml \
        --test_spec_uri=python/tests/e2e_tests/configs/e2e_tests.yaml \

    # Run specific tests by name
    python -m python.tests.e2e_tests.e2e_test \
        --compiled_pipeline_path=/tmp/gigl/my_pipeline.yaml \
        --test_spec_uri=python/tests/e2e_tests/configs/e2e_tests.yaml \
        --test_names=cora_nalp --test_names=dblp_nalp
"""

import argparse
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from google.cloud.aiplatform import PipelineJob

from gigl import __version__
from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.yaml_loader import load_resolved_yaml
from gigl.orchestration.kubeflow.kfp_orchestrator import KfpOrchestrator
from gigl.src.common.types import AppliedTaskIdentifier

logger = Logger()


@dataclass
class E2ETest:
    f"""
    Configuration for a single E2E test specification.

    Args:
        task_config_uri: The URI of the task config to use.
        resource_config_uri: The URI of the resource config to use.
        name_suffix: The suffix to add to the job name; job name will be of form: <test_name><name_suffix>,
            where <test_name> is the key in :attr:`E2ETestsSpec.tests`.
        start_at: Specify the component where to start the pipeline. Choices are defined in:
            :attr:`gigl.src.common.constants.components.GiGLComponents`.
        stop_after: Specify the component where to stop the pipeline. Choices are defined in:
            :attr:`gigl.src.common.constants.components.GiGLComponents`.
        wait_for_completion: Whether to wait for the pipeline run to finish or not.
        run_labels: Labels to associate with the pipeline run.
    """

    task_config_uri: str
    resource_config_uri: str
    name_suffix: str = (
        "_on_${now:}"  # Makes use of gigl.common.omegaconf_resolvers#now_resolver()
    )
    start_at: str = "config_populator"
    stop_after: Optional[str] = None
    wait_for_completion: bool = True
    run_labels: dict[str, str] = field(
        default_factory=lambda: {
            "gigl_commit": "${git_hash:}",
            "gigl_version": f"{__version__.replace('.', '_')}",
        }
    )


@dataclass
class E2ETestsSpec:
    """Root configuration containing multiple E2E test specifications."""

    tests: dict[str, E2ETest] = field(default_factory=dict)


def run_all_e2e_tests(
    tests: dict[str, E2ETest],
    compiled_pipeline_path: Uri,
) -> None:
    orchestrator = KfpOrchestrator()
    runs_to_wait_on: list[PipelineJob] = []
    for job_name, job in tests.items():
        full_job_name = f"{job_name}{job.name_suffix}"
        logger.info(f"Compiling and running job: {full_job_name}")
        logger.info(
            textwrap.dedent(
                f"""To reproduce, run:
            python -m gigl.orchestration.kubeflow.runner \
                --task_config_uri={job.task_config_uri} \
                --resource_config_uri={job.resource_config_uri} \
                --compiled_pipeline_path={compiled_pipeline_path} \
                --job_name={full_job_name} \
                --start_at={job.start_at} \
                --stop_after={job.stop_after} \
                --run_labels={job.run_labels} \
                {"--wait" if job.wait_for_completion else ""} \
                --action run
            """
            )
        )

        applied_task_identifier = AppliedTaskIdentifier(full_job_name)
        task_config_uri = UriFactory.create_uri(job.task_config_uri)
        resource_config_uri = UriFactory.create_uri(job.resource_config_uri)

        pipeline_job: PipelineJob = orchestrator.run(
            applied_task_identifier=applied_task_identifier,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            start_at=job.start_at,
            stop_after=job.stop_after,
            compiled_pipeline_path=compiled_pipeline_path,
            labels=job.run_labels,
        )
        if job.wait_for_completion:
            runs_to_wait_on.append(pipeline_job)

    if runs_to_wait_on:
        orchestrator.wait_for_completion(runs_to_wait_on)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_spec_uri",
        type=str,
        required=True,
        help="The test specs to run.",
    )
    parser.add_argument(
        "--compiled_pipeline_path",
        type=str,
        required=True,
        help="The compiled pipeline definition to use.",
    )
    parser.add_argument(
        "--test_names",
        type=str,
        default=[],
        action="append",
        help="""Optional names of specific tests to run. If not provided, all tests in the spec files will be run.
        Example: --test_names=cora_glt_udl_test --test_names=cora_nalp_test""",
    )

    args = parser.parse_args()
    logger.info(f"Starting e2e tests with args: {args}")

    test_spec_uri = UriFactory.create_uri(args.test_spec_uri)
    logger.info(f"Will load test spec from: {test_spec_uri}")
    logger.info(f"Will load compiled pipeline from: {test_spec_uri}")
    test_spec: E2ETestsSpec = load_resolved_yaml(test_spec_uri, E2ETestsSpec)
    available_tests = test_spec.tests.keys()
    filtered_tests: dict[str, E2ETest] = {}
    if args.test_names:
        for test_name in args.test_names:
            if test_name not in available_tests:
                raise ValueError(f"Test {test_name} not found in test spec.")
            filtered_tests[test_name] = test_spec.tests[test_name]
    else:
        filtered_tests = test_spec.tests

    compiled_pipeline_path = UriFactory.create_uri(args.compiled_pipeline_path)
    run_all_e2e_tests(
        tests=filtered_tests,
        compiled_pipeline_path=compiled_pipeline_path,
    )
