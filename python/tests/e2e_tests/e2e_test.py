"""
This script is used to run end-to-end (e2e) tests for the GiGL project.

Usage:
    python -m python.tests.e2e_tests.e2e_test --test_spec_uris <paths_to_test_specs.yaml> --container_image_cuda <cuda_image> --container_image_cpu <cpu_image> --container_image_dataflow <dataflow_image> [options]

Arguments:
    --test_spec_uris: URIs to YAML files containing test specifications. Supports both:
                     - Single test format: {test: {name, task_config_uri, ...}}
                     - Combined format: {tests: [{name, task_config_uri, ...}, ...]}
    --container_image_cuda: The CUDA container image to use.
    --container_image_cpu: The CPU container image to use.
    --container_image_dataflow: The Dataflow container image to use.
    --wait_for_all: If set, wait for all jobs to complete (overrides individual job settings).
    --pipeline_tag: Tag to apply to the pipeline definition.
    --additional_job_tag: Additional tag to append to job names for uniqueness.
    --test_names: Optional list of test names to run (supports partial matches). If not provided, runs all tests.

Examples:
    # Run all tests from combined config
    python -m python.tests.e2e_tests.e2e_test --test_spec_uris python/tests/e2e_tests/configs/e2e_tests.yaml --container_image_cuda my_cuda_image --container_image_cpu my_cpu_image --container_image_dataflow my_dataflow_image

    # Run specific tests by name
    python -m python.tests.e2e_tests.e2e_test --test_spec_uris python/tests/e2e_tests/configs/e2e_tests.yaml --container_image_cuda my_cuda_image --container_image_cpu my_cpu_image --container_image_dataflow my_dataflow_image --test_names cora_nalp dblp_nalp

    # Run with wait and custom tags
    python -m python.tests.e2e_tests.e2e_test --test_spec_uris python/tests/e2e_tests/configs/e2e_tests.yaml --container_image_cuda my_cuda_image --container_image_cpu my_cpu_image --container_image_dataflow my_dataflow_image --wait_for_all --additional_job_tag nightly
"""

from __future__ import annotations
import argparse
import concurrent.futures
import textwrap
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.kfp_orchestrator import KfpOrchestrator
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.file_loader import FileLoader
from scripts.build_and_push_docker_image import (
    build_and_push_cpu_image,
    build_and_push_cuda_image,
    build_and_push_dataflow_image,
)
from scripts.scala_packager import ScalaPackager, clean_scala_build_dirs

logger = Logger()

# Constants for job name generation
MAX_JOB_NAME_LENGTH = 51  # Kubernetes job name limit
UNIQUE_SUFFIX_LENGTH = 8  # Length of UUID suffix
MAX_BASE_NAME_LENGTH = MAX_JOB_NAME_LENGTH - UNIQUE_SUFFIX_LENGTH - 1  # -1 for underscore


@dataclass
class E2ETest:
    """Wrapper class storing information to run a GiGL e2e test.

    Thin wrapper around the the arguements of `KfpOrchestrator.compile` and `KfpOrchestrator.run`.

    Attributes:
        job_name: The name of the job to run.
        task_config_uri: The URI of the task configuration file.
        resource_config_uri: The URI of the resource configuration file.
        start_at: The component to start at in the pipeline.
        stop_after: The component to stop at in the pipeline.
        wait_for_completion: Whether to wait for the job to complete.
        requires_scala: Whether the job requires Scala. If no jobs to run require Scala, we skip the Scala packaging step.
    """

    job_name: AppliedTaskIdentifier
    task_config_uri: Uri
    resource_config_uri: Uri
    start_at: str = GiGLComponents.ConfigPopulator.value
    stop_after: str = GiGLComponents.PostProcessor.value
    wait_for_completion: bool = False
    requires_scala: bool = False

    @classmethod
    def from_dict(
        cls, json_dict: Dict[str, str], additional_job_tag: Optional[str] = None
    ) -> E2ETest:
        base_name = json_dict["name"]
        if additional_job_tag:
            base_name = f"{base_name}_{additional_job_tag}"

        # Generate a shorter, more readable unique suffix
        unique_suffix = str(uuid.uuid4()).replace("-", "")[:UNIQUE_SUFFIX_LENGTH]
        # Ensure job name fits within Kubernetes limits while preserving uniqueness
        job_name = f"{base_name[:MAX_BASE_NAME_LENGTH]}_{unique_suffix}"
        logger.info(f"For e2e test {json_dict['name']}, using job name: {job_name}")
        return cls(
            job_name=AppliedTaskIdentifier(job_name),
            task_config_uri=UriFactory.create_uri(json_dict["task_config_uri"]),
            resource_config_uri=UriFactory.create_uri(json_dict["resource_config_uri"]),
            start_at=json_dict.get("start_at", GiGLComponents.ConfigPopulator.value),
            stop_after=json_dict.get("stop_after", GiGLComponents.PostProcessor.value),
            wait_for_completion=json_dict.get("wait_for_completion", "False") == "True",
            requires_scala=json_dict.get("uses_in_memory_sampling", "False") == "True",
        )


def run_all_e2e_tests(
    jobs: List[E2ETest],
    container_image_cuda: str,
    container_image_cpu: str,
    container_image_dataflow: str,
    pipeline_tag: str,
) -> None:
    """Run all provided e2e tests.

    This function orchestrates the entire e2e testing process:
    1. Packages Scala components if needed
    2. Builds and pushes Docker images in parallel
    3. Compiles the pipeline once for all jobs
    4. Runs all jobs
    5. Waits for completion of jobs marked for waiting

    Args:
        jobs: List of E2ETest objects to run
        container_image_cuda: CUDA container image name
        container_image_cpu: CPU container image name
        container_image_dataflow: Dataflow container image name
        pipeline_tag: Tag to apply to the pipeline definition

    Raises:
        Exception: If Docker image compilation or job execution fails
    """
    if any(job.requires_scala for job in jobs):
        clean_scala_build_dirs()
        packager = ScalaPackager()
        packager.package_subgraph_sampler()
        packager.package_subgraph_sampler(use_spark35=True)
        packager.package_split_generator(use_spark35=True)
    else:
        logger.info("Skipping scala packaging.")

    image_builds = [
        ("CUDA", build_and_push_cuda_image, container_image_cuda),
        ("CPU", build_and_push_cpu_image, container_image_cpu),
        ("Dataflow", build_and_push_dataflow_image, container_image_dataflow),
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            (image_type, executor.submit(build_func, image_name))
            for image_type, build_func, image_name in image_builds
        ]

    failed_builds = []
    for image_type, future in futures:
        if future.exception():
            failed_builds.append((image_type, future.exception()))

    if failed_builds:
        logger.error("Failed to build the following Docker images:")
        for image_type, exception in failed_builds:
            logger.error(f"  {image_type}: {exception}")
        raise Exception(f"Could not compile {len(failed_builds)} Docker image(s).")

    jobs_by_name = {job.job_name: job for job in jobs}
    runs = {}
    orchestrator_client = KfpOrchestrator()

    # We can use the same pipeline definition for all jobs, so we compile it once.
    compiled_pipeline_path = orchestrator_client.compile(
        cpu_container_image=container_image_cpu,
        cuda_container_image=container_image_cuda,
        dataflow_container_image=container_image_dataflow,
        tag=pipeline_tag,
    )
    for job in jobs:
        logger.info(f"Compiling and running job: {job.job_name}")
        logger.info(
            textwrap.dedent(
                f"""To reproduce, run:
                                    python -m gigl.orchestration.kubeflow.runner \
                                    --container_image_cuda {container_image_cuda} \
                                    --container_image_cpu {container_image_cpu} \
                                    --container_image_dataflow {container_image_dataflow} \
                                    --job_name {job.job_name} \
                                    --task_config_uri {job.task_config_uri} \
                                    --resource_config_uri {job.resource_config_uri} \
                                    --start_at {job.start_at} \
                                    --stop_after {job.stop_after} \
                                    --pipeline_tag {pipeline_tag} \
                                    --action run
                                    """
            )
        )
        run = orchestrator_client.run(
            applied_task_identifier=job.job_name,
            task_config_uri=job.task_config_uri,
            resource_config_uri=job.resource_config_uri,
            start_at=job.start_at,
            stop_after=job.stop_after,
            compiled_pipeline_path=compiled_pipeline_path,
        )
        runs[job.job_name] = run

    jobs_to_wait_on = [job for job in jobs if job.wait_for_completion]
    runs_to_wait_on = [runs[job.job_name] for job in jobs_to_wait_on]

    if runs_to_wait_on:
        job_names = [job.job_name for job in jobs_to_wait_on]
        logger.info(f"Waiting for completion of {len(runs_to_wait_on)} job(s): {job_names}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_job = {
                executor.submit(orchestrator_client.wait_for_completion, run): job
                for job, run in zip(jobs_to_wait_on, runs_to_wait_on)
            }

        # Wait for all futures to complete
        concurrent.futures.wait(future_to_job.keys())

        failed_jobs = []
        for future, job in future_to_job.items():
            if future.exception():
                failed_jobs.append((job.job_name, future.exception()))

        if failed_jobs:
            logger.error(f"The following {len(failed_jobs)} job(s) failed:")
            for job_name, exception in failed_jobs:
                logger.error(f"  {job_name}: {exception}")
            raise Exception(f"{len(failed_jobs)} job(s) failed during execution.")

        logger.info(f"All {len(runs_to_wait_on)} job(s) completed successfully: {job_names}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_spec_uris",
        type=str,
        nargs="+",
        required=True,
        help="The test specs to run.",
    )
    parser.add_argument(
        "--container_image_cuda", type=str, required=True, help="The cuda container image to be built."
    )
    parser.add_argument(
        "--container_image_cpu", type=str, required=True, help="The cpu container image to be built."
    )
    parser.add_argument(
        "--container_image_dataflow",
        type=str,
        required=True,
        help="The dataflow container image to be built.",
    )
    parser.add_argument(
        "--wait_for_all",
        default=False,
        action="store_true",
        help="Wait for all jobs, if set will override job-level configuration.",
    )
    parser.add_argument(
        "--pipeline_tag",
        type=str,
        default="",
        help="Tag to apply to the pipeline definition (pipeline.yaml).",
    )
    parser.add_argument(
        "--additional_job_tag",
        type=str,
        default="",
        help="Additional tag to add to the job name.",
    )
    parser.add_argument(
        "--test_names",
        type=str,
        nargs="*",
        help="Optional list of specific test names to run. If not provided, all tests in the spec files will be run.",
    )

    args = parser.parse_args()
    logger.info(f"Starting e2e tests with args: {args}")
    jobs = []
    loader = FileLoader()
    for test_spec_uri in args.test_spec_uris:
        with loader.load_to_temp_file(UriFactory.create_uri(test_spec_uri)) as tf:
            with open(tf.name, "r") as f:
                test_spec_data = yaml.safe_load(f)

            # Handle both old format (single test) and new format (multiple tests)
            if "test" in test_spec_data:
                # Old format: single test specification
                test_job = E2ETest.from_dict(
                    test_spec_data["test"], additional_job_tag=args.additional_job_tag
                )
                if args.wait_for_all:
                    test_job.wait_for_completion = True
                jobs.append(test_job)
            elif "tests" in test_spec_data:
                # New format: multiple test specifications
                for test_config in test_spec_data["tests"]:
                    test_job = E2ETest.from_dict(
                        test_config, additional_job_tag=args.additional_job_tag
                    )
                    if args.wait_for_all:
                        test_job.wait_for_completion = True
                    jobs.append(test_job)
            else:
                logger.error(f"Invalid test specification format in {test_spec_uri}. Expected 'test' or 'tests' key.")
                exit(1)

    # Filter tests by name if specified
    if args.test_names:
        original_jobs = jobs[:]  # Keep a copy for error reporting
        original_count = len(jobs)
        test_name_set = set(args.test_names)

        # Match test names (partial matches allowed)
        jobs = [job for job in jobs if any(test_name in str(job.job_name) for test_name in test_name_set)]

        if not jobs:
            available_names = [str(job.job_name) for job in original_jobs]
            logger.error(f"No tests found matching the specified names: {args.test_names}")
            logger.info(f"Available tests are: {available_names}")
            exit(1)

        filtered_names = [str(job.job_name) for job in jobs]
        logger.info(f"Filtered to {len(jobs)} test(s) from {original_count} total: {filtered_names}")

    run_all_e2e_tests(
        jobs=jobs,
        container_image_cuda=args.container_image_cuda,
        container_image_cpu=args.container_image_cpu,
        container_image_dataflow=args.container_image_dataflow,
        pipeline_tag=args.pipeline_tag,
    )
