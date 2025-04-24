"""
This script is used to run a Kubeflow pipeline on VAI.
You have options to RUN a pipeline, COMPILE a pipeline, or RUN a pipeline without compiling it
i.e. you have a precompiled pipeline somewhere.

RUNNING A PIPELINE:
    python gigl.orchestration.kubeflow.runner --action=run  ...args
    The following arguments are required:
        --task_config_uri: GCS URI to template_or_frozen_config_uri.
        --resource_config_uri: GCS URI to resource_config_uri.
        --container_image_cuda: GiGL source code image compiled for use with cuda. See containers/Dockerfile.src
        --container_image_cpu: GiGL source code image compiled for use with cpu. See containers/Dockerfile.src
        --container_image_dataflow: GiGL source code image compiled for use with dataflow. See containers/Dockerfile.dataflow.src
    The folowing arguments are optional:
        --job_name: The name to give to the KFP job. Default is "gigl_run_at_<current_time>"
        --start_at: The component to start the pipeline at. Default is config_populator. See gigl.src.common.constants.components.GiGLComponents
        --stop_after: The component to stop the pipeline at. Default is None.
        --pipeline_tag: Optional tag, which is provided will be used to tag the pipeline description.
        --compiled_pipeline_path: The path to where to store the compiled pipeline to.
        --wait: Wait for the pipeline run to finish.
        --additional_job_args: Additional job arguments for the pipeline components, by component.
            The value has to be of form: "<gigl_component>.<arg_name>=<value>". Where <gigl_component> is one of the
            string representations of component specified in gigl.src.common.constants.components.GiGLComponents
            This argument can be repeated.
            Example:
            --additional_job_args=subgraph_sampler.additional_spark35_jar_file_uris='gs://path/to/jar'
            --additional_job_args=split_generator.some_other_arg='value'
            This passes additional_spark35_jar_file_uris="gs://path/to/jar" to subgraph_sampler at compile time and
            some_other_arg="value" to split_generator at compile time.

    You can alternatively run_no_compile if you have a precompiled pipeline somewhere.
    python gigl.orchestration.kubeflow.runner --action=run_no_compile ...args
    The following arguments are required:
        --task_config_uri
        --resource_config_uri
        --compiled_pipeline_path: The path to a pre-compiled pipeline; can be gcs URI (gs://...), or a local path
    The following arguments are optional:
        --job_name
        --start_at
        --stop_after
        --pipeline_tag
        --wait

COMPILING A PIPELINE:
    A strict subset of running a pipeline,
    python gigl.orchestration.kubeflow.runner --action=compile ...args
    The following arguments are required:
        --container_image_cuda
        --container_image_cpu
        --container_image_dataflow
    The following arguments are optional:
        --compiled_pipeline_path: The path to where to store the compiled pipeline to.
        --pipeline_tag: Optional tag, which is provided will be used to tag the pipeline description.
        --additional_job_args: Additional job arguments for the pipeline components, by component.
            The value has to be of form: "<gigl_component>.<arg_name>=<value>". Where <gigl_component> is one of the
            string representations of component specified in gigl.src.common.constants.components.GiGLComponents
            This argument can be repeated.
            Example:
            --additional_job_args=subgraph_sampler.additional_spark35_jar_file_uris='gs://path/to/jar'
            --additional_job_args=split_generator.some_other_arg='value'
            This passes additional_spark35_jar_file_uris="gs://path/to/jar" to subgraph_sampler at compile time and
            some_other_arg="value" to split_generator at compile time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from enum import Enum
from typing import List

from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.kfp_orchestrator import (
    DEFAULT_KFP_COMPILED_PIPELINE_DEST_PATH,
    KfpOrchestrator,
)
from gigl.orchestration.kubeflow.kfp_pipeline import SPECED_COMPONENTS
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.time import current_formatted_datetime

DEFAULT_JOB_NAME = f"gigl_run_at_{current_formatted_datetime()}"
DEFAULT_START_AT = GiGLComponents.ConfigPopulator.value


class Action(Enum):
    RUN = "run"
    COMPILE = "compile"
    RUN_NO_COMPILE = "run_no_compile"

    @staticmethod
    def from_string(s: str) -> Action:
        try:
            return Action(s)
        except KeyError:
            raise ValueError()


_REQUIRED_RUN_FLAGS = frozenset(
    [
        "task_config_uri",
        "resource_config_uri",
        "container_image_cuda",
        "container_image_cpu",
        "container_image_dataflow",
    ]
)
_REQUIRED_RUN_NO_COMPILE_FLAGS = frozenset(
    [
        "task_config_uri",
        "resource_config_uri",
        "compiled_pipeline_path",
    ]
)
_REQUIRED_COMPILE_FLAGS = frozenset(
    [
        "container_image_cuda",
        "container_image_cpu",
        "container_image_dataflow",
    ]
)

logger = Logger()


def _parse_additional_job_args(
    additional_job_args: List[str],
) -> dict[GiGLComponents, dict[str, str]]:
    """
    Parse the additional job arguments for the pipeline components, by component.
    Args:
        additional_job_args List[str]: Each element is of form: "<gigl_component>.<arg_name>=<value>"
            Where <gigl_component> is one of the string representations of component specified in
            gigl.src.common.constants.components.GiGLComponents
            Example:
            ["subgraph_sampler.additional_spark35_jar_file_uris=gs://path/to/jar", "split_generator.some_other_arg=value"].

    Returns dict[GiGLComponents, dict[str, str]]: The parsed additional job arguments.
            Example for the example above: {
                GiGLComponents.SubgraphSampler: {
                    "additional_spark35_jar_file_uris"="gs://path/to/jar",
                },
                GiGLComponents.SplitGenerator: {
                    "some_other_arg": "value",
                },
            }
    """
    result: dict[GiGLComponents, dict[str, str]] = defaultdict(dict)
    for job_arg in additional_job_args:
        component_dot_arg, value = job_arg.split("=", 1)
        component_str, arg = component_dot_arg.split(".", 1)  # Handle nested keys
        component = GiGLComponents(component_str)
        # Build the nested dictionary dynamically
        result[component][arg] = value

    logger.info(f"Parsed additional job args: {result}")
    return dict(result)  # Ensure the default dict is converted to a regular dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the KF pipeline for GNN preprocessing/training/inference"
    )
    parser.add_argument(
        "--container_image_cuda",
        help="The docker image name and tag to use for cuda pipeline components ",
    )
    parser.add_argument(
        "--container_image_cpu",
        help="The docker image name and tag to use for cpu pipeline components ",
    )
    parser.add_argument(
        "--container_image_dataflow",
        help="The docker image name and tag to use for the worker harness in dataflow ",
    )
    parser.add_argument(
        "--job_name",
        help="Runtime argument for running the pipeline. The name to give to the KFP job.",
        default=DEFAULT_JOB_NAME,
    )
    parser.add_argument(
        "--start_at",
        help="Runtime argument for running the pipeline. Specify the component where to start the pipeline.",
        choices=SPECED_COMPONENTS,
        default=DEFAULT_START_AT,
    )
    parser.add_argument(
        "--stop_after",
        help="Runtime argument for running the pipeline. Specify the component where to stop the pipeline.",
        choices=SPECED_COMPONENTS,
        default=None,
    )
    parser.add_argument(
        "--task_config_uri",
        help="Runtime argument for running the pipeline. GCS URI to template_or_frozen_config_uri.",
    )
    parser.add_argument(
        "--resource_config_uri",
        help="Runtine argument for resource and env specifications of each component",
    )
    parser.add_argument(
        "--action",
        type=Action.from_string,
        choices=list(Action),
        required=True,
    )
    parser.add_argument(
        "--wait",
        help="Wait for the pipeline run to finish",
        action="store_true",
    )
    parser.add_argument(
        "--pipeline_tag", "-t", help="Tag for the pipeline definition", default=None
    )
    parser.add_argument(
        "--compiled_pipeline_path",
        help="A custom URI that points to where you want the compiled pipeline is to be saved to."
        + "In the case you want to run an existing pipeline that you are not compiling, this is the path to the compiled pipeline.",
        default=DEFAULT_KFP_COMPILED_PIPELINE_DEST_PATH.uri,
    )
    parser.add_argument(
        "--additional_job_args",
        action="append",  # Allow multiple occurrences of this argument
        default=[],
        help="""Additional pipeline job arguments by component of form: "gigl_component.key=value,gigl_component.key_2=value_2"
        Example: --additional_job_args=subgraph_sampler.additional_spark35_jar_file_uris='gs://path/to/jar'
            --additional_job_args=split_generator.some_other_arg='value'
        This passes additional_spark35_jar_file_uris="gs://path/to/jar" to subgraph_sampler at compile time and
        some_other_arg="value" to split_generator at compile time.
        """,
    )

    args = parser.parse_args()
    logger.info(f"Beginning runner.py with args: {args}")

    parsed_additional_job_args = _parse_additional_job_args(args.additional_job_args)

    # Assert correctness of args
    required_flags: frozenset[str]
    if args.action == Action.RUN:
        required_flags = _REQUIRED_RUN_FLAGS
    elif args.action == Action.RUN_NO_COMPILE:
        required_flags = _REQUIRED_RUN_NO_COMPILE_FLAGS
    elif args.action == Action.COMPILE:
        required_flags = _REQUIRED_COMPILE_FLAGS

    missing_flags = []
    for flag in required_flags:
        if not hasattr(args, flag):
            missing_flags.append(flag)
    if missing_flags:
        raise ValueError(
            f"Missing the following flags for a {args.action} command: {missing_flags}. "
            + f"All required flags are: {list(required_flags)}"
        )

    compiled_pipeline_path = UriFactory.create_uri(args.compiled_pipeline_path)
    if args.action in (Action.RUN, Action.RUN_NO_COMPILE):
        orchestrator = KfpOrchestrator()

        task_config_uri = UriFactory.create_uri(args.task_config_uri)
        resource_config_uri = UriFactory.create_uri(args.resource_config_uri)
        applied_task_identifier = AppliedTaskIdentifier(args.job_name)

        if args.action == Action.RUN:
            path = orchestrator.compile(
                cuda_container_image=args.container_image_cuda,
                cpu_container_image=args.container_image_cpu,
                dataflow_container_image=args.container_image_dataflow,
                dst_compiled_pipeline_path=compiled_pipeline_path,
                additional_job_args=parsed_additional_job_args,
                tag=args.pipeline_tag,
            )
            assert (
                path == compiled_pipeline_path
            ), f"Compiled pipeline path {path} does not match provided path {compiled_pipeline_path}"

        run = orchestrator.run(
            applied_task_identifier=applied_task_identifier,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            start_at=args.start_at,
            stop_after=args.stop_after,
            compiled_pipeline_path=compiled_pipeline_path,
        )

        if args.wait:
            orchestrator.wait_for_completion(run=run)

    elif args.action == Action.COMPILE:
        pipeline_bundle_path = KfpOrchestrator.compile(
            cuda_container_image=args.container_image_cuda,
            cpu_container_image=args.container_image_cpu,
            dataflow_container_image=args.container_image_dataflow,
            dst_compiled_pipeline_path=compiled_pipeline_path,
            additional_job_args=parsed_additional_job_args,
            tag=args.pipeline_tag,
        )
        logger.info(
            f"Pipeline finished compiling, exported to: {pipeline_bundle_path.uri}"
        )
    else:
        raise ValueError(f"Unknown action: {args.action}")
