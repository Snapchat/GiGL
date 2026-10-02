import argparse
from pathlib import Path
from typing import Optional

from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.proto_utils import ProtoUtils, proto_to_yaml
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.common.utils.gigl_runtime import initialize_gigl_runtime
from gigl.src.validation_check.libs.frozen_config_path_checks import (
    assert_preprocessed_metadata_exists,
    assert_split_generator_output_exists,
    assert_subgraph_sampler_output_exists,
    assert_trained_model_exists,
)
from gigl.src.validation_check.libs.gbml_and_resource_config_compatibility_checks import (
    check_inferencer_graph_store_compatibility,
    check_trainer_graph_store_compatibility,
)
from gigl.src.validation_check.libs.name_checks import (
    check_if_kfp_pipeline_job_name_valid,
)
from gigl.src.validation_check.libs.resource_config_checks import (
    check_if_inferencer_resource_config_valid,
    check_if_preprocessor_resource_config_valid,
    check_if_shared_resource_config_valid,
    check_if_split_generator_resource_config_valid,
    check_if_subgraph_sampler_resource_config_valid,
    check_if_trainer_resource_config_valid,
)
from gigl.src.validation_check.libs.template_config_checks import (
    check_if_data_preprocessor_config_cls_valid,
    check_if_graph_metadata_valid,
    check_if_inferencer_cls_valid,
    check_if_post_processor_cls_valid,
    check_if_preprocessed_metadata_valid,
    check_if_split_generator_config_valid,
    check_if_subgraph_sampler_config_valid,
    check_if_task_metadata_valid,
    check_if_trainer_cls_valid,
    check_pipeline_has_valid_start_and_stop_flags,
)
from snapchat.research.gbml import gbml_config_pb2
from snapchat.research.gbml.gigl_resource_config_pb2 import GiglResourceConfig

START_STOP_COMPONENT_TO_CLS_CHECKS_MAP = {
    # TODO: (svij-sc) Add checks as needed, otherwise we default to below anyways
    (GiGLComponents.SubgraphSampler.value, GiGLComponents.SubgraphSampler.value): [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_subgraph_sampler_config_valid,
    ],
}

START_COMPONENT_TO_CLS_CHECKS_MAP = {
    GiGLComponents.ConfigPopulator.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_data_preprocessor_config_cls_valid,
        check_if_subgraph_sampler_config_valid,
        check_if_split_generator_config_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.DataPreprocessor.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_data_preprocessor_config_cls_valid,
        check_if_subgraph_sampler_config_valid,
        check_if_split_generator_config_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.SubgraphSampler.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_preprocessed_metadata_valid,
        check_if_subgraph_sampler_config_valid,
        check_if_split_generator_config_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.SplitGenerator.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_preprocessed_metadata_valid,
        check_if_split_generator_config_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.Trainer.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_preprocessed_metadata_valid,
        check_if_trainer_cls_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.Inferencer.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_preprocessed_metadata_valid,
        check_if_inferencer_cls_valid,
        check_if_post_processor_cls_valid,
    ],
    GiGLComponents.PostProcessor.value: [
        check_if_graph_metadata_valid,
        check_if_task_metadata_valid,
        check_if_post_processor_cls_valid,
    ],
}

START_COMPONENT_TO_ASSET_CHECKS_MAP = {
    GiGLComponents.SubgraphSampler.value: [
        assert_preprocessed_metadata_exists,
    ],
    GiGLComponents.SplitGenerator.value: [
        assert_preprocessed_metadata_exists,
        assert_subgraph_sampler_output_exists,
    ],
    GiGLComponents.Trainer.value: [
        assert_preprocessed_metadata_exists,
        assert_subgraph_sampler_output_exists,
        assert_split_generator_output_exists,
    ],
    GiGLComponents.Inferencer.value: [
        assert_preprocessed_metadata_exists,
        assert_subgraph_sampler_output_exists,
        assert_trained_model_exists,
    ],
}

START_STOP_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP = {
    (GiGLComponents.SubgraphSampler.value, GiGLComponents.SubgraphSampler.value): [
        check_if_shared_resource_config_valid,
        check_if_subgraph_sampler_resource_config_valid,
    ],
}

START_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP = {
    GiGLComponents.ConfigPopulator.value: [
        check_if_shared_resource_config_valid,
        check_if_preprocessor_resource_config_valid,
        check_if_subgraph_sampler_resource_config_valid,
        check_if_split_generator_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.DataPreprocessor.value: [
        check_if_shared_resource_config_valid,
        check_if_preprocessor_resource_config_valid,
        check_if_subgraph_sampler_resource_config_valid,
        check_if_split_generator_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.SubgraphSampler.value: [
        check_if_shared_resource_config_valid,
        check_if_subgraph_sampler_resource_config_valid,
        check_if_split_generator_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.SplitGenerator.value: [
        check_if_shared_resource_config_valid,
        check_if_split_generator_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.Trainer.value: [
        check_if_shared_resource_config_valid,
        check_if_trainer_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.Inferencer.value: [
        check_if_shared_resource_config_valid,
        check_if_inferencer_resource_config_valid,
    ],
    GiGLComponents.PostProcessor.value: [
        check_if_shared_resource_config_valid,
    ],
}

# Resource config checks to skip when using live subgraph sampling backend
RESOURCE_CONFIG_CHECKS_TO_SKIP_WITH_LIVE_SGS_BACKEND = [
    check_if_subgraph_sampler_resource_config_valid,
    check_if_split_generator_resource_config_valid,
]

logger = Logger()

# Map of start components to graph store compatibility checks to run
# Only run trainer checks when starting at or before Trainer
# Only run inferencer checks when starting at or before Inferencer
START_COMPONENT_TO_GRAPH_STORE_COMPATIBILITY_CHECKS = {
    GiGLComponents.ConfigPopulator.value: [
        check_trainer_graph_store_compatibility,
        check_inferencer_graph_store_compatibility,
    ],
    GiGLComponents.DataPreprocessor.value: [
        check_trainer_graph_store_compatibility,
        check_inferencer_graph_store_compatibility,
    ],
    GiGLComponents.SubgraphSampler.value: [
        check_trainer_graph_store_compatibility,
        check_inferencer_graph_store_compatibility,
    ],
    GiGLComponents.SplitGenerator.value: [
        check_trainer_graph_store_compatibility,
        check_inferencer_graph_store_compatibility,
    ],
    GiGLComponents.Trainer.value: [
        check_trainer_graph_store_compatibility,
        check_inferencer_graph_store_compatibility,
    ],
    GiGLComponents.Inferencer.value: [
        check_inferencer_graph_store_compatibility,
    ],
    # PostProcessor doesn't need graph store compatibility checks
}

# Map of (start, stop) component tuples to graph store compatibility checks

STOP_COMPONENT_TO_GRAPH_STORE_COMPATIBILITY_CHECKS_TO_SKIP = {
    GiGLComponents.Trainer.value: [
        check_inferencer_graph_store_compatibility,
    ],
}


def _run_gbml_and_resource_config_compatibility_checks(
    start_at: str,
    stop_after: Optional[str],
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    resource_config_wrapper: GiglResourceConfigWrapper,
) -> None:
    """
    Run compatibility checks between GbmlConfig and GiglResourceConfig.

    These checks verify that graph store mode configurations are consistent
    across both the template config (GbmlConfig) and resource config (GiglResourceConfig).

    Args:
        start_at: The component to start at.
        stop_after: Optional component to stop after.
        gbml_config_pb_wrapper: The GbmlConfig wrapper (template config).
        resource_config_wrapper: The GiglResourceConfig wrapper (resource config).
    """
    # Get the appropriate compatibility checks based on start/stop components
    compatibility_checks = set(
        START_COMPONENT_TO_GRAPH_STORE_COMPATIBILITY_CHECKS.get(start_at, [])
    )
    if stop_after in STOP_COMPONENT_TO_GRAPH_STORE_COMPATIBILITY_CHECKS_TO_SKIP:
        for skipped_check in STOP_COMPONENT_TO_GRAPH_STORE_COMPATIBILITY_CHECKS_TO_SKIP[
            stop_after
        ]:
            compatibility_checks.discard(skipped_check)

    for check in compatibility_checks:
        check(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            resource_config_wrapper=resource_config_wrapper,
        )


def _validate_resolved_configs(
    job_name: str,
    start_at: str,
    task_config: gbml_config_pb2.GbmlConfig,
    resource_config: GiglResourceConfig,
    stop_after: Optional[str] = None,
) -> bool:
    # check if job_name is valid
    check_if_kfp_pipeline_job_name_valid(job_name=job_name)
    gbml_config_pb_wrapper = GbmlConfigPbWrapper(task_config)

    gbml_config_pb: gbml_config_pb2.GbmlConfig = gbml_config_pb_wrapper.gbml_config_pb

    should_use_live_sgs_backend = gbml_config_pb_wrapper.should_use_glt_backend

    resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
    resource_config_pb: GiglResourceConfig = resource_config_wrapper.resource_config
    # check if start_at and stop_after aligns with live subgraph sampling backend use
    check_pipeline_has_valid_start_and_stop_flags(
        start_at=start_at,
        stop_after=stop_after,
        gbml_config_wrapper=gbml_config_pb_wrapper,
    )
    # check user defined classes and their runtime args

    if (
        stop_after is not None
        and (start_at, stop_after) in START_STOP_COMPONENT_TO_CLS_CHECKS_MAP
    ):
        cls_checks = START_STOP_COMPONENT_TO_CLS_CHECKS_MAP[(start_at, stop_after)]
    else:
        cls_checks = START_COMPONENT_TO_CLS_CHECKS_MAP.get(start_at, [])
    for cls_check in cls_checks:
        cls_check(gbml_config_pb=gbml_config_pb)

    # check the existence of needed assets
    for asset_check in START_COMPONENT_TO_ASSET_CHECKS_MAP.get(start_at, []):
        asset_check(gbml_config_pb=gbml_config_pb)
    # check if user-provided resource config is valid

    # Skip SGS and split resource config checks when using live subgraph sampling backend
    resource_config_checks_to_skip = (
        RESOURCE_CONFIG_CHECKS_TO_SKIP_WITH_LIVE_SGS_BACKEND
        if should_use_live_sgs_backend
        else []
    )

    if (
        stop_after is not None
        and (start_at, stop_after) in START_STOP_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP
    ):
        resource_config_checks = START_STOP_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP[
            (start_at, stop_after)
        ]
    else:
        resource_config_checks = START_COMPONENT_TO_RESOURCE_CONFIG_CHECKS_MAP.get(
            start_at, []
        )

    for resource_config_check in resource_config_checks:
        if resource_config_check not in resource_config_checks_to_skip:
            resource_config_check(resource_config_pb=resource_config_pb)
        else:
            logger.info(
                f"Skipping resource config check {resource_config_check.__name__} because we are using live subgraph sampling backend."
            )

    # check compatibility between template config and resource config for graph store mode
    # These checks ensure that if graph store mode is enabled in one config, it's also enabled in the other
    _run_gbml_and_resource_config_compatibility_checks(
        start_at=start_at,
        stop_after=stop_after,
        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        resource_config_wrapper=resource_config_wrapper,
    )

    # check if trained model file exist when skipping training
    if gbml_config_pb.shared_config.should_skip_training == True:
        assert_trained_model_exists(gbml_config_pb=gbml_config_pb)

    logger.info("[✅ SUCCESS] All checks passed successfully.")
    return should_use_live_sgs_backend


def resolve_configs(
    source_task_config_uri: Uri,
    source_resource_config_uri: Uri,
) -> tuple[gbml_config_pb2.GbmlConfig, GiglResourceConfig]:
    """Resolve task and resource configs into self-contained protobufs."""
    proto_utils = ProtoUtils()
    task_config = proto_utils.read_proto_from_yaml(
        uri=source_task_config_uri,
        proto_cls=gbml_config_pb2.GbmlConfig,
    )
    resource_config = proto_utils.read_proto_from_yaml(
        uri=source_resource_config_uri,
        proto_cls=GiglResourceConfig,
    )
    if resource_config.WhichOneof("shared_resource") == "shared_resource_config_uri":
        # Inline legacy external shared resources so downstream components
        # only need the published snapshot.
        resource_config.shared_resource_config.CopyFrom(
            GiglResourceConfigWrapper(resource_config).shared_resource_config
        )
    return task_config, resource_config


def kfp_validation_checks(
    job_name: str,
    source_task_config_uri: Uri,
    start_at: str,
    source_resource_config_uri: Uri,
    stop_after: Optional[str] = None,
) -> tuple[gbml_config_pb2.GbmlConfig, GiglResourceConfig, bool]:
    task_config, resource_config = resolve_configs(
        source_task_config_uri=source_task_config_uri,
        source_resource_config_uri=source_resource_config_uri,
    )
    should_use_live_sgs_backend = _validate_resolved_configs(
        job_name=job_name,
        start_at=start_at,
        task_config=task_config,
        resource_config=resource_config,
        stop_after=stop_after,
    )
    return task_config, resource_config, should_use_live_sgs_backend


def materialize_composed_config_snapshots(
    job_name: str,
    task_config: gbml_config_pb2.GbmlConfig,
    resource_config: GiglResourceConfig,
    task_config_source: str,
    resource_config_source: str,
) -> tuple[GcsUri, GcsUri]:
    """Write resolved config snapshots to stable GCS paths.

    Args:
        job_name: Name for the pipeline run.
        task_config: Resolved task config protobuf.
        resource_config: Resolved resource config protobuf.
        task_config_source: Provenance label for the task config source.
        resource_config_source: Provenance label for the resource config source.

    Returns:
        The task and resource snapshot URIs.
    """
    resource_config_wrapper = GiglResourceConfigWrapper(resource_config)
    snapshot_root = (
        resource_config_wrapper.temp_assets_regional_bucket_path
        / job_name
        / "config_validator"
    )
    composed_task_config_snapshot_uri = snapshot_root / "resolved_task_config.yaml"
    composed_resource_config_snapshot_uri = (
        snapshot_root / "resolved_resource_config.yaml"
    )
    gcs_utils = GcsUtils(project=resource_config_wrapper.project)
    gcs_utils.upload_from_string(
        gcs_path=composed_task_config_snapshot_uri,
        content=f"# Resolved Hydra config from: {task_config_source}\n"
        + proto_to_yaml(task_config),
    )
    gcs_utils.upload_from_string(
        gcs_path=composed_resource_config_snapshot_uri,
        content=f"# Resolved Hydra config from: {resource_config_source}\n"
        + proto_to_yaml(resource_config),
    )
    return (
        composed_task_config_snapshot_uri,
        composed_resource_config_snapshot_uri,
    )


def _source_config_label(uri: Uri, docker_uri: Optional[str]) -> str:
    """Include the image for local configs baked into a container."""
    if isinstance(uri, LocalUri) and docker_uri:
        return f"{docker_uri}:{uri}"
    return f"{uri}"


def _write_kfp_output(path: str, value: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Checks if config files and assets are valid for a GiGL pipeline run."
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="Unique identifier for the job name",
    )
    parser.add_argument(
        "--source_task_config_uri",
        type=str,
        help="User-supplied template or frozen task config URI to compose",
    )
    parser.add_argument(
        "--start_at",
        type=str,
        help="Specify the component where to start the pipeline",
    )
    parser.add_argument(
        "--stop_after",
        type=str,
        help="Specify the component where to stop the pipeline",
    )
    parser.add_argument(
        "--source_resource_config_uri",
        type=str,
        help="User-supplied resource config URI to compose",
    )
    parser.add_argument(
        "--output_file_path_composed_task_config_snapshot_uri",
        type=str,
        required=True,
        help="KFP output path for the composed task config snapshot URI",
    )
    parser.add_argument(
        "--output_file_path_composed_resource_config_snapshot_uri",
        type=str,
        required=True,
        help="KFP output path for the composed resource config snapshot URI",
    )
    parser.add_argument(
        "--output_file_path_should_use_glt_backend",
        type=str,
        required=True,
        help="KFP output path for the GLT backend decision",
    )
    parser.add_argument(
        "--cpu_docker_uri",
        type=str,
        default=None,
        help="Uri to dockerized source code compiled for cpu at runtime",
    )
    parser.add_argument(
        "--cuda_docker_uri",
        type=str,
        default=None,
        help="Uri to dockerized source code compiled for gpu at runtime",
    )
    args = parser.parse_args()

    source_task_config_uri = UriFactory.create_uri(args.source_task_config_uri)
    source_resource_config_uri = UriFactory.create_uri(args.source_resource_config_uri)

    check_if_kfp_pipeline_job_name_valid(job_name=args.job_name)
    task_config, resource_config = resolve_configs(
        source_task_config_uri=source_task_config_uri,
        source_resource_config_uri=source_resource_config_uri,
    )
    (
        composed_task_config_snapshot_uri,
        composed_resource_config_snapshot_uri,
    ) = materialize_composed_config_snapshots(
        job_name=args.job_name,
        task_config=task_config,
        resource_config=resource_config,
        task_config_source=_source_config_label(
            source_task_config_uri, args.cpu_docker_uri
        ),
        resource_config_source=_source_config_label(
            source_resource_config_uri, args.cpu_docker_uri
        ),
    )

    # Validation imports user-defined classes, so initialize the runtime from
    # the same snapshots that downstream components consume.
    initialize_gigl_runtime(
        applied_task_identifier=args.job_name,
        task_config_uri=composed_task_config_snapshot_uri,
        resource_config_uri=composed_resource_config_snapshot_uri,
        service_name=args.job_name,
        component=GiGLComponents.ConfigValidator,
        cpu_docker_uri=args.cpu_docker_uri,
        cuda_docker_uri=args.cuda_docker_uri,
    )

    should_use_glt_backend = _validate_resolved_configs(
        job_name=args.job_name,
        start_at=args.start_at,
        task_config=task_config,
        resource_config=resource_config,
        stop_after=args.stop_after,
    )
    _write_kfp_output(
        args.output_file_path_composed_task_config_snapshot_uri,
        composed_task_config_snapshot_uri.uri,
    )
    _write_kfp_output(
        args.output_file_path_composed_resource_config_snapshot_uri,
        composed_resource_config_snapshot_uri.uri,
    )
    _write_kfp_output(
        args.output_file_path_should_use_glt_backend,
        "true" if should_use_glt_backend else "false",
    )
