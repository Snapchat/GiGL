from typing import Optional, Union

from google.cloud.aiplatform_v1.types.accelerator_type import AcceleratorType

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from gigl.src.validation_check.libs.utils import (
    assert_proto_field_value_is_truthy,
    assert_proto_has_field,
)
from snapchat.research.gbml import gigl_resource_config_pb2

logger = Logger()


def _check_if_dataflow_resource_config_valid(
    dataflow_resource_config_pb: gigl_resource_config_pb2.DataflowResourceConfig,
) -> None:
    """
    Checks if the provided Dataflow resource configuration is valid.

    Args:
        dataflow_resource_config_pb (gigl_resource_config_pb2.DataflowResourceConfig): The dataflow resource configuration to be checked.

    Returns:
        None
    """
    for field in ["num_workers", "max_num_workers", "disk_size_gb", "machine_type"]:
        assert_proto_field_value_is_truthy(
            proto=dataflow_resource_config_pb, field_name=field
        )


def _check_if_spark_resource_config_valid(
    spark_resource_config_pb: gigl_resource_config_pb2.SparkResourceConfig,
) -> None:
    """
    Checks if the provided Spark resource configuration is valid.

    Args:
        spark_resource_config_pb (gigl_resource_config_pb2.SparkResourceConfig): The Spark resource configuration protobuf object.

    Returns:
        None
    """
    for field in ["machine_type", "num_local_ssds", "num_replicas"]:
        assert_proto_field_value_is_truthy(
            proto=spark_resource_config_pb, field_name=field
        )


def check_if_shared_resource_config_valid(
    resource_config_pb: gigl_resource_config_pb2.GiglResourceConfig,
) -> None:
    """
    Check if SharedResourceConfig specification is valid:
     - SharedResourceConfig or a SharedResourceConfig uri must be accessible in the resource config.
     - CommonComputeConfig must have appropriate fields defined.

    Args:
        resource_config_pb (gigl_resource_config_pb2.GiglResourceConfig): The resource config to be checked.

    Returns:
        None
    """

    logger.info("Config validation check: if resource config shared_resource is valid.")
    wrapper = GiglResourceConfigWrapper(resource_config=resource_config_pb)
    assert wrapper.shared_resource_config, (
        "Invalid 'shared_resource_config'; must provide shared_resource_config."
    )
    assert_proto_has_field(
        proto=wrapper.shared_resource_config, field_name="common_compute_config"
    )
    common_compute_config_pb = wrapper.shared_resource_config.common_compute_config
    for field in [
        "project",
        "region",
        "temp_assets_bucket",
        "temp_regional_assets_bucket",
        "perm_assets_bucket",
        "temp_assets_bq_dataset_name",
        "embedding_bq_dataset_name",
        "gcp_service_account_email",
        "dataflow_runner",
    ]:
        assert_proto_field_value_is_truthy(
            proto=common_compute_config_pb, field_name=field
        )


def check_if_preprocessor_resource_config_valid(
    resource_config_pb: gigl_resource_config_pb2.GiglResourceConfig,
) -> None:
    logger.info(
        "Config validation check: if resource config preprocessor_config is valid."
    )
    preprocessor_config: gigl_resource_config_pb2.DataPreprocessorConfig = (
        resource_config_pb.preprocessor_config
    )
    _check_if_dataflow_resource_config_valid(
        dataflow_resource_config_pb=preprocessor_config.node_preprocessor_config
    )
    _check_if_dataflow_resource_config_valid(
        dataflow_resource_config_pb=preprocessor_config.edge_preprocessor_config
    )


def check_if_subgraph_sampler_resource_config_valid(
    resource_config_pb: gigl_resource_config_pb2.GiglResourceConfig,
) -> None:
    logger.info(
        "Config validation check: if resource config subgraph_sampler_config is valid."
    )
    _check_if_spark_resource_config_valid(
        spark_resource_config_pb=resource_config_pb.subgraph_sampler_config
    )


def check_if_split_generator_resource_config_valid(
    resource_config_pb: gigl_resource_config_pb2.GiglResourceConfig,
) -> None:
    logger.info(
        "Config validation check: if resource config split_generator_config is valid."
    )
    _check_if_spark_resource_config_valid(
        spark_resource_config_pb=resource_config_pb.split_generator_config
    )


def check_if_trainer_resource_config_valid(
    resource_config_pb: gigl_resource_config_pb2.GiglResourceConfig,
) -> None:
    logger.info("Config validation check: if resource config trainer_config is valid.")
    wrapper = GiglResourceConfigWrapper(resource_config=resource_config_pb)
    assert wrapper.trainer_config, (
        "Invalid 'trainer_config'; must provide trainer_config."
    )

    trainer_config: Union[
        gigl_resource_config_pb2.LocalResourceConfig,
        gigl_resource_config_pb2.VertexAiResourceConfig,
        gigl_resource_config_pb2.KFPResourceConfig,
        gigl_resource_config_pb2.VertexAiGraphStoreConfig,
        gigl_resource_config_pb2.CustomResourceConfig,
    ] = wrapper.trainer_config
    if isinstance(trainer_config, gigl_resource_config_pb2.CustomResourceConfig):
        logger.info(
            "Skipping trainer machine-shape validation: trainer_config is a "
            "CustomResourceConfig (launcher-pluggable; no concrete machine "
            "spec to validate)."
        )
        return
    _validate_machine_config(config=trainer_config)


def check_if_inferencer_resource_config_valid(
    resource_config_pb: gigl_resource_config_pb2.GiglResourceConfig,
) -> None:
    logger.info(
        "Config validation check: if resource config inferencer_config is valid."
    )
    resource_config_wrapper = GiglResourceConfigWrapper(
        resource_config=resource_config_pb
    )
    inferencer_config = resource_config_wrapper.inferencer_config
    if isinstance(inferencer_config, gigl_resource_config_pb2.CustomResourceConfig):
        logger.info(
            "Skipping inferencer machine-shape validation: inferencer_config "
            "is a CustomResourceConfig (launcher-pluggable; no concrete "
            "machine spec to validate)."
        )
        return
    _validate_machine_config(config=inferencer_config)


def _validate_vertex_ai_resource_config(
    vertex_ai_resource_config_pb: gigl_resource_config_pb2.VertexAiResourceConfig,
) -> None:
    """
    Checks if the provided Vertex AI resource configuration is valid.
    """
    assert_proto_field_value_is_truthy(
        proto=vertex_ai_resource_config_pb, field_name="machine_type"
    )


def _validate_accelerator_type(
    proto_config: Union[
        gigl_resource_config_pb2.VertexAiResourceConfig,
        gigl_resource_config_pb2.KFPResourceConfig,
    ],
) -> None:
    """
    Checks if the provided accelerator type is valid.
    """
    if proto_config.gpu_type == AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name:  # type: ignore
        assert (
            proto_config.gpu_limit == 0
        ), f"""gpu_limit must be equal to 0 for cpu training/inference, indicated by provided gpu_type {proto_config.gpu_type}.
            Got gpu_limit {proto_config.gpu_limit}"""
    else:
        assert (
            proto_config.gpu_limit > 0
        ), f"""gpu_limit must be greater than 0 for gpu training/inference, indicated by provided gpu_type {proto_config.gpu_type}.
            Got gpu_limit {proto_config.gpu_limit}. Use gpu_type {AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name} for cpu training."""  # type: ignore


def _validate_cloud_machine_config(
    config: Union[
        gigl_resource_config_pb2.VertexAiResourceConfig,
        gigl_resource_config_pb2.KFPResourceConfig,
    ],
) -> None:
    """
    Checks if the provided cloud machine configuration is valid.
    """
    _validate_accelerator_type(proto_config=config)
    for field in [
        "gpu_type",
        "num_replicas",
    ]:
        assert_proto_field_value_is_truthy(proto=config, field_name=field)


def _validate_machine_config(
    config: Union[
        gigl_resource_config_pb2.LocalResourceConfig,
        gigl_resource_config_pb2.VertexAiResourceConfig,
        gigl_resource_config_pb2.KFPResourceConfig,
        gigl_resource_config_pb2.VertexAiGraphStoreConfig,
        gigl_resource_config_pb2.DataflowResourceConfig,
    ],
) -> None:
    if isinstance(config, gigl_resource_config_pb2.LocalResourceConfig):
        assert_proto_field_value_is_truthy(proto=config, field_name="num_workers")
    elif isinstance(config, gigl_resource_config_pb2.DataflowResourceConfig):
        _check_if_dataflow_resource_config_valid(dataflow_resource_config_pb=config)
    elif isinstance(config, gigl_resource_config_pb2.KFPResourceConfig):
        for field in [
            "cpu_request",
            "memory_request",
        ]:
            assert_proto_field_value_is_truthy(proto=config, field_name=field)
        _validate_cloud_machine_config(config=config)
    elif isinstance(config, gigl_resource_config_pb2.VertexAiResourceConfig):
        _validate_vertex_ai_resource_config(vertex_ai_resource_config_pb=config)
        _validate_accelerator_type(proto_config=config)
        _validate_cloud_machine_config(config=config)
    elif isinstance(config, gigl_resource_config_pb2.VertexAiGraphStoreConfig):
        _validate_vertex_ai_resource_config(
            vertex_ai_resource_config_pb=config.graph_store_pool
        )
        _validate_accelerator_type(proto_config=config.graph_store_pool)
        _validate_cloud_machine_config(config=config.graph_store_pool)

        _validate_vertex_ai_resource_config(
            vertex_ai_resource_config_pb=config.compute_pool
        )
        _validate_accelerator_type(proto_config=config.compute_pool)
        _validate_cloud_machine_config(config=config.compute_pool)
    else:
        raise ValueError(
            f"""Expected distributed config to be one of {gigl_resource_config_pb2.LocalResourceConfig.__name__},
            {gigl_resource_config_pb2.VertexAiResourceConfig.__name__},
            or {gigl_resource_config_pb2.KFPResourceConfig.__name__}.
            or {gigl_resource_config_pb2.VertexAiGraphStoreConfig.__name__}.
            Got {type(config)}"""
        )


def check_if_custom_resource_config_dry_run_valid(
    resource_config_pb: gigl_resource_config_pb2.GiglResourceConfig,
    task_config_uri: Uri,
    resource_config_uri: Uri,
    applied_task_identifier: str,
    cpu_docker_uri: Optional[str],
    cuda_docker_uri: Optional[str],
    component: GiGLComponents,
) -> None:
    """Invoke the custom launcher with ``is_dry_run=True`` for early validation.

    Resolves the component's resource config through the wrapper; if it is not
    a ``CustomResourceConfig`` this helper is a no-op. Otherwise it dispatches
    through ``launch_custom(..., is_dry_run=True)`` so the user-supplied
    launcher can validate its inputs without actually spawning remote jobs.

    The import of ``launch_custom`` is intentionally lazy: the dry-run hook is
    only reachable when the caller opts in via
    ``--check_custom_launcher_dry_run``, and keeping the import inside the
    function ensures ``assert_yaml_configs_parse`` (and other static config
    validators) do not transitively pull in launcher-side dependencies (which
    may be cluster-management clients such as a Ray platform SDK).

    Auth note: dry-run submission may call out to managed services that the
    launcher integrates with; the submitter must have whatever credentials
    those services require. See the custom launcher's own documentation for
    specifics.

    Args:
        resource_config_pb: The resource config to inspect. The trainer or
            inferencer oneof (depending on ``component``) is pulled out of the
            wrapper and, if it resolves to ``CustomResourceConfig``, dispatched
            to the launcher.
        task_config_uri: URI of the GbmlConfig YAML.
        resource_config_uri: URI of the GiglResourceConfig YAML.
        applied_task_identifier: Stable identifier for the job.
        cpu_docker_uri: Optional CPU Docker image URI forwarded to the launcher.
        cuda_docker_uri: Optional CUDA Docker image URI forwarded to the launcher.
        component: Which GiGL component to dry-run. Must be Trainer or
            Inferencer; other components never carry a ``CustomResourceConfig``.

    Raises:
        ValueError: If ``component`` is not Trainer or Inferencer.
    """
    # Lazy import — assert_yaml_configs_parse must stay import-free of
    # launcher-side deps (the resolved launcher may pull in a cluster SDK).
    from gigl.src.common.custom_launcher import launch_custom

    if component not in {GiGLComponents.Trainer, GiGLComponents.Inferencer}:
        raise ValueError(
            f"check_if_custom_resource_config_dry_run_valid only supports "
            f"Trainer and Inferencer components; got {component}."
        )

    wrapper = GiglResourceConfigWrapper(resource_config=resource_config_pb)
    component_config: Union[
        gigl_resource_config_pb2.LocalResourceConfig,
        gigl_resource_config_pb2.VertexAiResourceConfig,
        gigl_resource_config_pb2.KFPResourceConfig,
        gigl_resource_config_pb2.VertexAiGraphStoreConfig,
        gigl_resource_config_pb2.DataflowResourceConfig,
        gigl_resource_config_pb2.CustomResourceConfig,
    ]
    if component == GiGLComponents.Trainer:
        component_config = wrapper.trainer_config
    else:
        component_config = wrapper.inferencer_config

    if not isinstance(component_config, gigl_resource_config_pb2.CustomResourceConfig):
        logger.info(
            f"Skipping custom-launcher dry-run for {component.value}: "
            f"{type(component_config).__name__} is not a CustomResourceConfig."
        )
        return

    logger.info(
        f"Invoking custom launcher dry-run for {component.value} via "
        f"{component_config.launcher_fn}."
    )
    launch_custom(
        custom_resource_config=component_config,
        applied_task_identifier=applied_task_identifier,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        process_command="",
        process_runtime_args={},
        cpu_docker_uri=cpu_docker_uri,
        cuda_docker_uri=cuda_docker_uri,
        component=component,
        is_dry_run=True,
    )


def check_if_trainer_graph_store_storage_command_valid(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> None:
    """
    Validates that storage_command is set when graph store mode is enabled for trainer.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper to check.

    Raises:
        AssertionError: If graph store mode is enabled but storage_command is missing.
    """
    logger.info(
        "Config validation check: if trainer graph store storage_command is valid."
    )
    trainer_config = gbml_config_pb_wrapper.gbml_config_pb.trainer_config
    if trainer_config.HasField("graph_store_storage_config"):
        storage_command = trainer_config.graph_store_storage_config.command
        if not storage_command:
            raise AssertionError(
                "GbmlConfig.trainer_config.graph_store_storage_config.storage_command must be set "
                "when using graph store mode for trainer."
            )


def check_if_inferencer_graph_store_storage_command_valid(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
) -> None:
    """
    Validates that storage_command is set when graph store mode is enabled for inferencer.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper to check.

    Raises:
        AssertionError: If graph store mode is enabled but storage_command is missing.
    """
    logger.info(
        "Config validation check: if inferencer graph store storage_command is valid."
    )
    inferencer_config = gbml_config_pb_wrapper.gbml_config_pb.inferencer_config
    if inferencer_config.HasField("graph_store_storage_config"):
        storage_command = inferencer_config.graph_store_storage_config.command
        if not storage_command:
            raise AssertionError(
                "GbmlConfig.inferencer_config.graph_store_storage_config.storage_command must be set "
                "when using graph store mode for inferencer."
            )
