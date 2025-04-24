from typing import Union

from google.cloud.aiplatform_v1.types.accelerator_type import AcceleratorType

from gigl.common.logger import Logger
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
    assert (
        wrapper.shared_resource_config
    ), "Invalid 'shared_resource_config'; must provide shared_resource_config."
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
    assert (
        wrapper.trainer_config
    ), "Invalid 'trainer_config'; must provide trainer_config."

    trainer_config: Union[
        gigl_resource_config_pb2.LocalResourceConfig,
        gigl_resource_config_pb2.VertexAiResourceConfig,
        gigl_resource_config_pb2.KFPResourceConfig,
    ] = wrapper.trainer_config

    if isinstance(trainer_config, gigl_resource_config_pb2.LocalResourceConfig):
        assert_proto_field_value_is_truthy(
            proto=trainer_config, field_name="num_workers"
        )
    else:
        # Case where trainer config is gigl_resource_config_pb2.VertexAiResourceConfig or gigl_resource_config_pb2.KFPResourceConfig
        if isinstance(trainer_config, gigl_resource_config_pb2.VertexAiResourceConfig):
            assert_proto_field_value_is_truthy(
                proto=trainer_config, field_name="machine_type"
            )
        elif isinstance(trainer_config, gigl_resource_config_pb2.KFPResourceConfig):
            for field in [
                "cpu_request",
                "memory_request",
            ]:
                assert_proto_field_value_is_truthy(
                    proto=trainer_config, field_name=field
                )
        else:
            raise ValueError(
                f"""Expected distributed trainer config to be one of {gigl_resource_config_pb2.LocalResourceConfig.__name__}, 
                {gigl_resource_config_pb2.VertexAiResourceConfig.__name__}, 
                or {gigl_resource_config_pb2.KFPResourceConfig.__name__}. 
                Got {type(trainer_config)}"""
            )

        for field in [
            "gpu_type",
            "num_replicas",
        ]:
            assert_proto_field_value_is_truthy(proto=trainer_config, field_name=field)

        if trainer_config.gpu_type == AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name:  # type: ignore
            assert (
                trainer_config.gpu_limit == 0
            ), f"""gpu_limit must be equal to 0 for cpu training, indicated by provided gpu_type {trainer_config.gpu_type}. 
                Got gpu_limit {trainer_config.gpu_limit}"""
        else:
            assert (
                trainer_config.gpu_limit > 0
            ), f"""gpu_limit must be greater than 0 for gpu training, indicated by provided gpu_type {trainer_config.gpu_type}. 
                Got gpu_limit {trainer_config.gpu_limit}. Use gpu_type {AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name} for cpu training."""  # type: ignore


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
    if isinstance(inferencer_config, gigl_resource_config_pb2.DataflowResourceConfig):
        _check_if_dataflow_resource_config_valid(
            dataflow_resource_config_pb=inferencer_config
        )
    elif isinstance(inferencer_config, gigl_resource_config_pb2.VertexAiResourceConfig):
        assert_proto_field_value_is_truthy(
            proto=inferencer_config, field_name="machine_type"
        )
        assert_proto_field_value_is_truthy(
            proto=inferencer_config, field_name="gpu_type"
        )
        assert_proto_field_value_is_truthy(
            proto=inferencer_config, field_name="num_replicas"
        )
        if inferencer_config.gpu_type == AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name:  # type: ignore
            assert (
                inferencer_config.gpu_limit == 0
            ), f"""gpu_limit must be equal to 0 for cpu training, indicated by provided gpu_type {inferencer_config.gpu_type}. 
                Got gpu_limit {inferencer_config.gpu_limit}"""
        else:
            assert (
                inferencer_config.gpu_limit > 0
            ), f"""gpu_limit must be greater than 0 for gpu training, indicated by provided gpu_type {inferencer_config.gpu_type}. 
                Got gpu_limit {inferencer_config.gpu_limit}. Use gpu_type {AcceleratorType.ACCELERATOR_TYPE_UNSPECIFIED.name} for cpu training."""  # type: ignore
    elif isinstance(inferencer_config, gigl_resource_config_pb2.LocalResourceConfig):
        assert_proto_field_value_is_truthy(
            proto=inferencer_config, field_name="num_workers"
        )
    else:
        raise ValueError(
            f"""Expected inferencer config to be one of {gigl_resource_config_pb2.DataflowResourceConfig.__name__}, 
            {gigl_resource_config_pb2.VertexAiResourceConfig.__name__}, 
            or {gigl_resource_config_pb2.LocalResourceConfig.__name__}. 
            Got {type(inferencer_config)}"""
        )
