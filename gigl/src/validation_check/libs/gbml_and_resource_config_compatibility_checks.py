"""
Compatibility checks between GbmlConfig (template config) and GiglResourceConfig (resource config).

These checks ensure that graph store mode configurations are consistent across both configs.
If graph store mode is set up for trainer or inferencer in one config, it must be set up in the other.
"""

from typing import Literal

from google.protobuf.message import Message

from gigl.common.logger import Logger
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.gigl_resource_config import (
    GiglResourceConfigWrapper,
)
from snapchat.research.gbml import gigl_resource_config_pb2

logger = Logger()


def _gbml_config_has_graph_store(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    component: Literal["trainer", "inferencer"],
) -> bool:
    """
    Check if the GbmlConfig has graph_store_storage_config set for inferencer.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper to check.

    Returns:
        True if graph_store_storage_config is set for inferencer, False otherwise.
    """
    if component == "inferencer":
        config: Message = gbml_config_pb_wrapper.gbml_config_pb.inferencer_config
    elif component == "trainer":
        config = gbml_config_pb_wrapper.gbml_config_pb.trainer_config
    else:
        raise ValueError(
            f"Invalid component: {component}. Must be 'inferencer' or 'trainer'."
        )
    return config.HasField("graph_store_storage_config")


def _resource_config_has_graph_store(
    resource_config_wrapper: GiglResourceConfigWrapper,
    component: Literal["trainer", "inferencer"],
) -> bool:
    """
    Check if the GiglResourceConfig has VertexAiGraphStoreConfig set for the given component.

    Args:
        resource_config_wrapper: The resource config wrapper to check.

    Returns:
        True if VertexAiGraphStoreConfig is set for trainer, False otherwise.
    """
    if component == "trainer":
        config: Message = resource_config_wrapper.trainer_config
    elif component == "inferencer":
        config = resource_config_wrapper.inferencer_config
    else:
        raise ValueError(
            f"Invalid component: {component}. Must be 'trainer' or 'inferencer'."
        )
    return isinstance(config, gigl_resource_config_pb2.VertexAiGraphStoreConfig)


def check_trainer_graph_store_compatibility(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    resource_config_wrapper: GiglResourceConfigWrapper,
) -> None:
    """
    Check that trainer graph store mode is consistently configured across both configs.

    If graph_store_storage_config is set in GbmlConfig.trainer_config, then
    VertexAiGraphStoreConfig must be set in GiglResourceConfig.trainer_resource_config,
    and vice versa. Also validates that storage_command is set when graph store mode is enabled.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper (template config).
        resource_config_wrapper: The GiglResourceConfig wrapper (resource config).

    Raises:
        AssertionError: If graph store configurations are not compatible or storage_command is missing.
    """
    logger.info(
        "Config validation check: trainer graph store compatibility between template and resource configs."
    )

    gbml_has_graph_store = _gbml_config_has_graph_store(
        gbml_config_pb_wrapper, "trainer"
    )
    resource_has_graph_store = _resource_config_has_graph_store(
        resource_config_wrapper, "trainer"
    )

    if gbml_has_graph_store != resource_has_graph_store:
        raise AssertionError(
            f"If one of GbmlConfig.trainer_config.graph_store_storage_config or GiglResourceConfig.trainer_resource_config is set, the other must also be set. GbmlConfig.trainer_config.graph_store_storage_config is set: {gbml_has_graph_store}, GiglResourceConfig.trainer_resource_config is set: {resource_has_graph_store}."
        )


def check_inferencer_graph_store_compatibility(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    resource_config_wrapper: GiglResourceConfigWrapper,
) -> None:
    """
    Check that inferencer graph store mode is consistently configured across both configs.

    If graph_store_storage_config is set in GbmlConfig.inferencer_config, then
    VertexAiGraphStoreConfig must be set in GiglResourceConfig.inferencer_resource_config,
    and vice versa. Also validates that storage_command is set when graph store mode is enabled.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper (template config).
        resource_config_wrapper: The GiglResourceConfig wrapper (resource config).

    Raises:
        AssertionError: If graph store configurations are not compatible or storage_command is missing.
    """
    logger.info(
        "Config validation check: inferencer graph store compatibility between template and resource configs."
    )

    gbml_has_graph_store = _gbml_config_has_graph_store(
        gbml_config_pb_wrapper, "inferencer"
    )
    resource_has_graph_store = _resource_config_has_graph_store(
        resource_config_wrapper, "inferencer"
    )

    if gbml_has_graph_store != resource_has_graph_store:
        raise AssertionError(
            f"If one of GbmlConfig.inferencer_config.graph_store_storage_config or GiglResourceConfig.inferencer_resource_config is set, the other must also be set. GbmlConfig.inferencer_config.graph_store_storage_config is set: {gbml_has_graph_store}, GiglResourceConfig.inferencer_resource_config is set: {resource_has_graph_store}."
        )


def check_custom_resource_config_requires_glt_backend(
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    resource_config_wrapper: GiglResourceConfigWrapper,
) -> None:
    """Enforce that ``CustomResourceConfig`` is only used with the GLT (v2) backend.

    The v1 trainer/inferencer dispatchers never consult the
    ``custom_trainer_config`` / ``custom_inferencer_config`` oneof, so pairing
    a ``CustomResourceConfig`` with a task config that has
    ``should_use_glt_backend=False`` would silently fall through the v1 path
    and fail at runtime. Catch it up-front here so the failure is loud and
    actionable at validation time.

    Note on naming: the wrapper exposes ``should_use_glt_backend`` (bool) but
    the raw YAML key users set is ``feature_flags.should_run_glt_backend``.
    The wrapper translates one into the other; this check always reads the
    wrapper property and never the raw map.

    Args:
        gbml_config_pb_wrapper: The GbmlConfig wrapper (template config).
        resource_config_wrapper: The GiglResourceConfig wrapper (resource config).

    Raises:
        ValueError: If either the trainer or inferencer resource config is a
            ``CustomResourceConfig`` and ``should_use_glt_backend`` is False.
    """
    logger.info(
        "Config validation check: CustomResourceConfig requires GLT (v2) backend."
    )
    trainer_is_custom = isinstance(
        resource_config_wrapper.trainer_config,
        gigl_resource_config_pb2.CustomResourceConfig,
    )
    inferencer_is_custom = isinstance(
        resource_config_wrapper.inferencer_config,
        gigl_resource_config_pb2.CustomResourceConfig,
    )
    if not (trainer_is_custom or inferencer_is_custom):
        return

    if not gbml_config_pb_wrapper.should_use_glt_backend:
        offending: list[str] = []
        if trainer_is_custom:
            offending.append("trainer_resource_config.custom_trainer_config")
        if inferencer_is_custom:
            offending.append("inferencer_resource_config.custom_inferencer_config")
        raise ValueError(
            "CustomResourceConfig is only wired into the GLT (v2) dispatchers "
            "(glt_trainer.py / glt_inferencer.py); the v1 trainer/inferencer "
            "never consult the custom oneof and would fall through to an "
            "'Unsupported resource config' error at runtime. The following "
            f"custom resource configs were set: {offending}, but the task "
            "config has should_use_glt_backend=False (raw YAML key: "
            "feature_flags.should_run_glt_backend). Either set "
            "feature_flags.should_run_glt_backend='True' in the task config, "
            "or replace the CustomResourceConfig with a built-in resource "
            "config."
        )
