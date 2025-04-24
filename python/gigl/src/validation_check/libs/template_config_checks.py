import re
from typing import Any, Dict, Optional

from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.common.utils import os_utils
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.constants.components import GLT_BACKEND_UNSUPPORTED_COMPONENTS
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.subgraph_sampling_strategy import (
    SubgraphSamplingStrategyPbWrapper,
)
from gigl.src.common.types.pb_wrappers.task_metadata import TaskMetadataPbWrapper
from gigl.src.common.types.task_metadata import TaskMetadataType
from gigl.src.data_preprocessor.lib.data_preprocessor_config import (
    DataPreprocessorConfig,
)
from gigl.src.inference.v1.lib.base_inferencer import BaseInferencer
from gigl.src.post_process.lib.base_post_processor import BasePostProcessor
from gigl.src.training.v1.lib.base_trainer import BaseTrainer
from gigl.src.validation_check.libs.utils import assert_proto_field_value_is_truthy
from snapchat.research.gbml import gbml_config_pb2, preprocessed_metadata_pb2

logger = Logger()


def check_if_kfp_pipeline_job_name_valid(job_name: str) -> None:
    """
    Check if kfp pipeline job name valid. It is used to start spark cluster and must match pattern.
    The kfp pipeline job name is also used to generate AppliedTaskIdentifier for each component.
    """
    logger.info("Config validation check: if job_name valid.")
    if not bool(re.match(r"^(?:[a-z](?:[-_a-z0-9]{0,49}[a-z0-9])?)$", job_name)):
        raise ValueError(
            f"Invalid 'job_name'. Only lowercase letters, numbers, and dashes are allowed. "
            f"The value must start with lowercase letter or number and end with a lowercase letter or number. "
            f"'job_name' provided: {job_name} ."
        )


def check_pipeline_has_valid_start_and_stop_flags(
    start_at: str,
    stop_after: Optional[str],
    task_config_uri: str,
) -> None:
    """
    Check if start_at and stop_after are valid with current static (gigl) or dynamic (glt) backend
    """
    gbml_config_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=UriFactory.create_uri(task_config_uri)
    )
    components = [start_at] if stop_after is None else [start_at, stop_after]
    for component in components:
        if gbml_config_wrapper.should_use_experimental_glt_backend:
            if component in GLT_BACKEND_UNSUPPORTED_COMPONENTS:
                raise ValueError(
                    f"Invalid component {component} for GLT Backend"
                    f"GLT Backend does not support components {GLT_BACKEND_UNSUPPORTED_COMPONENTS}."
                )


def check_if_runtime_args_all_str(args_name: str, runtime_args: Dict[str, Any]) -> None:
    """
    Check if all values of the given runtime arguements are string.
    """
    for arg_key, arg_value in runtime_args.items():
        if type(arg_value) is not str:
            raise ValueError(
                f"Invalid type for runtime arguements under {args_name}, should be string. "
                f"Got {arg_value} with type {type(arg_value)} for {arg_key}."
            )


def check_if_task_metadata_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if taskMetadata specification is valid.
    """
    logger.info("Config validation check: if taskMetadata is valid.")
    pb_wrapper = TaskMetadataPbWrapper(gbml_config_pb.task_metadata)
    assert (
        pb_wrapper.task_metadata_type is not None
    ), "Invalid 'taskMetadata'; must be provided."
    task_metadata_type = pb_wrapper.task_metadata_type

    # We need to check if the types in the task_metadata are valid according to the graph_metadata.
    graph_metadata_pb = gbml_config_pb.graph_metadata

    task_metadata_pb = pb_wrapper.task_metadata
    if task_metadata_type == TaskMetadataType.NODE_BASED_TASK:
        assert isinstance(
            task_metadata_pb,
            gbml_config_pb2.GbmlConfig.TaskMetadata.NodeBasedTaskMetadata,
        ), f"Found 'taskMetadata' of type {task_metadata_type}, but pb is of type {type(task_metadata_pb)}; must be {gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata}."
        assert (
            len(task_metadata_pb.supervision_node_types) > 0
        ), "Must provide at least one supervision node type."
        for node_type in task_metadata_pb.supervision_node_types:
            assert (
                node_type in graph_metadata_pb.node_types
            ), f"Invalid supervision node type: {node_type}; not found in graphMetadata node types {graph_metadata_pb.node_types}."
    elif task_metadata_type == TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK:
        assert isinstance(
            task_metadata_pb,
            gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata,
        ), f"Found 'taskMetadata' of type {task_metadata_type}, but pb is of type {type(task_metadata_pb)}; must be {gbml_config_pb2.GbmlConfig.TaskMetadata.NodeAnchorBasedLinkPredictionTaskMetadata}."
        assert (
            len(task_metadata_pb.supervision_edge_types) > 0
        ), "Must provide at least one supervision edge type."
        graph_metadata_pb_edge_types = [
            GbmlProtosTranslator.edge_type_from_EdgeTypePb(edge_type_pb=edge_type_pb)
            for edge_type_pb in graph_metadata_pb.edge_types
        ]
        for edge_type_pb in task_metadata_pb.supervision_edge_types:
            edge_type = GbmlProtosTranslator.edge_type_from_EdgeTypePb(
                edge_type_pb=edge_type_pb
            )
            assert (
                edge_type in graph_metadata_pb_edge_types
            ), f"Invalid supervision edge type: {edge_type}; not found in graphMetadata edge types {graph_metadata_pb_edge_types}."
    else:
        raise ValueError(
            f"Invalid 'taskMetadata'; must be one of {[TaskMetadataType.NODE_BASED_TASK, TaskMetadataType.NODE_ANCHOR_BASED_LINK_PREDICTION_TASK]}.",
            f"{TaskMetadataType.LINK_BASED_TASK} is not yet supported.",
        )


def check_if_preprocessed_metadata_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if preprocessedMetadata is valid.
    """
    logger.info("Config validation check: if preprocessedMetadata is valid.")
    if not gbml_config_pb.shared_config.preprocessed_metadata_uri:
        raise ValueError("Invalid 'preprocessedMetadata'; must be provided.")
    proto_utils = ProtoUtils()
    pb: preprocessed_metadata_pb2.PreprocessedMetadata = (
        proto_utils.read_proto_from_yaml(
            uri=UriFactory.create_uri(
                gbml_config_pb.shared_config.preprocessed_metadata_uri
            ),
            proto_cls=preprocessed_metadata_pb2.PreprocessedMetadata,
        )
    )

    # Check that the number of preprocessed node types is nonzero, and every type exists in the graph metadata.
    assert (
        len(pb.condensed_node_type_to_preprocessed_metadata) > 0
    ), "preprocessedMetadata found with no node types."
    for condensed_node_type in pb.condensed_node_type_to_preprocessed_metadata:
        assert (
            condensed_node_type in gbml_config_pb.graph_metadata.condensed_node_type_map
        ), (
            f"Invalid condensed node type {condensed_node_type} in preprocessedMetadata: ",
            f"No matching entry in graphMetadata.",
        )

    # Check that the number of preprocessed edge types is nonzero, and every type exists in the graph metadata.
    assert (
        len(pb.condensed_edge_type_to_preprocessed_metadata) > 0
    ), "preprocessedMetadata found with no edge types."
    for condensed_edge_type in pb.condensed_edge_type_to_preprocessed_metadata:
        assert (
            condensed_edge_type in gbml_config_pb.graph_metadata.condensed_edge_type_map
        ), (
            f"Invalid condensed edge type {condensed_edge_type} in preprocessedMetadata: ",
            f"No matching entry in graphMetadata.",
        )

    for (
        _,
        node_metadata_output,
    ) in pb.condensed_node_type_to_preprocessed_metadata.items():
        # Check if all required fields are present in the node metadata output.
        for field in [
            "node_id_key",
            "schema_uri",
            "tfrecord_uri_prefix",
            "transform_fn_assets_uri",
            "enumerated_node_ids_bq_table",
            "enumerated_node_data_bq_table",
        ]:
            assert_proto_field_value_is_truthy(
                proto=node_metadata_output, field_name=field
            )

    for (
        _,
        edge_metadata_output,
    ) in pb.condensed_edge_type_to_preprocessed_metadata.items():
        # Check if all required fields are present in the edge metadata output.
        for field in ["src_node_id_key", "dst_node_id_key", "main_edge_info"]:
            assert_proto_field_value_is_truthy(
                proto=edge_metadata_output, field_name=field
            )

        # Check if all required fields are present in the main edge info.
        for field in [
            "schema_uri",
            "tfrecord_uri_prefix",
            "transform_fn_assets_uri",
            "enumerated_edge_data_bq_table",
        ]:
            assert_proto_field_value_is_truthy(
                proto=edge_metadata_output.main_edge_info, field_name=field
            )


def check_if_graph_metadata_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if GraphMetadata specification is valid.
    """
    logger.info("Config validation check: if graphMetadata is valid.")
    graph_metadata_pb = gbml_config_pb.graph_metadata
    if not graph_metadata_pb:
        raise ValueError("Invalid 'graphMetadata'; must be provided.")

    assert (
        graph_metadata_pb.node_types
    ), "Must provide at least one node type in graphMetadata."
    assert (
        graph_metadata_pb.edge_types
    ), "Must provide at least one edge type in graphMetadata."


def check_if_data_preprocessor_config_cls_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if dataPreprocessorArgs are all string.
    Check if dataPreprocessorConfigClsPath is valid and importable.
    """
    logger.info(
        "Config validation check: if dataPreprocessorConfigClsPath and its args are valid."
    )
    data_preprocessor_config_cls_path = (
        gbml_config_pb.dataset_config.data_preprocessor_config.data_preprocessor_config_cls_path
    )
    runtime_args: Dict[str, str] = dict(
        gbml_config_pb.dataset_config.data_preprocessor_config.data_preprocessor_args
    )
    check_if_runtime_args_all_str(
        args_name="dataPreprocessorArgs", runtime_args=runtime_args
    )
    try:
        data_preprocessor_cls = os_utils.import_obj(data_preprocessor_config_cls_path)
        data_preprocessor_config: DataPreprocessorConfig = data_preprocessor_cls(
            **runtime_args
        )
        assert isinstance(data_preprocessor_config, DataPreprocessorConfig)
    except Exception as e:
        raise ValueError(
            f"Invalid 'dataPreprocessorConfigClsPath' in frozen config: datasetConfig - dataPreprocessorConfig. "
            f"'dataPreprocessorConfigClsPath' provided: {data_preprocessor_config_cls_path}. "
            f"Error: {e}"
        )


def check_if_trainer_cls_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if trainerArgs are all string.
    Check if trainerClsPath is valid and importable.
    """
    logger.info("Config validation check: if trainerClsPath and its args are valid.")
    gbml_config_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
    if gbml_config_wrapper.should_use_experimental_glt_backend:
        logger.warning(
            "Skipping trainer class validation as GLT Backend is not implemented yet. "
            + "Trainer class may actually be a path to a script so, the paradigm is different."
            + "This is temporary to unblock testing and will be refactored in the future."
        )
        return
    trainer_cls_path = gbml_config_pb.trainer_config.trainer_cls_path
    runtime_args: Dict[str, str] = dict(gbml_config_pb.trainer_config.trainer_args)
    check_if_runtime_args_all_str(args_name="trainerArgs", runtime_args=runtime_args)
    try:
        trainer_cls = os_utils.import_obj(trainer_cls_path)
        trainer: BaseTrainer = trainer_cls(**runtime_args)
        assert isinstance(trainer, BaseTrainer)
    except Exception as e:
        raise ValueError(
            f"Invalid 'trainerClsPath' in frozen config: trainerConfig - trainerClsPath. "
            f"'trainerClsPath' provided: {trainer_cls_path}. "
            f"Error: {e}"
        )


def check_if_inferencer_cls_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if inferencerArgs are all string.
    Check if inferencerClsPath is valid and importable.
    """
    logger.info("Config validation check: if inferencerClsPath and its args are valid.")
    inferencer_cls_path = gbml_config_pb.inferencer_config.inferencer_cls_path
    runtime_args: Dict[str, str] = dict(
        gbml_config_pb.inferencer_config.inferencer_args
    )
    check_if_runtime_args_all_str(args_name="inferencerArgs", runtime_args=runtime_args)

    gbml_config_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
    if gbml_config_wrapper.should_use_experimental_glt_backend:
        logger.warning(
            "Skipping inferencer class validation as GLT Backend is enabled. "
            + "Inferencer class may actually be a path to a script so, the paradigm is different."
            + "This is temporary to unblock testing and will be refactored in the future."
        )
        return

    try:
        inferencer_cls = os_utils.import_obj(inferencer_cls_path)
        inferencer_instance: BaseInferencer = inferencer_cls(**runtime_args)
        assert isinstance(inferencer_instance, BaseInferencer)
    except Exception as e:
        raise ValueError(
            f"Invalid 'inferencerClsPath' in frozen config: inferencerConfig - inferencerClsPath. "
            f"'inferencerClsPath' provided: {inferencer_cls_path}. "
            f"Error: {e}"
        )


def check_if_split_generator_config_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if splitGeneratorConfig is valid.
    """
    logger.info("Config validation check: if splitGeneratorConfig is valid.")
    gbml_config_wrapper = GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)
    if gbml_config_wrapper.should_use_experimental_glt_backend:
        logger.warning(
            "Skipping splitGeneratorConfig validation as GLT Backend is enabled."
        )
        return

    assigner_cls_path = (
        gbml_config_pb.dataset_config.split_generator_config.assigner_cls_path
    )
    split_strategy_cls_path = (
        gbml_config_pb.dataset_config.split_generator_config.split_strategy_cls_path
    )

    if not assigner_cls_path or not split_strategy_cls_path:
        raise ValueError(
            "Invalid class paths or class paths not provided in splitGeneratorConfig."
        )

    assigner_args = dict(
        gbml_config_pb.dataset_config.split_generator_config.assigner_args
    )
    check_if_runtime_args_all_str(args_name="assignerArgs", runtime_args=assigner_args)


def check_if_subgraph_sampler_config_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if subgraphSamplerConfig is valid.
    """
    logger.info("Config validation check: if subgraphSamplerConfig is valid.")
    subgraph_sampler_config = gbml_config_pb.dataset_config.subgraph_sampler_config

    if subgraph_sampler_config.HasField("subgraph_sampling_strategy"):
        subgraph_sampling_strategy_pb_wrapper = SubgraphSamplingStrategyPbWrapper(
            subgraph_sampler_config.subgraph_sampling_strategy
        )
        subgraph_sampling_strategy_pb_wrapper.validate_dags(
            graph_metadata_pb=gbml_config_pb.graph_metadata,
            task_metadata_pb=gbml_config_pb.task_metadata,
        )
    else:
        num_hops = subgraph_sampler_config.num_hops
        num_neighbors_to_sample = subgraph_sampler_config.num_neighbors_to_sample

        if num_hops <= 0:
            raise ValueError("Invalid numHops in subgraphSamplerConfig.")
        if num_neighbors_to_sample <= 0:
            raise ValueError("Invalid numNeighborsToSample in subgraphSamplerConfig.")

    num_positive_samples = subgraph_sampler_config.num_positive_samples

    num_user_defined_positive_samples = (
        subgraph_sampler_config.num_user_defined_positive_samples
    )

    num_user_defined_negative_samples = (
        subgraph_sampler_config.num_user_defined_negative_samples
    )

    num_max_training_samples_to_output = (
        subgraph_sampler_config.num_max_training_samples_to_output
    )

    if num_positive_samples < 0:
        raise ValueError("Invalid numPositiveSamples in subgraphSamplerConfig.")
    if num_user_defined_positive_samples < 0:
        raise ValueError(
            "Invalid numUserDefinedPositiveSamples in subgraphSamplerConfig."
        )
    if num_user_defined_negative_samples < 0:
        raise ValueError(
            "Invalid numUserDefinedNegativeSamples in subgraphSamplerConfig."
        )
    if num_user_defined_positive_samples > 0 and num_positive_samples > 0:
        raise ValueError(
            "Can provide either num_positive_samples, or num_user_defined_positive_samples; not both."
        )
    assert (
        sum([num_user_defined_positive_samples, num_positive_samples]) > 0
    ), "Must provide either num_positive_samples, or num_user_defined_positive_samples."

    if num_max_training_samples_to_output < 0:
        raise ValueError(
            "Invalid numMaxTrainingSamplesToOutput in subgraphSamplerConfig."
        )


def check_if_post_processor_cls_valid(
    gbml_config_pb: gbml_config_pb2.GbmlConfig,
) -> None:
    """
    Check if postProcessorArgs are all string.
    Check if postProcessorClsPath is valid and importable.
    """
    logger.info(
        "Config validation check: if postProcessorClsPath and its args are valid."
    )
    post_processor_cls_path = (
        gbml_config_pb.post_processor_config.post_processor_cls_path
    )
    if not post_processor_cls_path:
        logger.info(
            "No post processor class provided - skipping checks for post processor"
        )
        return

    runtime_args: Dict[str, str] = dict(
        gbml_config_pb.post_processor_config.post_processor_args
    )
    check_if_runtime_args_all_str(
        args_name="postProcessorArgs", runtime_args=runtime_args
    )
    try:
        post_processor_cls = os_utils.import_obj(post_processor_cls_path)
        post_processor_instance: BasePostProcessor = post_processor_cls(**runtime_args)
        assert isinstance(post_processor_instance, BasePostProcessor)
    except Exception as e:
        raise ValueError(
            f"Invalid 'postProcessorClsPath' in frozen config and/or postProcessorArgs could not sucessfully "
            f"initialize the 'postProcessorClsPath' provided: {post_processor_cls_path}. "
            f"Error: {e}"
        )
