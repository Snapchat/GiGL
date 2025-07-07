import os
from typing import Dict, Final, List, NamedTuple, Optional

import kfp
import kfp.dsl.pipeline_channel
from kfp.dsl import PipelineTask

import gigl.src.common.constants.local_fs as local_fs_constants
from gigl.common import LocalUri
from gigl.common.logger import Logger
from gigl.common.types.resource_config import CommonPipelineComponentConfigs
from gigl.orchestration.kubeflow.utils.glt_backend import (
    check_glt_backend_eligibility_component,
)
from gigl.orchestration.kubeflow.utils.log_metrics import log_metrics_to_ui
from gigl.orchestration.kubeflow.utils.resource import add_task_resource_requirements
from gigl.src.common.constants.components import GiGLComponents

_COMPONENTS_BASE_PATH: Final[str] = os.path.join(
    local_fs_constants.get_gigl_root_directory(),
    "orchestration",
    "kubeflow",
)

logger = Logger()

SPECED_COMPONENTS: Final[List[str]] = [
    GiGLComponents.ConfigValidator.value,
    GiGLComponents.ConfigPopulator.value,
    GiGLComponents.SubgraphSampler.value,
    GiGLComponents.DataPreprocessor.value,
    GiGLComponents.SplitGenerator.value,
    GiGLComponents.Inferencer.value,
    GiGLComponents.PostProcessor.value,
    GiGLComponents.Trainer.value,
]

_speced_component_root: Final[LocalUri] = LocalUri.join(
    _COMPONENTS_BASE_PATH, "components"
)
_speced_component_op_dict: Final[Dict[GiGLComponents, kfp.components.YamlComponent]] = {
    GiGLComponents(component): kfp.components.load_component_from_file(
        LocalUri.join(_speced_component_root, component, "component.yaml").uri
    )
    for component in SPECED_COMPONENTS
}


def my_python_component(image: str):
    def _c():
        import time

        from gigl.common.logger import Logger

        logger = Logger()
        logger.info("This is a Python component that sleeps for 1 second.")
        logger.info("Starting sleep...")
        time.sleep(1)

    return kfp.dsl.component(
        _c,
        base_image=image,
    )


def decompose_task_config_uri_to_protos(image: str):
    def decompose(
        task_config_uri: str,
    ) -> NamedTuple(
        "outputs",
        node_type_to_condensed_node_type=str,
        edge_type_to_condensed_edge_type=str,
        data_preprocessor_class_path=str,
        data_preprocessor_args=str,
        preprocessed_metadata_uri=str,
    ):
        """
        Decomposes the task config URI into its constituent parts and writes them to output paths.
        """
        import json
        from typing import NamedTuple

        from gigl.common import UriFactory
        from gigl.common.logger import Logger
        from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper

        logger = Logger()

        logger.info(f"Task config uri: {task_config_uri}")
        gbml_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
            gbml_config_uri=UriFactory.create_uri(task_config_uri)
        )
        condensed_node_type_str = "#".join(
            f"{k}:{v}"
            for k, v in gbml_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map.items()
        )
        logger.info(f"Condensed Node Types: {condensed_node_type_str}")
        condensed_edge_type_str = "#".join(
            f"{k[0]},{k[1]},{k[2]}:{v}"
            for k, v in gbml_pb_wrapper.graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map.items()
        )
        logger.info(f"Condensed Edge Types: {condensed_edge_type_str}")
        condensed_node_type_to_condensed_node_type = condensed_node_type_str
        condensed_edge_type_to_condensed_edge_type = condensed_edge_type_str
        data_preprocessor_class_path = (
            gbml_pb_wrapper.dataset_config.data_preprocessor_config.data_preprocessor_config_cls_path
        )
        data_preprocessor_args = dict(
            gbml_pb_wrapper.dataset_config.data_preprocessor_config.data_preprocessor_args
        )
        preprocessed_metadata_uri = (
            gbml_pb_wrapper.shared_config.preprocessed_metadata_uri
        )
        logger.info(f"Data Preprocessor Class Path: {data_preprocessor_class_path}")
        logger.info(f"Data Preprocessor Args: {data_preprocessor_args}")
        logger.info(f"Preprocessed Metadata URI: {preprocessed_metadata_uri}")
        outputs = NamedTuple(
            "outputs",
            node_type_to_condensed_node_type=str,
            edge_type_to_condensed_edge_type=str,
            data_preprocessor_class_path=str,
            data_preprocessor_args=str,
            preprocessed_metadata_uri=str,
        )
        return outputs(
            condensed_node_type_to_condensed_node_type,
            condensed_edge_type_to_condensed_edge_type,
            data_preprocessor_class_path,
            json.dumps(data_preprocessor_args),
            preprocessed_metadata_uri,
        )

    return kfp.dsl.component(
        decompose,
        base_image=image,
    )


def _generate_component_task(
    component: GiGLComponents,
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    start_at: Optional[str] = None,
    stop_after: Optional[str] = None,
) -> PipelineTask:
    component_task: PipelineTask
    if component == GiGLComponents.ConfigPopulator:
        component_task = _speced_component_op_dict[component](
            job_name=job_name,
            template_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            **common_pipeline_component_configs.additional_job_args.get(component, {}),
        )

    elif component == GiGLComponents.ConfigValidator:
        component_task = _speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=task_config_uri,
            start_at=start_at,
            resource_config_uri=resource_config_uri,
            stop_after=stop_after,
            **common_pipeline_component_configs.additional_job_args.get(component, {}),
        )
    elif component == GiGLComponents.SubgraphSampler:
        component_task = _speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            custom_worker_image_uri=common_pipeline_component_configs.dataflow_container_image,
            **common_pipeline_component_configs.additional_job_args.get(component, {}),
        )
    elif component == GiGLComponents.Trainer:
        component_task = _speced_component_op_dict[component](
            job_name=job_name,
            config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            cpu_docker_uri=common_pipeline_component_configs.cpu_container_image,
            cuda_docker_uri=common_pipeline_component_configs.cuda_container_image,
            **common_pipeline_component_configs.additional_job_args.get(component, {}),
        )
    elif component == GiGLComponents.DataPreprocessor:
        # logger.info(f"Task config uri: {task_config_uri}")
        # gbml_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        #         gbml_config_uri=UriFactory.create_uri(task_config_uri.value)
        # )
        # condensed_node_type_str = "#".join(f"{k}:{v}" for k, v in gbml_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map.items())
        # logger.info(f"Condensed Node Types: {condensed_node_type_str}")
        # condensed_edge_type_str = "#".join(f"{k[0]},{k[1]}{k[2]}:{v}" for k, v in gbml_pb_wrapper.graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map.items())
        # logger.info(f"Condensed Edge Types: {condensed_edge_type_str}")
        prepared_outputs = decompose_task_config_uri_to_protos(
            image=common_pipeline_component_configs.cpu_container_image,
        )(task_config_uri=task_config_uri)
        component_task = _speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            custom_worker_image_uri=common_pipeline_component_configs.dataflow_container_image,
            data_preprocessor_class_path=prepared_outputs.outputs[
                "data_preprocessor_class_path"
            ],
            data_preprocessor_args=prepared_outputs.outputs["data_preprocessor_args"],
            preprocessed_metadata_uri=prepared_outputs.outputs[
                "preprocessed_metadata_uri"
            ],
            node_type_to_condensed_node_type=prepared_outputs.outputs[
                "node_type_to_condensed_node_type"
            ],
            edge_type_to_condensed_edge_type=prepared_outputs.outputs[
                "edge_type_to_condensed_edge_type"
            ],
            data_preprocessor_num_shards=0,
            **common_pipeline_component_configs.additional_job_args.get(component, {}),
        )
    elif component == GiGLComponents.Inferencer:
        component_task = _speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            custom_worker_image_uri=common_pipeline_component_configs.dataflow_container_image,
            cpu_docker_uri=common_pipeline_component_configs.cpu_container_image,
            cuda_docker_uri=common_pipeline_component_configs.cuda_container_image,
            **common_pipeline_component_configs.additional_job_args.get(component, {}),
        )
    else:
        component_task = _speced_component_op_dict[component](
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            **common_pipeline_component_configs.additional_job_args.get(component, {}),
        )
    add_task_resource_requirements(
        task=component_task,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    return component_task


def generate_pipeline(
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    tag: Optional[str] = None,
):
    """
    Generates a KFP pipeline definition for GiGL.
    Args:
        common_pipeline_component_configs (CommonPipelineComponentConfigs): Shared configuration between components.
        tag (Optiona[str]): Optional tag, which is provided will be used to tag the pipeline description.

    Returns:
        An @kfp.dsl.pipeline decorated function to generated a pipeline.
    """

    @kfp.dsl.pipeline(
        name="GiGL_Pipeline",
        description="GiGL Pipeline" if not tag else f"GiGL Pipeline @ {tag}",
    )
    def pipeline(
        job_name: str,
        template_or_frozen_config_uri: str,
        resource_config_uri: str,
        start_at: str = GiGLComponents.ConfigPopulator.value,
        stop_after: Optional[str] = None,
    ):
        """
        test_vai_launch = create_custom_training_job_from_component(
            my_python_component(common_pipeline_component_configs.cpu_container_image),
            replica_count=4,
        )(project=get_, location="us-central1")
        """
        validation_check_task = _generate_component_task(
            component=GiGLComponents.ConfigValidator,
            job_name=job_name,
            task_config_uri=template_or_frozen_config_uri,
            start_at=start_at,
            stop_after=stop_after,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
        )
        should_use_glt = check_glt_backend_eligibility_component(
            task_config_uri=template_or_frozen_config_uri,
            base_image=common_pipeline_component_configs.cpu_container_image,
        )

        with kfp.dsl.Condition(start_at == GiGLComponents.ConfigPopulator.value):
            config_populator_task = _create_config_populator_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                should_use_glt_runtime_param=should_use_glt,
                stop_after=stop_after,
            )
            config_populator_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.DataPreprocessor.value):
            data_preprocessor_task = _create_data_preprocessor_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
                should_use_glt_runtime_param=should_use_glt,
            )
            data_preprocessor_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.SubgraphSampler.value):
            subgraph_sampler_task = _create_subgraph_sampler_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            subgraph_sampler_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.SplitGenerator.value):
            split_generator_task = _create_split_generator_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            split_generator_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.Trainer.value):
            trainer_task = _create_trainer_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            trainer_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.Inferencer.value):
            inferencer_task = _create_inferencer_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            inferencer_task.after(validation_check_task)

        with kfp.dsl.Condition(start_at == GiGLComponents.PostProcessor.value):
            post_processor_task = _create_post_processor_task_op(
                job_name=job_name,
                task_config_uri=template_or_frozen_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
            )
            post_processor_task.after(validation_check_task)

    return pipeline


def _create_config_populator_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    should_use_glt_runtime_param: kfp.dsl.pipeline_channel.PipelineChannel,
    stop_after: Optional[str] = None,
) -> PipelineTask:
    config_populator_task = _generate_component_task(
        component=GiGLComponents.ConfigPopulator,
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
        stop_after=stop_after,
    )
    frozen_gbml_config_uri = config_populator_task.outputs["frozen_gbml_config_uri"]

    with kfp.dsl.Condition(stop_after != GiGLComponents.ConfigPopulator.value):
        data_preprocessor_task = _create_data_preprocessor_task_op(
            job_name=job_name,
            task_config_uri=frozen_gbml_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
            should_use_glt_runtime_param=should_use_glt_runtime_param,
            stop_after=stop_after,
        )
        data_preprocessor_task.after(config_populator_task)
    return config_populator_task


def _create_data_preprocessor_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    should_use_glt_runtime_param: kfp.dsl.pipeline_channel.PipelineChannel,
    stop_after: Optional[str] = None,
) -> PipelineTask:
    data_preprocessor_task = _generate_component_task(
        component=GiGLComponents.DataPreprocessor,
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )
    add_task_resource_requirements(
        task=data_preprocessor_task,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    with kfp.dsl.Condition(stop_after != GiGLComponents.DataPreprocessor.value):
        with kfp.dsl.Condition(should_use_glt_runtime_param == False):
            subgraph_sampler_task = _create_subgraph_sampler_task_op(
                job_name=job_name,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            subgraph_sampler_task.after(data_preprocessor_task)
        # If we are using the GLT runtime, we skip the subgraph sampler and split generator
        # and go straight to the GLT trainer
        with kfp.dsl.Condition(should_use_glt_runtime_param == True):
            glt_trainer_task = _create_trainer_task_op(
                job_name=job_name,
                task_config_uri=task_config_uri,
                resource_config_uri=resource_config_uri,
                common_pipeline_component_configs=common_pipeline_component_configs,
                stop_after=stop_after,
            )
            glt_trainer_task.after(data_preprocessor_task)

    return data_preprocessor_task


def _create_subgraph_sampler_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> PipelineTask:
    subgraph_sampler_task = _generate_component_task(
        component=GiGLComponents.SubgraphSampler,
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    with kfp.dsl.Condition(stop_after != GiGLComponents.SubgraphSampler.value):
        split_generator_task = _create_split_generator_task_op(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
            stop_after=stop_after,
        )
        split_generator_task.after(subgraph_sampler_task)

    return subgraph_sampler_task


def _create_split_generator_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> PipelineTask:
    split_generator_task: PipelineTask
    split_generator_task = _generate_component_task(
        component=GiGLComponents.SplitGenerator,
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    with kfp.dsl.Condition(stop_after != GiGLComponents.SplitGenerator.value):
        trainer_task = _create_trainer_task_op(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
            stop_after=stop_after,
        )
        trainer_task.after(split_generator_task)

    return split_generator_task


def _create_inferencer_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> PipelineTask:
    inferencer_task = _generate_component_task(
        component=GiGLComponents.Inferencer,
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )
    add_task_resource_requirements(
        task=inferencer_task,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    with kfp.dsl.Condition(stop_after != GiGLComponents.Inferencer.value):
        post_processor_task = _create_post_processor_task_op(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
        )
        post_processor_task.after(inferencer_task)

    return inferencer_task


def _create_trainer_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
    stop_after: Optional[str] = None,
) -> PipelineTask:
    trainer_task = _generate_component_task(
        component=GiGLComponents.Trainer,
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )

    log_metrics_component = log_metrics_to_ui(
        task_config_uri=task_config_uri,
        component_name=GiGLComponents.Trainer.value,
        base_image=common_pipeline_component_configs.cpu_container_image,
    )
    log_metrics_component.set_display_name(name="Log Trainer Eval Metrics")
    log_metrics_component.after(trainer_task)

    with kfp.dsl.Condition(stop_after != GiGLComponents.Trainer.value):
        inference_task = _create_inferencer_task_op(
            job_name=job_name,
            task_config_uri=task_config_uri,
            resource_config_uri=resource_config_uri,
            common_pipeline_component_configs=common_pipeline_component_configs,
            stop_after=stop_after,
        )
        inference_task.after(trainer_task)
    return trainer_task


def _create_post_processor_task_op(
    job_name: str,
    task_config_uri: str,
    resource_config_uri: str,
    common_pipeline_component_configs: CommonPipelineComponentConfigs,
) -> PipelineTask:
    post_processor_task = _generate_component_task(
        component=GiGLComponents.PostProcessor,
        job_name=job_name,
        task_config_uri=task_config_uri,
        resource_config_uri=resource_config_uri,
        common_pipeline_component_configs=common_pipeline_component_configs,
    )
    log_metrics_component = log_metrics_to_ui(
        task_config_uri=task_config_uri,
        component_name=GiGLComponents.PostProcessor.value,
        base_image=common_pipeline_component_configs.cpu_container_image,
    )
    log_metrics_component.set_display_name(name="Log PostProcessor Eval Metrics")
    log_metrics_component.after(post_processor_task)
    return post_processor_task
