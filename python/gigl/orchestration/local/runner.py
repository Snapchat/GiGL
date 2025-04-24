from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.env.dep_constants import GIGL_DATAFLOW_IMAGE
from gigl.src.common.constants.components import GiGLComponents
from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.metrics_service_provider import initialize_metrics
from gigl.src.config_populator.config_populator import ConfigPopulator
from gigl.src.data_preprocessor.data_preprocessor import DataPreprocessor
from gigl.src.inference.inferencer import Inferencer
from gigl.src.split_generator.split_generator import SplitGenerator
from gigl.src.subgraph_sampler.subgraph_sampler import SubgraphSampler
from gigl.src.training.trainer import Trainer
from gigl.src.validation_check.config_validator import (
    START_COMPONENT_TO_ASSET_CHECKS_MAP,
    START_COMPONENT_TO_CLS_CHECKS_MAP,
)
from snapchat.research.gbml import gbml_config_pb2

logger = Logger()


@dataclass
class PipelineConfig:
    applied_task_identifier: AppliedTaskIdentifier
    task_config_uri: Uri
    resource_config_uri: Uri
    custom_cuda_docker_uri: Optional[str] = None
    custom_cpu_docker_uri: Optional[str] = None
    dataflow_docker_uri: Optional[str] = GIGL_DATAFLOW_IMAGE


class Runner:
    """
    Orchestration of GiGL Pipeline with local execution.

    Args:
        pipeline_config (PipelineConfig): Configuration for the pipeline.
        start_at (str): Component to start the pipeline from. Default is config_populator.
    """

    @staticmethod
    def run(
        pipeline_config: PipelineConfig,
        start_at: str = GiGLComponents.ConfigPopulator.value,
    ):
        logger.info(
            f"Running pipeline from component {start_at} with parameters: \n"
            f"job_name: {pipeline_config.applied_task_identifier}\n"
            f"task_config_uri: {pipeline_config.task_config_uri}\n"
            f"resource_config_uri: {pipeline_config.resource_config_uri}\n"
            f"dataflow_docker_uri: {pipeline_config.dataflow_docker_uri}"
        )

        initialize_metrics(
            task_config_uri=pipeline_config.task_config_uri,
            service_name=pipeline_config.applied_task_identifier,
        )

        if start_at == GiGLComponents.ConfigPopulator.value:
            frozen_config_uri = Runner.run_config_populator(pipeline_config)
            pipeline_config.task_config_uri = frozen_config_uri
        else:
            Runner.config_check(start_at, pipeline_config)

        component_map: OrderedDict[GiGLComponents, Callable] = OrderedDict(
            {
                GiGLComponents.ConfigPopulator.value: Runner.run_config_populator,
                GiGLComponents.DataPreprocessor.value: Runner.run_data_preprocessor,
                GiGLComponents.SubgraphSampler.value: Runner.run_subgraph_sampler,
                GiGLComponents.SplitGenerator.value: Runner.run_split_generator,
                GiGLComponents.Trainer.value: Runner.run_trainer,
                GiGLComponents.Inferencer.value: Runner.run_inferencer,
            }
        )

        started: bool = False
        for component, method in component_map.items():
            if component == start_at:
                started = True
            if started:
                method(pipeline_config)

    @staticmethod
    def config_check(start_at: str, pipeline_config: PipelineConfig):
        proto_utils = ProtoUtils()
        gbml_config_pb: gbml_config_pb2.GbmlConfig = proto_utils.read_proto_from_yaml(
            uri=pipeline_config.task_config_uri, proto_cls=gbml_config_pb2.GbmlConfig
        )

        for cls_check in START_COMPONENT_TO_CLS_CHECKS_MAP.get(start_at, []):
            cls_check(gbml_config_pb=gbml_config_pb)

        for asset_check in START_COMPONENT_TO_ASSET_CHECKS_MAP.get(start_at, []):
            asset_check(gbml_config_pb=gbml_config_pb)

    @staticmethod
    def run_config_populator(pipeline_config: PipelineConfig) -> Uri:
        logger.info("Running Config Populator...")
        config_populator = ConfigPopulator()

        return config_populator.run(
            applied_task_identifier=pipeline_config.applied_task_identifier,
            task_config_uri=pipeline_config.task_config_uri,
            resource_config_uri=pipeline_config.resource_config_uri,
        )

    @staticmethod
    def run_data_preprocessor(pipeline_config: PipelineConfig) -> None:
        logger.info("Running Data Preprocessor...")
        data_preprocessor = DataPreprocessor()
        data_preprocessor.run(
            applied_task_identifier=pipeline_config.applied_task_identifier,
            task_config_uri=pipeline_config.task_config_uri,
            resource_config_uri=pipeline_config.resource_config_uri,
            custom_worker_image_uri=pipeline_config.dataflow_docker_uri,
        )

    @staticmethod
    def run_subgraph_sampler(pipeline_config: PipelineConfig) -> None:
        logger.info("Running Subgraph Sampler...")
        subgraph_sampler = SubgraphSampler()
        subgraph_sampler.run(
            applied_task_identifier=pipeline_config.applied_task_identifier,
            task_config_uri=pipeline_config.task_config_uri,
            resource_config_uri=pipeline_config.resource_config_uri,
        )

    @staticmethod
    def run_split_generator(pipeline_config: PipelineConfig) -> None:
        logger.info("Running Split Generator...")
        split_generator = SplitGenerator()
        split_generator.run(
            applied_task_identifier=pipeline_config.applied_task_identifier,
            task_config_uri=pipeline_config.task_config_uri,
            resource_config_uri=pipeline_config.resource_config_uri,
        )

    @staticmethod
    def run_trainer(pipeline_config: PipelineConfig) -> None:
        logger.info("Running Trainer...")
        trainer = Trainer()
        trainer.run(
            applied_task_identifier=pipeline_config.applied_task_identifier,
            task_config_uri=pipeline_config.task_config_uri,
            resource_config_uri=pipeline_config.resource_config_uri,
            cpu_docker_uri=pipeline_config.custom_cpu_docker_uri,
            cuda_docker_uri=pipeline_config.custom_cuda_docker_uri,
        )

    @staticmethod
    def run_inferencer(pipeline_config: PipelineConfig) -> None:
        logger.info("Running Inferencer...")
        inferencer = Inferencer()
        inferencer.run(
            applied_task_identifier=pipeline_config.applied_task_identifier,
            task_config_uri=pipeline_config.task_config_uri,
            resource_config_uri=pipeline_config.resource_config_uri,
            custom_worker_image_uri=pipeline_config.dataflow_docker_uri,
            cpu_docker_uri=pipeline_config.custom_cpu_docker_uri,
            cuda_docker_uri=pipeline_config.custom_cuda_docker_uri,
        )
