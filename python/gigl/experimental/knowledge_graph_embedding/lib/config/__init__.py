from __future__ import annotations

from dataclasses import dataclass, field

import torch
from applied_tasks.knowledge_graph_embedding.lib.config.evaluation import (
    EvaluationPhaseConfig,
)
from applied_tasks.knowledge_graph_embedding.lib.config.graph import (
    EnumeratedGraphData,
    GraphConfig,
    RawGraphData,
)
from applied_tasks.knowledge_graph_embedding.lib.config.model import ModelConfig
from applied_tasks.knowledge_graph_embedding.lib.config.run import RunConfig
from applied_tasks.knowledge_graph_embedding.lib.config.training import TrainConfig
from google.protobuf.json_format import ParseDict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gigl.common.logger import Logger
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from snapchat.research.gbml import graph_schema_pb2

logger = Logger()


@dataclass
class HeterogeneousGraphSparseEmbeddingConfig:
    run: RunConfig
    graph: GraphConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    validation: EvaluationPhaseConfig = field(default_factory=EvaluationPhaseConfig)
    testing: EvaluationPhaseConfig = field(default_factory=EvaluationPhaseConfig)

    @staticmethod
    def from_omegaconf(config: DictConfig) -> HeterogeneousGraphSparseEmbeddingConfig:
        """
        Create a HeterogeneousGraphSparseEmbeddingConfig object from an OmegaConf DictConfig.
        Args:
            config: The OmegaConf DictConfig object containing the configuration.
        Returns:
            A HeterogeneousGraphSparseEmbeddingConfig object.
        """

        # Build the GraphConfig
        graph_metadata = OmegaConf.select(config, "dataset.metadata", default=None)
        assert graph_metadata is not None, "Graph metadata is required in the config."

        graph_metadata_dict = OmegaConf.to_container(graph_metadata, resolve=True)
        pb = ParseDict(
            js_dict=graph_metadata_dict, message=graph_schema_pb2.GraphMetadata()
        )
        graph_metadata = GraphMetadataPbWrapper(graph_metadata_pb=pb)

        raw_graph_data = OmegaConf.select(
            config, "dataset.raw_graph_data", default=None
        )
        if raw_graph_data:
            raw_node_data = [instantiate(entry) for entry in raw_graph_data.node_data]
            raw_edge_data = [instantiate(entry) for entry in raw_graph_data.edge_data]

        enumerated_graph_data = OmegaConf.select(
            config, "dataset.enumerated_graph_data", default=None
        )
        if enumerated_graph_data:
            enumerated_node_data = [
                instantiate(entry) for entry in enumerated_graph_data.node_data
            ]
            enumerated_edge_data = [
                instantiate(entry) for entry in enumerated_graph_data.edge_data
            ]

        graph_config = GraphConfig(
            metadata=graph_metadata,
            raw_graph_data=RawGraphData(
                node_data=raw_node_data, edge_data=raw_edge_data
            )
            if raw_graph_data
            else None,
            enumerated_graph_data=EnumeratedGraphData(
                node_data=enumerated_node_data, edge_data=enumerated_edge_data
            )
            if enumerated_graph_data
            else None,
        )

        # Build the RunConfig
        run_config_info = OmegaConf.select(config, "run", default=None)
        assert run_config_info is not None, "Run config is required in the config."
        run_config: RunConfig = instantiate(run_config_info)

        structured_config = OmegaConf.merge(
            OmegaConf.structured(ModelConfig), config.model
        )
        model_config: ModelConfig = OmegaConf.to_object(structured_config)

        structured_config = OmegaConf.merge(
            OmegaConf.structured(TrainConfig), config.training
        )
        train_config: TrainConfig = OmegaConf.to_object(structured_config)

        structured_config = OmegaConf.merge(
            OmegaConf.structured(EvaluationPhaseConfig), config.validation
        )
        validation_config: EvaluationPhaseConfig = OmegaConf.to_object(
            structured_config
        )

        structured_config = OmegaConf.merge(
            OmegaConf.structured(EvaluationPhaseConfig), config.testing
        )
        testing_config: EvaluationPhaseConfig = OmegaConf.to_object(structured_config)

        config = HeterogeneousGraphSparseEmbeddingConfig(
            run=run_config,
            graph=graph_config,
            model=model_config,
            training=train_config,
            validation=validation_config,
            testing=testing_config,
        )

        return config

    def __post_init__(self):
        num_processes_per_machine = self.training.distributed.num_processes_per_machine
        if (
            self.run.should_use_cuda
            and num_processes_per_machine > torch.cuda.device_count()
        ):
            logger.warning(
                f"""Requested CUDA training and {num_processes_per_machine} processes per machine,
                but only {torch.cuda.device_count()} GPUs available.  Reducing number of processes per machine to
                {torch.cuda.device_count()}."""
            )
            self.training.distributed.num_processes_per_machine = (
                torch.cuda.device_count()
            )
