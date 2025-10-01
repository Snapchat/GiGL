from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, cast

import torch
from google.protobuf.json_format import ParseDict
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gigl.common.logger import Logger
from gigl.experimental.knowledge_graph_embedding.lib.config.evaluation import (
    EvaluationPhaseConfig,
)
from gigl.experimental.knowledge_graph_embedding.lib.config.graph import (
    EnumeratedGraphData,
    GraphConfig,
    RawGraphData,
)
from gigl.experimental.knowledge_graph_embedding.lib.config.model import ModelConfig
from gigl.experimental.knowledge_graph_embedding.lib.config.run import RunConfig
from gigl.experimental.knowledge_graph_embedding.lib.config.training import TrainConfig
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)
from snapchat.research.gbml import graph_schema_pb2

logger = Logger()


@dataclass
class HeterogeneousGraphSparseEmbeddingConfig:
    """
    Main configuration class for heterogeneous graph sparse embedding training.

    This configuration orchestrates all aspects of knowledge graph embedding model training,
    including the graph data structure, model architecture, training parameters, and
    evaluation settings.

    Attributes:
        run (RunConfig): Runtime configuration specifying execution environment (GPU/CPU usage).
        graph (GraphConfig): Graph configuration containing metadata and data references for nodes and edges.
        model (ModelConfig): Model architecture configuration including embedding dimensions and operators.
            Defaults to ModelConfig() with standard settings.
        training (TrainConfig): Training configuration with optimization, sampling, and distributed settings.
            Defaults to TrainConfig() with standard settings.
        validation (EvaluationPhaseConfig): Evaluation configuration for validation phase during training.
            Defaults to EvaluationPhaseConfig() with standard settings.
        testing (EvaluationPhaseConfig): Evaluation configuration for final model testing phase.
            Defaults to EvaluationPhaseConfig() with standard settings.
    """

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
        graph_data: Optional[RawGraphData] = None
        if raw_graph_data:
            raw_node_data = [instantiate(entry) for entry in raw_graph_data.node_data]
            raw_edge_data = [instantiate(entry) for entry in raw_graph_data.edge_data]
            graph_data = RawGraphData(node_data=raw_node_data, edge_data=raw_edge_data)

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
            enumerated_graph_data = EnumeratedGraphData(
                node_data=enumerated_node_data, edge_data=enumerated_edge_data
            )

        graph_config = GraphConfig(
            metadata=graph_metadata,
            raw_graph_data=graph_data,
            enumerated_graph_data=enumerated_graph_data,
        )

        # Build the RunConfig
        run_config_info = OmegaConf.select(config, "run", default=None)
        assert run_config_info is not None, "Run config is required in the config."
        run_config: RunConfig = instantiate(run_config_info)

        structured_config = OmegaConf.merge(
            OmegaConf.structured(ModelConfig), config.model
        )
        model_config: ModelConfig = cast(
            ModelConfig, OmegaConf.to_object(structured_config)
        )

        structured_config = OmegaConf.merge(
            OmegaConf.structured(TrainConfig), config.training
        )
        train_config: TrainConfig = cast(
            TrainConfig, OmegaConf.to_object(structured_config)
        )

        structured_config = OmegaConf.merge(
            OmegaConf.structured(EvaluationPhaseConfig), config.validation
        )
        validation_config: EvaluationPhaseConfig = cast(
            EvaluationPhaseConfig, OmegaConf.to_object(structured_config)
        )

        structured_config = OmegaConf.merge(
            OmegaConf.structured(EvaluationPhaseConfig), config.testing
        )
        testing_config: EvaluationPhaseConfig = cast(
            EvaluationPhaseConfig, OmegaConf.to_object(structured_config)
        )

        final_config = HeterogeneousGraphSparseEmbeddingConfig(
            run=run_config,
            graph=graph_config,
            model=model_config,
            training=train_config,
            validation=validation_config,
            testing=testing_config,
        )

        return final_config

    def __post_init__(self) -> None:
        """
        Post-initialization validation and adjustment of configuration parameters.

        Automatically adjusts the number of processes per machine for distributed training
        if the requested number exceeds available GPU devices. Issues a warning when
        reducing the process count to match hardware availability.

        Raises:
            No exceptions are raised, but warnings are logged for configuration adjustments.
        """
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
