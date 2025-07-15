from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Union

import torch
import torch_geometric.data

from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.translators.training_samples_protos_translator import (
    SupervisedNodeClassificationSample,
    TrainingSamplesProtosTranslator,
)
from gigl.src.common.types.graph_data import Node, NodeId
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.training.v1.lib.data_loaders.common import DataloaderConfig
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    LoopyIterableDataset,
    TfRecordsIterableDataset,
)
from gigl.src.training.v1.lib.data_loaders.utils import cast_graph_for_training
from snapchat.research.gbml import graph_schema_pb2, training_samples_schema_pb2


@dataclass
class SupervisedNodeClassificationBatch:
    graph: Union[
        torch_geometric.data.Data, torch_geometric.data.hetero_data.HeteroData
    ]  # batch-coalesced graph data used for message passing
    root_node_indices: torch.LongTensor  # dtype: int64, shape: [num_root_nodes, ]
    root_nodes: List[Node]  # len(root_nodes) == number of graphs in Batch
    root_node_labels: Optional[
        torch.LongTensor
    ] = None  # dtype: int64, shape: [num_root_nodes, ]

    @staticmethod
    def preprocess_node_classification_sample_fn(
        sample_pb: training_samples_schema_pb2.SupervisedNodeClassificationSample,
        builder: GraphBuilder,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> SupervisedNodeClassificationSample:
        samples = TrainingSamplesProtosTranslator.training_samples_from_SupervisedNodeClassificationSamplePb(
            samples=[sample_pb],
            graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            builder=builder,
        )
        sample = samples[0]  # Only 1 element since we are only processing 1 thing
        return sample

    @staticmethod
    def preprocess_node_classification_raw_sample_fn(
        raw_data: bytes,
        builder: GraphBuilder,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> SupervisedNodeClassificationSample:
        sample_pb = training_samples_schema_pb2.SupervisedNodeClassificationSample()
        sample_pb.ParseFromString(raw_data)
        return (
            SupervisedNodeClassificationBatch.preprocess_node_classification_sample_fn(
                sample_pb=sample_pb,
                builder=builder,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            )
        )

    @staticmethod
    def collate_pyg_node_classification_minibatch(
        builder: GraphBuilder,
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
        samples: List[SupervisedNodeClassificationSample],
    ) -> SupervisedNodeClassificationBatch:
        ordered_root_nodes: List[Node] = list()
        ordered_labels: List[int] = list()
        for sample in samples:
            ordered_root_nodes.append(sample.root_node)
            # If a sample has a label, add it.
            if len(sample.y):
                ordered_labels.append(sample.y[0].label)

            graph_data = sample.x
            if not isinstance(graph_data, PygGraphData):
                raise NotImplementedError(
                    f"Subgraph must be of type {PygGraphData.__name__}:"
                    f"instead found type {type(graph_data)}."
                )
            builder.add_graph_data(graph_data=graph_data)
        batch_graph_data = builder.build()

        node_mapping = batch_graph_data.global_node_to_subgraph_node_mapping
        _batch_ordered_root_node_indices: List[NodeId] = [
            node_mapping[root_node].id for root_node in ordered_root_nodes
        ]
        batch_graph_data.coalesce()
        batch_graph_data = cast_graph_for_training(
            batch_graph_data=batch_graph_data,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
            batch_type=SupervisedNodeClassificationBatch.__name__,
            should_register_edge_features=builder.should_register_edge_features,
        )
        ordered_root_node_labels = (
            torch.LongTensor(ordered_labels) if len(ordered_labels) else None
        )
        return SupervisedNodeClassificationBatch(
            graph=batch_graph_data,
            root_nodes=ordered_root_nodes,
            root_node_indices=torch.LongTensor(_batch_ordered_root_node_indices),
            root_node_labels=ordered_root_node_labels,
        )

    @staticmethod
    def process_raw_pyg_samples_and_collate_fn(
        batch: List[training_samples_schema_pb2.SupervisedNodeClassificationSample],
        builder: GraphBuilder,
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
    ) -> SupervisedNodeClassificationBatch:
        ordered_root_nodes: List[Node] = []
        ordered_labels: List[int] = []
        graph_samples: List[graph_schema_pb2.Graph] = []
        for sample in batch:
            root_node, _ = GbmlProtosTranslator.node_from_NodePb(
                graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
                node_pb=sample.root_node,
            )
            ordered_root_nodes.append(root_node)
            if sample.root_node_labels:
                ordered_labels.append(sample.root_node_labels[0].label)
            graph_samples.append(sample.neighborhood)

        batch_graph_data = GbmlProtosTranslator.graph_data_from_GraphPb(
            samples=graph_samples,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            builder=builder,
        )

        if not isinstance(batch_graph_data, PygGraphData):
            raise NotImplementedError(
                f"Subgraph must be of type {PygGraphData.__name__}:"
                f"instead found type {type(batch_graph_data)}."
            )

        node_mapping = batch_graph_data.global_node_to_subgraph_node_mapping
        _batch_ordered_root_node_indices: List[NodeId] = [
            node_mapping[root_node].id for root_node in ordered_root_nodes
        ]
        batch_graph_data.coalesce()
        batch_graph_data = cast_graph_for_training(
            batch_graph_data=batch_graph_data,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
            batch_type=SupervisedNodeClassificationBatch.__name__,
            should_register_edge_features=builder.should_register_edge_features,
        )
        ordered_root_node_labels = (
            torch.LongTensor(ordered_labels) if len(ordered_labels) else None
        )

        return SupervisedNodeClassificationBatch(
            graph=batch_graph_data,
            root_nodes=ordered_root_nodes,
            root_node_indices=torch.LongTensor(_batch_ordered_root_node_indices),
            root_node_labels=ordered_root_node_labels,
        )

    @staticmethod
    def get_default_data_loader(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_builder: GraphBuilder,
        config: DataloaderConfig,
    ) -> torch.utils.data.DataLoader:
        preprocess_raw_sample_fn = partial(
            SupervisedNodeClassificationBatch.preprocess_node_classification_raw_sample_fn,
            builder=graph_builder,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )
        _iterable_training_dataset = TfRecordsIterableDataset(
            tf_record_uris=config.uris,  # type: ignore
            process_raw_sample_fn=preprocess_raw_sample_fn,
            seed=config.seed,
        )
        iterable_training_dataset: Union[
            LoopyIterableDataset[SupervisedNodeClassificationSample],
            TfRecordsIterableDataset[SupervisedNodeClassificationSample],
        ]
        if config.should_loop:
            iterable_training_dataset = LoopyIterableDataset(
                iterable_dataset=_iterable_training_dataset
            )
        else:
            iterable_training_dataset = _iterable_training_dataset

        collate_fn = partial(
            SupervisedNodeClassificationBatch.collate_pyg_node_classification_minibatch,
            graph_builder,
            gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
        )

        return torch.utils.data.DataLoader(
            iterable_training_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=config.pin_memory,
        )
