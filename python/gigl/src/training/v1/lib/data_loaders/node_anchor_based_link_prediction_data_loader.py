from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torch_geometric.data

from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.translators.training_samples_protos_translator import (
    NodeAnchorBasedLinkPredictionSample,
    TrainingSamplesProtosTranslator,
)
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    Node,
    NodeId,
)
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
from snapchat.research.gbml import training_samples_schema_pb2


@dataclass
class NodeAnchorBasedLinkPredictionBatch:
    @dataclass
    class BatchSupervisionEdgeData:
        root_node_to_target_node_id: Dict[NodeId, torch.LongTensor] = field(
            default_factory=dict
        )  # maps root nodes to target node for positive or negative edges
        label_edge_features: Optional[Dict[NodeId, torch.FloatTensor]] = field(
            default_factory=dict
        )  # maps root nodes to edge features for or negative positive edges

    graph: Union[
        torch_geometric.data.Data, torch_geometric.data.hetero_data.HeteroData
    ]  # batch-coalesced graph data used for message passing
    root_node_indices: (
        torch.LongTensor
    )  # lists root node indices within the batch for whom to compute loss
    pos_supervision_edge_data: Dict[CondensedEdgeType, BatchSupervisionEdgeData]
    hard_neg_supervision_edge_data: Dict[CondensedEdgeType, BatchSupervisionEdgeData]
    condensed_node_type_to_subgraph_id_to_global_node_id: Dict[
        CondensedNodeType, Dict[NodeId, NodeId]
    ]  # for each condensed node type, maps subgraph node id to global node id

    @staticmethod
    def preprocess_node_anchor_based_link_prediction_sample_fn(
        sample_pb: training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
        builder: GraphBuilder,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> NodeAnchorBasedLinkPredictionSample:
        samples = TrainingSamplesProtosTranslator.training_samples_from_NodeAnchorBasedLinkPredictionSamplePb(
            samples=[sample_pb],
            graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
            builder=builder,
        )
        sample = samples[0]  # Only 1 element since we are only processing 1 thing
        return sample

    @staticmethod
    def preprocess_node_anchor_based_link_prediction_raw_sample_fn(
        raw_data: bytes,
        builder: GraphBuilder,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> NodeAnchorBasedLinkPredictionSample:
        sample_pb = training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample()
        sample_pb.ParseFromString(raw_data)
        return NodeAnchorBasedLinkPredictionBatch.preprocess_node_anchor_based_link_prediction_sample_fn(
            sample_pb=sample_pb,
            builder=builder,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )

    @staticmethod
    def collate_pyg_node_anchor_based_link_prediction_minibatch(
        builder: GraphBuilder,
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
        samples: List[NodeAnchorBasedLinkPredictionSample],
    ) -> NodeAnchorBasedLinkPredictionBatch:
        """
        We coalesce the various sample subgraphs to build a single unified neighborhood, which we use for message
        passing.  Coalescing has a few notable properties:
         - By coalescing, overlaps between multiple samples' subgraphs will be handled gracefully, and we will only
           conduct message passing over these edges once.  If we do not coalesce, an edge e which appears in k samples'
            subgraphs would result in a k-factor duplication of edges, edge features and messages. Likewise, a node n
            which appears in k samples' subgraphs would result in a k-factor duplication of node features.
        - By coalescing, the batch may have a node connected to more than the number of sampled neighbors (k) specified
          in SubgraphSampler config.  This is because two samples may both reference the same node and each have k
          different sampled edges.  Hence, the union of those two samples would result in the node having 2k neighbors.
        :param samples:
        :return:
        """
        for sample in samples:
            graph_data = sample.subgraph
            if not isinstance(graph_data, PygGraphData):
                raise NotImplementedError(
                    f"Subgraph must be of type {PygGraphData.__name__}:"
                    f"instead found type {type(graph_data)}."
                )
            builder.add_graph_data(graph_data=graph_data)
        batch_graph_data = builder.build()

        _batch_root_nodes: List[NodeId] = list()
        pos_supervision_edge_data: Dict[
            CondensedEdgeType,
            NodeAnchorBasedLinkPredictionBatch.BatchSupervisionEdgeData,
        ] = defaultdict(NodeAnchorBasedLinkPredictionBatch.BatchSupervisionEdgeData)
        hard_neg_supervision_edge_data: Dict[
            CondensedEdgeType,
            NodeAnchorBasedLinkPredictionBatch.BatchSupervisionEdgeData,
        ] = defaultdict(NodeAnchorBasedLinkPredictionBatch.BatchSupervisionEdgeData)
        condensed_node_type_to_subgraph_id_to_global_node_id: Dict[
            CondensedNodeType, Dict[NodeId, NodeId]
        ] = defaultdict(dict)
        node_mapping: Dict[
            Node, Node
        ] = batch_graph_data.global_node_to_subgraph_node_mapping
        for node_with_global_id, node_with_subgraph_id in node_mapping.items():
            condensed_node_type: CondensedNodeType = (
                graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                    node_with_global_id.type
                ]
            )
            condensed_node_type_to_subgraph_id_to_global_node_id[condensed_node_type][
                node_with_subgraph_id.id
            ] = node_with_global_id.id

        for sample in samples:
            # Root node
            subgraph_root_node = node_mapping[sample.root_node].id
            _batch_root_nodes.append(subgraph_root_node)

            for (
                condensed_edge_type,
                condensed_edge_type_to_supervision_edge_data,
            ) in sample.condensed_edge_type_to_supervision_edge_data.items():
                node_type = (
                    graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                        condensed_edge_type
                    ].dst_node_type
                )
                # Map each root node to its positive nodes (dst nodes of the positive edges).
                _subgraph_pos_nodes: List[NodeId] = []
                for (
                    pos_node_id
                ) in condensed_edge_type_to_supervision_edge_data.pos_nodes:
                    pos_node = Node(type=node_type, id=pos_node_id)
                    _subgraph_pos_nodes.append(node_mapping[pos_node].id)
                pos_supervision_edge_data[
                    condensed_edge_type
                ].root_node_to_target_node_id[subgraph_root_node] = torch.LongTensor(
                    _subgraph_pos_nodes
                )

                # Map each root node to its hard negative edges (dst nodes of the hard negative edges).
                _subgraph_hard_neg_nodes: List[NodeId] = []
                for (
                    hard_neg_node_id
                ) in condensed_edge_type_to_supervision_edge_data.hard_neg_nodes:
                    hard_neg_node = Node(type=node_type, id=hard_neg_node_id)
                    _subgraph_hard_neg_nodes.append(node_mapping[hard_neg_node].id)
                hard_neg_supervision_edge_data[
                    condensed_edge_type
                ].root_node_to_target_node_id[subgraph_root_node] = torch.LongTensor(
                    _subgraph_hard_neg_nodes
                )

                # Map each root node to its positive edge features.
                if preprocessed_metadata_pb_wrapper.has_pos_edge_features(
                    condensed_edge_type=condensed_edge_type
                ):
                    pos_supervision_edge_data[condensed_edge_type].label_edge_features[subgraph_root_node] = condensed_edge_type_to_supervision_edge_data.pos_edge_features  # type: ignore
                else:
                    pos_supervision_edge_data[
                        condensed_edge_type
                    ].label_edge_features = None

                # Map each root node to its hard negative edge features.
                if preprocessed_metadata_pb_wrapper.has_hard_neg_edge_features(
                    condensed_edge_type=condensed_edge_type
                ):
                    hard_neg_supervision_edge_data[condensed_edge_type].label_edge_features[subgraph_root_node] = condensed_edge_type_to_supervision_edge_data.hard_neg_edge_features  # type: ignore
                else:
                    hard_neg_supervision_edge_data[
                        condensed_edge_type
                    ].label_edge_features = None

        batch_root_nodes = torch.LongTensor(_batch_root_nodes)
        batch_graph_data.coalesce()
        batch_graph_data = cast_graph_for_training(
            batch_graph_data=batch_graph_data,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
            batch_type=NodeAnchorBasedLinkPredictionBatch.__name__,
            should_register_edge_features=builder.should_register_edge_features,
        )

        return NodeAnchorBasedLinkPredictionBatch(
            root_node_indices=batch_root_nodes,
            graph=batch_graph_data,
            pos_supervision_edge_data=pos_supervision_edge_data,
            hard_neg_supervision_edge_data=hard_neg_supervision_edge_data,
            condensed_node_type_to_subgraph_id_to_global_node_id=condensed_node_type_to_subgraph_id_to_global_node_id,
        )

    @staticmethod
    def get_default_data_loader(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_builder: GraphBuilder,
        config: DataloaderConfig,
    ) -> torch.utils.data.DataLoader:
        preprocess_raw_sample_fn = partial(
            NodeAnchorBasedLinkPredictionBatch.preprocess_node_anchor_based_link_prediction_raw_sample_fn,
            builder=graph_builder,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        )
        _iterable_training_dataset = TfRecordsIterableDataset(
            tf_record_uris=config.uris,  # type: ignore
            process_raw_sample_fn=preprocess_raw_sample_fn,
            seed=config.seed,
        )
        iterable_training_dataset: Union[
            LoopyIterableDataset[NodeAnchorBasedLinkPredictionSample],
            TfRecordsIterableDataset[NodeAnchorBasedLinkPredictionSample],
        ]
        if config.should_loop:
            iterable_training_dataset = LoopyIterableDataset(
                iterable_dataset=_iterable_training_dataset
            )
        else:
            iterable_training_dataset = _iterable_training_dataset

        collate_fn = partial(
            NodeAnchorBasedLinkPredictionBatch.collate_pyg_node_anchor_based_link_prediction_minibatch,
            graph_builder,
            gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
        )

        return torch.utils.data.DataLoader(
            iterable_training_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=collate_fn,  # type: ignore
            persistent_workers=False,
            pin_memory=config.pin_memory,
        )
