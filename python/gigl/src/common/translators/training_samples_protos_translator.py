from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch

from gigl.common.logger import Logger
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.gbml_graph_protocol import GbmlGraphDataProtocol
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.graph_data import CondensedEdgeType, Edge, Node, NodeId
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from snapchat.research.gbml import training_samples_schema_pb2

logger = Logger()


# TODO: (svij-sc) replace with SupervisedNodeClassificationSampleWrapper instead
class SupervisedNodeClassificationSample(NamedTuple):
    x: GbmlGraphDataProtocol  # TODO(nshah-sc): rename to subgraph to clarify this is a graph object, not features.
    root_node: Node
    y: List[training_samples_schema_pb2.Label]


# TODO: (mkolodner-sc) Rename due to overlapping name with training_samples_schema_proto message
@dataclass
class NodeAnchorBasedLinkPredictionSample:
    @dataclass
    class SampleSupervisionEdgeData:
        pos_nodes: List[NodeId]  # target nodes for pos edges
        hard_neg_nodes: List[NodeId]  # target nodes for hard neg edges
        pos_edge_features: Optional[torch.FloatTensor]  # features for pos edges
        hard_neg_edge_features: Optional[
            torch.FloatTensor
        ]  # features for hard neg edges

    root_node: Node  # root node for this sample
    subgraph: GbmlGraphDataProtocol  # subgraph with features used for message passing
    # mapping of edge type to positive and negative nodes and edge features
    condensed_edge_type_to_supervision_edge_data: Dict[
        CondensedEdgeType, SampleSupervisionEdgeData
    ]


class RootedNodeNeighborhoodSample(NamedTuple):
    root_node: Node  # root node for this sample
    subgraph: GbmlGraphDataProtocol  # subgraph with features used for message passing


class TrainingSamplesProtosTranslator:
    @staticmethod
    def training_samples_from_SupervisedNodeClassificationSamplePb(
        samples: List[training_samples_schema_pb2.SupervisedNodeClassificationSample],
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        builder: GraphBuilder,
    ) -> List[SupervisedNodeClassificationSample]:
        training_classification_samples: List[SupervisedNodeClassificationSample] = []
        for sample in samples:
            graph_data: GbmlGraphDataProtocol = (
                GbmlProtosTranslator.graph_data_from_GraphPb(
                    samples=[sample.neighborhood],
                    graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
                    builder=builder,
                )
            )
            root_node, _ = GbmlProtosTranslator.node_from_NodePb(
                node_pb=sample.root_node,
                graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            )
            labels = [label for label in sample.root_node_labels]
            training_classification_samples.append(
                SupervisedNodeClassificationSample(
                    x=graph_data, root_node=root_node, y=labels
                )
            )
        return training_classification_samples

    @staticmethod
    def training_samples_from_NodeAnchorBasedLinkPredictionSamplePb(
        samples: List[training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample],
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
        builder: GraphBuilder,
    ) -> List[NodeAnchorBasedLinkPredictionSample]:
        training_samples: List[NodeAnchorBasedLinkPredictionSample] = []
        for sample in samples:
            condensed_supervision_edge_type_to_pos_nodes: Dict[
                CondensedEdgeType, List[NodeId]
            ] = defaultdict(list)
            condensed_supervision_edge_type_to_hard_neg_nodes: Dict[
                CondensedEdgeType, List[NodeId]
            ] = defaultdict(list)
            condensed_supervision_edge_type_to_pos_edge_feats: Dict[
                CondensedEdgeType, List[torch.FloatTensor]
            ] = defaultdict(list)
            condensed_supervision_edge_type_to_hard_neg_edge_feats: Dict[
                CondensedEdgeType, List[torch.FloatTensor]
            ] = defaultdict(list)
            condensed_edge_type_to_supervision_edge_data: Dict[
                CondensedEdgeType,
                NodeAnchorBasedLinkPredictionSample.SampleSupervisionEdgeData,
            ] = {}
            graph_data: GbmlGraphDataProtocol = (
                GbmlProtosTranslator.graph_data_from_GraphPb(
                    samples=[sample.neighborhood],
                    graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
                    builder=builder,
                )
            )
            root_node, _ = GbmlProtosTranslator.node_from_NodePb(
                node_pb=sample.root_node,
                graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            )

            # TODO (tzhao-sc): this would allow the dataloader to load samples without any pos,
            #              which is meaningless for training and only useful for global metrics
            #              like AUC in validation and testing. TBD whether we want to allow
            #              this or filter those out in Split Generator.

            for pos_edge_pb in sample.pos_edges:
                pos_edge: Tuple[
                    Edge, Optional[torch.Tensor]
                ] = GbmlProtosTranslator.edge_from_EdgePb(
                    graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
                    edge_pb=pos_edge_pb,
                )
                node_id = pos_edge[0].dst_node.id
                condensed_edge_type = (
                    graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[
                        pos_edge[0].edge_type
                    ]
                )
                condensed_supervision_edge_type_to_pos_nodes[
                    condensed_edge_type
                ].append(node_id)
                if preprocessed_metadata_pb_wrapper.has_pos_edge_features(
                    condensed_edge_type
                ):
                    condensed_supervision_edge_type_to_pos_edge_feats[
                        condensed_edge_type
                    ].append(
                        pos_edge[1]  # type: ignore
                    )

            for hard_neg_edge_pb in sample.hard_neg_edges:
                hard_neg_edge: Tuple[
                    Edge, Optional[torch.Tensor]
                ] = GbmlProtosTranslator.edge_from_EdgePb(
                    graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
                    edge_pb=hard_neg_edge_pb,
                )
                node_id = hard_neg_edge[0].dst_node.id
                condensed_edge_type = (
                    graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[
                        hard_neg_edge[0].edge_type
                    ]
                )
                condensed_supervision_edge_type_to_hard_neg_nodes[
                    condensed_edge_type
                ].append(node_id)

                if preprocessed_metadata_pb_wrapper.has_hard_neg_edge_features(
                    condensed_edge_type
                ):
                    condensed_supervision_edge_type_to_hard_neg_edge_feats[
                        condensed_edge_type
                    ].append(
                        hard_neg_edge[1]  # type: ignore
                    )

            for condensed_edge_type in graph_metadata_pb_wrapper.condensed_edge_types:
                condensed_edge_type_to_supervision_edge_data[
                    condensed_edge_type
                ] = NodeAnchorBasedLinkPredictionSample.SampleSupervisionEdgeData(
                    pos_nodes=condensed_supervision_edge_type_to_pos_nodes[
                        condensed_edge_type
                    ],
                    hard_neg_nodes=condensed_supervision_edge_type_to_hard_neg_nodes[
                        condensed_edge_type
                    ],
                    pos_edge_features=(
                        torch.stack(  # type: ignore
                            condensed_supervision_edge_type_to_pos_edge_feats[  # type: ignore
                                condensed_edge_type
                            ]
                        )
                        if len(
                            condensed_supervision_edge_type_to_pos_edge_feats[
                                condensed_edge_type
                            ]
                        )
                        > 0
                        else None
                    ),
                    hard_neg_edge_features=(
                        torch.stack(  # type: ignore
                            condensed_supervision_edge_type_to_hard_neg_edge_feats[  # type: ignore
                                condensed_edge_type
                            ]
                        )
                        if len(
                            condensed_supervision_edge_type_to_hard_neg_edge_feats[
                                condensed_edge_type
                            ]
                        )
                        > 0
                        else None
                    ),
                )

            training_samples.append(
                NodeAnchorBasedLinkPredictionSample(
                    subgraph=graph_data,
                    root_node=root_node,
                    condensed_edge_type_to_supervision_edge_data=condensed_edge_type_to_supervision_edge_data,
                )
            )
        return training_samples

    @staticmethod
    def training_samples_from_RootedNodeNeighborhoodPb(
        samples: List[training_samples_schema_pb2.RootedNodeNeighborhood],
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        builder: GraphBuilder,
    ) -> List[RootedNodeNeighborhoodSample]:
        training_samples: List[RootedNodeNeighborhoodSample] = []
        for sample in samples:
            graph_data: GbmlGraphDataProtocol = (
                GbmlProtosTranslator.graph_data_from_GraphPb(
                    samples=[sample.neighborhood],
                    graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
                    builder=builder,
                )
            )
            root_node, _ = GbmlProtosTranslator.node_from_NodePb(
                node_pb=sample.root_node,
                graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            )
            training_samples.append(
                RootedNodeNeighborhoodSample(subgraph=graph_data, root_node=root_node)
            )
        return training_samples
