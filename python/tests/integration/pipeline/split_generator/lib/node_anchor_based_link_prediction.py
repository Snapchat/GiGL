from dataclasses import dataclass
from typing import Generic, List, Optional, Set

from gigl.common.logger import Logger
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder, TGraph
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.model import GraphBackend
from gigl.src.common.types.pb_wrappers.graph_data_types import EdgePbWrapper
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from snapchat.research.gbml import training_samples_schema_pb2

logger = Logger()


@dataclass
class NodeAnchorBasedLinkPredictionSplitData(Generic[TGraph]):
    graph: TGraph
    pos_edges: Set[EdgePbWrapper]
    hard_neg_edges: Set[EdgePbWrapper]


def build_single_data_split_subgraph_from_samples(
    split_main_samples: Optional[
        List[training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample]
    ],
    split_random_negatives: Optional[
        List[training_samples_schema_pb2.RootedNodeNeighborhood]
    ],
    graph_metadata_wrapper: GraphMetadataPbWrapper,
    graph_builder: GraphBuilder,
) -> NodeAnchorBasedLinkPredictionSplitData:
    """
    Build a NodeAnchorBasedLinkPredictionSplitData object encompassing nodes/edges from
    split_main_samples and split_random_negatives, which correspond to main and random negative samples.
    :param split_random_negatives:
    :param split_main_samples:
    :param graph_metadata_wrapper:
    :return:
    """

    pos_edges: Set[EdgePbWrapper] = set()
    hard_neg_edges: Set[EdgePbWrapper] = set()

    # accumulate graph data from all the split's main and random samples.
    if split_main_samples:
        # Parse info about message passing graph, pos_edges and hard_neg_edges from main samples.
        for unsup_node_anchor_based_link_pred_sample_pb in split_main_samples:
            # Accumulate (featureless) pos edges
            for edge_pb in unsup_node_anchor_based_link_pred_sample_pb.pos_edges:
                pos_edges.add(EdgePbWrapper(pb=edge_pb).dehydrate())

            # Accumulate (featureless) hard_neg edges
            for edge_pb in unsup_node_anchor_based_link_pred_sample_pb.hard_neg_edges:
                hard_neg_edges.add(EdgePbWrapper(pb=edge_pb).dehydrate())

            # Accumulate nodes and edges
            split_main_graph = GbmlProtosTranslator.graph_data_from_GraphPb(
                samples=[unsup_node_anchor_based_link_pred_sample_pb.neighborhood],
                graph_metadata_pb_wrapper=GraphMetadataPbWrapper(
                    graph_metadata_wrapper.graph_metadata_pb
                ),
                builder=GraphBuilderFactory.get_graph_builder(
                    backend_name=GraphBackend.PYG
                ),
            )
            graph_builder.add_graph_data(graph_data=split_main_graph)

    if split_random_negatives:
        # Parse info about message passing graph from random negatives.
        for rooted_neighborhood_sample_pb in split_random_negatives:
            # Accumulate nodes and edges
            split_random_graph = GbmlProtosTranslator.graph_data_from_GraphPb(
                samples=[rooted_neighborhood_sample_pb.neighborhood],
                graph_metadata_pb_wrapper=GraphMetadataPbWrapper(
                    graph_metadata_wrapper.graph_metadata_pb
                ),
                builder=GraphBuilderFactory.get_graph_builder(
                    backend_name=GraphBackend.PYG
                ),
            )
            graph_builder.add_graph_data(graph_data=split_random_graph)

    graph = graph_builder.build()

    # At this point, `graph` contains all the message passing edges visible in both main samples and random negatives.
    return NodeAnchorBasedLinkPredictionSplitData(
        graph=graph, pos_edges=pos_edges, hard_neg_edges=hard_neg_edges
    )


def log_node_anchor_based_link_prediction_split_details(
    train_split: NodeAnchorBasedLinkPredictionSplitData,
    val_split: NodeAnchorBasedLinkPredictionSplitData,
    test_split: NodeAnchorBasedLinkPredictionSplitData,
):
    """
    Log some high-level metrics about each train/val/test NodeAnchorBasedLinkPredictionSplitData.
    :param train_split:
    :param val_split:
    :param test_split:
    :return:
    """
    logger.info(
        f"Train split: {train_split.graph.num_nodes} nodes, "  # type: ignore
        f"{train_split.graph.num_edges} edges used in message passing."  # type: ignore
        f" ( {len(train_split.pos_edges)} + supervision edges, "
        f"{len(train_split.hard_neg_edges)} - supervision edges )"
    )
    logger.info(
        f"Val split: {val_split.graph.num_nodes} nodes, "  # type: ignore
        f"{val_split.graph.num_edges} edges used in message passing."  # type: ignore
        f" ( {len(val_split.pos_edges)} + supervision edges, "
        f"{len(val_split.hard_neg_edges)} - supervision edges )"
    )
    logger.info(
        f"Test split: {test_split.graph.num_nodes} nodes, "  # type: ignore
        f"{test_split.graph.num_edges} edges used in message passing."  # type: ignore
        f" ({len(test_split.pos_edges)} + supervision edges, "
        f"{len(test_split.hard_neg_edges)} - supervision edges )"
    )
