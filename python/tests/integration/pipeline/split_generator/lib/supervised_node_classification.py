from dataclasses import dataclass
from enum import Enum
from typing import Generic, List, Optional, Set

from gigl.common.logger import Logger
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder, TGraph
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.model import GraphBackend
from gigl.src.common.types.pb_wrappers.graph_data_types import NodePbWrapper
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from snapchat.research.gbml import training_samples_schema_pb2

logger = Logger()


class NodeClassificationSettingType(Enum):
    TRANSDUCTIVE = 1
    INDUCTIVE = 2


@dataclass
class NodeClassificationSplitData(Generic[TGraph]):
    graph: TGraph
    labeled_nodes: Set[NodePbWrapper]


def build_single_data_split_subgraph_from_dataset_samples(
    split_samples: Optional[
        List[training_samples_schema_pb2.SupervisedNodeClassificationSample]
    ],
    graph_metadata_wrapper: GraphMetadataPbWrapper,
    graph_builder: GraphBuilder,
) -> NodeClassificationSplitData:
    """
    Build a NodeClassificationSplitData object encompassing nodes/edges from split_samples,
    which is a list of samples from a particular split (train, val, test).
    :param split_samples:
    :param graph_metadata_wrapper:
    :return:
    """
    labeled_nodes = set()

    if split_samples:
        for sample_pb in split_samples:
            # Accumulate labeled nodes
            if sample_pb.root_node_labels:
                labeled_nodes.add(NodePbWrapper(pb=sample_pb.root_node))

            # Accumulate nodes and edges
            graph_data = GbmlProtosTranslator.graph_data_from_GraphPb(
                samples=[sample_pb.neighborhood],
                graph_metadata_pb_wrapper=GraphMetadataPbWrapper(
                    graph_metadata_wrapper.graph_metadata_pb
                ),
                builder=GraphBuilderFactory.get_graph_builder(
                    backend_name=GraphBackend.PYG
                ),
            )
            graph_builder.add_graph_data(graph_data=graph_data)

    graph = graph_builder.build()
    return NodeClassificationSplitData(graph=graph, labeled_nodes=labeled_nodes)


def log_node_classification_split_details(
    train_split: NodeClassificationSplitData,
    val_split: NodeClassificationSplitData,
    test_split: NodeClassificationSplitData,
):
    """
    Log some high-level metrics about each train/val/test NodeClassificationSplitData.
    :param train_split:
    :param val_split:
    :param test_split:
    :return:
    """
    logger.info(
        f"Train split: {train_split.graph.num_nodes} nodes "  # type: ignore
        f"({len(train_split.labeled_nodes)} labeled), "
        f"{train_split.graph.num_edges} edges."  # type: ignore
    )
    logger.info(
        f"Val split: {val_split.graph.num_nodes} nodes "  # type: ignore
        f"({len(val_split.labeled_nodes)} labeled), "
        f"{val_split.graph.num_edges} edges."  # type: ignore
    )
    logger.info(
        f"Test split: {test_split.graph.num_nodes} nodes "  # type: ignore
        f"({len(test_split.labeled_nodes)} labeled), "
        f"{test_split.graph.num_edges} edges."  # type: ignore
    )
