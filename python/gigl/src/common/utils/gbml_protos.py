from typing import List, Optional, Tuple

from gigl.src.common.types.pb_wrappers.graph_data_types import (
    EdgePbWrapper,
    GraphPbWrapper,
    NodePbWrapper,
)
from snapchat.research.gbml import training_samples_schema_pb2


class TrainingSamplesSchemaProtoUtils:
    @staticmethod
    def build_NodeAnchorBasedLinkPredictionSamplePb(
        target_node: NodePbWrapper,
        target_neighborhood: GraphPbWrapper,
        pos_neighborhoods: List[Tuple[EdgePbWrapper, GraphPbWrapper]],
        hard_neg_neighborhoods: Optional[
            List[Tuple[EdgePbWrapper, GraphPbWrapper]]
        ] = None,
        random_neg_neighborhoods: Optional[
            List[Tuple[EdgePbWrapper, GraphPbWrapper]]
        ] = None,
    ) -> training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample:
        training_sample = (
            training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample(
                root_node=target_node.pb,
            )
        )

        neighborhoods = [target_neighborhood]
        for pos_sample, pos_neighborhood in pos_neighborhoods:
            training_sample.pos_edges.append(pos_sample.pb)
            neighborhoods.append(pos_neighborhood)
        if hard_neg_neighborhoods:
            for hard_neg_sample, hard_neg_neighborhood in hard_neg_neighborhoods:
                training_sample.hard_neg_edges.append(hard_neg_sample.pb)
                neighborhoods.append(hard_neg_neighborhood)
        if random_neg_neighborhoods:
            for (
                random_neg_sample,
                random_neg_neighborhood,
            ) in random_neg_neighborhoods:
                training_sample.neg_edges.append(random_neg_sample.pb)
                neighborhoods.append(random_neg_neighborhood)

        merged_neighborhood = GraphPbWrapper.merge_subgraphs(subgraphs=neighborhoods)
        training_sample.neighborhood.CopyFrom(merged_neighborhood.pb)

        return training_sample

    @staticmethod
    def build_SupervisedNodeClassificationSamplePb(
        target_node: NodePbWrapper,
        neighborhood: GraphPbWrapper,
        node_labels: List[training_samples_schema_pb2.Label],
    ) -> training_samples_schema_pb2.SupervisedNodeClassificationSample:
        if node_labels:
            return training_samples_schema_pb2.SupervisedNodeClassificationSample(
                root_node=target_node.pb,
                neighborhood=neighborhood.pb,
                root_node_labels=node_labels,
            )
        else:
            return training_samples_schema_pb2.SupervisedNodeClassificationSample(
                root_node=target_node.pb,
                neighborhood=neighborhood.pb,
            )

    @staticmethod
    def build_SupervisedLinkBasedTaskSamplePb() -> (
        training_samples_schema_pb2.SupervisedLinkBasedTaskSample
    ):
        return NotImplemented

    @staticmethod
    def build_RootedNodeNeighborhoodPb(
        root_node: NodePbWrapper,
        neighborhood: GraphPbWrapper,
    ) -> training_samples_schema_pb2.RootedNodeNeighborhood:
        return training_samples_schema_pb2.RootedNodeNeighborhood(
            root_node=root_node.pb, neighborhood=neighborhood.pb
        )
