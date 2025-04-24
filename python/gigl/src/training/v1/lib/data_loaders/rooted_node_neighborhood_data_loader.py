from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Set, Union

import torch
import torch_geometric.data

from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.translators.training_samples_protos_translator import (
    RootedNodeNeighborhoodSample,
    TrainingSamplesProtosTranslator,
)
from gigl.src.common.types.graph_data import CondensedNodeType, Node, NodeId, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.training.v1.lib.data_loaders.common import DataloaderConfig
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    CombinedIterableDatasets,
    LoopyIterableDataset,
    TfRecordsIterableDataset,
)
from gigl.src.training.v1.lib.data_loaders.utils import cast_graph_for_training
from snapchat.research.gbml import graph_schema_pb2, training_samples_schema_pb2


@dataclass
class RootedNodeNeighborhoodBatch:
    graph: Union[
        torch_geometric.data.Data, torch_geometric.data.hetero_data.HeteroData
    ]  # batch-coalesced graph data used for message passing
    condensed_node_type_to_root_node_indices_map: Dict[
        CondensedNodeType, torch.LongTensor
    ]  # maps condensed node type to root node indices within the batch for whom to compute loss
    root_nodes: List[Node]
    condensed_node_type_to_subgraph_id_to_global_node_id: Dict[
        CondensedNodeType, Dict[NodeId, NodeId]
    ]  # for each condensed node type, maps subgraph node id to global node id

    @staticmethod
    def preprocess_rooted_node_neighborhood_sample_fn(
        sample_pb: training_samples_schema_pb2.RootedNodeNeighborhood,
        builder: GraphBuilder,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> RootedNodeNeighborhoodSample:
        samples = TrainingSamplesProtosTranslator.training_samples_from_RootedNodeNeighborhoodPb(
            samples=[sample_pb],
            graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            builder=builder,
        )
        sample = samples[0]  # Only 1 element since we are only processing 1 thing
        return sample

    @staticmethod
    def preprocess_rooted_node_neighborhood_raw_sample_fn(
        raw_data: bytes,
        builder: GraphBuilder,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    ) -> RootedNodeNeighborhoodSample:
        sample_pb = training_samples_schema_pb2.RootedNodeNeighborhood()
        sample_pb.ParseFromString(raw_data)
        return (
            RootedNodeNeighborhoodBatch.preprocess_rooted_node_neighborhood_sample_fn(
                sample_pb=sample_pb,
                builder=builder,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            )
        )

    @staticmethod
    def collate_pyg_rooted_node_neighborhood_minibatch(
        builder: GraphBuilder,
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
        samples: List[Dict[NodeType, RootedNodeNeighborhoodSample]],
    ) -> RootedNodeNeighborhoodBatch:
        """
        We coalesce the various sample subgraphs to build a single unified neighborhood, which we use for message
        passing.  By coalescing, overlaps between multiple samples' subgraphs will be handled gracefully, and
        we will only conduct message passing over these edges once.  If we do not coalesce, an edge e which appears in
        k samples' subgraphs would result in a k-factor duplication of edges, edge features and messages. Likewise, a
        node n which appears in k samples' subgraphs would result in a k-factor duplication of node features.
        :param samples:
        :return:
        """
        # TODO (mkolodner-sc) Investigate ways to customize batch size for each node type
        ordered_root_nodes: List[Node] = []
        unique_node_types: Set[NodeType] = set()
        for node_type_to_sample_map in samples:
            for node_type, sample in node_type_to_sample_map.items():
                ordered_root_nodes.append(sample.root_node)
                unique_node_types.add(node_type)
                graph_data = sample.subgraph
                if not isinstance(graph_data, PygGraphData):
                    raise NotImplementedError(
                        f"Subgraph must be of type {PygGraphData.__name__}:"
                        f"instead found type {type(graph_data)}."
                    )
                builder.add_graph_data(graph_data=graph_data)
        batch_graph_data = builder.build()

        node_mapping: Dict[
            Node, Node
        ] = batch_graph_data.global_node_to_subgraph_node_mapping

        condensed_node_type_to_subgraph_id_to_global_node_id: Dict[
            CondensedNodeType, Dict[NodeId, NodeId]
        ] = defaultdict(dict)
        for node_with_global_id, node_with_subgraph_id in node_mapping.items():
            condensed_node_type: CondensedNodeType = (
                graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                    node_with_global_id.type
                ]
            )
            condensed_node_type_to_subgraph_id_to_global_node_id[condensed_node_type][
                node_with_subgraph_id.id
            ] = node_with_global_id.id

        # We separate root node indices based on the node type of the root node
        condensed_node_type_to_root_node_indices_map: Dict[
            CondensedNodeType, torch.LongTensor
        ] = {}
        for node_type in unique_node_types:
            root_node_indices_list: List[NodeId] = [
                node_mapping[ordered_root_node].id
                for ordered_root_node in ordered_root_nodes
                if ordered_root_node.type == node_type
            ]
            condensed_node_type = (
                graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                    node_type
                ]
            )
            condensed_node_type_to_root_node_indices_map[
                condensed_node_type
            ] = torch.LongTensor(root_node_indices_list)
        batch_graph_data.coalesce()
        batch_graph_data = cast_graph_for_training(
            batch_graph_data=batch_graph_data,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
            batch_type=RootedNodeNeighborhoodBatch.__name__,
            should_register_edge_features=builder.should_register_edge_features,
        )

        return RootedNodeNeighborhoodBatch(
            condensed_node_type_to_root_node_indices_map=condensed_node_type_to_root_node_indices_map,
            root_nodes=ordered_root_nodes,
            graph=batch_graph_data,
            condensed_node_type_to_subgraph_id_to_global_node_id=condensed_node_type_to_subgraph_id_to_global_node_id,
        )

    @staticmethod
    def process_raw_pyg_samples_and_collate_fn(
        batch: List[training_samples_schema_pb2.RootedNodeNeighborhood],
        builder: GraphBuilder,
        graph_metadata_pb_wrapper: GraphMetadataPbWrapper,
        preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
    ) -> RootedNodeNeighborhoodBatch:
        ordered_root_nodes: List[Node] = []
        unique_node_types: Set[NodeType] = set()
        graph_samples: List[graph_schema_pb2.Graph] = []

        for sample in batch:
            root_node = Node(
                type=graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                    CondensedNodeType(sample.root_node.condensed_node_type)
                ],
                id=NodeId(sample.root_node.node_id),
            )
            ordered_root_nodes.append(root_node)
            unique_node_types.add(root_node.type)
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

        condensed_node_type_to_subgraph_id_to_global_node_id: Dict[
            CondensedNodeType, Dict[NodeId, NodeId]
        ] = defaultdict(dict)

        for node_with_global_id, node_with_subgraph_id in node_mapping.items():
            condensed_node_type: CondensedNodeType = (
                graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                    node_with_global_id.type
                ]
            )
            condensed_node_type_to_subgraph_id_to_global_node_id[condensed_node_type][
                node_with_subgraph_id.id
            ] = node_with_global_id.id

        condensed_node_type_to_root_node_indices_map: Dict[
            CondensedNodeType, torch.LongTensor
        ] = {
            graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                node_type
            ]: torch.LongTensor(
                [
                    node_mapping[root_node].id
                    for root_node in ordered_root_nodes
                    if root_node.type == node_type
                ]
            )
            for node_type in unique_node_types
        }

        batch_graph_data.coalesce()
        batch_graph_data = cast_graph_for_training(
            batch_graph_data=batch_graph_data,
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
            batch_type=RootedNodeNeighborhoodBatch.__name__,
            should_register_edge_features=builder.should_register_edge_features,
        )

        return RootedNodeNeighborhoodBatch(
            graph=batch_graph_data,
            condensed_node_type_to_root_node_indices_map=condensed_node_type_to_root_node_indices_map,
            root_nodes=ordered_root_nodes,
            condensed_node_type_to_subgraph_id_to_global_node_id=dict(
                condensed_node_type_to_subgraph_id_to_global_node_id
            ),
        )

    @staticmethod
    def get_default_data_loader(
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_builder: GraphBuilder,
        config: DataloaderConfig,
    ) -> torch.utils.data.DataLoader:
        """
        We often want to set should_loop = True because we want to be able to fetch random negatives on demand
        for each main-sample batch without worrying about this DataLoader "running out" of data. If this
        dataset were not "loopy", then we could run into a scenario where e.g. the main-sample dataloader
        has 20 batches, but the random-negative dataloader only has 10.  This pacing issue would cause us to
        not be able to fetch random negatives for the last 10 main-sample batches, undesirably.
        """
        iterable_dataset_map: Dict[
            NodeType,
            torch.utils.data.IterableDataset[RootedNodeNeighborhoodSample],
        ] = {}
        assert isinstance(config.uris, dict)
        for node_type, tf_record_uris in config.uris.items():
            _iterable_dataset: torch.utils.data.IterableDataset[
                RootedNodeNeighborhoodSample
            ]
            preprocess_raw_sample_fn = partial(
                RootedNodeNeighborhoodBatch.preprocess_rooted_node_neighborhood_raw_sample_fn,
                builder=graph_builder,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            )
            _iterable_dataset = TfRecordsIterableDataset(
                tf_record_uris=tf_record_uris,
                process_raw_sample_fn=preprocess_raw_sample_fn,
                seed=config.seed,
            )

            if config.should_loop:
                _iterable_dataset = LoopyIterableDataset(
                    iterable_dataset=_iterable_dataset
                )
            iterable_dataset_map[node_type] = _iterable_dataset

        iterable_training_dataset: CombinedIterableDatasets[
            RootedNodeNeighborhoodSample
        ] = CombinedIterableDatasets(
            iterable_dataset_map=iterable_dataset_map  # type: ignore
        )

        collate_fn = partial(
            RootedNodeNeighborhoodBatch.collate_pyg_rooted_node_neighborhood_minibatch,
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
