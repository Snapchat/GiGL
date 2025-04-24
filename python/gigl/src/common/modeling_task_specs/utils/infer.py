from collections import defaultdict
from typing import Dict, List, Set, Union

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data.hetero_data import HeteroData

from gigl.src.common.models.layers.decoder import LinkPredictionDecoder
from gigl.src.common.models.layers.loss import ModelResultType
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    NodeId,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.task_inputs import (
    BatchCombinedScores,
    BatchEmbeddings,
    BatchScores,
    InputBatch,
    NodeAnchorBasedLinkPredictionTaskInputs,
)
from gigl.src.training.v1.lib.data_loaders.node_anchor_based_link_prediction_data_loader import (
    NodeAnchorBasedLinkPredictionBatch,
)
from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)

# TODO (mkolodner-sc) Move PyG Logic to PyG-specific location


def infer_training_batch(
    model: Union[torch.nn.parallel.DistributedDataParallel, nn.Module],
    training_batch: Union[
        NodeAnchorBasedLinkPredictionBatch,
        RootedNodeNeighborhoodBatch,
        Data,
        HeteroData,
    ],
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    device: torch.device,
) -> Dict[CondensedNodeType, torch.Tensor]:
    # Compute embeddings for all nodes in the main and random batches.
    if isinstance(training_batch, NodeAnchorBasedLinkPredictionBatch) or isinstance(
        training_batch, RootedNodeNeighborhoodBatch
    ):
        training_batch = training_batch.graph

    node_type_to_condensed_node_type_map = (
        gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map
    )
    supervision_node_types = (
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_node_types(
            should_include_src_nodes=True,
            should_include_dst_nodes=True,
        )
    )
    output_node_types = [NodeType(node_type) for node_type in supervision_node_types]

    training_batch = training_batch.to(device=device)
    node_type_to_embeddings: Dict[NodeType, torch.Tensor] = model(
        data=training_batch, output_node_types=output_node_types, device=device
    )
    return {
        node_type_to_condensed_node_type_map[node_type]: node_type_to_embeddings[
            node_type
        ]
        for node_type in node_type_to_embeddings
    }


def infer_root_embeddings(
    model: Union[torch.nn.parallel.DistributedDataParallel, nn.Module],
    graph: Union[Data, HeteroData],
    root_node_indices: torch.LongTensor,
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    device: torch.device,
) -> torch.Tensor:
    batch_graph = graph.to(device=device)
    batch_root_node_indices = root_node_indices.to(device=device)
    output_node_types = list(
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_node_types(
            should_include_src_nodes=True,
            should_include_dst_nodes=False,
        )
    )
    # TODO (mkolodner) Add support for multiple root_node_indices node types in Stage 3 HGS
    if len(output_node_types) != 1:
        raise NotImplementedError(
            "Stage 3 HGS is not yet supported -- training can only be performed with one unique source node type."
        )
    node_type_to_embeddings: Dict[NodeType, torch.Tensor] = model(
        data=batch_graph, output_node_types=output_node_types, device=device
    )
    out = node_type_to_embeddings[output_node_types[0]]
    embed = out[batch_root_node_indices]
    return embed


def infer_task_inputs(
    model: Union[torch.nn.parallel.DistributedDataParallel, nn.Module],
    gbml_config_pb_wrapper: GbmlConfigPbWrapper,
    main_batch: NodeAnchorBasedLinkPredictionBatch,
    random_neg_batch: RootedNodeNeighborhoodBatch,
    should_eval: bool,
    device: torch.device,
) -> NodeAnchorBasedLinkPredictionTaskInputs:
    # Initializing empty container values
    batch_scores: List[Dict[CondensedEdgeType, BatchScores]] = []
    batch_combined_scores: Dict[CondensedEdgeType, BatchCombinedScores] = {}

    pos_embeddings: Dict[CondensedEdgeType, torch.FloatTensor] = {}
    hard_neg_embeddings: Dict[CondensedEdgeType, torch.FloatTensor] = {}
    repeated_anchor_embeddings: Dict[CondensedEdgeType, torch.FloatTensor] = {}

    _pos_embeddings: Dict[CondensedEdgeType, List[torch.FloatTensor]] = defaultdict(
        list
    )
    _hard_neg_embeddings: Dict[
        CondensedEdgeType, List[torch.FloatTensor]
    ] = defaultdict(list)

    _positive_ids: Dict[CondensedEdgeType, List[torch.LongTensor]] = defaultdict(list)
    _hard_neg_ids: Dict[CondensedEdgeType, List[torch.LongTensor]] = defaultdict(list)

    # Map of Condensed Edge Type to list of num_pos_nodes for retrieval calculation
    repeated_anchor_count: Dict[CondensedEdgeType, List[int]] = defaultdict(list)

    # Populate main_batch and RNN task inputs field
    input_batch = InputBatch(main_batch=main_batch, random_neg_batch=random_neg_batch)

    batch_result_types: Set[ModelResultType]
    decoder: LinkPredictionDecoder
    # Unwrap any DDP layers
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        decoder = model.module.decode
        batch_result_types = model.module.tasks.result_types
    else:
        decoder = model.decode
        batch_result_types = model.tasks.result_types

    # If we only have losses which only require the input batch, don't forward here and return the
    # input batch immediately to minimize computation we don't need, such as encoding and decoding.
    should_forward_batch: bool = (
        ModelResultType.batch_scores in batch_result_types
        or ModelResultType.batch_embeddings in batch_result_types
        or ModelResultType.batch_combined_scores in batch_result_types
        or should_eval
    )

    if not should_forward_batch:
        return NodeAnchorBasedLinkPredictionTaskInputs(
            input_batch=input_batch,
            batch_embeddings=None,
            batch_scores=batch_scores,
            batch_combined_scores=batch_combined_scores,
        )

    # Forward input batch through model

    main_embeddings: Dict[CondensedNodeType, torch.Tensor] = infer_training_batch(
        model=model,
        training_batch=main_batch,
        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        device=device,
    )
    random_neg_embeddings = infer_training_batch(
        model=model,
        training_batch=random_neg_batch,
        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
        device=device,
    )

    main_batch_node_id_mapping: Dict[
        CondensedNodeType, Dict[NodeId, NodeId]
    ] = main_batch.condensed_node_type_to_subgraph_id_to_global_node_id
    random_negative_batch_node_id_mapping: Dict[
        CondensedNodeType, Dict[NodeId, NodeId]
    ] = random_neg_batch.condensed_node_type_to_subgraph_id_to_global_node_id

    # Getting all condensed anchor node types for getting query embeddings
    anchor_node_types = list(
        gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_node_types(
            should_include_src_nodes=True,
            should_include_dst_nodes=False,
        )
    )
    condensed_anchor_node_types = [
        gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
            node_type
        ]
        for node_type in anchor_node_types
    ]
    # TODO (mkolodner-sc) Add support for multiple root_node_indices node types in Stage 3 HGS
    if len(condensed_anchor_node_types) != 1:
        raise NotImplementedError(
            "Stage 3 HGS is not yet supported -- training can only be performed with one unique source node type."
        )

    main_batch_root_node_indices = main_batch.root_node_indices.to(device=device)
    query_embeddings = main_embeddings[condensed_anchor_node_types[0]][
        main_batch_root_node_indices
    ]

    # Getting RNN Embeddings and Scores
    random_neg_root_embeddings: Dict[CondensedNodeType, torch.FloatTensor] = {}
    random_neg_scores: Dict[CondensedNodeType, torch.FloatTensor] = {}

    for (
        condensed_node_type
    ) in random_neg_batch.condensed_node_type_to_root_node_indices_map:
        random_neg_root_node_indices = (
            random_neg_batch.condensed_node_type_to_root_node_indices_map[
                condensed_node_type
            ].to(device=device)
        )
        random_neg_root_embeddings[condensed_node_type] = (
            random_neg_embeddings[condensed_node_type][random_neg_root_node_indices]  # type: ignore
            if random_neg_root_node_indices.numel()
            else torch.FloatTensor([]).to(device=device)
        )
        if ModelResultType.batch_scores in batch_result_types or should_eval:
            random_neg_scores[condensed_node_type] = (
                decoder(
                    query_embeddings, random_neg_root_embeddings[condensed_node_type]
                )
                if random_neg_root_embeddings[condensed_node_type].numel()
                else torch.FloatTensor([]).to(device=device)
            )

    # Loop through all root nodes and populate ids, embeddings, and scores per condensed edge type
    for root_node_idx, root_node in enumerate(main_batch_root_node_indices):
        root_node = torch.unsqueeze(root_node, 0)  # shape=[1]
        _batch_scores: Dict[CondensedEdgeType, BatchScores] = {}
        for (
            supervision_edge_type
        ) in (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
        ):
            condensed_supervision_edge_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[
                supervision_edge_type
            ]
            (
                condensed_anchor_node_type,
                condensed_supervision_target_node_type,
            ) = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_condensed_node_types[
                condensed_supervision_edge_type
            ]
            pos_nodes: torch.LongTensor = main_batch.pos_supervision_edge_data[
                condensed_supervision_edge_type
            ].root_node_to_target_node_id[
                root_node.item()
            ]  # shape=[num_pos_nodes]

            hard_neg_nodes: (
                torch.LongTensor
            ) = main_batch.hard_neg_supervision_edge_data[
                condensed_supervision_edge_type
            ].root_node_to_target_node_id[
                root_node.item()
            ]  # shape=[num_hard_neg_nodes]

            repeated_anchor_count[condensed_supervision_edge_type].append(
                pos_nodes.numel()
            )

            if pos_nodes.numel():
                _pos_embeddings[condensed_supervision_edge_type].append(main_embeddings[condensed_supervision_target_node_type][pos_nodes])  # type: ignore
                _positive_ids[condensed_supervision_edge_type].append(pos_nodes)

            if hard_neg_nodes.numel():
                _hard_neg_embeddings[condensed_supervision_edge_type].append(main_embeddings[condensed_supervision_target_node_type][hard_neg_nodes])  # type: ignore
                _hard_neg_ids[condensed_supervision_edge_type].append(hard_neg_nodes)

            # If any tasks need batch score information, decode embeddings into scores
            if ModelResultType.batch_scores in batch_result_types or should_eval:
                pos_scores = (
                    decoder(
                        main_embeddings[condensed_anchor_node_type][root_node],
                        main_embeddings[condensed_supervision_target_node_type][
                            pos_nodes
                        ],
                    )
                    if pos_nodes.numel()
                    else torch.FloatTensor([]).to(device=device)
                )
                hard_neg_scores = (
                    decoder(
                        main_embeddings[condensed_anchor_node_type][root_node],
                        main_embeddings[condensed_supervision_target_node_type][
                            hard_neg_nodes
                        ],
                    )
                    if hard_neg_nodes.numel()
                    else torch.FloatTensor([]).to(device=device)
                )
                random_neg_scores_root = random_neg_scores[
                    condensed_supervision_target_node_type
                ][[root_node_idx], :].to(device=device)
                _batch_scores[condensed_supervision_edge_type] = BatchScores(
                    pos_scores=pos_scores,
                    hard_neg_scores=hard_neg_scores,
                    random_neg_scores=random_neg_scores_root,  # type: ignore
                )

        if ModelResultType.batch_scores in batch_result_types or should_eval:
            batch_scores.append(_batch_scores)

    # Loop through all condensed edge types and collapse lists of same type into single tensor
    for (
        supervision_edge_type
    ) in gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types():
        condensed_supervision_edge_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[
            supervision_edge_type
        ]
        (
            condensed_anchor_node_type,
            condensed_supervision_target_node_type,
        ) = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_condensed_node_types[
            condensed_supervision_edge_type
        ]
        pos_embeddings[condensed_supervision_edge_type] = (
            torch.cat(tuple(_pos_embeddings[condensed_supervision_edge_type]))  # type: ignore
            if len(_pos_embeddings[condensed_supervision_edge_type])
            else torch.tensor([])
        )
        hard_neg_embeddings[condensed_supervision_edge_type] = (
            torch.cat(tuple(_hard_neg_embeddings[condensed_supervision_edge_type]))  # type: ignore
            if len(_hard_neg_embeddings[condensed_supervision_edge_type])
            else torch.tensor([])
        )

        repeated_anchor_embeddings[
            condensed_supervision_edge_type
        ] = query_embeddings.repeat_interleave(
            torch.tensor(repeated_anchor_count[condensed_supervision_edge_type]).to(device=device), dim=0  # type: ignore
        )

        # If needed, calculate task inputs for retrieval loss per condensed edge type
        if ModelResultType.batch_combined_scores in batch_result_types:
            candidate_embeddings = torch.cat(
                (
                    pos_embeddings[condensed_supervision_edge_type].to(device=device),
                    hard_neg_embeddings[condensed_supervision_edge_type].to(
                        device=device
                    ),
                    random_neg_root_embeddings[
                        condensed_supervision_target_node_type
                    ].to(device=device),
                )
            )

            repeated_subgraph_query_ids = (
                main_batch_root_node_indices.repeat_interleave(
                    torch.tensor(
                        repeated_anchor_count[condensed_supervision_edge_type]
                    ).to(device=device)
                )
            )

            repeated_global_query_ids = torch.tensor(
                [
                    main_batch_node_id_mapping[condensed_anchor_node_type][
                        node_id.item()
                    ]
                    for node_id in repeated_subgraph_query_ids
                ]
            ).to(device=device)

            subgraph_positive_ids = (
                torch.cat(tuple(_positive_ids[condensed_supervision_edge_type]))
                if len(_positive_ids[condensed_supervision_edge_type])
                else torch.tensor([])
            )

            global_positive_ids = torch.tensor(
                [
                    main_batch_node_id_mapping[condensed_supervision_target_node_type][
                        node_id.item()
                    ]
                    for node_id in subgraph_positive_ids
                ]
            ).to(device=device)

            subgraph_hard_neg_ids = (
                torch.cat(tuple(_hard_neg_ids[condensed_supervision_edge_type]))
                if len(_hard_neg_ids[condensed_supervision_edge_type])
                else torch.tensor([])
            )

            global_hard_neg_ids = torch.tensor(
                [
                    main_batch_node_id_mapping[condensed_supervision_target_node_type][
                        node_id.item()
                    ]
                    for node_id in subgraph_hard_neg_ids
                ]
            ).to(device=device)

            random_neg_root_node_indices = (
                random_neg_batch.condensed_node_type_to_root_node_indices_map[
                    condensed_supervision_target_node_type
                ].to(device=device)
            )

            subgraph_random_neg_ids = (
                random_neg_root_node_indices
                if random_neg_root_node_indices.numel()
                else torch.tensor([])
            )

            global_random_neg_ids = torch.tensor(
                [
                    random_negative_batch_node_id_mapping[
                        condensed_supervision_target_node_type
                    ][node_id.item()]
                    for node_id in subgraph_random_neg_ids
                ]
            ).to(device=device)

            repeated_candidate_scores = (
                decoder(
                    repeated_anchor_embeddings[condensed_supervision_edge_type],
                    candidate_embeddings,
                )
                if repeated_anchor_embeddings[condensed_supervision_edge_type].numel()
                else torch.tensor([])
            )

            batch_combined_scores[
                condensed_supervision_edge_type
            ] = BatchCombinedScores(
                repeated_candidate_scores=repeated_candidate_scores,
                positive_ids=global_positive_ids,  # type: ignore
                hard_neg_ids=global_hard_neg_ids,  # type: ignore
                random_neg_ids=global_random_neg_ids,  # type: ignore
                repeated_query_ids=repeated_global_query_ids,  # type: ignore
                num_unique_query_ids=main_batch_root_node_indices.shape[0],
            )

    # Populate all computed embeddings for task input
    batch_embeddings = BatchEmbeddings(
        query_embeddings=query_embeddings,  # type: ignore
        repeated_query_embeddings=repeated_anchor_embeddings,  # type: ignore
        pos_embeddings=pos_embeddings,  # type: ignore
        hard_neg_embeddings=hard_neg_embeddings,  # type: ignore
        random_neg_embeddings=random_neg_root_embeddings,  # type: ignore
    )

    return NodeAnchorBasedLinkPredictionTaskInputs(
        input_batch=input_batch,
        batch_embeddings=batch_embeddings,
        batch_scores=batch_scores,
        batch_combined_scores=batch_combined_scores,
    )
