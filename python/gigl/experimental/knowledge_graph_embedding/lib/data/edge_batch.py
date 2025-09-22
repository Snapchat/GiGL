from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
import torchrec

from gigl.experimental.knowledge_graph_embedding.common.graph_dataset import (
    CONDENSED_EDGE_TYPE_FIELD,
    DST_FIELD,
    SRC_FIELD,
    HeterogeneousGraphEdgeDict,
)
from gigl.experimental.knowledge_graph_embedding.common.torchrec.batch import (
    DataclassBatch,
)
from gigl.experimental.knowledge_graph_embedding.lib.config.dataloader import (
    DataloaderConfig,
)
from gigl.experimental.knowledge_graph_embedding.lib.config.sampling import (
    SamplingConfig,
)
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    LoopyIterableDataset,
)


@dataclass
class EdgeBatch(DataclassBatch):
    """
    A class for representing a batch of edges in a heterogeneous graph.
    This can be derived from input edge tensors, and contains logic to build
    a torchrec KeyedJaggedTensor (used for sharded embedding lookups) and
    other metadata tensors which are required to train KGE models.
    """

    src_dst_pairs: torchrec.KeyedJaggedTensor
    condensed_edge_types: torch.Tensor
    labels: torch.Tensor

    @staticmethod
    def from_edge_tensors(
        edges: torch.Tensor,
        condensed_edge_types: torch.Tensor,
        edge_labels: torch.Tensor,
        condensed_node_type_to_node_type_map: dict[CondensedNodeType, NodeType],
        condensed_edge_type_to_condensed_node_type_map: dict[
            CondensedEdgeType, tuple[CondensedNodeType, CondensedNodeType]
        ],
    ) -> EdgeBatch:
        """
        Creates an EdgeBatch from edge tensors.
        We create an EdgeBatch of len(2 * edges) by creating a src-dst pair for each edge in the batch.

        Args:
            edges (torch.Tensor): A tensor of edges.
            condensed_edge_types (torch.Tensor): A tensor of condensed edge types.
            edge_labels (torch.Tensor): A tensor of edge labels.
            condensed_node_type_to_node_type_map (dict[CondensedNodeType, NodeType]): A mapping from condensed node types to node types.
            condensed_edge_type_to_condensed_node_type_map (dict[CondensedEdgeType, tuple[CondensedNodeType, CondensedNodeType]]): A mapping from condensed edge types to condensed node types.
        """

        num_edges = edges.size(0)
        # We canonicalize the order of keys so all KJTs are constructed the same way.
        # This ensures that when they are processed by EmbeddingBagCollections, the outputs are consistently ordered.
        cnt_keys = sorted(list(condensed_node_type_to_node_type_map.keys()))
        lengths: dict[CondensedNodeType, list[int]] = {
            key: [0] * (2 * num_edges) for key in cnt_keys
        }
        values: dict[CondensedNodeType, list[int]] = {key: [] for key in cnt_keys}

        for i, (edge, condensed_edge_type) in enumerate(
            zip(edges, condensed_edge_types)
        ):
            src, dst = edge[0].item(), edge[1].item()
            src_cnt, dst_cnt = condensed_edge_type_to_condensed_node_type_map[
                condensed_edge_type.item()
            ]
            values[src_cnt].append(src)
            values[dst_cnt].append(dst)
            lengths[src_cnt][2 * i] = 1
            lengths[dst_cnt][2 * i + 1] = 1

        lengths_tensor: dict[CondensedNodeType, torch.Tensor] = dict()
        values_tensor: dict[CondensedNodeType, torch.Tensor] = dict()
        for key in cnt_keys:
            lengths_tensor[key] = torch.tensor(lengths[key], dtype=torch.int32)
            values_tensor[key] = torch.tensor(values[key], dtype=torch.int32)

        # Flatten tensors
        src_dst_pairs = torchrec.KeyedJaggedTensor(
            keys=cnt_keys,
            values=torch.cat([values_tensor[cnt] for cnt in cnt_keys], dim=0),
            lengths=torch.cat([lengths_tensor[cnt] for cnt in cnt_keys], dim=0),
        )

        return EdgeBatch(
            src_dst_pairs=src_dst_pairs,
            condensed_edge_types=condensed_edge_types,
            labels=edge_labels,
        )

    def to_edge_tensors(
        self,
        condensed_edge_type_to_condensed_node_type_map: dict[
            CondensedEdgeType, tuple[CondensedNodeType, CondensedNodeType]
        ],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reconstructs the edge tensors from the EdgeBatch.
        This is used for debugging and sanity checking the EdgeBatch.
        """

        # Get the edge tensors from the edge batch
        src_dst_pairs_kjt = self.src_dst_pairs
        condensed_edge_types = self.condensed_edge_types
        edge_labels = self.labels

        node_types = src_dst_pairs_kjt.keys()  # the unique node types
        num_node_types = len(
            node_types
        )  # the num of unique node types (= num of embedding tables)
        num_edges = int(
            len(src_dst_pairs_kjt.lengths()) / num_node_types / 2
        )  # len(lengths) == 2 * num_node_types * num_edges
        assert (
            num_edges == len(edge_labels) == len(condensed_edge_types)
        ), f"The number of edges, edge labels and edge types should be equal.  Got {num_edges, len(edge_labels), len(condensed_edge_types)}"

        reconstructed_edges = []
        src_dst_pairs_kjt_view = src_dst_pairs_kjt.lengths().view(
            num_node_types, num_edges * 2
        )

        condensed_node_types_for_edges = src_dst_pairs_kjt_view.argmax(dim=0).view(
            -1, 2
        )
        for condensed_edge_type, condensed_node_types_in_edges in zip(
            condensed_edge_types, condensed_node_types_for_edges
        ):
            (
                expected_src_cnt,
                expected_dst_cnt,
            ) = condensed_edge_type_to_condensed_node_type_map[
                condensed_edge_type.item()
            ]
            assert (
                condensed_node_types_in_edges[0].item() == expected_src_cnt
                and condensed_node_types_in_edges[1].item() == expected_dst_cnt
            ), f"Expected condensed node types for edge type {condensed_edge_type} to be {expected_src_cnt, expected_dst_cnt}, but got {condensed_node_types_in_edges}"

        condensed_node_types_for_edges = src_dst_pairs_kjt_view.argmax(dim=0).tolist()
        src_dst_pairs_values_iters = {
            node_type: iter(jagged.values())
            for node_type, jagged in src_dst_pairs_kjt.to_dict().items()
        }
        for condensed_node_type in condensed_node_types_for_edges:
            reconstructed_edges.append(
                next(src_dst_pairs_values_iters[condensed_node_type])
            )
        reconstructed_edges_tensor = torch.tensor(
            reconstructed_edges, dtype=torch.int32
        )
        reconstructed_edges_tensor = reconstructed_edges_tensor.view(-1, 2)

        return reconstructed_edges_tensor, condensed_edge_types, edge_labels

    @staticmethod
    def build_data_loader(
        dataset: torch.utils.data.IterableDataset,
        sampling_config: SamplingConfig,
        dataloader_config: DataloaderConfig,
        graph_metadata: GraphMetadataPbWrapper,
        condensed_node_type_to_vocab_size_map: dict[CondensedNodeType, int],
        pin_memory: bool,
        should_loop: bool = True,
    ):
        dataset = (
            LoopyIterableDataset(iterable_dataset=dataset) if should_loop else dataset
        )

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=sampling_config.positive_edge_batch_size,
            collate_fn=partial(
                collate_edge_batch_from_heterogeneous_graph_edge_dict,
                condensed_edge_type_to_condensed_node_type_map=graph_metadata.condensed_edge_type_to_condensed_node_types,
                condensed_node_type_to_vocab_size_map=condensed_node_type_to_vocab_size_map,
                condensed_node_type_to_node_type_map=graph_metadata.condensed_node_type_to_node_type_map,
                num_random_negatives_per_edge=sampling_config.num_random_negatives_per_edge,
            ),
            pin_memory=pin_memory,
            num_workers=dataloader_config.num_workers,
        )


def collate_edge_batch_from_heterogeneous_graph_edge_dict(
    inputs: list[HeterogeneousGraphEdgeDict],
    condensed_edge_type_to_condensed_node_type_map: dict[
        CondensedEdgeType, tuple[CondensedNodeType, CondensedNodeType]
    ],
    condensed_node_type_to_vocab_size_map: dict[CondensedNodeType, int],
    condensed_node_type_to_node_type_map: dict[CondensedNodeType, NodeType],
    num_random_negatives_per_edge: int = 0,
) -> EdgeBatch:
    """
    This is a collate function for the EdgeBatch.
    It takes a list of heterogeneous graph edge dictionaries (read from upstream dataset),
    converts them to tensors for "positive" edges, samples "negative" edges if applicable,
    and constructs an EdgeBatch (containing a TorchRec KeyedJaggedTensor and metadata).

    Args:
        inputs (list[HeterogeneousGraphEdgeDict]): The input data.
        condensed_edge_type_to_condensed_node_type_map (dict[CondensedEdgeType, tuple[CondensedNodeType, CondensedNodeType]]): A mapping from condensed edge types to condensed node types.
        condensed_node_type_to_vocab_size_map (dict[CondensedNodeType, int]): A mapping from condensed node types to vocab sizes.
        condensed_node_type_to_node_type_map (dict[CondensedNodeType, NodeType]): A mapping from condensed node types to node types.
        num_negative_samples_per_edge (int): The number of negative samples to generate for each positive edge.

    Returns:
        EdgeBatch: The collated EdgeBatch.
    """

    # Convert the input data to tensors
    pos_edges, pos_condensed_edge_types, pos_labels = build_tensors_from_edge_dicts(
        inputs
    )

    # Generative negative edges for the positive edges.
    if num_random_negatives_per_edge == 0:
        # If no negative samples are required, return the positive edges only.
        neg_edges = torch.empty((0, 2), dtype=torch.int32)
        neg_condensed_edge_types = torch.empty(0, dtype=torch.int32)
        neg_labels = torch.empty(0, dtype=torch.int32)
    else:
        (
            neg_edges,
            neg_condensed_edge_types,
            neg_labels,
        ) = relationwise_batch_random_negative_sampling(
            condensed_edge_type_to_condensed_node_type_map=condensed_edge_type_to_condensed_node_type_map,
            condensed_node_type_to_vocab_size_map=condensed_node_type_to_vocab_size_map,
            num_negatives_per_condensed_edge_type=num_random_negatives_per_edge,
        )

    # Construct the EdgeBatch which the model will consume.
    edge_batch = EdgeBatch.from_edge_tensors(
        edges=torch.vstack((pos_edges, neg_edges)),
        condensed_edge_types=torch.hstack(
            (pos_condensed_edge_types, neg_condensed_edge_types)
        ),
        edge_labels=torch.hstack((pos_labels, neg_labels)),
        condensed_node_type_to_node_type_map=condensed_node_type_to_node_type_map,
        condensed_edge_type_to_condensed_node_type_map=condensed_edge_type_to_condensed_node_type_map,
    )

    return edge_batch


def build_tensors_from_edge_dicts(
    inputs: list[HeterogeneousGraphEdgeDict],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a list of HeterogeneousGraphEdgeDict into tensors.

    Args:
        inputs (list[HeterogeneousGraphEdgeDict]): A list of edge dictionaries.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - edges (torch.Tensor): A tensor of shape [num_edges, 2] containing the source and destination node IDs.
            - condensed_edge_types (torch.Tensor): A tensor of shape [num_edges] containing the condensed edge types.
            - labels (torch.Tensor): A tensor of shape [num_edges] containing labels (all set to 1).
    """

    # Determine the number of edges
    num_edges = len(inputs)

    # Preallocate torch tensors
    edges = torch.empty((num_edges, 2), dtype=torch.int32)
    condensed_edge_types = torch.empty(num_edges, dtype=torch.int32)

    # Fill the preallocated torch tensors using direct indexing
    for i, row in enumerate(inputs):
        edges[i, 0] = int(row[SRC_FIELD])
        edges[i, 1] = int(row[DST_FIELD])
        condensed_edge_types[i] = int(row[CONDENSED_EDGE_TYPE_FIELD])

    # Create labels tensor directly
    labels = torch.ones(num_edges, dtype=torch.int32)
    return edges, condensed_edge_types, labels


def relationwise_batch_random_negative_sampling(
    condensed_edge_type_to_condensed_node_type_map: dict[
        CondensedEdgeType, tuple[CondensedNodeType, CondensedNodeType]
    ],
    condensed_node_type_to_vocab_size_map: dict[CondensedNodeType, int],
    num_negatives_per_condensed_edge_type: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs random negative sampling for each edge type.

    This function generates `num_negatives_per_condensed_edge_type` with src and dst selected at random from the
    vocabulary associated with the node types, as defined by the edge type and provided type-to-vocabulary maps.

    These can be consumed in model training as negative samples which are shared across edges.

    Args:
        condensed_edge_type_to_condensed_node_type_map (dict[int, tuple[int, int]]): A mapping from each edge type
            to a tuple of (source_node_type, destination_node_type) [R].
        condensed_node_type_to_vocab_size_map (dict[int, int]): A mapping from each node type to the size of its vocabulary.
        num_negatives_per_condensed_edge_type (int): The number of negative edges to sample per edge type [K].

    Returns:
        negative_edges (Tensor): A tensor of shape [R * K] containing negative edges.
        negative_edge_types (Tensor): A tensor of shape [R * K] containing the edge type
            for each negative edge.
        negative_labels (Tensor): A tensor of zeros with shape [R * K], suitable for
            use in contrastive or classification losses.
    """

    negative_condensed_edge_types = torch.tensor(
        list(condensed_edge_type_to_condensed_node_type_map.keys()), dtype=torch.int32
    ).repeat_interleave(num_negatives_per_condensed_edge_type)

    negative_edges = torch.zeros(
        negative_condensed_edge_types.numel(), 2, dtype=torch.int
    )

    # Labels are all 0 for negatives
    negative_labels = torch.zeros_like(negative_condensed_edge_types, dtype=torch.int)

    if num_negatives_per_condensed_edge_type:
        # Corrupt nodes in-place based on edge type and corruption side
        for (
            condensed_edge_type,
            condensed_node_types,
        ) in condensed_edge_type_to_condensed_node_type_map.items():
            relation_mask = (
                negative_condensed_edge_types == condensed_edge_type
            )  # [E * K]
            src_cnt, dst_cnt = condensed_node_types

            # Sample uniformly from the vocabulary.
            src_vocab_size = condensed_node_type_to_vocab_size_map[src_cnt]
            dst_vocab_size = condensed_node_type_to_vocab_size_map[dst_cnt]
            rand_src_inds = (
                torch.rand(size=(num_negatives_per_condensed_edge_type,))
                * src_vocab_size
            ).to(torch.int)
            rand_dst_inds = (
                torch.rand(size=(num_negatives_per_condensed_edge_type,))
                * dst_vocab_size
            ).to(torch.int)
            negative_edges[relation_mask, 0] = rand_src_inds
            negative_edges[relation_mask, 1] = rand_dst_inds

    return negative_edges, negative_condensed_edge_types, negative_labels
