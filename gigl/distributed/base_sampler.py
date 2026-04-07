from collections import defaultdict
from dataclasses import dataclass
from typing import Union

import torch
from graphlearn_torch.distributed import DistNeighborSampler as GLTDistNeighborSampler
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import NodeType

from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
)
from gigl.utils.data_splitters import PADDING_NODE


def _stable_unique_preserve_order(nodes: torch.Tensor) -> torch.Tensor:
    """Return unique 1-D values while preserving first-occurrence order.

    Args:
        nodes: A 1-D tensor of node IDs (may contain duplicates).

    Returns:
        A 1-D tensor of unique node IDs in first-occurrence order.

    Raises:
        ValueError: If ``nodes`` is not 1-D.
    """
    if nodes.dim() != 1:
        raise ValueError(
            f"Expected a 1-D tensor of node ids, got shape {tuple(nodes.shape)}."
        )
    if nodes.numel() <= 1:
        return nodes

    unique_nodes, inverse = torch.unique(nodes, sorted=False, return_inverse=True)
    first_positions = torch.full(
        (unique_nodes.numel(),),
        fill_value=nodes.numel(),
        dtype=torch.long,
        device=nodes.device,
    )
    positions = torch.arange(nodes.numel(), device=nodes.device)
    first_positions.scatter_reduce_(
        0,
        inverse,
        positions,
        reduce="amin",
        include_self=True,
    )
    stable_order = torch.argsort(first_positions)
    return unique_nodes[stable_order]


@dataclass
class SampleLoopInputs:
    """Inputs prepared for the neighbor sampling loop in _sample_from_nodes.

    This dataclass holds the processed inputs that are passed to the core
    sampling loop. It allows _prepare_sample_loop_inputs to customize what nodes
    are sampled from and what metadata is attached to the output, without
    duplicating the sampling loop logic.

    Attributes:
        nodes_to_sample: For homogeneous graphs, a tensor of node IDs. For
            heterogeneous graphs, a dict mapping node types to tensors.
            For ABLP, this also includes supervision nodes (positive/negative labels).
        metadata: Metadata dict to attach to the sampler output (e.g., label tensors).
    """

    nodes_to_sample: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
    metadata: dict[str, torch.Tensor]


class BaseDistNeighborSampler(GLTDistNeighborSampler):
    """Base class for GiGL distributed samplers.

    Extends GLT's DistNeighborSampler with shared utilities for preparing
    sampling inputs, including ABLP (anchor-based link prediction) support.

    Subclasses must override ``_sample_from_nodes`` with their specific
    sampling strategy (e.g., k-hop neighbor sampling, PPR-based sampling).
    """

    def _prepare_sample_loop_inputs(
        self,
        inputs: NodeSamplerInput,
    ) -> SampleLoopInputs:
        """Prepare inputs for the sampling loop.

        Handles both standard NodeSamplerInput and ABLPNodeSamplerInput.
        For ABLP inputs, adds supervision nodes to the sampling seeds and
        builds label metadata.

        Args:
            inputs: Either a NodeSamplerInput or ABLPNodeSamplerInput.

        Returns:
            SampleLoopInputs containing the nodes to sample from and any
            metadata related to the task (e.g., label tensors for ABLP).
        """
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type

        if isinstance(inputs, ABLPNodeSamplerInput):
            return self._prepare_ablp_inputs(inputs, input_seeds, input_type)

        # For homogeneous graphs (input_type is None), return tensor directly.
        # For heterogeneous graphs, return dict mapping node type to tensor.
        if input_type is None:
            return SampleLoopInputs(
                nodes_to_sample=input_seeds,
                metadata={},
            )
        return SampleLoopInputs(
            nodes_to_sample={input_type: input_seeds},
            metadata={},
        )

    def _prepare_ablp_inputs(
        self,
        inputs: ABLPNodeSamplerInput,
        input_seeds: torch.Tensor,
        input_type: NodeType,
    ) -> SampleLoopInputs:
        """Prepare ABLP inputs with supervision nodes and label metadata.

        Args:
            inputs: The ABLPNodeSamplerInput containing label information.
            input_seeds: The anchor node seeds (already moved to device).
            input_type: The node type of the anchor seeds.

        Returns:
            SampleLoopInputs with supervision nodes included in nodes_to_sample
            and label tensors in metadata.
        """
        # Since GLT swaps src/dst for edge_dir = "out",
        # and GiGL assumes that supervision edge types are always
        # (anchor_node_type, to, supervision_node_type),
        # we need to index into supervision edge types accordingly.
        label_edge_index = 0 if self.edge_dir == "in" else 2

        # Build metadata and input nodes from positive/negative labels.
        # We need to sample from the supervision nodes as well, and ensure
        # that we are sampling from the correct node type.
        metadata: dict[str, torch.Tensor] = {}
        input_seeds_builder: dict[
            Union[str, NodeType], list[torch.Tensor]
        ] = defaultdict(list)
        input_seeds_builder[input_type].append(input_seeds)

        for edge_type, label_tensor in inputs.positive_label_by_edge_types.items():
            filtered_label_tensor = label_tensor[label_tensor != PADDING_NODE].to(
                self.device
            )
            input_seeds_builder[edge_type[label_edge_index]].append(
                filtered_label_tensor
            )
            # Update the metadata per positive label edge type.
            # We do this because GLT only supports dict[str, torch.Tensor] for metadata.
            metadata[
                f"{POSITIVE_LABEL_METADATA_KEY}{str(tuple(edge_type))}"
            ] = label_tensor

        for edge_type, label_tensor in inputs.negative_label_by_edge_types.items():
            filtered_label_tensor = label_tensor[label_tensor != PADDING_NODE].to(
                self.device
            )
            input_seeds_builder[edge_type[label_edge_index]].append(
                filtered_label_tensor
            )
            # Update the metadata per negative label edge type.
            # We do this because GLT only supports dict[str, torch.Tensor] for metadata.
            metadata[
                f"{NEGATIVE_LABEL_METADATA_KEY}{str(tuple(edge_type))}"
            ] = label_tensor

        nodes_to_sample: dict[Union[str, NodeType], torch.Tensor] = {
            # Keep first-occurrence order so anchor seeds remain at the front of
            # their node type; graph-transformer paths rely on that convention.
            node_type: _stable_unique_preserve_order(
                torch.cat(seeds, dim=0).to(self.device)
            )
            for node_type, seeds in input_seeds_builder.items()
        }

        return SampleLoopInputs(
            nodes_to_sample=nodes_to_sample,
            metadata=metadata,
        )

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """Sample subgraph from seed nodes.

        Subclasses must override this method with their specific sampling
        strategy.

        Args:
            inputs: The seed nodes to sample from.

        Raises:
            NotImplementedError: Always — subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _sample_from_nodes."
        )
