import asyncio
import gc
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import DistNeighborSampler as GLTDistNeighborSampler
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NeighborOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import EdgeType, NodeType
from graphlearn_torch.utils import count_dict, merge_dict, reverse_edge_type

from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
)
from gigl.utils.data_splitters import PADDING_NODE


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


class DistNeighborSampler(GLTDistNeighborSampler):
    """GiGL's distributed neighbor sampler supporting both standard and ABLP inputs.

    Extends GLT's DistNeighborSampler and overrides _sample_from_nodes to support
    both NodeSamplerInput (standard neighbor sampling) and ABLPNodeSamplerInput
    (anchor-based link prediction with supervision nodes).

    For ABLPNodeSamplerInput, supervision nodes (positive/negative labels) are
    added to the sampling seeds, and label information is included in the output
    metadata.
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

        # As a perf optimization, we *could* have `nodes_to_sample` be only the
        # unique nodes, but since torch.unique() calls a sort, we should
        # investigate if it's worth it.
        # TODO(kmonte, mkolodner-sc): Investigate if this is worth it.
        nodes_to_sample: dict[Union[str, NodeType], torch.Tensor] = {
            node_type: torch.cat(seeds, dim=0).to(self.device)
            for node_type, seeds in input_seeds_builder.items()
        }

        # Memory cleanup — only del loop vars if any labels were processed
        has_labels = bool(
            inputs.positive_label_by_edge_types or inputs.negative_label_by_edge_types
        )
        if has_labels:
            del filtered_label_tensor, label_tensor
        for value in input_seeds_builder.values():
            value.clear()
        input_seeds_builder.clear()
        del input_seeds_builder
        gc.collect()

        return SampleLoopInputs(
            nodes_to_sample=nodes_to_sample,
            metadata=metadata,
        )

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Optional[SampleMessage]:
        """Sample subgraph from seed nodes.

        Supports both NodeSamplerInput and ABLPNodeSamplerInput. For ABLP,
        supervision nodes are included in sampling and label metadata is
        attached to the output.
        """
        sample_loop_inputs = self._prepare_sample_loop_inputs(inputs)
        input_type = inputs.input_type
        nodes_to_sample = sample_loop_inputs.nodes_to_sample
        metadata = sample_loop_inputs.metadata

        self.max_input_size: int = max(self.max_input_size, inputs.node.numel())
        inducer = self._acquire_inducer()
        is_hetero = self.dist_graph.data_cls == "hetero"

        output: NeighborOutput
        if is_hetero:
            assert input_type is not None
            assert isinstance(nodes_to_sample, dict)
            out_nodes_hetero: dict[NodeType, list[torch.Tensor]] = {}
            out_rows_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            out_cols_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            out_edges_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            num_sampled_nodes_hetero: dict[NodeType, list[torch.Tensor]] = {}
            num_sampled_edges_hetero: dict[EdgeType, list[torch.Tensor]] = {}

            src_dict = inducer.init_node(nodes_to_sample)
            # Use the original anchor seeds (inputs.node) for batch tracking,
            # not the deduped nodes_to_sample. For ABLP, nodes_to_sample includes
            # supervision nodes which should not be part of the batch.
            batch = {input_type: inputs.node.to(self.device)}

            merge_dict(src_dict, out_nodes_hetero)
            count_dict(src_dict, num_sampled_nodes_hetero, 1)

            for i in range(self.num_hops):
                task_dict: dict[EdgeType, asyncio.Task] = {}
                nbr_dict: dict[EdgeType, list[torch.Tensor]] = {}
                edge_dict: dict[EdgeType, torch.Tensor] = {}
                for etype in self.edge_types:
                    req_num = self.num_neighbors[etype][i]
                    if self.edge_dir == "in":
                        srcs = src_dict.get(etype[-1], None)
                        if srcs is not None and srcs.numel() > 0:
                            task_dict[
                                reverse_edge_type(etype)
                            ] = self._loop.create_task(
                                self._sample_one_hop(srcs, req_num, etype)
                            )
                    elif self.edge_dir == "out":
                        srcs = src_dict.get(etype[0], None)
                        if srcs is not None and srcs.numel() > 0:
                            task_dict[etype] = self._loop.create_task(
                                self._sample_one_hop(srcs, req_num, etype)
                            )

                for etype, task in task_dict.items():
                    output = await task
                    if output.nbr.numel() == 0:
                        continue
                    nbr_dict[etype] = [src_dict[etype[0]], output.nbr, output.nbr_num]
                    if output.edge is not None:
                        edge_dict[etype] = output.edge

                if len(nbr_dict) == 0:
                    continue
                nodes_dict, rows_dict, cols_dict = inducer.induce_next(nbr_dict)
                merge_dict(nodes_dict, out_nodes_hetero)
                merge_dict(rows_dict, out_rows_hetero)
                merge_dict(cols_dict, out_cols_hetero)
                merge_dict(edge_dict, out_edges_hetero)
                count_dict(nodes_dict, num_sampled_nodes_hetero, i + 2)
                count_dict(cols_dict, num_sampled_edges_hetero, i + 1)
                src_dict = nodes_dict

            sample_output = HeteroSamplerOutput(
                node={
                    ntype: torch.cat(nodes) for ntype, nodes in out_nodes_hetero.items()
                },
                row={etype: torch.cat(rows) for etype, rows in out_rows_hetero.items()},
                col={etype: torch.cat(cols) for etype, cols in out_cols_hetero.items()},
                edge=(
                    {etype: torch.cat(eids) for etype, eids in out_edges_hetero.items()}
                    if self.with_edge
                    else None
                ),
                batch=batch,
                num_sampled_nodes=num_sampled_nodes_hetero,
                num_sampled_edges=num_sampled_edges_hetero,
                input_type=input_type,
                metadata=metadata,
            )
        else:
            assert (
                input_type is None
            ), f"Expected input_type to be None for homogeneous graph, got {input_type}"
            assert isinstance(nodes_to_sample, torch.Tensor)

            srcs = inducer.init_node(nodes_to_sample)
            # Use the original anchor seeds (inputs.node) for batch tracking,
            # not the deduped nodes_to_sample. For ABLP, nodes_to_sample includes
            # supervision nodes which should not be part of the batch.
            batch = inputs.node.to(self.device)
            out_nodes: list[torch.Tensor] = []
            out_edges: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            num_sampled_nodes: list[torch.Tensor] = []
            num_sampled_edges: list[torch.Tensor] = []
            out_nodes.append(srcs)
            num_sampled_nodes.append(srcs.size(0))

            for req_num in self.num_neighbors:
                output = await self._sample_one_hop(srcs, req_num, None)
                if output.nbr.numel() == 0:
                    break
                nodes, rows, cols = inducer.induce_next(
                    srcs, output.nbr, output.nbr_num
                )
                out_nodes.append(nodes)
                out_edges.append((rows, cols, output.edge))
                num_sampled_nodes.append(nodes.size(0))
                num_sampled_edges.append(cols.size(0))
                srcs = nodes

            if not out_edges:
                sample_output = SamplerOutput(
                    node=torch.cat(out_nodes),
                    row=torch.empty(0, dtype=torch.long, device=self.device),
                    col=torch.empty(0, dtype=torch.long, device=self.device),
                    edge=(
                        torch.empty(0, dtype=torch.long, device=self.device)
                        if self.with_edge
                        else None
                    ),
                    batch=batch,
                    num_sampled_nodes=num_sampled_nodes,
                    num_sampled_edges=num_sampled_edges,
                    metadata=metadata,
                )
            else:
                sample_output = SamplerOutput(
                    node=torch.cat(out_nodes),
                    row=torch.cat([e[0] for e in out_edges]),
                    col=torch.cat([e[1] for e in out_edges]),
                    edge=(
                        torch.cat([e[2] for e in out_edges]) if self.with_edge else None
                    ),
                    batch=batch,
                    num_sampled_nodes=num_sampled_nodes,
                    num_sampled_edges=num_sampled_edges,
                    metadata=metadata,
                )

        self.inducer_pool.put(inducer)
        return sample_output
