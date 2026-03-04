import asyncio
import gc
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.distributed import DistNeighborSampler
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
class PreparedSamplingInputs:
    """Prepared inputs for the sampling loop.

    Attributes:
        input_seeds: The original anchor node seeds.
        input_type: The node type of the anchor seeds.
        nodes_to_sample: Dict mapping node types to tensors of nodes to include
            in sampling. May include additional nodes beyond input_seeds
            (e.g., supervision nodes in ABLP).
        metadata: Metadata dict to include in the sampler output.
        use_original_seeds_for_batch: If True, use input_seeds as the batch
            (ABLP behavior). If False, use the inducer result as the batch
            (GLT base behavior). Defaults to False for GLT compatibility.
    """

    input_seeds: torch.Tensor
    input_type: NodeType
    nodes_to_sample: dict[Union[str, NodeType], torch.Tensor]
    metadata: dict[str, torch.Tensor]
    use_original_seeds_for_batch: bool = False


class GiglDistNeighborSampler(DistNeighborSampler):
    """GiGL's base distributed neighbor sampler with template method pattern.

    Extends GLT's DistNeighborSampler and overrides _sample_from_nodes to use
    a hook (_prepare_sampling_inputs) that subclasses can override to customize
    input preparation without duplicating the core sampling loop.

    The default implementation behaves identically to the base GLT sampler.
    Subclasses can override _prepare_sampling_inputs to add additional nodes
    to the sampling (e.g., supervision nodes) or populate metadata.
    """

    def _prepare_sampling_inputs(
        self,
        inputs: NodeSamplerInput,
    ) -> PreparedSamplingInputs:
        """Prepare inputs for the sampling loop.

        Override this method in subclasses to customize input preparation.
        The default implementation uses the input seeds directly with no
        additional nodes or metadata.

        Args:
            inputs: The node sampler input.

        Returns:
            PreparedSamplingInputs containing the seeds, node type, nodes to
            sample from, and metadata.
        """
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type
        return PreparedSamplingInputs(
            input_seeds=input_seeds,
            input_type=input_type,
            nodes_to_sample={input_type: input_seeds},
            metadata={},
        )

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Optional[SampleMessage]:
        """Sample subgraph from seed nodes.

        Uses _prepare_sampling_inputs hook to allow subclasses to customize
        which nodes to sample from and what metadata to include.
        """
        prepared = self._prepare_sampling_inputs(inputs)
        input_seeds = prepared.input_seeds
        input_type = prepared.input_type
        nodes_to_sample = prepared.nodes_to_sample
        metadata = prepared.metadata
        use_original_seeds_for_batch = prepared.use_original_seeds_for_batch

        self.max_input_size = max(self.max_input_size, input_seeds.numel())
        inducer = self._acquire_inducer()
        is_hetero = self.dist_graph.data_cls == "hetero"

        output: NeighborOutput
        if is_hetero:
            assert input_type is not None
            out_nodes_hetero: dict[NodeType, list[torch.Tensor]] = {}
            out_rows_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            out_cols_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            out_edges_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            num_sampled_nodes_hetero: dict[NodeType, list[torch.Tensor]] = {}
            num_sampled_edges_hetero: dict[EdgeType, list[torch.Tensor]] = {}

            src_dict = inducer.init_node(nodes_to_sample)
            # GLT uses src_dict (inducer result) for batch; ABLP uses original seeds
            batch = (
                {input_type: input_seeds} if use_original_seeds_for_batch else src_dict
            )

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
                len(nodes_to_sample) == 1
            ), f"Expected 1 input node type, got {len(nodes_to_sample)}"
            assert (
                input_type == list(nodes_to_sample.keys())[0]
            ), f"Expected input type {input_type}, got {list(nodes_to_sample.keys())[0]}"

            srcs = inducer.init_node(nodes_to_sample[input_type])
            # GLT uses srcs (inducer result) for batch; ABLP uses original seeds
            batch = input_seeds if use_original_seeds_for_batch else srcs
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
                    row=torch.tensor([]).to(self.device),
                    col=torch.tensor([]).to(self.device),
                    edge=(torch.tensor([]).to(self.device) if self.with_edge else None),
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


class DistABLPNeighborSampler(GiglDistNeighborSampler):
    """ABLP-specific neighbor sampler that adds supervision nodes to sampling.

    Overrides _prepare_sampling_inputs to parse ABLPNodeSamplerInput, adding
    supervision nodes (positive/negative labels) to the sampling seeds and
    including label information in the output metadata.
    """

    def _prepare_sampling_inputs(
        self,
        inputs: NodeSamplerInput,
    ) -> PreparedSamplingInputs:
        """Prepare ABLP inputs with supervision nodes and label metadata.

        Parses ABLPNodeSamplerInput to extract positive/negative label nodes,
        adds them to the sampling seeds, and builds metadata with label tensors.

        Args:
            inputs: Must be an ABLPNodeSamplerInput.

        Returns:
            PreparedSamplingInputs with supervision nodes included in
            nodes_to_sample and label tensors in metadata.
        """
        assert isinstance(inputs, ABLPNodeSamplerInput)
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type

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

        # Memory cleanup
        del filtered_label_tensor, label_tensor
        for value in input_seeds_builder.values():
            value.clear()
        input_seeds_builder.clear()
        del input_seeds_builder
        gc.collect()

        return PreparedSamplingInputs(
            input_seeds=input_seeds,
            input_type=input_type,
            nodes_to_sample=nodes_to_sample,
            metadata=metadata,
            use_original_seeds_for_batch=True,
        )
