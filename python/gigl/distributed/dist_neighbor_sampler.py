import asyncio
from typing import Optional

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

from gigl.distributed.sampler import ABLPNodeSamplerInput
from gigl.utils.data_splitters import PADDING_NODE

# TODO (mkolodner-sc): Investigate upstreaming this change back to GLT


class DistABLPNeighborSampler(DistNeighborSampler):
    """
    We inherit from the GLT DistNeighborSampler base class and override the _sample_from_nodes function. Specifically, we
    introduce functionality to read parse ABLPNodeSamplerInput, which contains information about the supervision nodes and node types
    that we also want to fanout around. We add the supervision nodes to the initial fanout seeds, and inject the label information into the
    output SampleMessage metadata.
    """

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Optional[SampleMessage]:
        assert isinstance(inputs, ABLPNodeSamplerInput)
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type
        supervision_node_type = inputs.supervision_node_type
        positive_labels = inputs.positive_labels.to(self.device)
        negative_labels = (
            inputs.negative_labels.to(self.device)
            if inputs.negative_labels is not None
            else None
        )

        positive_seeds = positive_labels[positive_labels != PADDING_NODE]
        negative_seeds: Optional[torch.Tensor]
        if negative_labels is not None:
            negative_seeds = negative_labels[negative_labels != PADDING_NODE]
        else:
            negative_seeds = None
        self.max_input_size: int = max(self.max_input_size, input_seeds.numel())
        inducer = self._acquire_inducer()
        is_hetero = self.dist_graph.data_cls == "hetero"
        metadata: dict[str, torch.Tensor] = {"positive_labels": positive_labels}
        if negative_labels is not None:
            metadata["negative_labels"] = negative_labels
        if input_type == supervision_node_type:
            combined_seeds: tuple[torch.Tensor, ...]
            if negative_seeds is not None:
                combined_seeds = (input_seeds, positive_seeds, negative_seeds)
            else:
                combined_seeds = (input_seeds, positive_seeds)
            input_nodes = {input_type: torch.cat(combined_seeds, dim=0)}
        else:
            if negative_seeds is None:
                input_nodes = {
                    input_type: input_seeds,
                    supervision_node_type: positive_seeds,
                }
            else:
                input_nodes = {
                    input_type: input_seeds,
                    supervision_node_type: torch.cat(
                        (positive_seeds, negative_seeds), dim=0
                    ),
                }
        output: NeighborOutput
        if is_hetero:
            assert input_type is not None
            out_nodes_hetero: dict[NodeType, list[torch.Tensor]] = {}
            out_rows_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            out_cols_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            out_edges_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            num_sampled_nodes_hetero: dict[NodeType, list[torch.Tensor]] = {}
            num_sampled_edges_hetero: dict[EdgeType, list[torch.Tensor]] = {}
            src_dict = inducer.init_node(input_nodes)
            batch = src_dict
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
            assert input_type == supervision_node_type
            srcs = inducer.init_node(input_nodes[input_type])
            batch = srcs
            out_nodes: list[torch.Tensor] = []
            out_edges: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
            num_sampled_nodes: list[torch.Tensor] = []
            num_sampled_edges: list[torch.Tensor] = []
            out_nodes.append(srcs)
            num_sampled_nodes.append(srcs.size(0))
            # Sample subgraph.
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

            sample_output = SamplerOutput(
                node=torch.cat(out_nodes),
                row=torch.cat([e[0] for e in out_edges]),
                col=torch.cat([e[1] for e in out_edges]),
                edge=(torch.cat([e[2] for e in out_edges]) if self.with_edge else None),
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
                metadata=metadata,
            )

        # Reclaim inducer into pool.
        self.inducer_pool.put(inducer)

        return sample_output
