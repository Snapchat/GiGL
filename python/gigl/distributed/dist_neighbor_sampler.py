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
from graphlearn_torch.utils import count_dict, merge_dict, reverse_edge_type

from gigl.types.sampler import LabeledNodeSamplerInput
from gigl.utils.data_splitters import PADDING_NODE


class DistLinkPredictionNeighborSampler(DistNeighborSampler):
    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Optional[SampleMessage]:
        assert isinstance(inputs, LabeledNodeSamplerInput)
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type
        supervision_node_type = inputs.supervision_node_type
        labeled_seeds = inputs.positive_labels[
            inputs.positive_labels != PADDING_NODE
        ].to(self.device)
        if inputs.negative_labels is not None:
            negative_seeds = inputs.negative_labels[
                inputs.negative_labels != PADDING_NODE
            ].to(self.device)
            labeled_seeds = torch.cat((labeled_seeds, negative_seeds), dim=0)
        self.max_input_size = max(self.max_input_size, input_seeds.numel())
        inducer = self._acquire_inducer()
        is_hetero = self.dist_graph.data_cls == "hetero"
        if input_type == supervision_node_type:
            input_nodes = {input_type: torch.cat((input_seeds, labeled_seeds), dim=0)}
        else:
            input_nodes = {
                input_type: input_seeds,
                supervision_node_type: labeled_seeds,
            }
        if is_hetero:
            assert input_type is not None
            out_nodes, out_rows, out_cols, out_edges = {}, {}, {}, {}
            num_sampled_nodes, num_sampled_edges = {}, {}
            src_dict = inducer.init_node(input_nodes)
            batch = src_dict
            merge_dict(src_dict, out_nodes)
            count_dict(src_dict, num_sampled_nodes, 1)

            for i in range(self.num_hops):
                task_dict, nbr_dict, edge_dict = {}, {}, {}
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
                    output: NeighborOutput = await task
                    if output.nbr.numel() == 0:
                        continue
                    nbr_dict[etype] = [src_dict[etype[0]], output.nbr, output.nbr_num]
                    if output.edge is not None:
                        edge_dict[etype] = output.edge

                if len(nbr_dict) == 0:
                    continue
                nodes_dict, rows_dict, cols_dict = inducer.induce_next(nbr_dict)
                merge_dict(nodes_dict, out_nodes)
                merge_dict(rows_dict, out_rows)
                merge_dict(cols_dict, out_cols)
                merge_dict(edge_dict, out_edges)
                count_dict(nodes_dict, num_sampled_nodes, i + 2)
                count_dict(cols_dict, num_sampled_edges, i + 1)
                src_dict = nodes_dict

            sample_output = HeteroSamplerOutput(
                node={ntype: torch.cat(nodes) for ntype, nodes in out_nodes.items()},
                row={etype: torch.cat(rows) for etype, rows in out_rows.items()},
                col={etype: torch.cat(cols) for etype, cols in out_cols.items()},
                edge=(
                    {etype: torch.cat(eids) for etype, eids in out_edges.items()}
                    if self.with_edge
                    else None
                ),
                batch=batch,
                num_sampled_nodes=num_sampled_nodes,
                num_sampled_edges=num_sampled_edges,
                input_type=input_type,
                metadata={},
            )
        else:
            assert input_type == supervision_node_type
            srcs = inducer.init_node(input_nodes[input_type])
            batch = srcs
            out_nodes, out_edges = [], []
            num_sampled_nodes, num_sampled_edges = [], []
            out_nodes.append(srcs)
            num_sampled_nodes.append(srcs.size(0))
            # Sample subgraph.
            for req_num in self.num_neighbors:
                output: NeighborOutput = await self._sample_one_hop(srcs, req_num, None)
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
                metadata={},
            )

        # Reclaim inducer into pool.
        self.inducer_pool.put(inducer)

        return sample_output
