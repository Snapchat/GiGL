import asyncio
import gc
import heapq
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

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

        # Since GLT swaps src/dst for edge_dir = "out",
        # and GiGL assumes that supervision edge types are always (anchor_node_type, to, supervision_node_type),
        # we need to index into supervision edge types accordingly.
        label_edge_index = 0 if self.edge_dir == "in" else 2

        # Go through the positive and negative labels and add them to the metadata and input seeds builder.
        # We need to sample from the supervision nodes as well, and ensure that we are sampling from the correct node type.
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
        # As a perf optimization, we *could* have `input_nodes` be only the unique nodes,
        # but since torch.unique() calls a sort, we should investigate if it's worth it.
        # TODO(kmonte, mkolodner-sc): Investigate if this is worth it.
        input_nodes: dict[Union[str, NodeType], torch.Tensor] = {
            node_type: torch.cat(seeds, dim=0).to(self.device)
            for node_type, seeds in input_seeds_builder.items()
        }
        del filtered_label_tensor, label_tensor
        for value in input_seeds_builder.values():
            value.clear()
        input_seeds_builder.clear()
        del input_seeds_builder
        gc.collect()

        self.max_input_size: int = max(self.max_input_size, input_seeds.numel())
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
            src_dict = inducer.init_node(input_nodes)
            batch = {input_type: input_seeds}
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
                len(input_nodes) == 1
            ), f"Expected 1 input node type, got {len(input_nodes)}"
            assert (
                input_type == list(input_nodes.keys())[0]
            ), f"Expected input type {input_type}, got {list(input_nodes.keys())[0]}"
            srcs = inducer.init_node(input_nodes[input_type])
            batch = input_seeds
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


class DistPPRNeighborSampler(DistNeighborSampler):
    """
    Personalized PageRank (PPR) based neighbor sampler that inherits from GLT DistNeighborSampler.

    Instead of uniform random sampling, this sampler uses PPR scores to select the most
    relevant neighbors for each seed node. The PPR algorithm approximates the stationary
    distribution of a random walk with restart probability alpha.

    Args:
        alpha: Restart probability (teleport probability back to seed). Higher values
               keep samples closer to seeds. Typical values: 0.15-0.25.
        eps: Convergence threshold. Smaller values give more accurate PPR scores
             but require more computation. Typical values: 1e-4 to 1e-6.
        max_ppr_nodes: Maximum number of nodes to return per seed based on PPR scores.
        default_node_id: Node ID to use when fewer than max_ppr_nodes are found.
        default_weight: Weight to assign to padding nodes.
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.15,
        eps: float = 1e-5,
        max_ppr_nodes: int = 50,
        default_node_id: int = -1,
        default_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.eps = eps
        self.max_ppr_nodes = max_ppr_nodes
        self.default_node_id = default_node_id
        self.default_weight = default_weight
        self._alpha_eps = alpha * eps

    async def _get_neighbors_for_nodes(
        self,
        nodes: torch.Tensor,
        edge_type: Optional[EdgeType] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch neighbors for a batch of nodes.

        Returns:
            Tuple of (neighbors, neighbor_counts) where neighbors is a flattened tensor
            and neighbor_counts[i] gives the number of neighbors for nodes[i].
        """
        # Use the underlying sampling infrastructure to get all neighbors
        # We request a large number to effectively get all neighbors
        output: NeighborOutput = await self._sample_one_hop(
            nodes, req_num=-1, edge_type=edge_type  # -1 typically means all neighbors
        )
        return output.nbr, output.nbr_num

    async def _compute_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        edge_type: Optional[EdgeType] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute PPR scores for seed nodes using the push-based approximation algorithm.

        This implements the Forward Push algorithm (Andersen et al., 2006) which
        iteratively pushes probability mass from nodes with high residual to their
        neighbors.

        Args:
            seed_nodes: Tensor of seed node IDs [batch_size]
            edge_type: Optional edge type for heterogeneous graphs

        Returns:
            Tuple of (neighbor_ids, ppr_weights) both of shape [batch_size, max_ppr_nodes]
        """
        device = seed_nodes.device
        batch_size = seed_nodes.size(0)

        # Initialize PPR scores (p) and residuals (r) for each seed
        # Using dictionaries for sparse representation
        p: List[Dict[int, float]] = [defaultdict(float) for _ in range(batch_size)]
        r: List[Dict[int, float]] = [defaultdict(float) for _ in range(batch_size)]

        # Initialize queues with seed nodes
        # Use both a list (for ordering) and a set (for O(1) membership check)
        q: List[List[int]] = [[] for _ in range(batch_size)]
        q_set: List[Set[int]] = [set() for _ in range(batch_size)]

        # Batch convert seeds to Python ints (single boundary crossing)
        seed_list = seed_nodes.tolist()

        # Initialize residuals: r[i][seed] = alpha for each seed
        for i, seed in enumerate(seed_list):
            r[i][seed] = self.alpha
            q[i].append(seed)
            q_set[i].add(seed)

        # Cache for neighbor lookups to avoid redundant fetches
        # Store as Python lists to avoid repeated .item() calls
        neighbor_cache: Dict[int, List[int]] = {}

        # Count total nodes in queues
        num_nodes_in_queue = batch_size

        while num_nodes_in_queue > 0:
            # Collect all unique nodes that need neighbor lookups
            nodes_to_lookup: List[int] = []
            nodes_to_lookup_set: Set[int] = set()
            for i in range(batch_size):
                for node_id in q[i]:
                    if (
                        node_id not in neighbor_cache
                        and node_id not in nodes_to_lookup_set
                    ):
                        nodes_to_lookup.append(node_id)
                        nodes_to_lookup_set.add(node_id)

            # Batch fetch neighbors for all nodes not in cache
            if nodes_to_lookup:
                lookup_tensor = torch.tensor(
                    nodes_to_lookup, dtype=torch.long, device=device
                )
                neighbors, neighbor_counts = await self._get_neighbors_for_nodes(
                    lookup_tensor, edge_type
                )

                # Batch convert to Python lists (single boundary crossing)
                neighbors_list = neighbors.tolist()
                counts_list = neighbor_counts.tolist()

                # Populate cache with Python lists
                offset = 0
                for node_id, count in zip(nodes_to_lookup, counts_list):
                    neighbor_cache[node_id] = neighbors_list[offset : offset + count]
                    offset += count

            # Process one node from each non-empty queue
            for i in range(batch_size):
                if not q[i]:
                    continue

                u_node = q[i].pop()
                q_set[i].discard(u_node)
                num_nodes_in_queue -= 1

                # Get residual for this node
                res_u = r[i].get(u_node, 0.0)

                # Push to PPR score and reset residual
                p[i][u_node] += res_u
                r[i][u_node] = 0.0

                # Get neighbors from cache (already Python list)
                neighbor_list = neighbor_cache.get(u_node, [])
                neighbor_count = len(neighbor_list)

                if neighbor_count == 0:
                    continue

                # Distribute residual to neighbors
                push_value = (1 - self.alpha) * res_u / neighbor_count

                for v_node in neighbor_list:
                    r[i][v_node] += push_value

                    # Check if v_node should be added to queue
                    res_v = r[i].get(v_node, 0.0)

                    # Get degree of v_node for threshold check
                    if v_node in neighbor_cache:
                        v_degree = len(neighbor_cache[v_node])
                    else:
                        # If not in cache, we'll fetch it next iteration
                        v_degree = 1  # Conservative estimate

                    # Add to queue if residual exceeds threshold (O(1) set lookup)
                    if res_v >= self._alpha_eps * v_degree:
                        if v_node not in q_set[i]:
                            q[i].append(v_node)
                            q_set[i].add(v_node)
                            num_nodes_in_queue += 1

        # Extract top-k nodes by PPR score for each seed using heapq (O(n log k))
        out_neighbor_ids = torch.full(
            (batch_size, self.max_ppr_nodes),
            self.default_node_id,
            dtype=torch.long,
            device=device,
        )
        out_weights = torch.full(
            (batch_size, self.max_ppr_nodes),
            self.default_weight,
            dtype=torch.float,
            device=device,
        )

        for i in range(batch_size):
            # Use heapq.nlargest for O(n log k) instead of O(n log n) full sort
            top_k = heapq.nlargest(self.max_ppr_nodes, p[i].items(), key=lambda x: x[1])

            for j, (node_id, weight) in enumerate(top_k):
                out_neighbor_ids[i, j] = node_id
                out_weights[i, j] = weight

        return out_neighbor_ids, out_weights

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Optional[SampleMessage]:
        """
        Override the base sampling method to use PPR-based neighbor selection.
        """
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type

        is_hetero = self.dist_graph.data_cls == "hetero"
        metadata: Dict[str, torch.Tensor] = {}

        if is_hetero:
            assert input_type is not None

            # Compute PPR for each edge type
            ppr_neighbors: Dict[EdgeType, Tuple[torch.Tensor, torch.Tensor]] = {}

            for etype in self.edge_types:
                if self.edge_dir == "in":
                    # For incoming edges, we sample from destination node type
                    if input_type == etype[-1]:
                        ppr_neighbors[etype] = await self._compute_ppr_scores(
                            input_seeds, etype
                        )
                elif self.edge_dir == "out":
                    # For outgoing edges, we sample from source node type
                    if input_type == etype[0]:
                        ppr_neighbors[etype] = await self._compute_ppr_scores(
                            input_seeds, etype
                        )

            # Build the sampler output from PPR results
            out_nodes_hetero: Dict[NodeType, List[torch.Tensor]] = defaultdict(list)
            out_rows_hetero: Dict[EdgeType, List[torch.Tensor]] = defaultdict(list)
            out_cols_hetero: Dict[EdgeType, List[torch.Tensor]] = defaultdict(list)

            # Add input seeds
            out_nodes_hetero[input_type].append(input_seeds)

            inducer = self._acquire_inducer()

            for etype, (neighbor_ids, weights) in ppr_neighbors.items():
                # Flatten and deduplicate neighbors
                flat_neighbors = neighbor_ids.flatten()
                valid_mask = flat_neighbors != self.default_node_id
                valid_neighbors = flat_neighbors[valid_mask]

                if valid_neighbors.numel() > 0:
                    # Determine the node type for these neighbors
                    neighbor_type = etype[0] if self.edge_dir == "in" else etype[-1]
                    out_nodes_hetero[neighbor_type].append(valid_neighbors.unique())

                    # Store PPR weights in metadata for later use
                    metadata[f"ppr_weights_{etype}"] = weights

            # Concatenate nodes per type
            node_dict = {
                ntype: torch.cat(nodes).unique()
                for ntype, nodes in out_nodes_hetero.items()
                if nodes
            }

            sample_output = HeteroSamplerOutput(
                node=node_dict,
                row={},  # PPR doesn't necessarily maintain edge structure
                col={},
                edge=None,
                batch={input_type: input_seeds},
                num_sampled_nodes={
                    ntype: [nodes.size(0)] for ntype, nodes in node_dict.items()
                },
                num_sampled_edges={},
                input_type=input_type,
                metadata=metadata,
            )

            self.inducer_pool.put(inducer)

        else:
            # Homogeneous graph case
            neighbor_ids, weights = await self._compute_ppr_scores(input_seeds, None)

            # Flatten and get unique neighbors
            flat_neighbors = neighbor_ids.flatten()
            valid_mask = flat_neighbors != self.default_node_id
            valid_neighbors = flat_neighbors[valid_mask].unique()

            all_nodes = torch.cat([input_seeds, valid_neighbors])

            metadata["ppr_weights"] = weights
            metadata["ppr_neighbor_ids"] = neighbor_ids

            sample_output = SamplerOutput(
                node=all_nodes,
                row=torch.tensor([], dtype=torch.long, device=self.device),
                col=torch.tensor([], dtype=torch.long, device=self.device),
                edge=None,
                batch=input_seeds,
                num_sampled_nodes=[input_seeds.size(0), valid_neighbors.size(0)],
                num_sampled_edges=[],
                metadata=metadata,
            )

        return sample_output
