import asyncio
import gc
import heapq
import time
from collections import defaultdict
from typing import Optional, Set, Tuple, Union

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

_PPR_HOMOGENEOUS_NODE_TYPE = "ppr_homogeneous_node_type"
_PPR_HOMOGENEOUS_EDGE_TYPE = (
    _PPR_HOMOGENEOUS_NODE_TYPE,
    "to",
    _PPR_HOMOGENEOUS_NODE_TYPE,
)


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

    This sampler supports both homogeneous and heterogeneous graphs. For heterogeneous graphs,
    the PPR algorithm traverses across all edge types, switching edge types based on the
    current node type and the configured edge direction.

    Args:
        alpha: Restart probability (teleport probability back to seed). Higher values
               keep samples closer to seeds. Typical values: 0.15-0.25.
        eps: Convergence threshold. Smaller values give more accurate PPR scores
             but require more computation. Typical values: 1e-3 to 1e-6.
        max_ppr_nodes: Maximum number of nodes to return per seed based on PPR scores.
        default_node_id: Node ID to use when fewer than max_ppr_nodes are found.
        default_weight: Weight to assign to padding nodes.
        num_nbrs_per_hop: Maximum number of neighbors to fetch per hop.
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.5,
        eps: float = 1e-3,
        max_ppr_nodes: int = 50,
        default_node_id: int = -1,
        default_weight: float = 0.0,
        num_nbrs_per_hop: int = 10000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._eps = eps
        self._max_ppr_nodes = max_ppr_nodes
        self._default_node_id = default_node_id
        self._default_weight = default_weight
        self._alpha_eps = alpha * eps
        self._num_nbrs_per_hop = num_nbrs_per_hop

        # Build mapping from node type to edge types that can be traversed from that node type.
        self._node_type_to_edge_types: dict[NodeType, list[EdgeType]] = defaultdict(
            list
        )

        if hasattr(self, "edge_types") and self.edge_types is not None:
            self._is_homogeneous = False
            # Heterogeneous case: map each node type to its outgoing/incoming edge types
            for etype in self.edge_types:
                if self.edge_dir == "in":
                    # For incoming edges, we traverse FROM the destination node type
                    anchor_type = etype[-1]
                else:  # "out"
                    # For outgoing edges, we traverse FROM the source node type
                    anchor_type = etype[0]

                self._node_type_to_edge_types[anchor_type].append(etype)
        else:
            self._node_type_to_edge_types[_PPR_HOMOGENEOUS_NODE_TYPE] = [
                _PPR_HOMOGENEOUS_EDGE_TYPE
            ]
            self._is_homogeneous = True

    def _get_neighbor_type(self, edge_type: EdgeType) -> NodeType:
        """Get the node type of neighbors reached via an edge type."""
        return edge_type[0] if self.edge_dir == "in" else edge_type[-1]

    async def _get_neighbors_for_nodes(
        self,
        nodes: torch.Tensor,
        edge_type: EdgeType,
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
            srcs=nodes,
            num_nbr=self._num_nbrs_per_hop,
            etype=edge_type if edge_type is not _PPR_HOMOGENEOUS_EDGE_TYPE else None,
        )
        return output.nbr, output.nbr_num

    async def _batch_fetch_neighbors(
        self,
        nodes_by_edge_type: dict[EdgeType, Set[int]],
        neighbor_target: dict[Tuple[int, EdgeType], list[int]],
        degree_target: dict[Tuple[int, EdgeType], int],
        device: torch.device,
    ) -> int:
        """
        Batch fetch neighbors for nodes grouped by edge type.

        Fetches neighbors for all nodes in nodes_by_edge_type, populating
        neighbor_target with neighbor lists and degree_target with counts.

        Args:
            nodes_by_edge_type: Dict mapping edge type to set of node IDs to fetch
            neighbor_target: Dict to populate with (node_id, edge_type) -> neighbor list
            degree_target: Dict to populate with (node_id, edge_type) -> degree count
            device: Torch device for tensor creation

        Returns:
            Number of neighbor lookup calls made
        """
        num_lookups = 0
        for etype, node_ids in nodes_by_edge_type.items():
            if not node_ids:
                continue
            nodes_list = list(node_ids)
            lookup_tensor = torch.tensor(nodes_list, dtype=torch.long, device=device)

            neighbors, neighbor_counts = await self._get_neighbors_for_nodes(
                lookup_tensor,
                etype,
            )
            num_lookups += 1

            neighbors_list = neighbors.tolist()
            counts_list = neighbor_counts.tolist()
            del neighbors, neighbor_counts

            # neighbors_list is a flat concatenation of all neighbors for all looked-up nodes.
            # We use offset to slice out each node's neighbors: node i's neighbors are at
            # neighbors_list[offset : offset + count], then we advance offset by count.
            offset = 0
            for node_id, count in zip(nodes_list, counts_list):
                cache_key = (node_id, etype)
                neighbor_target[cache_key] = neighbors_list[offset : offset + count]
                degree_target[cache_key] = count
                offset += count

        return num_lookups

    async def _compute_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: Optional[NodeType] = None,
    ) -> Tuple[
        Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        dict[str, float],
    ]:
        """
        Compute PPR scores for seed nodes using the push-based approximation algorithm.

        This implements the Forward Push algorithm (Andersen et al., 2006) which
        iteratively pushes probability mass from nodes with high residual to their
        neighbors. For heterogeneous graphs, the algorithm traverses across all
        edge types, switching based on the current node type.

        Algorithm Overview (each iteration of the main loop):
            1. Fetch neighbors: Drain all nodes from the queue, group by edge type,
               and perform a batched neighbor lookup to populate neighbor/degree caches.
            2. Push residual: For each queued node, add its residual to its PPR score,
               reset its residual to zero, then distribute (1-alpha) * residual to
               all neighbors proportionally by degree.
            3. Batch fetch degrees: Group all neighbors that received residual and
               perform a batched lookup to get their degrees (needed for threshold check).
            4. Re-queue high-residual nodes: For each neighbor that received residual,
               check if residual >= alpha * eps * total_degree. If so, add to queue
               for processing in the next iteration.

        Args:
            seed_nodes: Tensor of seed node IDs [batch_size]
            seed_node_type: Node type of seed nodes. Should be None for homogeneous graphs.

        Returns:
            Tuple of (neighbor_ids_by_type, ppr_weights_by_type, stats) where:
                - neighbor_ids_by_type: Union[torch.Tensor, dict mapping node type -> [batch_size, max_ppr_nodes]]
                - ppr_weights_by_type: Union[torch.Tensor, dict mapping node type -> [batch_size, max_ppr_nodes]]
                - stats: dict with iteration statistics for runtime tuning
        """
        if seed_node_type is None:
            seed_node_type = _PPR_HOMOGENEOUS_NODE_TYPE
        device = seed_nodes.device
        batch_size = seed_nodes.size(0)

        # PPR scores: p[i][(node_id, node_type)] = score
        p: list[dict[Tuple[int, NodeType], float]] = [
            defaultdict(float) for _ in range(batch_size)
        ]
        # Residuals: r[i][(node_id, node_type)] = residual
        r: list[dict[Tuple[int, NodeType], float]] = [
            defaultdict(float) for _ in range(batch_size)
        ]

        # Queue stores (node_id, node_type) tuples
        q: list[Set[Tuple[int, NodeType]]] = [set() for _ in range(batch_size)]

        seed_list = seed_nodes.tolist()

        # Initialize residuals: r[i][(seed, seed_type)] = alpha for each seed
        for i, seed in enumerate(seed_list):
            r[i][(seed, seed_node_type)] = self._alpha
            q[i].add((seed, seed_node_type))

        # Cache keyed by (node_id, edge_type) since same node can have different neighbors per edge type
        neighbor_cache: dict[Tuple[int, EdgeType], list[int]] = {}
        degree_cache: dict[Tuple[int, EdgeType], int] = {}

        num_nodes_in_queue = batch_size

        # Statistics for running forward push
        ppr_num_iterations = 0
        ppr_total_nodes_processed = 0
        ppr_num_neighbor_lookups = 0

        # Timing statistics (in seconds)
        total_network_time = 0.0
        total_for_loop_time = 0.0
        batch_start_time = time.perf_counter()

        while num_nodes_in_queue > 0:
            ppr_num_iterations += 1

            # Drain all nodes from all queues and group by edge type for batched lookups
            loop_start = time.perf_counter()
            nodes_to_process: list[Set[Tuple[int, NodeType]]] = [
                set() for _ in range(batch_size)
            ]
            nodes_by_edge_type: dict[EdgeType, Set[int]] = defaultdict(set)

            for i in range(batch_size):
                if q[i]:
                    nodes_to_process[i] = q[i]
                    q[i] = set()
                    num_nodes_in_queue -= len(nodes_to_process[i])

                    # Group nodes by edge type for batched lookups
                    for node_id, node_type in nodes_to_process[i]:
                        edge_types_for_node = self._node_type_to_edge_types[node_type]
                        for etype in edge_types_for_node:
                            cache_key = (node_id, etype)
                            if cache_key not in neighbor_cache:
                                nodes_by_edge_type[etype].add(node_id)
            total_for_loop_time += time.perf_counter() - loop_start

            # Batch fetch neighbors per edge type
            network_start = time.perf_counter()
            ppr_num_neighbor_lookups += await self._batch_fetch_neighbors(
                nodes_by_edge_type, neighbor_cache, degree_cache, device
            )
            total_network_time += time.perf_counter() - network_start

            # Collect neighbors needing degree lookups while processing
            neighbors_needing_degree: dict[EdgeType, Set[int]] = defaultdict(set)

            # Process nodes and push residual
            loop_start = time.perf_counter()
            for i in range(batch_size):
                for u_node, u_type in nodes_to_process[i]:
                    ppr_total_nodes_processed += 1

                    key_u = (u_node, u_type)
                    res_u = r[i].get(key_u, 0.0)

                    # Push to PPR score and reset residual
                    p[i][key_u] += res_u
                    r[i][key_u] = 0.0

                    # For each edge type from this node type, push residual to neighbors
                    edge_types_for_node = self._node_type_to_edge_types[u_type]

                    # Calculate total degree across all edge types for proper probability distribution
                    total_degree = sum(
                        degree_cache[(u_node, etype)] for etype in edge_types_for_node
                    )

                    if total_degree == 0:
                        continue

                    # Push residual proportionally based on degree per edge type
                    for etype in edge_types_for_node:
                        cache_key = (u_node, etype)
                        neighbor_list = neighbor_cache[cache_key]
                        neighbor_count = degree_cache[cache_key]

                        if neighbor_count == 0:
                            continue

                        # Determine the type of the neighbors
                        v_type = self._get_neighbor_type(etype)

                        # Distribute residual to neighbors, weighted by edge type contribution
                        push_value = (1 - self._alpha) * res_u / total_degree

                        for v_node in neighbor_list:
                            key_v = (v_node, v_type)
                            r[i][key_v] += push_value

                            # Collect neighbors needing degree lookups inline
                            edge_types_for_v = self._node_type_to_edge_types.get(
                                v_type, []
                            )
                            for v_etype in edge_types_for_v:
                                v_cache_key = (v_node, v_etype)
                                if v_cache_key not in degree_cache:
                                    neighbors_needing_degree[v_etype].add(v_node)
            total_for_loop_time += time.perf_counter() - loop_start

            # Batch fetch neighbors for all v_nodes not in cache to get their degrees.
            # We keep neighbors temporarily and only promote to neighbor_cache if they pass the residual threshold.
            # Doing it this way is more performant than always storing neighbors in the neighbor cache, and leads to ~20x less
            # nodes polluting the neighbor cache.
            # Note that it may be more performant to only fetch neighbor counts instead of both neighbors
            # and neighbor counts. However, GLT doesn't expose a lever for only getting neighbor counts, so we
            # fetch all neighbors and neighbor counts initially.
            # TODO (mkolodner-sc): Investigate and potentially upstream an item to GLT to expose lever for only getting neighbor counts.
            network_start = time.perf_counter()
            temp_neighbors: dict[Tuple[int, EdgeType], list[int]] = {}
            ppr_num_neighbor_lookups += await self._batch_fetch_neighbors(
                neighbors_needing_degree, temp_neighbors, degree_cache, device
            )
            total_network_time += time.perf_counter() - network_start

            # Add high-residual neighbors to queue
            loop_start = time.perf_counter()
            for i in range(batch_size):
                for u_node, u_type in nodes_to_process[i]:
                    edge_types_for_node = self._node_type_to_edge_types.get(u_type, [])
                    for etype in edge_types_for_node:
                        cache_key = (u_node, etype)
                        neighbor_list = neighbor_cache[cache_key]
                        v_type = self._get_neighbor_type(etype)

                        for v_node in neighbor_list:
                            key_v = (v_node, v_type)

                            if key_v in q[i]:
                                continue

                            res_v = r[i].get(key_v, 0.0)
                            if res_v == 0.0:
                                continue

                            # Sum degrees across all edge types from v_type for threshold check
                            edge_types_for_v = self._node_type_to_edge_types.get(
                                v_type, []
                            )
                            total_v_degree = sum(
                                degree_cache[(v_node, v_etype)]
                                for v_etype in edge_types_for_v
                            )

                            if res_v >= self._alpha_eps * total_v_degree:
                                q[i].add(key_v)
                                num_nodes_in_queue += 1

                                # Promote temp_neighbors to neighbor_cache if available
                                for v_etype in edge_types_for_v:
                                    temp_key = (v_node, v_etype)
                                    if temp_key in temp_neighbors:
                                        neighbor_cache[temp_key] = temp_neighbors[
                                            temp_key
                                        ]
            total_for_loop_time += time.perf_counter() - loop_start

            del temp_neighbors

        # Extract top-k nodes by PPR score, grouped by node type
        # Collect all node types that appear in results
        loop_start = time.perf_counter()
        all_node_types: Set[NodeType] = set()
        for i in range(batch_size):
            for node_id, node_type in p[i].keys():
                all_node_types.add(node_type)

        out_neighbor_ids: Union[torch.Tensor, dict[NodeType, torch.Tensor]] = {}
        out_weights: Union[torch.Tensor, dict[NodeType, torch.Tensor]] = {}

        for ntype in all_node_types:
            ntype_neighbor_ids = torch.full(
                (batch_size, self._max_ppr_nodes),
                self._default_node_id,
                dtype=torch.long,
                device=device,
            )
            ntype_weights = torch.full(
                (batch_size, self._max_ppr_nodes),
                self._default_weight,
                dtype=torch.float,
                device=device,
            )

            for i in range(batch_size):
                # Filter to nodes of this type
                type_scores = {
                    node_id: score
                    for (node_id, node_type), score in p[i].items()
                    if node_type == ntype
                }
                top_k = heapq.nlargest(
                    self._max_ppr_nodes, type_scores.items(), key=lambda x: x[1]
                )

                for j, (node_id, weight) in enumerate(top_k):
                    ntype_neighbor_ids[i, j] = node_id
                    ntype_weights[i, j] = weight

            out_neighbor_ids[ntype] = ntype_neighbor_ids
            out_weights[ntype] = ntype_weights

        if self._is_homogeneous:
            assert (
                len(all_node_types) == 1
                and _PPR_HOMOGENEOUS_NODE_TYPE in all_node_types
            )
            out_neighbor_ids = out_neighbor_ids[_PPR_HOMOGENEOUS_NODE_TYPE]
            out_weights = out_weights[_PPR_HOMOGENEOUS_NODE_TYPE]
        total_for_loop_time += time.perf_counter() - loop_start

        # Calculate total batch time
        total_batch_time = time.perf_counter() - batch_start_time

        # Collect PPR iteration statistics for runtime tuning (includes timing stats)
        ppr_stats: dict[str, float] = {
            "ppr_num_iterations": ppr_num_iterations,
            "ppr_total_nodes_processed": ppr_total_nodes_processed,
            "ppr_num_neighbor_lookups": ppr_num_neighbor_lookups,
            "ppr_neighbor_cache_size": len(neighbor_cache),
            "ppr_degree_cache_size": len(degree_cache),
            # Timing stats in milliseconds
            "ppr_total_time_ms": total_batch_time * 1000,
            "ppr_network_time_ms": total_network_time * 1000,
            "ppr_for_loop_time_ms": total_for_loop_time * 1000,
        }

        return out_neighbor_ids, out_weights, ppr_stats

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Optional[SampleMessage]:
        """
        Override the base sampling method to use PPR-based neighbor selection.

        For heterogeneous graphs, PPR traverses across all edge types, switching
        edge types based on the current node type. This allows discovering nodes
        of different types through multi-hop traversal.
        """
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type

        is_hetero = self.dist_graph.data_cls == "hetero"
        metadata: dict[str, torch.Tensor] = {}

        # Compute PPR scores - unified for both homogeneous and heterogeneous
        # For hetero: pass input_type as seed_node_type
        # For homo: pass None as seed_node_type
        seed_node_type = input_type if is_hetero else None
        (
            neighbor_ids_by_type,
            weights_by_type,
            ppr_stats,
        ) = await self._compute_ppr_scores(input_seeds, seed_node_type)

        # Add PPR stats to metadata as tensors (required by GLT SampleQueue)
        metadata["ppr_num_iterations"] = torch.tensor(
            [ppr_stats["ppr_num_iterations"]],
            dtype=torch.long,
            device=self.device,
        )
        metadata["ppr_total_nodes_processed"] = torch.tensor(
            [ppr_stats["ppr_total_nodes_processed"]],
            dtype=torch.long,
            device=self.device,
        )
        metadata["ppr_num_neighbor_lookups"] = torch.tensor(
            [ppr_stats["ppr_num_neighbor_lookups"]],
            dtype=torch.long,
            device=self.device,
        )
        metadata["ppr_neighbor_cache_size"] = torch.tensor(
            [ppr_stats["ppr_neighbor_cache_size"]],
            dtype=torch.long,
            device=self.device,
        )
        metadata["ppr_degree_cache_size"] = torch.tensor(
            [ppr_stats["ppr_degree_cache_size"]],
            dtype=torch.long,
            device=self.device,
        )
        # Add timing stats to metadata (in milliseconds)
        metadata["ppr_total_time_ms"] = torch.tensor(
            [ppr_stats["ppr_total_time_ms"]],
            dtype=torch.float,
            device=self.device,
        )
        metadata["ppr_network_time_ms"] = torch.tensor(
            [ppr_stats["ppr_network_time_ms"]],
            dtype=torch.float,
            device=self.device,
        )
        metadata["ppr_for_loop_time_ms"] = torch.tensor(
            [ppr_stats["ppr_for_loop_time_ms"]],
            dtype=torch.float,
            device=self.device,
        )

        if is_hetero:
            assert isinstance(neighbor_ids_by_type, dict)
            assert isinstance(weights_by_type, dict)
            assert input_type is not None

            # Build the sampler output from PPR results
            out_nodes_hetero: dict[NodeType, list[torch.Tensor]] = defaultdict(list)

            # Add input seeds
            out_nodes_hetero[input_type].append(input_seeds)

            inducer = self._acquire_inducer()

            for ntype, neighbor_ids in neighbor_ids_by_type.items():
                weights = weights_by_type[ntype]

                # Flatten and deduplicate neighbors
                flat_neighbors = neighbor_ids.flatten()
                valid_mask = flat_neighbors != self._default_node_id
                valid_neighbors = flat_neighbors[valid_mask]

                if valid_neighbors.numel() > 0:
                    out_nodes_hetero[ntype].append(valid_neighbors.unique())

                    # Store PPR weights in metadata for later use
                    metadata[f"ppr_weights_{ntype}"] = weights
                    metadata[f"ppr_neighbor_ids_{ntype}"] = neighbor_ids

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
                edge={},  # Empty dict instead of None - GLT SampleQueue requires all values to be tensors
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
            assert isinstance(neighbor_ids_by_type, torch.Tensor)
            assert isinstance(weights_by_type, torch.Tensor)

            neighbor_ids = neighbor_ids_by_type
            weights = weights_by_type

            # Flatten and get unique neighbors
            flat_neighbors = neighbor_ids.flatten()
            valid_mask = flat_neighbors != self._default_node_id
            valid_neighbors = flat_neighbors[valid_mask].unique()

            all_nodes = torch.cat([input_seeds, valid_neighbors])

            metadata["ppr_weights"] = weights
            metadata["ppr_neighbor_ids"] = neighbor_ids

            sample_output = SamplerOutput(
                node=all_nodes,
                row=torch.tensor([], dtype=torch.long, device=self.device),
                col=torch.tensor([], dtype=torch.long, device=self.device),
                edge=torch.tensor(
                    [], dtype=torch.long, device=self.device
                ),  # Empty tensor instead of None - GLT SampleQueue requires all values to be tensors
                batch=input_seeds,
                num_sampled_nodes=[input_seeds.size(0), valid_neighbors.size(0)],
                num_sampled_edges=[],
                metadata=metadata,
            )

        return sample_output
