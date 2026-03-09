import heapq
from collections import defaultdict
from typing import Optional, Union

import torch
from graphlearn_torch.channel import SampleMessage
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NeighborOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import EdgeType, NodeType
from graphlearn_torch.utils import merge_dict

from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler

_PPR_HOMOGENEOUS_NODE_TYPE = "ppr_homogeneous_node_type"
_PPR_HOMOGENEOUS_EDGE_TYPE = (
    _PPR_HOMOGENEOUS_NODE_TYPE,
    "to",
    _PPR_HOMOGENEOUS_NODE_TYPE,
)


class DistPPRNeighborSampler(DistNeighborSampler):
    """
    Personalized PageRank (PPR) based neighbor sampler that inherits from GLT DistNeighborSampler.

    Instead of uniform random sampling, this sampler uses PPR scores to select the most
    relevant neighbors for each seed node. The PPR algorithm approximates the stationary
    distribution of a random walk with restart probability alpha.

    This sampler supports both homogeneous and heterogeneous graphs. For heterogeneous graphs,
    the PPR algorithm traverses across all edge types, switching edge types based on the
    current node type and the configured edge direction.

    Degree tensors are sourced automatically from the dataset at initialization time.

    Args:
        alpha: Restart probability (teleport probability back to seed). Higher values
               keep samples closer to seeds. Typical values: 0.15-0.25.
        eps: Convergence threshold. Smaller values give more accurate PPR scores
             but require more computation. Typical values: 1e-4 to 1e-6.
        max_ppr_nodes: Maximum number of nodes to return per seed based on PPR scores.
        default_node_id: Node ID to use when fewer than max_ppr_nodes are found.
        default_weight: Weight to assign to padding nodes.
        num_nbrs_per_hop: Maximum number of neighbors to fetch per hop.
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.5,
        eps: float = 1e-4,
        max_ppr_nodes: int = 50,
        default_node_id: int = -1,
        default_weight: float = 0.0,
        num_nbrs_per_hop: int = 100000,
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

        assert isinstance(
            self.data, DistDataset
        ), "DistPPRNeighborSampler requires a GiGL DistDataset to access degree tensors."
        self._degree_tensors = self.data.degree_tensor

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

    def _get_degree_from_tensor(self, node_id: int, edge_type: EdgeType) -> int:
        """
        Look up the TRUE degree of a node for a specific edge type from in-memory tensors.

        This returns the actual node degree (not capped), which is mathematically correct
        for PPR algorithm calculations.

        Args:
            node_id: The ID of the node to look up.
            edge_type: The edge type to get the degree for.

        Returns:
            The true degree of the node for the given edge type.
        """
        if self._is_homogeneous:
            # For homogeneous graphs, degree_tensors is a single tensor
            assert isinstance(self._degree_tensors, torch.Tensor)
            if node_id >= len(self._degree_tensors):
                return 0
            return int(self._degree_tensors[node_id].item())
        else:
            # For heterogeneous graphs, degree_tensors is a dict keyed by edge type
            assert isinstance(self._degree_tensors, dict)
            if edge_type not in self._degree_tensors:
                return 0
            degree_tensor = self._degree_tensors[edge_type]
            if node_id >= len(degree_tensor):
                return 0
            return int(degree_tensor[node_id].item())

    def _get_neighbor_type(self, edge_type: EdgeType) -> NodeType:
        """Get the node type of neighbors reached via an edge type."""
        return edge_type[0] if self.edge_dir == "in" else edge_type[-1]

    async def _get_neighbors_for_nodes(
        self,
        nodes: torch.Tensor,
        edge_type: EdgeType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch neighbors for a batch of nodes.

        Returns:
            tuple of (neighbors, neighbor_counts) where neighbors is a flattened tensor
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
        nodes_by_edge_type: dict[EdgeType, set[int]],
        neighbor_target: dict[tuple[int, EdgeType], list[int]],
        device: torch.device,
    ) -> int:
        """
        Batch fetch neighbors for nodes grouped by edge type.

        Fetches neighbors for all nodes in nodes_by_edge_type, populating
        neighbor_target with neighbor lists. Degrees are looked up separately
        from the in-memory degree_tensors.

        Args:
            nodes_by_edge_type: Dict mapping edge type to set of node IDs to fetch
            neighbor_target: Dict to populate with (node_id, edge_type) -> neighbor list
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
                offset += count

        return num_lookups

    async def _compute_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: Optional[NodeType] = None,
    ) -> tuple[
        Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        Union[torch.Tensor, dict[NodeType, torch.Tensor]],
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
            tuple of (neighbor_ids_by_type, ppr_weights_by_type) where:
                - neighbor_ids_by_type: Union[torch.Tensor, dict mapping node type -> [batch_size, max_ppr_nodes]]
                - ppr_weights_by_type: Union[torch.Tensor, dict mapping node type -> [batch_size, max_ppr_nodes]]
        """
        if seed_node_type is None:
            seed_node_type = _PPR_HOMOGENEOUS_NODE_TYPE
        device = seed_nodes.device
        batch_size = seed_nodes.size(0)

        # PPR scores: p[i][(node_id, node_type)] = score
        p: list[dict[tuple[int, NodeType], float]] = [
            defaultdict(float) for _ in range(batch_size)
        ]
        # Residuals: r[i][(node_id, node_type)] = residual
        r: list[dict[tuple[int, NodeType], float]] = [
            defaultdict(float) for _ in range(batch_size)
        ]

        # Queue stores (node_id, node_type) tuples
        q: list[set[tuple[int, NodeType]]] = [set() for _ in range(batch_size)]

        seed_list = seed_nodes.tolist()

        # Initialize residuals: r[i][(seed, seed_type)] = alpha for each seed
        for i, seed in enumerate(seed_list):
            r[i][(seed, seed_node_type)] = self._alpha
            q[i].add((seed, seed_node_type))

        # Cache keyed by (node_id, edge_type) since same node can have different neighbors per edge type
        neighbor_cache: dict[tuple[int, EdgeType], list[int]] = {}

        num_nodes_in_queue = batch_size

        while num_nodes_in_queue > 0:
            # Drain all nodes from all queues and group by edge type for batched lookups
            nodes_to_process: list[set[tuple[int, NodeType]]] = [
                set() for _ in range(batch_size)
            ]
            nodes_by_edge_type: dict[EdgeType, set[int]] = defaultdict(set)

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

            # Batch fetch neighbors per edge type
            await self._batch_fetch_neighbors(
                nodes_by_edge_type, neighbor_cache, device
            )

            # Process nodes and push residual
            for i in range(batch_size):
                for u_node, u_type in nodes_to_process[i]:
                    key_u = (u_node, u_type)
                    res_u = r[i].get(key_u, 0.0)

                    # Push to PPR score and reset residual
                    p[i][key_u] += res_u
                    r[i][key_u] = 0.0

                    # For each edge type from this node type, push residual to neighbors
                    edge_types_for_node = self._node_type_to_edge_types[u_type]

                    # Calculate total degree across all edge types for proper probability distribution
                    # Degrees are looked up directly from in-memory tensors
                    total_degree = sum(
                        self._get_degree_from_tensor(u_node, etype)
                        for etype in edge_types_for_node
                    )

                    if total_degree == 0:
                        continue

                    # Push residual proportionally based on degree per edge type
                    for etype in edge_types_for_node:
                        cache_key = (u_node, etype)
                        neighbor_list = neighbor_cache[cache_key]
                        neighbor_count = self._get_degree_from_tensor(u_node, etype)

                        if neighbor_count == 0:
                            continue

                        # Determine the type of the neighbors
                        v_type = self._get_neighbor_type(etype)

                        # Distribute residual to neighbors, weighted by edge type contribution
                        push_value = (1 - self._alpha) * res_u / total_degree

                        for v_node in neighbor_list:
                            key_v = (v_node, v_type)
                            r[i][key_v] += push_value

            # Add high-residual neighbors to queue
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
                                self._get_degree_from_tensor(v_node, v_etype)
                                for v_etype in edge_types_for_v
                            )

                            if res_v >= self._alpha_eps * total_v_degree:
                                q[i].add(key_v)
                                num_nodes_in_queue += 1

        # Extract top-k nodes by PPR score, grouped by node type
        all_node_types: set[NodeType] = set()
        for i in range(batch_size):
            for node_id, node_type in p[i].keys():
                all_node_types.add(node_type)

        out_neighbor_ids_dict: dict[NodeType, torch.Tensor] = {}
        out_weights_dict: dict[NodeType, torch.Tensor] = {}

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

            out_neighbor_ids_dict[ntype] = ntype_neighbor_ids
            out_weights_dict[ntype] = ntype_weights

        out_neighbor_ids: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        out_weights: Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        if self._is_homogeneous:
            assert (
                len(all_node_types) == 1
                and _PPR_HOMOGENEOUS_NODE_TYPE in all_node_types
            )
            out_neighbor_ids = out_neighbor_ids_dict[_PPR_HOMOGENEOUS_NODE_TYPE]
            out_weights = out_weights_dict[_PPR_HOMOGENEOUS_NODE_TYPE]
        else:
            out_neighbor_ids = out_neighbor_ids_dict
            out_weights = out_weights_dict

        return out_neighbor_ids, out_weights

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Optional[SampleMessage]:
        """
        Override the base sampling method to use PPR-based neighbor selection.

        Supports both NodeSamplerInput and ABLPNodeSamplerInput. For ABLP, PPR
        scores are computed from both anchor and supervision nodes, so the sampled
        subgraph includes neighbors relevant to all seed types.

        For heterogeneous graphs, PPR traverses across all edge types, switching
        edge types based on the current node type. PPR weights are stored in
        metadata keyed as ``ppr_weights_{seed_type}_{neighbor_type}`` and
        ``ppr_neighbor_ids_{seed_type}_{neighbor_type}``.

        The ``ppr_neighbor_ids`` tensors are locally indexed — each value is a
        0-based index into ``data[ntype].node`` (the global-ID array produced by
        GLT's collate step), so downstream models can directly index into
        ``data[ntype].x`` without a separate global→local remapping step.

        The inducer is used to perform deduplication and local-index assignment
        in-place during sampling, avoiding a post-hoc lookup pass.
        """
        sample_loop_inputs = self._prepare_sample_loop_inputs(inputs)
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type
        is_hetero = self.dist_graph.data_cls == "hetero"
        metadata = sample_loop_inputs.metadata
        nodes_to_sample = sample_loop_inputs.nodes_to_sample

        inducer = self._acquire_inducer()

        if is_hetero:
            assert isinstance(nodes_to_sample, dict)
            assert input_type is not None

            # Register all seeds with the inducer; src_dict maps NodeType -> global IDs
            src_dict = inducer.init_node(nodes_to_sample)

            # Compute PPR for each seed type; build nbr_dict for a single inducer.induce_next
            # call using virtual edge types (seed_type, 'ppr', ntype).
            nbr_dict: dict[EdgeType, list[torch.Tensor]] = {}
            valid_counts_per_pair: dict[tuple[NodeType, NodeType], torch.Tensor] = {}
            all_ppr_neighbor_ids: dict[tuple[NodeType, NodeType], torch.Tensor] = {}

            for seed_type, seed_nodes in nodes_to_sample.items():
                nbr_ids_by_type, nbr_weights_by_type = await self._compute_ppr_scores(
                    seed_nodes, seed_type
                )
                assert isinstance(nbr_ids_by_type, dict)
                assert isinstance(nbr_weights_by_type, dict)

                for ntype, neighbor_ids in nbr_ids_by_type.items():
                    valid_mask = neighbor_ids != self._default_node_id
                    valid_counts = valid_mask.sum(dim=1)
                    flat_valid_nbrs = neighbor_ids[valid_mask]

                    valid_counts_per_pair[(seed_type, ntype)] = valid_counts
                    all_ppr_neighbor_ids[(seed_type, ntype)] = neighbor_ids
                    metadata[f"ppr_weights_{seed_type}_{ntype}"] = nbr_weights_by_type[
                        ntype
                    ]

                    # Only add to nbr_dict if there are actual neighbors; induce_next
                    # will deduplicate across seed types automatically.
                    if flat_valid_nbrs.numel() > 0:
                        virtual_etype: EdgeType = (seed_type, "ppr", ntype)
                        nbr_dict[virtual_etype] = [
                            src_dict[seed_type],
                            flat_valid_nbrs,
                            valid_counts,
                        ]

            new_nodes_dict, _rows_dict, cols_dict = inducer.induce_next(nbr_dict)

            # node_dict = seeds + newly discovered PPR neighbors (no duplicates)
            out_nodes_hetero: dict[NodeType, list[torch.Tensor]] = defaultdict(list)
            merge_dict(src_dict, out_nodes_hetero)
            merge_dict(new_nodes_dict, out_nodes_hetero)
            node_dict = {
                ntype: torch.cat(nodes)
                for ntype, nodes in out_nodes_hetero.items()
                if nodes
            }

            # Reconstruct locally-indexed ppr_neighbor_ids from inducer cols.
            # cols_dict[(seed_type, 'ppr', ntype)] holds local destination indices
            # for all edges, in the same flat order as flat_valid_nbrs was built.
            for (
                seed_type,
                ntype,
            ), original_neighbor_ids in all_ppr_neighbor_ids.items():
                valid_counts = valid_counts_per_pair[(seed_type, ntype)]
                ppr_ids_local = torch.full_like(original_neighbor_ids, -1)
                virtual_etype = (seed_type, "ppr", ntype)
                cols = cols_dict.get(virtual_etype)
                if cols is not None:
                    offset = 0
                    for i, count in enumerate(valid_counts.tolist()):
                        count = int(count)
                        ppr_ids_local[i, :count] = cols[offset : offset + count]
                        offset += count
                metadata[f"ppr_neighbor_ids_{seed_type}_{ntype}"] = ppr_ids_local

            sample_output = HeteroSamplerOutput(
                node=node_dict,
                row={},  # PPR doesn't maintain edge structure
                col={},
                edge={},  # Empty dict — GLT SampleQueue requires all values to be tensors
                batch={input_type: input_seeds},
                num_sampled_nodes={
                    ntype: [nodes.size(0)] for ntype, nodes in node_dict.items()
                },
                num_sampled_edges={},
                input_type=input_type,
                metadata=metadata,
            )

        else:
            assert isinstance(nodes_to_sample, torch.Tensor)

            # Register seeds; srcs holds their global IDs (local indices 0..N-1 assigned internally)
            srcs = inducer.init_node(nodes_to_sample)

            homo_neighbor_ids, homo_weights = await self._compute_ppr_scores(
                nodes_to_sample, None
            )
            assert isinstance(homo_neighbor_ids, torch.Tensor)
            assert isinstance(homo_weights, torch.Tensor)

            valid_mask = homo_neighbor_ids != self._default_node_id
            valid_counts = valid_mask.sum(dim=1)
            flat_valid_nbrs = homo_neighbor_ids[valid_mask]

            # induce_next deduplicates flat_valid_nbrs against already-seen nodes
            # and returns local destination indices (cols) for each neighbor edge.
            new_nodes, _rows, cols = inducer.induce_next(
                srcs, flat_valid_nbrs, valid_counts
            )
            all_nodes = torch.cat([srcs, new_nodes])

            # Reconstruct ppr_neighbor_ids_local with shape [batch_size, max_ppr_nodes].
            # cols is flat; we slice it per seed using valid_counts to get each seed's
            # local neighbor indices.
            ppr_neighbor_ids_local = torch.full_like(homo_neighbor_ids, -1)
            offset = 0
            for i, count in enumerate(valid_counts.tolist()):
                count = int(count)
                ppr_neighbor_ids_local[i, :count] = cols[offset : offset + count]
                offset += count

            metadata["ppr_weights"] = homo_weights
            metadata["ppr_neighbor_ids"] = ppr_neighbor_ids_local

            sample_output = SamplerOutput(
                node=all_nodes,
                row=torch.tensor([], dtype=torch.long, device=self.device),
                col=torch.tensor([], dtype=torch.long, device=self.device),
                edge=torch.tensor(
                    [], dtype=torch.long, device=self.device
                ),  # Empty tensor — GLT SampleQueue requires all values to be tensors
                batch=input_seeds,
                num_sampled_nodes=[srcs.size(0), new_nodes.size(0)],
                num_sampled_edges=[],
                metadata=metadata,
            )

        self.inducer_pool.put(inducer)
        return sample_output
