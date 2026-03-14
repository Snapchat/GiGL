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

# Sentinel type names for homogeneous graphs.  The PPR algorithm uses
# dict[NodeType, ...] internally for both homo and hetero graphs; these
# sentinels let the homogeneous path reuse the same dict-based code.
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
        num_nbrs_per_hop: Maximum number of neighbors to fetch per hop.
        total_degree_dtype: Dtype for precomputed total-degree tensors. Defaults to
            ``torch.int32``, which supports total degrees up to ~2 billion. Use a
            larger dtype if nodes have exceptionally high aggregate degrees.
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.5,
        eps: float = 1e-4,
        max_ppr_nodes: int = 50,
        num_nbrs_per_hop: int = 100000,
        total_degree_dtype: torch.dtype = torch.int32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._eps = eps
        self._max_ppr_nodes = max_ppr_nodes
        self._requeue_threshold_factor = alpha * eps
        self._num_nbrs_per_hop = num_nbrs_per_hop

        assert isinstance(
            self.data, DistDataset
        ), "DistPPRNeighborSampler requires a GiGL DistDataset to access degree tensors."
        degree_tensors = self.data.degree_tensor

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

        # Precompute total degree per node type: the sum of degrees across all
        # edge types traversable from that node type.  This is a graph-level
        # property used on every PPR iteration, so computing it once at init
        # avoids per-node summation and cache lookups in the hot loop.
        self._total_degree_by_node_type: dict[
            NodeType, torch.Tensor
        ] = self._build_total_degree_tensors(degree_tensors, total_degree_dtype)

    def _build_total_degree_tensors(
        self,
        degree_tensors: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
        dtype: torch.dtype,
    ) -> dict[NodeType, torch.Tensor]:
        """Build total-degree tensors by summing per-edge-type degrees for each node type.

        For homogeneous graphs, the total degree is just the single degree tensor.
        For heterogeneous graphs, it sums degree tensors across all edge types
        traversable from each node type, padding shorter tensors with zeros.

        Args:
            degree_tensors: Per-edge-type degree tensors from the dataset.
            dtype: Dtype for the output tensors.

        Returns:
            Dict mapping node type to a 1-D tensor of total degrees.
        """
        result: dict[NodeType, torch.Tensor] = {}

        if self._is_homogeneous:
            assert isinstance(degree_tensors, torch.Tensor)
            result[_PPR_HOMOGENEOUS_NODE_TYPE] = degree_tensors.to(dtype)
        else:
            assert isinstance(degree_tensors, dict)
            for node_type, edge_types in self._node_type_to_edge_types.items():
                max_len = 0
                for et in edge_types:
                    if et not in degree_tensors:
                        raise ValueError(
                            f"Edge type {et} not found in degree tensors. "
                            f"Available: {list(degree_tensors.keys())}"
                        )
                    max_len = max(max_len, len(degree_tensors[et]))

                summed = torch.zeros(max_len, dtype=dtype)
                for et in edge_types:
                    et_degrees = degree_tensors[et]
                    summed[: len(et_degrees)] += et_degrees.to(dtype)
                result[node_type] = summed

        return result

    def _get_total_degree(self, node_id: int, node_type: NodeType) -> int:
        """Look up the precomputed total degree of a node.

        Args:
            node_id: The ID of the node to look up.
            node_type: The node type.

        Returns:
            The total degree (sum across all edge types) for the node.

        Raises:
            ValueError: If the node ID is out of range, indicating corrupted
                graph data or a sampler bug.
        """
        degree_tensor = self._total_degree_by_node_type[node_type]
        if node_id >= len(degree_tensor):
            raise ValueError(
                f"Node ID {node_id} exceeds total degree tensor length "
                f"({len(degree_tensor)}) for node type {node_type}."
            )
        return int(degree_tensor[node_id].item())

    def _get_destination_type(self, edge_type: EdgeType) -> NodeType:
        """Get the node type at the destination end of an edge type."""
        return edge_type[0] if self.edge_dir == "in" else edge_type[-1]

    async def _batch_fetch_neighbors(
        self,
        nodes_by_edge_type: dict[EdgeType, set[int]],
        neighbor_target: dict[tuple[int, EdgeType], list[int]],
        device: torch.device,
    ) -> None:
        """
        Batch fetch neighbors for nodes grouped by edge type.

        Fetches neighbors for all nodes in nodes_by_edge_type, populating
        neighbor_target with neighbor lists. Degrees are looked up separately
        from the in-memory degree_tensors.

        Args:
            nodes_by_edge_type: Dict mapping edge type to set of node IDs to fetch
            neighbor_target: Dict to populate with (node_id, edge_type) -> neighbor list
            device: Torch device for tensor creation
        """
        for etype, node_ids in nodes_by_edge_type.items():
            if not node_ids:
                continue
            nodes_list = list(node_ids)
            lookup_tensor = torch.tensor(nodes_list, dtype=torch.long, device=device)

            # _sample_one_hop expects None for homogeneous graphs, not the PPR sentinel.
            output: NeighborOutput = await self._sample_one_hop(
                srcs=lookup_tensor,
                num_nbr=self._num_nbrs_per_hop,
                etype=etype if etype != _PPR_HOMOGENEOUS_EDGE_TYPE else None,
            )
            neighbors = output.nbr
            neighbor_counts = output.nbr_num

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

    async def _compute_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: Optional[NodeType] = None,
    ) -> tuple[
        Union[torch.Tensor, dict[NodeType, torch.Tensor]],
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
            2. Push residual + re-queue (single pass): For each queued node, add its
               residual to its PPR score, reset its residual to zero, then distribute
               (1-alpha) * residual to all neighbors proportionally by degree. After
               each push, immediately check if the neighbor's accumulated residual
               exceeds alpha * eps * total_degree; if so, add it to the queue for
               the next iteration. Total degree lookups are cached across the entire
               PPR computation to avoid redundant summation.

        Args:
            seed_nodes: Tensor of seed node IDs [batch_size]
            seed_node_type: Node type of seed nodes. Should be None for homogeneous graphs.

        Returns:
            tuple of (flat_neighbor_ids, flat_weights, valid_counts) where each is either
            a 1-D tensor (homogeneous) or a dict mapping NodeType to a 1-D tensor
            (heterogeneous):
                - flat_neighbor_ids: global neighbor IDs in top-k order, concatenated
                  across all seeds. Length equals sum(valid_counts).
                - flat_weights: corresponding PPR scores, same length as flat_neighbor_ids.
                - valid_counts: number of PPR neighbors found per seed [batch_size].
        """
        if seed_node_type is None:
            seed_node_type = _PPR_HOMOGENEOUS_NODE_TYPE
        device = seed_nodes.device
        batch_size = seed_nodes.size(0)

        # Per-seed PPR state, nested by node type for efficient type-grouped access.
        #
        # ppr_scores[i][node_type][node_id] = accumulated PPR score for node_id
        #   of type node_type, relative to seed i.  Updated each iteration by
        #   absorbing the node's residual.
        #
        # residuals[i][node_type][node_id] = unconverged probability mass at node_id
        #   of type node_type for seed i.  Each iteration, a node's residual is
        #   absorbed into its PPR score and then distributed to its neighbors.
        #
        # queue[i][node_type] = set of node IDs whose residual exceeds the
        #   convergence threshold (alpha * eps * total_degree).  The algorithm
        #   terminates when all queues are empty.  A set is used because multiple
        #   neighbors can push residual to the same node in one iteration —
        #   deduplication avoids redundant processing, and the O(1) membership
        #   check matters since it runs in the innermost loop.
        ppr_scores: list[dict[NodeType, dict[int, float]]] = [
            defaultdict(lambda: defaultdict(float)) for _ in range(batch_size)
        ]
        residuals: list[dict[NodeType, dict[int, float]]] = [
            defaultdict(lambda: defaultdict(float)) for _ in range(batch_size)
        ]
        queue: list[dict[NodeType, set[int]]] = [
            defaultdict(set) for _ in range(batch_size)
        ]

        seed_list = seed_nodes.tolist()

        for i, seed in enumerate(seed_list):
            residuals[i][seed_node_type][seed] = self._alpha
            queue[i][seed_node_type].add(seed)

        # Cache keyed by (node_id, edge_type) since same node can have different neighbors per edge type
        neighbor_cache: dict[tuple[int, EdgeType], list[int]] = {}

        num_nodes_in_queue = batch_size
        one_minus_alpha = 1 - self._alpha

        while num_nodes_in_queue > 0:
            # Drain all nodes from all queues and group by edge type for batched lookups
            queued_nodes: list[dict[NodeType, set[int]]] = [
                defaultdict(set) for _ in range(batch_size)
            ]
            nodes_by_edge_type: dict[EdgeType, set[int]] = defaultdict(set)

            for i in range(batch_size):
                if queue[i]:
                    queued_nodes[i] = queue[i]
                    queue[i] = defaultdict(set)
                    for node_type, node_ids in queued_nodes[i].items():
                        num_nodes_in_queue -= len(node_ids)
                        edge_types_for_node = self._node_type_to_edge_types[node_type]
                        for node_id in node_ids:
                            for etype in edge_types_for_node:
                                cache_key = (node_id, etype)
                                if cache_key not in neighbor_cache:
                                    nodes_by_edge_type[etype].add(node_id)

            await self._batch_fetch_neighbors(
                nodes_by_edge_type, neighbor_cache, device
            )

            # Push residual to neighbors and re-queue in a single pass.  This
            # is safe because each seed's state is independent, and residuals
            # are always positive so the merged loop can never miss a re-queue.
            for i in range(batch_size):
                for source_type, source_nodes in queued_nodes[i].items():
                    for source_node in source_nodes:
                        source_residual = residuals[i][source_type].get(
                            source_node, 0.0
                        )

                        ppr_scores[i][source_type][source_node] += source_residual
                        residuals[i][source_type][source_node] = 0.0

                        edge_types_for_node = self._node_type_to_edge_types[source_type]

                        total_degree = self._get_total_degree(source_node, source_type)

                        if total_degree == 0:
                            continue

                        residual_per_neighbor = (
                            one_minus_alpha * source_residual / total_degree
                        )

                        for etype in edge_types_for_node:
                            cache_key = (source_node, etype)
                            neighbor_list = neighbor_cache[cache_key]
                            if not neighbor_list:
                                continue

                            neighbor_type = self._get_destination_type(etype)

                            for neighbor_node in neighbor_list:
                                residuals[i][neighbor_type][
                                    neighbor_node
                                ] += residual_per_neighbor

                                requeue_threshold = (
                                    self._requeue_threshold_factor
                                    * self._get_total_degree(
                                        neighbor_node, neighbor_type
                                    )
                                )
                                should_requeue = (
                                    neighbor_node not in queue[i][neighbor_type]
                                    and residuals[i][neighbor_type][neighbor_node]
                                    >= requeue_threshold
                                )
                                if should_requeue:
                                    queue[i][neighbor_type].add(neighbor_node)
                                    num_nodes_in_queue += 1

        # Extract top-k nodes by PPR score, grouped by node type.
        # Build flat tensors directly (no padding) — valid_counts[i] records how many
        # neighbors seed i actually has, so callers can recover per-seed slices.
        all_node_types = self._node_type_to_edge_types.keys()

        flat_ids_by_ntype: dict[NodeType, torch.Tensor] = {}
        flat_weights_by_ntype: dict[NodeType, torch.Tensor] = {}
        valid_counts_by_ntype: dict[NodeType, torch.Tensor] = {}

        for ntype in all_node_types:
            flat_ids: list[int] = []
            flat_weights: list[float] = []
            valid_counts: list[int] = []

            for i in range(batch_size):
                type_scores = ppr_scores[i].get(ntype, {})
                top_k = heapq.nlargest(
                    self._max_ppr_nodes, type_scores.items(), key=lambda x: x[1]
                )
                for node_id, weight in top_k:
                    flat_ids.append(node_id)
                    flat_weights.append(weight)
                valid_counts.append(len(top_k))

            flat_ids_by_ntype[ntype] = torch.tensor(
                flat_ids, dtype=torch.long, device=device
            )
            flat_weights_by_ntype[ntype] = torch.tensor(
                flat_weights, dtype=torch.float, device=device
            )
            valid_counts_by_ntype[ntype] = torch.tensor(
                valid_counts, dtype=torch.long, device=device
            )

        if self._is_homogeneous:
            assert (
                len(flat_ids_by_ntype) == 1
                and _PPR_HOMOGENEOUS_NODE_TYPE in flat_ids_by_ntype
            )
            return (
                flat_ids_by_ntype[_PPR_HOMOGENEOUS_NODE_TYPE],
                flat_weights_by_ntype[_PPR_HOMOGENEOUS_NODE_TYPE],
                valid_counts_by_ntype[_PPR_HOMOGENEOUS_NODE_TYPE],
            )
        else:
            return flat_ids_by_ntype, flat_weights_by_ntype, valid_counts_by_ntype

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
        edge types based on the current node type.

        Output format (PyG edge-index style, no padding):

        - ``ppr_neighbor_ids`` (homo) / ``ppr_neighbor_ids_{seed_type}_{ntype}`` (hetero):
          shape ``[2, num_edges]`` — row 0 is local seed indices, row 1 is local
          neighbor indices.  Both index into ``data[ntype].node``.
        - ``ppr_weights`` (homo) / ``ppr_weights_{seed_type}_{ntype}`` (hetero):
          shape ``[num_edges]`` — PPR score for each edge, aligned with the columns
          of ``ppr_neighbor_ids``.

        Local indices are produced by the inducer (see below), so row 1 of
        ``ppr_neighbor_ids`` directly indexes into ``data[ntype].x`` without any
        additional global→local remapping.

        **Why the inducer is used for local-index assignment:**

        The inducer is GLT's C++ data structure (backed by a per-node-type hash map)
        that maintains a single global-ID → local-index mapping for the entire
        subgraph being built.  We use it here instead of a Python dict for two reasons:

        1. **Consistency across seed types.** For heterogeneous ABLP inputs,
           ``_compute_ppr_scores`` is called once per seed type (anchors, supervision
           nodes, …).  A node reachable from multiple seed types must receive the
           *same* local index in ``node_dict[ntype]`` regardless of which seed type
           discovered it.  The inducer is shared across all those calls, so it
           guarantees this automatically.

        2. **Performance.** The inducer's C++ hash map is faster than a Python dict
           for per-node lookups on large graphs, and its lifecycle is already managed
           by GLT's inducer pool (``_acquire_inducer`` / ``inducer_pool.put``).

        The API used here mirrors GLT's own ``DistNeighborSampler._sample_from_nodes``:

        - ``inducer.init_node(seeds)`` registers seed nodes and returns their global
          IDs (local indices 0, 1, … are assigned internally).
        - ``inducer.induce_next(srcs, flat_nbrs, counts)`` (homo) or
          ``inducer.induce_next(nbr_dict)`` (hetero) deduplicates neighbors against
          all previously seen nodes and returns:

            - ``new_nodes``: global IDs of nodes not yet registered.
            - ``cols``: flat local destination indices for *every* neighbor edge,
              in the same order as the input ``flat_nbrs``.  Combined with
              ``repeat_interleave``-expanded seed indices, this forms the
              ``[2, num_edges]`` edge-index tensor directly.
        """
        sample_loop_inputs = self._prepare_sample_loop_inputs(inputs)
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type
        is_hetero = self.dist_graph.data_cls == "hetero"
        metadata = sample_loop_inputs.metadata
        nodes_to_sample = sample_loop_inputs.nodes_to_sample

        # Acquired once per sample; returned to the pool at the end.  The inducer
        # maintains the shared global→local index map for this entire subgraph.
        inducer = self._acquire_inducer()

        if is_hetero:
            assert isinstance(nodes_to_sample, dict)
            assert input_type is not None

            # Register all seeds (anchors + supervision nodes for ABLP) with the
            # inducer first, so they occupy the lowest local indices.  src_dict maps
            # NodeType -> global IDs (same values as nodes_to_sample).
            src_dict = inducer.init_node(nodes_to_sample)

            # Compute PPR for each seed type, collecting flat global neighbor IDs,
            # weights, and per-seed counts.  Build nbr_dict for a single
            # inducer.induce_next call using virtual edge types (seed_type, 'ppr', ntype)
            # — the inducer only cares about etype[0] and etype[-1] as source/dest
            # node types, so the relation name is arbitrary.
            nbr_dict: dict[EdgeType, list[torch.Tensor]] = {}
            all_flat_weights: dict[tuple[NodeType, NodeType], torch.Tensor] = {}
            all_valid_counts: dict[tuple[NodeType, NodeType], torch.Tensor] = {}

            for seed_type, seed_nodes in nodes_to_sample.items():
                (
                    flat_ids_by_type,
                    flat_weights_by_type,
                    valid_counts_by_type,
                ) = await self._compute_ppr_scores(seed_nodes, seed_type)
                assert isinstance(flat_ids_by_type, dict)
                assert isinstance(flat_weights_by_type, dict)
                assert isinstance(valid_counts_by_type, dict)

                for ntype, flat_ids in flat_ids_by_type.items():
                    valid_counts = valid_counts_by_type[ntype]
                    all_flat_weights[(seed_type, ntype)] = flat_weights_by_type[ntype]
                    all_valid_counts[(seed_type, ntype)] = valid_counts

                    # Skip empty pairs; induce_next handles deduplication across
                    # seed types so a neighbor reachable from multiple seed types
                    # gets one consistent local index in node_dict[ntype].
                    if flat_ids.numel() > 0:
                        virtual_etype: EdgeType = (seed_type, "ppr", ntype)
                        nbr_dict[virtual_etype] = [
                            src_dict[seed_type],
                            flat_ids,
                            valid_counts,
                        ]

            # induce_next assigns local indices to all neighbors not yet registered,
            # deduplicating across all virtual edge types in one pass.
            # new_nodes_dict: newly discovered global IDs per node type.
            # cols_dict: flat local destination indices per virtual edge type,
            #            in the same order the flat neighbors were provided.
            new_nodes_dict, _rows_dict, cols_dict = inducer.induce_next(nbr_dict)

            # node_dict = seeds (already in src_dict) + newly discovered PPR
            # neighbors.  merge_dict appends tensors into lists; cat collapses them.
            out_nodes_hetero: dict[NodeType, list[torch.Tensor]] = defaultdict(list)
            merge_dict(src_dict, out_nodes_hetero)
            merge_dict(new_nodes_dict, out_nodes_hetero)
            node_dict = {
                ntype: torch.cat(nodes)
                for ntype, nodes in out_nodes_hetero.items()
                if nodes
            }

            # Build PyG-style edge-index output per (seed_type, ntype) pair.
            # cols_dict[(seed_type, 'ppr', ntype)] gives flat local dst indices in
            # the same order as the flat neighbors passed to induce_next.
            # repeat_interleave expands seed local indices to match.
            for (seed_type, ntype), flat_weights in all_flat_weights.items():
                valid_counts = all_valid_counts[(seed_type, ntype)]
                virtual_etype = (seed_type, "ppr", ntype)
                cols = cols_dict.get(virtual_etype)
                if cols is not None:
                    seed_batch_size = nodes_to_sample[seed_type].size(0)
                    src_indices = torch.repeat_interleave(
                        torch.arange(seed_batch_size, device=self.device), valid_counts
                    )
                    ppr_edge_index = torch.stack([src_indices, cols])
                else:
                    ppr_edge_index = torch.zeros(
                        2, 0, dtype=torch.long, device=self.device
                    )
                    flat_weights = torch.zeros(0, dtype=torch.float, device=self.device)
                metadata[f"ppr_neighbor_ids_{seed_type}_{ntype}"] = ppr_edge_index
                metadata[f"ppr_weights_{seed_type}_{ntype}"] = flat_weights

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

            # Register seeds; local indices 0..N-1 are assigned internally.
            # srcs holds their global IDs (same values as nodes_to_sample).
            srcs = inducer.init_node(nodes_to_sample)

            (
                homo_flat_ids,
                homo_flat_weights,
                homo_valid_counts,
            ) = await self._compute_ppr_scores(nodes_to_sample, None)
            assert isinstance(homo_flat_ids, torch.Tensor)
            assert isinstance(homo_flat_weights, torch.Tensor)
            assert isinstance(homo_valid_counts, torch.Tensor)

            # induce_next deduplicates homo_flat_ids against already-seen nodes
            # (the seeds registered above) and returns:
            #   new_nodes: global IDs of nodes not yet registered.
            #   cols: flat local destination indices for every neighbor, in the
            #         same order as homo_flat_ids.
            new_nodes, _rows, cols = inducer.induce_next(
                srcs, homo_flat_ids, homo_valid_counts
            )
            all_nodes = torch.cat([srcs, new_nodes])

            # Build PyG-style edge-index: row 0 = local seed indices (expanded via
            # repeat_interleave), row 1 = local neighbor indices from inducer cols.
            src_indices = torch.repeat_interleave(
                torch.arange(nodes_to_sample.size(0), device=self.device),
                homo_valid_counts,
            )
            ppr_edge_index = torch.stack([src_indices, cols])

            metadata["ppr_neighbor_ids"] = ppr_edge_index
            metadata["ppr_weights"] = homo_flat_weights

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
