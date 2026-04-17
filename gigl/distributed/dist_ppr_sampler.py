# TODO (mkolodner-sc): The forward push loop in _compute_ppr_scores is the
# main throughput bottleneck — both the queue drain (preparing batched node
# lookups by edge type) and the residual push/requeue pass are pure Python
# dict/set operations in tight nested loops.  Moving these to a C++ extension
# (e.g. pybind11) would eliminate per-operation Python overhead and enable
# cache-friendly memory access patterns.

# TODO (mkolodner-sc): Investigate whether concurrency for _sample_one_hop and _compute_ppr_scores will
# yield performance benefits.

import heapq
from collections import defaultdict
from typing import Optional, Union

import torch
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NeighborOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import EdgeType, NodeType
from graphlearn_torch.utils import merge_dict

from gigl.distributed.base_sampler import BaseDistNeighborSampler
from gigl.types.graph import is_label_edge_type

# Trailing "." is an intentional separator.  These constants are used both to
# write metadata keys (f"{KEY}{repr(edge_type)}" → e.g. "ppr_edge_index.('user', 'to', 'story')")
# and as the strip prefix in extract_edge_type_metadata (key[len(prefix):] must
# yield a bare EdgeType repr for ast.literal_eval).
PPR_EDGE_INDEX_METADATA_KEY = "ppr_edge_index."
PPR_WEIGHT_METADATA_KEY = "ppr_weight."

# Sentinel type names for homogeneous graphs.  The PPR algorithm uses
# dict[NodeType, ...] internally for both homo and hetero graphs; these
# sentinels let the homogeneous path reuse the same dict-based code.
# TODO (mkolodner-sc): The sentinel approach adds an extra dict lookup on
# every operation in the hot loop for homogeneous graphs (always resolving
# the same single key).  Profile whether this overhead is meaningful
# compared to the neighbor fetch and residual update costs, and consider
# splitting into separate homo/hetero loop implementations if so.
_PPR_HOMOGENEOUS_NODE_TYPE = "ppr_homogeneous_node_type"
_PPR_HOMOGENEOUS_EDGE_TYPE = (
    _PPR_HOMOGENEOUS_NODE_TYPE,
    "to",
    _PPR_HOMOGENEOUS_NODE_TYPE,
)


class DistPPRNeighborSampler(BaseDistNeighborSampler):
    """Personalized PageRank (PPR) based distributed neighbor sampler.

    Extends BaseGiGLSampler (which provides shared input preparation utilities)
    and overrides _sample_from_nodes with PPR-based neighbor selection.

    Instead of uniform random sampling, this sampler uses Personalized PageRank
    (PPR) scores to select the most relevant neighbors for each seed node. PPR
    scores are approximated here using the Forward Push algorithm (Andersen et
    al., 2006).

    This sampler supports both homogeneous and heterogeneous graphs. For heterogeneous graphs,
    the PPR algorithm traverses across all edge types, switching edge types based on the
    current node type and the configured edge direction.

    The ``edge_index`` and ``edge_attr`` fields on the output Data/HeteroData
    objects are populated with PPR seed-to-neighbor relationships (not edges
    in the original graph). ``N`` is the total number of (seed, neighbor)
    pairs across all seeds in the batch.

    **Homogeneous (Data):**
        - ``data.edge_index``: ``[2, N]`` int64 — row 0 is local seed indices,
          row 1 is local neighbor indices.
        - ``data.edge_attr``: ``[N]`` float — PPR score for each pair.

    **Heterogeneous (HeteroData)** — one PPR edge type per
    ``(seed_type, neighbor_type)`` pair, with ``"ppr"`` as the relation:
        - ``data[(seed_type, "ppr", ntype)].edge_index``: same format as above.
        - ``data[(seed_type, "ppr", ntype)].edge_attr``: same format as above.

    Args:
        alpha: Restart probability (teleport probability back to seed). Higher values
               keep samples closer to seeds. Typical values: 0.15-0.25.
        eps: Convergence threshold. Smaller values give more accurate PPR scores
             but require more computation. Typical values: 1e-4 to 1e-6.
        max_ppr_nodes: Maximum number of nodes to return per seed based on PPR scores.
        num_neighbors_per_hop: Maximum number of neighbors to fetch per hop.
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
        num_neighbors_per_hop: int = 100_000,
        total_degree_dtype: torch.dtype = torch.int32,
        degree_tensors: Union[torch.Tensor, dict[EdgeType, torch.Tensor]],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._eps = eps
        self._max_ppr_nodes = max_ppr_nodes
        self._requeue_threshold_factor = alpha * eps
        self._num_neighbors_per_hop = num_neighbors_per_hop

        # Build mapping from node type to edge types that can be traversed from that node type.
        self._node_type_to_edge_types: dict[NodeType, list[EdgeType]] = defaultdict(
            list
        )

        # GLT's DistNeighborSampler only sets self.edge_types for heterogeneous
        # graphs (when dist_graph.data_cls == 'hetero'), so we use that as the
        # heterogeneity check.
        if self.dist_graph.data_cls == "hetero":
            self._is_homogeneous = False
            # Heterogeneous case: map each node type to its outgoing/incoming edge types.
            # Label edge types (injected by ABLP for supervision) are excluded: including
            # them would let PPR walks cross label edges, leaking ground-truth targets into
            # the sampled neighborhood.
            for etype in self.edge_types:
                if is_label_edge_type(etype):
                    continue
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
        # TODO (mkolodner-sc): This trades memory for throughput — we
        # materialize a tensor per node type to avoid recomputing total degree
        # on every neighbor during sampling.  Computing it here (rather than in
        # the dataset) also keeps the door open for edge-specific degree
        # strategies.  If memory becomes a bottleneck, revisit this.
        self._node_type_to_total_degree: dict[NodeType, torch.Tensor] = (
            self._build_total_degree_tensors(degree_tensors, total_degree_dtype)
        )

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
            # Single edge type: degree values fit directly in the target dtype.
            result[_PPR_HOMOGENEOUS_NODE_TYPE] = degree_tensors.to(dtype)
        else:
            assert isinstance(degree_tensors, dict)
            dtype_max = torch.iinfo(dtype).max
            for node_type, edge_types in self._node_type_to_edge_types.items():
                max_len = 0
                for et in edge_types:
                    if et not in degree_tensors:
                        raise ValueError(
                            f"Edge type {et} not found in degree tensors. "
                            f"Available: {list(degree_tensors.keys())}"
                        )
                    max_len = max(max_len, len(degree_tensors[et]))

                # Each degree tensor is indexed by node ID (derived from CSR
                # indptr), so index i in every edge type's tensor refers to
                # the same node.  Element-wise summation gives the total degree
                # per node across all edge types.  Shorter tensors are padded
                # implicitly (only the first len(et_degrees) entries are added).
                # Sum in int64: aggregate degrees are bounded by partition size
                # and fit comfortably within int64 range in practice.
                summed = torch.zeros(max_len, dtype=torch.int64)
                for et in edge_types:
                    et_degrees = degree_tensors[et]
                    summed[: len(et_degrees)] += et_degrees.to(torch.int64)
                result[node_type] = summed.clamp(max=dtype_max).to(dtype)

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
        # Destination-only node types (no outgoing edges) are absent from
        # _node_type_to_total_degree because total degree is only computed for
        # traversable source types.  Returning 0 here is correct: such nodes
        # act as terminals — they accumulate PPR score but never push residual
        # further.
        if node_type not in self._node_type_to_total_degree:
            return 0
        degree_tensor = self._node_type_to_total_degree[node_type]
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
        nodes_to_lookup: dict[EdgeType, set[int]],
        device: torch.device,
    ) -> dict[tuple[int, EdgeType], list[int]]:
        """Batch fetch neighbors for nodes grouped by edge type.

        Issues one ``_sample_one_hop`` call per edge type (not per node), so all
        nodes of the same edge type are fetched in a single RPC round-trip. Each
        node's neighbor list is capped at ``self._num_neighbors_per_hop``.

        Args:
            nodes_to_lookup: Dict mapping each edge type to the set of node IDs
                whose neighbors should be fetched via that edge type.  Only nodes
                absent from the caller's ``neighbor_cache`` should be included.
            device: Torch device for intermediate tensor creation.

        Returns:
            Dict mapping ``(node_id, edge_type)`` to the list of neighbor node IDs
            returned by ``_sample_one_hop``.  Only nodes that appeared in
            ``nodes_to_lookup`` are present; edge types with an empty node set are
            skipped entirely.

        Example::

            nodes_to_lookup = {
                ("user", "buys", "item"): {0, 3},
                ("item", "bought_by", "user"): {7},
            }
            # Might return (neighbor lists depend on graph structure):
            {
                (0, ("user", "buys", "item")): [5, 9, 2],
                (3, ("user", "buys", "item")): [1],
                (7, ("item", "bought_by", "user")): [0, 3],
            }
        """
        result: dict[tuple[int, EdgeType], list[int]] = {}
        for etype, node_ids in nodes_to_lookup.items():
            if not node_ids:
                continue
            nodes_list = list(node_ids)
            lookup_tensor = torch.tensor(nodes_list, dtype=torch.long, device=device)

            # _sample_one_hop expects None for homogeneous graphs, not the PPR sentinel.
            output: NeighborOutput = await self._sample_one_hop(
                srcs=lookup_tensor,
                num_nbr=self._num_neighbors_per_hop,
                etype=etype if etype != _PPR_HOMOGENEOUS_EDGE_TYPE else None,
            )
            neighbors = output.nbr
            neighbor_counts = output.nbr_num

            # TODO (mkolodner-sc): Investigate performance of a vectorized version of the below code
            neighbors_list = neighbors.tolist()
            counts_list = neighbor_counts.tolist()
            del neighbors, neighbor_counts

            # neighbors_list is a flat concatenation of all neighbors for all looked-up nodes.
            # We use offset to slice out each node's neighbors: node i's neighbors are at
            # neighbors_list[offset : offset + count], then we advance offset by count.
            offset = 0
            for node_id, count in zip(nodes_list, counts_list):
                result[(node_id, etype)] = neighbors_list[offset : offset + count]
                offset += count

        return result

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
            seed_nodes: Tensor of seed node IDs, shape ``[batch_size]``.
            seed_node_type: Node type of seed nodes.  Pass ``None`` for
                homogeneous graphs (internally mapped to a sentinel type).

        Returns:
            A 3-tuple ``(flat_neighbor_ids, flat_weights, valid_counts)``.
            For homogeneous graphs each element is a 1-D tensor; for
            heterogeneous graphs each element is a ``dict[NodeType, Tensor]``
            where each tensor has the same structure as the homogeneous case.

            - ``flat_neighbor_ids``: global neighbor IDs selected by top-k PPR
              score, concatenated across seeds.  For batch of size ``B`` with
              ``C_i`` neighbors for seed ``i``, shape is
              ``[sum(C_0, ..., C_{B-1})]``.
            - ``flat_weights``: PPR scores corresponding to each entry in
              ``flat_neighbor_ids``, same shape.
            - ``valid_counts``: number of PPR neighbors contributed by each
              seed, shape ``[batch_size]``.  Used to slice the flat tensors into
              per-seed groups: seed ``i``'s neighbors are at
              ``flat_neighbor_ids[sum(valid_counts[:i]) : sum(valid_counts[:i+1])]``.

        Example::

            # 4 seeds, valid_counts = [1, 3, 2, 0]  →  6 total (seed, neighbor) pairs
            flat_neighbor_ids = tensor([d0, d1a, d1b, d1c, d2a, d2b])
            flat_weights      = tensor([w0, w1a, w1b, w1c, w2a, w2b])
            valid_counts      = tensor([1,  3,   2,   0])
        """
        if seed_node_type is None:
            seed_node_type = _PPR_HOMOGENEOUS_NODE_TYPE
        device = seed_nodes.device
        batch_size = seed_nodes.size(0)

        # Per-seed PPR state, nested by node type for efficient type-grouped access.

        # ppr_scores[i][node_type][node_id] = accumulated PPR score for node_id
        # of type node_type, relative to seed i.  Updated each iteration by
        # absorbing the node's residual.
        ppr_scores: list[dict[NodeType, dict[int, float]]] = [
            defaultdict(lambda: defaultdict(float)) for _ in range(batch_size)
        ]

        # residuals[i][node_type][node_id] = unconverged probability mass at node_id
        # of type node_type for seed i.  Each iteration, a node's residual is
        # absorbed into its PPR score and then distributed to its neighbors.
        residuals: list[dict[NodeType, dict[int, float]]] = [
            defaultdict(lambda: defaultdict(float)) for _ in range(batch_size)
        ]

        # queue[i][node_type] = set of node IDs whose residual exceeds the
        # convergence threshold (alpha * eps * total_degree).  The algorithm
        # terminates when all queues are empty.  A set is used because multiple
        # neighbors can push residual to the same node in one iteration —
        # deduplication avoids redundant processing, and the O(1) membership
        # check matters since it runs in the innermost loop.
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
            nodes_to_lookup: dict[EdgeType, set[int]] = defaultdict(set)

            for seed_idx in range(batch_size):
                if queue[seed_idx]:
                    queued_nodes[seed_idx] = queue[seed_idx]
                    queue[seed_idx] = defaultdict(set)
                    for node_type, node_ids in queued_nodes[seed_idx].items():
                        num_nodes_in_queue -= len(node_ids)
                        # We fetch neighbors for ALL edge types originating
                        # from this node type, not just the edge type that
                        # caused the node to be queued.  This is required for
                        # correctness: forward push distributes residual to
                        # all neighbors proportionally by total degree, so
                        # every edge type must be considered.
                        # Destination-only types have no entry in _node_type_to_edge_types;
                        # .get() returns [] so we skip neighbor lookup for them.
                        edge_types_for_node = self._node_type_to_edge_types.get(
                            node_type, []
                        )
                        for node_id in node_ids:
                            for etype in edge_types_for_node:
                                cache_key = (node_id, etype)
                                if cache_key not in neighbor_cache:
                                    # TODO (mkolodner-sc): Investigate switching from set to list
                                    # here.  _sample_one_hop handles duplicates correctly (second
                                    # write to result[(node_id, etype)] is a no-op overwrite), so
                                    # dedup is not required for correctness.  A list would avoid
                                    # per-add hash cost and the set->list->tensor conversion in
                                    # _batch_fetch_neighbors, though at the cost of redundant
                                    # network calls for any duplicate nodes across seeds.
                                    nodes_to_lookup[etype].add(node_id)

            fetched_neighbors = await self._batch_fetch_neighbors(
                nodes_to_lookup=nodes_to_lookup,
                device=device,
            )
            # fetched_neighbors is intentionally NOT merged into neighbor_cache
            # upfront.  We only promote entries when a node is requeued — see
            # the should_requeue block below.

            # Push residual to neighbors and re-queue in a single pass.  This
            # is safe because each seed's state is independent, and residuals
            # are always positive so the merged loop can never miss a re-queue.
            for seed_idx in range(batch_size):
                for source_type, source_nodes in queued_nodes[seed_idx].items():
                    for source_node in source_nodes:
                        source_residual = residuals[seed_idx][source_type].get(
                            source_node, 0.0
                        )

                        ppr_scores[seed_idx][source_type][source_node] += (
                            source_residual
                        )
                        residuals[seed_idx][source_type][source_node] = 0.0

                        # Same destination-only guard as in the queue drain loop above.
                        edge_types_for_node = self._node_type_to_edge_types.get(
                            source_type, []
                        )

                        total_degree = self._get_total_degree(source_node, source_type)

                        if total_degree == 0:
                            continue

                        residual_per_neighbor = (
                            one_minus_alpha * source_residual / total_degree
                        )

                        for etype in edge_types_for_node:
                            cache_key = (source_node, etype)
                            # fetched_neighbors and neighbor_cache are mutually
                            # exclusive per iteration: the queue drain only adds
                            # a node to nodes_to_lookup if it is absent from
                            # neighbor_cache, so a key appears in at most one.
                            neighbor_list = fetched_neighbors.get(
                                cache_key, neighbor_cache.get(cache_key, [])
                            )
                            if not neighbor_list:
                                continue

                            neighbor_type = self._get_destination_type(etype)

                            for neighbor_node in neighbor_list:
                                residuals[seed_idx][neighbor_type][neighbor_node] += (
                                    residual_per_neighbor
                                )

                                requeue_threshold = (
                                    self._requeue_threshold_factor
                                    * self._get_total_degree(
                                        neighbor_node, neighbor_type
                                    )
                                )
                                should_requeue = (
                                    neighbor_node not in queue[seed_idx][neighbor_type]
                                    and residuals[seed_idx][neighbor_type][
                                        neighbor_node
                                    ]
                                    >= requeue_threshold
                                )
                                if should_requeue:
                                    queue[seed_idx][neighbor_type].add(neighbor_node)
                                    num_nodes_in_queue += 1
                                    # Promote this node's neighbor lists to the
                                    # persistent cache: it will be processed next
                                    # iteration, so caching now avoids a re-fetch.
                                    # Nodes that are never requeued (typically
                                    # high-degree) are never promoted, keeping
                                    # their large neighbor lists out of the cache.
                                    for (
                                        promote_etype
                                    ) in self._node_type_to_edge_types.get(
                                        neighbor_type, []
                                    ):
                                        promote_key = (neighbor_node, promote_etype)
                                        if (
                                            promote_key in fetched_neighbors
                                            and promote_key not in neighbor_cache
                                        ):
                                            neighbor_cache[promote_key] = (
                                                fetched_neighbors[promote_key]
                                            )

        # Extract top-k nodes by PPR score, grouped by node type.
        # Results are three flat tensors per node type (no padding):
        #   - flat_ids:      [id_seed0_0, id_seed0_1, ..., id_seed1_0, ...]
        #   - flat_weights:  [wt_seed0_0, wt_seed0_1, ..., wt_seed1_0, ...]
        #   - valid_counts:  [count_seed0, count_seed1, ...]
        #
        # valid_counts[i] records how many top-k neighbors seed i contributed.
        # The inducer uses valid_counts to slice flat_ids into per-seed groups
        # and assign local indices.  Example:
        #
        #   4 seeds, valid_counts = [1, 6, 2, 1]  (10 total pairs)
        #   flat_ids = [d0a, d1a, d1b, d1c, d1d, d1e, d1f, d2a, d2b, d3a]
        #
        #   seed 0 owns flat_ids[0:1],  seed 1 owns flat_ids[1:7],
        #   seed 2 owns flat_ids[7:9],  seed 3 owns flat_ids[9:10]
        # _node_type_to_edge_types only contains source types; destination-only
        # types are absent but may have accumulated PPR scores during the walk.
        # We union with all types seen in ppr_scores so they appear in the output.
        all_node_types: set[NodeType] = set(self._node_type_to_edge_types.keys())
        for seed_ppr in ppr_scores:
            all_node_types.update(seed_ppr.keys())

        ntype_to_flat_ids: dict[NodeType, torch.Tensor] = {}
        ntype_to_flat_weights: dict[NodeType, torch.Tensor] = {}
        ntype_to_valid_counts: dict[NodeType, torch.Tensor] = {}

        for ntype in all_node_types:
            flat_ids: list[int] = []
            flat_weights: list[float] = []
            valid_counts: list[int] = []

            for i in range(batch_size):
                type_scores = ppr_scores[i].get(ntype, {})
                top_k = heapq.nlargest(
                    self._max_ppr_nodes, type_scores.items(), key=lambda x: x[1]
                )
                if top_k:
                    ids, weights = zip(*top_k)
                    flat_ids.extend(ids)
                    flat_weights.extend(weights)
                valid_counts.append(len(top_k))

            ntype_to_flat_ids[ntype] = torch.tensor(
                flat_ids, dtype=torch.long, device=device
            )
            ntype_to_flat_weights[ntype] = torch.tensor(
                flat_weights, dtype=torch.float, device=device
            )
            ntype_to_valid_counts[ntype] = torch.tensor(
                valid_counts, dtype=torch.long, device=device
            )

        if self._is_homogeneous:
            assert (
                len(ntype_to_flat_ids) == 1
                and _PPR_HOMOGENEOUS_NODE_TYPE in ntype_to_flat_ids
            )
            return (
                ntype_to_flat_ids[_PPR_HOMOGENEOUS_NODE_TYPE],
                ntype_to_flat_weights[_PPR_HOMOGENEOUS_NODE_TYPE],
                ntype_to_valid_counts[_PPR_HOMOGENEOUS_NODE_TYPE],
            )
        else:
            return ntype_to_flat_ids, ntype_to_flat_weights, ntype_to_valid_counts

    async def _sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
    ) -> Union[SamplerOutput, HeteroSamplerOutput]:
        """
        Override the base sampling method to use PPR-based neighbor selection.

        Supports both NodeSamplerInput and ABLPNodeSamplerInput. For ABLP, PPR
        scores are computed from both anchor and supervision nodes, so the sampled
        subgraph includes neighbors relevant to all seed types.

        For heterogeneous graphs, PPR traverses across all edge types, switching
        edge types based on the current node type.

        See the class docstring for the output format (``edge_index`` and
        ``edge_attr`` fields on the output Data/HeteroData).

        Local indices are produced by the inducer (see below), so row 1 of
        ``edge_index`` directly indexes into ``data[ntype].x`` without any
        additional global→local remapping.

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

            - ``new_nodes``: global IDs of nodes not previously registered
              with the inducer (i.e., not seeds and not returned by a prior
              ``induce_next`` call).
            - ``rows``: flat local source indices, expanded to match ``flat_nbrs``.
            - ``cols``: flat local destination indices for every neighbor,
              in the same order as ``flat_nbrs``.  Together, ``rows`` and
              ``cols`` form the ``[2, num_edges]`` edge-index tensor directly.
        """
        sample_loop_inputs = self._prepare_sample_loop_inputs(inputs)
        input_seeds = inputs.node.to(self.device)
        input_type = inputs.input_type
        is_hetero = self.dist_graph.data_cls == "hetero"
        metadata = sample_loop_inputs.metadata
        nodes_to_sample = sample_loop_inputs.nodes_to_sample

        # The inducer is GLT's C++ data structure that maintains a global-ID →
        # local-index mapping for the subgraph being built.  It serves two roles:
        #
        # 1. Deduplication: when the same global node ID appears from multiple
        #    seeds or seed types, induce_next assigns it a single local index.
        #    This ensures node_dict[ntype] has no duplicates.
        #
        # 2. Local index assignment: init_node registers seeds at local indices
        #    0..N-1.  induce_next then assigns the next available indices to
        #    neighbors not previously registered with the inducer.  The
        #    returned "cols" tensor contains the local destination index for
        #    every neighbor (including those already registered), which we
        #    use directly as row 1 of the PyG edge-index tensor.
        #
        # Acquired once per sample call; returned to the pool at the end.
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
            # inducer.induce_next call using PPR edge types (seed_type, 'ppr', ntype)
            # — the inducer only cares about etype[0] and etype[-1] as source/dest
            # node types, so the relation name is arbitrary.
            nbr_dict: dict[EdgeType, list[torch.Tensor]] = {}
            ppr_edge_type_to_flat_weights: dict[EdgeType, torch.Tensor] = {}

            for seed_type, seed_nodes in nodes_to_sample.items():
                (
                    ntype_to_flat_ids,
                    ntype_to_flat_weights,
                    ntype_to_valid_counts,
                ) = await self._compute_ppr_scores(seed_nodes, seed_type)
                assert isinstance(ntype_to_flat_ids, dict)
                assert isinstance(ntype_to_flat_weights, dict)
                assert isinstance(ntype_to_valid_counts, dict)

                for ntype, flat_ids in ntype_to_flat_ids.items():
                    ppr_edge_type: EdgeType = (seed_type, "ppr", ntype)
                    valid_counts = ntype_to_valid_counts[ntype]
                    ppr_edge_type_to_flat_weights[ppr_edge_type] = (
                        ntype_to_flat_weights[ntype]
                    )

                    # Skip empty pairs; induce_next handles deduplication across
                    # seed types so a neighbor reachable from multiple seed types
                    # gets one consistent local index in node_dict[ntype].
                    if flat_ids.numel() > 0:
                        nbr_dict[ppr_edge_type] = [
                            src_dict[seed_type],
                            flat_ids,
                            valid_counts,
                        ]

            # induce_next processes all PPR edge types in nbr_dict in one
            # pass, assigning local indices to neighbors not yet registered and
            # deduplicating nodes seen from multiple seed types.  Returns:
            #   new_nodes_dict[NodeType] -> global IDs of nodes not previously
            #                              registered with the inducer
            #   rows_dict[EdgeType]     -> flat local source indices per virtual
            #                              edge type, expanded to match flat_ids
            #   cols_dict[EdgeType]     -> flat local destination indices, one
            #                              per neighbor in the same order as the
            #                              flat_ids passed in nbr_dict
            new_nodes_dict, rows_dict, cols_dict = inducer.induce_next(nbr_dict)

            # node_dict = seeds (already in src_dict) + PPR neighbors not
            # previously registered.  merge_dict appends tensors into lists;
            # cat collapses them.
            out_nodes_hetero: dict[NodeType, list[torch.Tensor]] = defaultdict(list)
            merge_dict(src_dict, out_nodes_hetero)
            merge_dict(new_nodes_dict, out_nodes_hetero)
            node_dict = {
                ntype: torch.cat(nodes)
                for ntype, nodes in out_nodes_hetero.items()
                if nodes
            }

            # Build PyG-style edge-index output per PPR edge type.
            # rows_dict and cols_dict are keyed by PPR edge type and give
            # flat local source/destination indices respectively, aligned with
            # the flat_ids order passed to induce_next.
            for ppr_edge_type, flat_weights in ppr_edge_type_to_flat_weights.items():
                rows = rows_dict.get(ppr_edge_type)
                cols = cols_dict.get(ppr_edge_type)
                if rows is not None and cols is not None:
                    edge_index = torch.stack([rows, cols])
                else:
                    edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                    flat_weights = torch.zeros(0, dtype=torch.float, device=self.device)
                etype_str = repr(ppr_edge_type)
                metadata[f"{PPR_EDGE_INDEX_METADATA_KEY}{etype_str}"] = edge_index
                metadata[f"{PPR_WEIGHT_METADATA_KEY}{etype_str}"] = flat_weights

            sample_output = HeteroSamplerOutput(
                node=node_dict,
                # row/col/edge are left empty rather than populated with PPR edges because
                # the virtual (seed_type, "ppr", neighbor_type) edge types are unknown to
                # GLT: meaning the collate functions would fail trying to process them.
                # Instead, edge_index and edge_attr tensors are passed through metadata and
                # attached directly to the data object in the loader's _collate_fn.
                row={},
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
            #   new_nodes: global IDs of nodes not previously registered
            #             with the inducer.
            #   rows: flat local source indices (one per neighbor, expanded).
            #   cols: flat local destination indices for every neighbor, in the
            #         same order as homo_flat_ids.
            new_nodes, rows, cols = inducer.induce_next(
                srcs, homo_flat_ids, homo_valid_counts
            )
            all_nodes = torch.cat([srcs, new_nodes])

            ppr_edge_index = torch.stack([rows, cols])

            metadata["edge_index"] = ppr_edge_index
            metadata["edge_attr"] = homo_flat_weights

            sample_output = SamplerOutput(
                node=all_nodes,
                # row/col/edge are left empty for parity with the hetero case above.
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
