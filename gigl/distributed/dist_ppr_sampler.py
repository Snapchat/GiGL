# TODO (mkolodner-sc): Investigate whether concurrency for _sample_one_hop and _compute_ppr_scores will
# yield performance benefits.

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

from gigl.csrc.distributed import PPRForwardPushState
from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler
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


# TODO (mkolodner-sc): Consider introducing a BaseGiGLSampler that owns
# shared utilities like _prepare_sample_loop_inputs, with KHopSampler and
# PPRSampler as siblings.  Currently DistPPRNeighborSampler inherits from
# DistNeighborSampler (the k-hop sampler), which bundles generic utilities
# with k-hop-specific sampling logic.


class DistPPRNeighborSampler(DistNeighborSampler):
    """
    Personalized PageRank (PPR) based neighbor sampler that inherits from GLT DistNeighborSampler.

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
        self._node_type_to_total_degree: dict[
            NodeType, torch.Tensor
        ] = self._build_total_degree_tensors(degree_tensors, total_degree_dtype)

        # Build integer ID mappings for the C++ forward-push kernel.  String
        # NodeType / EdgeType keys are only used at the Python boundary
        # (translating to/from _sample_one_hop); all hot-loop state inside
        # PPRForwardPushState is indexed by int32 IDs.
        #
        # We include both source types (have outgoing edges) and destination-only
        # types (no outgoing edges, but may accumulate PPR score during the walk)
        # so the kernel can index residual/ppr_score tables for any node it sees.
        _all_node_types: list[NodeType] = sorted(
            {nt for nt in self._node_type_to_edge_types}
            | {
                self._get_destination_type(et)
                for etypes in self._node_type_to_edge_types.values()
                for et in etypes
            }
        )
        # dict.fromkeys preserves insertion order while deduplicating.
        _all_edge_types: list[EdgeType] = list(
            dict.fromkeys(
                et for etypes in self._node_type_to_edge_types.values() for et in etypes
            )
        )

        self._node_type_to_id: dict[NodeType, int] = {
            nt: i for i, nt in enumerate(_all_node_types)
        }
        self._ntype_id_to_ntype: list[NodeType] = _all_node_types
        self._etype_to_etype_id: dict[EdgeType, int] = {
            et: i for i, et in enumerate(_all_edge_types)
        }
        self._etype_id_to_etype: list[EdgeType] = _all_edge_types

        self._node_type_id_to_edge_type_ids: list[list[int]] = [
            [
                self._etype_to_etype_id[et]
                for et in self._node_type_to_edge_types.get(nt, [])
            ]
            for nt in _all_node_types
        ]
        self._edge_type_id_to_dst_ntype_id: list[int] = [
            self._node_type_to_id[self._get_destination_type(et)]
            for et in _all_edge_types
        ]
        # Degree tensors indexed by ntype_id.  Destination-only types get an empty
        # tensor; the C++ kernel returns 0 for those, matching _get_total_degree.
        self._degree_tensors_for_cpp: list[torch.Tensor] = [
            self._node_type_to_total_degree.get(nt, torch.zeros(0, dtype=torch.int32))
            for nt in _all_node_types
        ]

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

    def _get_destination_type(self, edge_type: EdgeType) -> NodeType:
        """Get the node type at the destination end of an edge type."""
        return edge_type[0] if self.edge_dir == "in" else edge_type[-1]

    async def _batch_fetch_neighbors(
        self,
        nodes_by_etype_id: dict[int, torch.Tensor],
        device: torch.device,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batch fetch neighbors for nodes grouped by integer edge type ID.

        Issues one ``_sample_one_hop`` call per edge type (not per node), so all
        nodes of the same edge type are fetched in a single RPC round-trip. Each
        node's neighbor list is capped at ``self._num_neighbors_per_hop``.

        Args:
            nodes_by_etype_id: Dict mapping integer edge type ID to a 1-D int64
                tensor of node IDs to fetch neighbors for.  Comes directly from
                ``drain_queue()``; node IDs are already deduplicated.
            device: Torch device for intermediate tensor creation.

        Returns:
            Dict mapping etype_id to ``(node_ids, flat_neighbors, counts)`` as
            int64 tensors, ready to pass directly to ``push_residuals``.
            ``flat_neighbors`` is the flat concatenation of all neighbor lists
            for that edge type; ``counts[i]`` is the neighbor count for
            ``node_ids[i]``.

        Example::

            nodes_by_etype_id = {
                2: tensor([0, 3]),   # etype_id 2 → nodes 0 and 3
                5: tensor([7]),      # etype_id 5 → node 7
            }
            # Might return (neighbor lists depend on graph structure):
            {
                2: (tensor([0, 3]), tensor([5, 9, 2, 1]), tensor([3, 1])),
                5: (tensor([7]),    tensor([0, 3]),        tensor([2])),
            }
        """
        result: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for eid, node_ids_tensor in nodes_by_etype_id.items():
            etype = self._etype_id_to_etype[eid]
            # _sample_one_hop expects None for homogeneous graphs, not the PPR sentinel.
            output: NeighborOutput = await self._sample_one_hop(
                srcs=node_ids_tensor.to(device),
                num_nbr=self._num_neighbors_per_hop,
                etype=etype if etype != _PPR_HOMOGENEOUS_EDGE_TYPE else None,
            )
            result[eid] = (node_ids_tensor, output.nbr, output.nbr_num)

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

        ppr_state = PPRForwardPushState(
            seed_nodes,
            self._node_type_to_id[seed_node_type],
            self._alpha,
            self._requeue_threshold_factor,
            self._node_type_id_to_edge_type_ids,
            self._edge_type_id_to_dst_ntype_id,
            self._degree_tensors_for_cpp,
        )

        while True:
            # drain_queue returns None when the queue is truly empty (convergence),
            # or a dict (possibly empty) when nodes were drained.  An empty dict
            # means all drained nodes either had cached neighbors or no outgoing
            # edges — we still call push_residuals to flush their residuals into
            # ppr_scores_.
            drain_result: dict[int, torch.Tensor] | None = ppr_state.drain_queue()
            if drain_result is None:
                break

            nodes_by_etype_id: dict[int, torch.Tensor] = drain_result
            if nodes_by_etype_id:
                fetched_by_etype_id = await self._batch_fetch_neighbors(
                    nodes_by_etype_id, device
                )
            else:
                fetched_by_etype_id = {}

            ppr_state.push_residuals(fetched_by_etype_id)

        # Translate ntype_id integer keys back to NodeType strings for the rest
        # of the pipeline, and move tensors to the correct device.
        ntype_to_flat_ids: dict[NodeType, torch.Tensor] = {}
        ntype_to_flat_weights: dict[NodeType, torch.Tensor] = {}
        ntype_to_valid_counts: dict[NodeType, torch.Tensor] = {}

        for ntype_id, (flat_ids, flat_weights, valid_counts) in ppr_state.extract_top_k(
            self._max_ppr_nodes
        ).items():
            ntype = self._ntype_id_to_ntype[ntype_id]
            ntype_to_flat_ids[ntype] = flat_ids.to(device)
            ntype_to_flat_weights[ntype] = flat_weights.to(device)
            ntype_to_valid_counts[ntype] = valid_counts.to(device)

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
                    ppr_edge_type_to_flat_weights[
                        ppr_edge_type
                    ] = ntype_to_flat_weights[ntype]

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
