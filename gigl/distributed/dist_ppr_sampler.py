import asyncio
import math
from collections import defaultdict
from typing import Optional, Sequence, Union, cast

import torch

# TODO: Once gigl_core has a stable Python interface, re-export PPRForwardPush
# under a gigl.core namespace rather than importing directly from the C++ extension.
from gigl_core import PPRForwardPush
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import EdgeType, NodeType
from graphlearn_torch.utils import merge_dict

from gigl.distributed.base_sampler import BaseDistNeighborSampler
from gigl.types.graph import DEFAULT_HOMOGENEOUS_NODE_TYPE, is_label_edge_type

# Trailing "." is an intentional separator.  These constants are used both to
# write metadata keys (f"{KEY}{repr(edge_type)}" → e.g. "ppr_edge_index.('user', 'to', 'story')")
# and as the strip prefix in extract_edge_type_metadata (key[len(prefix):] must
# yield a bare EdgeType repr for ast.literal_eval).
PPR_EDGE_INDEX_METADATA_KEY = "ppr_edge_index."
PPR_WEIGHT_METADATA_KEY = "ppr_weight."

# Sentinel edge type for homogeneous graphs.  The PPR algorithm uses
# dict[NodeType, ...] internally for both homo and hetero graphs; the
# DEFAULT_HOMOGENEOUS_NODE_TYPE sentinel lets the homogeneous path reuse
# the same dict-based code.
_PPR_HOMOGENEOUS_EDGE_TYPE = (
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    "to",
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
)
_PPRResult = tuple[
    Union[torch.Tensor, dict[NodeType, torch.Tensor]],
    Union[torch.Tensor, dict[NodeType, torch.Tensor]],
    Union[torch.Tensor, dict[NodeType, torch.Tensor]],
]
_HeteroPPRResult = tuple[
    dict[NodeType, torch.Tensor],
    dict[NodeType, torch.Tensor],
    dict[NodeType, torch.Tensor],
]
_TypedPPRScoreMap = dict[NodeType, list[dict[int, list[float]]]]
_TypedPPRCandidates = dict[NodeType, list[list[list[tuple[int, float]]]]]
_TypedPPRMergeState = tuple[_TypedPPRScoreMap, _TypedPPRCandidates]
_TypedPPRChannelKey = Union[EdgeType, tuple[EdgeType, ...]]
_TypedPPRChannelGroups = list[tuple[tuple[EdgeType, ...], int]]


class DistPPRNeighborSampler(BaseDistNeighborSampler):
    """Personalized PageRank (PPR) based distributed neighbor sampler.

    Extends BaseGiGLSampler (which provides shared input preparation utilities)
    and overrides _sample_from_nodes with PPR-based neighbor selection.

    Instead of uniform random sampling, this sampler uses Personalized PageRank
    (PPR) scores to select the most relevant neighbors for each seed node. PPR
    scores are approximated here using the Forward Push algorithm (Andersen et
    al., 2006).

    Residual top-up provides a cheaper way to increase returned sequence volume
    without lowering ``eps``.  Lower ``eps`` thresholds re-enqueue more
    low-residual nodes, but also increase push iterations and neighbor-fetch
    work.  Top-up instead fills unused output slots with positive-residual nodes
    already discovered during Forward Push; these are the nodes that are closest
    to being re-enqueued if the threshold were lower.

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
        - ``data[(seed_type, "ppr", ntype)].edge_attr``: scalar PPR score for
          regular PPR. For typed PPR, edge attrs are multi-column:
          ``[best_calibrated_score, calibrated_channel_scores..., channel_presence_bits...]``.
          Typed-PPR scores are calibrated within each channel/seed pool and
          globally ranked by the best calibrated score. Channel columns follow
          the insertion order of ``typed_channel_quotas``.

    Args:
        alpha: Restart probability (teleport probability back to seed). Higher values
               keep samples closer to seeds. Typical values: 0.15-0.25.
        eps: Convergence threshold. Smaller values give more accurate PPR scores
             but require more computation. Typical values: 1e-4 to 1e-6.
        max_ppr_nodes: Maximum number of nodes to return per seed. If finalized
            PPR scores produce fewer than this cap and residual top-up is
            enabled, discovered residual candidates fill the remaining slots
            with score ``ppr_score + residual``.  Returned nodes are sorted by
            emitted score, but residual candidates do not displace finalized
            PPR nodes when finalized scores already fill the cap.
        enable_residual_topup: Whether to include residual candidates discovered
            during Forward Push when fewer than ``max_ppr_nodes`` finalized PPR
            scores are available.
        num_neighbors_per_hop: Maximum number of neighbors to fetch per hop.
        typed_channel_quotas: Optional top-k quotas for typed PPR traversal
            channels. Each channel may contribute up to its quota to the
            candidate pool; the final returned sequence is still capped by
            ``max_ppr_nodes``. Quotas may sum above ``max_ppr_nodes`` to give
            sparse or overlapping channels room to fill the sequence.
            Example::

                typed_channel_quotas = {
                    ("user", "views", "item"): 64,
                    (
                        ("user", "likes", "item"),
                        ("user", "shares", "item"),
                    ): 32,
                }

        degree_tensors: Pre-computed total-degree tensors (int32). Homogeneous
            graphs use a single tensor; heterogeneous graphs use tensors keyed
            by NodeType. The colocated and graph-store loader paths retrieve
            these through ``DistDataset.degree_tensor`` and move them to shared
            memory before worker handoff.
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.5,
        eps: float = 1e-4,
        max_ppr_nodes: int = 50,
        enable_residual_topup: bool = True,
        num_neighbors_per_hop: int = 100_000,
        degree_tensors: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        max_fetch_iterations: Optional[int] = None,
        typed_channel_quotas: Optional[dict[_TypedPPRChannelKey, int]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._max_ppr_nodes = max_ppr_nodes
        self._enable_residual_topup = enable_residual_topup
        self._requeue_threshold_factor = alpha * eps
        self._num_neighbors_per_hop = num_neighbors_per_hop
        self._max_fetch_iterations = max_fetch_iterations

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
            self._node_type_to_edge_types[DEFAULT_HOMOGENEOUS_NODE_TYPE] = [
                _PPR_HOMOGENEOUS_EDGE_TYPE
            ]
            self._is_homogeneous = True

        typed_channel_groups, self._typed_ppr_channel_quotas = (
            self._parse_typed_channel_quota_groups(typed_channel_quotas)
        )
        if self._typed_ppr_channel_quotas is not None:
            if self._is_homogeneous:
                raise ValueError(
                    "Typed PPR channel quotas are only supported for heterogeneous PPR sampling."
                )

        # Convert the public homogeneous/heterogeneous degree-tensor shape to
        # the node-type keyed form used internally by PPR.
        self._node_type_to_total_degree = self._convert_degree_tensors_to_dict(
            degree_tensors
        )

        # Build integer ID mappings for the C++ forward-push kernel.  String
        # NodeType / EdgeType keys are only used at the Python boundary
        # (translating to/from _sample_one_hop); all hot-loop state inside
        # PPRForwardPush is indexed by int32 IDs.
        #
        # We include both source types (have outgoing edges) and destination-only
        # types (no outgoing edges, but may accumulate PPR score during the walk)
        # so the kernel can index residual/ppr_score tables for any node it sees.
        source_node_types: set[NodeType] = set(self._node_type_to_edge_types.keys())
        destination_node_types: set[NodeType] = {
            self._get_destination_type(et)
            for etypes in self._node_type_to_edge_types.values()
            for et in etypes
        }
        all_node_types: list[NodeType] = sorted(
            source_node_types | destination_node_types
        )
        all_edge_types: list[EdgeType] = sorted(
            {et for etypes in self._node_type_to_edge_types.values() for et in etypes}
        )

        self._node_type_to_id: dict[NodeType, int] = {
            nt: i for i, nt in enumerate(all_node_types)
        }
        self._ntype_id_to_ntype: list[NodeType] = all_node_types
        self._etype_to_etype_id: dict[EdgeType, int] = {
            et: i for i, et in enumerate(all_edge_types)
        }
        self._etype_id_to_etype: list[EdgeType] = all_edge_types

        self._node_type_id_to_edge_type_ids: list[list[int]] = [
            [
                self._etype_to_etype_id[et]
                for et in self._node_type_to_edge_types.get(nt, [])
            ]
            for nt in all_node_types
        ]
        self._edge_type_id_to_dst_ntype_id: list[int] = [
            self._node_type_to_id[self._get_destination_type(et)]
            for et in all_edge_types
        ]
        # Degree tensors indexed by ntype_id.  Destination-only types get an empty
        # tensor; the C++ kernel returns 0 for those, matching _get_total_degree.
        self._degree_tensors_for_cpp: list[torch.Tensor] = [
            self._node_type_to_total_degree.get(nt, torch.zeros(0, dtype=torch.int32))
            for nt in all_node_types
        ]

        if typed_channel_groups is not None:
            self._typed_ppr_channel_to_node_type_id_to_edge_type_ids = (
                self._build_edge_type_channel_group_edge_type_ids(
                    typed_channel_groups,
                )
            )
        else:
            self._typed_ppr_channel_to_node_type_id_to_edge_type_ids = []

    def _build_edge_type_channel_group_edge_type_ids(
        self,
        edge_type_groups: _TypedPPRChannelGroups,
    ) -> list[list[list[int]]]:
        """Build per-node-type edge-type allowlists for canonical edge-type channels."""
        known_edge_types = set(self._etype_to_etype_id.keys())
        edge_type_group_to_node_type_id_to_edge_type_ids: list[list[list[int]]] = []
        for edge_types, _ in edge_type_groups:
            unknown_edge_types = set(edge_types) - known_edge_types
            if unknown_edge_types:
                raise ValueError(
                    "typed_channel_quotas includes non-traversable edge types "
                    f"{sorted(unknown_edge_types)!r}. Edge types must exist in the "
                    "graph and must not be label edge types."
                )

            edge_type_set = set(edge_types)
            node_type_id_to_edge_type_ids = [
                [
                    self._etype_to_etype_id[et]
                    for et in self._node_type_to_edge_types.get(nt, [])
                    if et in edge_type_set
                ]
                for nt in self._ntype_id_to_ntype
            ]
            if not any(node_type_id_to_edge_type_ids):
                raise ValueError(
                    f"typed_channel_quotas includes edge-type channel={edge_types!r}, "
                    "but no traversable edge types exist for that channel."
                )
            edge_type_group_to_node_type_id_to_edge_type_ids.append(
                node_type_id_to_edge_type_ids
            )
        return edge_type_group_to_node_type_id_to_edge_type_ids

    @staticmethod
    def _parse_typed_channel_quota_groups(
        typed_channel_quotas: Optional[dict[_TypedPPRChannelKey, int]],
    ) -> tuple[Optional[_TypedPPRChannelGroups], Optional[list[int]]]:
        """Validate quotas and return edge-type groups plus aligned quota values."""
        if not typed_channel_quotas:
            return None, None

        typed_channel_groups: _TypedPPRChannelGroups = []
        typed_channel_quota_list: list[int] = []
        invalid_quotas: dict[_TypedPPRChannelKey, object] = {}

        def is_canonical_edge_type(value: object) -> bool:
            return (
                isinstance(value, tuple)
                and len(value) == 3
                and all(isinstance(part, str) for part in value)
            )

        for edge_type_key, quota in typed_channel_quotas.items():
            if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
                invalid_quotas[edge_type_key] = quota
                continue
            if is_canonical_edge_type(edge_type_key):
                edge_types = (cast(EdgeType, edge_type_key),)
            elif (
                isinstance(edge_type_key, tuple)
                and edge_type_key
                and all(
                    is_canonical_edge_type(edge_type) for edge_type in edge_type_key
                )
            ):
                edge_types = cast(tuple[EdgeType, ...], edge_type_key)
            else:
                raise ValueError(
                    "typed_channel_quotas keys must be a canonical edge type "
                    "(src_type, relation, dst_type) or a non-empty tuple of "
                    f"canonical edge types, got {edge_type_key!r}."
                )
            typed_channel_groups.append((edge_types, quota))
            typed_channel_quota_list.append(quota)

        if invalid_quotas:
            raise ValueError(
                "typed_channel_quotas must contain only positive integer quotas, "
                f"got {invalid_quotas}."
            )
        return typed_channel_groups, typed_channel_quota_list

    def _convert_degree_tensors_to_dict(
        self,
        degree_tensors: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
    ) -> dict[NodeType, torch.Tensor]:
        """Convert degree tensors to the node-type keyed shape PPR uses."""
        if isinstance(degree_tensors, torch.Tensor):
            if not self._is_homogeneous:
                raise ValueError(
                    "Expected degree tensors keyed by node type for heterogeneous PPR sampling."
                )
            return {DEFAULT_HOMOGENEOUS_NODE_TYPE: degree_tensors}

        missing_anchor_types = set(self._node_type_to_edge_types.keys()) - set(
            degree_tensors.keys()
        )
        if missing_anchor_types:
            raise ValueError(
                f"Missing PPR degree tensors for node types: {missing_anchor_types}"
            )
        return degree_tensors

    def _get_destination_type(self, edge_type: EdgeType) -> NodeType:
        """Get the node type at the destination end of an edge type."""
        return edge_type[0] if self.edge_dir == "in" else edge_type[-1]

    async def _batch_fetch_neighbors(
        self,
        nodes_by_etype_id: dict[int, torch.Tensor],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batch fetch neighbors for nodes grouped by integer edge type ID.

        Issues one one-hop request per edge type in the frontier. Each node's
        neighbor list is capped at ``self._num_neighbors_per_hop``.

        Args:
            nodes_by_etype_id: Dict mapping integer edge type ID to a 1-D int64
                tensor of node IDs to fetch neighbors for.  Comes directly from
                ``drain_queue()``; node IDs are already deduplicated.

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
        eids: list[int] = []
        sample_tasks = []
        for eid in nodes_by_etype_id:
            etype = self._etype_id_to_etype[eid]
            # The lower sampler expects None only for true homogeneous graphs.
            # Labeled homogeneous ABLP graphs are hetero-backed because label
            # edges are represented as separate edge types, so they still need
            # the explicit default edge type here.
            rpc_etype = (
                None
                if self._is_homogeneous and etype == _PPR_HOMOGENEOUS_EDGE_TYPE
                else etype
            )
            eids.append(eid)
            sample_tasks.append(
                self._sample_one_hop(
                    nodes_by_etype_id[eid],
                    self._num_neighbors_per_hop,
                    rpc_etype,
                )
            )

        outputs = await asyncio.gather(*sample_tasks)
        return {
            eid: (nodes_by_etype_id[eid], output.nbr, output.nbr_num)
            for eid, output in zip(eids, outputs)
        }

    @staticmethod
    def _union_nodes_by_etype_id(
        nodes_by_etype_by_channel: Sequence[dict[int, torch.Tensor]],
    ) -> dict[int, torch.Tensor]:
        """Union channel frontier nodes by edge type for a shared one-hop fetch."""
        node_parts_by_etype_id: dict[int, list[torch.Tensor]] = defaultdict(list)
        for nodes_by_etype_id in nodes_by_etype_by_channel:
            for etype_id, nodes in nodes_by_etype_id.items():
                if nodes.numel() > 0:
                    node_parts_by_etype_id[etype_id].append(nodes)

        union_nodes_by_etype_id: dict[int, torch.Tensor] = {}
        for etype_id, node_parts in node_parts_by_etype_id.items():
            if len(node_parts) == 1:
                union_nodes_by_etype_id[etype_id] = node_parts[0]
            else:
                union_nodes_by_etype_id[etype_id] = torch.unique(
                    torch.cat(node_parts),
                    sorted=False,
                )
        return union_nodes_by_etype_id

    def _new_ppr_forward_push_state(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: NodeType,
        node_type_id_to_edge_type_ids: Optional[list[list[int]]] = None,
    ) -> PPRForwardPush:
        return PPRForwardPush(
            seed_nodes,
            self._node_type_to_id[seed_node_type],
            self._alpha,
            self._requeue_threshold_factor,
            (
                node_type_id_to_edge_type_ids
                if node_type_id_to_edge_type_ids is not None
                else self._node_type_id_to_edge_type_ids
            ),
            self._edge_type_id_to_dst_ntype_id,
            self._degree_tensors_for_cpp,
        )

    def _extract_ppr_state_top_k(
        self,
        ppr_state,
        device: torch.device,
        max_ppr_nodes: int,
    ) -> _PPRResult:
        """Extract PPR neighbors from a completed C++ Forward Push state.

        The C++ kernel indexes node types by compact integer IDs for speed.
        This helper translates those IDs back to GiGL node-type keys and
        preserves the homogeneous return shape expected by the rest of the
        sampler.

        ``max_ppr_nodes`` is the combined per-seed cap across finalized PPR and
        residual top-up candidates. If residual top-up is enabled, C++ derives
        the residual candidate budget from this cap after selecting finalized
        PPR nodes.

        Returns:
            ``(flat_ids, flat_weights, valid_counts)`` for homogeneous graphs,
            or three dictionaries keyed by node type for heterogeneous graphs.
            ``flat_ids`` and ``flat_weights`` are concatenated across seeds;
            ``valid_counts`` stores how many selected nodes belong to each seed.
        """
        # Translate ntype_id integer keys back to NodeType strings for the rest
        # of the pipeline, and move tensors to the correct device.
        ntype_to_flat_ids: dict[NodeType, torch.Tensor] = {}
        ntype_to_flat_weights: dict[NodeType, torch.Tensor] = {}
        ntype_to_valid_counts: dict[NodeType, torch.Tensor] = {}

        extracted_results = ppr_state.extract_top_k_with_residual_top_up(
            max_ppr_nodes,
            self._enable_residual_topup,
        )

        for ntype_id, (
            flat_ids,
            flat_weights,
            valid_counts,
        ) in extracted_results.items():
            ntype = self._ntype_id_to_ntype[ntype_id]
            ntype_to_flat_ids[ntype] = flat_ids.to(device)
            ntype_to_flat_weights[ntype] = flat_weights.to(device)
            ntype_to_valid_counts[ntype] = valid_counts.to(device)

        if self._is_homogeneous:
            assert (
                len(ntype_to_flat_ids) == 1
                and DEFAULT_HOMOGENEOUS_NODE_TYPE in ntype_to_flat_ids
            )
            return (
                ntype_to_flat_ids[DEFAULT_HOMOGENEOUS_NODE_TYPE],
                ntype_to_flat_weights[DEFAULT_HOMOGENEOUS_NODE_TYPE],
                ntype_to_valid_counts[DEFAULT_HOMOGENEOUS_NODE_TYPE],
            )
        else:
            return (
                ntype_to_flat_ids,
                ntype_to_flat_weights,
                ntype_to_valid_counts,
            )

    async def _compute_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: Optional[NodeType] = None,
    ) -> _PPRResult:
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
            seed_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        device = seed_nodes.device

        ppr_state = self._new_ppr_forward_push_state(
            seed_nodes=seed_nodes,
            seed_node_type=seed_node_type,
        )

        fetch_iteration_count = 0
        loop = asyncio.get_running_loop()
        nodes_by_etype_id = ppr_state.drain_queue()

        # drain_queue returns None when the queue is truly empty (convergence),
        # or a dict (possibly empty) when nodes were drained.  An empty dict
        # means all drained nodes either had cached neighbors or no outgoing
        # edges — we still push residuals to flush them into ppr_scores_.
        while nodes_by_etype_id is not None:
            fetch_budget_remaining = (
                self._max_fetch_iterations is None
                or fetch_iteration_count < self._max_fetch_iterations
            )
            if nodes_by_etype_id and fetch_budget_remaining:
                fetched_by_etype_id = await self._batch_fetch_neighbors(
                    nodes_by_etype_id
                )
                fetch_iteration_count += 1
            else:
                # Fetch budget exhausted; push_residuals will use the existing neighbor cache.
                fetched_by_etype_id = {}

            await loop.run_in_executor(
                None,
                ppr_state.push_residuals,
                fetched_by_etype_id,
            )
            nodes_by_etype_id = ppr_state.drain_queue()

        return self._extract_ppr_state_top_k(
            ppr_state,
            device,
            max_ppr_nodes=self._max_ppr_nodes,
        )

    async def _compute_ppr_scores_for_sampler_mode(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: Optional[NodeType] = None,
    ) -> _PPRResult:
        """Compute regular or typed PPR scores depending on sampler options."""
        if self._typed_ppr_channel_quotas is None:
            return await self._compute_ppr_scores(seed_nodes, seed_node_type)

        if seed_node_type is None:
            raise ValueError("Typed PPR requires an explicit heterogeneous seed type.")
        return await self._compute_typed_ppr_scores(seed_nodes, seed_node_type)

    async def _compute_typed_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: NodeType,
    ) -> _HeteroPPRResult:
        """Run typed-channel-restricted PPR states and merge their results."""
        assert self._typed_ppr_channel_quotas is not None
        channel_quotas = self._typed_ppr_channel_quotas
        device = seed_nodes.device
        ppr_states = [
            self._new_ppr_forward_push_state(
                seed_nodes=seed_nodes,
                seed_node_type=seed_node_type,
                node_type_id_to_edge_type_ids=node_type_id_to_edge_type_ids,
            )
            for node_type_id_to_edge_type_ids in (
                self._typed_ppr_channel_to_node_type_id_to_edge_type_ids
            )
        ]
        fetch_iteration_counts = [0 for _ in ppr_states]
        loop = asyncio.get_running_loop()
        nodes_by_channel: list[Optional[dict[int, torch.Tensor]]] = [
            ppr_state.drain_queue() for ppr_state in ppr_states
        ]

        while any(
            nodes_by_etype_id is not None for nodes_by_etype_id in nodes_by_channel
        ):
            fetched_by_channel: list[
                dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            ] = [dict() for _ in ppr_states]
            fetch_channel_indices: list[int] = []
            nodes_by_etype_by_fetch_channel: list[dict[int, torch.Tensor]] = []

            for channel_idx, nodes_by_etype_id in enumerate(nodes_by_channel):
                if nodes_by_etype_id is None:
                    continue

                fetch_budget_remaining = (
                    self._max_fetch_iterations is None
                    or fetch_iteration_counts[channel_idx] < self._max_fetch_iterations
                )
                has_nodes_to_fetch = any(
                    nodes.numel() > 0 for nodes in nodes_by_etype_id.values()
                )
                if has_nodes_to_fetch and fetch_budget_remaining:
                    fetch_channel_indices.append(channel_idx)
                    nodes_by_etype_by_fetch_channel.append(nodes_by_etype_id)
                    fetch_iteration_counts[channel_idx] += 1

            if nodes_by_etype_by_fetch_channel:
                union_nodes_by_etype_id = self._union_nodes_by_etype_id(
                    nodes_by_etype_by_fetch_channel
                )
                if union_nodes_by_etype_id:
                    union_fetched_by_etype_id = await self._batch_fetch_neighbors(
                        union_nodes_by_etype_id
                    )
                    for channel_idx, nodes_by_etype_id in zip(
                        fetch_channel_indices,
                        nodes_by_etype_by_fetch_channel,
                    ):
                        fetched_by_channel[channel_idx] = {
                            etype_id: union_fetched_by_etype_id[etype_id]
                            for etype_id in nodes_by_etype_id
                            if etype_id in union_fetched_by_etype_id
                        }

            active_channel_indices = [
                channel_idx
                for channel_idx, nodes_by_etype_id in enumerate(nodes_by_channel)
                if nodes_by_etype_id is not None
            ]
            push_tasks = [
                loop.run_in_executor(
                    None,
                    ppr_states[channel_idx].push_residuals,
                    fetched_by_channel[channel_idx],
                )
                for channel_idx in active_channel_indices
            ]
            if push_tasks:
                await asyncio.gather(*push_tasks)
                for channel_idx in active_channel_indices:
                    nodes_by_channel[channel_idx] = ppr_states[
                        channel_idx
                    ].drain_queue()

        residual_topup_nodes = self._max_ppr_nodes if self._enable_residual_topup else 0
        topup_candidate_limits = [
            channel_quota + residual_topup_nodes for channel_quota in channel_quotas
        ]
        channel_results = [
            self._extract_ppr_state_top_k(
                ppr_state,
                device=device,
                max_ppr_nodes=channel_quota,
                residual_topup_nodes=residual_topup_nodes,
                max_total_nodes=topup_candidate_limit,
            )
            for ppr_state, channel_quota, topup_candidate_limit in zip(
                ppr_states,
                channel_quotas,
                topup_candidate_limits,
            )
        ]
        return self._merge_typed_ppr_results_with_topup(
            channel_results=channel_results,
            base_channel_quotas=channel_quotas,
            topup_candidate_limits=topup_candidate_limits,
            num_seeds=seed_nodes.numel(),
            device=seed_nodes.device,
        )

    def _merge_typed_ppr_results_with_topup(
        self,
        channel_results: Sequence[_PPRResult],
        base_channel_quotas: Sequence[int],
        topup_candidate_limits: Sequence[int],
        num_seeds: int,
        device: torch.device,
    ) -> _HeteroPPRResult:
        """Merge typed PPR once, preserving base ordering and appending residual top-up.

        ``channel_results`` are extracted as finalized-PPR candidates followed
        by residual-backed top-up candidates for each channel source. The base
        merge preserves the configured channel quotas. If that result is short,
        residual-backed candidates from all channel sources compete globally for
        the remaining ``max_ppr_nodes`` slots.
        """
        num_channels = len(topup_candidate_limits)
        num_edge_attr_features = 1 + (2 * num_channels)
        base_scores, base_candidates = self._build_typed_ppr_merge_state(
            num_seeds=num_seeds,
            num_channels=num_channels,
        )
        extended_scores, extended_candidates = self._build_typed_ppr_merge_state(
            num_seeds=num_seeds,
            num_channels=num_channels,
        )
        self._populate_typed_ppr_topup_merge_states(
            channel_results=channel_results,
            base_channel_quotas=base_channel_quotas,
            topup_candidate_limits=topup_candidate_limits,
            base_scores=base_scores,
            base_candidates=base_candidates,
            extended_scores=extended_scores,
            extended_candidates=extended_candidates,
            num_edge_attr_features=num_edge_attr_features,
            num_channels=num_channels,
        )

        ntype_to_flat_ids_out: dict[NodeType, torch.Tensor] = {}
        ntype_to_flat_weights_out: dict[NodeType, torch.Tensor] = {}
        ntype_to_valid_counts_out: dict[NodeType, torch.Tensor] = {}
        ntypes = set(base_scores.keys()) | set(extended_scores.keys())

        for ntype in ntypes:
            flat_ids: list[int] = []
            flat_weights: list[list[float]] = []
            valid_counts: list[int] = []
            base_seed_scores_by_ntype = base_scores.get(ntype)
            base_candidates_by_ntype = base_candidates.get(ntype)
            extended_seed_scores_by_ntype = extended_scores.get(ntype)
            extended_candidates_by_ntype = extended_candidates.get(ntype)

            for seed_index in range(num_seeds):
                selected_nodes: list[int] = []
                selected_node_ids: set[int] = set()

                if (
                    base_seed_scores_by_ntype is not None
                    and base_candidates_by_ntype is not None
                ):
                    base_selected_nodes = self._select_typed_ppr_node_ids(
                        seed_scores=base_seed_scores_by_ntype[seed_index],
                        candidates_by_channel=base_candidates_by_ntype[seed_index],
                        channel_quotas=base_channel_quotas,
                    )
                    selected_nodes.extend(base_selected_nodes)
                    selected_node_ids.update(base_selected_nodes)
                    flat_weights.extend(
                        base_seed_scores_by_ntype[seed_index][node_id]
                        for node_id in base_selected_nodes
                    )

                if (
                    len(selected_nodes) < self._max_ppr_nodes
                    and extended_seed_scores_by_ntype is not None
                    and extended_candidates_by_ntype is not None
                ):
                    extended_selected_nodes = self._select_typed_ppr_node_ids(
                        seed_scores=extended_seed_scores_by_ntype[seed_index],
                        candidates_by_channel=extended_candidates_by_ntype[seed_index],
                        channel_quotas=topup_candidate_limits,
                    )
                    for node_id in extended_selected_nodes:
                        if len(selected_nodes) >= self._max_ppr_nodes:
                            break
                        if node_id in selected_node_ids:
                            continue
                        selected_nodes.append(node_id)
                        selected_node_ids.add(node_id)
                        flat_weights.append(
                            extended_seed_scores_by_ntype[seed_index][node_id]
                        )

                valid_counts.append(len(selected_nodes))
                flat_ids.extend(selected_nodes)

            ntype_to_flat_ids_out[ntype] = torch.tensor(
                flat_ids,
                dtype=torch.long,
                device=device,
            )
            ntype_to_flat_weights_out[ntype] = torch.tensor(
                flat_weights,
                dtype=torch.double,
                device=device,
            ).reshape(
                len(flat_weights),
                num_edge_attr_features,
            )
            ntype_to_valid_counts_out[ntype] = torch.tensor(
                valid_counts,
                dtype=torch.long,
                device=device,
            )

        return (
            ntype_to_flat_ids_out,
            ntype_to_flat_weights_out,
            ntype_to_valid_counts_out,
        )

    def _build_typed_ppr_merge_state(
        self,
        num_seeds: int,
        num_channels: int,
    ) -> _TypedPPRMergeState:
        merged_scores: _TypedPPRScoreMap = defaultdict(
            lambda: [dict() for _ in range(num_seeds)]
        )
        channel_candidates: _TypedPPRCandidates = defaultdict(
            lambda: [[[] for _ in range(num_channels)] for _ in range(num_seeds)]
        )
        return merged_scores, channel_candidates

    def _populate_typed_ppr_topup_merge_states(
        self,
        channel_results: Sequence[_PPRResult],
        base_channel_quotas: Sequence[int],
        topup_candidate_limits: Sequence[int],
        base_scores: _TypedPPRScoreMap,
        base_candidates: _TypedPPRCandidates,
        extended_scores: _TypedPPRScoreMap,
        extended_candidates: _TypedPPRCandidates,
        num_edge_attr_features: int,
        num_channels: int,
    ) -> None:
        for channel_idx, (
            ntype_to_flat_ids,
            ntype_to_flat_weights,
            ntype_to_valid_counts,
        ) in enumerate(channel_results):
            assert isinstance(ntype_to_flat_ids, dict)
            assert isinstance(ntype_to_flat_weights, dict)
            assert isinstance(ntype_to_valid_counts, dict)

            base_channel_quota = base_channel_quotas[channel_idx]
            topup_candidate_limit = topup_candidate_limits[channel_idx]

            for ntype, flat_ids in ntype_to_flat_ids.items():
                ids_cpu = flat_ids.detach().cpu().tolist()
                weights_cpu = ntype_to_flat_weights[ntype].detach().cpu().tolist()
                counts_cpu = ntype_to_valid_counts[ntype].detach().cpu().tolist()
                offset = 0

                for seed_index, count in enumerate(counts_cpu):
                    base_nodes_and_scores: list[tuple[int, float]] = []
                    extended_nodes_and_scores: list[tuple[int, float]] = []
                    extended_max_score = 0.0
                    candidate_count = min(count, topup_candidate_limit)

                    for candidate_idx, (node_id, raw_score) in enumerate(
                        zip(
                            ids_cpu[offset : offset + candidate_count],
                            weights_cpu[offset : offset + candidate_count],
                        )
                    ):
                        if not math.isfinite(raw_score):
                            continue
                        score = min(max(raw_score, 0.0), 1.0)
                        extended_nodes_and_scores.append((node_id, score))
                        extended_max_score = max(extended_max_score, score)
                        if candidate_idx < base_channel_quota:
                            base_nodes_and_scores.append((node_id, score))

                    offset += count

                    # Both base and residual top-up candidates are calibrated using
                    # the extended candidate pool's max score. This keeps finalized
                    # PPR nodes and residual top-up nodes on the same per-channel
                    # scale before the global typed-PPR merge.
                    self._add_typed_ppr_seed_candidates(
                        seed_scores=base_scores[ntype][seed_index],
                        seed_channel_candidates=base_candidates[ntype][seed_index][
                            channel_idx
                        ],
                        seed_nodes_and_scores=base_nodes_and_scores,
                        max_score=extended_max_score,
                        channel_idx=channel_idx,
                        num_edge_attr_features=num_edge_attr_features,
                        num_channels=num_channels,
                    )
                    self._add_typed_ppr_seed_candidates(
                        seed_scores=extended_scores[ntype][seed_index],
                        seed_channel_candidates=extended_candidates[ntype][seed_index][
                            channel_idx
                        ],
                        seed_nodes_and_scores=extended_nodes_and_scores,
                        max_score=extended_max_score,
                        channel_idx=channel_idx,
                        num_edge_attr_features=num_edge_attr_features,
                        num_channels=num_channels,
                    )

    def _add_typed_ppr_seed_candidates(
        self,
        seed_scores: dict[int, list[float]],
        seed_channel_candidates: list[tuple[int, float]],
        seed_nodes_and_scores: Sequence[tuple[int, float]],
        max_score: float,
        channel_idx: int,
        num_edge_attr_features: int,
        num_channels: int,
    ) -> None:
        for node_id, score in seed_nodes_and_scores:
            calibrated_score = score / max_score if max_score > 0 else 0.0
            score_features = seed_scores.get(node_id)
            if score_features is None:
                score_features = [0.0] * num_edge_attr_features
                seed_scores[node_id] = score_features
            score_features[0] = max(score_features[0], calibrated_score)
            channel_score_idx = 1 + channel_idx
            channel_presence_idx = 1 + num_channels + channel_idx
            score_features[channel_score_idx] = max(
                score_features[channel_score_idx], calibrated_score
            )
            score_features[channel_presence_idx] = 1.0
            seed_channel_candidates.append((node_id, calibrated_score))

    def _select_typed_ppr_node_ids(
        self,
        seed_scores: dict[int, list[float]],
        candidates_by_channel: Sequence[Sequence[tuple[int, float]]],
        channel_quotas: Sequence[int],
    ) -> list[int]:
        """Select typed-PPR nodes by channel quota, then global calibrated rank.

        Each channel first contributes its own top candidates by that channel's
        calibrated score. The pooled candidates are then globally ranked by the
        best calibrated score across all channels for the seed.
        """
        selected_node_ids: set[int] = set()
        selected_nodes: list[int] = []

        sorted_candidates_by_channel = [
            sorted(
                candidates,
                key=lambda item: (-item[1], item[0]),
            )
            for candidates in candidates_by_channel
        ]
        global_candidates: list[tuple[float, float, int, int]] = []
        for channel_idx, candidates in enumerate(sorted_candidates_by_channel):
            channel_quota = channel_quotas[channel_idx]
            for node_id, calibrated_score in candidates[:channel_quota]:
                global_candidates.append(
                    (
                        seed_scores[node_id][0],
                        calibrated_score,
                        channel_idx,
                        node_id,
                    )
                )

        for _, _, _, node_id in sorted(
            global_candidates,
            key=lambda item: (-item[0], -item[1], item[2], item[3]),
        ):
            if len(selected_nodes) >= self._max_ppr_nodes:
                break
            if node_id in selected_node_ids:
                continue
            selected_node_ids.add(node_id)
            selected_nodes.append(node_id)

        return selected_nodes

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

            # Compute PPR for all seed types concurrently, collecting flat global
            # neighbor IDs, weights, and per-seed counts.  Build nbr_dict for a
            # single inducer.induce_next call using PPR edge types
            # (seed_type, 'ppr', ntype) — the inducer only cares about etype[0]
            # and etype[-1] as source/dest node types, so the relation name is
            # arbitrary.
            #
            # Each seed type's PPR computation is entirely independent: it creates
            # its own PPRForwardPush and only reads shared sampler attributes
            # (degree tensors, edge-type maps) which are immutable after __init__.
            # Running them with asyncio.gather allows their fetch phases to overlap,
            # which is most beneficial when there are 2+ distinct seed node types
            # (e.g. cross-type supervision edges like user→story).
            seed_types = list(nodes_to_sample.keys())
            ppr_results = await asyncio.gather(
                *[
                    self._compute_ppr_scores_for_sampler_mode(
                        nodes_to_sample[seed_type], seed_type
                    )
                    for seed_type in seed_types
                ]
            )

            nbr_dict: dict[EdgeType, list[torch.Tensor]] = {}
            ppr_edge_type_to_flat_weights: dict[EdgeType, torch.Tensor] = {}

            for seed_type, (
                ntype_to_flat_ids,
                ntype_to_flat_weights,
                ntype_to_valid_counts,
            ) in zip(seed_types, ppr_results):
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
            for (
                ppr_edge_type,
                flat_weights,
            ) in ppr_edge_type_to_flat_weights.items():
                rows = rows_dict.get(ppr_edge_type)
                cols = cols_dict.get(ppr_edge_type)
                if rows is not None and cols is not None:
                    edge_index = torch.stack([rows, cols])
                else:
                    edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)
                    flat_weights = flat_weights.new_zeros((0, *flat_weights.shape[1:]))
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
            if isinstance(nodes_to_sample, torch.Tensor):
                homogeneous_nodes_to_sample = nodes_to_sample
            elif isinstance(nodes_to_sample, dict):
                node_types = set(nodes_to_sample.keys())
                if node_types != {DEFAULT_HOMOGENEOUS_NODE_TYPE}:
                    raise ValueError(
                        f"Expected only {DEFAULT_HOMOGENEOUS_NODE_TYPE} for homogeneous PPR sampling, "
                        f"received node types: {node_types}"
                    )
                homogeneous_nodes_to_sample = nodes_to_sample[
                    DEFAULT_HOMOGENEOUS_NODE_TYPE
                ]
            else:
                raise TypeError(
                    f"Expected Tensor or node-type mapping for homogeneous PPR sampling, got {type(nodes_to_sample)}"
                )

            # Register seeds; local indices 0..N-1 are assigned internally.
            # srcs holds their global IDs (same values as nodes_to_sample).
            srcs = inducer.init_node(homogeneous_nodes_to_sample)

            (
                homo_flat_ids,
                homo_flat_weights,
                homo_valid_counts,
            ) = await self._compute_ppr_scores_for_sampler_mode(
                homogeneous_nodes_to_sample, None
            )
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
