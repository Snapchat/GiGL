import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union, cast

import torch

# TODO: Once gigl_core has a stable Python interface, re-export PPRForwardPush
# under a gigl.core namespace rather than importing directly from the C++ extension.
from gigl_core import (
    PPRForwardPush,
    drain_typed_ppr_channel_queues,
    extract_original_edges_from_ppr_caches,
    extract_typed_top_k_with_residual_top_up,
)
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NeighborOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import EdgeType, NodeType
from graphlearn_torch.utils import merge_dict, reverse_edge_type

from gigl.distributed.base_sampler import BaseDistNeighborSampler
from gigl.distributed.utils.dist_typed_sampler import (
    TypedPPRChannelKey,
    TypedPPRChannelTraversalMaps,
    build_edge_type_channel_group_edge_type_ids,
    compute_typed_channel_target_counts,
    parse_typed_channel_ratio_groups,
)
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

PPRForwardPushFetchMap = dict[
    int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
]
PPRTensorOutput = Union[torch.Tensor, dict[NodeType, torch.Tensor]]
PPRCacheState = Optional[Union[PPRForwardPush, list[PPRForwardPush]]]


@dataclass(frozen=True)
class PPRResult:
    """C++ PPR extraction output plus optional completed cache state.

    Homogeneous extraction uses tensors directly; heterogeneous extraction uses
    dictionaries keyed by node type.
    """

    node_ids: PPRTensorOutput
    weights: PPRTensorOutput
    valid_counts: PPRTensorOutput
    cache_state: PPRCacheState


@dataclass(frozen=True)
class SampledEdgeResult:
    """GLT sampled-edge fields for original edges attached to PPR output."""

    rows: dict[EdgeType, torch.Tensor]
    cols: dict[EdgeType, torch.Tensor]
    edge_ids: Optional[dict[EdgeType, torch.Tensor]]
    num_sampled_edges: dict[EdgeType, list[int]]


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

    Internal execution follows the same shape for regular and typed PPR. Regular
    PPR owns one C++ ``PPRForwardPush`` state per seed type: ``drain_queue``
    exposes the next frontier, Python performs the distributed neighbor fetch,
    ``push_residuals`` updates the state, and C++ extraction emits the final
    top-k plus residual top-up output. Typed PPR runs one ``PPRForwardPush``
    traversal per configured channel. Its typed drain step unions the channel
    frontiers before the same distributed fetch, pushes results back into the
    active channel states, and then uses one typed C++ extraction step to apply
    channel target counts, deduplicate shared candidates, and emit the final
    typed edge-attribute features.

    The ``edge_index`` and ``edge_attr`` fields on the output Data/HeteroData
    objects are populated with PPR seed-to-neighbor relationships (not edges
    in the original graph). ``N`` is the total number of (seed, neighbor)
    pairs across all seeds in the batch.

    **Homogeneous (Data):**
        - ``data.edge_index``: ``[2, N]`` int64 — row 0 is local seed indices,
          row 1 is local neighbor indices.
        - ``data.edge_attr``: ``[N, 2]`` float — ``[ppr_score, hop_proximity]``
          for each pair. ``hop_proximity`` is ``1 / (1 + hop)``: ``1.0`` for
          the anchor, ``0.5`` for 1-hop, and so on.

    **Heterogeneous (HeteroData)** — one PPR edge type per
    ``(seed_type, neighbor_type)`` pair, with ``"ppr"`` as the relation:
        - ``data[(seed_type, "ppr", neighbor_type)].edge_index``: same format as above.
        - ``data[(seed_type, "ppr", neighbor_type)].edge_attr``:
          ``[ppr_score, hop_proximity]`` for regular PPR. For typed PPR, edge
          attrs are multi-column:
          ``[best_score, hop_proximity, (channel_score, channel_hop_proximity,
          channel_presence), ...]``.
          Scores use the same PPR mass scale as regular PPR output.
          Channel columns follow the insertion order of
          ``typed_channel_ratios``. Column 0 is the scalar best score for
          consumers that need a single PPR weight, and column 1 is always the
          global hop proximity. Per-channel hop proximity is ``1 / (1 + hop)``
          when that channel reached the node, and ``0`` when it did not.
          For present channels, the original hop count can be recovered as
          ``(1 - proximity) / proximity``; use the presence bit before applying
          this inverse because missing channels have proximity ``0``.
        - When ``include_sampled_edges`` is enabled, original graph edge types are
          included alongside virtual PPR edges. The sampler emits only original
          edges that were fetched during PPR traversal and whose source and
          destination are both in the final PPR-selected node set. These original
          edges are emitted through GLT's regular sampled-edge channel, so their
          final HeteroData edge orientation follows the same ``edge_dir``
          convention as k-hop sampling.

    Args:
        alpha: Restart probability (teleport probability back to seed). Higher values
               keep samples closer to seeds. Typical values: 0.15-0.25.
        eps: Convergence threshold. Smaller values give more accurate PPR scores
             but require more computation. Typical values: 1e-4 to 1e-6.
        max_ppr_nodes: Maximum number of nodes to return per seed. If finalized
            PPR scores produce fewer than this cap and residual top-up is
            enabled, discovered residual candidates fill the remaining slots
            with score ``ppr_score + residual``. Returned nodes are sorted by
            emitted score, but residual candidates do not displace finalized
            PPR nodes when finalized scores already fill the cap.
        enable_residual_topup: Whether to include residual candidates discovered
            during Forward Push when fewer than ``max_ppr_nodes`` finalized PPR
            scores are available.
        num_neighbors_per_hop: Maximum number of neighbors to fetch per hop.
        typed_channel_ratios: Optional target proportions for typed PPR
            traversal channels. If not provided, PPR treats all eligible edge
            types as one shared traversal space and emits a single scalar PPR
            score per output row.
            Keys may be either a single canonical edge type
            ``(src_type, relation, dst_type)`` or a tuple of canonical edge
            types. Each key defines one traversal channel that may use only
            those exact edge types. Edge types may appear in multiple channels
            when those channels intentionally overlap.
            Channel order follows the insertion order of this mapping, and
            typed ``edge_attr`` channel columns use that same order. If the
            mapping is produced from an unordered config source, construct it
            deterministically before passing it to the sampler. Values are
            positive ratios that must sum to ``1.0``. The sampler converts
            ratios to per-channel target counts from ``max_ppr_nodes``.
            Finalized PPR candidates and residual top-up candidates both obey
            these target counts. If the same node appears in multiple channels,
            it is attributed to the channel where it has the highest emitted
            PPR score for that seed. If sparse channels or duplicate nodes
            leave unused target slots, the remaining slots are redistributed
            globally by score so the returned sequence can still fill up to,
            but never exceed, ``max_ppr_nodes``.
            Example::

                typed_channel_ratios = {
                    ("user", "views", "item"): 0.6,
                    (
                        ("user", "likes", "item"),
                        ("user", "shares", "item"),
                    ): 0.4,
                }

            With ``max_ppr_nodes=200``, this example targets 120 nodes
            attributed to the views channel and 80 nodes attributed to the
            grouped likes/shares channel. The views channel traverses only
            ``("user", "views", "item")`` edges. The grouped likes/shares
            channel traverses either likes or shares edges as one channel.
            Both finalized PPR rows and residual top-up rows fill those same
            targets. The targets are best-effort rather than strict per-seed
            guarantees: if a channel cannot provide enough unique candidates,
            unused slots are filled by the remaining highest-scoring candidates
            from any channel, up to ``max_ppr_nodes`` total rows.

        degree_tensors: Pre-computed total-degree tensors (int32). Homogeneous
            graphs use a single tensor; heterogeneous graphs use tensors keyed
            by NodeType. The colocated and graph-store loader paths retrieve
            these through ``DistDataset.degree_tensor`` and move them to shared
            memory before worker handoff.
        include_sampled_edges: Whether heterogeneous PPR output should include
            original graph edges alongside virtual PPR edges. The sampler emits
            only original edges that were fetched during PPR traversal and whose
            source and destination are both in the final PPR-selected node set.
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
        typed_channel_ratios: Optional[dict[TypedPPRChannelKey, float]] = None,
        include_sampled_edges: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        if isinstance(max_ppr_nodes, bool) or max_ppr_nodes < 0:
            raise ValueError(
                f"max_ppr_nodes must be non-negative, got {max_ppr_nodes}."
            )
        self._max_ppr_nodes = max_ppr_nodes
        self._enable_residual_topup = enable_residual_topup
        self._requeue_threshold_factor = alpha * eps
        self._num_neighbors_per_hop = num_neighbors_per_hop
        self._max_fetch_iterations = max_fetch_iterations
        self._include_sampled_edges = include_sampled_edges

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
            for edge_type in self.edge_types:
                if is_label_edge_type(edge_type):
                    continue
                if self.edge_dir == "in":
                    # For incoming edges, we traverse FROM the destination node type
                    anchor_type = edge_type[-1]
                else:  # "out"
                    # For outgoing edges, we traverse FROM the source node type
                    anchor_type = edge_type[0]

                self._node_type_to_edge_types[anchor_type].append(edge_type)
        else:
            self._node_type_to_edge_types[DEFAULT_HOMOGENEOUS_NODE_TYPE] = [
                _PPR_HOMOGENEOUS_EDGE_TYPE
            ]
            self._is_homogeneous = True

        typed_channel_groups, typed_channel_ratio_list = (
            parse_typed_channel_ratio_groups(typed_channel_ratios)
        )
        self._typed_ppr_channel_target_counts = (
            compute_typed_channel_target_counts(
                typed_channel_ratio_list,
                max_ppr_nodes,
            )
            if typed_channel_ratio_list is not None
            else None
        )
        if self._typed_ppr_channel_target_counts is not None:
            if self._is_homogeneous:
                raise ValueError(
                    "Typed PPR channel ratios are only supported for heterogeneous PPR sampling."
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
            self._get_destination_type(edge_type)
            for edge_types in self._node_type_to_edge_types.values()
            for edge_type in edge_types
        }
        all_node_types: list[NodeType] = sorted(
            source_node_types | destination_node_types
        )
        all_edge_types: list[EdgeType] = sorted(
            {
                edge_type
                for edge_types in self._node_type_to_edge_types.values()
                for edge_type in edge_types
            }
        )

        self._node_type_to_id: dict[NodeType, int] = {
            node_type: node_type_id
            for node_type_id, node_type in enumerate(all_node_types)
        }
        self._ntype_id_to_ntype: list[NodeType] = all_node_types
        self._etype_to_etype_id: dict[EdgeType, int] = {
            edge_type: edge_type_id
            for edge_type_id, edge_type in enumerate(all_edge_types)
        }
        self._etype_id_to_etype: list[EdgeType] = all_edge_types

        self._node_type_id_to_edge_type_ids: list[list[int]] = [
            [
                self._etype_to_etype_id[edge_type]
                for edge_type in self._node_type_to_edge_types.get(node_type, [])
            ]
            for node_type in all_node_types
        ]
        self._edge_type_id_to_dst_ntype_id: list[int] = [
            self._node_type_to_id[self._get_destination_type(edge_type)]
            for edge_type in all_edge_types
        ]
        # Degree tensors indexed by ntype_id.  Destination-only types get an empty
        # tensor; the C++ kernel returns 0 for those, matching _get_total_degree.
        self._degree_tensors_for_cpp: list[torch.Tensor] = [
            self._node_type_to_total_degree.get(
                node_type, torch.zeros(0, dtype=torch.int32)
            )
            for node_type in all_node_types
        ]

        self._typed_ppr_channel_to_node_type_id_to_edge_type_ids: TypedPPRChannelTraversalMaps = []
        if typed_channel_groups is not None:
            self._typed_ppr_channel_to_node_type_id_to_edge_type_ids = (
                build_edge_type_channel_group_edge_type_ids(
                    edge_type_groups=typed_channel_groups,
                    edge_type_to_edge_type_id=self._etype_to_etype_id,
                    node_type_to_edge_types=self._node_type_to_edge_types,
                    node_types=self._ntype_id_to_ntype,
                )
            )

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
        nodes_by_edge_type_id: dict[int, torch.Tensor],
    ) -> PPRForwardPushFetchMap:
        """Batch fetch neighbors for nodes grouped by integer edge type ID.

        Issues one one-hop request per edge type in the frontier. Each node's
        neighbor list is capped at ``self._num_neighbors_per_hop``.

        Args:
            nodes_by_edge_type_id: Dict mapping integer edge type ID to a 1-D int64
                tensor of node IDs to fetch neighbors for.  Comes directly from
                ``drain_queue()`` as CPU tensors; node IDs are already deduplicated.

        Returns:
            Dict mapping edge type ID to ``(source_nodes, flat_neighbors,
            counts, optional_edge_ids)``. ``flat_neighbors`` is the flat
            concatenation of all neighbor lists for that edge type; ``counts[i]``
            is the neighbor count for ``source_nodes[i]``.

        Example::

            nodes_by_edge_type_id = {
                2: tensor([0, 3]),   # edge type ID 2 -> nodes 0 and 3
                5: tensor([7]),      # edge type ID 5 -> node 7
            }
            # Might return (neighbor lists depend on graph structure):
            {
                2: (tensor([0, 3]), tensor([5, 9, 2, 1]), tensor([3, 1]), None),
                5: (tensor([7]), tensor([0, 3]), tensor([2]), None),
            }
        """
        edge_type_ids: list[int] = []
        sample_tasks = []
        for edge_type_id in nodes_by_edge_type_id:
            edge_type = self._etype_id_to_etype[edge_type_id]
            # _sample_one_hop expects None only for true homogeneous graphs.
            # Labeled homogeneous ABLP graphs are hetero-backed because label
            # edges are represented as separate edge types, so they still need
            # the explicit default edge type here.
            rpc_edge_type = (
                None
                if self._is_homogeneous and edge_type == _PPR_HOMOGENEOUS_EDGE_TYPE
                else edge_type
            )
            edge_type_ids.append(edge_type_id)
            # drain_queue materializes CPU frontier tensors; _sample_one_hop can
            # consume them directly, so avoid a sampler-device round trip here.
            sample_tasks.append(
                self._sample_one_hop(
                    srcs=nodes_by_edge_type_id[edge_type_id],
                    num_nbr=self._num_neighbors_per_hop,
                    etype=rpc_edge_type,
                )
            )
        outputs: list[NeighborOutput] = await asyncio.gather(*sample_tasks)
        return {
            edge_type_id: (
                nodes_by_edge_type_id[edge_type_id],
                output.nbr,
                output.nbr_num,
                output.edge if self._include_sampled_edges else None,
            )
            for edge_type_id, output in zip(edge_type_ids, outputs)
        }

    def _extract_ppr_state_top_k(
        self,
        ppr_state,
        device: torch.device,
    ) -> tuple[
        Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        Union[torch.Tensor, dict[NodeType, torch.Tensor]],
    ]:
        """Extract PPR neighbors from a completed C++ Forward Push state.

        The C++ kernel indexes node types by compact integer IDs for speed.
        This helper translates those IDs back to GiGL node-type keys and
        preserves the homogeneous return shape expected by the rest of the
        sampler.

        ``max_ppr_nodes`` is the maximum number of nodes returned for each
        source node. If residual top-up is enabled, residual candidates count
        against this cap.

        Returns:
            ``(flat_ids, flat_weights, valid_counts)`` for homogeneous graphs,
            or three dictionaries keyed by node type for heterogeneous graphs.
            ``flat_ids`` and ``flat_weights`` are concatenated across seeds;
            ``valid_counts`` stores how many selected nodes belong to each seed.
        """
        # Translate integer node-type IDs back to NodeType strings for the rest
        # of the pipeline, and move tensors to the correct device.
        node_type_to_flat_ids: dict[NodeType, torch.Tensor] = {}
        node_type_to_flat_weights: dict[NodeType, torch.Tensor] = {}
        node_type_to_valid_counts: dict[NodeType, torch.Tensor] = {}

        extracted_results = ppr_state.extract_top_k_with_residual_top_up(
            self._max_ppr_nodes,
            self._enable_residual_topup,
        )

        for node_type_id, (
            flat_ids,
            flat_weights,
            valid_counts,
        ) in extracted_results.items():
            node_type = self._ntype_id_to_ntype[node_type_id]
            # TODO: If these copies become a bottleneck, evaluate
            # non_blocking=True together with pinned extraction output tensors.
            node_type_to_flat_ids[node_type] = flat_ids.to(device)
            node_type_to_flat_weights[node_type] = flat_weights.to(device)
            node_type_to_valid_counts[node_type] = valid_counts.to(device)

        if self._is_homogeneous:
            return (
                node_type_to_flat_ids[DEFAULT_HOMOGENEOUS_NODE_TYPE],
                node_type_to_flat_weights[DEFAULT_HOMOGENEOUS_NODE_TYPE],
                node_type_to_valid_counts[DEFAULT_HOMOGENEOUS_NODE_TYPE],
            )
        else:
            return (
                node_type_to_flat_ids,
                node_type_to_flat_weights,
                node_type_to_valid_counts,
            )

    def _extract_typed_ppr_state_top_k(
        self,
        ppr_states,
        typed_ppr_channel_target_counts: list[int],
        device: torch.device,
    ) -> tuple[
        dict[NodeType, torch.Tensor],
        dict[NodeType, torch.Tensor],
        dict[NodeType, torch.Tensor],
    ]:
        """Extract typed PPR results and move output tensors to the sampler device."""
        extracted_results = extract_typed_top_k_with_residual_top_up(
            ppr_states,
            typed_ppr_channel_target_counts,
            self._enable_residual_topup,
        )
        node_type_to_flat_ids: dict[NodeType, torch.Tensor] = {}
        node_type_to_flat_weights: dict[NodeType, torch.Tensor] = {}
        node_type_to_valid_counts: dict[NodeType, torch.Tensor] = {}
        for node_type_id, (
            flat_ids,
            flat_weights,
            valid_counts,
        ) in extracted_results.items():
            node_type = self._ntype_id_to_ntype[node_type_id]
            node_type_to_flat_ids[node_type] = flat_ids.to(device)
            node_type_to_flat_weights[node_type] = flat_weights.to(device)
            node_type_to_valid_counts[node_type] = valid_counts.to(device)
        return (
            node_type_to_flat_ids,
            node_type_to_flat_weights,
            node_type_to_valid_counts,
        )

    async def _compute_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: Optional[NodeType] = None,
    ) -> PPRResult:
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
            A ``PPRResult``. For homogeneous graphs, ``node_ids``, ``weights``,
            and ``valid_counts`` are tensors. For heterogeneous graphs, they are
            ``dict[NodeType, Tensor]`` objects where each tensor has the same
            structure as the homogeneous case.

            - ``flat_neighbor_ids``: global neighbor IDs selected by top-k PPR
              score, concatenated across seeds.  For batch of size ``B`` with
              ``C_i`` neighbors for seed ``i``, shape is
              ``[sum(C_0, ..., C_{B-1})]``.
            - ``flat_edge_attrs``: ``[ppr_score, hop_proximity]`` rows
              corresponding to each entry in ``flat_neighbor_ids``, shape
              ``[sum(C_0, ..., C_{B-1}), 2]``.
            - ``valid_counts``: number of PPR neighbors contributed by each
              seed, shape ``[batch_size]``.  Used to slice the flat tensors into
              per-seed groups: seed ``i``'s neighbors are at
              ``flat_neighbor_ids[sum(valid_counts[:i]) : sum(valid_counts[:i+1])]``.
            - ``cache_state``: the completed C++ PPR state when
              ``include_sampled_edges`` is enabled, otherwise ``None``. Its
              neighbor cache is used by the optional original-edge output path
              without triggering extra graph-store reads.

        Example::

            # 4 seeds, valid_counts = [1, 3, 2, 0]  →  6 total (seed, neighbor) pairs
            flat_neighbor_ids = tensor([d0, d1a, d1b, d1c, d2a, d2b])
            flat_edge_attrs   = tensor([[w0, p0], [w1a, p1a], ..., [w2b, p2b]])
            valid_counts      = tensor([1, 3, 2, 0])
        """
        if seed_node_type is None:
            seed_node_type = DEFAULT_HOMOGENEOUS_NODE_TYPE
        device = seed_nodes.device
        loop = asyncio.get_running_loop()

        ppr_state = await loop.run_in_executor(
            None,
            PPRForwardPush,
            seed_nodes,
            self._node_type_to_id[seed_node_type],
            self._alpha,
            self._requeue_threshold_factor,
            self._node_type_id_to_edge_type_ids,
            self._edge_type_id_to_dst_ntype_id,
            self._degree_tensors_for_cpp,
        )

        fetch_iteration_count = 0

        # TODO: If Python/C++ boundary overhead in this loop becomes a blocker,
        # consider a coarser C++ PPR iteration API while keeping Python
        # responsible for async distributed neighbor fetches.
        while True:
            # drain_queue returns None when the queue is truly empty (convergence),
            # or a dict (possibly empty) when nodes were drained.  An empty dict
            # means all drained nodes either had cached neighbors or no outgoing
            # edges — we still call push_residuals to flush their residuals into
            # ppr_scores_.
            # The pybind wrapper releases the GIL, but a direct call would still
            # occupy this coroutine's event-loop thread until C++ returns.
            nodes_by_edge_type_id = await loop.run_in_executor(
                None, ppr_state.drain_queue
            )
            if nodes_by_edge_type_id is None:
                break

            fetch_budget_remaining = (
                self._max_fetch_iterations is None
                or fetch_iteration_count < self._max_fetch_iterations
            )
            if nodes_by_edge_type_id and fetch_budget_remaining:
                fetched_by_edge_type_id = await self._batch_fetch_neighbors(
                    nodes_by_edge_type_id
                )
                fetch_iteration_count += 1
            else:
                # Fetch budget exhausted; push_residuals will use the existing neighbor cache.
                fetched_by_edge_type_id = {}

            await loop.run_in_executor(
                None,
                ppr_state.push_residuals,
                fetched_by_edge_type_id,
            )

        node_ids, weights, valid_counts = await loop.run_in_executor(
            None,
            self._extract_ppr_state_top_k,
            ppr_state,
            device,
        )
        return PPRResult(
            node_ids=node_ids,
            weights=weights,
            valid_counts=valid_counts,
            cache_state=ppr_state if self._include_sampled_edges else None,
        )

    async def _compute_typed_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: NodeType,
        typed_ppr_channel_target_counts: list[int],
    ) -> PPRResult:
        """Run one PPR traversal per typed channel and extract the merged result.

        Each channel receives the same seed nodes but a different edge-type
        traversal allowlist. Fetch frontiers are unioned across active channels
        per iteration so shared graph neighborhoods are fetched once and reused
        by every channel that requested them. After convergence, C++ deduplicates
        candidates, attributes each node to its strongest channel, fills channel
        target counts, redistributes unused slots by score, and emits typed
        edge-attribute features in one extraction step.

        Args:
            seed_nodes: Global node IDs for the seed batch.
            seed_node_type: Heterogeneous node type for ``seed_nodes``.
            typed_ppr_channel_target_counts: Per-channel target output counts, aligned
                with ``self._typed_ppr_channel_to_node_type_id_to_edge_type_ids``.

        Returns:
            Heterogeneous PPR extraction output with typed edge-attribute
            feature vectors, plus completed C++ PPR states when
            ``include_sampled_edges`` is enabled, otherwise
            ``None``. Their neighbor caches are used by the optional
            original-edge output path without triggering extra graph-store
            reads.
        """
        device = seed_nodes.device
        loop = asyncio.get_running_loop()

        # Build one Forward Push state per typed channel. All states use the
        # same seeds, restart probability, degree tensors, and destination-type
        # map; only the per-node-type edge traversal allowlist differs.
        def build_ppr_states() -> list[PPRForwardPush]:
            return [
                PPRForwardPush(
                    seed_nodes,
                    self._node_type_to_id[seed_node_type],
                    self._alpha,
                    self._requeue_threshold_factor,
                    node_type_id_to_edge_type_ids,
                    self._edge_type_id_to_dst_ntype_id,
                    self._degree_tensors_for_cpp,
                )
                for node_type_id_to_edge_type_ids in (
                    self._typed_ppr_channel_to_node_type_id_to_edge_type_ids
                )
            ]

        ppr_states = await loop.run_in_executor(None, build_ppr_states)
        fetch_iteration_counts = [0 for _ in ppr_states]
        max_fetch_iterations = (
            self._max_fetch_iterations if self._max_fetch_iterations is not None else -1
        )

        while True:
            (
                drained_channel_indices,
                fetch_channel_indices,
                edge_type_ids_by_fetch_channel,
                unioned_node_ids_by_edge_type_id,
            ) = await loop.run_in_executor(
                None,
                drain_typed_ppr_channel_queues,
                ppr_states,
                fetch_iteration_counts,
                max_fetch_iterations,
            )
            if not drained_channel_indices:
                break

            fetched_by_channel: list[PPRForwardPushFetchMap] = [
                dict() for _ in ppr_states
            ]

            if unioned_node_ids_by_edge_type_id:
                union_fetched_by_edge_type_id = await self._batch_fetch_neighbors(
                    unioned_node_ids_by_edge_type_id
                )
                for channel_index, edge_type_ids in zip(
                    fetch_channel_indices,
                    edge_type_ids_by_fetch_channel,
                    strict=True,
                ):
                    fetch_iteration_counts[channel_index] += 1
                    fetched_by_channel[channel_index] = {
                        edge_type_id: union_fetched_by_edge_type_id[edge_type_id]
                        for edge_type_id in edge_type_ids
                    }

            # Push every non-converged channel. The fetched_by_channel entry is
            # empty for channels that have no new fetch work; PPRForwardPush will
            # use its cached neighbors in that case.
            push_tasks = [
                loop.run_in_executor(
                    None,
                    ppr_states[channel_index].push_residuals,
                    fetched_by_channel[channel_index],
                )
                for channel_index in drained_channel_indices
            ]
            await asyncio.gather(*push_tasks)

        node_ids, weights, valid_counts = await loop.run_in_executor(
            None,
            self._extract_typed_ppr_state_top_k,
            ppr_states,
            typed_ppr_channel_target_counts,
            device,
        )
        return PPRResult(
            node_ids=node_ids,
            weights=weights,
            valid_counts=valid_counts,
            cache_state=ppr_states if self._include_sampled_edges else None,
        )

    def _extract_original_edges_from_ppr_caches(
        self,
        node_dict: dict[NodeType, torch.Tensor],
        ppr_cache_states: list[PPRForwardPush],
    ) -> SampledEdgeResult:
        """Extract original edges over selected nodes from C++ PPR caches.

        This intentionally does not issue another graph-store request; it extracts
        sampled edge rows already cached while computing PPR.

        Returns:
            GLT sampled-edge dictionaries:
            ``rows`` maps each original edge type to local source indices.
            ``cols`` maps each original edge type to local destination indices.
            ``edge_ids`` maps each original edge type to edge IDs when ``with_edge``
            is enabled, otherwise it is ``None``.
            ``num_sampled_edges`` maps each original edge type to GLT's per-hop edge
            counts list, using one count because these edges are attached as a
            single sampled-edge layer.
        """
        selected_node_ids_by_node_type_id = {
            self._node_type_to_id[node_type]: node_ids.cpu()
            for node_type, node_ids in node_dict.items()
        }
        extracted_edges = extract_original_edges_from_ppr_caches(
            ppr_cache_states,
            selected_node_ids_by_node_type_id,
            self.with_edge,
        )

        rows_dict: dict[EdgeType, torch.Tensor] = {}
        cols_dict: dict[EdgeType, torch.Tensor] = {}
        edge_dict: dict[EdgeType, torch.Tensor] = {}
        for edge_type_id, (rows, cols, edge_ids) in extracted_edges.items():
            edge_type = self._etype_id_to_etype[edge_type_id]
            output_edge_type = (
                reverse_edge_type(edge_type) if self.edge_dir == "in" else edge_type
            )
            rows_dict[output_edge_type] = rows.to(self.device)
            cols_dict[output_edge_type] = cols.to(self.device)
            if edge_ids is not None:
                edge_dict[output_edge_type] = edge_ids.to(self.device)

        if not rows_dict:
            empty_edge_dict = {} if self.with_edge else None
            return SampledEdgeResult(
                rows={},
                cols={},
                edge_ids=empty_edge_dict,
                num_sampled_edges={},
            )

        num_sampled_edges = {
            edge_type: [int(cols.size(0))] for edge_type, cols in cols_dict.items()
        }
        return SampledEdgeResult(
            rows=rows_dict,
            cols=cols_dict,
            edge_ids=edge_dict if self.with_edge else None,
            num_sampled_edges=num_sampled_edges,
        )

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
        ``edge_index`` directly indexes into ``data[node_type].x`` without any
        additional global→local remapping.

        The inducer is GLT's C++ data structure (backed by a per-node-type hash map)
        that maintains a single global-ID → local-index mapping for the entire
        subgraph being built.  We use it here instead of a Python dict for two reasons:

        1. **Consistency across seed types.** For heterogeneous ABLP inputs,
           ``_compute_ppr_scores`` is called once per seed type (anchors, supervision
           nodes, …).  A node reachable from multiple seed types must receive the
           *same* local index in ``node_dict[node_type]`` regardless of which seed type
           discovered it.  The inducer is shared across all those calls, so it
           guarantees this automatically.

        2. **Performance.** The inducer's C++ hash map is faster than a Python dict
           for per-node lookups on large graphs, and its lifecycle is already managed
           by GLT's inducer pool (``_acquire_inducer`` / ``inducer_pool.put``).

        The API used here mirrors GLT's own ``DistNeighborSampler._sample_from_nodes``:

        - ``inducer.init_node(seeds)`` registers seed nodes and returns their global
          IDs (local indices 0, 1, … are assigned internally).
        - ``inducer.induce_next(source_nodes, flat_neighbors, counts)`` (homogeneous)
          or ``inducer.induce_next(neighbor_dict)`` (heterogeneous) deduplicates
          neighbors against all previously seen nodes and returns:

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
        #    This ensures node_dict[node_type] has no duplicates.
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
            nodes_by_seed_type = nodes_to_sample

            # Register all seeds (anchors + supervision nodes for ABLP) with the
            # inducer first, so they occupy the lowest local indices.  source_dict maps
            # NodeType -> global IDs (same values as nodes_to_sample).
            source_dict: dict[NodeType, torch.Tensor] = inducer.init_node(
                nodes_by_seed_type
            )

            # Compute PPR for all seed types concurrently, collecting flat global
            # neighbor IDs, weights, and per-seed counts.  Build neighbor_dict for a
            # single inducer.induce_next call using PPR edge types
            # (seed_type, 'ppr', neighbor_type). The inducer only cares about
            # source and destination node types, so the relation name is arbitrary.
            #
            # Each seed type's PPR computation is entirely independent: it creates
            # its own PPRForwardPush and only reads shared sampler attributes
            # (degree tensors, edge-type maps) which are immutable after __init__.
            # Running them with asyncio.gather allows their fetch phases to overlap,
            # which is most beneficial when there are 2+ distinct seed node types
            # (e.g. cross-type supervision edges like user→story).
            seed_types = list(nodes_by_seed_type.keys())
            typed_ppr_channel_target_counts = self._typed_ppr_channel_target_counts
            if typed_ppr_channel_target_counts is None:
                ppr_results = await asyncio.gather(
                    *[
                        self._compute_ppr_scores(
                            nodes_by_seed_type[seed_type],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                            seed_type,
                        )
                        for seed_type in seed_types
                    ]
                )
            else:
                ppr_results = await asyncio.gather(
                    *[
                        self._compute_typed_ppr_scores(
                            nodes_by_seed_type[seed_type],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                            seed_type,
                            typed_ppr_channel_target_counts,
                        )
                        for seed_type in seed_types
                    ]
                )
            neighbor_dict: dict[EdgeType, list[torch.Tensor]] = {}
            ppr_edge_type_to_flat_weights: dict[EdgeType, torch.Tensor] = {}

            for seed_type, ppr_result in zip(seed_types, ppr_results):
                assert isinstance(ppr_result.node_ids, dict)
                assert isinstance(ppr_result.weights, dict)
                assert isinstance(ppr_result.valid_counts, dict)
                node_type_to_flat_ids = cast(
                    dict[NodeType, torch.Tensor], ppr_result.node_ids
                )
                node_type_to_flat_weights = cast(
                    dict[NodeType, torch.Tensor], ppr_result.weights
                )
                node_type_to_valid_counts = cast(
                    dict[NodeType, torch.Tensor], ppr_result.valid_counts
                )

                for node_type, flat_ids in node_type_to_flat_ids.items():
                    ppr_edge_type: EdgeType = (seed_type, "ppr", node_type)
                    valid_counts = node_type_to_valid_counts[node_type]
                    ppr_edge_type_to_flat_weights[ppr_edge_type] = (
                        node_type_to_flat_weights[node_type]
                    )

                    # Skip empty pairs; induce_next handles deduplication across
                    # seed types so a neighbor reachable from multiple seed types
                    # gets one consistent local index in node_dict[node_type].
                    if flat_ids.numel() > 0:
                        neighbor_dict[ppr_edge_type] = [
                            source_dict[seed_type],
                            flat_ids,
                            valid_counts,
                        ]

            # induce_next processes all PPR edge types in neighbor_dict in one
            # pass, assigning local indices to neighbors not yet registered and
            # deduplicating nodes seen from multiple seed types.  Returns:
            #   new_nodes_dict[NodeType] -> global IDs of nodes not previously
            #                              registered with the inducer
            #   rows_dict[EdgeType]     -> flat local source indices per virtual
            #                              edge type, expanded to match flat_ids
            #   cols_dict[EdgeType]     -> flat local destination indices, one
            #                              per neighbor in the same order as the
            #                              flat_ids passed in neighbor_dict
            new_nodes_dict, rows_dict, cols_dict = inducer.induce_next(neighbor_dict)

            # node_dict = seeds (already in source_dict) + PPR neighbors not
            # previously registered.  merge_dict appends tensors into lists;
            # cat collapses them.
            heterogeneous_output_nodes: dict[NodeType, list[torch.Tensor]] = (
                defaultdict(list)
            )
            merge_dict(source_dict, heterogeneous_output_nodes)
            merge_dict(new_nodes_dict, heterogeneous_output_nodes)
            node_dict = {
                node_type: torch.cat(nodes)
                for node_type, nodes in heterogeneous_output_nodes.items()
                if nodes
            }

            sampled_edges = SampledEdgeResult(
                rows={},
                cols={},
                edge_ids={} if self.with_edge else None,
                num_sampled_edges={},
            )
            if self._include_sampled_edges:
                ppr_cache_states: list[PPRForwardPush] = []
                for ppr_result in ppr_results:
                    ppr_cache_state = ppr_result.cache_state
                    if isinstance(ppr_cache_state, PPRForwardPush):
                        ppr_cache_states.append(ppr_cache_state)
                    elif ppr_cache_state is not None:
                        ppr_cache_states.extend(ppr_cache_state)
                # Keep the cache scan off the asyncio event-loop thread. The
                # pybind wrapper releases the GIL around the C++ extraction.
                sampled_edges = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._extract_original_edges_from_ppr_caches,
                    node_dict,
                    ppr_cache_states,
                )

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
                edge_type_repr = repr(ppr_edge_type)
                metadata[f"{PPR_EDGE_INDEX_METADATA_KEY}{edge_type_repr}"] = edge_index
                metadata[f"{PPR_WEIGHT_METADATA_KEY}{edge_type_repr}"] = flat_weights

            sample_output = HeteroSamplerOutput(
                node=node_dict,
                # Virtual PPR edge types are unknown to GLT, so they are passed
                # through metadata and attached in the loader's _collate_fn.
                # Optional original edges use regular GLT row/col output because
                # those edge types are part of the dataset schema.
                row=sampled_edges.rows,
                col=sampled_edges.cols,
                edge=sampled_edges.edge_ids,
                batch={input_type: input_seeds},
                num_sampled_nodes={
                    node_type: [nodes.size(0)] for node_type, nodes in node_dict.items()
                },
                num_sampled_edges=sampled_edges.num_sampled_edges,
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
            # source_nodes holds their global IDs (same values as nodes_to_sample).
            source_nodes = inducer.init_node(homogeneous_nodes_to_sample)

            ppr_result = await self._compute_ppr_scores(homogeneous_nodes_to_sample, None)
            homogeneous_flat_ids = ppr_result.node_ids
            homogeneous_flat_weights = ppr_result.weights
            homogeneous_valid_counts = ppr_result.valid_counts
            assert isinstance(homogeneous_flat_ids, torch.Tensor)
            assert isinstance(homogeneous_flat_weights, torch.Tensor)
            assert isinstance(homogeneous_valid_counts, torch.Tensor)

            # induce_next deduplicates homogeneous_flat_ids against already-seen nodes
            # (the seeds registered above) and returns:
            #   new_nodes: global IDs of nodes not previously registered
            #             with the inducer.
            #   rows: flat local source indices (one per neighbor, expanded).
            #   cols: flat local destination indices for every neighbor, in the
            #         same order as homogeneous_flat_ids.
            new_nodes, rows, cols = inducer.induce_next(
                source_nodes, homogeneous_flat_ids, homogeneous_valid_counts
            )
            all_nodes = torch.cat([source_nodes, new_nodes])

            ppr_edge_index = torch.stack([rows, cols])

            metadata["edge_index"] = ppr_edge_index
            metadata["edge_attr"] = homogeneous_flat_weights

            sample_output = SamplerOutput(
                node=all_nodes,
                # row/col/edge are left empty for parity with the hetero case above.
                row=torch.tensor([], dtype=torch.long, device=self.device),
                col=torch.tensor([], dtype=torch.long, device=self.device),
                edge=torch.tensor(
                    [], dtype=torch.long, device=self.device
                ),  # Empty tensor — GLT SampleQueue requires all values to be tensors
                batch=input_seeds,
                num_sampled_nodes=[source_nodes.size(0), new_nodes.size(0)],
                num_sampled_edges=[],
                metadata=metadata,
            )

        self.inducer_pool.put(inducer)
        return sample_output
