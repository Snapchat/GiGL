import asyncio
from collections import defaultdict
from typing import Optional, Union

import torch

# TODO: Once gigl_core has a stable Python interface, re-export PPRForwardPush
# under a gigl.core namespace rather than importing directly from the C++ extension.
from gigl_core import PPRForwardPush, drain_typed_ppr_channel_queues
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)
from graphlearn_torch.typing import EdgeType, NodeType
from graphlearn_torch.utils import merge_dict

from gigl.distributed.base_sampler import BaseDistNeighborSampler
from gigl.distributed.utils.distributed_typed_sampler import (
    HeteroPPRResult,
    PPRResult,
    TypedPPRChannelKey,
    build_edge_type_channel_group_edge_type_ids,
    merge_typed_ppr_results,
    parse_typed_channel_quota_groups,
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
        - ``data[(seed_type, "ppr", neighbor_type)].edge_index``: same format as above.
        - ``data[(seed_type, "ppr", neighbor_type)].edge_attr``: scalar PPR
          score for regular PPR. For typed PPR, edge attrs are multi-column:
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
        typed_channel_quotas: Optional[dict[TypedPPRChannelKey, int]] = None,
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

        typed_channel_groups, self._typed_ppr_channel_quotas = (
            parse_typed_channel_quota_groups(typed_channel_quotas)
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

        if typed_channel_groups is not None:
            self._typed_ppr_channel_to_node_type_id_to_edge_type_ids = (
                build_edge_type_channel_group_edge_type_ids(
                    edge_type_groups=typed_channel_groups,
                    edge_type_to_edge_type_id=self._etype_to_etype_id,
                    node_type_to_edge_types=self._node_type_to_edge_types,
                    node_types=self._ntype_id_to_ntype,
                )
            )
        else:
            self._typed_ppr_channel_to_node_type_id_to_edge_type_ids = []

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
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Batch fetch neighbors for nodes grouped by integer edge type ID.

        Issues one one-hop request per edge type in the frontier. Each node's
        neighbor list is capped at ``self._num_neighbors_per_hop``.

        Args:
            nodes_by_edge_type_id: Dict mapping integer edge type ID to a 1-D int64
                tensor of node IDs to fetch neighbors for.  Comes directly from
                ``drain_queue()``; node IDs are already deduplicated.

        Returns:
            Dict mapping edge type ID to ``(node_ids, flat_neighbors, counts)``
            as int64 tensors, ready to pass directly to ``push_residuals``.
            ``flat_neighbors`` is the flat concatenation of all neighbor lists
            for that edge type; ``counts[i]`` is the neighbor count for
            ``node_ids[i]``.

        Example::

            nodes_by_edge_type_id = {
                2: tensor([0, 3]),   # edge type ID 2 -> nodes 0 and 3
                5: tensor([7]),      # edge type ID 5 -> node 7
            }
            # Might return (neighbor lists depend on graph structure):
            {
                2: (tensor([0, 3]), tensor([5, 9, 2, 1]), tensor([3, 1])),
                5: (tensor([7]),    tensor([0, 3]),        tensor([2])),
            }
        """
        edge_type_ids: list[int] = []
        sample_tasks = []
        for edge_type_id in nodes_by_edge_type_id:
            edge_type = self._etype_id_to_etype[edge_type_id]
            # The lower sampler expects None only for true homogeneous graphs.
            # Labeled homogeneous ABLP graphs are hetero-backed because label
            # edges are represented as separate edge types, so they still need
            # the explicit default edge type here.
            rpc_edge_type = (
                None
                if self._is_homogeneous and edge_type == _PPR_HOMOGENEOUS_EDGE_TYPE
                else edge_type
            )
            edge_type_ids.append(edge_type_id)
            sample_tasks.append(
                self._sample_one_hop(
                    nodes_by_edge_type_id[edge_type_id],
                    self._num_neighbors_per_hop,
                    rpc_edge_type,
                )
            )

        outputs = await asyncio.gather(*sample_tasks)
        return {
            edge_type_id: (
                nodes_by_edge_type_id[edge_type_id],
                output.nbr,
                output.nbr_num,
            )
            for edge_type_id, output in zip(edge_type_ids, outputs)
        }

    def _extract_ppr_state_top_k(
        self,
        ppr_state,
        device: torch.device,
        ppr_node_limit: int,
    ) -> PPRResult:
        """Extract PPR neighbors from a completed C++ Forward Push state.

        The C++ kernel indexes node types by compact integer IDs for speed.
        This helper translates those IDs back to GiGL node-type keys and
        preserves the homogeneous return shape expected by the rest of the
        sampler.

        ``ppr_node_limit`` is the combined cap for this extraction. If residual
        top-up is enabled, C++ derives the residual candidate budget from this
        cap after selecting finalized PPR nodes.

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
            ppr_node_limit,
            self._enable_residual_topup,
        )

        for node_type_id, (
            flat_ids,
            flat_weights,
            valid_counts,
        ) in extracted_results.items():
            node_type = self._ntype_id_to_ntype[node_type_id]
            node_type_to_flat_ids[node_type] = flat_ids.to(device)
            node_type_to_flat_weights[node_type] = flat_weights.to(device)
            node_type_to_valid_counts[node_type] = valid_counts.to(device)

        if self._is_homogeneous:
            assert (
                len(node_type_to_flat_ids) == 1
                and DEFAULT_HOMOGENEOUS_NODE_TYPE in node_type_to_flat_ids
            )
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

        ppr_state = PPRForwardPush(
            seed_nodes,
            self._node_type_to_id[seed_node_type],
            self._alpha,
            self._requeue_threshold_factor,
            self._node_type_id_to_edge_type_ids,
            self._edge_type_id_to_dst_ntype_id,
            self._degree_tensors_for_cpp,
        )

        fetch_iteration_count = 0
        loop = asyncio.get_running_loop()
        nodes_by_edge_type_id = ppr_state.drain_queue()

        # drain_queue returns None when the queue is truly empty (convergence),
        # or a dict (possibly empty) when nodes were drained.  An empty dict
        # means all drained nodes either had cached neighbors or no outgoing
        # edges — we still push residuals to flush them into ppr_scores_.
        while nodes_by_edge_type_id is not None:
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
            nodes_by_edge_type_id = ppr_state.drain_queue()

        return self._extract_ppr_state_top_k(
            ppr_state,
            device,
            ppr_node_limit=self._max_ppr_nodes,
        )

    async def _compute_typed_ppr_scores(
        self,
        seed_nodes: torch.Tensor,
        seed_node_type: NodeType,
    ) -> HeteroPPRResult:
        """Run one PPR state per typed channel and merge the channel results.

        Each channel receives the same seed nodes but a different edge-type
        traversal allowlist. Fetch frontiers are unioned across active channels
        per iteration so shared graph neighborhoods are fetched once and reused
        by every channel that requested them.

        Args:
            seed_nodes: Global node IDs for the seed batch.
            seed_node_type: Heterogeneous node type for ``seed_nodes``.

        Returns:
            Heterogeneous PPR extraction output with typed edge-attribute
            feature vectors.
        """
        assert self._typed_ppr_channel_quotas is not None
        channel_quotas = self._typed_ppr_channel_quotas
        device = seed_nodes.device

        # Build one Forward Push state per typed channel. All states use the
        # same seeds, restart probability, degree tensors, and destination-type
        # map; only the per-node-type edge traversal allowlist differs.
        ppr_states = [
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
        fetch_iteration_counts = [0 for _ in ppr_states]
        max_fetch_iterations = (
            self._max_fetch_iterations if self._max_fetch_iterations is not None else -1
        )
        loop = asyncio.get_running_loop()

        while True:
            (
                active_channel_indices,
                fetch_channel_indices,
                edge_type_ids_by_fetch_channel,
                union_nodes_by_edge_type_id,
            ) = drain_typed_ppr_channel_queues(
                ppr_states,
                fetch_iteration_counts,
                max_fetch_iterations,
            )
            if not active_channel_indices:
                break

            fetched_by_channel: list[
                dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
            ] = [dict() for _ in ppr_states]

            if union_nodes_by_edge_type_id:
                union_fetched_by_edge_type_id = await self._batch_fetch_neighbors(
                    union_nodes_by_edge_type_id
                )
                for channel_index, edge_type_ids in zip(
                    fetch_channel_indices,
                    edge_type_ids_by_fetch_channel,
                ):
                    fetch_iteration_counts[channel_index] += 1
                    fetched_by_channel[channel_index] = {
                        edge_type_id: union_fetched_by_edge_type_id[edge_type_id]
                        for edge_type_id in edge_type_ids
                        if edge_type_id in union_fetched_by_edge_type_id
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
                for channel_index in active_channel_indices
            ]
            if push_tasks:
                await asyncio.gather(*push_tasks)

        channel_results = [
            self._extract_ppr_state_top_k(
                ppr_state,
                device=device,
                ppr_node_limit=channel_quota,
            )
            for ppr_state, channel_quota in zip(ppr_states, channel_quotas)
        ]
        return merge_typed_ppr_results(
            channel_results=channel_results,
            channel_quotas=channel_quotas,
            max_ppr_nodes=self._max_ppr_nodes,
            device=device,
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

            # Register all seeds (anchors + supervision nodes for ABLP) with the
            # inducer first, so they occupy the lowest local indices.  source_dict maps
            # NodeType -> global IDs (same values as nodes_to_sample).
            source_dict = inducer.init_node(nodes_to_sample)

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
            seed_types = list(nodes_to_sample.keys())
            if self._typed_ppr_channel_quotas is None:
                ppr_results = await asyncio.gather(
                    *[
                        self._compute_ppr_scores(nodes_to_sample[seed_type], seed_type)
                        for seed_type in seed_types
                    ]
                )
            else:
                ppr_results = await asyncio.gather(
                    *[
                        self._compute_typed_ppr_scores(
                            nodes_to_sample[seed_type], seed_type
                        )
                        for seed_type in seed_types
                    ]
                )

            neighbor_dict: dict[EdgeType, list[torch.Tensor]] = {}
            ppr_edge_type_to_flat_weights: dict[EdgeType, torch.Tensor] = {}

            for seed_type, (
                node_type_to_flat_ids,
                node_type_to_flat_weights,
                node_type_to_valid_counts,
            ) in zip(seed_types, ppr_results):
                assert isinstance(node_type_to_flat_ids, dict)
                assert isinstance(node_type_to_flat_weights, dict)
                assert isinstance(node_type_to_valid_counts, dict)

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
                    node_type: [nodes.size(0)] for node_type, nodes in node_dict.items()
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
            # source_nodes holds their global IDs (same values as nodes_to_sample).
            source_nodes = inducer.init_node(homogeneous_nodes_to_sample)

            (
                homogeneous_flat_ids,
                homogeneous_flat_weights,
                homogeneous_valid_counts,
            ) = await self._compute_ppr_scores(homogeneous_nodes_to_sample, None)
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
