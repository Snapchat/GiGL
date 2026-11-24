"""Unit tests for DistPPRNeighborSampler correctness via DistNeighborLoader and DistABLPLoader.

Verifies that the PPR scores produced by the distributed sampler match
NetworkX's ``pagerank`` with personalization — a well-tested, independent
PPR implementation.

Note on compatability with NetworkX:

Both our forward push algorithm (Andersen et al., 2006) and NetworkX's
``pagerank`` (power iteration) compute Personalized PageRank — they are
different solvers for the same quantity.  With a small residual tolerance
(eps=1e-6), forward push converges close enough that per-node scores match
NetworkX within atol=1e-6 (observed deltas are ~1e-7).

Another note is that our ``alpha`` is the *restart* (teleport) probability — the probability of
jumping back to the seed at each step.  NetworkX's ``alpha`` is the *damping
factor* — the probability of following an edge.  These are complements::

    nx_alpha = 1 - our_alpha

Finally, with ``edge_dir="in"``, the PPR walk from node v follows *incoming* edges —
it moves to nodes u where edge (u, v) exists in the graph.  NetworkX's
``pagerank`` follows *outgoing* edges.  To make NetworkX traverse the same
neighbors as the sampler, we reverse the edges when building the reference
graph (add dst→src instead of src→dst).  When ``edge_dir="out"``, no
reversal is needed since both follow the original edge direction.
"""

import heapq
from collections import defaultdict
from typing import Literal, Optional

import networkx as nx
import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from graphlearn_torch.typing import reverse_edge_type
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.dist_ppr_sampler import (
    PPRResult,
    cap_ppr_result_to_top_k,
    cap_rows_to_per_seed_top_k,
)
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.sampler_options import PPRSamplerOptions
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
)
from tests.test_assets.distributed.test_dataset import (
    STORY,
    STORY_TO_USER,
    USER,
    USER_TO_STORY,
    create_heterogeneous_dataset,
    create_heterogeneous_dataset_for_ablp,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

# ---------------------------------------------------------------------------
# Homogeneous test graph (5 nodes, undirected edges stored as bidirectional)
#
#     0 --- 1 --- 3
#     |     |
#     2 --- +
#     |
#     4
#
# Undirected edges: {0-1, 0-2, 1-2, 1-3, 2-4}
# ---------------------------------------------------------------------------
_TEST_EDGE_INDEX = torch.tensor(
    [
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 4],
        [1, 0, 2, 0, 2, 1, 3, 1, 4, 2],
    ]
)
_NUM_TEST_NODES = 5

# ---------------------------------------------------------------------------
# Heterogeneous bipartite test graph (3 users, 3 stories)
#   USER_TO_STORY: user 0 -> {story 0, story 1}
#                  user 1 -> {story 1, story 2}
#                  user 2 -> {story 0, story 2}
#   STORY_TO_USER: reverse of USER_TO_STORY
# ---------------------------------------------------------------------------
_TEST_HETERO_EDGE_INDICES = {
    USER_TO_STORY: torch.tensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 0, 2]]),
    STORY_TO_USER: torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 0, 1, 1, 2]]),
}
_NUM_TEST_USERS = 3
_NUM_TEST_STORIES = 3

_TEST_ALPHA = 0.5
_TEST_EPS = 1e-6
_TEST_MAX_PPR_NODES = 5


# ---------------------------------------------------------------------------
# Reference PPR implementations (NetworkX-based)
# ---------------------------------------------------------------------------
def _build_reference_graph(edge_dir: Literal["in", "out"] = "in") -> nx.DiGraph:
    """Build a NetworkX DiGraph matching the homogeneous test edge_index.

    With ``edge_dir="in"``, edges are reversed (dst→src) so that NetworkX's
    outgoing-edge traversal matches GLT's incoming-edge PPR walk.  With
    ``edge_dir="out"``, edges keep their original direction (src→dst).

    See the module docstring for a full explanation of why reversal is needed.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(_NUM_TEST_NODES))
    src = _TEST_EDGE_INDEX[0].tolist()
    dst = _TEST_EDGE_INDEX[1].tolist()
    if edge_dir == "in":
        graph.add_edges_from(zip(dst, src))
    else:
        graph.add_edges_from(zip(src, dst))
    return graph


def _reference_ppr(
    graph: nx.DiGraph,
    seed: int,
    alpha: float,
    max_ppr_nodes: int,
) -> dict[int, float]:
    """Compute reference PPR scores for a homogeneous graph using NetworkX.

    See the module docstring for the alpha mapping rationale.

    Args:
        graph: NetworkX DiGraph with edges oriented for the sampling direction.
        seed: Seed node ID.
        alpha: Restart probability (our convention).
        max_ppr_nodes: Maximum number of top-scoring nodes to return.

    Returns:
        Dict mapping node_id -> PPR score for the top-k nodes.
    """
    personalization = {n: 0.0 for n in graph.nodes()}
    personalization[seed] = 1.0

    scores = nx.pagerank(
        graph, alpha=1 - alpha, personalization=personalization, tol=1e-12
    )
    top_k = heapq.nlargest(max_ppr_nodes, scores.items(), key=lambda x: x[1])
    return dict(top_k)


def _build_hetero_reference_graph(edge_dir: Literal["in", "out"] = "in") -> nx.DiGraph:
    """Build a NetworkX DiGraph for the heterogeneous test graph.

    Nodes are ``(type_str, id)`` tuples.  Edge direction is handled the same
    way as :func:`_build_reference_graph` — see the module docstring for the
    full explanation of why reversal is needed for ``edge_dir="in"``.
    """
    graph = nx.DiGraph()
    for i in range(_NUM_TEST_USERS):
        graph.add_node((str(USER), i))
    for i in range(_NUM_TEST_STORIES):
        graph.add_node((str(STORY), i))

    for edge_type, edge_index in _TEST_HETERO_EDGE_INDICES.items():
        src_type, _, dst_type = edge_type
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        if edge_dir == "in":
            for s, d in zip(src, dst):
                graph.add_edge((str(dst_type), d), (str(src_type), s))
        else:
            for s, d in zip(src, dst):
                graph.add_edge((str(src_type), s), (str(dst_type), d))

    return graph


def _reference_ppr_hetero(
    graph: nx.DiGraph,
    seed: int,
    seed_type: str,
    alpha: float,
    max_ppr_nodes: int,
) -> dict[str, dict[int, float]]:
    """Compute reference PPR scores for a heterogeneous graph using NetworkX.

    See the module docstring for the alpha mapping rationale.

    Args:
        graph: NetworkX DiGraph with ``(type_str, id)`` tuple nodes.
        seed: Seed node ID.
        seed_type: Node type string of the seed.
        alpha: Restart probability (our convention).
        max_ppr_nodes: Maximum top-scoring nodes to return per node type.

    Returns:
        Dict mapping node_type_str -> {node_id: PPR score} for top-k per type.
    """
    personalization = {n: 0.0 for n in graph.nodes()}
    personalization[(seed_type, seed)] = 1.0

    scores = nx.pagerank(
        graph, alpha=1 - alpha, personalization=personalization, tol=1e-12
    )

    type_to_scores: dict[str, dict[int, float]] = defaultdict(dict)
    for (ntype, nid), score in scores.items():
        type_to_scores[ntype][nid] = score

    result: dict[str, dict[int, float]] = {}
    for ntype, type_scores in type_to_scores.items():
        top_k = heapq.nlargest(max_ppr_nodes, type_scores.items(), key=lambda x: x[1])
        result[ntype] = dict(top_k)

    return result


# ---------------------------------------------------------------------------
# Shared verification helpers
# ---------------------------------------------------------------------------
def _extract_hetero_ppr_scores(
    datum: HeteroData,
    seed_type: str,
    node_types: list[str],
) -> dict[str, dict[int, float]]:
    """Extract and validate PPR metadata from a HeteroData batch.

    Verifies tensor shapes and invariants (positive weights, valid indices),
    maps local indices to global IDs, and returns scores grouped by node type.

    Args:
        datum: A single HeteroData batch (batch_size=1).
        seed_type: The seed node type used to key PPR metadata attributes.
        node_types: Node types to extract PPR scores for.

    Returns:
        Dict mapping node_type_str -> {global_node_id: ppr_score}.
    """
    ntype_to_sampler_ppr: dict[str, dict[int, float]] = {}
    for ntype in node_types:
        ppr_edge_type = (seed_type, "ppr", ntype)
        assert ppr_edge_type in datum.edge_types, (
            f"Missing PPR edge type {ppr_edge_type} on HeteroData"
        )

        ppr_edge_index = datum[ppr_edge_type].edge_index
        ppr_edge_attr = datum[ppr_edge_type].edge_attr

        assert ppr_edge_index.dim() == 2 and ppr_edge_index.size(0) == 2, (
            f"Expected [2, X] edge_index, got shape {list(ppr_edge_index.shape)}"
        )
        assert ppr_edge_attr.dim() == 2
        assert ppr_edge_attr.size(1) == 2
        assert ppr_edge_index.size(1) == ppr_edge_attr.size(0)
        ppr_weights = ppr_edge_attr[:, 0]
        ppr_hop_proximities = ppr_edge_attr[:, 1]
        assert (ppr_weights > 0).all(), f"PPR weights for {ntype} must be positive"
        _assert_valid_hop_proximities(ppr_hop_proximities)
        assert (ppr_edge_index[0] == 0).all(), (
            "All src indices must be 0 for batch_size=1"
        )

        global_node_ids = datum[ntype].node
        type_ppr: dict[int, float] = {}
        for j in range(ppr_edge_index.size(1)):
            local_dst = ppr_edge_index[1, j].item()
            global_dst = global_node_ids[local_dst].item()
            type_ppr[global_dst] = ppr_weights[j].item()
        ntype_to_sampler_ppr[str(ntype)] = type_ppr

    return ntype_to_sampler_ppr


def _assert_valid_hop_proximities(hop_proximities: torch.Tensor) -> None:
    """Assert PPR hop proximity is bounded and invertible for emitted rows."""
    assert (hop_proximities > 0).all()
    assert (hop_proximities <= 1).all()
    recovered_hops = (1 - hop_proximities) / hop_proximities
    assert torch.allclose(recovered_hops, recovered_hops.round())


def _assert_typed_channel_hop_proximities(
    channel_hop_proximities: torch.Tensor,
    channel_presence: torch.Tensor,
) -> None:
    """Assert typed-channel hop proximity is bounded and zero when missing."""
    assert (channel_hop_proximities >= 0).all()
    assert (channel_hop_proximities <= 1).all()
    present_channel_mask = channel_presence == 1
    assert present_channel_mask.any(dim=1).all()
    assert (channel_hop_proximities[present_channel_mask] > 0).all()
    recovered_present_hops = (
        1 - channel_hop_proximities[present_channel_mask]
    ) / channel_hop_proximities[present_channel_mask]
    assert torch.allclose(recovered_present_hops, recovered_present_hops.round())
    assert (channel_hop_proximities[~present_channel_mask] == 0).all()


def _assert_ppr_scores_match_reference(
    ntype_to_sampler_ppr: dict[str, dict[int, float]],
    reference_ppr: dict[str, dict[int, float]],
    seed_id: int,
) -> None:
    """Assert sampler PPR scores match reference scores per node type.

    Checks that top-k node sets are identical and that per-node scores
    are within atol=2e-6.  The forward push error per node is bounded by
    the per-node requeue threshold alpha * eps * degree; for max degree 3,
    alpha=0.5, eps=1e-6 the per-node threshold is ~1.5e-6.  Tolerance is
    set to 2e-6 to provide a small margin above this bound.

    Args:
        ntype_to_sampler_ppr: Sampler output from :func:`_extract_hetero_ppr_scores`.
        reference_ppr: Reference output from :func:`_reference_ppr_hetero`.
        seed_id: Global seed node ID (for error messages).
    """
    for ntype_str in reference_ppr:
        assert set(ntype_to_sampler_ppr[ntype_str].keys()) == set(
            reference_ppr[ntype_str].keys()
        ), (
            f"{seed_id}, type {ntype_str}: top-k node sets differ.\n"
            f"  Sampler:   {sorted(ntype_to_sampler_ppr[ntype_str].keys())}\n"
            f"  Reference: {sorted(reference_ppr[ntype_str].keys())}"
        )

        for node_id in reference_ppr[ntype_str]:
            ref_score = reference_ppr[ntype_str][node_id]
            sam_score = ntype_to_sampler_ppr[ntype_str][node_id]
            assert abs(sam_score - ref_score) < 2e-6, (
                f"{seed_id}, type {ntype_str}, node {node_id}: "
                f"sampler={sam_score:.8f} vs reference={ref_score:.8f}"
            )


def _assert_typed_ppr_edge_attrs(
    datum: HeteroData,
    seed_type: str,
    node_types: list[str],
    num_channels: int,
) -> None:
    """Assert typed PPR edge attributes have the expected channel layout."""
    expected_width = 2 + (3 * num_channels)
    for node_type in node_types:
        ppr_edge_type = (seed_type, "ppr", node_type)
        assert ppr_edge_type in datum.edge_types, (
            f"Missing PPR edge type {ppr_edge_type} on HeteroData"
        )
        ppr_edge_index = datum[ppr_edge_type].edge_index
        ppr_edge_attr = datum[ppr_edge_type].edge_attr

        assert ppr_edge_index.dim() == 2 and ppr_edge_index.size(0) == 2, (
            f"Expected [2, X] edge_index, got shape {list(ppr_edge_index.shape)}"
        )
        assert ppr_edge_attr.dim() == 2, (
            f"Expected 2D typed PPR edge_attr, got {ppr_edge_attr.dim()}D"
        )
        assert ppr_edge_attr.size(0) == ppr_edge_index.size(1)
        assert ppr_edge_attr.size(1) == expected_width, (
            f"Expected typed PPR edge_attr width {expected_width}, "
            f"got {ppr_edge_attr.size(1)}"
        )

        if ppr_edge_attr.size(0) == 0:
            continue

        best_scores = ppr_edge_attr[:, 0]
        hop_proximities = ppr_edge_attr[:, 1]
        channel_features = ppr_edge_attr[:, 2:].reshape(
            ppr_edge_attr.size(0), num_channels, 3
        )
        channel_scores = channel_features[:, :, 0]
        channel_hop_proximities = channel_features[:, :, 1]
        channel_presence = channel_features[:, :, 2]
        assert (best_scores >= 0).all()
        assert (best_scores <= 1).all()
        _assert_valid_hop_proximities(hop_proximities)
        assert (channel_scores >= 0).all()
        assert (channel_scores <= 1).all()
        _assert_typed_channel_hop_proximities(channel_hop_proximities, channel_presence)
        if best_scores.size(0) > 1:
            assert (best_scores[:-1] >= best_scores[1:]).all()

        assert ((channel_presence == 0) | (channel_presence == 1)).all()


# ---------------------------------------------------------------------------
# Spawned process functions
# ---------------------------------------------------------------------------
def _run_ppr_loader_correctness_check(
    _: int,
    alpha: float,
    max_ppr_nodes: int,
    edge_dir: Literal["in", "out"],
) -> None:
    """Iterate homogeneous PPR loader and verify each batch against NetworkX PPR."""
    create_test_process_group()

    dataset = create_homogeneous_dataset(edge_index=_TEST_EDGE_INDEX, edge_dir=edge_dir)

    # This dataset has no edge features, so with_edge is derived as False. PPR
    # outputs come from metadata (ppr_edge_index/ppr_edge_attr), not sampled edge
    # ids, so with_edge must not affect the PPR results verified below.
    assert dataset.edge_features is None, (
        "PPR fixture must be edge-feature-free so with_edge is False."
    )

    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[],  # Unused by PPR sampler; required by interface
        sampler_options=PPRSamplerOptions(
            alpha=alpha,
            eps=_TEST_EPS,
            max_ppr_nodes=max_ppr_nodes,
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )

    reference_graph = _build_reference_graph(edge_dir)

    batches_checked = 0
    for datum in loader:
        assert isinstance(datum, Data)

        # PPR sampling does not count per-hop neighbors, so num_sampled_edges
        # should be absent or empty on all PPR output batches.
        assert (
            not hasattr(datum, "num_sampled_edges") or len(datum.num_sampled_edges) == 0
        ), (
            f"Expected empty num_sampled_edges for PPR output, got {datum.num_sampled_edges}"
        )

        assert hasattr(datum, "edge_index"), "Missing edge_index on Data"
        assert hasattr(datum, "edge_attr"), "Missing edge_attr on Data"

        ppr_edge_index = datum.edge_index
        ppr_edge_attr = datum.edge_attr

        assert ppr_edge_index.dim() == 2 and ppr_edge_index.size(0) == 2, (
            f"Expected [2, X] edge_index, got shape {list(ppr_edge_index.shape)}"
        )
        assert ppr_edge_attr.dim() == 2, (
            f"Expected 2D PPR edge_attr, got {ppr_edge_attr.dim()}D"
        )
        assert ppr_edge_attr.size(1) == 2
        assert ppr_edge_index.size(1) == ppr_edge_attr.size(0), (
            f"Edge count mismatch: {ppr_edge_index.size(1)} vs {ppr_edge_attr.size(0)}"
        )
        ppr_weights = ppr_edge_attr[:, 0]
        ppr_hop_proximities = ppr_edge_attr[:, 1]
        assert (ppr_weights > 0).all(), "PPR weights must be positive"
        _assert_valid_hop_proximities(ppr_hop_proximities)
        assert (ppr_edge_index[0] == 0).all(), (
            "All src indices must be 0 for batch_size=1"
        )

        # Map local indices to global IDs
        global_node_ids = datum.node
        seed_global_id = datum.batch[0].item()

        sampler_ppr: dict[int, float] = {}
        for j in range(ppr_edge_index.size(1)):
            local_dst = ppr_edge_index[1, j].item()
            global_dst = global_node_ids[local_dst].item()
            sampler_ppr[global_dst] = ppr_weights[j].item()

        # Compute reference PPR
        reference_ppr = _reference_ppr(
            graph=reference_graph,
            seed=seed_global_id,
            alpha=alpha,
            max_ppr_nodes=max_ppr_nodes,
        )

        # Verify same top-k node set
        assert set(sampler_ppr.keys()) == set(reference_ppr.keys()), (
            f"Seed {seed_global_id}: top-k node sets differ.\n"
            f"  Sampler:   {sorted(sampler_ppr.keys())}\n"
            f"  Reference: {sorted(reference_ppr.keys())}"
        )

        # Forward push is an approximation; with eps=1e-6 the per-node
        # requeue threshold is alpha * eps * degree.  For this test graph
        # (max degree 3, alpha=0.5, eps=1e-6) the per-node threshold is
        # ~1.5e-6.  Tolerance is set to 2e-6 to provide a small margin
        # above this bound.
        for node_id in reference_ppr:
            ref_score = reference_ppr[node_id]
            sam_score = sampler_ppr[node_id]
            assert abs(sam_score - ref_score) < 2e-6, (
                f"Seed {seed_global_id}, node {node_id}: "
                f"sampler={sam_score:.8f} vs reference={ref_score:.8f}"
            )

        batches_checked += 1

    assert batches_checked == _NUM_TEST_NODES, (
        f"Expected {_NUM_TEST_NODES} batches, got {batches_checked}"
    )
    shutdown_rpc()


def _run_ppr_hetero_loader_correctness_check(
    _: int,
    alpha: float,
    max_ppr_nodes: int,
    edge_dir: Literal["in", "out"],
) -> None:
    """Iterate heterogeneous PPR loader and verify each batch against NetworkX PPR."""
    create_test_process_group()

    dataset = create_heterogeneous_dataset(
        edge_indices=_TEST_HETERO_EDGE_INDICES, edge_dir=edge_dir
    )

    node_ids = dataset.node_ids
    assert isinstance(node_ids, dict)

    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(USER, node_ids[USER]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[],  # Unused by PPR sampler; required by interface
        sampler_options=PPRSamplerOptions(
            alpha=alpha,
            eps=_TEST_EPS,
            max_ppr_nodes=max_ppr_nodes,
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )

    reference_graph = _build_hetero_reference_graph(edge_dir)

    batches_checked = 0
    for datum in loader:
        assert isinstance(datum, HeteroData)

        # PPR sampling does not count per-hop neighbors, so num_sampled_edges
        # should be absent or empty on all PPR output batches.
        assert (
            not hasattr(datum, "num_sampled_edges") or len(datum.num_sampled_edges) == 0
        ), (
            f"Expected empty num_sampled_edges for PPR output, got {datum.num_sampled_edges}"
        )

        seed_global_id = datum[USER].batch[0].item()

        ntype_to_sampler_ppr = _extract_hetero_ppr_scores(
            datum, str(USER), [USER, STORY]
        )

        reference_ppr = _reference_ppr_hetero(
            graph=reference_graph,
            seed=seed_global_id,
            seed_type=str(USER),
            alpha=alpha,
            max_ppr_nodes=max_ppr_nodes,
        )

        _assert_ppr_scores_match_reference(
            ntype_to_sampler_ppr, reference_ppr, seed_global_id
        )

        for edge_type in datum.edge_types:
            assert edge_type[1] == "ppr", (
                f"Non-PPR edge type {edge_type} found in PPR sampler output"
            )

        batches_checked += 1

    assert batches_checked == _NUM_TEST_USERS, (
        f"Expected {_NUM_TEST_USERS} batches, got {batches_checked}"
    )
    shutdown_rpc()


def _run_typed_ppr_loader_shape_check(_: int) -> None:
    """Verify typed PPR runs through DistNeighborLoader into HeteroData."""
    create_test_process_group()

    dataset = create_heterogeneous_dataset(
        edge_indices=_TEST_HETERO_EDGE_INDICES,
        edge_dir="out",
    )
    node_ids = dataset.node_ids
    assert isinstance(node_ids, dict)

    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(USER, node_ids[USER]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[],
        sampler_options=PPRSamplerOptions(
            alpha=_TEST_ALPHA,
            eps=_TEST_EPS,
            max_ppr_nodes=_TEST_MAX_PPR_NODES,
            typed_channel_ratios={
                USER_TO_STORY: 0.5,
                STORY_TO_USER: 0.5,
            },
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )

    batches_checked = 0
    for datum in loader:
        assert isinstance(datum, HeteroData)
        _assert_typed_ppr_edge_attrs(
            datum=datum,
            seed_type=str(USER),
            node_types=[USER, STORY],
            num_channels=2,
        )
        for edge_type in datum.edge_types:
            assert edge_type[1] == "ppr", (
                f"Non-PPR edge type {edge_type} found in PPR sampler output"
            )
        batches_checked += 1
    assert batches_checked == _NUM_TEST_USERS, (
        f"Expected {_NUM_TEST_USERS} batches, got {batches_checked}"
    )

    empty_story_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(USER, node_ids[USER][:1]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        num_neighbors=[],
        sampler_options=PPRSamplerOptions(
            alpha=_TEST_ALPHA,
            eps=_TEST_EPS,
            max_ppr_nodes=_TEST_MAX_PPR_NODES,
            typed_channel_ratios={
                STORY_TO_USER: 0.5,
                (STORY_TO_USER,): 0.5,
            },
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )
    empty_story_datum = next(iter(empty_story_loader))
    assert isinstance(empty_story_datum, HeteroData)
    _assert_typed_ppr_edge_attrs(
        datum=empty_story_datum,
        seed_type=str(USER),
        node_types=[USER, STORY],
        num_channels=2,
    )
    empty_story_edge_attr = empty_story_datum[(str(USER), "ppr", STORY)].edge_attr
    assert tuple(empty_story_edge_attr.shape) == (0, 8)

    shutdown_rpc()


def _run_ppr_ablp_loader_correctness_check(
    _: int,
    alpha: float,
    max_ppr_nodes: int,
    edge_dir: Literal["in", "out"],
) -> None:
    """Iterate ABLP PPR loader and verify anchor-seed PPR against NetworkX reference.

    Checks both anchor (USER) seed PPR scores for correctness against NetworkX,
    and verifies that supervision (STORY) seed PPR metadata is present with
    valid shapes.  Also confirms that ABLP-specific output (y_positive) is
    produced alongside PPR metadata.

    The ABLP dataset is created inside this spawned process because the
    splitter requires torch.distributed to be initialized.
    """
    create_test_process_group()

    dataset = create_heterogeneous_dataset_for_ablp(
        positive_labels={0: [0, 1], 1: [1, 2], 2: [0, 2]},
        train_node_ids=[0, 1],
        val_node_ids=[2],
        test_node_ids=[],
        edge_indices=_TEST_HETERO_EDGE_INDICES,
        edge_dir=edge_dir,
    )

    train_node_ids = dataset.train_node_ids
    if train_node_ids is None or isinstance(train_node_ids, torch.Tensor):
        raise TypeError(
            f"Expected train_node_ids to be a dictionary, got {type(train_node_ids)}"
        )

    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[],  # Unused by PPR sampler; required by interface
        input_nodes=(USER, train_node_ids[USER]),
        supervision_edge_type=USER_TO_STORY,
        sampler_options=PPRSamplerOptions(
            alpha=alpha,
            eps=_TEST_EPS,
            max_ppr_nodes=max_ppr_nodes,
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )

    reference_graph = _build_hetero_reference_graph(edge_dir=edge_dir)

    batches_checked = 0
    for datum in loader:
        assert isinstance(datum, HeteroData)

        # PPR sampling does not count per-hop neighbors, so num_sampled_edges
        # should be absent or empty on all PPR output batches.
        assert (
            not hasattr(datum, "num_sampled_edges") or len(datum.num_sampled_edges) == 0
        ), (
            f"Expected empty num_sampled_edges for PPR output, got {datum.num_sampled_edges}"
        )

        # ABLP should produce positive labels alongside PPR metadata
        assert hasattr(datum, "y_positive"), "Missing y_positive on HeteroData"

        seed_global_id = datum[USER].batch[0].item()

        # --- Verify anchor (USER) seed PPR correctness against NetworkX ---
        ntype_to_sampler_ppr = _extract_hetero_ppr_scores(
            datum, str(USER), [USER, STORY]
        )

        reference_ppr = _reference_ppr_hetero(
            graph=reference_graph,
            seed=seed_global_id,
            seed_type=str(USER),
            alpha=alpha,
            max_ppr_nodes=max_ppr_nodes,
        )

        _assert_ppr_scores_match_reference(
            ntype_to_sampler_ppr, reference_ppr, seed_global_id
        )

        # --- Verify supervision (STORY) seed PPR metadata ---
        # ABLP adds supervision nodes as additional seeds, producing PPR metadata
        # keyed by the STORY seed type.  We only check shapes here (not correctness
        # against NetworkX) because the supervision seeds vary per batch depending
        # on the label edges, making deterministic reference computation complex.
        for ntype in [USER, STORY]:
            ppr_edge_type = (STORY, "ppr", ntype)
            assert ppr_edge_type in datum.edge_types, (
                f"Missing PPR edge type {ppr_edge_type} on HeteroData"
            )

            ppr_edge_index = datum[ppr_edge_type].edge_index
            ppr_edge_attr = datum[ppr_edge_type].edge_attr

            assert ppr_edge_index.dim() == 2 and ppr_edge_index.size(0) == 2
            assert ppr_edge_attr.dim() == 2
            assert ppr_edge_attr.size(1) == 2
            assert ppr_edge_index.size(1) == ppr_edge_attr.size(0)
            if ppr_edge_attr.numel() > 0:
                ppr_weights = ppr_edge_attr[:, 0]
                ppr_hop_proximities = ppr_edge_attr[:, 1]
                assert (ppr_weights > 0).all()
                _assert_valid_hop_proximities(ppr_hop_proximities)
            assert (ppr_edge_index[1] >= 0).all()
            assert (ppr_edge_index[1] < datum[ntype].node.size(0)).all()

        for edge_type in datum.edge_types:
            assert edge_type[1] == "ppr", (
                f"Non-PPR edge type {edge_type} found in PPR sampler output"
            )

        batches_checked += 1

    assert batches_checked > 0, "Expected at least one ABLP batch"
    shutdown_rpc()


def _run_ppr_labeled_homogeneous_ablp_loader_check(_: int) -> None:
    """Verify PPR works for labeled homogeneous DistABLPLoader inputs."""
    create_test_process_group()

    dataset = create_heterogeneous_dataset_for_ablp(
        positive_labels={0: [1], 1: [2], 2: [0]},
        negative_labels={0: [2], 1: [0], 2: [1]},
        train_node_ids=[0, 1],
        val_node_ids=[2],
        test_node_ids=[],
        edge_indices={DEFAULT_HOMOGENEOUS_EDGE_TYPE: _TEST_EDGE_INDEX},
        src_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
        dst_node_type=DEFAULT_HOMOGENEOUS_NODE_TYPE,
        supervision_edge_type=DEFAULT_HOMOGENEOUS_EDGE_TYPE,
        edge_dir="out",
    )

    train_node_ids = dataset.train_node_ids
    assert isinstance(train_node_ids, dict)

    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[],
        input_nodes=train_node_ids[DEFAULT_HOMOGENEOUS_NODE_TYPE],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        sampler_options=PPRSamplerOptions(
            alpha=_TEST_ALPHA,
            eps=_TEST_EPS,
            max_ppr_nodes=_TEST_MAX_PPR_NODES,
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )

    datum = next(iter(loader))
    assert isinstance(datum, Data)
    assert hasattr(datum, "edge_index"), "Missing PPR edge_index on Data"
    assert hasattr(datum, "edge_attr"), "Missing PPR edge_attr on Data"
    assert hasattr(datum, "y_positive"), "Missing y_positive on Data"
    assert hasattr(datum, "y_negative"), "Missing y_negative on Data"
    assert datum.edge_index.dim() == 2
    assert datum.edge_index.size(0) == 2
    assert datum.edge_attr.dim() == 2
    assert datum.edge_attr.size(1) == 2
    assert datum.edge_index.size(1) == datum.edge_attr.size(0)

    shutdown_rpc()


# ---------------------------------------------------------------------------
# Bug regression runners
# ---------------------------------------------------------------------------
def _run_ppr_destination_only_node_type(_: int) -> None:
    """Verify PPR output includes destination-only node types.

    Uses a one-directional USER->STORY graph (no STORY->USER edges) so that
    STORY is a destination-only type with no outgoing edges.  Asserts that
    STORY nodes appear in the PPR output despite having no outgoing edges and
    being absent from _node_type_to_edge_types.
    """
    create_test_process_group()

    # One-directional graph: USER->STORY only, so STORY is destination-only.
    # user 0 -> story 0, user 0 -> story 1, user 1 -> story 0
    edge_index = torch.tensor([[0, 0, 1], [0, 1, 0]])
    dataset = create_heterogeneous_dataset(
        edge_indices={USER_TO_STORY: edge_index},
        edge_dir="out",
    )

    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[],
        input_nodes=(USER, torch.tensor([0])),
        sampler_options=PPRSamplerOptions(
            alpha=_TEST_ALPHA,
            eps=_TEST_EPS,
            max_ppr_nodes=_TEST_MAX_PPR_NODES,
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )

    datum = next(iter(loader))
    assert isinstance(datum, HeteroData)

    # STORY must appear in the PPR output even though it has no outgoing edges.
    ppr_edge_type = (USER, "ppr", STORY)
    assert ppr_edge_type in datum.edge_types, (
        f"Missing PPR edge type {ppr_edge_type} — destination-only STORY was dropped"
    )
    assert datum[ppr_edge_type].edge_index.shape[1] > 0, (
        "Expected at least one PPR edge to STORY"
    )
    assert (datum[ppr_edge_type].edge_attr[:, 0] > 0).all()

    shutdown_rpc()


def _run_ppr_loader_preserves_original_edges_for_selected_nodes(_: int) -> None:
    """Verify opt-in hetero PPR output includes original edges over selected nodes."""
    create_test_process_group()

    edge_index = torch.tensor([[0, 0, 0], [0, 1, 2]])
    dataset = create_heterogeneous_dataset(
        edge_indices={USER_TO_STORY: edge_index},
        edge_dir="out",
    )
    node_ids = dataset.node_ids
    assert isinstance(node_ids, dict)

    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(USER, torch.tensor([0])),
        num_neighbors=[],
        sampler_options=PPRSamplerOptions(
            alpha=_TEST_ALPHA,
            eps=_TEST_EPS,
            max_ppr_nodes=2,
            include_sampled_edges=True,
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )

    datum = next(iter(loader))
    assert isinstance(datum, HeteroData)

    ppr_edge_type = (USER, "ppr", STORY)
    original_output_edge_type = reverse_edge_type(USER_TO_STORY)
    assert ppr_edge_type in datum.edge_types
    # Original edges are emitted through GLT's regular sampled-edge path, so
    # they follow the same edge_dir="out" orientation convention as k-hop.
    assert original_output_edge_type in datum.edge_types

    ppr_story_edge_index = datum[ppr_edge_type].edge_index
    original_edge_index = datum[original_output_edge_type].edge_index
    story_node_ids = datum[STORY].node

    ppr_story_ids = {
        int(story_node_ids[local_dst].item())
        for local_dst in ppr_story_edge_index[1].tolist()
    }
    original_story_ids = {
        int(story_node_ids[local_src].item())
        for local_src in original_edge_index[0].tolist()
    }

    assert original_story_ids
    assert original_story_ids <= ppr_story_ids
    assert len(original_story_ids) < edge_index.size(1)
    assert [int(count) for count in datum.num_sampled_nodes[USER]] == [1, 0]
    assert [int(count) for count in datum.num_sampled_nodes[STORY]] == [
        0,
        story_node_ids.size(0),
    ]
    assert [
        int(count) for count in datum.num_sampled_edges[original_output_edge_type]
    ] == [original_edge_index.size(1)]

    shutdown_rpc()


def _run_ppr_ablp_label_edges_do_not_affect_anchor_ppr(_: int) -> None:
    """Verify that ABLP label edges are excluded from anchor-seed PPR walks.

    Uses a graph where user 0's positive label target (story 1) is NOT
    reachable via message-passing edges.  Asserts that story 1 does not appear
    in user 0's PPR neighborhood, confirming that label edge types are not
    traversed during PPR walks.
    """
    create_test_process_group()

    # Message-passing graph: user 0 <-> story 0, user 1 <-> story 2
    # Positive label edges: user 0 -> story 1 (NOT in message graph)
    dataset = create_heterogeneous_dataset_for_ablp(
        positive_labels={0: [1], 1: [2]},
        train_node_ids=[0],
        val_node_ids=[1],
        test_node_ids=[],
        edge_indices={
            USER_TO_STORY: torch.tensor([[0, 1], [0, 2]]),
            STORY_TO_USER: torch.tensor([[0, 2], [0, 1]]),
        },
        edge_dir="out",
    )

    train_node_ids = dataset.train_node_ids
    assert isinstance(train_node_ids, dict)

    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[],
        input_nodes=(USER, train_node_ids[USER]),  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        supervision_edge_type=USER_TO_STORY,
        sampler_options=PPRSamplerOptions(
            alpha=_TEST_ALPHA,
            eps=_TEST_EPS,
            max_ppr_nodes=_TEST_MAX_PPR_NODES,
        ),
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )

    datum = next(iter(loader))
    assert isinstance(datum, HeteroData)

    sampler_ppr = _extract_hetero_ppr_scores(datum, str(USER), [USER, STORY])

    # story 1 is reachable only via the positive label edge from user 0.
    # It must not appear in user 0's PPR neighborhood.
    assert 1 not in sampler_ppr[str(STORY)], (
        "story 1 appeared in user 0's PPR output — label edge was incorrectly traversed"
    )

    # story 0 is reachable via message-passing and must be present.
    assert 0 in sampler_ppr[str(STORY)], (
        "story 0 missing from user 0's PPR output — message-passing edge was not traversed"
    )

    shutdown_rpc()


# ---------------------------------------------------------------------------
# Global top-K cap: three node types so a seed emits rows of several types.
# ---------------------------------------------------------------------------
_TAG = NodeType("tag")
_STORY_TO_TAG = EdgeType(STORY, Relation("to"), _TAG)
_TAG_TO_STORY = EdgeType(_TAG, Relation("to"), STORY)
_CAP_NUM_USERS = 8
_CAP_NUM_STORIES = 8
_CAP_NUM_TAGS = 6
_CAP_MAX_PPR_NODES = 2
_CAP_NODE_TYPES = [USER, STORY, _TAG]
_CAP_FEATURE_DIM = 3
_CAP_NODE_TYPE_OFFSET = {USER: 0, STORY: 100, _TAG: 200}


def _build_cap_test_dataset():
    """Builds a seeded random user-story-tag graph with node features that encode node identity."""
    generator = torch.Generator().manual_seed(7)

    def random_bipartite_edges(
        num_sources: int, num_destinations: int, out_degree: int
    ) -> torch.Tensor:
        sources = torch.arange(num_sources).repeat_interleave(out_degree)
        destinations = torch.cat(
            [
                torch.randperm(num_destinations, generator=generator)[:out_degree]
                for _ in range(num_sources)
            ]
        )
        return torch.stack([sources, destinations])

    user_to_story = random_bipartite_edges(_CAP_NUM_USERS, _CAP_NUM_STORIES, 3)
    story_to_tag = random_bipartite_edges(_CAP_NUM_STORIES, _CAP_NUM_TAGS, 2)
    edge_indices = {
        USER_TO_STORY: user_to_story,
        STORY_TO_USER: user_to_story.flip(0),
        _STORY_TO_TAG: story_to_tag,
        _TAG_TO_STORY: story_to_tag.flip(0),
    }
    num_nodes = {USER: _CAP_NUM_USERS, STORY: _CAP_NUM_STORIES, _TAG: _CAP_NUM_TAGS}
    node_features = {
        node_type: (
            _CAP_NODE_TYPE_OFFSET[node_type] + torch.arange(count, dtype=torch.float32)
        ).unsqueeze(1)
        * torch.arange(1, _CAP_FEATURE_DIM + 1, dtype=torch.float32)
        for node_type, count in num_nodes.items()
    }
    return create_heterogeneous_dataset(
        edge_indices=edge_indices, node_features=node_features, edge_dir="out"
    )


def _read_consumer_view(
    datum: HeteroData,
    seed_global_id: int,
    num_tokens: int,
    drop_seed: bool,
) -> tuple[dict, int]:
    """Reads a batch the way the graph transformer does: the top rows per seed by score.

    Rows of every ``(USER, "ppr", node_type)`` edge type are ranked by ``float32`` score, with ties broken
    by node type and global id. The consumer keeps the first ``num_tokens`` of them, after dropping the
    seed itself when ``drop_seed`` is set.

    Returns:
        A tuple of:

        - the consumer view: the ``tokens`` as an ordered list of ``(node_type, global_id)``, the PPR
          ``attributes`` and node ``features`` of each token, and the original ``edges`` among tokens
          and the seed.
        - the total number of PPR rows in the batch.
    """
    rows: list[tuple[float, str, int, tuple[float, ...]]] = []
    for node_type in _CAP_NODE_TYPES:
        ppr_edge_type = (USER, "ppr", node_type)
        if ppr_edge_type not in datum.edge_types:
            continue
        edge_index = datum[ppr_edge_type].edge_index
        edge_attr = datum[ppr_edge_type].edge_attr
        if edge_index.size(1) == 0:
            # A node type without any sampled node has no ``node`` attribute at all.
            continue
        global_ids = datum[node_type].node[edge_index[1]]
        for global_id, attributes in zip(global_ids.tolist(), edge_attr):
            rows.append(
                (
                    float(attributes[0].float()),
                    str(node_type),
                    global_id,
                    tuple(attributes.tolist()),
                )
            )
    num_rows = len(rows)
    if drop_seed:
        rows = [row for row in rows if (row[1], row[2]) != (str(USER), seed_global_id)]
    rows.sort(key=lambda row: (-row[0], row[1], row[2]))
    rows = rows[:num_tokens]

    tokens = [(row[1], row[2]) for row in rows]
    attributes_by_token = {(row[1], row[2]): row[3] for row in rows}
    node_ids = {
        node_type: datum[node_type].node
        for node_type in datum.node_types
        if "node" in datum[node_type]
    }
    features_by_token = {}
    for node_type_name, global_id in tokens:
        local_index = node_ids[node_type_name].tolist().index(global_id)
        features_by_token[(node_type_name, global_id)] = tuple(
            datum[node_type_name].x[local_index].tolist()
        )
    kept_nodes = set(tokens) | {(str(USER), seed_global_id)}
    edges = set()
    for edge_type in datum.edge_types:
        if edge_type[1] == "ppr" or "edge_index" not in datum[edge_type]:
            continue
        source_type, _, destination_type = edge_type
        source_ids = node_ids[source_type][datum[edge_type].edge_index[0]].tolist()
        destination_ids = node_ids[destination_type][
            datum[edge_type].edge_index[1]
        ].tolist()
        for source_id, destination_id in zip(source_ids, destination_ids):
            if (source_type, source_id) in kept_nodes and (
                destination_type,
                destination_id,
            ) in kept_nodes:
                edges.add((edge_type, source_id, destination_id))
    consumer_view = {
        "tokens": tokens,
        "attributes": attributes_by_token,
        "features": features_by_token,
        "edges": edges,
    }
    return consumer_view, num_rows


def _collect_consumer_views(
    dataset,
    sampler_options: PPRSamplerOptions,
    num_tokens: int,
    drop_seed: bool,
) -> dict[int, tuple[dict, int]]:
    """Samples every user as its own batch and returns each seed's consumer view and PPR row count."""
    loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(USER, torch.arange(_CAP_NUM_USERS)),
        num_neighbors=[],
        sampler_options=sampler_options,
        pin_memory_device=torch.device("cpu"),
        batch_size=1,
    )
    views: dict[int, tuple[dict, int]] = {}
    for datum in loader:
        assert isinstance(datum, HeteroData)
        seed_global_id = int(datum[USER].batch[0])
        views[seed_global_id] = _read_consumer_view(
            datum, seed_global_id, num_tokens, drop_seed
        )
    return views


def _run_ppr_global_top_k_cap_consumer_check(
    _: int,
    typed_channel_ratios: Optional[dict],
    consumer_tokens: int,
    drop_seed: bool,
    expect_identical: bool,
) -> None:
    """Compares the capped and uncapped samplers through a top-``consumer_tokens`` consumer.

    When ``expect_identical`` is set, every seed's tokens, PPR attributes, node features and original
    edges must match. Otherwise the consumer reads past the cap and at least one seed must differ.
    """
    create_test_process_group()
    dataset = _build_cap_test_dataset()

    def sampler_options(global_top_k_cap: bool) -> PPRSamplerOptions:
        return PPRSamplerOptions(
            alpha=_TEST_ALPHA,
            eps=_TEST_EPS,
            max_ppr_nodes=_CAP_MAX_PPR_NODES,
            typed_channel_ratios=typed_channel_ratios,
            include_sampled_edges=True,
            global_top_k_cap=global_top_k_cap,
        )

    uncapped = _collect_consumer_views(
        dataset, sampler_options(False), consumer_tokens, drop_seed
    )
    capped = _collect_consumer_views(
        dataset, sampler_options(True), consumer_tokens, drop_seed
    )

    assert uncapped.keys() == capped.keys() == set(range(_CAP_NUM_USERS))
    # The cap only drops rows, and never more than the ``max_ppr_nodes + 1`` rows it promises to keep.
    for seed_global_id in uncapped:
        num_uncapped_rows = uncapped[seed_global_id][1]
        num_capped_rows = capped[seed_global_id][1]
        assert num_capped_rows <= num_uncapped_rows
        assert num_capped_rows >= min(num_uncapped_rows, _CAP_MAX_PPR_NODES + 1)
    # Guard against a vacuous comparison: the cap must actually drop rows here.
    assert any(capped[seed][1] < uncapped[seed][1] for seed in uncapped)

    mismatched_seeds = [
        seed for seed in uncapped if uncapped[seed][0] != capped[seed][0]
    ]
    if expect_identical:
        assert not mismatched_seeds, (
            f"Capped output differs for seeds {mismatched_seeds}"
        )
    else:
        assert mismatched_seeds, "Expected a consumer reading past the cap to differ"

    shutdown_rpc()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class DistPPRSamplerTest(TestCase):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    @parameterized.expand(
        [
            param("edge_dir_in", edge_dir="in"),
            param("edge_dir_out", edge_dir="out"),
        ]
    )
    def test_ppr_sampler_correctness_homogeneous(self, _, edge_dir: str) -> None:
        """Verify PPR scores match NetworkX pagerank on a small homogeneous graph."""
        mp.spawn(
            fn=_run_ppr_loader_correctness_check,
            args=(_TEST_ALPHA, _TEST_MAX_PPR_NODES, edge_dir),
        )

    @parameterized.expand(
        [
            param("edge_dir_in", edge_dir="in"),
            param("edge_dir_out", edge_dir="out"),
        ]
    )
    def test_ppr_sampler_correctness_heterogeneous(self, _, edge_dir: str) -> None:
        """Verify PPR scores match NetworkX pagerank on a heterogeneous bipartite graph."""
        mp.spawn(
            fn=_run_ppr_hetero_loader_correctness_check,
            args=(_TEST_ALPHA, _TEST_MAX_PPR_NODES, edge_dir),
        )

    @parameterized.expand(
        [
            param("edge_dir_in", edge_dir="in"),
            param("edge_dir_out", edge_dir="out"),
        ]
    )
    def test_ppr_sampler_ablp_correctness(self, _, edge_dir: str) -> None:
        """Verify PPR scores through DistABLPLoader on a heterogeneous graph."""
        mp.spawn(
            fn=_run_ppr_ablp_loader_correctness_check,
            args=(_TEST_ALPHA, _TEST_MAX_PPR_NODES, edge_dir),
        )

    def test_ppr_sampler_destination_only_node_type(self) -> None:
        """Verify PPR output includes destination-only node types."""
        mp.spawn(fn=_run_ppr_destination_only_node_type, args=())

    def test_ppr_sampler_can_preserve_original_edges_for_selected_nodes(self) -> None:
        """Verify PPR can retain original hetero edges over selected nodes."""
        mp.spawn(
            fn=_run_ppr_loader_preserves_original_edges_for_selected_nodes,
            args=(),
        )

    def test_typed_ppr_sampler_loader_outputs_channel_attrs(self) -> None:
        """Verify typed PPR runs end-to-end through the loader."""
        mp.spawn(fn=_run_typed_ppr_loader_shape_check, args=())

    def test_ppr_sampler_ablp_ignores_label_edges_for_anchor_ppr(self) -> None:
        """Verify ABLP label edges are excluded from anchor-seed PPR walks."""
        mp.spawn(fn=_run_ppr_ablp_label_edges_do_not_affect_anchor_ppr, args=())

    def test_ppr_sampler_homogeneous_ablp(self) -> None:
        """Verify PPR handles homogeneous ABLP seed dictionaries."""
        mp.spawn(fn=_run_ppr_labeled_homogeneous_ablp_loader_check, args=())

    @parameterized.expand(
        [
            param("untyped_ppr", None, _CAP_MAX_PPR_NODES, False),
            param("untyped_ppr_with_the_seed_dropped", None, _CAP_MAX_PPR_NODES, True),
            param(
                "typed_ppr_channels",
                {
                    (USER_TO_STORY, STORY_TO_USER): 0.5,
                    (USER_TO_STORY, STORY_TO_USER, _STORY_TO_TAG, _TAG_TO_STORY): 0.5,
                },
                _CAP_MAX_PPR_NODES,
                False,
            ),
        ]
    )
    def test_a_consumer_of_the_top_max_ppr_nodes_rows_sees_the_same_output_with_the_cap(
        self,
        _,
        typed_channel_ratios: Optional[dict],
        consumer_tokens: int,
        drop_seed: bool,
    ) -> None:
        """A consumer reading the top ``max_ppr_nodes`` rows per seed sees the same output with the cap."""
        mp.spawn(
            fn=_run_ppr_global_top_k_cap_consumer_check,
            args=(typed_channel_ratios, consumer_tokens, drop_seed, True),
        )

    def test_a_consumer_reading_past_max_ppr_nodes_rows_sees_different_output_with_the_cap(
        self,
    ) -> None:
        """Documents why the cap is unsafe for a consumer reading more than ``max_ppr_nodes`` rows."""
        mp.spawn(
            fn=_run_ppr_global_top_k_cap_consumer_check,
            args=(None, 3 * _CAP_MAX_PPR_NODES, False, False),
        )


def _by_node_type(tensors) -> dict[NodeType, torch.Tensor]:
    """Narrows a ``PPRResult`` field to its node-type-keyed form."""
    assert isinstance(tensors, dict)
    return tensors


class PerSeedTopKCapTest(TestCase):
    """Capping keeps each seed's top rows across node types, and every row tied at the cutoff."""

    @parameterized.expand(
        [
            param(
                "a_seed_with_fewer_rows_than_top_k_keeps_them_all",
                [0.4, 0.9],
                [2],
                3,
                [True, True],
            ),
            param(
                "a_seed_with_exactly_top_k_rows_keeps_them_all",
                [0.4, 0.9, 0.1],
                [3],
                3,
                [True, True, True],
            ),
            param(
                "the_lowest_rows_of_a_longer_seed_are_dropped",
                [0.1, 0.9, 0.5, 0.7],
                [4],
                2,
                [False, True, False, True],
            ),
            param(
                "every_row_tied_at_the_cutoff_is_kept",
                [0.9, 0.5, 0.5, 0.5, 0.1],
                [5],
                2,
                [True, True, True, True, False],
            ),
            param(
                "a_tie_inside_top_k_does_not_widen_the_cap",
                [0.5, 0.5, 0.1],
                [3],
                2,
                [True, True, False],
            ),
            param("a_top_k_of_zero_keeps_nothing", [0.5, 0.9], [2], 0, [False, False]),
        ]
    )
    def test_a_seed_keeps_its_top_k_rows_and_every_row_tied_at_the_cutoff(
        self,
        _,
        scores: list[float],
        valid_counts: list[int],
        top_k: int,
        expected_keep: list[bool],
    ) -> None:
        keep = cap_rows_to_per_seed_top_k(
            {USER: torch.tensor(scores)}, {USER: torch.tensor(valid_counts)}, top_k
        )
        self.assertEqual(keep[USER].tolist(), expected_keep)

    def test_each_seed_is_capped_across_all_of_its_node_types(self) -> None:
        # Seed 0 owns rows {0.2, 0.8} of "user" and {0.9} of "story": its top 2 are 0.9 and 0.8.
        # Seed 1 owns one row and seed 2 owns two rows, so neither is over the cap.
        scores = {
            USER: torch.tensor([0.2, 0.8, 0.6]),
            STORY: torch.tensor([0.9, 0.3, 0.4]),
        }
        valid_counts = {USER: torch.tensor([2, 1, 0]), STORY: torch.tensor([1, 0, 2])}
        keep = cap_rows_to_per_seed_top_k(scores, valid_counts, top_k=2)
        self.assertEqual(keep[USER].tolist(), [False, True, True])
        self.assertEqual(keep[STORY].tolist(), [True, True, True])

    def test_seed_without_rows_does_not_shift_other_seeds(self) -> None:
        scores = {USER: torch.tensor([0.9, 0.1, 0.5, 0.4])}
        valid_counts = {USER: torch.tensor([0, 2, 0, 2])}
        keep = cap_rows_to_per_seed_top_k(scores, valid_counts, top_k=1)
        self.assertEqual(keep[USER].tolist(), [True, False, True, False])

    def test_surviving_rows_keep_their_relative_order(self) -> None:
        node_ids = torch.tensor([10, 11, 12, 13, 14, 15])
        scores = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.7, 0.2])
        keep = cap_rows_to_per_seed_top_k(
            {USER: scores}, {USER: torch.tensor([3, 3])}, top_k=2
        )
        self.assertEqual(node_ids[keep[USER]].tolist(), [11, 12, 13, 14])

    def test_no_node_types_or_no_rows_gives_empty_masks(self) -> None:
        self.assertEqual(cap_rows_to_per_seed_top_k({}, {}, top_k=3), {})
        keep = cap_rows_to_per_seed_top_k(
            {USER: torch.empty(0), STORY: torch.empty(0)},
            {USER: torch.tensor([0, 0]), STORY: torch.tensor([0, 0])},
            top_k=3,
        )
        self.assertEqual(keep[USER].shape, (0,))
        self.assertEqual(keep[STORY].shape, (0,))

    def test_float64_scores_that_tie_at_float32_precision_are_all_kept(self) -> None:
        # float64 scores that differ by less than float32 resolution tie once ranked at float32.
        near_tied = torch.tensor([1.0, 1.0 + 1e-12, 0.5], dtype=torch.float64)
        self.assertGreater(near_tied[1], near_tied[0])
        keep = cap_rows_to_per_seed_top_k(
            {USER: near_tied}, {USER: torch.tensor([3])}, top_k=1
        )
        self.assertEqual(keep[USER].tolist(), [True, True, False])

        # The same scores, far enough apart for float32, are not tied.
        distinct = torch.tensor([1.0, 1.0 + 1e-6, 0.5], dtype=torch.float64)
        keep = cap_rows_to_per_seed_top_k(
            {USER: distinct}, {USER: torch.tensor([3])}, top_k=1
        )
        self.assertEqual(keep[USER].tolist(), [False, True, False])

        # float32 input is ranked as-is.
        keep = cap_rows_to_per_seed_top_k(
            {USER: distinct.float()}, {USER: torch.tensor([3])}, top_k=1
        )
        self.assertEqual(keep[USER].tolist(), [False, True, False])

    def test_capping_a_ppr_result_drops_the_rows_from_every_field(self) -> None:
        ppr_result = PPRResult(
            node_ids={
                USER: torch.tensor([10, 11, 12, 20]),
                STORY: torch.tensor([30, 31]),
            },
            # Column 0 is the score; column 1 is a payload equal to the row's node id that must follow it.
            weights={
                USER: torch.tensor([[0.9, 10], [0.5, 11], [0.4, 12], [0.8, 20]]),
                STORY: torch.tensor([[0.7, 30], [0.6, 31]]),
            },
            valid_counts={USER: torch.tensor([3, 1]), STORY: torch.tensor([1, 1])},
            is_expanded={USER: torch.tensor([1, 0, 1, 1]), STORY: torch.tensor([0, 1])},
            cache_state=None,
        )
        capped = cap_ppr_result_to_top_k(ppr_result, top_k=3)
        # Seed 0 keeps its 0.9, 0.7 and 0.5 rows; seed 1 has two rows and keeps both.
        self.assertEqual(_by_node_type(capped.node_ids)[USER].tolist(), [10, 11, 20])
        self.assertEqual(_by_node_type(capped.node_ids)[STORY].tolist(), [30, 31])
        self.assertEqual(_by_node_type(capped.valid_counts)[USER].tolist(), [2, 1])
        self.assertEqual(_by_node_type(capped.valid_counts)[STORY].tolist(), [1, 1])
        self.assertEqual(_by_node_type(capped.valid_counts)[USER].dtype, torch.int64)
        self.assertEqual(_by_node_type(capped.is_expanded)[USER].tolist(), [1, 0, 1])
        self.assertEqual(_by_node_type(capped.is_expanded)[STORY].tolist(), [0, 1])
        self.assertEqual(
            _by_node_type(capped.weights)[USER][:, 1].tolist(), [10, 11, 20]
        )
        self.assertEqual(_by_node_type(capped.weights)[STORY][:, 1].tolist(), [30, 31])

    def test_capping_a_typed_ppr_result_ranks_on_its_best_score_column(
        self,
    ) -> None:
        # Typed PPR rows carry extra per-channel columns; only column 0 (the best score) ranks them.
        ppr_result = PPRResult(
            node_ids={USER: torch.tensor([1, 2, 3])},
            weights={
                USER: torch.tensor([[0.1, 0.9, 0.9], [0.8, 0.0, 0.0], [0.5, 0.1, 0.1]])
            },
            valid_counts={USER: torch.tensor([3])},
            is_expanded={USER: torch.tensor([1, 1, 1])},
            cache_state=None,
        )
        capped = cap_ppr_result_to_top_k(ppr_result, top_k=2)
        self.assertEqual(_by_node_type(capped.node_ids)[USER].tolist(), [2, 3])
        self.assertEqual(_by_node_type(capped.valid_counts)[USER].tolist(), [2])


if __name__ == "__main__":
    absltest.main()
