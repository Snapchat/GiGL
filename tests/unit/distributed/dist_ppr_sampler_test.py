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
from typing import Literal

import networkx as nx
import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from parameterized import param, parameterized
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.sampler_options import PPRSamplerOptions
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
        assert (
            ppr_edge_type in datum.edge_types
        ), f"Missing PPR edge type {ppr_edge_type} on HeteroData"

        ppr_edge_index = datum[ppr_edge_type].edge_index
        ppr_weights = datum[ppr_edge_type].edge_attr

        assert (
            ppr_edge_index.dim() == 2 and ppr_edge_index.size(0) == 2
        ), f"Expected [2, X] edge_index, got shape {list(ppr_edge_index.shape)}"
        assert ppr_weights.dim() == 1
        assert ppr_edge_index.size(1) == ppr_weights.size(0)
        assert (ppr_weights > 0).all(), f"PPR weights for {ntype} must be positive"
        assert (
            ppr_edge_index[0] == 0
        ).all(), "All src indices must be 0 for batch_size=1"

        global_node_ids = datum[ntype].node
        type_ppr: dict[int, float] = {}
        for j in range(ppr_edge_index.size(1)):
            local_dst = ppr_edge_index[1, j].item()
            global_dst = global_node_ids[local_dst].item()
            type_ppr[global_dst] = ppr_weights[j].item()
        ntype_to_sampler_ppr[str(ntype)] = type_ppr

    return ntype_to_sampler_ppr


def _assert_ppr_scores_match_reference(
    ntype_to_sampler_ppr: dict[str, dict[int, float]],
    reference_ppr: dict[str, dict[int, float]],
    seed_id: int,
) -> None:
    """Assert sampler PPR scores match reference scores per node type.

    Checks that top-k node sets are identical and that per-node scores
    are within atol=1e-5.  The forward push error per node is bounded by
    O(alpha * eps * degree); for max degree 3, alpha=0.5, eps=1e-6 the
    theoretical bound is ~1.5e-6, so 1e-5 provides a safety margin.

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
            assert abs(sam_score - ref_score) < 1e-5, (
                f"{seed_id}, type {ntype_str}, node {node_id}: "
                f"sampler={sam_score:.8f} vs reference={ref_score:.8f}"
            )


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

        assert hasattr(datum, "edge_index"), "Missing edge_index on Data"
        assert hasattr(datum, "edge_attr"), "Missing edge_attr on Data"

        ppr_edge_index = datum.edge_index
        ppr_weights = datum.edge_attr

        assert (
            ppr_edge_index.dim() == 2 and ppr_edge_index.size(0) == 2
        ), f"Expected [2, X] edge_index, got shape {list(ppr_edge_index.shape)}"
        assert ppr_weights.dim() == 1, f"Expected 1D weights, got {ppr_weights.dim()}D"
        assert ppr_edge_index.size(1) == ppr_weights.size(
            0
        ), f"Edge count mismatch: {ppr_edge_index.size(1)} vs {ppr_weights.size(0)}"
        assert (ppr_weights > 0).all(), "PPR weights must be positive"
        assert (
            ppr_edge_index[0] == 0
        ).all(), "All src indices must be 0 for batch_size=1"

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

        # Forward push is an approximation; with eps=1e-6 the per-node error
        # is bounded by O(alpha * eps * degree).  For this test graph
        # (max degree 3, alpha=0.5, eps=1e-6) the theoretical bound is ~1.5e-6.
        # Tolerance is set to 1e-5 to provide a safety margin above that bound.
        for node_id in reference_ppr:
            ref_score = reference_ppr[node_id]
            sam_score = sampler_ppr[node_id]
            assert abs(sam_score - ref_score) < 1e-5, (
                f"Seed {seed_global_id}, node {node_id}: "
                f"sampler={sam_score:.8f} vs reference={ref_score:.8f}"
            )

        batches_checked += 1

    assert (
        batches_checked == _NUM_TEST_NODES
    ), f"Expected {_NUM_TEST_NODES} batches, got {batches_checked}"
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
        input_nodes=(USER, node_ids[USER]),
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
            assert (
                edge_type[1] == "ppr"
            ), f"Non-PPR edge type {edge_type} found in PPR sampler output"

        batches_checked += 1

    assert (
        batches_checked == _NUM_TEST_USERS
    ), f"Expected {_NUM_TEST_USERS} batches, got {batches_checked}"
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
    assert isinstance(train_node_ids, dict)

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
            assert (
                ppr_edge_type in datum.edge_types
            ), f"Missing PPR edge type {ppr_edge_type} on HeteroData"

            ppr_edge_index = datum[ppr_edge_type].edge_index
            ppr_weights = datum[ppr_edge_type].edge_attr

            assert ppr_edge_index.dim() == 2 and ppr_edge_index.size(0) == 2
            assert ppr_weights.dim() == 1
            assert ppr_edge_index.size(1) == ppr_weights.size(0)
            if ppr_weights.numel() > 0:
                assert (ppr_weights > 0).all()
            assert (ppr_edge_index[1] >= 0).all()
            assert (ppr_edge_index[1] < datum[ntype].node.size(0)).all()

        for edge_type in datum.edge_types:
            assert (
                edge_type[1] == "ppr"
            ), f"Non-PPR edge type {edge_type} found in PPR sampler output"

        batches_checked += 1

    assert batches_checked > 0, "Expected at least one ABLP batch"
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
    assert (
        ppr_edge_type in datum.edge_types
    ), f"Missing PPR edge type {ppr_edge_type} — destination-only STORY was dropped"
    assert (
        datum[ppr_edge_type].edge_index.shape[1] > 0
    ), "Expected at least one PPR edge to STORY"
    assert (datum[ppr_edge_type].edge_attr > 0).all()

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
        input_nodes=(USER, train_node_ids[USER]),
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
    assert (
        1 not in sampler_ppr[str(STORY)]
    ), "story 1 appeared in user 0's PPR output — label edge was incorrectly traversed"

    # story 0 is reachable via message-passing and must be present.
    assert (
        0 in sampler_ppr[str(STORY)]
    ), "story 0 missing from user 0's PPR output — message-passing edge was not traversed"

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

    def test_ppr_sampler_ablp_ignores_label_edges_for_anchor_ppr(self) -> None:
        """Verify ABLP label edges are excluded from anchor-seed PPR walks."""
        mp.spawn(fn=_run_ppr_ablp_label_edges_do_not_affect_anchor_ppr, args=())


if __name__ == "__main__":
    absltest.main()
