#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gigl {

// Per-seed, per-node-type PPR algorithm state.
// Grouping all four tables into one struct is a logical convenience: a single
// _state[seedIdx][nodeTypeId] access reaches all four tables for a given (seed, ntype)
// pair, rather than indexing four separate 2D arrays.  Note that unordered_map and
// unordered_set heap-allocate their bucket storage, so the actual key-value data is
// not co-located in memory — only the control-plane metadata (size, bucket pointer)
// lives inside the struct.
struct SeedNodeTypeState {
    std::unordered_map<int32_t, double> pprScores; // absorbed PPR mass
    std::unordered_map<int32_t, double> residuals; // unabsorbed mass waiting to push
    std::unordered_set<int32_t> queue;             // nodes queued for the next drain
    std::unordered_set<int32_t> queuedNodes;       // snapshot captured by drainQueue()
};

// Batched drain result for typed-PPR channels.
struct DrainedTypedPPRQueues {
    // Channels that drained queued nodes and need pushResiduals() this iteration.
    std::vector<int32_t> activeChannelIndices;

    // Channels that have non-empty uncached frontiers and remaining fetch budget.
    std::vector<int32_t> fetchChannelIndices;

    // Edge types requested by each fetch channel, aligned with fetchChannelIndices.
    std::vector<std::vector<int32_t>> edgeTypeIdsByFetchChannel;

    // Unioned node frontier for one shared distributed neighbor fetch.
    std::unordered_map<int32_t, torch::Tensor> unionNodesByEdgeTypeId;
};

// C++ kernel for PPR Forward Push (Andersen et al., 2006).
// Hot-loop state lives here; distributed neighbor fetches are driven from Python.
//
// Call sequence per batch:
//   1. PPRForwardPush(seedNodes, ...)
//   while True:
//   2. drainQueue()                         → nodes needing neighbor lookup
//   3. <Python: _batch_fetch_neighbors()>
//   4. pushResiduals(fetchedByEtypeId)
//   5. extractTopKWithResidualTopUp(maxPPRNodes, enableResidualTopUp)
class PPRForwardPush {
public:
    PPRForwardPush(const torch::Tensor& seedNodes,
                   int32_t seedNodeTypeId,
                   double alpha,
                   double requeueThresholdFactor,
                   std::vector<std::vector<int32_t>> nodeTypeToEdgeTypeIds,
                   std::vector<int32_t> edgeTypeToDstNtypeId,
                   std::vector<torch::Tensor> degreeTensors);

    // Drain queued nodes and return {etype_id: int64 node tensor} for neighbor lookup.
    // Returns nullopt when the queue is empty (convergence). Empty map means all nodes
    // were cache-hits; call pushResiduals({}) to continue.
    std::optional<std::unordered_map<int32_t, torch::Tensor>> drainQueue();

    // Push residuals given fetched neighbor data.
    // fetchedByEtypeId: {etype_id: (node_ids[N], flat_nbrs[sum(counts)], counts[N])}
    // TODO: Move these repeated tensor tuple/map types into aliases in a follow-up
    // refactor-only PR.
    void pushResiduals(
        const std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>& fetchedByEtypeId);

    // Return top-k PPR nodes plus residual-mass top-up nodes, sorted by score.
    //
    // Residual top-up does not issue new neighbor fetches.  It only reads the
    // residual table already built by Forward Push.  This gives callers a way
    // to fill short sequences with nodes that were discovered but did not cross
    // the requeue threshold, without lowering eps and running more push steps.
    // Scores are emitted on the
    // same mass scale as PPR scores: ppr_score(node) + residual(node), i.e. the
    // score the node would have if the remaining residual at that node were
    // absorbed locally.  Residual candidates only fill the requested top-up
    // budget; they do not displace selected finalized-PPR nodes. The returned
    // set is selected by this two-phase policy, then sorted by emitted score;
    // it is not a global top-k over ppr_score + residual when maxPPRNodes is tight.
    // maxPPRNodes is the final per-seed cap across finalized PPR and residual
    // top-up candidates.
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> extractTopKWithResidualTopUp(
        int32_t maxPPRNodes, bool enableResidualTopUp);

private:
    // Total out-degree of a node across all edge types. Returns 0 for sink nodes.
    [[nodiscard]] int32_t getTotalDegree(int32_t nodeId, int32_t nodeTypeId) const;

    double _alpha;
    double _requeueThresholdFactor; // alpha * eps; per-node requeue threshold = factor * degree

    // NOTE: int32_t is used for batch size, node IDs, and type IDs throughout this class.
    // All of this code will break silently (overflow) if batch size or node IDs exceed ~2B
    // (INT32_MAX = 2,147,483,647).  This is not a realistic concern today, but if graph
    // scale ever approaches that threshold, these should be widened to int64_t.
    int32_t _batchSize;          // number of seed nodes in the current batch
    int32_t _numNodeTypes;       // total distinct node types (1 for homogeneous graphs)
    int32_t _numNodesInQueue{0}; // running count of queued nodes across all seeds and types

    // Graph structure — set at construction, read-only during the algorithm.
    // _nodeTypeToEdgeTypeIds[ntype_id] → list of edge type IDs that originate from that node type.
    // _edgeTypeToDstNtypeId[etype_id]  → destination node type ID for that edge type.
    // _degreeTensors[ntype_id]         → int32 tensor of total out-degrees, indexed by node ID.
    std::vector<std::vector<int32_t>> _nodeTypeToEdgeTypeIds;
    std::vector<int32_t> _edgeTypeToDstNtypeId;
    std::vector<torch::Tensor> _degreeTensors;

    // Per-seed, per-node-type PPR state.  Indexed as _state[seedIdx][nodeTypeId].
    // 2D vector: both dimensions are dense sequential integers bounded at construction,
    // so array indexing is O(1) with no hashing (contrast with _neighborCache below).
    //
    // int32_t is used for node and type IDs throughout to match PyG/GLT's signed-integer
    // convention (torch.int32 / torch.int64).  Signed types also make nodeId >= 0 checks
    // meaningful — an unsigned type would make that guard tautological.
    //
    // Sized [_batchSize][_numNodeTypes] at construction and never resized,
    // so [seedIdx][nodeTypeId] indexing is always safe within the loop bounds.
    std::vector<std::vector<SeedNodeTypeState>> _state;

    // Neighbor lists keyed by packKey(nodeId, edgeTypeId).
    // Hash map: nodeId is a sparse graph ID from a large graph, so a dense array is
    // impractical (contrast with _state above).  Populated incrementally; avoids re-fetching.
    std::unordered_map<uint64_t, std::vector<int32_t>> _neighborCache;
};

// Drain several independent PPR states and union their fetch frontier by edge type.
// Used by typed PPR to keep per-channel PPR state separate while issuing one
// shared neighbor fetch for duplicate channel frontier requests.
DrainedTypedPPRQueues drainTypedPPRChannelQueues(const std::vector<PPRForwardPush*>& states,
                                                 const std::vector<int32_t>& fetchIterationCounts,
                                                 int32_t maxFetchIterations);

} // namespace gigl
