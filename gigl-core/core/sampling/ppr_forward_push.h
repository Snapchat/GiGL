#pragma once

#include <torch/torch.h>

#include <algorithm>
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
    std::unordered_map<int32_t, double> pprScores;   // absorbed PPR mass
    std::unordered_map<int32_t, double> residuals;   // unabsorbed mass waiting to push
    std::unordered_set<int32_t> queue;               // nodes queued for the next drain
    std::unordered_set<int32_t> queuedNodes;         // snapshot captured by drainQueue()
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
//   5. extractTopK(maxPprNodes)
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
    // fetchedByEtypeId: {etype_id: (node_ids[N], flat_nbrs[sum(counts)], counts[N], flat_weights[sum(counts)])}
    // flat_weights is empty (numel()==0) for uniform-residual mode; non-empty for
    // weight-proportional mode.  _hasWeights is latched true on the first call with a
    // non-empty flat_weights and never reset within one PPRForwardPush lifetime.
    void pushResiduals(const std::unordered_map<
                       int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>&
                           fetchedByEtypeId);

    // Return top-k PPR nodes per seed per node type.
    // Result: {ntype_id: (flat_ids, flat_weights, valid_counts)} — one entry per node type,
    // including types unreachable in this batch (empty tensors, all-zero valid_counts).
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
    extractTopK(int32_t maxPprNodes);

   private:
    // Total out-degree of a node across all edge types. Returns 0 for sink nodes.
    [[nodiscard]] int32_t getTotalDegree(int32_t nodeId, int32_t nodeTypeId) const;

    double _alpha;
    double _requeueThresholdFactor;  // alpha * eps; per-node requeue threshold = factor * degree

    // NOTE: int32_t is used for batch size, node IDs, and type IDs throughout this class.
    // All of this code will break silently (overflow) if batch size or node IDs exceed ~2B
    // (INT32_MAX = 2,147,483,647).  This is not a realistic concern today, but if graph
    // scale ever approaches that threshold, these should be widened to int64_t.
    int32_t _batchSize;       // number of seed nodes in the current batch
    int32_t _numNodeTypes;    // total distinct node types (1 for homogeneous graphs)
    int32_t _numNodesInQueue{0};  // running count of queued nodes across all seeds and types

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

    // True once any pushResiduals call receives a non-empty flat_weights tensor.
    // Latched true for the object lifetime; never reset.
    bool _hasWeights{false};

    // Per-edge weights parallel to _neighborCache: _weightCache[packKey(node, etype)][i]
    // is the weight of the i-th cached neighbor.  Only populated in weighted mode.
    std::unordered_map<uint64_t, std::vector<double>> _weightCache;

};

}  // namespace gigl
