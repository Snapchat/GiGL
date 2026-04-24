#pragma once

#include <torch/torch.h>

#include <algorithm>      // std::partial_sort, std::min
#include <cstdint>        // Fixed-width integer types: int32_t, int64_t, uint32_t, uint64_t
#include <optional>       // std::optional for nullable return values
#include <tuple>          // std::tuple for multi-value returns
#include <unordered_map>  // std::unordered_map — like Python dict, O(1) average lookup
#include <unordered_set>  // std::unordered_set — like Python set, O(1) average lookup
#include <vector>         // std::vector — like Python list, contiguous in memory

// Combine (node_id, etype_id) into a single 64-bit integer for use as a hash
// map key.  A single 64-bit integer is cheaper to hash than a pair of two
// integers (std::unordered_map has no built-in pair hash).
//
// Bit layout:
//   bits 63–32: node_id  (upper half)
//   bits 31– 0: etype_id (lower half)
//
// Both inputs are cast through uint32_t before packing.  Without this, a
// negative int32_t (e.g. -1 = 0xFFFFFFFF) would be sign-extended to a full
// 64-bit value, corrupting the upper bits when shifted.  Reinterpreting as
// uint32_t first treats the bit pattern as-is (no sign extension).
static inline uint64_t packKey(int32_t nodeId, int32_t etypeId) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(nodeId)) << 32) |
           static_cast<uint32_t>(etypeId);
}

// C++ kernel for the PPR Forward Push algorithm (Andersen et al., 2006).
//
// All hot-loop state (scores, residuals, queue, neighbor cache) lives inside
// this object.  The distributed neighbor fetch is kept in Python because it
// involves async RPC calls that C++ cannot drive directly.
//
// Owned state: _pprScores, _residuals, _queue, _queuedNodes, _neighborCache.
// Python retains ownership of: the distributed neighbor fetch (_batch_fetch_neighbors).
//
// Typical call sequence per batch:
//   1.  PPRForwardPushState(seedNodes, ...)   — init per-seed residuals / queue
//   while True:
//   2.  drainQueue()                          — drain queue → nodes needing lookup
//   3.  <Python: _batch_fetch_neighbors(...)>  — distributed RPC fetch (stays in Python)
//   4.  pushResiduals(fetchedByEtypeId)        — push residuals, update queue
//   5.  extractTopK(maxPprNodes)               — top-k selection per seed per node type
class PPRForwardPushState {
   public:
    PPRForwardPushState(const torch::Tensor& seedNodes,
                        int32_t seedNodeTypeId,
                        double alpha,
                        double requeueThresholdFactor,
                        std::vector<std::vector<int32_t>> nodeTypeToEdgeTypeIds,
                        std::vector<int32_t> edgeTypeToDstNtypeId,
                        std::vector<torch::Tensor> degreeTensors);

    // Drain all queued nodes and return {etype_id: tensor[node_ids]} for batch
    // neighbor lookup.  Also snapshots the drained nodes into _queuedNodes for
    // use by pushResiduals().
    //
    // Return value semantics:
    //   - std::nullopt   → queue was already empty; convergence achieved; stop the loop.
    //   - empty map      → nodes were drained but all were cached; call pushResiduals({}).
    //   - non-empty map  → {etype_id → 1-D int64 tensor of node IDs} needing neighbor lookup.
    std::optional<std::unordered_map<int32_t, torch::Tensor>> drainQueue();

    // Push residuals to neighbors given the fetched neighbor data.
    //
    // fetchedByEtypeId: {etype_id: (node_ids_tensor, flat_nbrs_tensor, counts_tensor)}
    //   - node_ids_tensor:  [N]           int64 — source node IDs fetched for this edge type
    //   - flat_nbrs_tensor: [sum(counts)] int64 — all neighbor lists concatenated flat
    //   - counts_tensor:    [N]           int64 — neighbor count for each source node
    void pushResiduals(const std::unordered_map<
                       int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>&
                           fetchedByEtypeId);

    // Extract top-k PPR nodes per seed per node type.
    //
    // Returns {ntype_id: (flat_ids_tensor, flat_weights_tensor, valid_counts_tensor)}.
    // Only node types that received any PPR score are included in the output.
    //
    // Output layout for a batch of B seeds:
    //   flat_ids[0 : valid_counts[0]]                 → top-k nodes for seed 0
    //   flat_ids[valid_counts[0] : valid_counts[0]+valid_counts[1]] → top-k for seed 1
    //   ...
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
    extractTopK(int32_t maxPprNodes);

    // Returns _nodesDrainedPerIteration built up across all drainQueue() calls.
    [[nodiscard]] const std::vector<int32_t>& getNodesDrainedPerIteration() const;

   private:
    // Look up the total (across all edge types) out-degree of a node.
    // Returns 0 for destination-only node types (no outgoing edges).
    [[nodiscard]] int32_t getTotalDegree(int32_t nodeId, int32_t ntypeId) const;

    // -------------------------------------------------------------------------
    // Scalar algorithm parameters
    // -------------------------------------------------------------------------
    double _alpha;                    // Restart probability
    double _oneMinusAlpha;            // 1 - alpha, precomputed to avoid repeated subtraction
    double _requeueThresholdFactor;   // alpha * eps; multiplied by degree to get per-node threshold

    int32_t _batchSize;                    // Number of seeds in the current batch
    int32_t _numNodeTypes;                 // Total number of node types (homo + hetero)
    int32_t _numNodesInQueue{0};           // Running count of nodes across all seeds / types

    // -------------------------------------------------------------------------
    // Graph structure (read-only after construction)
    // -------------------------------------------------------------------------
    std::vector<std::vector<int32_t>> _nodeTypeToEdgeTypeIds;
    std::vector<int32_t> _edgeTypeToDstNtypeId;
    std::vector<torch::Tensor> _degreeTensors;

    // -------------------------------------------------------------------------
    // Per-seed, per-node-type PPR state (indexed [seed_idx][ntype_id])
    // -------------------------------------------------------------------------
    std::vector<std::vector<std::unordered_map<int32_t, double>>> _pprScores;
    std::vector<std::vector<std::unordered_map<int32_t, double>>> _residuals;
    std::vector<std::vector<std::unordered_set<int32_t>>> _queue;
    std::vector<std::vector<std::unordered_set<int32_t>>> _queuedNodes;

    // -------------------------------------------------------------------------
    // Neighbor cache
    // -------------------------------------------------------------------------
    std::unordered_map<uint64_t, std::vector<int32_t>> _neighborCache;

    // -------------------------------------------------------------------------
    // Diagnostics (populated during the algorithm; read after convergence)
    // -------------------------------------------------------------------------
    // Total nodes drained (across all seeds and node types) in each drainQueue()
    // call.  One entry per loop iteration; useful for understanding convergence speed.
    std::vector<int32_t> _nodesDrainedPerIteration;
};
