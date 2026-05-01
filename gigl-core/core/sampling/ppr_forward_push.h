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
// Grouping all four tables into one struct keeps related data co-located in memory:
// when the push loop accesses pprScores, residuals, queue, and queuedNodes for the
// same (seed, ntype) pair, they are in the same cache line region rather than spread
// across four separate top-level vectors.
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
//   1. PPRForwardPushState(seedNodes, ...)
//   while True:
//   2. drainQueue()                         → nodes needing neighbor lookup
//   3. <Python: _batch_fetch_neighbors()>
//   4. pushResiduals(fetchedByEtypeId)
//   5. extractTopK(maxPprNodes)
class PPRForwardPushState {
   public:
    PPRForwardPushState(const torch::Tensor& seedNodes,
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
    void pushResiduals(const std::unordered_map<
                       int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>&
                           fetchedByEtypeId);

    // Return top-k PPR nodes per seed per node type.
    // Result: {ntype_id: (flat_ids, flat_weights, valid_counts)} — one entry per node type,
    // including types unreachable in this batch (empty tensors, all-zero valid_counts).
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
    extractTopK(int32_t maxPprNodes);

    // Total nodes drained per drainQueue() call, across all seeds and node types.
    [[nodiscard]] const std::vector<int32_t>& getNodesDrainedPerIteration() const;

   private:
    // Total out-degree of a node across all edge types. Returns 0 for sink nodes.
    [[nodiscard]] int32_t getTotalDegree(int32_t nodeId, int32_t nodeTypeId) const;

    double _alpha;
    double _requeueThresholdFactor;  // alpha * eps; per-node requeue threshold = factor * degree

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
    //
    //   outer vector  [_batchSize]    — one entry per seed node in the batch
    //   inner vector  [_numNodeTypes] — one SeedNodeTypeState per node type
    //
    // int32_t is used for node and type IDs throughout to match PyG/GLT's signed-integer
    // convention (torch.int32 / torch.int64).  Signed types also make nodeId >= 0 checks
    // meaningful — an unsigned type would make that guard tautological.
    //
    // Sized [_batchSize][_numNodeTypes] at construction and never resized,
    // so [seedIdx][nodeTypeId] indexing is always safe within the loop bounds.
    std::vector<std::vector<SeedNodeTypeState>> _state;

    // Neighbor lists fetched from the distributed graph store, keyed by packKey(node_id, etype_id).
    // Populated incrementally as nodes are processed; avoids re-fetching the same node twice.
    std::unordered_map<uint64_t, std::vector<int32_t>> _neighborCache;

    // Total nodes drained (across all seeds and types) per drainQueue() call.
    std::vector<int32_t> _nodesDrainedPerIteration;
};

}  // namespace gigl
