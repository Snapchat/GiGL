#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    // Result: {ntype_id: (flat_ids, flat_weights, valid_counts)} — one entry per active ntype.
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
    extractTopK(int32_t maxPprNodes);

    // Total nodes drained per drainQueue() call, across all seeds and node types.
    [[nodiscard]] const std::vector<int32_t>& getNodesDrainedPerIteration() const;

   private:
    // Total out-degree of a node across all edge types. Returns 0 for sink nodes.
    [[nodiscard]] int32_t getTotalDegree(int32_t nodeId, int32_t ntypeId) const;

    double _alpha;
    double _requeueThresholdFactor;   // alpha * eps; multiplied by degree for per-node threshold

    int32_t _batchSize;
    int32_t _numNodeTypes;
    int32_t _numNodesInQueue{0};

    // Graph structure (read-only after construction)
    std::vector<std::vector<int32_t>> _nodeTypeToEdgeTypeIds;
    std::vector<int32_t> _edgeTypeToDstNtypeId;
    std::vector<torch::Tensor> _degreeTensors;

    // Per-seed, per-node-type PPR state [seed_idx][ntype_id]
    std::vector<std::vector<std::unordered_map<int32_t, double>>> _pprScores;
    std::vector<std::vector<std::unordered_map<int32_t, double>>> _residuals;
    std::vector<std::vector<std::unordered_set<int32_t>>> _queue;
    std::vector<std::vector<std::unordered_set<int32_t>>> _queuedNodes;  // snapshot from drainQueue

    std::unordered_map<uint64_t, std::vector<int32_t>> _neighborCache;
    std::vector<int32_t> _nodesDrainedPerIteration;
};
