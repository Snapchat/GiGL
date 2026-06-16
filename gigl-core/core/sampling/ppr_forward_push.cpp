#include "ppr_forward_push.h"

#include <torch/torch.h>

#include <climits>
#include <cstdint>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gigl {

// Pack (node_id, etype_id) into a single uint64 for use as a hash key.
// Inputs are cast through uint32_t to avoid sign-extension of negative int32 values.
static uint64_t packKey(int32_t nodeId, int32_t edgeTypeId) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(nodeId)) << 32) | static_cast<uint32_t>(edgeTypeId);
}

PPRForwardPush::PPRForwardPush(const torch::Tensor& seedNodes,
                               int32_t seedNodeTypeId,
                               double alpha,
                               double requeueThresholdFactor,
                               std::vector<std::vector<int32_t>> nodeTypeToEdgeTypeIds,
                               std::vector<int32_t> edgeTypeToDstNtypeId,
                               std::vector<torch::Tensor> degreeTensors)
    : _alpha(alpha),
      _requeueThresholdFactor(requeueThresholdFactor),
      // std::move transfers ownership of each vector into the member variable
      // without copying its contents — equivalent to Python's list hand-off
      // when you no longer need the original.
      _nodeTypeToEdgeTypeIds(std::move(nodeTypeToEdgeTypeIds)),
      _edgeTypeToDstNtypeId(std::move(edgeTypeToDstNtypeId)),
      _degreeTensors(std::move(degreeTensors)) {
    TORCH_CHECK(seedNodes.dim() == 1, "seedNodes must be 1D");
    // int32_t is sufficient: batch sizes approaching 2B seeds are not a realistic concern.
    _batchSize = static_cast<int32_t>(seedNodes.size(0));
    _numNodeTypes = static_cast<int32_t>(_nodeTypeToEdgeTypeIds.size());

    TORCH_CHECK(seedNodeTypeId >= 0, "seedNodeTypeId ", seedNodeTypeId, " is negative.");
    TORCH_CHECK(
        seedNodeTypeId < _numNodeTypes, "seedNodeTypeId ", seedNodeTypeId, " out of range [0, ", _numNodeTypes, ").");
    auto numEdgeTypes = static_cast<int32_t>(_edgeTypeToDstNtypeId.size());
    for (int32_t edgeTypeId = 0; edgeTypeId < numEdgeTypes; ++edgeTypeId) {
        int32_t dstNodeTypeId = _edgeTypeToDstNtypeId[edgeTypeId];
        TORCH_CHECK(dstNodeTypeId >= 0, "edgeTypeToDstNtypeId[", edgeTypeId, "] = ", dstNodeTypeId, " is negative.");
        TORCH_CHECK(dstNodeTypeId < _numNodeTypes,
                    "edgeTypeToDstNtypeId[",
                    edgeTypeId,
                    "] = ",
                    dstNodeTypeId,
                    " out of range [0, ",
                    _numNodeTypes,
                    ").");
    }
    for (int32_t nodeTypeId = 0; nodeTypeId < _numNodeTypes; ++nodeTypeId) {
        for (int32_t edgeTypeId : _nodeTypeToEdgeTypeIds[nodeTypeId]) {
            TORCH_CHECK(edgeTypeId >= 0,
                        "nodeTypeToEdgeTypeIds[",
                        nodeTypeId,
                        "] contains negative edge type id ",
                        edgeTypeId,
                        ".");
            TORCH_CHECK(edgeTypeId < numEdgeTypes,
                        "nodeTypeToEdgeTypeIds[",
                        nodeTypeId,
                        "] contains edge type id ",
                        edgeTypeId,
                        " out of range [0, ",
                        numEdgeTypes,
                        ").");
        }
    }

    // Allocate per-seed, per-node-type state.
    // .assign(n, val) fills a vector with n independent copies of val — like [val for _ in range(n)] in Python.
    _state.assign(_batchSize, std::vector<SeedNodeTypeState>(_numNodeTypes));

    // accessor<dtype, ndim>() returns a typed view into the tensor's data that
    // supports [i] indexing with bounds checking in debug builds.
    auto seedNodeAcc = seedNodes.accessor<int64_t, 1>();
    _numNodesInQueue = _batchSize;
    for (int32_t seedIdx = 0; seedIdx < _batchSize; ++seedIdx) {
        auto seedNodeId = static_cast<int32_t>(seedNodeAcc[seedIdx]);
        // PPR initialisation: each seed starts with residual = alpha (the
        // restart probability).  The first push will move alpha into ppr_score
        // and distribute (1-alpha)*alpha to the seed's neighbors.
        _state[seedIdx][seedNodeTypeId].residuals[seedNodeId] = _alpha;
        _state[seedIdx][seedNodeTypeId].queue.insert(seedNodeId);
    }
}

std::optional<std::unordered_map<int32_t, torch::Tensor>> PPRForwardPush::drainQueue() {
    if (_numNodesInQueue == 0) {
        return std::nullopt;
    }

    // Reset the snapshot from the previous iteration.
    // TODO: if this loop becomes a bottleneck, consider parallelising with
    // std::for_each(std::execution::par_unseq, ...) or adding vectorisation hints.
    for (auto& perSeedState : _state) {
        for (auto& nodeTypeState : perSeedState) {
            nodeTypeState.queuedNodes.clear();
        }
    }

    // nodesToLookup[edgeTypeId] = set of node IDs that need a neighbor fetch for
    // edge type edgeTypeId this round.  Using a set deduplicates nodes that appear
    // in multiple seeds' queues: we only fetch each (node, etype) pair once.
    std::unordered_map<int32_t, std::unordered_set<int32_t>> nodesToLookup;

    // TODO: For homogeneous graphs _numNodeTypes == 1, so the inner loop always
    // executes exactly once (nodeTypeId=0).  std::vector indexing is cheap, but a
    // dedicated homogeneous code path could eliminate the loop entirely.  Profile
    // before splitting.
    for (int32_t seedIdx = 0; seedIdx < _batchSize; ++seedIdx) {
        for (int32_t nodeTypeId = 0; nodeTypeId < _numNodeTypes; ++nodeTypeId) {
            auto& seedNodeTypeState = _state[seedIdx][nodeTypeId];
            if (seedNodeTypeState.queue.empty()) {
                continue;
            }

            // Move the live queue into the snapshot in O(1) — avoids copying all node IDs.
            // The explicit clear() after move is defensive: the standard only guarantees
            // a moved-from container is "valid but unspecified", not necessarily empty.
            seedNodeTypeState.queuedNodes = std::move(seedNodeTypeState.queue);
            seedNodeTypeState.queue.clear();
            _numNodesInQueue -= static_cast<int32_t>(seedNodeTypeState.queuedNodes.size());

            for (int32_t nodeId : seedNodeTypeState.queuedNodes) {
                for (int32_t edgeTypeId : _nodeTypeToEdgeTypeIds[nodeTypeId]) {
                    if (_neighborCache.find(packKey(nodeId, edgeTypeId)) == _neighborCache.end()) {
                        nodesToLookup[edgeTypeId].insert(nodeId);
                    }
                }
            }
        }
    }

    std::unordered_map<int32_t, torch::Tensor> result;
    for (const auto& [edgeTypeId, nodeSet] : nodesToLookup) {
        std::vector<int64_t> nodeIdsToLookup(nodeSet.begin(), nodeSet.end());
        result[edgeTypeId] = torch::tensor(nodeIdsToLookup, torch::kLong);
    }
    return result;
}

void PPRForwardPush::pushResiduals(
    const std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>&
        fetchedByEtypeId) {
    // Step 1: Unpack the input map into C++ maps keyed by packKey(nodeId, edgeTypeId)
    // for fast lookup during the residual-push loop below.
    // fetchedWeights is populated only in weighted mode (_hasWeights becomes true on
    // the first call that includes a non-empty flat_weights tensor).
    std::unordered_map<uint64_t, std::vector<int32_t>> fetched;
    std::unordered_map<uint64_t, std::vector<double>> fetchedWeights;

    for (const auto& [edgeTypeId, neighborTensors] : fetchedByEtypeId) {
        const auto& nodeIdsTensor = std::get<0>(neighborTensors);
        const auto& flatNeighborIdsTensor = std::get<1>(neighborTensors);
        const auto& countsTensor = std::get<2>(neighborTensors);
        const auto& flatWeightsTensor = std::get<3>(neighborTensors);

        bool etypeHasWeights = flatWeightsTensor.numel() > 0;
        if (etypeHasWeights) {
            _hasWeights = true;
        }

        // accessor<int64_t, 1>() gives a bounds-checked, typed 1-D view into
        // each tensor's data — equivalent to iterating over a NumPy array.
        auto nodeIdsAccessor = nodeIdsTensor.accessor<int64_t, 1>();
        auto flatNeighborIdsAccessor = flatNeighborIdsTensor.accessor<int64_t, 1>();
        auto countsAccessor = countsTensor.accessor<int64_t, 1>();
        // Raw pointer for weights avoids a conditional accessor construction.
        const double* flatWeightsPtr = etypeHasWeights ? flatWeightsTensor.data_ptr<double>() : nullptr;

        // Walk the flat neighbor list, slicing out each node's neighbors using
        // the running offset into the concatenated flat buffer.
        int64_t offset = 0;
        for (int64_t nodeIdx = 0; nodeIdx < nodeIdsTensor.size(0); ++nodeIdx) {
            auto nodeId = static_cast<int32_t>(nodeIdsAccessor[nodeIdx]);
            int64_t count = countsAccessor[nodeIdx];
            uint64_t key = packKey(nodeId, edgeTypeId);

            std::vector<int32_t> neighborIds(count);
            for (int64_t neighborIdx = 0; neighborIdx < count; ++neighborIdx) {
                neighborIds[neighborIdx] = static_cast<int32_t>(flatNeighborIdsAccessor[offset + neighborIdx]);
            }
            fetched[key] = std::move(neighborIds);

            if (flatWeightsPtr != nullptr) {
                std::vector<double> neighborWeights(count);
                for (int64_t neighborIdx = 0; neighborIdx < count; ++neighborIdx) {
                    double weight = flatWeightsPtr[offset + neighborIdx];
                    TORCH_CHECK(weight >= 0.0, "PPR edge weights must be non-negative.");
                    neighborWeights[neighborIdx] = weight;
                }
                fetchedWeights[key] = std::move(neighborWeights);
            }

            offset += count;
        }
    }

    // Promote neighbor and weight lists for a newly re-queued node into the persistent cache.
    // Called from both uniform and weighted paths — the _hasWeights guard inside handles
    // whether _weightCache is populated.
    auto promoteToCache = [&](int32_t neighborNodeId, int32_t dstNodeTypeId) {
        for (int32_t neighborEdgeTypeId : _nodeTypeToEdgeTypeIds[dstNodeTypeId]) {
            uint64_t packedKey = packKey(neighborNodeId, neighborEdgeTypeId);
            if (_neighborCache.find(packedKey) == _neighborCache.end()) {
                auto fetchedNeighborEntry = fetched.find(packedKey);
                if (fetchedNeighborEntry != fetched.end()) {
                    _neighborCache[packedKey] = fetchedNeighborEntry->second;
                    if (_hasWeights) {
                        auto fetchedWeightsNeighborEntry = fetchedWeights.find(packedKey);
                        if (fetchedWeightsNeighborEntry != fetchedWeights.end()) {
                            _weightCache[packedKey] = fetchedWeightsNeighborEntry->second;
                        }
                    }
                }
            }
        }
    };

    // Step 2: For every node that was in the queue (captured in _queuedNodes
    // by drainQueue()), apply one PPR push step:
    //   a. Absorb residual into the PPR score.
    //   b. Compute the normalisation factor (neighbor count or total weight).
    //   c. Distribute (1-alpha) * residual to each neighbor: uniformly when
    //      _hasWeights is false; proportionally to edge weight when true.
    //   d. Enqueue any neighbor whose residual now exceeds the requeue threshold.
    for (int32_t seedIdx = 0; seedIdx < _batchSize; ++seedIdx) {
        for (int32_t nodeTypeId = 0; nodeTypeId < _numNodeTypes; ++nodeTypeId) {
            auto& srcNodeTypeState = _state[seedIdx][nodeTypeId];
            if (srcNodeTypeState.queuedNodes.empty()) {
                continue;
            }

            for (int32_t sourceNodeId : srcNodeTypeState.queuedNodes) {
                auto residualIter = srcNodeTypeState.residuals.find(sourceNodeId);
                double sourceResidual = (residualIter != srcNodeTypeState.residuals.end()) ? residualIter->second : 0.0;

                // a. Absorb: move residual into the PPR score.
                srcNodeTypeState.pprScores[sourceNodeId] += sourceResidual;
                srcNodeTypeState.residuals[sourceNodeId] = 0.0;

                if (!_hasWeights) {
                    // --- Uniform path ---
                    // b. Count total fetched/cached neighbors across all edge types for
                    // this source node.  We normalise by the number of neighbors we
                    // actually retrieved, not the true degree, so residual is fully
                    // distributed among known neighbors rather than leaking to unfetched
                    // ones (which matters when num_neighbors_per_hop < true_degree).
                    int32_t totalFetched = 0;
                    for (int32_t edgeTypeId : _nodeTypeToEdgeTypeIds[nodeTypeId]) {
                        auto fetchedEntry = fetched.find(packKey(sourceNodeId, edgeTypeId));
                        if (fetchedEntry != fetched.end()) {
                            totalFetched += static_cast<int32_t>(fetchedEntry->second.size());
                        } else {
                            auto cachedEntry = _neighborCache.find(packKey(sourceNodeId, edgeTypeId));
                            if (cachedEntry != _neighborCache.end()) {
                                totalFetched += static_cast<int32_t>(cachedEntry->second.size());
                            }
                        }
                    }
                    // Two cases reach here:
                    //   1. True sink node (no outgoing edges): absorbing the full residual is correct.
                    //   2. Budget exhausted, no cache entry: the (1-α)·r that should flow to
                    //      neighbors has nowhere to go, so it gets absorbed into src's score instead.
                    //      This overstates src and understates its neighbors.  This is expected
                    //      behavior when max_fetch_iterations is set, which intentionally trades
                    //      theoretical PPR correctness for better throughput.
                    if (totalFetched == 0) {
                        continue;
                    }

                    double residualPerNeighbor = (1.0 - _alpha) * sourceResidual / static_cast<double>(totalFetched);

                    for (int32_t edgeTypeId : _nodeTypeToEdgeTypeIds[nodeTypeId]) {
                        // Invariant: fetched and _neighborCache are mutually exclusive for
                        // any given (node, etype) key within one iteration.  drainQueue()
                        // only requests a fetch for nodes absent from _neighborCache, so a
                        // key is in at most one of the two.
                        //
                        // Neighbor list for this (src, edgeTypeId) pair, borrowed from whichever
                        // map holds it.  reference_wrapper is used because std::optional cannot
                        // hold a reference directly, and we want to avoid copying the vector —
                        // the data already exists in fetched or _neighborCache and both outlive
                        // this loop body.  Access via neighborList->get().
                        std::optional<std::reference_wrapper<const std::vector<int32_t>>> neighborList;
                        auto fetchedEntry = fetched.find(packKey(sourceNodeId, edgeTypeId));
                        if (fetchedEntry != fetched.end()) {
                            neighborList = std::cref(fetchedEntry->second);
                        } else {
                            auto cachedEntry = _neighborCache.find(packKey(sourceNodeId, edgeTypeId));
                            if (cachedEntry != _neighborCache.end()) {
                                neighborList = std::cref(cachedEntry->second);
                            }
                        }
                        if (!neighborList || neighborList->get().empty()) {
                            continue;
                        }

                        int32_t dstNodeTypeId = _edgeTypeToDstNtypeId[edgeTypeId];

                        // c. Accumulate residual for each neighbor and re-enqueue if threshold
                        // exceeded.
                        auto& dstNodeTypeState = _state[seedIdx][dstNodeTypeId];
                        for (int32_t neighborNodeId : neighborList->get()) {
                            dstNodeTypeState.residuals[neighborNodeId] += residualPerNeighbor;

                            double threshold = _requeueThresholdFactor *
                                               static_cast<double>(getTotalDegree(neighborNodeId, dstNodeTypeId));

                            if (dstNodeTypeState.queue.find(neighborNodeId) == dstNodeTypeState.queue.end() &&
                                dstNodeTypeState.residuals[neighborNodeId] >= threshold) {
                                dstNodeTypeState.queue.insert(neighborNodeId);
                                ++_numNodesInQueue;
                                promoteToCache(neighborNodeId, dstNodeTypeId);
                            }
                        }
                    }
                } else {
                    // --- Weighted path ---
                    // b. Sum total weight of fetched/cached neighbors across all edge types.
                    // We normalise by total fetched weight rather than by true out-weight so
                    // that the residual is fully distributed among known neighbors, consistent
                    // with how the uniform path handles truncated neighbor lists.
                    double totalFetchedWeight = 0.0;
                    for (int32_t edgeTypeId : _nodeTypeToEdgeTypeIds[nodeTypeId]) {
                        uint64_t key = packKey(sourceNodeId, edgeTypeId);
                        auto fetchedWeightsEntry = fetchedWeights.find(key);
                        if (fetchedWeightsEntry != fetchedWeights.end()) {
                            for (double w : fetchedWeightsEntry->second) {
                                totalFetchedWeight += w;
                            }
                        } else {
                            auto cachedWeightsEntry = _weightCache.find(key);
                            if (cachedWeightsEntry != _weightCache.end()) {
                                for (double w : cachedWeightsEntry->second) {
                                    totalFetchedWeight += w;
                                }
                            }
                        }
                    }
                    // Sink node or all-zero-weight edges: absorb residual, nothing to distribute.
                    if (totalFetchedWeight == 0.0) {
                        continue;
                    }

                    double baseResidual = (1.0 - _alpha) * sourceResidual;

                    for (int32_t edgeTypeId : _nodeTypeToEdgeTypeIds[nodeTypeId]) {
                        uint64_t key = packKey(sourceNodeId, edgeTypeId);

                        std::optional<std::reference_wrapper<const std::vector<int32_t>>> neighborList;
                        std::optional<std::reference_wrapper<const std::vector<double>>> weightList;

                        auto fetchedEntry = fetched.find(key);
                        if (fetchedEntry != fetched.end()) {
                            neighborList = std::cref(fetchedEntry->second);
                            auto fetchedWeightsEntry = fetchedWeights.find(key);
                            if (fetchedWeightsEntry != fetchedWeights.end()) {
                                weightList = std::cref(fetchedWeightsEntry->second);
                            }
                        } else {
                            auto cachedEntry = _neighborCache.find(key);
                            if (cachedEntry != _neighborCache.end()) {
                                neighborList = std::cref(cachedEntry->second);
                                auto cachedWeightsEntry = _weightCache.find(key);
                                if (cachedWeightsEntry != _weightCache.end()) {
                                    weightList = std::cref(cachedWeightsEntry->second);
                                }
                            }
                        }
                        if (!neighborList || neighborList->get().empty()) {
                            continue;
                        }

                        int32_t dstNodeTypeId = _edgeTypeToDstNtypeId[edgeTypeId];
                        auto& dstNodeTypeState = _state[seedIdx][dstNodeTypeId];

                        // c. Accumulate weight-proportional residual for each neighbor.
                        // weightList is always populated alongside neighborList in weighted mode:
                        // fetched[key] and fetchedWeights[key] are set together in Step 1,
                        // and _neighborCache[key] and _weightCache[key] are promoted together.
                        TORCH_INTERNAL_ASSERT(weightList.has_value(),
                                              "weightList must be populated alongside neighborList in weighted mode");
                        const auto& neighbors = neighborList->get();
                        const auto& weights = weightList->get();
                        TORCH_INTERNAL_ASSERT(weights.size() == neighbors.size(),
                                              "weightList and neighborList must have the same size");
                        for (int32_t i = 0; i < static_cast<int32_t>(neighbors.size()); ++i) {
                            if (weights[i] == 0.0) {
                                continue;
                            }
                            int32_t neighborNodeId = neighbors[i];
                            double residualContribution = baseResidual * weights[i] / totalFetchedWeight;
                            if (residualContribution == 0.0) {
                                continue;
                            }
                            dstNodeTypeState.residuals[neighborNodeId] += residualContribution;

                            double threshold = _requeueThresholdFactor *
                                               static_cast<double>(getTotalDegree(neighborNodeId, dstNodeTypeId));

                            if (dstNodeTypeState.queue.find(neighborNodeId) == dstNodeTypeState.queue.end() &&
                                dstNodeTypeState.residuals[neighborNodeId] >= threshold) {
                                dstNodeTypeState.queue.insert(neighborNodeId);
                                ++_numNodesInQueue;
                                promoteToCache(neighborNodeId, dstNodeTypeId);
                            }
                        }
                    }
                }
            }
        }
    }
}

std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> PPRForwardPush::extractTopK(
    int32_t maxPprNodes) {
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    // Emit an entry for every node type, even if unreachable in this batch (empty tensors,
    // all-zero valid_counts).  This keeps the output shape consistent across batches so
    // downstream model architectures see a fixed set of PPR edge types every iteration.
    for (int32_t nodeTypeId = 0; nodeTypeId < _numNodeTypes; ++nodeTypeId) {
        std::vector<int64_t> flatIds;
        std::vector<double> flatWeights;
        std::vector<int64_t> validCounts;

        for (int32_t seedIdx = 0; seedIdx < _batchSize; ++seedIdx) {
            const auto& scores = _state[seedIdx][nodeTypeId].pprScores;
            int32_t topK = std::min(maxPprNodes, static_cast<int32_t>(scores.size()));
            if (topK > 0) {
                std::vector<std::pair<int32_t, double>> scorePairs(scores.begin(), scores.end());
                std::partial_sort(scorePairs.begin(),
                                  scorePairs.begin() + topK,
                                  scorePairs.end(),
                                  [](const auto& a, const auto& b) { return a.second > b.second; });

                for (int32_t rankIdx = 0; rankIdx < topK; ++rankIdx) {
                    flatIds.push_back(static_cast<int64_t>(scorePairs[rankIdx].first));
                    flatWeights.push_back(scorePairs[rankIdx].second);
                }
            }
            validCounts.push_back(static_cast<int64_t>(topK));
        }

        result[nodeTypeId] = {torch::tensor(flatIds, torch::kLong),
                              torch::tensor(flatWeights, torch::kDouble),
                              torch::tensor(validCounts, torch::kLong)};
    }
    return result;
}

int32_t PPRForwardPush::getTotalDegree(int32_t nodeId, int32_t nodeTypeId) const {
    TORCH_CHECK(nodeTypeId >= 0, "nodeTypeId ", nodeTypeId, " is negative, which indicates a sampler bug.");
    TORCH_CHECK(nodeTypeId < static_cast<int32_t>(_degreeTensors.size()),
                "nodeTypeId ",
                nodeTypeId,
                " out of range [0, ",
                _degreeTensors.size(),
                "). This indicates a construction bug in the sampler.");
    const auto& degreeTensor = _degreeTensors[nodeTypeId];
    if (degreeTensor.numel() == 0) {
        return 0;
    }
    TORCH_CHECK(nodeId >= 0, "Node ID ", nodeId, " is negative, which indicates a sampler bug.");
    TORCH_CHECK(nodeId < static_cast<int32_t>(degreeTensor.size(0)),
                "Node ID ",
                nodeId,
                " out of range for degree tensor of ntype_id ",
                nodeTypeId,
                " (size=",
                degreeTensor.size(0),
                "). This indicates corrupted graph data or a sampler bug.");
    if (degreeTensor.scalar_type() == torch::kInt) {
        return degreeTensor.data_ptr<int32_t>()[nodeId];
    }
    if (degreeTensor.scalar_type() == torch::kLong) {
        return static_cast<int32_t>(std::min<int64_t>(degreeTensor.data_ptr<int64_t>()[nodeId], INT32_MAX));
    }
    TORCH_CHECK(false,
                "Unsupported degree tensor dtype: ",
                degreeTensor.scalar_type(),
                ". Expected torch.int32 or torch.int64.");
    return 0; // unreachable; suppresses compiler warning
}

} // namespace gigl
