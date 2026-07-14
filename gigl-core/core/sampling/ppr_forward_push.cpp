#include "ppr_forward_push.h"

#include <torch/torch.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
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

DrainedTypedPPRQueues drainTypedPPRChannelQueues(const std::vector<PPRForwardPush*>& states,
                                                 const std::vector<int32_t>& fetchIterationCounts,
                                                 int32_t maxFetchIterations) {
    TORCH_CHECK(states.size() == fetchIterationCounts.size(),
                "Expected one fetch iteration count per PPR state, got ",
                fetchIterationCounts.size(),
                " counts for ",
                states.size(),
                " states.");

    DrainedTypedPPRQueues drained;
    std::unordered_map<int32_t, std::unordered_set<int64_t>> nodeIdsByEdgeTypeId;

    for (size_t channelIndex = 0; channelIndex < states.size(); ++channelIndex) {
        PPRForwardPush* state = states[channelIndex];
        TORCH_CHECK(state != nullptr, "PPR state pointer must not be null.");

        auto nodesByEdgeTypeId = state->drainQueue();
        if (!nodesByEdgeTypeId.has_value()) {
            continue;
        }

        drained.activeChannelIndices.push_back(static_cast<int32_t>(channelIndex));

        bool fetchBudgetRemaining = maxFetchIterations < 0 || fetchIterationCounts[channelIndex] < maxFetchIterations;
        if (!fetchBudgetRemaining) {
            continue;
        }

        std::vector<int32_t> requestedEdgeTypeIds;
        for (const auto& [edgeTypeId, nodes] : nodesByEdgeTypeId.value()) {
            if (nodes.numel() == 0) {
                continue;
            }
            TORCH_CHECK(nodes.dim() == 1, "drainQueue() must return 1-D node tensors.");
            TORCH_CHECK(nodes.scalar_type() == torch::kInt64, "drainQueue() must return int64 node tensors.");

            requestedEdgeTypeIds.push_back(edgeTypeId);
            auto contiguousNodes = nodes.contiguous();
            auto nodeAccessor = contiguousNodes.accessor<int64_t, 1>();
            auto& nodeIds = nodeIdsByEdgeTypeId[edgeTypeId];
            for (int64_t nodeIndex = 0; nodeIndex < contiguousNodes.size(0); ++nodeIndex) {
                nodeIds.insert(nodeAccessor[nodeIndex]);
            }
        }

        if (!requestedEdgeTypeIds.empty()) {
            drained.fetchChannelIndices.push_back(static_cast<int32_t>(channelIndex));
            drained.edgeTypeIdsByFetchChannel.push_back(std::move(requestedEdgeTypeIds));
        }
    }

    for (const auto& [edgeTypeId, nodeIds] : nodeIdsByEdgeTypeId) {
        std::vector<int64_t> nodeIdsToLookup(nodeIds.begin(), nodeIds.end());
        drained.unionNodesByEdgeTypeId[edgeTypeId] = torch::tensor(nodeIdsToLookup, torch::kLong);
    }
    return drained;
}

void PPRForwardPush::pushResiduals(
    const std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>& fetchedByEtypeId) {
    // Step 1: Persist fetched neighbor lists in the per-state cache. drainQueue()
    // consults this cache before requesting future lookups, so storing every
    // fetched row here avoids re-fetching a (node, edge type) pair if it re-enters
    // the frontier later in the same PPR channel.
    for (const auto& [edgeTypeId, neighborTensors] : fetchedByEtypeId) {
        const auto& nodeIdsTensor = std::get<0>(neighborTensors);
        const auto& flatNeighborIdsTensor = std::get<1>(neighborTensors);
        const auto& countsTensor = std::get<2>(neighborTensors);

        // accessor<int64_t, 1>() gives a bounds-checked, typed 1-D view into
        // each tensor's data — equivalent to iterating over a NumPy array.
        auto nodeIdsAccessor = nodeIdsTensor.accessor<int64_t, 1>();
        auto flatNeighborIdsAccessor = flatNeighborIdsTensor.accessor<int64_t, 1>();
        auto countsAccessor = countsTensor.accessor<int64_t, 1>();

        // Walk the flat neighbor list, slicing out each node's neighbors using
        // the running offset into the concatenated flat buffer.
        int64_t offset = 0;
        for (int64_t nodeIdx = 0; nodeIdx < nodeIdsTensor.size(0); ++nodeIdx) {
            auto nodeId = static_cast<int32_t>(nodeIdsAccessor[nodeIdx]);
            int64_t count = countsAccessor[nodeIdx];
            std::vector<int32_t> neighborIds(count);
            for (int64_t neighborIdx = 0; neighborIdx < count; ++neighborIdx) {
                neighborIds[neighborIdx] = static_cast<int32_t>(flatNeighborIdsAccessor[offset + neighborIdx]);
            }
            uint64_t cacheKey = packKey(nodeId, edgeTypeId);
            if (_neighborCache.find(cacheKey) == _neighborCache.end()) {
                _neighborCache.emplace(cacheKey, std::move(neighborIds));
            }
            offset += count;
        }
    }

    // Step 2: For every node that was in the queue (captured in _queuedNodes
    // by drainQueue()), apply one PPR push step:
    //   a. Absorb residual into the PPR score.
    //   b. Distribute (1-alpha) * residual equally to each neighbor.
    //   c. Enqueue any neighbor whose residual now exceeds the requeue threshold.
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

                // b. Count total cached neighbors across all edge types for
                // this source node.  We normalise by the number of neighbors we
                // actually retrieved, not the true degree, so residual is fully
                // distributed among known neighbors rather than leaking to unfetched
                // ones (which matters when num_neighbors_per_hop < true_degree).
                int32_t totalCachedNeighbors = 0;
                for (int32_t edgeTypeId : _nodeTypeToEdgeTypeIds[nodeTypeId]) {
                    auto cachedEntry = _neighborCache.find(packKey(sourceNodeId, edgeTypeId));
                    if (cachedEntry != _neighborCache.end()) {
                        totalCachedNeighbors += static_cast<int32_t>(cachedEntry->second.size());
                    }
                }
                // Two cases reach here:
                //   1. True sink node (no outgoing edges): absorbing the full residual is correct.
                //   2. Budget exhausted, no cache entry: the (1-α)·r that should flow to
                //      neighbors has nowhere to go, so it gets absorbed into src's score instead.
                //      This overstates src and understates its neighbors.  This is expected
                //      behavior when max_fetch_iterations is set, which intentionally trades
                //      theoretical PPR correctness for better throughput.
                if (totalCachedNeighbors == 0) {
                    continue;
                }

                double residualPerNeighbor =
                    (1.0 - _alpha) * sourceResidual / static_cast<double>(totalCachedNeighbors);

                for (int32_t edgeTypeId : _nodeTypeToEdgeTypeIds[nodeTypeId]) {
                    // Neighbor list for this (src, edgeTypeId) pair, borrowed from whichever
                    // map holds it.  reference_wrapper is used because std::optional cannot
                    // hold a reference directly, and we want to avoid copying the vector —
                    // the data already exists in _neighborCache and outlives this loop body.
                    // Access via neighborList->get().
                    std::optional<std::reference_wrapper<const std::vector<int32_t>>> neighborList;
                    auto cachedEntry = _neighborCache.find(packKey(sourceNodeId, edgeTypeId));
                    if (cachedEntry != _neighborCache.end()) {
                        neighborList = std::cref(cachedEntry->second);
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
                        }
                    }
                }
            }
        }
    }
}

static std::vector<std::pair<int32_t, double>> selectPPRPairsWithResidualTopUp(const SeedNodeTypeState& nodeTypeState,
                                                                               int32_t maxPPRNodes,
                                                                               int32_t maxResidualNodes,
                                                                               int32_t maxTotalNodes) {
    const auto& scores = nodeTypeState.pprScores;
    const auto higherScore = [](const auto& a, const auto& b) {
        return a.second > b.second;
    };

    int32_t totalNodeLimit = maxPPRNodes + maxResidualNodes;
    if (maxTotalNodes >= 0) {
        totalNodeLimit = maxTotalNodes;
    }

    const int32_t topK = std::min(maxPPRNodes, static_cast<int32_t>(scores.size()));
    const int32_t residualTopUpBudget = std::min(maxResidualNodes, std::max<int32_t>(0, totalNodeLimit - topK));
    std::vector<std::pair<int32_t, double>> selectedPairs;
    selectedPairs.reserve(static_cast<size_t>(topK + residualTopUpBudget));
    std::unordered_set<int32_t> selectedPPRNodeIds;
    if (topK > 0) {
        if (residualTopUpBudget > 0) {
            selectedPPRNodeIds.reserve(static_cast<size_t>(topK));
        }
        std::vector<std::pair<int32_t, double>> scorePairs(scores.begin(), scores.end());
        // Selection is intentionally two-phase: finalized nodes are selected
        // first by raw PPR score, and residual candidates only compete for
        // the remaining budget.
        if (maxResidualNodes > 0) {
            // The final emitted order is sorted by ppr_score + residual after
            // top-up candidates are selected, so this pass only needs to
            // partition out the raw-PPR top K.
            if (topK < static_cast<int32_t>(scorePairs.size())) {
                std::nth_element(scorePairs.begin(),
                                 scorePairs.begin() + topK,
                                 scorePairs.end(),
                                 higherScore);
            }
        } else {
            std::partial_sort(scorePairs.begin(),
                              scorePairs.begin() + topK,
                              scorePairs.end(),
                              higherScore);
        }

        for (int32_t rankIdx = 0; rankIdx < topK; ++rankIdx) {
            int32_t nodeId = scorePairs[rankIdx].first;
            double outputScore = scorePairs[rankIdx].second;
            if (maxResidualNodes > 0) {
                auto residualIter = nodeTypeState.residuals.find(nodeId);
                if (residualIter != nodeTypeState.residuals.end()) {
                    outputScore += residualIter->second;
                }
            }
            selectedPairs.emplace_back(nodeId, outputScore);
            if (residualTopUpBudget > 0) {
                selectedPPRNodeIds.insert(nodeId);
            }
        }
    }

    if (residualTopUpBudget > 0) {
        std::vector<std::pair<int32_t, double>> residualPairs;
        residualPairs.reserve(nodeTypeState.residuals.size());
        for (const auto& [nodeId, residual] : nodeTypeState.residuals) {
            if (residual <= 0.0 || selectedPPRNodeIds.find(nodeId) != selectedPPRNodeIds.end()) {
                continue;
            }

            auto scoreIter = scores.find(nodeId);
            double pprScore = (scoreIter != scores.end()) ? scoreIter->second : 0.0;
            double outputScore = pprScore + residual;
            residualPairs.emplace_back(nodeId, outputScore);
        }

        const int32_t residualTopK = std::min(residualTopUpBudget, static_cast<int32_t>(residualPairs.size()));
        if (residualTopK > 0) {
            // Residual candidates only need selection here; selected finalized
            // and residual rows are sorted together below.
            if (residualTopK < static_cast<int32_t>(residualPairs.size())) {
                std::nth_element(residualPairs.begin(),
                                 residualPairs.begin() + residualTopK,
                                 residualPairs.end(),
                                 higherScore);
            }

            for (int32_t rankIdx = 0; rankIdx < residualTopK; ++rankIdx) {
                selectedPairs.emplace_back(residualPairs[rankIdx].first, residualPairs[rankIdx].second);
            }
        }
    }

    if (maxResidualNodes > 0 && selectedPairs.size() > 1) {
        std::sort(selectedPairs.begin(), selectedPairs.end(), higherScore);
    }
    return selectedPairs;
}

static double clampTypedPPRScore(double score) {
    if (!std::isfinite(score)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::min(std::max(score, 0.0), 1.0);
}

static void addTypedPPRSeedCandidates(std::unordered_map<int32_t, std::vector<double>>& seedScores,
                                      std::vector<std::pair<int32_t, double>>& seedChannelCandidates,
                                      const std::vector<std::pair<int32_t, double>>& nodesAndScores,
                                      double maxScore,
                                      int32_t channelIndex,
                                      int32_t numEdgeAttrFeatures,
                                      int32_t numChannels) {
    for (const auto& [nodeId, rawScore] : nodesAndScores) {
        double score = clampTypedPPRScore(rawScore);
        if (!std::isfinite(score)) {
            continue;
        }
        double calibratedScore = maxScore > 0.0 ? score / maxScore : 0.0;
        auto scoreIter = seedScores.find(nodeId);
        if (scoreIter == seedScores.end()) {
            scoreIter = seedScores.emplace(nodeId, std::vector<double>(numEdgeAttrFeatures, 0.0)).first;
        }
        auto& scoreFeatures = scoreIter->second;
        scoreFeatures[0] = std::max(scoreFeatures[0], calibratedScore);
        int32_t channelScoreIndex = 1 + channelIndex;
        int32_t channelPresenceIndex = 1 + numChannels + channelIndex;
        scoreFeatures[channelScoreIndex] = std::max(scoreFeatures[channelScoreIndex], calibratedScore);
        scoreFeatures[channelPresenceIndex] = 1.0;
        seedChannelCandidates.emplace_back(nodeId, calibratedScore);
    }
}

static std::vector<int32_t> selectTypedPPRNodeIds(
    const std::unordered_map<int32_t, std::vector<double>>& seedScores,
    std::vector<std::vector<std::pair<int32_t, double>>> candidatesByChannel,
    const std::vector<int32_t>& channelQuotas,
    int32_t maxPPRNodes) {
    struct GlobalCandidate {
        double bestCalibratedScore;
        double channelCalibratedScore;
        int32_t channelIndex;
        int32_t nodeId;
    };

    std::vector<GlobalCandidate> globalCandidates;
    for (int32_t channelIndex = 0; channelIndex < static_cast<int32_t>(candidatesByChannel.size()); ++channelIndex) {
        auto& candidates = candidatesByChannel[channelIndex];
        std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) {
                return a.second > b.second;
            }
            return a.first < b.first;
        });

        int32_t channelQuota = channelQuotas[channelIndex];
        int32_t numCandidates = std::min(channelQuota, static_cast<int32_t>(candidates.size()));
        for (int32_t candidateIndex = 0; candidateIndex < numCandidates; ++candidateIndex) {
            int32_t nodeId = candidates[candidateIndex].first;
            auto scoreIter = seedScores.find(nodeId);
            if (scoreIter == seedScores.end()) {
                continue;
            }
            globalCandidates.push_back({scoreIter->second[0], candidates[candidateIndex].second, channelIndex, nodeId});
        }
    }

    std::sort(globalCandidates.begin(), globalCandidates.end(), [](const auto& a, const auto& b) {
        if (a.bestCalibratedScore != b.bestCalibratedScore) {
            return a.bestCalibratedScore > b.bestCalibratedScore;
        }
        if (a.channelCalibratedScore != b.channelCalibratedScore) {
            return a.channelCalibratedScore > b.channelCalibratedScore;
        }
        if (a.channelIndex != b.channelIndex) {
            return a.channelIndex < b.channelIndex;
        }
        return a.nodeId < b.nodeId;
    });

    std::unordered_set<int32_t> selectedNodeIds;
    std::vector<int32_t> selectedNodes;
    int32_t selectedReserveSize = std::min(maxPPRNodes, static_cast<int32_t>(globalCandidates.size()));
    selectedNodeIds.reserve(static_cast<size_t>(selectedReserveSize));
    selectedNodes.reserve(static_cast<size_t>(selectedReserveSize));
    for (const auto& candidate : globalCandidates) {
        if (static_cast<int32_t>(selectedNodes.size()) >= maxPPRNodes) {
            break;
        }
        if (selectedNodeIds.find(candidate.nodeId) != selectedNodeIds.end()) {
            continue;
        }
        selectedNodeIds.insert(candidate.nodeId);
        selectedNodes.push_back(candidate.nodeId);
    }
    return selectedNodes;
}

std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> PPRForwardPush::
    extractTopKWithResidualTopUp(int32_t maxPPRNodes, bool enableResidualTopUp) {
    TORCH_CHECK(maxPPRNodes >= 0, "maxPPRNodes must be non-negative, got ", maxPPRNodes, ".");
    int32_t residualTopUpNodes = enableResidualTopUp ? maxPPRNodes : 0;

    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    // Emit an entry for every node type, even if unreachable in this batch (empty tensors,
    // all-zero valid_counts).  This keeps the output shape consistent across batches so
    // downstream model architectures see a fixed set of PPR edge types every iteration.
    for (int32_t nodeTypeId = 0; nodeTypeId < _numNodeTypes; ++nodeTypeId) {
        std::vector<int64_t> flatIds;
        std::vector<double> flatWeights;
        std::vector<int64_t> validCounts;

        for (int32_t seedIdx = 0; seedIdx < _batchSize; ++seedIdx) {
            const auto& nodeTypeState = _state[seedIdx][nodeTypeId];
            auto selectedPairs =
                selectPPRPairsWithResidualTopUp(nodeTypeState, maxPPRNodes, residualTopUpNodes, maxPPRNodes);
            for (const auto& [nodeId, score] : selectedPairs) {
                flatIds.push_back(static_cast<int64_t>(nodeId));
                flatWeights.push_back(score);
            }
            validCounts.push_back(static_cast<int64_t>(selectedPairs.size()));
        }

        result[nodeTypeId] = {torch::tensor(flatIds, torch::kLong),
                              torch::tensor(flatWeights, torch::kDouble),
                              torch::tensor(validCounts, torch::kLong)};
    }
    return result;
}

std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> extractTypedTopKWithResidualTopUp(
    const std::vector<PPRForwardPush*>& states,
    const std::vector<int32_t>& channelQuotas,
    int32_t maxPPRNodes,
    bool enableResidualTopUp) {
    TORCH_CHECK(maxPPRNodes >= 0, "maxPPRNodes must be non-negative, got ", maxPPRNodes, ".");
    TORCH_CHECK(!states.empty(), "Typed PPR extraction requires at least one channel state.");
    TORCH_CHECK(states.size() == channelQuotas.size(),
                "Expected one channel quota per typed PPR state, got ",
                channelQuotas.size(),
                " quotas for ",
                states.size(),
                " states.");

    const auto* firstState = states.front();
    TORCH_CHECK(firstState != nullptr, "PPR state pointer must not be null.");
    int32_t batchSize = firstState->_batchSize;
    int32_t numNodeTypes = firstState->_numNodeTypes;
    int32_t numChannels = static_cast<int32_t>(states.size());
    int32_t numEdgeAttrFeatures = 1 + (2 * numChannels);
    int32_t residualTopUpNodes = enableResidualTopUp ? maxPPRNodes : 0;

    for (int32_t channelIndex = 0; channelIndex < numChannels; ++channelIndex) {
        TORCH_CHECK(channelQuotas[channelIndex] >= 0,
                    "channelQuotas must be non-negative, got ",
                    channelQuotas[channelIndex],
                    " at channel ",
                    channelIndex,
                    ".");
        TORCH_CHECK(states[channelIndex] != nullptr, "PPR state pointer must not be null.");
        TORCH_CHECK(states[channelIndex]->_batchSize == batchSize,
                    "All typed PPR channel states must have the same batch size.");
        TORCH_CHECK(states[channelIndex]->_numNodeTypes == numNodeTypes,
                    "All typed PPR channel states must have the same node-type count.");
    }

    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    std::vector<int32_t> topUpChannelQuotas(static_cast<size_t>(numChannels), maxPPRNodes);

    for (int32_t nodeTypeId = 0; nodeTypeId < numNodeTypes; ++nodeTypeId) {
        std::vector<int64_t> flatIds;
        std::vector<double> flatFeatureValues;
        std::vector<int64_t> validCounts;

        for (int32_t seedIdx = 0; seedIdx < batchSize; ++seedIdx) {
            std::unordered_map<int32_t, std::vector<double>> baseScores;
            std::unordered_map<int32_t, std::vector<double>> extendedScores;
            std::vector<std::vector<std::pair<int32_t, double>>> baseCandidatesByChannel(
                static_cast<size_t>(numChannels));
            std::vector<std::vector<std::pair<int32_t, double>>> extendedCandidatesByChannel(
                static_cast<size_t>(numChannels));

            for (int32_t channelIndex = 0; channelIndex < numChannels; ++channelIndex) {
                const auto& nodeTypeState = states[channelIndex]->_state[seedIdx][nodeTypeId];
                int32_t channelPPRLimit = std::min(channelQuotas[channelIndex], maxPPRNodes);
                auto selectedPairs =
                    selectPPRPairsWithResidualTopUp(nodeTypeState, channelPPRLimit, residualTopUpNodes, maxPPRNodes);

                double extendedMaxScore = 0.0;
                std::vector<std::pair<int32_t, double>> baseNodesAndScores;
                std::vector<std::pair<int32_t, double>> extendedNodesAndScores;
                baseNodesAndScores.reserve(static_cast<size_t>(
                    std::min(channelQuotas[channelIndex], static_cast<int32_t>(selectedPairs.size()))));
                extendedNodesAndScores.reserve(selectedPairs.size());
                for (int32_t candidateIndex = 0; candidateIndex < static_cast<int32_t>(selectedPairs.size());
                     ++candidateIndex) {
                    double score = clampTypedPPRScore(selectedPairs[candidateIndex].second);
                    if (!std::isfinite(score)) {
                        continue;
                    }
                    extendedNodesAndScores.emplace_back(selectedPairs[candidateIndex].first, score);
                    extendedMaxScore = std::max(extendedMaxScore, score);
                    if (candidateIndex < channelQuotas[channelIndex]) {
                        baseNodesAndScores.emplace_back(selectedPairs[candidateIndex].first, score);
                    }
                }

                addTypedPPRSeedCandidates(baseScores,
                                          baseCandidatesByChannel[channelIndex],
                                          baseNodesAndScores,
                                          extendedMaxScore,
                                          channelIndex,
                                          numEdgeAttrFeatures,
                                          numChannels);
                addTypedPPRSeedCandidates(extendedScores,
                                          extendedCandidatesByChannel[channelIndex],
                                          extendedNodesAndScores,
                                          extendedMaxScore,
                                          channelIndex,
                                          numEdgeAttrFeatures,
                                          numChannels);
            }

            std::vector<int32_t> selectedNodes;
            std::unordered_set<int32_t> selectedNodeIds;
            auto baseSelectedNodes =
                selectTypedPPRNodeIds(baseScores, baseCandidatesByChannel, channelQuotas, maxPPRNodes);
            selectedNodes.reserve(static_cast<size_t>(maxPPRNodes));
            selectedNodeIds.reserve(static_cast<size_t>(maxPPRNodes));
            for (int32_t nodeId : baseSelectedNodes) {
                if (static_cast<int32_t>(selectedNodes.size()) >= maxPPRNodes) {
                    break;
                }
                selectedNodes.push_back(nodeId);
                selectedNodeIds.insert(nodeId);
                flatIds.push_back(static_cast<int64_t>(nodeId));
                const auto& features = baseScores.at(nodeId);
                flatFeatureValues.insert(flatFeatureValues.end(), features.begin(), features.end());
            }

            if (static_cast<int32_t>(selectedNodes.size()) < maxPPRNodes) {
                auto extendedSelectedNodes =
                    selectTypedPPRNodeIds(extendedScores, extendedCandidatesByChannel, topUpChannelQuotas, maxPPRNodes);
                for (int32_t nodeId : extendedSelectedNodes) {
                    if (static_cast<int32_t>(selectedNodes.size()) >= maxPPRNodes) {
                        break;
                    }
                    if (selectedNodeIds.find(nodeId) != selectedNodeIds.end()) {
                        continue;
                    }
                    selectedNodes.push_back(nodeId);
                    selectedNodeIds.insert(nodeId);
                    flatIds.push_back(static_cast<int64_t>(nodeId));
                    const auto& features = extendedScores.at(nodeId);
                    flatFeatureValues.insert(flatFeatureValues.end(), features.begin(), features.end());
                }
            }

            validCounts.push_back(static_cast<int64_t>(selectedNodes.size()));
        }

        auto flatWeights =
            torch::tensor(flatFeatureValues, torch::kDouble)
                .reshape({static_cast<int64_t>(flatIds.size()), static_cast<int64_t>(numEdgeAttrFeatures)});
        result[nodeTypeId] = {
            torch::tensor(flatIds, torch::kLong),
            flatWeights,
            torch::tensor(validCounts, torch::kLong),
        };
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
