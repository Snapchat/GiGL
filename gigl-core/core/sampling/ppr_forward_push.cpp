#include "ppr_forward_push.h"

#include <torch/torch.h>

#include <algorithm>
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

TypedPPRQueueDrainResult drainTypedPPRChannelQueues(const std::vector<PPRForwardPush*>& states,
                                                    const std::vector<int32_t>& fetchIterationCounts,
                                                    int32_t maxFetchIterations) {
    TORCH_CHECK(states.size() == fetchIterationCounts.size(),
                "Expected one fetch iteration count per PPR state, got ",
                fetchIterationCounts.size(),
                " counts for ",
                states.size(),
                " states.");

    TypedPPRQueueDrainResult queueDrainResult;
    std::unordered_map<int32_t, std::unordered_set<int64_t>> unionedFrontierNodeIdsByEdgeTypeId;

    // TODO: If typed queue draining becomes a measured blocker, evaluate
    // parallelizing this channel loop with per-thread frontier maps and a
    // deterministic merge into queueDrainResult.
    for (size_t channelIndex = 0; channelIndex < states.size(); ++channelIndex) {
        PPRForwardPush* state = states[channelIndex];

        auto channelFrontierByEdgeTypeId = state->drainQueue();
        if (!channelFrontierByEdgeTypeId.has_value()) {
            continue;
        }

        queueDrainResult.drainedChannelIndices.push_back(static_cast<int32_t>(channelIndex));

        bool fetchBudgetRemaining = maxFetchIterations < 0 || fetchIterationCounts[channelIndex] < maxFetchIterations;
        if (!fetchBudgetRemaining) {
            continue;
        }

        std::vector<int32_t> requestedEdgeTypeIds;
        for (const auto& [edgeTypeId, nodes] : channelFrontierByEdgeTypeId.value()) {
            requestedEdgeTypeIds.push_back(edgeTypeId);
            auto nodeAccessor = nodes.accessor<int64_t, 1>();
            auto& unionedFrontierNodeIds = unionedFrontierNodeIdsByEdgeTypeId[edgeTypeId];
            for (int64_t nodeIndex = 0; nodeIndex < nodes.size(0); ++nodeIndex) {
                unionedFrontierNodeIds.insert(nodeAccessor[nodeIndex]);
            }
        }

        if (!requestedEdgeTypeIds.empty()) {
            queueDrainResult.fetchChannelIndices.push_back(static_cast<int32_t>(channelIndex));
            queueDrainResult.edgeTypeIdsByFetchChannel.push_back(std::move(requestedEdgeTypeIds));
        }
    }

    for (const auto& [edgeTypeId, unionedFrontierNodeIds] : unionedFrontierNodeIdsByEdgeTypeId) {
        std::vector<int64_t> nodeIdsToLookup(unionedFrontierNodeIds.begin(), unionedFrontierNodeIds.end());
        queueDrainResult.unionedNodeIdsByEdgeTypeId[edgeTypeId] = torch::tensor(nodeIdsToLookup, torch::kLong);
    }
    return queueDrainResult;
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

// Helper function for selecting one seed/node-type's finalized PPR rows.
//
// Inputs:
//   nodeTypeState: finalized PPR scores and residuals for one seed/node type.
//   finalizedPPRNodeLimit: maximum finalized-PPR rows to select before top-up.
//
// Expected output: (node_id, raw_ppr_score) pairs selected by raw PPR score.
// The order is unspecified; callers that emit these directly should sort the
// returned vector before writing output tensors.
static std::vector<std::pair<int32_t, double>> selectFinalizedPPRPairs(const SeedNodeTypeState& nodeTypeState,
                                                                       int32_t finalizedPPRNodeLimit) {
    const auto& scores = nodeTypeState.pprScores;

    const int32_t numReturnedPairs = std::min(finalizedPPRNodeLimit, static_cast<int32_t>(scores.size()));
    std::vector<std::pair<int32_t, double>> selectedPairs;
    selectedPairs.reserve(static_cast<size_t>(numReturnedPairs));
    if (numReturnedPairs > 0) {
        std::vector<std::pair<int32_t, double>> scorePairs(scores.begin(), scores.end());
        if (numReturnedPairs < static_cast<int32_t>(scorePairs.size())) {
            std::nth_element(scorePairs.begin(),
                             scorePairs.begin() + numReturnedPairs,
                             scorePairs.end(),
                             [](const auto& a, const auto& b) { return a.second > b.second; });
        }

        for (int32_t rankIdx = 0; rankIdx < numReturnedPairs; ++rankIdx) {
            selectedPairs.emplace_back(scorePairs[rankIdx].first, scorePairs[rankIdx].second);
        }
    }

    return selectedPairs;
}

// Helper function for extending one seed/node-type's selected PPR rows with
// residual top-up candidates.
//
// Inputs:
//   nodeTypeState: finalized PPR scores and residuals for one seed/node type.
//   selectedPairs: mutable finalized-PPR rows already selected for this seed.
//   sequenceLength: maximum total rows after residual top-up.
//
// Expected output: selectedPairs has up to sequenceLength rows after appending
// highest-scoring residual candidates that are not already selected. This helper
// does not sort selectedPairs; callers sort only when their output needs it.
static void appendResidualTopUpPairs(const SeedNodeTypeState& nodeTypeState,
                                     std::vector<std::pair<int32_t, double>>& selectedPairs,
                                     int32_t sequenceLength) {
    const int32_t residualTopUpBudget =
        std::max<int32_t>(0, sequenceLength - static_cast<int32_t>(selectedPairs.size()));
    if (residualTopUpBudget > 0) {
        const std::unordered_map<int32_t, double>& pprScoresByNodeId = nodeTypeState.pprScores;
        std::unordered_set<int32_t> selectedPPRNodeIds;
        selectedPPRNodeIds.reserve(selectedPairs.size());
        for (const auto& selectedPair : selectedPairs) {
            selectedPPRNodeIds.insert(selectedPair.first);
        }

        std::vector<std::pair<int32_t, double>> residualPairs;
        residualPairs.reserve(nodeTypeState.residuals.size());
        for (const auto& [nodeId, residual] : nodeTypeState.residuals) {
            // Forward push residuals are non-negative in normal operation. Pushed
            // nodes remain in the map with zero residual, so skip drained entries
            // and any unexpected non-positive values.
            if (residual <= 0.0 || selectedPPRNodeIds.find(nodeId) != selectedPPRNodeIds.end()) {
                continue;
            }

            std::unordered_map<int32_t, double>::const_iterator pprScoreIter = pprScoresByNodeId.find(nodeId);
            double pprScore = (pprScoreIter != pprScoresByNodeId.end()) ? pprScoreIter->second : 0.0;
            double outputScore = pprScore + residual;
            residualPairs.emplace_back(nodeId, outputScore);
        }

        const int32_t residualTopK = std::min(residualTopUpBudget, static_cast<int32_t>(residualPairs.size()));
        if (residualTopK > 0) {
            if (residualTopK < static_cast<int32_t>(residualPairs.size())) {
                std::nth_element(residualPairs.begin(),
                                 residualPairs.begin() + residualTopK,
                                 residualPairs.end(),
                                 [](const auto& a, const auto& b) { return a.second > b.second; });
            }

            for (int32_t rankIdx = 0; rankIdx < residualTopK; ++rankIdx) {
                selectedPairs.emplace_back(residualPairs[rankIdx].first, residualPairs[rankIdx].second);
            }
        }
    }
}

// Helper function for moving finalized-PPR rows onto the same score scale as
// residual top-up rows.
//
// Inputs:
//   nodeTypeState: residual table for one seed/node type.
//   selectedPairs: mutable finalized-PPR rows with raw PPR scores.
//
// Expected output: selectedPairs scores are updated in-place to
// ppr_score + residual(node) when residual mass exists for that node.
static void addResidualMassToPPRPairs(const SeedNodeTypeState& nodeTypeState,
                                      std::vector<std::pair<int32_t, double>>& selectedPairs) {
    for (auto& [nodeId, score] : selectedPairs) {
        auto residualIter = nodeTypeState.residuals.find(nodeId);
        if (residualIter != nodeTypeState.residuals.end()) {
            score += residualIter->second;
        }
    }
}

// Helper function for adding one channel's extracted PPR candidates into a
// selection-only candidate list.
//
// This intentionally does not write edge_attr features. Typed extraction uses
// it for the base quota-biased pass, where residual top-up must not affect which
// nodes consume per-channel quota slots.
//
// Inputs:
//   channelSelectionCandidates: mutable sortable candidates for this channel quota.
//   nodesAndScores: extracted (node_id, score) candidates for one channel.
//   maxScore: largest score in the channel, used to calibrate scores to [0, 1].
//
// Expected output: channelSelectionCandidates contains this channel's calibrated
// candidates for later quota selection.
static void addTypedPPRChannelCandidates(std::vector<std::pair<int32_t, double>>& channelSelectionCandidates,
                                         const std::vector<std::pair<int32_t, double>>& nodesAndScores,
                                         double maxScore) {
    for (const auto& [nodeId, score] : nodesAndScores) {
        double calibratedScore = maxScore > 0.0 ? score / maxScore : 0.0;
        channelSelectionCandidates.emplace_back(nodeId, calibratedScore);
    }
}

// Helper function for adding one channel's extracted PPR candidates into the
// emitted typed feature table and a fill-pass candidate list.
//
// Unlike addTypedPPRChannelCandidates, this helper owns the output view: it
// populates the edge_attr row for every node it sees. When residual top-up is
// enabled, callers pass residual-aware scores here so base-selected and top-up
// rows are emitted on the same ppr_score + residual scale.
//
// Inputs:
//   outputScoresByNodeId: mutable map from node ID to emitted edge_attr features.
//   channelOutputCandidates: mutable sortable candidates for the fill pass.
//   nodesAndScores: extracted (node_id, score) candidates for one channel.
//   maxScore: largest score in the channel, used to calibrate scores to [0, 1].
//   channelIndex: index of the typed PPR channel being added.
//   numChannels: total typed PPR channels, used to derive feature width.
//
// Expected output: outputScoresByNodeId has this channel's score/presence
// features merged in, and channelOutputCandidates contains this channel's
// calibrated candidates.
static void addTypedPPRSeedFeaturesAndCandidates(std::unordered_map<int32_t, std::vector<double>>& outputScoresByNodeId,
                                                 std::vector<std::pair<int32_t, double>>& channelOutputCandidates,
                                                 const std::vector<std::pair<int32_t, double>>& nodesAndScores,
                                                 double maxScore,
                                                 int32_t channelIndex,
                                                 int32_t numChannels) {
    // Typed edge_attr rows store one best score across channels, followed by
    // one score and one presence bit per channel.
    int32_t numEdgeAttrFeatures = 1 + (2 * numChannels);
    for (const auto& [nodeId, score] : nodesAndScores) {
        double calibratedScore = maxScore > 0.0 ? score / maxScore : 0.0;
        auto scoreIter = outputScoresByNodeId.find(nodeId);
        if (scoreIter == outputScoresByNodeId.end()) {
            scoreIter = outputScoresByNodeId.emplace(nodeId, std::vector<double>(numEdgeAttrFeatures, 0.0)).first;
        }
        auto& scoreFeatures = scoreIter->second;

        // Feature layout:
        // [best_calibrated_score, per-channel scores..., channel presence bits...].
        scoreFeatures[0] = std::max(scoreFeatures[0], calibratedScore);
        int32_t channelScoreIndex = 1 + channelIndex;
        int32_t channelPresenceIndex = 1 + numChannels + channelIndex;

        // Record this node's score for the current channel and mark that the
        // channel reached the node. A node may be seen multiple times in the
        // same channel view, so keep the strongest calibrated score.
        scoreFeatures[channelScoreIndex] = std::max(scoreFeatures[channelScoreIndex], calibratedScore);
        scoreFeatures[channelPresenceIndex] = 1.0;

        // Keep a per-channel sortable candidate list so channel quotas can be
        // applied before the global cross-channel dedup pass.
        channelOutputCandidates.emplace_back(nodeId, calibratedScore);
    }
}

// Helper function for applying typed channel quotas and cross-channel dedup.
//
// Inputs:
//   candidatesByChannel: mutable (node_id, calibrated_score) candidates per channel.
//   channelQuotas: maximum number of candidates to consider from each channel.
//   maxPPRNodes: maximum number of deduplicated node IDs to return for a seed.
//
// Expected output: node IDs selected after per-channel quota filtering, then
// globally ranked by best calibrated score. Tie breakers keep output deterministic.
static std::vector<int32_t> selectTypedPPRNodeIds(
    std::vector<std::vector<std::pair<int32_t, double>>>& candidatesByChannel,
    const std::vector<int32_t>& channelQuotas,
    int32_t maxPPRNodes) {
    struct GlobalCandidate {
        double bestCalibratedScore;
        double channelCalibratedScore;
        int32_t channelIndex;
        int32_t nodeId;
    };

    // Per-channel ordering is by the channel-local score. Global ordering is by
    // the node's best score across all channels, then by the candidate's own
    // channel score and stable IDs for deterministic ties.
    const auto higherScorePair = [](const auto& a, const auto& b) {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first < b.first;
    };
    const auto higherGlobalCandidate = [](const auto& a, const auto& b) {
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
    };

    // Bound reserve sizes by the number of rows that can survive per-channel
    // quotas. Current callers already cap channels to these limits, but this
    // keeps the helper efficient if a future caller passes larger vectors.
    size_t candidateRowReserveSize = 0;
    for (int32_t channelIndex = 0; channelIndex < static_cast<int32_t>(candidatesByChannel.size()); ++channelIndex) {
        candidateRowReserveSize += static_cast<size_t>(
            std::min(channelQuotas[channelIndex], static_cast<int32_t>(candidatesByChannel[channelIndex].size())));
    }

    // First apply per-channel quotas. candidateRows may still contain the same
    // node multiple times if multiple channels reached it, while
    // bestCalibratedScoreByNodeId tracks the cross-channel score used for final
    // ranking.
    std::unordered_map<int32_t, double> bestCalibratedScoreByNodeId;
    bestCalibratedScoreByNodeId.reserve(candidateRowReserveSize);
    std::vector<GlobalCandidate> candidateRows;
    candidateRows.reserve(candidateRowReserveSize);
    for (int32_t channelIndex = 0; channelIndex < static_cast<int32_t>(candidatesByChannel.size()); ++channelIndex) {
        auto& candidates = candidatesByChannel[channelIndex];
        int32_t channelQuota = channelQuotas[channelIndex];
        int32_t numCandidates = std::min(channelQuota, static_cast<int32_t>(candidates.size()));
        if (numCandidates <= 0) {
            continue;
        }

        // Only partition when a channel has more rows than its quota. We do not
        // need a sorted per-channel prefix; we only need the top quota rows
        // before the global cross-channel rank.
        if (numCandidates < static_cast<int32_t>(candidates.size())) {
            std::nth_element(candidates.begin(), candidates.begin() + numCandidates, candidates.end(), higherScorePair);
        }

        for (int32_t candidateIndex = 0; candidateIndex < numCandidates; ++candidateIndex) {
            int32_t nodeId = candidates[candidateIndex].first;
            double calibratedScore = candidates[candidateIndex].second;
            candidateRows.push_back({0.0, calibratedScore, channelIndex, nodeId});
            auto [bestScoreIter, inserted] = bestCalibratedScoreByNodeId.emplace(nodeId, calibratedScore);
            if (!inserted) {
                bestScoreIter->second = std::max(bestScoreIter->second, calibratedScore);
            }
        }
    }

    // Collapse duplicate node IDs before global ranking. If the same node is
    // present through multiple channels, keep the row that would sort highest
    // under the final comparator; its bestCalibratedScore is still the best
    // score observed for that node across every quota-surviving channel.
    std::unordered_map<int32_t, GlobalCandidate> bestCandidateByNodeId;
    bestCandidateByNodeId.reserve(bestCalibratedScoreByNodeId.size());
    for (auto candidate : candidateRows) {
        candidate.bestCalibratedScore = bestCalibratedScoreByNodeId.at(candidate.nodeId);
        auto [candidateIter, inserted] = bestCandidateByNodeId.emplace(candidate.nodeId, candidate);
        if (!inserted && higherGlobalCandidate(candidate, candidateIter->second)) {
            candidateIter->second = candidate;
        }
    }

    // Move unique candidates into a vector so the final top-k ranking can use
    // partial_sort. This avoids sorting the full candidate set when
    // maxPPRNodes is smaller than the number of unique channel candidates.
    std::vector<GlobalCandidate> globalCandidates;
    globalCandidates.reserve(bestCandidateByNodeId.size());
    for (const auto& candidateEntry : bestCandidateByNodeId) {
        globalCandidates.push_back(candidateEntry.second);
    }

    int32_t selectedReserveSize = std::min(maxPPRNodes, static_cast<int32_t>(globalCandidates.size()));
    if (selectedReserveSize < static_cast<int32_t>(globalCandidates.size())) {
        std::partial_sort(globalCandidates.begin(),
                          globalCandidates.begin() + selectedReserveSize,
                          globalCandidates.end(),
                          higherGlobalCandidate);
    } else if (globalCandidates.size() > 1) {
        std::sort(globalCandidates.begin(), globalCandidates.end(), higherGlobalCandidate);
    }

    // globalCandidates is already ranked through selectedReserveSize, so emit
    // only node IDs in final order. Dedup already happened above.
    std::vector<int32_t> selectedNodes;
    selectedNodes.reserve(static_cast<size_t>(selectedReserveSize));
    for (int32_t candidateIndex = 0; candidateIndex < selectedReserveSize; ++candidateIndex) {
        selectedNodes.push_back(globalCandidates[candidateIndex].nodeId);
    }
    return selectedNodes;
}

std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> PPRForwardPush::
    extractTopKWithResidualTopUp(int32_t maxPPRNodes, bool enableResidualTopUp) {
    TORCH_CHECK(maxPPRNodes >= 0, "maxPPRNodes must be non-negative, got ", maxPPRNodes, ".");

    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> extractedPPRByNodeTypeId;
    // Emit an entry for every node type, even if unreachable in this batch (empty tensors,
    // all-zero valid_counts).  This keeps the output shape consistent across batches so
    // downstream model architectures see a fixed set of PPR edge types every iteration.
    for (int32_t nodeTypeId = 0; nodeTypeId < _numNodeTypes; ++nodeTypeId) {
        std::vector<int64_t> flatIds;
        std::vector<double> flatWeights;
        std::vector<int64_t> validCounts;

        for (int32_t seedIdx = 0; seedIdx < _batchSize; ++seedIdx) {
            const auto& nodeTypeState = _state[seedIdx][nodeTypeId];
            auto selectedPairs = selectFinalizedPPRPairs(nodeTypeState, maxPPRNodes);
            if (enableResidualTopUp) {
                addResidualMassToPPRPairs(nodeTypeState, selectedPairs);
                appendResidualTopUpPairs(nodeTypeState, selectedPairs, maxPPRNodes);
            }

            // The selection helpers use nth_element, which selects the right rows
            // but does not order them. Sort the selected rows once to preserve the
            // emitted ordering contract. With residual top-up enabled, this matches
            // the previous behavior: selected finalized and top-up rows are ordered
            // together by emitted score, so top-up rows may interleave after selection.
            if (selectedPairs.size() > 1) {
                std::sort(selectedPairs.begin(), selectedPairs.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });
            }

            for (const auto& [nodeId, score] : selectedPairs) {
                flatIds.push_back(static_cast<int64_t>(nodeId));
                flatWeights.push_back(score);
            }
            validCounts.push_back(static_cast<int64_t>(selectedPairs.size()));
        }

        extractedPPRByNodeTypeId[nodeTypeId] = {torch::tensor(flatIds, torch::kLong),
                                                torch::tensor(flatWeights, torch::kDouble),
                                                torch::tensor(validCounts, torch::kLong)};
    }
    return extractedPPRByNodeTypeId;
}

std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> extractTypedTopKWithResidualTopUp(
    const std::vector<PPRForwardPush*>& states,
    const std::vector<int32_t>& channelQuotas,
    int32_t maxPPRNodes,
    bool enableResidualTopUp) {
    const auto* firstState = states.front();
    int32_t batchSize = firstState->_batchSize;
    int32_t numNodeTypes = firstState->_numNodeTypes;
    int32_t numChannels = static_cast<int32_t>(states.size());
    // Typed edge_attr rows store one best score across channels, followed by
    // one score and one presence bit per channel.
    int32_t numEdgeAttrFeatures = 1 + (2 * numChannels);

    // Pre-size the per-seed feature map below. Output features may receive up
    // to maxPPRNodes candidates per channel when residual top-up is on.
    size_t outputCandidateReserveSize = 0;
    for (int32_t channelIndex = 0; channelIndex < numChannels; ++channelIndex) {
        int32_t channelPPRNodeBudget = std::min(channelQuotas[channelIndex], maxPPRNodes);
        outputCandidateReserveSize += static_cast<size_t>(enableResidualTopUp ? maxPPRNodes : channelPPRNodeBudget);
    }

    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> extractedTypedPPRByNodeTypeId;
    std::vector<int32_t> topUpChannelQuotas(static_cast<size_t>(numChannels), maxPPRNodes);

    for (int32_t nodeTypeId = 0; nodeTypeId < numNodeTypes; ++nodeTypeId) {
        std::vector<int64_t> flatIds;
        std::vector<double> flatFeatureValues;
        std::vector<int64_t> validCounts;

        for (int32_t seedIdx = 0; seedIdx < batchSize; ++seedIdx) {
            std::unordered_map<int32_t, std::vector<double>> outputScores;
            outputScores.reserve(outputCandidateReserveSize);
            // Keep the two typed views separate:
            //   - baseCandidatesByChannel is selection-only. It contains raw
            //     finalized PPR rows and drives the quota-biased first pass.
            //   - outputScores/outputCandidatesByChannel is the emitted view.
            //     It includes residual-aware scores when top-up is enabled, so
            //     base-selected and fill-selected rows share one score scale.
            std::vector<std::vector<std::pair<int32_t, double>>> baseCandidatesByChannel(
                static_cast<size_t>(numChannels));
            std::vector<std::vector<std::pair<int32_t, double>>> outputCandidatesByChannel(
                static_cast<size_t>(numChannels));

            for (int32_t channelIndex = 0; channelIndex < numChannels; ++channelIndex) {
                const auto& nodeTypeState = states[channelIndex]->_state[seedIdx][nodeTypeId];
                int32_t channelPPRNodeBudget = std::min(channelQuotas[channelIndex], maxPPRNodes);
                auto baseNodesAndScores = selectFinalizedPPRPairs(nodeTypeState, channelPPRNodeBudget);

                // Keep residual top-up out of the quota-biased base pass. The
                // output view starts from the same finalized candidates, then
                // adds residual mass/candidates for emitted features and the
                // later fill pass.
                auto outputNodesAndScores = baseNodesAndScores;
                if (enableResidualTopUp) {
                    addResidualMassToPPRPairs(nodeTypeState, outputNodesAndScores);
                    appendResidualTopUpPairs(nodeTypeState, outputNodesAndScores, maxPPRNodes);
                }

                double baseMaxScore = 0.0;
                for (const auto& nodeAndScore : baseNodesAndScores) {
                    baseMaxScore = std::max(baseMaxScore, nodeAndScore.second);
                }

                double outputMaxScore = 0.0;
                for (const auto& nodeAndScore : outputNodesAndScores) {
                    outputMaxScore = std::max(outputMaxScore, nodeAndScore.second);
                }

                baseCandidatesByChannel[channelIndex].reserve(baseNodesAndScores.size());
                outputCandidatesByChannel[channelIndex].reserve(outputNodesAndScores.size());
                addTypedPPRChannelCandidates(baseCandidatesByChannel[channelIndex], baseNodesAndScores, baseMaxScore);
                addTypedPPRSeedFeaturesAndCandidates(outputScores,
                                                     outputCandidatesByChannel[channelIndex],
                                                     outputNodesAndScores,
                                                     outputMaxScore,
                                                     channelIndex,
                                                     numChannels);
            }

            auto selectedNodes = selectTypedPPRNodeIds(baseCandidatesByChannel, channelQuotas, maxPPRNodes);
            int32_t selectedNodeCount = static_cast<int32_t>(selectedNodes.size());
            if (selectedNodeCount < maxPPRNodes) {
                std::unordered_set<int32_t> selectedNodeIds(selectedNodes.begin(), selectedNodes.end());
                auto topUpSelectedNodes =
                    selectTypedPPRNodeIds(outputCandidatesByChannel, topUpChannelQuotas, maxPPRNodes);
                for (int32_t nodeId : topUpSelectedNodes) {
                    if (selectedNodeCount >= maxPPRNodes) {
                        break;
                    }
                    if (selectedNodeIds.find(nodeId) != selectedNodeIds.end()) {
                        continue;
                    }
                    ++selectedNodeCount;
                    selectedNodeIds.insert(nodeId);
                    selectedNodes.push_back(nodeId);
                }
            }

            if (enableResidualTopUp && selectedNodes.size() > 1) {
                std::vector<std::pair<int32_t, double>> selectedNodesAndScores;
                selectedNodesAndScores.reserve(selectedNodes.size());
                for (int32_t nodeId : selectedNodes) {
                    selectedNodesAndScores.emplace_back(nodeId, outputScores.at(nodeId)[0]);
                }
                std::stable_sort(selectedNodesAndScores.begin(),
                                 selectedNodesAndScores.end(),
                                 [](const auto& a, const auto& b) { return a.second > b.second; });

                selectedNodes.clear();
                selectedNodes.reserve(selectedNodesAndScores.size());
                for (const auto& selectedNodeAndScore : selectedNodesAndScores) {
                    selectedNodes.push_back(selectedNodeAndScore.first);
                }
            }

            for (int32_t nodeId : selectedNodes) {
                flatIds.push_back(static_cast<int64_t>(nodeId));
                const auto& features = outputScores.at(nodeId);
                flatFeatureValues.insert(flatFeatureValues.end(), features.begin(), features.end());
            }

            validCounts.push_back(static_cast<int64_t>(selectedNodeCount));
        }

        auto flatWeights =
            torch::tensor(flatFeatureValues, torch::kDouble)
                .reshape({static_cast<int64_t>(flatIds.size()), static_cast<int64_t>(numEdgeAttrFeatures)});
        extractedTypedPPRByNodeTypeId[nodeTypeId] = {
            torch::tensor(flatIds, torch::kLong),
            flatWeights,
            torch::tensor(validCounts, torch::kLong),
        };
    }

    return extractedTypedPPRByNodeTypeId;
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
