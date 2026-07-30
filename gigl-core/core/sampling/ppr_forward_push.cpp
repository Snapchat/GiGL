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
    // Fetch requests are edge-type scoped: each edge type has its own adjacency
    // table and destination node type. Grouping by node type would merge
    // relation-specific fetches that must remain separate.
    std::unordered_map<int32_t, std::unordered_set<int64_t>> unionedSourceNodeIdsByEdgeTypeId;

    // TODO: If benchmarking shows typed PPR is materially slower than regular
    // heterogeneous PPR because of this drain loop, evaluate parallelizing
    // channels with per-thread frontier maps and a deterministic merge into
    // queueDrainResult.
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
            auto& unionedSourceNodeIds = unionedSourceNodeIdsByEdgeTypeId[edgeTypeId];
            for (int64_t nodeIndex = 0; nodeIndex < nodes.size(0); ++nodeIndex) {
                unionedSourceNodeIds.insert(nodeAccessor[nodeIndex]);
            }
        }

        if (!requestedEdgeTypeIds.empty()) {
            queueDrainResult.fetchChannelIndices.push_back(static_cast<int32_t>(channelIndex));
            queueDrainResult.edgeTypeIdsByFetchChannel.push_back(std::move(requestedEdgeTypeIds));
        }
    }

    for (const auto& [edgeTypeId, unionedSourceNodeIds] : unionedSourceNodeIdsByEdgeTypeId) {
        std::vector<int64_t> nodeIdsToLookup(unionedSourceNodeIds.begin(), unionedSourceNodeIds.end());
        queueDrainResult.unionedNodeIdsByEdgeTypeId[edgeTypeId] = torch::tensor(nodeIdsToLookup, torch::kLong);
    }
    return queueDrainResult;
}

void PPRForwardPush::pushResiduals(const NeighborFetchMap& fetchedByEtypeId) {
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

// Helper function for adding one channel's extracted PPR candidates into the
// emitted typed feature table and a selection candidate list.
//
// This helper writes the emitted edge_attr feature table for every node it
// sees. When residual top-up is enabled, callers pass residual-aware scores
// here so finalized and residual rows are emitted on the same ppr_score +
// residual scale.
//
// Inputs:
//   nodesAndScores: extracted (node_id, score) candidates for one channel.
//   maxScore: largest score in the channel, used to calibrate scores to [0, 1].
//   channelIndex: index of the typed PPR channel being added.
//   numChannels: total typed PPR channels, used to derive feature width.
//   outputScoresByNodeId: mutable map from node ID to emitted edge_attr features.
//   channelOutputCandidates: mutable sortable candidates for target-count selection.
//
// Expected output: outputScoresByNodeId has this channel's score/presence
// features merged in, and channelOutputCandidates contains this channel's
// calibrated candidates.
static void addTypedPPRSeedFeaturesAndCandidates(const std::vector<std::pair<int32_t, double>>& nodesAndScores,
                                                 double maxScore,
                                                 int32_t channelIndex,
                                                 int32_t numChannels,
                                                 std::unordered_map<int32_t, std::vector<double>>& outputScoresByNodeId,
                                                 std::vector<std::pair<int32_t, double>>& channelOutputCandidates) {
    if (nodesAndScores.empty()) {
        return;
    }
    TORCH_CHECK(maxScore > 0.0,
                "Typed PPR output has candidates but non-positive max score ",
                maxScore,
                ", which indicates invalid PPR state.");

    // Feature width is 1 + 2C:
    //   column 0: best calibrated score across channels, used as the scalar
    //             PPR weight by downstream ranking/sequence construction.
    //   columns [1, C]: calibrated score for each channel.
    //   columns [1 + C, 1 + 2C): presence bit for each channel.
    int32_t numEdgeAttrFeatures = 1 + (2 * numChannels);
    for (const auto& [nodeId, score] : nodesAndScores) {
        double calibratedScore = score / maxScore;
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
        // channel reached the node. Current extraction emits one row per node
        // per channel; max keeps the merge stable if a future caller ever
        // passes duplicates.
        scoreFeatures[channelScoreIndex] = std::max(scoreFeatures[channelScoreIndex], calibratedScore);
        scoreFeatures[channelPresenceIndex] = 1.0;

        // Keep a per-channel sortable candidate list so target counts can be
        // applied after cross-channel attribution.
        channelOutputCandidates.emplace_back(nodeId, calibratedScore);
    }
}

// Helper function for applying typed channel target counts and cross-channel dedup.
//
// Inputs:
//   candidatesByChannel: mutable (node_id, calibrated_score) candidates per channel.
//   channelTargetCounts: desired output count per attributed channel.
//   maxPPRNodes: maximum number of deduplicated node IDs to return for a seed.
//
// Expected output: node IDs selected by this policy:
//   1. Attribute duplicate nodes to the channel where they have the highest score.
//   2. Fill each channel up to its target count.
//   3. Redistribute unused slots to the remaining highest-scoring candidates.
//   4. Return the selected nodes globally ranked by best calibrated score.
static std::vector<int32_t> selectTypedPPRNodeIds(
    std::vector<std::vector<std::pair<int32_t, double>>>& candidatesByChannel,
    const std::vector<int32_t>& channelTargetCounts,
    int32_t maxPPRNodes) {
    struct AttributedCandidate {
        double bestCalibratedScore;
        double channelCalibratedScore;
        int32_t channelIndex;
        int32_t nodeId;
    };

    const auto higherAttributedCandidate = [](const auto& a, const auto& b) {
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

    size_t totalCandidateRows = 0;
    for (const auto& candidates : candidatesByChannel) {
        totalCandidateRows += candidates.size();
    }

    // Deduplicate before applying channel targets. A node that appears in
    // multiple channels is attributed to the channel where it has the strongest
    // calibrated score, which is the same score used for global ranking.
    std::unordered_map<int32_t, AttributedCandidate> bestCandidateByNodeId;
    bestCandidateByNodeId.reserve(totalCandidateRows);
    for (int32_t channelIndex = 0; channelIndex < static_cast<int32_t>(candidatesByChannel.size()); ++channelIndex) {
        for (const auto& [nodeId, calibratedScore] : candidatesByChannel[channelIndex]) {
            AttributedCandidate candidate{calibratedScore, calibratedScore, channelIndex, nodeId};
            auto [candidateIter, inserted] = bestCandidateByNodeId.emplace(nodeId, candidate);
            if (!inserted && higherAttributedCandidate(candidate, candidateIter->second)) {
                candidateIter->second = candidate;
            }
        }
    }

    std::vector<std::vector<AttributedCandidate>> candidatesByAttributedChannel(
        static_cast<size_t>(candidatesByChannel.size()));
    for (const auto& candidateEntry : bestCandidateByNodeId) {
        const auto& candidate = candidateEntry.second;
        candidatesByAttributedChannel[candidate.channelIndex].push_back(candidate);
    }

    std::vector<AttributedCandidate> selectedCandidates;
    selectedCandidates.reserve(
        static_cast<size_t>(std::min(maxPPRNodes, static_cast<int32_t>(bestCandidateByNodeId.size()))));
    std::unordered_set<int32_t> selectedNodeIds;
    selectedNodeIds.reserve(selectedCandidates.capacity());

    // First honor the target counts for each attributed channel. Selection is
    // local to each channel so we only spend target slots on that channel's
    // strongest unique nodes.
    for (int32_t channelIndex = 0; channelIndex < static_cast<int32_t>(candidatesByAttributedChannel.size());
         ++channelIndex) {
        auto& candidates = candidatesByAttributedChannel[channelIndex];
        int32_t remainingOutputSlots = maxPPRNodes - static_cast<int32_t>(selectedCandidates.size());
        if (remainingOutputSlots <= 0) {
            break;
        }
        int32_t numTargetCandidates = std::min(
            {channelTargetCounts[channelIndex], static_cast<int32_t>(candidates.size()), remainingOutputSlots});
        if (numTargetCandidates <= 0) {
            continue;
        }

        if (numTargetCandidates < static_cast<int32_t>(candidates.size())) {
            // Membership is enough here: the final selected set is sorted once
            // after target fill and redistribution.
            std::nth_element(candidates.begin(),
                             candidates.begin() + numTargetCandidates,
                             candidates.end(),
                             higherAttributedCandidate);
        }

        for (int32_t candidateIndex = 0; candidateIndex < numTargetCandidates; ++candidateIndex) {
            const auto& candidate = candidates[candidateIndex];
            selectedCandidates.push_back(candidate);
            selectedNodeIds.insert(candidate.nodeId);
        }
    }

    // Sparse channels or cross-channel duplicates can leave some target slots
    // unused. Redistribute that leftover capacity to the strongest remaining
    // candidates globally so sequence length is not sacrificed for ratios that
    // cannot be exactly filled on this seed.
    int32_t remainingOutputSlots = maxPPRNodes - static_cast<int32_t>(selectedCandidates.size());
    if (remainingOutputSlots > 0) {
        std::vector<AttributedCandidate> remainingCandidates;
        remainingCandidates.reserve(bestCandidateByNodeId.size() - selectedNodeIds.size());
        for (const auto& candidateEntry : bestCandidateByNodeId) {
            const auto& candidate = candidateEntry.second;
            if (selectedNodeIds.find(candidate.nodeId) == selectedNodeIds.end()) {
                remainingCandidates.push_back(candidate);
            }
        }

        int32_t numFillCandidates = std::min(remainingOutputSlots, static_cast<int32_t>(remainingCandidates.size()));
        if (numFillCandidates < static_cast<int32_t>(remainingCandidates.size())) {
            // Membership is enough here too; final output ordering happens once
            // after selectedCandidates is complete.
            std::nth_element(remainingCandidates.begin(),
                             remainingCandidates.begin() + numFillCandidates,
                             remainingCandidates.end(),
                             higherAttributedCandidate);
        }

        for (int32_t candidateIndex = 0; candidateIndex < numFillCandidates; ++candidateIndex) {
            selectedCandidates.push_back(remainingCandidates[candidateIndex]);
        }
    }

    // The target/fill passes decide membership; final output order is still by
    // best calibrated score so downstream sequence construction sees a ranked
    // PPR sequence rather than channel-grouped blocks.
    if (selectedCandidates.size() > 1) {
        std::sort(selectedCandidates.begin(), selectedCandidates.end(), higherAttributedCandidate);
    }

    std::vector<int32_t> selectedNodes;
    selectedNodes.reserve(selectedCandidates.size());
    for (const auto& candidate : selectedCandidates) {
        selectedNodes.push_back(candidate.nodeId);
    }
    return selectedNodes;
}

PPRExtractResult PPRForwardPush::extractTopKWithResidualTopUp(int32_t maxPPRNodes, bool enableResidualTopUp) {
    TORCH_CHECK(maxPPRNodes >= 0, "maxPPRNodes must be non-negative, got ", maxPPRNodes, ".");

    PPRExtractResult extractedPPRByNodeTypeId;
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

PPRExtractResult extractTypedTopKWithResidualTopUp(const std::vector<PPRForwardPush*>& states,
                                                   const std::vector<int32_t>& channelTargetCounts,
                                                   int32_t maxPPRNodes,
                                                   bool enableResidualTopUp) {
    // Typed channels are constructed from the same seed batch and graph schema;
    // only the edge-type traversal allowlist differs. The sampler calls typed
    // extraction only when typed-channel target counts are configured, so at
    // least one state exists. Use the first state as the shared schema source
    // for batch size and node-type count.
    const auto* firstState = states.front();
    int32_t batchSize = firstState->_batchSize;
    int32_t numNodeTypes = firstState->_numNodeTypes;
    int32_t numChannels = static_cast<int32_t>(states.size());
    // Feature width is 1 + 2C:
    //   column 0: best calibrated score across channels, used as the scalar
    //             PPR weight by downstream ranking/sequence construction.
    //   columns [1, C]: calibrated score for each channel.
    //   columns [1 + C, 1 + 2C): presence bit for each channel.
    int32_t numEdgeAttrFeatures = 1 + (2 * numChannels);

    // Pre-size the per-seed feature map below. Each channel can contribute up
    // to maxPPRNodes candidates before cross-channel dedup and target filling.
    size_t outputCandidateReserveSize = static_cast<size_t>(numChannels) * static_cast<size_t>(maxPPRNodes);

    PPRExtractResult extractedTypedPPRByNodeTypeId;

    for (int32_t nodeTypeId = 0; nodeTypeId < numNodeTypes; ++nodeTypeId) {
        std::vector<int64_t> flatIds;
        std::vector<double> flatFeatureValues;
        std::vector<int64_t> validCounts;

        for (int32_t seedIdx = 0; seedIdx < batchSize; ++seedIdx) {
            std::unordered_map<int32_t, std::vector<double>> outputScores;
            outputScores.reserve(outputCandidateReserveSize);
            // outputCandidatesByChannel includes finalized PPR rows, plus
            // residual-aware rows when top-up is enabled. Selection and emitted
            // features use this same view so both finalized and residual
            // candidates obey the per-channel target counts.
            std::vector<std::vector<std::pair<int32_t, double>>> outputCandidatesByChannel(
                static_cast<size_t>(numChannels));

            for (int32_t channelIndex = 0; channelIndex < numChannels; ++channelIndex) {
                const auto& nodeTypeState = states[channelIndex]->_state[seedIdx][nodeTypeId];
                auto outputNodesAndScores = selectFinalizedPPRPairs(nodeTypeState, maxPPRNodes);

                if (enableResidualTopUp) {
                    addResidualMassToPPRPairs(nodeTypeState, outputNodesAndScores);
                    appendResidualTopUpPairs(nodeTypeState, outputNodesAndScores, maxPPRNodes);
                }

                auto outputMaxScoreIter =
                    std::max_element(outputNodesAndScores.begin(),
                                     outputNodesAndScores.end(),
                                     [](const auto& a, const auto& b) { return a.second < b.second; });
                double outputMaxScore =
                    outputMaxScoreIter != outputNodesAndScores.end() ? outputMaxScoreIter->second : 0.0;

                outputCandidatesByChannel[channelIndex].reserve(outputNodesAndScores.size());
                addTypedPPRSeedFeaturesAndCandidates(outputNodesAndScores,
                                                     outputMaxScore,
                                                     channelIndex,
                                                     numChannels,
                                                     outputScores,
                                                     outputCandidatesByChannel[channelIndex]);
            }

            auto selectedNodes = selectTypedPPRNodeIds(outputCandidatesByChannel, channelTargetCounts, maxPPRNodes);
            int32_t selectedNodeCount = static_cast<int32_t>(selectedNodes.size());

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
