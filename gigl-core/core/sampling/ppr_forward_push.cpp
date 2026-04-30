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
static uint64_t packKey(int32_t nodeId, int32_t etypeId) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(nodeId)) << 32) |
           static_cast<uint32_t>(etypeId);
}

PPRForwardPushState::PPRForwardPushState(const torch::Tensor& seedNodes,
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

    TORCH_CHECK(seedNodeTypeId >= 0,
                "seedNodeTypeId ", seedNodeTypeId, " is negative.");
    TORCH_CHECK(seedNodeTypeId < _numNodeTypes,
                "seedNodeTypeId ", seedNodeTypeId, " out of range [0, ", _numNodeTypes, ").");
    auto numEdgeTypes = static_cast<int32_t>(_edgeTypeToDstNtypeId.size());
    for (int32_t eid = 0; eid < numEdgeTypes; ++eid) {
        int32_t dstNt = _edgeTypeToDstNtypeId[eid];
        TORCH_CHECK(dstNt >= 0,
                    "edgeTypeToDstNtypeId[", eid, "] = ", dstNt, " is negative.");
        TORCH_CHECK(dstNt < _numNodeTypes,
                    "edgeTypeToDstNtypeId[", eid, "] = ", dstNt,
                    " out of range [0, ", _numNodeTypes, ").");
    }
    for (int32_t nt = 0; nt < _numNodeTypes; ++nt) {
        for (int32_t eid : _nodeTypeToEdgeTypeIds[nt]) {
            TORCH_CHECK(eid >= 0,
                        "nodeTypeToEdgeTypeIds[", nt, "] contains negative edge type id ", eid, ".");
            TORCH_CHECK(eid < numEdgeTypes,
                        "nodeTypeToEdgeTypeIds[", nt, "] contains edge type id ", eid,
                        " out of range [0, ", numEdgeTypes, ").");
        }
    }

    // Allocate per-seed, per-node-type tables.
    // .assign(n, val) fills a vector with n copies of val — like [val] * n in Python.
    _pprScores.assign(_batchSize, std::vector<std::unordered_map<int32_t, double>>(_numNodeTypes));
    _residuals.assign(_batchSize, std::vector<std::unordered_map<int32_t, double>>(_numNodeTypes));
    _queue.assign(_batchSize, std::vector<std::unordered_set<int32_t>>(_numNodeTypes));
    _queuedNodes.assign(_batchSize, std::vector<std::unordered_set<int32_t>>(_numNodeTypes));

    // accessor<dtype, ndim>() returns a typed view into the tensor's data that
    // supports [i] indexing with bounds checking in debug builds.
    auto acc = seedNodes.accessor<int64_t, 1>();
    _numNodesInQueue = _batchSize;
    for (int32_t i = 0; i < _batchSize; ++i) {
        auto seed = static_cast<int32_t>(acc[i]);
        // PPR initialisation: each seed starts with residual = alpha (the
        // restart probability).  The first push will move alpha into ppr_score
        // and distribute (1-alpha)*alpha to the seed's neighbors.
        _residuals[i][seedNodeTypeId][seed] = _alpha;
        _queue[i][seedNodeTypeId].insert(seed);
    }
}

std::optional<std::unordered_map<int32_t, torch::Tensor>> PPRForwardPushState::drainQueue() {
    if (_numNodesInQueue == 0) {
        return std::nullopt;
    }

    // Reset the snapshot from the previous iteration.
    // TODO: if this loop becomes a bottleneck, consider parallelising with
    // std::for_each(std::execution::par_unseq, ...) or adding vectorisation hints.
    for (int32_t s = 0; s < _batchSize; ++s) {
        for (auto& qs : _queuedNodes[s]) {
            qs.clear();
        }
    }

    // nodesToLookup[eid] = set of node IDs that need a neighbor fetch for
    // edge type eid this round.  Using a set deduplicates nodes that appear
    // in multiple seeds' queues: we only fetch each (node, etype) pair once.
    std::unordered_map<int32_t, std::unordered_set<int32_t>> nodesToLookup;

    int32_t totalDrainedThisRound = 0;
    for (int32_t s = 0; s < _batchSize; ++s) {
        for (int32_t nt = 0; nt < _numNodeTypes; ++nt) {
            if (_queue[s][nt].empty()) {
                continue;
            }

            // Move the live queue into the snapshot in O(1) — avoids copying all node IDs.
            // The explicit clear() after move is defensive: the standard only guarantees
            // a moved-from container is "valid but unspecified", not necessarily empty.
            _queuedNodes[s][nt] = std::move(_queue[s][nt]);
            _queue[s][nt].clear();
            auto numDrained = static_cast<int32_t>(_queuedNodes[s][nt].size());
            totalDrainedThisRound += numDrained;
            _numNodesInQueue -= numDrained;

            for (int32_t nodeId : _queuedNodes[s][nt]) {
                for (int32_t eid : _nodeTypeToEdgeTypeIds[nt]) {
                    if (_neighborCache.find(packKey(nodeId, eid)) == _neighborCache.end()) {
                        nodesToLookup[eid].insert(nodeId);
                    }
                }
            }
        }
    }

    _nodesDrainedPerIteration.push_back(totalDrainedThisRound);

    std::unordered_map<int32_t, torch::Tensor> result;
    for (const auto& [eid, nodeSet] : nodesToLookup) {
        std::vector<int64_t> ids(nodeSet.begin(), nodeSet.end());
        result[eid] = torch::tensor(ids, torch::kLong);
    }
    return result;
}

const std::vector<int32_t>& PPRForwardPushState::getNodesDrainedPerIteration() const {
    return _nodesDrainedPerIteration;
}

void PPRForwardPushState::pushResiduals(
    const std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>& fetchedByEtypeId) {
    // Step 1: Unpack the input map into a C++ map keyed by packKey(nodeId, etypeId)
    // for fast lookup during the residual-push loop below.
    std::unordered_map<uint64_t, std::vector<int32_t>> fetched;
    for (const auto& [eid, tup] : fetchedByEtypeId) {
        const auto& nodeIdsT = std::get<0>(tup);
        const auto& flatNbrsT = std::get<1>(tup);
        const auto& countsT = std::get<2>(tup);

        // accessor<int64_t, 1>() gives a bounds-checked, typed 1-D view into
        // each tensor's data — equivalent to iterating over a NumPy array.
        auto nodeAcc = nodeIdsT.accessor<int64_t, 1>();
        auto nbrAcc = flatNbrsT.accessor<int64_t, 1>();
        auto cntAcc = countsT.accessor<int64_t, 1>();

        // Walk the flat neighbor list, slicing out each node's neighbors using
        // the running offset into the concatenated flat buffer.
        int64_t offset = 0;
        for (int64_t i = 0; i < nodeIdsT.size(0); ++i) {
            auto nid = static_cast<int32_t>(nodeAcc[i]);
            int64_t count = cntAcc[i];
            std::vector<int32_t> nbrs(count);
            for (int64_t j = 0; j < count; ++j) {
                nbrs[j] = static_cast<int32_t>(nbrAcc[offset + j]);
            }
            fetched[packKey(nid, eid)] = std::move(nbrs);
            offset += count;
        }
    }

    // Step 2: For every node that was in the queue (captured in _queuedNodes
    // by drainQueue()), apply one PPR push step:
    //   a. Absorb residual into the PPR score.
    //   b. Distribute (1-alpha) * residual equally to each neighbor.
    //   c. Enqueue any neighbor whose residual now exceeds the requeue threshold.
    for (int32_t s = 0; s < _batchSize; ++s) {
        for (int32_t nt = 0; nt < _numNodeTypes; ++nt) {
            if (_queuedNodes[s][nt].empty()) {
                continue;
            }

            for (int32_t src : _queuedNodes[s][nt]) {
                auto& srcRes = _residuals[s][nt];
                auto it = srcRes.find(src);
                double res = (it != srcRes.end()) ? it->second : 0.0;

                // a. Absorb: move residual into the PPR score.
                _pprScores[s][nt][src] += res;
                srcRes[src] = 0.0;

                // b. Count total fetched/cached neighbors across all edge types for
                // this source node.  We normalise by the number of neighbors we
                // actually retrieved, not the true degree, so residual is fully
                // distributed among known neighbors rather than leaking to unfetched
                // ones (which matters when num_neighbors_per_hop < true_degree).
                int32_t totalFetched = 0;
                for (int32_t eid : _nodeTypeToEdgeTypeIds[nt]) {
                    auto fetchedEntry = fetched.find(packKey(src, eid));
                    if (fetchedEntry != fetched.end()) {
                        totalFetched += static_cast<int32_t>(fetchedEntry->second.size());
                    } else {
                        auto cachedEntry = _neighborCache.find(packKey(src, eid));
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

                double resPerNbr = (1.0 - _alpha) * res / static_cast<double>(totalFetched);

                for (int32_t eid : _nodeTypeToEdgeTypeIds[nt]) {
                    // Invariant: fetched and _neighborCache are mutually exclusive for
                    // any given (node, etype) key within one iteration.  drainQueue()
                    // only requests a fetch for nodes absent from _neighborCache, so a
                    // key is in at most one of the two.
                    // Points at the neighbor list we will distribute residual to —
                    // either from `fetched` (new this iteration) or `_neighborCache`
                    // (seen in a previous iteration).  Null if neither has data for
                    // this (node, etype) pair.  Does not own any memory.
                    const std::vector<int32_t>* nbrList = nullptr;
                    auto fetchedEntry = fetched.find(packKey(src, eid));
                    if (fetchedEntry != fetched.end()) {
                        nbrList = &fetchedEntry->second;
                    } else {
                        auto cachedEntry = _neighborCache.find(packKey(src, eid));
                        if (cachedEntry != _neighborCache.end()) {
                            nbrList = &cachedEntry->second;
                        }
                    }
                    if (!nbrList || nbrList->empty()) {
                        continue;
                    }

                    int32_t dstNt = _edgeTypeToDstNtypeId[eid];

                    // c. Accumulate residual for each neighbor and re-enqueue if threshold
                    // exceeded.
                    for (int32_t nbr : *nbrList) {
                        _residuals[s][dstNt][nbr] += resPerNbr;

                        double threshold = _requeueThresholdFactor * static_cast<double>(getTotalDegree(nbr, dstNt));

                        if (_queue[s][dstNt].find(nbr) == _queue[s][dstNt].end() &&
                            _residuals[s][dstNt][nbr] >= threshold) {
                            _queue[s][dstNt].insert(nbr);
                            ++_numNodesInQueue;

                            // Promote neighbor lists to the persistent cache: this node will
                            // be processed next iteration, so caching avoids a re-fetch.
                            for (int32_t peid : _nodeTypeToEdgeTypeIds[dstNt]) {
                                uint64_t pk = packKey(nbr, peid);
                                if (_neighborCache.find(pk) == _neighborCache.end()) {
                                    auto fetchedNbrEntry = fetched.find(pk);
                                    if (fetchedNbrEntry != fetched.end()) {
                                        _neighborCache[pk] = fetchedNbrEntry->second;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> PPRForwardPushState::extractTopK(
    int32_t maxPprNodes) {
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    // Emit an entry for every node type, even if unreachable in this batch (empty tensors,
    // all-zero valid_counts).  This keeps the output shape consistent across batches so
    // downstream model architectures see a fixed set of PPR edge types every iteration.
    for (int32_t nt = 0; nt < _numNodeTypes; ++nt) {
        std::vector<int64_t> flatIds;
        std::vector<float> flatWeights;
        std::vector<int64_t> validCounts;

        for (int32_t seedIdx = 0; seedIdx < _batchSize; ++seedIdx) {
            const auto& scores = _pprScores[seedIdx][nt];
            int32_t topK = std::min(maxPprNodes, static_cast<int32_t>(scores.size()));
            if (topK > 0) {
                std::vector<std::pair<int32_t, double>> scorePairs(scores.begin(), scores.end());
                std::partial_sort(scorePairs.begin(), scorePairs.begin() + topK, scorePairs.end(),
                                  [](const auto& a, const auto& b) { return a.second > b.second; });

                for (int32_t i = 0; i < topK; ++i) {
                    flatIds.push_back(static_cast<int64_t>(scorePairs[i].first));
                    // Cast to float32 for output; internal scores stay double to
                    // avoid accumulated rounding errors in the push loop.
                    flatWeights.push_back(static_cast<float>(scorePairs[i].second));
                }
            }
            validCounts.push_back(static_cast<int64_t>(topK));
        }

        result[nt] = {torch::tensor(flatIds, torch::kLong),
                      torch::tensor(flatWeights, torch::kFloat),
                      torch::tensor(validCounts, torch::kLong)};
    }
    return result;
}

int32_t PPRForwardPushState::getTotalDegree(int32_t nodeId, int32_t ntypeId) const {
    if (ntypeId >= static_cast<int32_t>(_degreeTensors.size())) {
        return 0;
    }
    const auto& t = _degreeTensors[ntypeId];
    if (t.numel() == 0) {
        return 0;
    }
    TORCH_CHECK(nodeId >= 0,
                "Node ID ", nodeId, " is negative, which indicates a sampler bug.");
    TORCH_CHECK(nodeId < static_cast<int32_t>(t.size(0)),
                "Node ID ", nodeId, " out of range for degree tensor of ntype_id ",
                ntypeId, " (size=", t.size(0), "). This indicates corrupted graph data or a sampler bug.");
    if (t.scalar_type() == torch::kInt) {
        return t.data_ptr<int32_t>()[nodeId];
    }
    if (t.scalar_type() == torch::kLong) {
        return static_cast<int32_t>(
            std::min<int64_t>(t.data_ptr<int64_t>()[nodeId], INT32_MAX));
    }
    TORCH_CHECK(false, "Unsupported degree tensor dtype: ", t.scalar_type(),
                ". Expected torch.int32 or torch.int64.");
    return 0;  // unreachable; suppresses compiler warning
}

}  // namespace gigl
