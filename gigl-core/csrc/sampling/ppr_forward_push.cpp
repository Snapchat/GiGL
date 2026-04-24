#include "ppr_forward_push.h"

PPRForwardPushState::PPRForwardPushState(const torch::Tensor& seedNodes,
                                         int32_t seedNodeTypeId,
                                         double alpha,
                                         double requeueThresholdFactor,
                                         std::vector<std::vector<int32_t>> nodeTypeToEdgeTypeIds,
                                         std::vector<int32_t> edgeTypeToDstNtypeId,
                                         std::vector<torch::Tensor> degreeTensors)
    : _alpha(alpha),
      _oneMinusAlpha(1.0 - alpha),
      _requeueThresholdFactor(requeueThresholdFactor),
      // std::move transfers ownership of each vector into the member variable
      // without copying its contents — equivalent to Python's list hand-off
      // when you no longer need the original.
      _nodeTypeToEdgeTypeIds(std::move(nodeTypeToEdgeTypeIds)),
      _edgeTypeToDstNtypeId(std::move(edgeTypeToDstNtypeId)),
      _degreeTensors(std::move(degreeTensors)) {
    TORCH_CHECK(seedNodes.dim() == 1, "seedNodes must be 1D");
    _batchSize = static_cast<int32_t>(seedNodes.size(0));
    _numNodeTypes = static_cast<int32_t>(_nodeTypeToEdgeTypeIds.size());

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

            // Move the live queue into the snapshot (no data copy — O(1)).
            _queuedNodes[s][nt] = std::move(_queue[s][nt]);
            _queue[s][nt].clear();
            totalDrainedThisRound += static_cast<int32_t>(_queuedNodes[s][nt].size());
            _numNodesInQueue -= static_cast<int32_t>(_queuedNodes[s][nt].size());

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
    for (auto& [eid, nodeSet] : nodesToLookup) {
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
                    auto fi = fetched.find(packKey(src, eid));
                    if (fi != fetched.end()) {
                        totalFetched += static_cast<int32_t>(fi->second.size());
                    } else {
                        auto ci = _neighborCache.find(packKey(src, eid));
                        if (ci != _neighborCache.end()) {
                            totalFetched += static_cast<int32_t>(ci->second.size());
                        }
                    }
                }
                // Destination-only nodes (or nodes with no fetched neighbors) absorb
                // residual but do not push further.
                if (totalFetched == 0) {
                    continue;
                }

                double resPerNbr = _oneMinusAlpha * res / static_cast<double>(totalFetched);

                for (int32_t eid : _nodeTypeToEdgeTypeIds[nt]) {
                    // Invariant: fetched and _neighborCache are mutually exclusive for
                    // any given (node, etype) key within one iteration.  drainQueue()
                    // only requests a fetch for nodes absent from _neighborCache, so a
                    // key is in at most one of the two.
                    const std::vector<int32_t>* nbrList = nullptr;
                    auto fi = fetched.find(packKey(src, eid));
                    if (fi != fetched.end()) {
                        nbrList = &fi->second;
                    } else {
                        auto ci = _neighborCache.find(packKey(src, eid));
                        if (ci != _neighborCache.end()) {
                            nbrList = &ci->second;
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
                                    auto pfi = fetched.find(pk);
                                    if (pfi != fetched.end()) {
                                        _neighborCache[pk] = pfi->second;
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
    std::unordered_set<int32_t> active;
    for (int32_t s = 0; s < _batchSize; ++s) {
        for (int32_t nt = 0; nt < _numNodeTypes; ++nt) {
            if (!_pprScores[s][nt].empty()) {
                active.insert(nt);
            }
        }
    }

    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    for (int32_t nt : active) {
        std::vector<int64_t> flatIds;
        std::vector<float> flatWeights;
        std::vector<int64_t> validCounts;

        for (int32_t s = 0; s < _batchSize; ++s) {
            const auto& scores = _pprScores[s][nt];
            int32_t k = std::min(maxPprNodes, static_cast<int32_t>(scores.size()));
            if (k > 0) {
                std::vector<std::pair<int32_t, double>> items(scores.begin(), scores.end());
                std::partial_sort(items.begin(), items.begin() + k, items.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });

                for (int32_t i = 0; i < k; ++i) {
                    flatIds.push_back(static_cast<int64_t>(items[i].first));
                    // Cast to float32 for output; internal scores stay double to
                    // avoid accumulated rounding errors in the push loop.
                    flatWeights.push_back(static_cast<float>(items[i].second));
                }
            }
            validCounts.push_back(static_cast<int64_t>(k));
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
    TORCH_CHECK(nodeId < static_cast<int32_t>(t.size(0)),
                "Node ID ",
                nodeId,
                " out of range for degree tensor of ntype_id ",
                ntypeId,
                " (size=",
                t.size(0),
                "). This indicates corrupted graph data or a sampler bug.");
    // data_ptr<int32_t>() returns a raw C pointer to the tensor's int32 data buffer.
    return t.data_ptr<int32_t>()[nodeId];
}
