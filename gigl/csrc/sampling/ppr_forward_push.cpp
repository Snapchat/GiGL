#include "ppr_forward_push.h"

PPRForwardPushState::PPRForwardPushState(torch::Tensor seed_nodes,
                                         int32_t seed_node_type_id,
                                         double alpha,
                                         double requeue_threshold_factor,
                                         std::vector<std::vector<int32_t>> node_type_to_edge_type_ids,
                                         std::vector<int32_t> edge_type_to_dst_ntype_id,
                                         std::vector<torch::Tensor> degree_tensors)
    : alpha_(alpha),
      one_minus_alpha_(1.0 - alpha),
      requeue_threshold_factor_(requeue_threshold_factor),
      // std::move transfers ownership of each vector into the member variable
      // without copying its contents — equivalent to Python's list hand-off
      // when you no longer need the original.
      node_type_to_edge_type_ids_(std::move(node_type_to_edge_type_ids)),
      edge_type_to_dst_ntype_id_(std::move(edge_type_to_dst_ntype_id)),
      degree_tensors_(std::move(degree_tensors)) {
    TORCH_CHECK(seed_nodes.dim() == 1, "seed_nodes must be 1D");
    batch_size_ = static_cast<int32_t>(seed_nodes.size(0));
    num_node_types_ = static_cast<int32_t>(node_type_to_edge_type_ids_.size());

    // Allocate per-seed, per-node-type tables.
    // .assign(n, val) fills a vector with n copies of val — like [val] * n in Python.
    ppr_scores_.assign(batch_size_, std::vector<std::unordered_map<int32_t, double>>(num_node_types_));
    residuals_.assign(batch_size_, std::vector<std::unordered_map<int32_t, double>>(num_node_types_));
    queue_.assign(batch_size_, std::vector<std::unordered_set<int32_t>>(num_node_types_));
    queued_nodes_.assign(batch_size_, std::vector<std::unordered_set<int32_t>>(num_node_types_));

    // accessor<dtype, ndim>() returns a typed view into the tensor's data that
    // supports [i] indexing with bounds checking in debug builds.
    auto acc = seed_nodes.accessor<int64_t, 1>();
    num_nodes_in_queue_ = batch_size_;
    for (int32_t i = 0; i < batch_size_; ++i) {
        int32_t seed = static_cast<int32_t>(acc[i]);
        // PPR initialisation: each seed starts with residual = alpha (the
        // restart probability).  The first push will move alpha into ppr_score
        // and distribute (1-alpha)*alpha to the seed's neighbors.
        residuals_[i][seed_node_type_id][seed] = alpha_;
        queue_[i][seed_node_type_id].insert(seed);
    }
}

std::optional<std::unordered_map<int32_t, torch::Tensor>> PPRForwardPushState::drain_queue() {
    if (num_nodes_in_queue_ == 0) {
        return std::nullopt;
    }

    // Reset the snapshot from the previous iteration.
    for (int32_t s = 0; s < batch_size_; ++s)
        for (auto& qs : queued_nodes_[s])
            qs.clear();

    // nodes_to_lookup[eid] = set of node IDs that need a neighbor fetch for
    // edge type eid this round.  Using a set deduplicates nodes that appear
    // in multiple seeds' queues: we only fetch each (node, etype) pair once.
    std::unordered_map<int32_t, std::unordered_set<int32_t>> nodes_to_lookup;

    int32_t total_drained_this_round = 0;
    for (int32_t s = 0; s < batch_size_; ++s) {
        for (int32_t nt = 0; nt < num_node_types_; ++nt) {
            if (queue_[s][nt].empty())
                continue;

            // Move the live queue into the snapshot (no data copy — O(1)).
            queued_nodes_[s][nt] = std::move(queue_[s][nt]);
            queue_[s][nt].clear();
            total_drained_this_round += static_cast<int32_t>(queued_nodes_[s][nt].size());
            num_nodes_in_queue_ -= static_cast<int32_t>(queued_nodes_[s][nt].size());

            for (int32_t node_id : queued_nodes_[s][nt]) {
                for (int32_t eid : node_type_to_edge_type_ids_[nt]) {
                    if (neighbor_cache_.find(pack_key(node_id, eid)) == neighbor_cache_.end()) {
                        nodes_to_lookup[eid].insert(node_id);
                    }
                }
            }
        }
    }

    nodes_drained_per_iteration_.push_back(total_drained_this_round);

    std::unordered_map<int32_t, torch::Tensor> result;
    for (auto& [eid, node_set] : nodes_to_lookup) {
        std::vector<int64_t> ids(node_set.begin(), node_set.end());
        result[eid] = torch::tensor(ids, torch::kLong);
    }
    return result;
}

const std::vector<int32_t>& PPRForwardPushState::get_nodes_drained_per_iteration() const {
    return nodes_drained_per_iteration_;
}

void PPRForwardPushState::push_residuals(
    const std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>& fetched_by_etype_id) {
    // Step 1: Unpack the input map into a C++ map keyed by pack_key(node_id, etype_id)
    // for fast lookup during the residual-push loop below.
    std::unordered_map<uint64_t, std::vector<int32_t>> fetched;
    for (const auto& [eid, tup] : fetched_by_etype_id) {
        const auto& node_ids_t = std::get<0>(tup);
        const auto& flat_nbrs_t = std::get<1>(tup);
        const auto& counts_t = std::get<2>(tup);

        // accessor<int64_t, 1>() gives a bounds-checked, typed 1-D view into
        // each tensor's data — equivalent to iterating over a NumPy array.
        auto node_acc = node_ids_t.accessor<int64_t, 1>();
        auto nbr_acc = flat_nbrs_t.accessor<int64_t, 1>();
        auto cnt_acc = counts_t.accessor<int64_t, 1>();

        // Walk the flat neighbor list, slicing out each node's neighbors using
        // the running offset into the concatenated flat buffer.
        int64_t offset = 0;
        for (int64_t i = 0; i < node_ids_t.size(0); ++i) {
            int32_t nid = static_cast<int32_t>(node_acc[i]);
            int64_t count = cnt_acc[i];
            std::vector<int32_t> nbrs(count);
            for (int64_t j = 0; j < count; ++j)
                nbrs[j] = static_cast<int32_t>(nbr_acc[offset + j]);
            fetched[pack_key(nid, eid)] = std::move(nbrs);
            offset += count;
        }
    }

    // Step 2: For every node that was in the queue (captured in queued_nodes_
    // by drain_queue()), apply one PPR push step:
    //   a. Absorb residual into the PPR score.
    //   b. Distribute (1-alpha) * residual equally to each neighbor.
    //   c. Enqueue any neighbor whose residual now exceeds the requeue threshold.
    for (int32_t s = 0; s < batch_size_; ++s) {
        for (int32_t nt = 0; nt < num_node_types_; ++nt) {
            if (queued_nodes_[s][nt].empty())
                continue;

            for (int32_t src : queued_nodes_[s][nt]) {
                auto& src_res = residuals_[s][nt];
                auto it = src_res.find(src);
                double res = (it != src_res.end()) ? it->second : 0.0;

                // a. Absorb: move residual into the PPR score.
                ppr_scores_[s][nt][src] += res;
                src_res[src] = 0.0;

                // b. Count total fetched/cached neighbors across all edge types for
                // this source node.  We normalise by the number of neighbors we
                // actually retrieved, not the true degree, so residual is fully
                // distributed among known neighbors rather than leaking to unfetched
                // ones (which matters when num_neighbors_per_hop < true_degree).
                int32_t total_fetched = 0;
                for (int32_t eid : node_type_to_edge_type_ids_[nt]) {
                    auto fi = fetched.find(pack_key(src, eid));
                    if (fi != fetched.end()) {
                        total_fetched += static_cast<int32_t>(fi->second.size());
                    } else {
                        auto ci = neighbor_cache_.find(pack_key(src, eid));
                        if (ci != neighbor_cache_.end())
                            total_fetched += static_cast<int32_t>(ci->second.size());
                    }
                }
                // Destination-only nodes (or nodes with no fetched neighbors) absorb
                // residual but do not push further.
                if (total_fetched == 0)
                    continue;

                double res_per_nbr = one_minus_alpha_ * res / static_cast<double>(total_fetched);

                for (int32_t eid : node_type_to_edge_type_ids_[nt]) {
                    // Invariant: fetched and neighbor_cache_ are mutually exclusive for
                    // any given (node, etype) key within one iteration.  drain_queue()
                    // only requests a fetch for nodes absent from neighbor_cache_, so a
                    // key is in at most one of the two.
                    const std::vector<int32_t>* nbr_list = nullptr;
                    auto fi = fetched.find(pack_key(src, eid));
                    if (fi != fetched.end()) {
                        nbr_list = &fi->second;
                    } else {
                        auto ci = neighbor_cache_.find(pack_key(src, eid));
                        if (ci != neighbor_cache_.end())
                            nbr_list = &ci->second;
                    }
                    if (!nbr_list || nbr_list->empty())
                        continue;

                    int32_t dst_nt = edge_type_to_dst_ntype_id_[eid];

                    // c. Accumulate residual for each neighbor and re-enqueue if threshold
                    // exceeded.
                    for (int32_t nbr : *nbr_list) {
                        residuals_[s][dst_nt][nbr] += res_per_nbr;

                        double threshold =
                            requeue_threshold_factor_ * static_cast<double>(get_total_degree(nbr, dst_nt));

                        if (queue_[s][dst_nt].find(nbr) == queue_[s][dst_nt].end() &&
                            residuals_[s][dst_nt][nbr] >= threshold) {
                            queue_[s][dst_nt].insert(nbr);
                            ++num_nodes_in_queue_;

                            // Promote neighbor lists to the persistent cache: this node will
                            // be processed next iteration, so caching avoids a re-fetch.
                            for (int32_t peid : node_type_to_edge_type_ids_[dst_nt]) {
                                uint64_t pk = pack_key(nbr, peid);
                                if (neighbor_cache_.find(pk) == neighbor_cache_.end()) {
                                    auto pfi = fetched.find(pk);
                                    if (pfi != fetched.end())
                                        neighbor_cache_[pk] = pfi->second;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> PPRForwardPushState::extract_top_k(
    int32_t max_ppr_nodes) {
    std::unordered_set<int32_t> active;
    for (int32_t s = 0; s < batch_size_; ++s)
        for (int32_t nt = 0; nt < num_node_types_; ++nt)
            if (!ppr_scores_[s][nt].empty())
                active.insert(nt);

    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    for (int32_t nt : active) {
        std::vector<int64_t> flat_ids;
        std::vector<float> flat_weights;
        std::vector<int64_t> valid_counts;

        for (int32_t s = 0; s < batch_size_; ++s) {
            const auto& scores = ppr_scores_[s][nt];
            int32_t k = std::min(max_ppr_nodes, static_cast<int32_t>(scores.size()));
            if (k > 0) {
                std::vector<std::pair<int32_t, double>> items(scores.begin(), scores.end());
                std::partial_sort(items.begin(), items.begin() + k, items.end(), [](const auto& a, const auto& b) {
                    return a.second > b.second;
                });

                for (int32_t i = 0; i < k; ++i) {
                    flat_ids.push_back(static_cast<int64_t>(items[i].first));
                    // Cast to float32 for output; internal scores stay double to
                    // avoid accumulated rounding errors in the push loop.
                    flat_weights.push_back(static_cast<float>(items[i].second));
                }
            }
            valid_counts.push_back(static_cast<int64_t>(k));
        }

        result[nt] = {torch::tensor(flat_ids, torch::kLong),
                      torch::tensor(flat_weights, torch::kFloat),
                      torch::tensor(valid_counts, torch::kLong)};
    }
    return result;
}

int32_t PPRForwardPushState::get_total_degree(int32_t node_id, int32_t ntype_id) const {
    if (ntype_id >= static_cast<int32_t>(degree_tensors_.size()))
        return 0;
    const auto& t = degree_tensors_[ntype_id];
    if (t.numel() == 0)
        return 0;
    TORCH_CHECK(node_id < static_cast<int32_t>(t.size(0)),
                "Node ID ",
                node_id,
                " out of range for degree tensor of ntype_id ",
                ntype_id,
                " (size=",
                t.size(0),
                "). This indicates corrupted graph data or a sampler bug.");
    // data_ptr<int32_t>() returns a raw C pointer to the tensor's int32 data buffer.
    return t.data_ptr<int32_t>()[node_id];
}
