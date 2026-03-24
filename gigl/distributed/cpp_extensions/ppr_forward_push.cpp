#include <torch/extension.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// Pack (node_id, etype_id) into a single uint64_t lookup key.
// Requires both values fit in 32 bits — enforced by the Python caller.
static inline uint64_t pack_key(int32_t node_id, int32_t etype_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(node_id)) << 32) |
           static_cast<uint32_t>(etype_id);
}

// C++ kernel for the PPR Forward Push algorithm (Andersen et al., 2006).
//
// Owned state: ppr_scores, residuals, queue, queued_nodes, neighbor_cache.
// Python retains ownership of: the distributed neighbor fetch (_batch_fetch_neighbors).
//
// Typical call sequence per batch:
//   1.  PPRForwardPushState(seed_nodes, ...)   — init per-seed residuals / queue
//   while True:
//   2.  drain_queue()                          — drain queue → nodes needing lookup
//   3.  <Python: _batch_fetch_neighbors(...)>  — distributed RPC fetch (stays in Python)
//   4.  push_residuals(fetched_by_etype_id)    — push residuals, update queue
//   5.  extract_top_k(max_ppr_nodes)           — top-k selection per seed per node type
class PPRForwardPushState {
public:
    PPRForwardPushState(
        torch::Tensor seed_nodes,
        int32_t seed_node_type_id,
        double alpha,
        double requeue_threshold_factor,
        std::vector<std::vector<int32_t>> node_type_to_edge_type_ids,
        std::vector<int32_t> edge_type_to_dst_ntype_id,
        std::vector<torch::Tensor> degree_tensors
    )
        : alpha_(alpha),
          one_minus_alpha_(1.0 - alpha),
          requeue_threshold_factor_(requeue_threshold_factor),
          node_type_to_edge_type_ids_(std::move(node_type_to_edge_type_ids)),
          edge_type_to_dst_ntype_id_(std::move(edge_type_to_dst_ntype_id)),
          degree_tensors_(std::move(degree_tensors)) {

        TORCH_CHECK(seed_nodes.dim() == 1, "seed_nodes must be 1D");
        batch_size_     = static_cast<int32_t>(seed_nodes.size(0));
        num_node_types_ = static_cast<int32_t>(node_type_to_edge_type_ids_.size());

        ppr_scores_.assign(batch_size_,    std::vector<std::unordered_map<int32_t, double>>(num_node_types_));
        residuals_.assign(batch_size_,     std::vector<std::unordered_map<int32_t, double>>(num_node_types_));
        queue_.assign(batch_size_,         std::vector<std::unordered_set<int32_t>>(num_node_types_));
        queued_nodes_.assign(batch_size_,  std::vector<std::unordered_set<int32_t>>(num_node_types_));

        auto acc = seed_nodes.accessor<int64_t, 1>();
        num_nodes_in_queue_ = batch_size_;
        for (int32_t i = 0; i < batch_size_; ++i) {
            int32_t seed = static_cast<int32_t>(acc[i]);
            residuals_[i][seed_node_type_id][seed] = alpha_;
            queue_[i][seed_node_type_id].insert(seed);
        }
    }

    // Drain all queued nodes and return {etype_id: tensor[node_ids]} for batch
    // neighbor lookup.  Also snapshots the drained nodes into queued_nodes_ for
    // use by push_residuals().
    //
    // Returns None when the queue is truly empty (convergence signal).
    // Returns a dict (possibly empty) when nodes were drained but all had cached
    // neighbors or no outgoing edges — push_residuals must still be called to
    // flush their residuals into ppr_scores_.
    py::object drain_queue() {
        if (num_nodes_in_queue_ == 0) {
            return py::none();
        }

        for (int32_t s = 0; s < batch_size_; ++s)
            for (auto& qs : queued_nodes_[s]) qs.clear();

        std::unordered_map<int32_t, std::unordered_set<int32_t>> nodes_to_lookup;

        for (int32_t s = 0; s < batch_size_; ++s) {
            for (int32_t nt = 0; nt < num_node_types_; ++nt) {
                if (queue_[s][nt].empty()) continue;

                // Snapshot queue into queued_nodes, then reset queue.
                queued_nodes_[s][nt] = std::move(queue_[s][nt]);
                queue_[s][nt].clear();
                num_nodes_in_queue_ -= static_cast<int32_t>(queued_nodes_[s][nt].size());

                for (int32_t node_id : queued_nodes_[s][nt]) {
                    for (int32_t eid : node_type_to_edge_type_ids_[nt]) {
                        // Only add to lookup if not already in the persistent cache.
                        if (neighbor_cache_.find(pack_key(node_id, eid)) == neighbor_cache_.end()) {
                            nodes_to_lookup[eid].insert(node_id);
                        }
                    }
                }
            }
        }

        py::dict result;
        for (auto& [eid, node_set] : nodes_to_lookup) {
            std::vector<int64_t> ids(node_set.begin(), node_set.end());
            result[py::int_(eid)] = torch::tensor(ids, torch::kLong);
        }
        return result;
    }

    // Push residuals to neighbors given the fetched neighbor data.
    // fetched_by_etype_id: {etype_id: (node_ids_tensor, flat_nbrs_tensor, counts_tensor)}
    //   - node_ids_tensor:  [N] int64 — source node IDs fetched for this edge type
    //   - flat_nbrs_tensor: [sum(counts)] int64 — flat concatenation of all neighbor lists
    //   - counts_tensor:    [N] int64 — number of neighbors for each source node
    void push_residuals(py::dict fetched_by_etype_id) {
        // Build local fetched map: pack_key(node_id, etype_id) -> neighbor list.
        std::unordered_map<uint64_t, std::vector<int32_t>> fetched;
        for (auto item : fetched_by_etype_id) {
            int32_t eid      = item.first.cast<int32_t>();
            auto tup         = item.second.cast<py::tuple>();
            auto node_ids_t  = tup[0].cast<torch::Tensor>();
            auto flat_nbrs_t = tup[1].cast<torch::Tensor>();
            auto counts_t    = tup[2].cast<torch::Tensor>();

            auto node_acc  = node_ids_t.accessor<int64_t, 1>();
            auto nbr_acc   = flat_nbrs_t.accessor<int64_t, 1>();
            auto cnt_acc   = counts_t.accessor<int64_t, 1>();

            int64_t offset = 0;
            for (int64_t i = 0; i < node_ids_t.size(0); ++i) {
                int32_t nid   = static_cast<int32_t>(node_acc[i]);
                int64_t count = cnt_acc[i];
                std::vector<int32_t> nbrs(count);
                for (int64_t j = 0; j < count; ++j)
                    nbrs[j] = static_cast<int32_t>(nbr_acc[offset + j]);
                fetched[pack_key(nid, eid)] = std::move(nbrs);
                offset += count;
            }
        }

        for (int32_t s = 0; s < batch_size_; ++s) {
            for (int32_t nt = 0; nt < num_node_types_; ++nt) {
                if (queued_nodes_[s][nt].empty()) continue;

                for (int32_t src : queued_nodes_[s][nt]) {
                    auto& src_res = residuals_[s][nt];
                    auto it = src_res.find(src);
                    double res = (it != src_res.end()) ? it->second : 0.0;

                    ppr_scores_[s][nt][src] += res;
                    src_res[src] = 0.0;

                    int32_t total_deg = get_total_degree(src, nt);
                    if (total_deg == 0) continue;

                    double res_per_nbr = one_minus_alpha_ * res / static_cast<double>(total_deg);

                    for (int32_t eid : node_type_to_edge_type_ids_[nt]) {
                        // fetched and neighbor_cache are mutually exclusive per iteration:
                        // drain_queue only adds a node to nodes_to_lookup when absent from
                        // neighbor_cache, so a given key appears in at most one of the two.
                        const std::vector<int32_t>* nbr_list = nullptr;
                        auto fi = fetched.find(pack_key(src, eid));
                        if (fi != fetched.end()) {
                            nbr_list = &fi->second;
                        } else {
                            auto ci = neighbor_cache_.find(pack_key(src, eid));
                            if (ci != neighbor_cache_.end()) nbr_list = &ci->second;
                        }
                        if (!nbr_list || nbr_list->empty()) continue;

                        int32_t dst_nt = edge_type_to_dst_ntype_id_[eid];

                        for (int32_t nbr : *nbr_list) {
                            residuals_[s][dst_nt][nbr] += res_per_nbr;

                            double threshold = requeue_threshold_factor_ *
                                static_cast<double>(get_total_degree(nbr, dst_nt));

                            if (queue_[s][dst_nt].find(nbr) == queue_[s][dst_nt].end() &&
                                residuals_[s][dst_nt][nbr] >= threshold) {
                                queue_[s][dst_nt].insert(nbr);
                                ++num_nodes_in_queue_;

                                // Promote this node's neighbor lists to the persistent cache:
                                // it will be processed next iteration, so caching now avoids
                                // a re-fetch.  Nodes that are never requeued (typically
                                // high-degree) are never promoted, keeping their large neighbor
                                // lists out of the cache.
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

    // Extract top-k PPR nodes per seed per node type.
    // Returns {ntype_id: (flat_ids_tensor, flat_weights_tensor, valid_counts_tensor)}.
    // Only node types that received any PPR score are included in the output.
    py::dict extract_top_k(int32_t max_ppr_nodes) {
        std::unordered_set<int32_t> active;
        for (int32_t s = 0; s < batch_size_; ++s)
            for (int32_t nt = 0; nt < num_node_types_; ++nt)
                if (!ppr_scores_[s][nt].empty()) active.insert(nt);

        py::dict result;
        for (int32_t nt : active) {
            std::vector<int64_t> flat_ids;
            std::vector<float>   flat_weights;
            std::vector<int64_t> valid_counts;

            for (int32_t s = 0; s < batch_size_; ++s) {
                const auto& scores = ppr_scores_[s][nt];
                int32_t k = std::min(max_ppr_nodes, static_cast<int32_t>(scores.size()));
                if (k > 0) {
                    std::vector<std::pair<int32_t, double>> items(scores.begin(), scores.end());
                    std::partial_sort(items.begin(), items.begin() + k, items.end(),
                        [](const auto& a, const auto& b) { return a.second > b.second; });
                    for (int32_t i = 0; i < k; ++i) {
                        flat_ids.push_back(static_cast<int64_t>(items[i].first));
                        flat_weights.push_back(static_cast<float>(items[i].second));
                    }
                }
                valid_counts.push_back(static_cast<int64_t>(k));
            }

            result[py::int_(nt)] = py::make_tuple(
                torch::tensor(flat_ids, torch::kLong),
                torch::tensor(flat_weights, torch::kFloat),
                torch::tensor(valid_counts, torch::kLong)
            );
        }
        return result;
    }

private:
    int32_t get_total_degree(int32_t node_id, int32_t ntype_id) const {
        if (ntype_id >= static_cast<int32_t>(degree_tensors_.size())) return 0;
        const auto& t = degree_tensors_[ntype_id];
        if (t.numel() == 0) return 0;  // destination-only type: no outgoing edges
        TORCH_CHECK(
            node_id < static_cast<int32_t>(t.size(0)),
            "Node ID ", node_id, " out of range for degree tensor of ntype_id ", ntype_id,
            " (size=", t.size(0), "). This indicates corrupted graph data or a sampler bug."
        );
        return t.data_ptr<int32_t>()[node_id];
    }

    double  alpha_, one_minus_alpha_, requeue_threshold_factor_;
    int32_t batch_size_, num_node_types_, num_nodes_in_queue_{0};

    std::vector<std::vector<int32_t>> node_type_to_edge_type_ids_;
    std::vector<int32_t>              edge_type_to_dst_ntype_id_;
    std::vector<torch::Tensor>        degree_tensors_;

    // Per-seed, per-node-type PPR state (indexed [seed_idx][ntype_id]).
    // double precision avoids float32 rounding errors accumulating over 20-30
    // push iterations, which would otherwise cause ~1e-4 score errors vs the
    // true PPR.  Output weights are cast to float32 in extract_top_k.
    std::vector<std::vector<std::unordered_map<int32_t, double>>> ppr_scores_;
    std::vector<std::vector<std::unordered_map<int32_t, double>>> residuals_;
    std::vector<std::vector<std::unordered_set<int32_t>>>         queue_;
    // Snapshot of queue contents from the last drain_queue() call, used by push_residuals().
    std::vector<std::vector<std::unordered_set<int32_t>>>         queued_nodes_;

    // Persistent neighbor cache: pack_key(node_id, etype_id) -> neighbor list.
    // Only nodes that have been requeued (and thus will be processed again) are
    // promoted here from the per-iteration fetched map.
    std::unordered_map<uint64_t, std::vector<int32_t>> neighbor_cache_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PPRForwardPushState>(m, "PPRForwardPushState")
        .def(py::init<
            torch::Tensor,
            int32_t,
            double, double,
            std::vector<std::vector<int32_t>>,
            std::vector<int32_t>,
            std::vector<torch::Tensor>
        >())
        .def("drain_queue",    &PPRForwardPushState::drain_queue)
        .def("push_residuals", &PPRForwardPushState::push_residuals)
        .def("extract_top_k",  &PPRForwardPushState::extract_top_k);
}
