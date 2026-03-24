#include <torch/extension.h>  // PyTorch C++ API (tensors, TORCH_CHECK)
#include <pybind11/stl.h>      // Automatic conversion between C++ containers and Python types

#include <algorithm>    // std::partial_sort, std::min
#include <cstdint>      // Fixed-width integer types: int32_t, int64_t, uint32_t, uint64_t
#include <unordered_map>  // std::unordered_map — like Python dict, O(1) average lookup
#include <unordered_set>  // std::unordered_set — like Python set, O(1) average lookup
#include <vector>         // std::vector — like Python list, contiguous in memory

namespace py = pybind11;  // Alias for the pybind11 namespace (bridges C++ ↔ Python)

// Combine (node_id, etype_id) into a single 64-bit integer for use as a hash
// map key.  A single 64-bit integer is cheaper to hash than a pair of two
// integers (std::unordered_map has no built-in pair hash).
//
// Bit layout:
//   bits 63–32: node_id  (upper half)
//   bits 31– 0: etype_id (lower half)
//
// Both inputs are cast through uint32_t before packing.  Without this, a
// negative int32_t (e.g. -1 = 0xFFFFFFFF) would be sign-extended to a full
// 64-bit value, corrupting the upper bits when shifted.  Reinterpreting as
// uint32_t first treats the bit pattern as-is (no sign extension).
//
// `static inline` means: define this function here in the translation unit
// (not in a separate object file) and ask the compiler to inline it at each
// call site instead of generating a function call.
static inline uint64_t pack_key(int32_t node_id, int32_t etype_id) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(node_id)) << 32) |
           static_cast<uint32_t>(etype_id);
}

// C++ kernel for the PPR Forward Push algorithm (Andersen et al., 2006).
//
// All hot-loop state (scores, residuals, queue, neighbor cache) lives inside
// this object.  The distributed neighbor fetch is kept in Python because it
// involves async RPC calls that C++ cannot drive directly.
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
          // std::move transfers ownership of each vector into the member variable
          // without copying its contents — equivalent to Python's list hand-off
          // when you no longer need the original.
          node_type_to_edge_type_ids_(std::move(node_type_to_edge_type_ids)),
          edge_type_to_dst_ntype_id_(std::move(edge_type_to_dst_ntype_id)),
          degree_tensors_(std::move(degree_tensors)) {

        TORCH_CHECK(seed_nodes.dim() == 1, "seed_nodes must be 1D");
        batch_size_     = static_cast<int32_t>(seed_nodes.size(0));
        num_node_types_ = static_cast<int32_t>(node_type_to_edge_type_ids_.size());

        // Allocate per-seed, per-node-type tables.
        // .assign(n, val) fills a vector with n copies of val — like [val] * n in Python.
        // Each inner element is an empty hash map / hash set for that (seed, ntype) pair.
        ppr_scores_.assign(batch_size_,   std::vector<std::unordered_map<int32_t, double>>(num_node_types_));
        residuals_.assign(batch_size_,    std::vector<std::unordered_map<int32_t, double>>(num_node_types_));
        queue_.assign(batch_size_,        std::vector<std::unordered_set<int32_t>>(num_node_types_));
        queued_nodes_.assign(batch_size_, std::vector<std::unordered_set<int32_t>>(num_node_types_));

        // accessor<dtype, ndim>() returns a typed view into the tensor's data that
        // supports [i] indexing with bounds checking in debug builds.  Here we read
        // each seed node ID from the 1-D int64 tensor.
        auto acc = seed_nodes.accessor<int64_t, 1>();
        num_nodes_in_queue_ = batch_size_;
        for (int32_t i = 0; i < batch_size_; ++i) {
            // static_cast<int32_t>: explicit narrowing from int64 to int32.
            // The Python caller guarantees node IDs fit in 32 bits.
            int32_t seed = static_cast<int32_t>(acc[i]);
            // PPR initialisation: each seed starts with residual = alpha (the
            // restart probability).  The first push will move alpha into ppr_score
            // and distribute (1-alpha)*alpha to the seed's neighbors.
            residuals_[i][seed_node_type_id][seed] = alpha_;
            queue_[i][seed_node_type_id].insert(seed);
        }
    }

    // Drain all queued nodes and return {etype_id: tensor[node_ids]} for batch
    // neighbor lookup.  Also snapshots the drained nodes into queued_nodes_ for
    // use by push_residuals().
    //
    // Return value semantics (py::object can hold any Python value):
    //   - py::none()  → queue was already empty; convergence achieved; stop the loop.
    //   - py::dict{}  → nodes were drained.  The dict maps etype_id → 1-D int64
    //                   tensor of node IDs that need neighbor lookups this round.
    //                   May be empty if all drained nodes were already in the cache
    //                   or had no outgoing edges — push_residuals must still be called
    //                   to flush their accumulated residual into ppr_scores_.
    py::object drain_queue() {
        if (num_nodes_in_queue_ == 0) {
            return py::none();
        }

        // Reset the snapshot from the previous iteration.  `auto&` is a reference
        // (alias) to the existing set — clearing it modifies the original in-place
        // rather than operating on a copy.
        for (int32_t s = 0; s < batch_size_; ++s)
            for (auto& qs : queued_nodes_[s]) qs.clear();

        // nodes_to_lookup[eid] = set of node IDs that need a neighbor fetch for
        // edge type eid this round.  Using a set deduplicates nodes that appear
        // in multiple seeds' queues: we only fetch each (node, etype) pair once
        // regardless of how many seeds need it.
        std::unordered_map<int32_t, std::unordered_set<int32_t>> nodes_to_lookup;

        for (int32_t s = 0; s < batch_size_; ++s) {
            for (int32_t nt = 0; nt < num_node_types_; ++nt) {
                if (queue_[s][nt].empty()) continue;

                // Move the live queue into the snapshot (no data copy — O(1)).
                // queue_ is then reset to an empty set so new entries added by
                // push_residuals() in this same iteration don't interfere.
                queued_nodes_[s][nt] = std::move(queue_[s][nt]);
                queue_[s][nt].clear();
                num_nodes_in_queue_ -= static_cast<int32_t>(queued_nodes_[s][nt].size());

                for (int32_t node_id : queued_nodes_[s][nt]) {
                    for (int32_t eid : node_type_to_edge_type_ids_[nt]) {
                        // Only request a fetch if the neighbor list isn't already
                        // cached from a previous iteration.
                        if (neighbor_cache_.find(pack_key(node_id, eid)) == neighbor_cache_.end()) {
                            nodes_to_lookup[eid].insert(node_id);
                        }
                    }
                }
            }
        }

        // Convert to Python: {etype_id (int) → 1-D int64 tensor of node IDs}.
        // py::int_(eid) wraps a C++ int as a Python int so it can be used as a
        // dict key on the Python side.
        py::dict result;
        for (auto& [eid, node_set] : nodes_to_lookup) {
            // Copy the set into a vector first: torch::tensor() requires a
            // contiguous sequence, not an unordered_set iterator.
            std::vector<int64_t> ids(node_set.begin(), node_set.end());
            result[py::int_(eid)] = torch::tensor(ids, torch::kLong);
        }
        return result;
    }

    // Push residuals to neighbors given the fetched neighbor data.
    //
    // fetched_by_etype_id: {etype_id: (node_ids_tensor, flat_nbrs_tensor, counts_tensor)}
    //   - node_ids_tensor:  [N]           int64 — source node IDs fetched for this edge type
    //   - flat_nbrs_tensor: [sum(counts)] int64 — all neighbor lists concatenated flat
    //   - counts_tensor:    [N]           int64 — neighbor count for each source node
    //
    // For example, if nodes 3 and 7 were fetched for etype 0:
    //   node_ids  = [3, 7]
    //   flat_nbrs = [10, 11, 12, 20]   ← node 3 has nbrs {10,11,12}, node 7 has nbr {20}
    //   counts    = [3, 1]
    void push_residuals(py::dict fetched_by_etype_id) {
        // Step 1: Unpack the Python dict into a C++ map for fast lookup during
        // the residual-push loop below.
        // fetched: pack_key(node_id, etype_id) → neighbor list (as int32_t vector)
        std::unordered_map<uint64_t, std::vector<int32_t>> fetched;
        for (auto item : fetched_by_etype_id) {
            int32_t eid = item.first.cast<int32_t>();
            // .cast<py::tuple>() interprets the Python value as a tuple so we
            // can index into it with [0], [1], [2].
            auto tup         = item.second.cast<py::tuple>();
            auto node_ids_t  = tup[0].cast<torch::Tensor>();
            auto flat_nbrs_t = tup[1].cast<torch::Tensor>();
            auto counts_t    = tup[2].cast<torch::Tensor>();

            // accessor<int64_t, 1>() gives a bounds-checked, typed 1-D view into
            // each tensor's data — equivalent to iterating over a NumPy array.
            auto node_acc = node_ids_t.accessor<int64_t, 1>();
            auto nbr_acc  = flat_nbrs_t.accessor<int64_t, 1>();
            auto cnt_acc  = counts_t.accessor<int64_t, 1>();

            // Walk the flat neighbor list, slicing out each node's neighbors using
            // the running offset into the concatenated flat buffer.
            int64_t offset = 0;
            for (int64_t i = 0; i < node_ids_t.size(0); ++i) {
                int32_t nid   = static_cast<int32_t>(node_acc[i]);
                int64_t count = cnt_acc[i];
                std::vector<int32_t> nbrs(count);
                for (int64_t j = 0; j < count; ++j)
                    nbrs[j] = static_cast<int32_t>(nbr_acc[offset + j]);
                // std::move: hand off nbrs to the map without copying its contents.
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
                if (queued_nodes_[s][nt].empty()) continue;

                for (int32_t src : queued_nodes_[s][nt]) {
                    // `auto&` gives a reference to the residual map for this
                    // (seed, node_type) pair so we can read and write it without
                    // an extra hash lookup each time.
                    auto& src_res = residuals_[s][nt];
                    // .find() returns an iterator; .end() means "not found".
                    // We treat a missing entry as residual = 0.
                    auto it = src_res.find(src);
                    double res = (it != src_res.end()) ? it->second : 0.0;

                    // a. Absorb: move residual into the PPR score.
                    ppr_scores_[s][nt][src] += res;
                    src_res[src] = 0.0;

                    int32_t total_deg = get_total_degree(src, nt);
                    // Destination-only nodes (no outgoing edges) absorb residual
                    // into their PPR score but do not push further.
                    if (total_deg == 0) continue;

                    // b. Distribute: each neighbor of src (across all edge types
                    // from nt) receives an equal share of the pushed residual.
                    double res_per_nbr = one_minus_alpha_ * res / static_cast<double>(total_deg);

                    for (int32_t eid : node_type_to_edge_type_ids_[nt]) {
                        // Invariant: fetched and neighbor_cache_ are mutually exclusive for
                        // any given (node, etype) key within one iteration.  drain_queue()
                        // only requests a fetch for nodes absent from neighbor_cache_, so a
                        // key is in at most one of the two.  We check fetched first since it
                        // is the common case for newly-seen nodes.
                        //
                        // `const std::vector<int32_t>*` is a pointer to a neighbor list.
                        // We use a pointer (rather than copying the list) so we can check
                        // for absence with nullptr without allocating anything.
                        const std::vector<int32_t>* nbr_list = nullptr;
                        auto fi = fetched.find(pack_key(src, eid));
                        if (fi != fetched.end()) {
                            // `&fi->second` takes the address of the vector stored in
                            // the map — nbr_list now points to it without copying.
                            nbr_list = &fi->second;
                        } else {
                            auto ci = neighbor_cache_.find(pack_key(src, eid));
                            if (ci != neighbor_cache_.end()) nbr_list = &ci->second;
                        }
                        // Skip if no neighbor list is available (node has no edges of
                        // this type, or the fetch returned an empty list).
                        if (!nbr_list || nbr_list->empty()) continue;

                        int32_t dst_nt = edge_type_to_dst_ntype_id_[eid];

                        // c. For each neighbor, accumulate residual and check threshold.
                        // `*nbr_list` dereferences the pointer to access the vector.
                        for (int32_t nbr : *nbr_list) {
                            residuals_[s][dst_nt][nbr] += res_per_nbr;

                            double threshold = requeue_threshold_factor_ *
                                static_cast<double>(get_total_degree(nbr, dst_nt));

                            // Only enqueue if: (1) not already in queue for this
                            // iteration, and (2) residual exceeds the push threshold
                            // alpha * eps * degree.
                            if (queue_[s][dst_nt].find(nbr) == queue_[s][dst_nt].end() &&
                                residuals_[s][dst_nt][nbr] >= threshold) {
                                queue_[s][dst_nt].insert(nbr);
                                ++num_nodes_in_queue_;  // ++x is equivalent to x += 1

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
    //
    // Returns {ntype_id: (flat_ids_tensor, flat_weights_tensor, valid_counts_tensor)}.
    // Only node types that received any PPR score are included in the output.
    //
    // Output layout for a batch of B seeds (same structure as _batch_fetch_neighbors):
    //   flat_ids[0 : valid_counts[0]]                 → top-k nodes for seed 0
    //   flat_ids[valid_counts[0] : valid_counts[0]+valid_counts[1]] → top-k for seed 1
    //   ...
    py::dict extract_top_k(int32_t max_ppr_nodes) {
        // Collect node types that have any PPR score — skip types with no activity.
        std::unordered_set<int32_t> active;
        for (int32_t s = 0; s < batch_size_; ++s)
            for (int32_t nt = 0; nt < num_node_types_; ++nt)
                if (!ppr_scores_[s][nt].empty()) active.insert(nt);

        py::dict result;
        for (int32_t nt : active) {
            // Flat output vectors — entries for all seeds are concatenated.
            std::vector<int64_t> flat_ids;
            std::vector<float>   flat_weights;
            std::vector<int64_t> valid_counts;

            for (int32_t s = 0; s < batch_size_; ++s) {
                // `const auto&` is a read-only reference — we iterate the map
                // without copying it.
                const auto& scores = ppr_scores_[s][nt];
                // Cap k at the number of nodes that actually have a score.
                int32_t k = std::min(max_ppr_nodes, static_cast<int32_t>(scores.size()));
                if (k > 0) {
                    // Copy the map entries into a vector of (node_id, score) pairs
                    // so they can be sorted.  std::pair is like a Python 2-tuple.
                    std::vector<std::pair<int32_t, double>> items(scores.begin(), scores.end());

                    // std::partial_sort rearranges items so that the first k entries
                    // are the k largest — like Python's heapq.nlargest but in-place.
                    // The lambda `[](const auto& a, const auto& b) { return ...; }`
                    // is an anonymous comparator (like Python's `key=` argument).
                    // `.second` accesses the score (second element of the pair);
                    // `>` makes it descending (highest score first).
                    std::partial_sort(items.begin(), items.begin() + k, items.end(),
                        [](const auto& a, const auto& b) { return a.second > b.second; });

                    for (int32_t i = 0; i < k; ++i) {
                        flat_ids.push_back(static_cast<int64_t>(items[i].first));
                        // Cast to float32 for output; internal scores stay double to
                        // avoid accumulated rounding errors in the push loop above.
                        flat_weights.push_back(static_cast<float>(items[i].second));
                    }
                }
                valid_counts.push_back(static_cast<int64_t>(k));
            }

            // py::make_tuple wraps C++ values into a Python tuple.
            result[py::int_(nt)] = py::make_tuple(
                torch::tensor(flat_ids, torch::kLong),
                torch::tensor(flat_weights, torch::kFloat),
                torch::tensor(valid_counts, torch::kLong)
            );
        }
        return result;
    }

private:
    // Look up the total (across all edge types) out-degree of a node.
    // Returns 0 for destination-only node types (no outgoing edges).
    int32_t get_total_degree(int32_t node_id, int32_t ntype_id) const {
        if (ntype_id >= static_cast<int32_t>(degree_tensors_.size())) return 0;
        const auto& t = degree_tensors_[ntype_id];
        if (t.numel() == 0) return 0;  // destination-only type: no outgoing edges
        TORCH_CHECK(
            node_id < static_cast<int32_t>(t.size(0)),
            "Node ID ", node_id, " out of range for degree tensor of ntype_id ", ntype_id,
            " (size=", t.size(0), "). This indicates corrupted graph data or a sampler bug."
        );
        // data_ptr<int32_t>() returns a raw C pointer to the tensor's int32 data
        // buffer.  Direct pointer indexing ([node_id]) is safe here because we
        // validated the bounds with TORCH_CHECK above.
        return t.data_ptr<int32_t>()[node_id];
    }

    // -------------------------------------------------------------------------
    // Scalar algorithm parameters
    // -------------------------------------------------------------------------
    double  alpha_;                       // Restart probability
    double  one_minus_alpha_;             // 1 - alpha, precomputed to avoid repeated subtraction
    double  requeue_threshold_factor_;    // alpha * eps; multiplied by degree to get per-node threshold

    int32_t batch_size_;                  // Number of seeds in the current batch
    int32_t num_node_types_;              // Total number of node types (homo + hetero)
    int32_t num_nodes_in_queue_{0};       // Running count of nodes across all seeds / types

    // -------------------------------------------------------------------------
    // Graph structure (read-only after construction)
    // -------------------------------------------------------------------------
    // node_type_to_edge_type_ids_[ntype_id] → list of edge type IDs that can be
    // traversed from that node type (outgoing or incoming, depending on edge_dir).
    std::vector<std::vector<int32_t>> node_type_to_edge_type_ids_;
    // edge_type_to_dst_ntype_id_[eid] → node type ID at the destination end.
    std::vector<int32_t>              edge_type_to_dst_ntype_id_;
    // degree_tensors_[ntype_id][node_id] → total degree of that node across all
    // edge types traversable from its type.  Empty tensor means no outgoing edges.
    std::vector<torch::Tensor>        degree_tensors_;

    // -------------------------------------------------------------------------
    // Per-seed, per-node-type PPR state (indexed [seed_idx][ntype_id])
    // -------------------------------------------------------------------------
    // double precision avoids float32 rounding errors accumulating over 20-30
    // push iterations, which would otherwise cause ~1e-4 score errors vs the
    // true PPR.  Output weights are cast to float32 in extract_top_k.
    //
    // ppr_scores_[s][nt]: node_id → absorbed PPR score (Σ of residuals pushed so far)
    std::vector<std::vector<std::unordered_map<int32_t, double>>> ppr_scores_;
    // residuals_[s][nt]: node_id → unabsorbed probability mass waiting to be pushed
    std::vector<std::vector<std::unordered_map<int32_t, double>>> residuals_;
    // queue_[s][nt]: nodes whose residual exceeds the threshold and need a push next round
    std::vector<std::vector<std::unordered_set<int32_t>>>         queue_;
    // queued_nodes_[s][nt]: snapshot of queue_ taken by drain_queue() for the current round.
    // Separating it from queue_ lets push_residuals() enqueue new nodes into queue_ without
    // modifying the set currently being iterated.
    std::vector<std::vector<std::unordered_set<int32_t>>>         queued_nodes_;

    // -------------------------------------------------------------------------
    // Neighbor cache
    // -------------------------------------------------------------------------
    // Persistent cache: pack_key(node_id, etype_id) → neighbor list.
    // Only nodes that have been re-queued (and will therefore be processed again)
    // are promoted here from the per-iteration fetched map in push_residuals().
    // This avoids re-fetching neighbors for nodes processed in multiple iterations
    // while keeping large neighbor lists of high-degree (never-requeued) nodes
    // out of memory.
    std::unordered_map<uint64_t, std::vector<int32_t>> neighbor_cache_;
};

// Register PPRForwardPushState with Python via pybind11.
//
// TORCH_EXTENSION_NAME is set by PyTorch's setup() at build time to match the
// Python module name (e.g. "ppr_forward_push").  At import time, Python calls
// this function to populate the module with the C++ class.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PPRForwardPushState>(m, "PPRForwardPushState")
        // .def(py::init<...>()) exposes the constructor.  The template arguments
        // list the exact C++ parameter types so pybind11 can convert Python
        // arguments to the correct C++ types automatically.
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
