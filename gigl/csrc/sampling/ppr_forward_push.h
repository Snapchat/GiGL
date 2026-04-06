#pragma once

#include <torch/torch.h>

#include <algorithm>      // std::partial_sort, std::min
#include <cstdint>        // Fixed-width integer types: int32_t, int64_t, uint32_t, uint64_t
#include <optional>       // std::optional for nullable return values
#include <tuple>          // std::tuple for multi-value returns
#include <unordered_map>  // std::unordered_map — like Python dict, O(1) average lookup
#include <unordered_set>  // std::unordered_set — like Python set, O(1) average lookup
#include <vector>         // std::vector — like Python list, contiguous in memory

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
    PPRForwardPushState(torch::Tensor seed_nodes, int32_t seed_node_type_id, double alpha,
                        double requeue_threshold_factor,
                        std::vector<std::vector<int32_t>> node_type_to_edge_type_ids,
                        std::vector<int32_t> edge_type_to_dst_ntype_id,
                        std::vector<torch::Tensor> degree_tensors);

    // Drain all queued nodes and return {etype_id: tensor[node_ids]} for batch
    // neighbor lookup.  Also snapshots the drained nodes into queued_nodes_ for
    // use by push_residuals().
    //
    // Return value semantics:
    //   - std::nullopt   → queue was already empty; convergence achieved; stop the loop.
    //   - empty map      → nodes were drained but all were cached; call push_residuals({}).
    //   - non-empty map  → {etype_id → 1-D int64 tensor of node IDs} needing neighbor lookup.
    std::optional<std::unordered_map<int32_t, torch::Tensor>> drain_queue();

    // Push residuals to neighbors given the fetched neighbor data.
    //
    // fetched_by_etype_id: {etype_id: (node_ids_tensor, flat_nbrs_tensor, counts_tensor)}
    //   - node_ids_tensor:  [N]           int64 — source node IDs fetched for this edge type
    //   - flat_nbrs_tensor: [sum(counts)] int64 — all neighbor lists concatenated flat
    //   - counts_tensor:    [N]           int64 — neighbor count for each source node
    void push_residuals(const std::unordered_map<
                        int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>&
                            fetched_by_etype_id);

    // Extract top-k PPR nodes per seed per node type.
    //
    // Returns {ntype_id: (flat_ids_tensor, flat_weights_tensor, valid_counts_tensor)}.
    // Only node types that received any PPR score are included in the output.
    //
    // Output layout for a batch of B seeds:
    //   flat_ids[0 : valid_counts[0]]                 → top-k nodes for seed 0
    //   flat_ids[valid_counts[0] : valid_counts[0]+valid_counts[1]] → top-k for seed 1
    //   ...
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
    extract_top_k(int32_t max_ppr_nodes);

   private:
    // Look up the total (across all edge types) out-degree of a node.
    // Returns 0 for destination-only node types (no outgoing edges).
    int32_t get_total_degree(int32_t node_id, int32_t ntype_id) const;

    // -------------------------------------------------------------------------
    // Scalar algorithm parameters
    // -------------------------------------------------------------------------
    double alpha_;            // Restart probability
    double one_minus_alpha_;  // 1 - alpha, precomputed to avoid repeated subtraction
    double requeue_threshold_factor_;  // alpha * eps; multiplied by degree to get per-node threshold

    int32_t batch_size_;             // Number of seeds in the current batch
    int32_t num_node_types_;         // Total number of node types (homo + hetero)
    int32_t num_nodes_in_queue_{0};  // Running count of nodes across all seeds / types

    // -------------------------------------------------------------------------
    // Graph structure (read-only after construction)
    // -------------------------------------------------------------------------
    std::vector<std::vector<int32_t>> node_type_to_edge_type_ids_;
    std::vector<int32_t> edge_type_to_dst_ntype_id_;
    std::vector<torch::Tensor> degree_tensors_;

    // -------------------------------------------------------------------------
    // Per-seed, per-node-type PPR state (indexed [seed_idx][ntype_id])
    // -------------------------------------------------------------------------
    std::vector<std::vector<std::unordered_map<int32_t, double>>> ppr_scores_;
    std::vector<std::vector<std::unordered_map<int32_t, double>>> residuals_;
    std::vector<std::vector<std::unordered_set<int32_t>>> queue_;
    std::vector<std::vector<std::unordered_set<int32_t>>> queued_nodes_;

    // -------------------------------------------------------------------------
    // Neighbor cache
    // -------------------------------------------------------------------------
    std::unordered_map<uint64_t, std::vector<int32_t>> neighbor_cache_;

    // -------------------------------------------------------------------------
    // Diagnostics (populated during the algorithm; read after convergence)
    // -------------------------------------------------------------------------
    // Total nodes drained (across all seeds and node types) in each drain_queue()
    // call.  One entry per loop iteration; useful for understanding convergence speed.
    std::vector<int32_t> nodes_drained_per_iteration_;

   public:
    // Returns nodes_drained_per_iteration_ built up across all drain_queue() calls.
    const std::vector<int32_t>& get_nodes_drained_per_iteration() const;
};
