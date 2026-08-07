#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gigl {

// Neighbor fetch input from Python, keyed by integer edge type ID:
//   node_ids[N], flat_neighbor_ids[sum(counts)], counts[N], optional edge_ids[sum(counts)].
using NeighborFetchTensors = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>>;
using NeighborFetchMap = std::unordered_map<int32_t, NeighborFetchTensors>;

// PPR extraction output, keyed by integer node type ID:
//   ids, weights/edge_attr, valid_counts.
using PPRExtractTensors = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;
using PPRExtractResult = std::unordered_map<int32_t, PPRExtractTensors>;

// Original-edge extraction output, keyed by integer edge type ID:
//   rows, cols, optional edge_ids.
using OriginalEdgeExtractTensors = std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>>;
using OriginalEdgeExtractResult = std::unordered_map<int32_t, OriginalEdgeExtractTensors>;

// Per-seed, per-node-type PPR algorithm state.
// Grouping all four tables into one struct is a logical convenience: a single
// _state[seedIdx][nodeTypeId] access reaches all four tables for a given (seed, ntype)
// pair, rather than indexing four separate 2D arrays.  Note that unordered_map and
// unordered_set heap-allocate their bucket storage, so the actual key-value data is
// not co-located in memory — only the control-plane metadata (size, bucket pointer)
// lives inside the struct.
struct SeedNodeTypeState {
    std::unordered_map<int32_t, double> pprScores; // absorbed PPR mass
    std::unordered_map<int32_t, double> residuals; // unabsorbed mass waiting to push
    std::unordered_set<int32_t> queue;             // nodes queued for the next drain
    std::unordered_set<int32_t> queuedNodes;       // snapshot captured by drainQueue()
};

struct CachedNeighborList {
    std::vector<int32_t> neighborIds;
    std::optional<std::vector<int64_t>> edgeIds;
};

// Batched drain result for typed-PPR channels.
//
// Typed PPR keeps one PPRForwardPush state per channel.  During an iteration,
// each channel drains its own queue, but Python should issue at most one shared
// neighbor fetch per edge type. This struct carries both pieces of information:
// which channel states still need pushResiduals(), and the unioned frontier to
// fetch once for all channels that requested it.
struct TypedPPRQueueDrainResult {
    // Channels whose drainQueue() returned a value this iteration. Channel IDs
    // are positional indices into the states/channel-target vectors. Python
    // builds those vectors from typed-channel insertion order, and this function
    // appends indices in ascending order, so the ordering is stable.
    //
    // These channels need pushResiduals(), even when no fetch budget remains.
    // In that case Python passes an empty fetch map and the channel uses its
    // existing neighbor cache / budget-exhausted behavior, matching untyped PPR.
    std::vector<int32_t> drainedChannelIndices;

    // Subset of drainedChannelIndices that still have fetch budget and at least
    // one non-empty uncached frontier.
    std::vector<int32_t> fetchChannelIndices;

    // Edge types requested by each fetch channel, aligned with fetchChannelIndices.
    std::vector<std::vector<int32_t>> edgeTypeIdsByFetchChannel;

    // Unioned node frontier for one shared distributed neighbor fetch. This is
    // keyed by integer edge type ID, not node type ID, because neighbor fetches
    // are edge-type scoped; node type alone would lose the relation/destination
    // distinction for heterogeneous graphs. Tensor values are int64 source node
    // IDs to fetch.
    std::unordered_map<int32_t, torch::Tensor> unionedNodeIdsByEdgeTypeId;
};

// C++ kernel for PPR Forward Push (Andersen et al., 2006).
// Hot-loop state lives here; distributed neighbor fetches are driven from Python.
//
// Call sequence per batch:
//   1. PPRForwardPush(seedNodes, ...)
//   while True:
//   2. drainQueue()                         → nodes needing neighbor lookup
//   3. <Python: _batch_fetch_neighbors()>
//   4. pushResiduals(fetchedByEtypeId)
//   5. extractTopKWithResidualTopUp(maxPPRNodes, enableResidualTopUp)
class PPRForwardPush {
public:
    PPRForwardPush(const torch::Tensor& seedNodes,
                   int32_t seedNodeTypeId,
                   double alpha,
                   double requeueThresholdFactor,
                   std::vector<std::vector<int32_t>> nodeTypeToEdgeTypeIds,
                   std::vector<int32_t> edgeTypeToDstNtypeId,
                   std::vector<torch::Tensor> degreeTensors);

    // Drain queued nodes and return {etype_id: int64 node tensor} for neighbor lookup.
    // Returns nullopt when the queue is empty (convergence). Empty map means all nodes
    // were cache-hits; call pushResiduals({}) to continue.
    std::optional<std::unordered_map<int32_t, torch::Tensor>> drainQueue();

    // Push residuals given fetched neighbor data.
    void pushResiduals(const NeighborFetchMap& fetchedByEtypeId);

    // Return original graph edges from fetched adjacency whose endpoints are
    // both in the selected node set. The rows/cols are local indices into the
    // selected node tensors supplied by node type ID.
    OriginalEdgeExtractResult extractOriginalEdgesFromCache(
        const std::unordered_map<int32_t, torch::Tensor>& selectedNodeIdsByNodeTypeId, bool includeEdgeIds) const;

    // Return top-k PPR nodes plus residual-mass top-up nodes, sorted by score.
    //
    // Residual top-up does not issue new neighbor fetches.  It only reads the
    // residual table already built by Forward Push.  This gives callers a way
    // to fill short sequences with nodes that were discovered but did not cross
    // the requeue threshold, without lowering eps and running more push steps.
    // Scores are emitted on the
    // same mass scale as PPR scores: ppr_score(node) + residual(node), i.e. the
    // score the node would have if the remaining residual at that node were
    // absorbed locally.  Residual candidates only fill the requested top-up
    // budget; they do not displace selected finalized-PPR nodes. The returned
    // set is selected by this two-phase policy, then sorted by emitted score;
    // it is not a global top-k over ppr_score + residual when maxPPRNodes is tight.
    // maxPPRNodes is the final per-seed cap across finalized PPR and residual
    // top-up candidates.
    PPRExtractResult extractTopKWithResidualTopUp(int32_t maxPPRNodes, bool enableResidualTopUp);

    friend PPRExtractResult extractTypedTopKWithResidualTopUp(const std::vector<PPRForwardPush*>& states,
                                                              const std::vector<int32_t>& channelTargetCounts,
                                                              bool enableResidualTopUp);
    friend OriginalEdgeExtractResult extractOriginalEdgesFromPPRCaches(
        const std::vector<const PPRForwardPush*>& states,
        const std::unordered_map<int32_t, torch::Tensor>& selectedNodeIdsByNodeTypeId,
        bool includeEdgeIds);

private:
    // Total out-degree of a node across all edge types. Returns 0 for sink nodes.
    [[nodiscard]] int32_t getTotalDegree(int32_t nodeId, int32_t nodeTypeId) const;

    double _alpha;
    double _requeueThresholdFactor; // alpha * eps; per-node requeue threshold = factor * degree

    // NOTE: int32_t is used for batch size, node IDs, and type IDs throughout this class.
    // All of this code will break silently (overflow) if batch size or node IDs exceed ~2B
    // (INT32_MAX = 2,147,483,647).  This is not a realistic concern today, but if graph
    // scale ever approaches that threshold, these should be widened to int64_t.
    int32_t _batchSize;          // number of seed nodes in the current batch
    int32_t _numNodeTypes;       // total distinct node types (1 for homogeneous graphs)
    int32_t _numNodesInQueue{0}; // running count of queued nodes across all seeds and types

    // Graph structure — set at construction, read-only during the algorithm.
    // _nodeTypeToEdgeTypeIds[ntype_id] → list of edge type IDs that originate from that node type.
    // _edgeTypeToDstNtypeId[etype_id]  → destination node type ID for that edge type.
    // _degreeTensors[ntype_id]         → int32 tensor of total out-degrees, indexed by node ID.
    std::vector<std::vector<int32_t>> _nodeTypeToEdgeTypeIds;
    std::vector<int32_t> _edgeTypeToDstNtypeId;
    std::vector<int32_t> _edgeTypeToSrcNtypeId;
    std::vector<torch::Tensor> _degreeTensors;

    // Per-seed, per-node-type PPR state.  Indexed as _state[seedIdx][nodeTypeId].
    // 2D vector: both dimensions are dense sequential integers bounded at construction,
    // so array indexing is O(1) with no hashing (contrast with _neighborCache below).
    //
    // int32_t is used for node and type IDs throughout to match PyG/GLT's signed-integer
    // convention (torch.int32 / torch.int64).  Signed types also make nodeId >= 0 checks
    // meaningful — an unsigned type would make that guard tautological.
    //
    // Sized [_batchSize][_numNodeTypes] at construction and never resized,
    // so [seedIdx][nodeTypeId] indexing is always safe within the loop bounds.
    std::vector<std::vector<SeedNodeTypeState>> _state;

    // Neighbor lists keyed by packKey(nodeId, edgeTypeId).
    // Hash map: nodeId is a sparse graph ID from a large graph, so a dense array is
    // impractical (contrast with _state above).  Populated incrementally; avoids re-fetching.
    std::unordered_map<uint64_t, CachedNeighborList> _neighborCache;
};

// Helper function for draining several independent channel states for one
// typed-PPR iteration.
//
// This is the typed wrapper around PPRForwardPush::drainQueue(): it drains the
// independent channel states, records every channel that still needs
// pushResiduals(), and unions fetchable frontier nodes by edge type so Python
// can issue one shared distributed neighbor fetch for duplicate channel
// requests. Channel drains may run concurrently; the result merge happens in
// channel order to keep the returned channel-index lists deterministic.
//
// Inputs:
//   states: One PPRForwardPush per typed channel. Each state is mutated by its
//           drainQueue() call.
//   fetchIterationCounts: Number of distributed fetches already issued for each
//                         channel; aligned with states.
//   maxFetchIterations: -1 means unbounded; otherwise channels at this count
//                       still need pushResiduals() but contribute no new fetch
//                       frontier.
//
// Expected output: TypedPPRQueueDrainResult, whose drainedChannelIndices are
// the channels to push this iteration and whose unionedNodeIdsByEdgeTypeId is
// the shared fetch request.
TypedPPRQueueDrainResult drainTypedPPRChannelQueues(const std::vector<PPRForwardPush*>& states,
                                                    const std::vector<int32_t>& fetchIterationCounts,
                                                    int32_t maxFetchIterations);

// Helper function for extracting and merging completed typed-PPR channel states
// in one C++ step.
//
// For each seed/node-type, typed extraction builds one candidate view per
// channel. When residual top-up is enabled, residual candidates are included in
// that same view, so finalized PPR and residual top-up both obey the configured
// channel target counts. The merge uses emitted PPR scores for selection,
// deduplicates candidates seen through multiple channels by attributing each
// node to the channel where it has the highest score, fills each channel
// target, redistributes unused slots globally by score, and emits
// per-node-type tensors.
//
// Inputs:
//   states: Completed PPRForwardPush states, one per typed channel.
//   channelTargetCounts: Per-channel target output counts, aligned with states.
//                        Their sum is the maximum number of deduplicated nodes
//                        to return per seed.
//   enableResidualTopUp: Whether residual candidates may participate in target
//                        filling alongside finalized PPR candidates.
//
// Expected output: per-node-type tensors. Tuple values match
// extractTopKWithResidualTopUp:
//   ids: int64 node IDs, flattened across seeds.
//   weights: double feature matrix with columns
//            [best_score, per-channel scores..., presence bits...].
//   valid_counts: int64 count of selected nodes per seed.
PPRExtractResult extractTypedTopKWithResidualTopUp(const std::vector<PPRForwardPush*>& states,
                                                   const std::vector<int32_t>& channelTargetCounts,
                                                   bool enableResidualTopUp);

// Extract original graph edges from one or more completed PPR states. Multiple
// states are used for typed PPR, where each channel owns its own cache. Duplicate
// emitted edges shared by channels are emitted once.
OriginalEdgeExtractResult extractOriginalEdgesFromPPRCaches(
    const std::vector<const PPRForwardPush*>& states,
    const std::unordered_map<int32_t, torch::Tensor>& selectedNodeIdsByNodeTypeId,
    bool includeEdgeIds);

} // namespace gigl
