// One-hop neighbor lookup that serves remote sampling requests without Python.
//
// GLT answers another rank's neighbor request with a Python RPC callee, so torch RPC takes the
// interpreter lock to unpickle the request, run the lookup and pickle the reply. A TorchScript
// function that calls lookupNeighbors runs on the RPC thread with no interpreter lock.
//
// Each sampler registers the CSR topology of its partition once, keyed by the sampler's RPC
// callee id (identical on every rank) and the edge type. Lookups take global ids and return
// what GLT's CPURandomSampler::Sample / SampleWithEdge return for the same rows.

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

namespace gigl {

// The CSR of one edge type of one partition. The tensors are references to the dataset's
// storage, not copies.
struct RegisteredCsr {
    torch::Tensor indptr;                 // int64 [rowCount + 1]
    torch::Tensor indices;                // int32 or int64 [edgeCount]
    std::optional<torch::Tensor> edgeIds; // int64 [edgeCount]; none when the graph has no edge ids
    int64_t compressedIdOffset;           // global id of row 0
    bool withEdge;                        // whether replies carry edge ids
};

// Registers (or replaces) the CSR of (samplerId, edgeType). Validates dtypes, devices and shapes.
void registerCsr(int64_t samplerId,
                 const std::string& edgeType,
                 torch::Tensor indptr,
                 torch::Tensor indices,
                 std::optional<torch::Tensor> edgeIds,
                 int64_t compressedIdOffset,
                 bool withEdge);

// Removes every edge type registered under samplerId. In-flight lookups keep their tensors.
void unregisterCsr(int64_t samplerId);

// Seeds the generator that sub-samples long rows, as GLT's RandomSeedManager::setSeed does for
// its sampler: each RPC thread seeds its own generator with this value the first time it
// sub-samples. Threads that sub-sampled before the call keep their generator.
void setRandomSeed(uint32_t seed);

// Samples up to reqNum neighbors of each global id in `ids` (int64, CPU, 1-D).
//
// Returns (neighbors, counts, edgeIds): int64 flat neighbor ids, int64 per-id counts and int64
// edge ids (std::nullopt, which TorchScript returns as None, when the registration has withEdge
// false; an empty tensor when it has withEdge true and the sample is empty). An id outside the partition's rows
// has no neighbors. A row with at most reqNum neighbors is copied in stored order, exactly as
// GLT does. A row with more is sampled uniformly with replacement, as GLT does, from this
// thread's own generator, so those draws differ from GLT's.
std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> lookupNeighbors(int64_t samplerId,
                                                                                       const std::string& edgeType,
                                                                                       const torch::Tensor& ids,
                                                                                       int64_t reqNum);

} // namespace gigl
