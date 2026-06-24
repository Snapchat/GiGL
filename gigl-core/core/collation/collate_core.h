#pragma once

// Generic collation kernel for distributed neighbor loaders.
//
// Reproduces the per-batch tensor wrangling that a distributed neighbor loader
// performs when turning a flat sampler message (per-type id/feature/edge tensors)
// into the component tensors of a graph batch: per-type dict assembly, the
// edge-direction row/col swap, empty-edge filling, and per-hop count padding.
//
// All input tensors are assumed to already reside on the target device; this
// kernel never issues device transfers. Pure C++ lives here and in collate_core.cpp;
// python_collate_core.cpp handles only Python<->C++ type conversion.

#include <optional>

#include <torch/torch.h>

namespace gigl {
namespace collation {

// Sentinel used by the scaffold import test; removed once real entry points land.
int ping();

// Right-pad a 1-D per-hop count tensor with zeros to `targetLen` on its own device.
// Mirrors torch.nn.functional.pad(t, (0, targetLen - t.size(0))).
// Precondition: counts.dim() == 1 and counts.size(0) <= targetLen.
torch::Tensor padCount(const torch::Tensor& counts, int64_t targetLen);

// Build a length-`targetLen` zero count vector with the given options (dtype/device).
// Mirrors torch.tensor([0]*targetLen, device=...).
torch::Tensor zeroCount(int64_t targetLen, const torch::TensorOptions& options);

// Component tensors for a homogeneous graph batch. Optionals are nullopt when the
// corresponding sampler tensor was absent; the Python shim maps these to None.
struct HomogeneousCollateResult {
    torch::Tensor node;       // local->global ids
    torch::Tensor edgeIndex;  // [2, E]
    std::optional<torch::Tensor> eid;
    std::optional<torch::Tensor> x;          // node features
    std::optional<torch::Tensor> edgeAttr;   // edge features
    std::optional<torch::Tensor> batch;
    std::optional<torch::Tensor> numSampledNodes;  // passed through verbatim
    std::optional<torch::Tensor> numSampledEdges;  // passed through verbatim
};

// Reproduces the GLT homogeneous collate body + to_data assembly (no padding).
// `rows`/`cols` are stacked verbatim into edgeIndex = stack([rows, cols]); the caller
// performs the edge_dir swap by choosing which sampler tensor to pass as each argument
// (mirrors SamplerOutput(ids, cols, rows, ...) at dist_loader.py:446).
// All tensors are assumed already on the target device; no transfers are issued.
HomogeneousCollateResult collateHomogeneous(
    const torch::Tensor& ids,
    const torch::Tensor& rows,
    const torch::Tensor& cols,
    const std::optional<torch::Tensor>& eids,
    const std::optional<torch::Tensor>& nfeats,
    const std::optional<torch::Tensor>& efeats,
    const std::optional<torch::Tensor>& batch,
    const std::optional<torch::Tensor>& numSampledNodes,
    const std::optional<torch::Tensor>& numSampledEdges);

}  // namespace collation
}  // namespace gigl
