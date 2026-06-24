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

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

using EdgeTypeArray = std::array<std::string, 3>;

struct EdgeTypeArrayHash {
    std::size_t operator()(const EdgeTypeArray& e) const noexcept {
        std::size_t h = 1469598103934665603ULL;  // FNV-1a basis
        for (const auto& s : e) {
            for (char c : s) {
                h ^= static_cast<std::size_t>(static_cast<unsigned char>(c));
                h *= 1099511628211ULL;
            }
            h ^= 0x9e3779b97f4a7c15ULL;  // separator between tuple fields
        }
        return h;
    }
};

template <typename V>
using EdgeTypeMap = std::unordered_map<EdgeTypeArray, V, EdgeTypeArrayHash>;

// Component tensors for a heterogeneous graph batch. Keys mirror GLT/PyG:
// node/x keyed by node-type string; edges keyed by the reversed EdgeTypeArray.
struct HeterogeneousCollateResult {
    std::unordered_map<std::string, torch::Tensor> node;
    EdgeTypeMap<torch::Tensor> edgeIndex;       // [2,E]; absent types filled [2,0]
    EdgeTypeMap<torch::Tensor> edge;            // eids; only present types
    std::unordered_map<std::string, torch::Tensor> x;   // nfeats; only present types
    EdgeTypeMap<torch::Tensor> edgeAttr;        // efeats; only present types
    std::unordered_map<std::string, torch::Tensor> batch;  // {inputType: tensor} or empty
    std::unordered_map<std::string, torch::Tensor> numSampledNodes;  // padded
    EdgeTypeMap<torch::Tensor> numSampledEdges;  // padded
};

// Reproduces the GLT heterogeneous collate body (dist_loader.py:351-420) and the
// tensor-level parts of to_hetero_data (transform.py:60-115): per-type dict build,
// edge_dir row/col swap, get_edge_index empty-fill, and num_sampled_* padding.
// Metadata is assumed already stripped (GiGL strips #META keys upstream), so no
// metadata handling is performed. All tensors are assumed on the target device.
HeterogeneousCollateResult collateHeterogeneous(
    const std::unordered_map<std::string, torch::Tensor>& msg,
    const std::vector<std::string>& nodeTypes,
    const std::vector<std::pair<std::string, EdgeTypeArray>>& edgeTypeStrToRev,
    const std::vector<EdgeTypeArray>& reversedEdgeTypes,
    const std::string& inputType,
    bool hasBatch,
    int64_t batchSize);  // anchor-id slice length used when the ".batch" key is absent

}  // namespace collation
}  // namespace gigl
