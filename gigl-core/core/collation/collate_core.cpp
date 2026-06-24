#include "collate_core.h"

#include <torch/nn/functional/padding.h>

namespace gigl {
namespace collation {

int ping() {
    return 0;
}

torch::Tensor padCount(const torch::Tensor& counts, int64_t targetLen) {
    TORCH_CHECK(counts.dim() == 1, "per-hop count tensor must be 1-D");
    const int64_t current = counts.size(0);
    TORCH_CHECK(current <= targetLen, "per-hop count length exceeds target");
    if (current == targetLen) {
        return counts;
    }
    namespace F = torch::nn::functional;
    return F::pad(counts, F::PadFuncOptions({0, targetLen - current}));
}

torch::Tensor zeroCount(int64_t targetLen, const torch::TensorOptions& options) {
    return torch::zeros({targetLen}, options);
}

HomogeneousCollateResult collateHomogeneous(
    const torch::Tensor& ids,
    const torch::Tensor& rows,
    const torch::Tensor& cols,
    const std::optional<torch::Tensor>& eids,
    const std::optional<torch::Tensor>& nfeats,
    const std::optional<torch::Tensor>& efeats,
    const std::optional<torch::Tensor>& batch,
    const std::optional<torch::Tensor>& numSampledNodes,
    const std::optional<torch::Tensor>& numSampledEdges) {
    HomogeneousCollateResult result;
    result.node = ids;
    result.edgeIndex = torch::stack({rows, cols});
    result.eid = eids;
    result.x = nfeats;
    result.edgeAttr = efeats;
    result.batch = batch;
    result.numSampledNodes = numSampledNodes;
    result.numSampledEdges = numSampledEdges;
    return result;
}

}  // namespace collation
}  // namespace gigl
