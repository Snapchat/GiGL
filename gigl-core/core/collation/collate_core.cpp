#include "collate_core.h"

#include <torch/nn/functional/padding.h>

namespace {

// Look up a key in the message; return nullopt if absent (mirrors `key in msg`).
std::optional<torch::Tensor> tryGet(
    const std::unordered_map<std::string, torch::Tensor>& msg, const std::string& key) {
    auto it = msg.find(key);
    if (it == msg.end()) {
        return std::nullopt;
    }
    return it->second;
}

}  // namespace

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

HeterogeneousCollateResult collateHeterogeneous(
    const std::unordered_map<std::string, torch::Tensor>& msg,
    const std::vector<std::string>& nodeTypes,
    const std::vector<std::pair<std::string, EdgeTypeArray>>& edgeTypeStrToRev,
    const std::vector<EdgeTypeArray>& reversedEdgeTypes,
    const std::string& inputType,
    bool hasBatch,
    int64_t batchSize) {
    HeterogeneousCollateResult result;

    // --- nodes (dist_loader.py:356-365) ---
    for (const auto& ntype : nodeTypes) {
        if (auto ids = tryGet(msg, ntype + ".ids")) {
            result.node[ntype] = *ids;
        }
        if (auto nfeat = tryGet(msg, ntype + ".nfeats")) {
            result.x[ntype] = *nfeat;
        }
        if (auto nsn = tryGet(msg, ntype + ".num_sampled_nodes")) {
            result.numSampledNodes[ntype] = *nsn;  // padded below
        }
    }

    // --- edges + edge_dir swap (dist_loader.py:367-382) ---
    EdgeTypeMap<torch::Tensor> rowDict;
    EdgeTypeMap<torch::Tensor> colDict;
    for (const auto& [etypeStr, revEtype] : edgeTypeStrToRev) {
        auto rows = tryGet(msg, etypeStr + ".rows");
        auto cols = tryGet(msg, etypeStr + ".cols");
        if (rows && cols) {
            // The edge index is reversed: row<-cols, col<-rows.
            rowDict[revEtype] = *cols;
            colDict[revEtype] = *rows;
        }
        if (auto eids = tryGet(msg, etypeStr + ".eids")) {
            result.edge[revEtype] = *eids;
        }
        if (auto nse = tryGet(msg, etypeStr + ".num_sampled_edges")) {
            result.numSampledEdges[revEtype] = *nse;  // padded below
        }
        if (auto efeat = tryGet(msg, etypeStr + ".efeats")) {
            result.edgeAttr[revEtype] = *efeat;
        }
    }

    // --- batch (dist_loader.py:389-405); inputType is the anchor node type ---
    // GiGL loaders are NODE sampling; only the {inputType: batch} entry is produced.
    // batch_labels (nlabels) are not present for these loaders and are ignored here.
    // GLT writes the "{inputType}.batch" key ONLY when output.batch is not None
    // (dist_neighbor_sampler.py:781-783); when absent, GLT's NODE branch FALLS BACK to
    // node_dict[inputType][:batch_size] (dist_loader.py:397-399). We must reproduce that
    // fallback here, so the kernel takes batchSize and slices the anchor node ids.
    if (hasBatch) {
        if (auto b = tryGet(msg, inputType + ".batch")) {
            result.batch[inputType] = *b;
        } else {
            // Slice the first batchSize anchor node ids (matches node_dict[inputType][:batch_size]).
            auto nodeIt = result.node.find(inputType);
            TORCH_CHECK(nodeIt != result.node.end(),
                        "batch fallback requires anchor node ids for inputType");
            result.batch[inputType] = nodeIt->second.slice(/*dim=*/0, /*start=*/0, /*end=*/batchSize);
        }
    }

    // --- get_edge_index empty-fill (sampler/base.py:294-301) ---
    // GLT fills absent edge types with torch.empty((2,0)).to(self.device), where
    // self.device == to_device (dist_loader.py:417, sampler/base.py:299). Derive the
    // device from any present edge tensor; if NO edges were sampled at all, fall back to
    // any present NODE tensor's device (node tensors are on to_device after the dispatcher's
    // _move_msg_to_device). A bare CPU default would diverge from GLT on CUDA when a batch
    // has zero sampled edges.
    torch::TensorOptions edgeOpts = torch::TensorOptions().dtype(torch::kInt64);
    bool deviceFound = false;
    for (const auto& [et, t] : rowDict) {
        edgeOpts = torch::TensorOptions().dtype(torch::kInt64).device(t.device());
        deviceFound = true;
        break;
    }
    if (!deviceFound) {
        for (const auto& [nt, t] : result.node) {
            edgeOpts = torch::TensorOptions().dtype(torch::kInt64).device(t.device());
            deviceFound = true;
            break;
        }
    }
    for (const auto& revEtype : reversedEdgeTypes) {
        auto rIt = rowDict.find(revEtype);
        if (rIt != rowDict.end()) {
            result.edgeIndex[revEtype] = torch::stack({rIt->second, colDict.at(revEtype)});
        } else {
            result.edgeIndex[revEtype] = torch::empty({2, 0}, edgeOpts);
        }
    }

    // --- num_sampled_edges padding (transform.py:70-90) ---
    int64_t numHops = 0;
    for (const auto& [et, t] : result.numSampledEdges) {
        numHops = std::max<int64_t>(numHops, t.size(0));
    }
    for (const auto& revEtype : reversedEdgeTypes) {
        auto edgeIndexDevice = result.edgeIndex.at(revEtype).device();
        auto countOpts = torch::TensorOptions().dtype(torch::kInt64).device(edgeIndexDevice);
        auto it = result.numSampledEdges.find(revEtype);
        if (it == result.numSampledEdges.end()) {
            result.numSampledEdges[revEtype] = zeroCount(numHops, countOpts);
        } else {
            result.numSampledEdges[revEtype] = padCount(it->second, numHops);
        }
    }

    // --- num_sampled_nodes padding (transform.py:97-104) ---
    // PyG iterates node types present in the sampler output's node dict.
    for (const auto& ntype : nodeTypes) {
        auto nodeIt = result.node.find(ntype);
        if (nodeIt == result.node.end()) {
            continue;  // node type absent from this batch's node dict
        }
        auto nodeDevice = nodeIt->second.device();
        auto countOpts = torch::TensorOptions().dtype(torch::kInt64).device(nodeDevice);
        auto it = result.numSampledNodes.find(ntype);
        if (it == result.numSampledNodes.end()) {
            result.numSampledNodes[ntype] = zeroCount(numHops + 1, countOpts);
        } else {
            result.numSampledNodes[ntype] = padCount(it->second, numHops + 1);
        }
    }

    return result;
}

}  // namespace collation
}  // namespace gigl
