#include "rpc_neighbor_lookup.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace gigl {

namespace {

using CsrKey = std::pair<int64_t, std::string>;

std::shared_mutex& registryMutex() {
    static std::shared_mutex mutex;
    return mutex;
}

std::map<CsrKey, std::shared_ptr<const RegisteredCsr>>& registry() {
    static std::map<CsrKey, std::shared_ptr<const RegisteredCsr>> csrByKey;
    return csrByKey;
}

std::shared_ptr<const RegisteredCsr> findCsr(int64_t samplerId, const std::string& edgeType) {
    std::shared_lock<std::shared_mutex> lock(registryMutex());
    auto it = registry().find({samplerId, edgeType});
    if (it == registry().end()) {
        return nullptr;
    }
    return it->second;
}

// The seed set by setRandomSeed, or -1 when unset (then each thread seeds from std::random_device,
// as GLT's RandomSeedManager::getSeed does).
std::atomic<int64_t>& randomSeed() {
    static std::atomic<int64_t> seed{-1};
    return seed;
}

uint32_t threadSeed() {
    const int64_t seed = randomSeed().load();
    if (seed >= 0) {
        return static_cast<uint32_t>(seed);
    }
    return std::random_device{}();
}

// Uniform with replacement, as GLT's CPURandomSampler::UniformSample, but from this thread's own
// generator (GLT's generator is not reachable from here).
std::mt19937& threadEngine() {
    thread_local std::mt19937 engine(threadSeed());
    return engine;
}

template<typename ColT>
void copyOrSampleRow(const ColT* colBegin,
                     const int64_t* edgeIdBegin,
                     int64_t degree,
                     int64_t reqNum,
                     int64_t* outNeighbors,
                     int64_t* outEdgeIds) {
    if (reqNum < degree) {
        std::uniform_int_distribution<int64_t> distribution(0, degree - 1);
        auto& engine = threadEngine();
        for (int64_t i = 0; i < reqNum; ++i) {
            const int64_t position = distribution(engine);
            outNeighbors[i] = static_cast<int64_t>(colBegin[position]);
            if (outEdgeIds != nullptr) {
                outEdgeIds[i] = edgeIdBegin[position];
            }
        }
        return;
    }
    for (int64_t i = 0; i < degree; ++i) {
        outNeighbors[i] = static_cast<int64_t>(colBegin[i]);
    }
    if (outEdgeIds != nullptr) {
        std::copy(edgeIdBegin, edgeIdBegin + degree, outEdgeIds);
    }
}

template<typename ColT>
void gatherRows(const RegisteredCsr& csr,
                const int64_t* localRows,
                const int64_t* counts,
                int64_t batchSize,
                int64_t reqNum,
                int64_t* outNeighbors,
                int64_t* outEdgeIds) {
    const int64_t* rowPtr = csr.indptr.data_ptr<int64_t>();
    const ColT* colIdx = csr.indices.data_ptr<ColT>();
    const int64_t* edgeIds = nullptr;
    if (outEdgeIds != nullptr && csr.edgeIds.has_value()) {
        edgeIds = csr.edgeIds.value().data_ptr<int64_t>();
    }
    int64_t offset = 0;
    for (int64_t i = 0; i < batchSize; ++i) {
        if (counts[i] == 0) {
            continue;
        }
        const int64_t row = localRows[i];
        const int64_t start = rowPtr[row];
        const int64_t degree = rowPtr[row + 1] - start;
        int64_t* edgeOut = nullptr;
        const int64_t* edgeBegin = nullptr;
        if (outEdgeIds != nullptr) {
            edgeOut = outEdgeIds + offset;
            edgeBegin = edgeIds + start;
        }
        copyOrSampleRow<ColT>(colIdx + start, edgeBegin, degree, reqNum, outNeighbors + offset, edgeOut);
        offset += counts[i];
    }
}

} // namespace

void registerCsr(int64_t samplerId,
                 const std::string& edgeType,
                 torch::Tensor indptr,
                 torch::Tensor indices,
                 std::optional<torch::Tensor> edgeIds,
                 int64_t compressedIdOffset,
                 bool withEdge) {
    TORCH_CHECK(indptr.device().is_cpu(), "registerCsr: indptr must be on the CPU");
    TORCH_CHECK(indices.device().is_cpu(), "registerCsr: indices must be on the CPU");
    TORCH_CHECK(indptr.dim() == 1, "registerCsr: indptr must be 1-D");
    TORCH_CHECK(indptr.size(0) >= 1, "registerCsr: indptr must have at least one entry");
    TORCH_CHECK(indptr.scalar_type() == torch::kInt64, "registerCsr: indptr must be int64, got ", indptr.scalar_type());
    TORCH_CHECK(indptr.is_contiguous(), "registerCsr: indptr must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "registerCsr: indices must be contiguous");
    TORCH_CHECK(indices.dim() == 1, "registerCsr: indices must be 1-D");
    const bool indicesAreIntegers = indices.scalar_type() == torch::kInt64 || indices.scalar_type() == torch::kInt32;
    TORCH_CHECK(indicesAreIntegers, "registerCsr: indices must be int32 or int64, got ", indices.scalar_type());
    TORCH_CHECK(indptr[0].item<int64_t>() == 0, "registerCsr: indptr must start at 0");
    TORCH_CHECK(indptr[-1].item<int64_t>() <= indices.size(0), "registerCsr: indptr must end within indices");
    if (edgeIds.has_value() && edgeIds->numel() == 0) {
        // GLT treats an empty edge-id tensor as no edge ids.
        edgeIds.reset();
    }
    if (edgeIds.has_value()) {
        TORCH_CHECK(edgeIds->device().is_cpu(), "registerCsr: edge ids must be on the CPU");
        TORCH_CHECK(edgeIds->is_contiguous(), "registerCsr: edge ids must be contiguous");
        TORCH_CHECK(edgeIds->dim() == 1, "registerCsr: edge ids must be 1-D");
        TORCH_CHECK(edgeIds->scalar_type() == torch::kInt64, "registerCsr: edge ids must be int64");
        TORCH_CHECK(edgeIds->size(0) == indices.size(0), "registerCsr: edge ids and indices differ in length");
    }
    auto csr = std::make_shared<const RegisteredCsr>(
        RegisteredCsr{std::move(indptr), std::move(indices), std::move(edgeIds), compressedIdOffset, withEdge});
    std::unique_lock<std::shared_mutex> lock(registryMutex());
    registry()[{samplerId, edgeType}] = std::move(csr);
}

void unregisterCsr(int64_t samplerId) {
    std::unique_lock<std::shared_mutex> lock(registryMutex());
    auto& csrByKey = registry();
    for (auto it = csrByKey.begin(); it != csrByKey.end();) {
        if (it->first.first == samplerId) {
            it = csrByKey.erase(it);
        } else {
            ++it;
        }
    }
}

void setRandomSeed(uint32_t seed) {
    randomSeed().store(static_cast<int64_t>(seed));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lookupNeighbors(int64_t samplerId,
                                                                        const std::string& edgeType,
                                                                        const torch::Tensor& ids,
                                                                        int64_t reqNum) {
    // The registry lock is held only to copy the shared_ptr, so a replaced registration cannot
    // free tensors under an in-flight lookup.
    const auto csr = findCsr(samplerId, edgeType);
    TORCH_CHECK(
        csr != nullptr, "lookupNeighbors: no CSR registered for sampler ", samplerId, " edge type '", edgeType, "'");
    TORCH_CHECK(ids.device().is_cpu(), "lookupNeighbors: ids must be on the CPU");
    TORCH_CHECK(ids.dim() == 1, "lookupNeighbors: ids must be 1-D");
    TORCH_CHECK(ids.scalar_type() == torch::kInt64, "lookupNeighbors: ids must be int64, got ", ids.scalar_type());
    // GLT's sampler takes an int32 request size; a negative one means every neighbor.
    if (reqNum < 0 || reqNum > std::numeric_limits<int32_t>::max()) {
        reqNum = std::numeric_limits<int32_t>::max();
    }
    const torch::Tensor contiguousIds = ids.contiguous();
    const auto* idPtr = contiguousIds.data_ptr<int64_t>();
    const int64_t batchSize = contiguousIds.size(0);
    const int64_t rowCount = csr->indptr.size(0) - 1;
    const auto* rowPtr = csr->indptr.data_ptr<int64_t>();
    const auto longOptions = torch::TensorOptions().dtype(torch::kInt64);

    torch::Tensor counts = torch::empty({batchSize}, longOptions);
    auto* countPtr = counts.data_ptr<int64_t>();
    std::vector<int64_t> localRows(batchSize);
    int64_t total = 0;
    for (int64_t i = 0; i < batchSize; ++i) {
        // gigl.utils.csr.to_local_compressed_ids: ids outside [offset, offset + rowCount) have no row.
        const int64_t row = idPtr[i] - csr->compressedIdOffset;
        int64_t count = 0;
        if (row >= 0 && row < rowCount) {
            count = std::min(reqNum, rowPtr[row + 1] - rowPtr[row]);
        }
        localRows[i] = row;
        countPtr[i] = count;
        total += count;
    }

    torch::Tensor neighbors = torch::empty({total}, longOptions);
    torch::Tensor edgeIdsOut = torch::empty({0}, longOptions);
    int64_t* edgeOutPtr = nullptr;
    if (csr->withEdge && total > 0) {
        // GLT's SampleWithEdge raises the same way for a non-empty sample without edge ids.
        TORCH_CHECK(csr->edgeIds.has_value(), "lookupNeighbors: edge ids were requested but the graph has none");
        edgeIdsOut = torch::empty({total}, longOptions);
        edgeOutPtr = edgeIdsOut.data_ptr<int64_t>();
    }
    if (total > 0) {
        // Single-threaded on purpose: many RPC threads run lookups at once, and fanning each one
        // out into the intra-op pool would oversubscribe the cores.
        if (csr->indices.scalar_type() == torch::kInt32) {
            gatherRows<int32_t>(
                *csr, localRows.data(), countPtr, batchSize, reqNum, neighbors.data_ptr<int64_t>(), edgeOutPtr);
        } else {
            gatherRows<int64_t>(
                *csr, localRows.data(), countPtr, batchSize, reqNum, neighbors.data_ptr<int64_t>(), edgeOutPtr);
        }
    }
    return std::make_tuple(neighbors, counts, edgeIdsOut);
}

} // namespace gigl
