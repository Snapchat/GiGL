// Serves one-hop neighbor lookups for remote sampling requests. This is a copy of the logic GLT and GiGL run
// in Python and C++ for the same request, not a call into it:
//
// - GLT's CPURandomSampler::Sample / SampleWithEdge (a req_num < 0 means int32 max):
//   https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/csrc/cpu/random_sampler.cc#L24-L73
// - FillNbrsNum (count = min(req_num, degree); 0 for rows at or past the row count):
//   https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/csrc/cpu/random_sampler.cc#L75-L92
// - UniformSample (copy when degree <= req_num, else uniform with replacement):
//   https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/csrc/cpu/random_sampler.cc#L137-L176
// - RpcSamplingCallee.call, the Python callee this path replaces:
//   https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/python/distributed/dist_neighbor_sampler.py#L58-L77
// - NeighborSampler.sample_one_hop (shape of an empty output):
//   https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/python/sampler/neighbor_sampler.py#L146-L167
// - GiGL's patch gigl/scripts/patches/0001-glt-csr-col-count-and-int32-indices.patch rewrites those samplers as
//   templates over int32/int64 column ids, makes SampleWithEdge reject a non-empty sample without edge ids, and
//   reads the seed once per thread.
// - GiGL's global-to-local id translation: to_local_compressed_ids in gigl/utils/csr.py and
//   _GlobalIdNeighborSampler.sample_one_hop in gigl/distributed/base_sampler.py.
//
// The GLT links are pinned to the commit in GLT_COMMIT (gigl/scripts/glt_wheel_cache.py); update them when the pin
// moves.
//
// Why a copy and not a call: GLT's installed wheel is one pybind module (py_graphlearn_torch*.so) with no headers
// or linkable C++ library, so gigl-core cannot call CPURandomSampler. And the TorchScript RPC needs a registered
// torch operator, which GLT's sampler is not.

#include "neighbor_lookup.h"

#include <c10/util/ArrayRef.h>

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

// std::map rather than a flat hash map such as the ska::flat_hash_map in ppr_forward_push.h: the map holds one
// entry per (sampler, edge type) in the process, tens of entries, and is read once per RPC request, not once per
// node. A request takes about 12-26 us at 256 ids and 0.4-12 ms at 4,096-65,536 ids, so the lookup is noise.
// std::map also needs no hash for the (int64, string) key.
//
// The mutex is there because registerCsr can run while lookups for other samplers run on RPC threads: in
// graph-store mode every new channel builds a sampler, and so registers, inside a long-lived worker that is
// already serving, and std::map is not safe for a read during a write. It is a shared_mutex taken in shared mode
// by lookups, so readers do not block each other; it costs one atomic update per request and is held only to copy
// the shared_ptr (see findCsr).
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
//
// The seed follows GLT's RandomSeedManager:
//   https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/include/common.h#L36-L63
// and the engine in UniformSample:
//   https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/csrc/cpu/random_sampler.cc#L143-L146
// GiGL's patch seeds that engine once per thread, as this thread_local does.
// This file draws from uniform_int_distribution<int64_t> where GLT uses uniform_int_distribution<int>, so the
// draws differ from GLT's even from the same seed.
std::mt19937& threadEngine() {
    thread_local std::mt19937 engine(threadSeed());
    return engine;
}

// Fills one row's reply at `offset`. `columns` and `edgeIds` are the row's slice of the CSR; `edgeIds` is empty when
// the reply carries no edge ids. The row's reply is `min(reqNum, degree)` long, and `outEdgeIds` is only written when
// `edgeIds` is not empty.
template<typename ColT>
void copyOrSampleRow(c10::ArrayRef<ColT> columns,
                     c10::ArrayRef<int64_t> edgeIds,
                     int64_t reqNum,
                     at::TensorAccessor<int64_t, 1> outNeighbors,
                     at::TensorAccessor<int64_t, 1> outEdgeIds,
                     int64_t offset) {
    const int64_t degree = columns.size();
    const bool withEdgeIds = !edgeIds.empty();
    const int64_t rowLength = std::min(reqNum, degree);
    // The accessor's operator[] does not check bounds (and multiplies by a runtime stride), so the range is checked
    // once per row and the row is then written through a plain pointer.
    TORCH_CHECK(offset + rowLength <= outNeighbors.size(0), "lookupNeighbors: reply is shorter than the row");
    int64_t* rowNeighbors = &outNeighbors[offset];
    int64_t* rowEdgeIds = nullptr;
    if (withEdgeIds) {
        TORCH_CHECK(offset + rowLength <= outEdgeIds.size(0), "lookupNeighbors: edge id reply is shorter than the row");
        rowEdgeIds = &outEdgeIds[offset];
    }
    if (reqNum < degree) {
        std::uniform_int_distribution<int64_t> distribution(0, degree - 1);
        auto& engine = threadEngine();
        for (int64_t i = 0; i < reqNum; ++i) {
            const int64_t position = distribution(engine);
            rowNeighbors[i] = static_cast<int64_t>(columns[position]);
            if (withEdgeIds) {
                rowEdgeIds[i] = edgeIds[position];
            }
        }
        return;
    }
    std::copy(columns.begin(), columns.end(), rowNeighbors);
    if (withEdgeIds) {
        std::copy(edgeIds.begin(), edgeIds.end(), rowEdgeIds);
    }
}

// `outEdgeIds` is only written when `writeEdgeIds`; otherwise it may be over an empty tensor.
template<typename ColT>
void gatherRows(const RegisteredCsr& csr,
                const std::vector<int64_t>& localRows,
                c10::ArrayRef<int64_t> counts,
                int64_t reqNum,
                at::TensorAccessor<int64_t, 1> outNeighbors,
                bool writeEdgeIds,
                at::TensorAccessor<int64_t, 1> outEdgeIds) {
    const c10::ArrayRef<int64_t> rowPtr(csr.indptr.data_ptr<int64_t>(), csr.indptr.numel());
    const c10::ArrayRef<ColT> colIdx(csr.indices.data_ptr<ColT>(), csr.indices.numel());
    c10::ArrayRef<int64_t> edgeIds;
    if (writeEdgeIds) {
        edgeIds = c10::ArrayRef<int64_t>(csr.edgeIds.value().data_ptr<int64_t>(), csr.edgeIds.value().numel());
    }
    int64_t offset = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] == 0) {
            continue;
        }
        const int64_t row = localRows[i];
        const int64_t start = rowPtr[row];
        const int64_t degree = rowPtr[row + 1] - start;
        c10::ArrayRef<int64_t> rowEdgeIds;
        if (writeEdgeIds) {
            rowEdgeIds = edgeIds.slice(start, degree);
        }
        copyOrSampleRow<ColT>(colIdx.slice(start, degree), rowEdgeIds, reqNum, outNeighbors, outEdgeIds, offset);
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

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> lookupNeighbors(int64_t samplerId,
                                                                                       const std::string& edgeType,
                                                                                       const torch::Tensor& ids,
                                                                                       int64_t reqNum) {
    // The registry lock is held only to copy the shared_ptr, so a replaced registration cannot
    // free tensors under an in-flight lookup.
    const auto csr = findCsr(samplerId, edgeType);
    TORCH_CHECK(
        csr != nullptr, "lookupNeighbors: no CSR registered for sampler ", samplerId, " edge type '", edgeType, "'");
    return lookupNeighbors(*csr, ids, reqNum);
}

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>> lookupNeighbors(const RegisteredCsr& csr,
                                                                                       const torch::Tensor& ids,
                                                                                       int64_t reqNum) {
    TORCH_CHECK(ids.device().is_cpu(), "lookupNeighbors: ids must be on the CPU");
    TORCH_CHECK(ids.dim() == 1, "lookupNeighbors: ids must be 1-D");
    TORCH_CHECK(ids.scalar_type() == torch::kInt64, "lookupNeighbors: ids must be int64, got ", ids.scalar_type());
    // GLT's sampler takes an int32 request size; a negative one means every neighbor.
    if (reqNum < 0 || reqNum > std::numeric_limits<int32_t>::max()) {
        reqNum = std::numeric_limits<int32_t>::max();
    }
    const torch::Tensor contiguousIds = ids.contiguous();
    const c10::ArrayRef<int64_t> idView(contiguousIds.data_ptr<int64_t>(), contiguousIds.numel());
    const int64_t batchSize = contiguousIds.size(0);
    const int64_t rowCount = csr.indptr.size(0) - 1;
    const c10::ArrayRef<int64_t> rowPtr(csr.indptr.data_ptr<int64_t>(), csr.indptr.numel());
    const auto longOptions = torch::TensorOptions().dtype(torch::kInt64);

    torch::Tensor counts = torch::empty({batchSize}, longOptions);
    auto countOut = counts.accessor<int64_t, 1>();
    std::vector<int64_t> localRows(batchSize);
    int64_t total = 0;
    for (int64_t i = 0; i < batchSize; ++i) {
        // gigl.utils.csr.to_local_compressed_ids: ids outside [offset, offset + rowCount) have no row.
        const int64_t row = idView[i] - csr.compressedIdOffset;
        int64_t count = 0;
        if (row >= 0 && row < rowCount) {
            count = std::min(reqNum, rowPtr[row + 1] - rowPtr[row]);
        }
        localRows[i] = row;
        countOut[i] = count;
        total += count;
    }

    torch::Tensor neighbors = torch::empty({total}, longOptions);
    // None when the registration does not want edge ids; otherwise empty until a non-empty sample fills it.
    std::optional<torch::Tensor> edgeIdsOut;
    if (csr.withEdge) {
        edgeIdsOut = torch::empty({0}, longOptions);
    }
    bool writeEdgeIds = false;
    if (csr.withEdge && total > 0) {
        // GLT's SampleWithEdge raises the same way for a non-empty sample without edge ids.
        TORCH_CHECK(csr.edgeIds.has_value(), "lookupNeighbors: edge ids were requested but the graph has none");
        edgeIdsOut = torch::empty({total}, longOptions);
        writeEdgeIds = true;
    }
    if (total > 0) {
        // Single-threaded on purpose. Parallelism already comes from serving many requests at once on the RPC
        // thread pool (up to 16 threads per sampling worker: shared_dist_sampling_producer.py sets
        // num_rpc_threads = min(num_partitions, 16)). A request is one partition's share of one hop's frontier,
        // usually hundreds to a few thousand ids, so a per-request thread fan-out would cost more in wake-up than
        // it saves and would oversubscribe the cores. GLT's own at::parallel_for here also runs serially, because
        // GLT is not built with OpenMP (see the patch).
        //
        // A version written with torch ops (index, where, clamp, cumsum, repeat_interleave, gather) was measured
        // on a 1M-row, ~50M-edge CSR. On one thread it is 10-28x slower at 256 ids, and about 2x slower
        // (sub-sampled) to 7x slower (copy) at 4,096-65,536 ids. With 16 intra-op threads it wins only at 65,536
        // ids with sub-sampling (0.7x the time), by using cores the other RPC threads need. This loop does one
        // pass with no temporaries, and its row reads are data-dependent gathers, which SIMD does not help.
        auto neighborsOut = neighbors.accessor<int64_t, 1>();
        // Without edge ids to write, this is an empty tensor that gatherRows never touches.
        const torch::Tensor edgeIdsTensor = edgeIdsOut.value_or(torch::empty({0}, longOptions));
        auto edgeIdsAccessor = edgeIdsTensor.accessor<int64_t, 1>();
        const c10::ArrayRef<int64_t> countView(counts.data_ptr<int64_t>(), counts.numel());
        if (csr.indices.scalar_type() == torch::kInt32) {
            gatherRows<int32_t>(csr, localRows, countView, reqNum, neighborsOut, writeEdgeIds, edgeIdsAccessor);
        } else {
            gatherRows<int64_t>(csr, localRows, countView, reqNum, neighborsOut, writeEdgeIds, edgeIdsAccessor);
        }
    }
    return std::make_tuple(neighbors, counts, edgeIdsOut);
}

} // namespace gigl
