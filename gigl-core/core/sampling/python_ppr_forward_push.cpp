// Python bindings for PPRForwardPush.
//
// Pure C++ algorithm lives in ppr_forward_push.{h,cpp}; this file only handles
// type conversion between Python (pybind11) and C++ types, then delegates to
// the C++ implementation.

#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "ppr_forward_push.h"

namespace py = pybind11;

namespace gigl {

// pushResiduals receives Python-owned containers, so convert them while the GIL
// is held and release only around the C++ state update.
static void pushResidualsWrapper(PPRForwardPush& state, const py::dict& fetchedByEtypeId) {
    NeighborFetchMap neighborTensorsByEtypeId;
    // Dict iteration touches Python objects — GIL must be held here.
    for (auto item : fetchedByEtypeId) {
        auto edgeTypeId = item.first.cast<int32_t>();
        auto neighborTensors = item.second.cast<py::tuple>();
        auto neighborTensorCount = neighborTensors.size();
        if (neighborTensorCount != 3) {
            TORCH_CHECK(neighborTensorCount == 4,
                        "Expected neighbor fetch tuple of length 3 or 4, received ",
                        neighborTensorCount,
                        ".");
        }
        std::optional<torch::Tensor> edgeIds = std::nullopt;
        if (neighborTensorCount == 4 && !neighborTensors[3].is_none()) {
            edgeIds = neighborTensors[3].cast<torch::Tensor>();
        }
        neighborTensorsByEtypeId[edgeTypeId] = {neighborTensors[0].cast<torch::Tensor>(),
                                                neighborTensors[1].cast<torch::Tensor>(),
                                                neighborTensors[2].cast<torch::Tensor>(),
                                                edgeIds};
    }
    // C++ push only uses tensor accessor/data_ptr APIs — GIL-safe to release.
    // Releasing here lets the asyncio event loop process RPC completion callbacks
    // from other concurrent PPR coroutines while this push runs.
    // REQUIREMENT: no other thread may read or modify neighborTensorsByEtypeId or
    // the underlying tensor data while the GIL is released.  The caller (Python)
    // must not alias or mutate fetchedByEtypeId until push_residuals returns.
    {
        py::gil_scoped_release release;
        state.pushResiduals(neighborTensorsByEtypeId);
    }
}

static std::optional<std::unordered_map<int32_t, torch::Tensor>> drainQueueWrapper(PPRForwardPush& state) {
    std::optional<std::unordered_map<int32_t, torch::Tensor>> queueDrainResult;
    // drainQueue mutates only this PPRForwardPush instance and materializes CPU
    // tensors for frontier node IDs. pybind converts those tensor handles back
    // to Python tensors after return without copying the underlying storage.
    {
        py::gil_scoped_release release;
        queueDrainResult = state.drainQueue();
    }
    return queueDrainResult;
}

static PPRExtractResult extractTopKWithResidualTopUpWrapper(PPRForwardPush& state,
                                                            int32_t maxPPRNodes,
                                                            bool enableResidualTopUp) {
    PPRExtractResult result;
    // Extraction walks C++ state and builds torch tensors. Returning through
    // pybind creates Python container/wrapper objects, not tensor data copies.
    {
        py::gil_scoped_release release;
        result = state.extractTopKWithResidualTopUp(maxPPRNodes, enableResidualTopUp);
    }
    return result;
}

static py::tuple drainTypedPPRChannelQueuesWrapper(const py::sequence& states,
                                                   const std::vector<int32_t>& fetchIterationCounts,
                                                   int32_t maxFetchIterations) {
    std::vector<PPRForwardPush*> statePtrs;
    statePtrs.reserve(py::len(states));
    // Sequence iteration and casting touch Python objects, so keep the GIL
    // while copying raw C++ state pointers out of the Python container.
    for (py::handle stateObj : states) {
        statePtrs.push_back(&stateObj.cast<PPRForwardPush&>());
    }

    // C++ typed drain only reads/mutates PPRForwardPush states and builds C++
    // containers. Reacquire the GIL before constructing the Python tuple.
    // REQUIREMENT: no other thread may read or mutate these channel states
    // while the GIL is released. The typed sampler drains and pushes each
    // channel in a single sequenced loop iteration.
    TypedPPRQueueDrainResult queueDrainResult;
    {
        py::gil_scoped_release release;
        queueDrainResult = drainTypedPPRChannelQueues(statePtrs, fetchIterationCounts, maxFetchIterations);
    }
    // Pybind converts the temporary C++ containers into Python objects. Tensor
    // values are handles, so this does not copy tensor storage across the
    // Python/C++ boundary.
    return py::make_tuple(queueDrainResult.drainedChannelIndices,
                          queueDrainResult.fetchChannelIndices,
                          queueDrainResult.edgeTypeIdsByFetchChannel,
                          queueDrainResult.unionedNodeIdsByEdgeTypeId);
}

static PPRExtractResult extractTypedTopKWithResidualTopUpWrapper(const py::sequence& states,
                                                                 const std::vector<int32_t>& channelTargetCounts,
                                                                 bool enableResidualTopUp) {
    std::vector<PPRForwardPush*> statePtrs;
    statePtrs.reserve(py::len(states));
    // Sequence iteration and casting touch Python objects, so keep the GIL
    // while copying raw C++ state pointers out of the Python container.
    for (py::handle stateObj : states) {
        statePtrs.push_back(&stateObj.cast<PPRForwardPush&>());
    }

    // C++ extraction only reads the completed channel states and builds C++
    // tensors/containers. Reacquire the GIL before pybind converts the return.
    PPRExtractResult result;
    {
        py::gil_scoped_release release;
        result = extractTypedTopKWithResidualTopUp(statePtrs, channelTargetCounts, enableResidualTopUp);
    }
    return result;
}

static py::dict extractOriginalEdgesFromPPRCachesWrapper(const py::sequence& states,
                                                         const py::dict& selectedNodeIdsByNodeTypeId,
                                                         bool includeEdgeIds) {
    std::vector<const PPRForwardPush*> statePtrs;
    statePtrs.reserve(py::len(states));
    for (py::handle stateObj : states) {
        statePtrs.push_back(&stateObj.cast<PPRForwardPush&>());
    }

    std::unordered_map<int32_t, torch::Tensor> selectedNodeTensorsByNodeTypeId;
    for (auto item : selectedNodeIdsByNodeTypeId) {
        selectedNodeTensorsByNodeTypeId[item.first.cast<int32_t>()] = item.second.cast<torch::Tensor>();
    }

    OriginalEdgeExtractResult result;
    {
        py::gil_scoped_release release;
        result = extractOriginalEdgesFromPPRCaches(statePtrs, selectedNodeTensorsByNodeTypeId, includeEdgeIds);
    }

    // Building py::dict/py::tuple objects and pybind tensor wrappers touches the
    // Python C API, so the GIL must be held after the C++ extraction completes.
    py::dict pyResult;
    for (const auto& [edgeTypeId, tensors] : result) {
        py::object edgeIdsObject = py::none();
        const auto edgeIdsTensor = tensors.edgeIds.value_or(torch::Tensor());
        if (edgeIdsTensor.defined()) {
            edgeIdsObject = py::cast(edgeIdsTensor);
        }
        pyResult[py::int_(edgeTypeId)] = py::make_tuple(tensors.rows, tensors.cols, edgeIdsObject);
    }
    return pyResult;
}

} // namespace gigl

// TORCH_EXTENSION_NAME is set by PyTorch's build system to match the Python
// module name derived from this file's path (e.g. "ppr_forward_push").
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<gigl::PPRForwardPush>(m, "PPRForwardPush")
        .def(py::init<torch::Tensor,
                      int32_t,
                      double,
                      double,
                      std::vector<std::vector<int32_t>>,
                      std::vector<int32_t>,
                      std::vector<torch::Tensor>>(),
             // Constructor argument conversion happens before the C++ body; the
             // body only initializes PPR state and can run without the GIL.
             py::call_guard<py::gil_scoped_release>())
        .def("drain_queue", gigl::drainQueueWrapper)
        .def("push_residuals", gigl::pushResidualsWrapper)
        .def("extract_top_k_with_residual_top_up",
             &gigl::extractTopKWithResidualTopUpWrapper,
             py::arg("max_ppr_nodes"),
             py::arg("enable_residual_topup"));
    m.def("drain_typed_ppr_channel_queues",
          &gigl::drainTypedPPRChannelQueuesWrapper,
          py::arg("states"),
          py::arg("fetch_iteration_counts"),
          py::arg("max_fetch_iterations") = -1);
    m.def("extract_typed_top_k_with_residual_top_up",
          &gigl::extractTypedTopKWithResidualTopUpWrapper,
          py::arg("states"),
          py::arg("channel_target_counts"),
          py::arg("enable_residual_topup"));
    m.def("extract_original_edges_from_ppr_caches",
          &gigl::extractOriginalEdgesFromPPRCachesWrapper,
          py::arg("states"),
          py::arg("selected_node_ids_by_node_type_id"),
          py::arg("include_edge_ids"));
}
