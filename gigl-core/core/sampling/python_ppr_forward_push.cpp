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

static std::optional<std::unordered_map<int32_t, torch::Tensor>> drainQueueWrapper(PPRForwardPush& state) {
    // C++ drain only mutates the PPRForwardPush state and creates tensor outputs.
    // Reacquire the GIL before pybind converts the optional/map return value.
    // REQUIREMENT: no other thread may read or mutate this PPRForwardPush while
    // the GIL is released. The sampler owns each state within one coroutine.
    std::optional<std::unordered_map<int32_t, torch::Tensor>> drained;
    {
        py::gil_scoped_release release;
        drained = state.drainQueue();
    }
    return drained;
}

static void pushResidualsWrapper(PPRForwardPush& state, const py::dict& fetchedByEtypeId) {
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> neighborTensorsByEtypeId;
    // Dict iteration touches Python objects — GIL must be held here.
    for (auto item : fetchedByEtypeId) {
        auto edgeTypeId = item.first.cast<int32_t>();
        auto neighborTensors = item.second.cast<py::tuple>();
        neighborTensorsByEtypeId[edgeTypeId] = {neighborTensors[0].cast<torch::Tensor>(),
                                                neighborTensors[1].cast<torch::Tensor>(),
                                                neighborTensors[2].cast<torch::Tensor>()};
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

static std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
extractTopKWithResidualTopUpWrapper(PPRForwardPush& state, int32_t maxPPRNodes, bool enableResidualTopUp) {
    // C++ extraction only reads the completed state and builds C++
    // tensors/containers. Reacquire the GIL before pybind converts the return.
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
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
    DrainedTypedPPRQueues drained;
    {
        py::gil_scoped_release release;
        drained = drainTypedPPRChannelQueues(statePtrs, fetchIterationCounts, maxFetchIterations);
    }
    return py::make_tuple(std::move(drained.activeChannelIndices),
                          std::move(drained.fetchChannelIndices),
                          std::move(drained.edgeTypeIdsByFetchChannel),
                          std::move(drained.unionNodesByEdgeTypeId));
}

static std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>
extractTypedTopKWithResidualTopUpWrapper(const py::sequence& states,
                                         const std::vector<int32_t>& channelQuotas,
                                         int32_t maxPPRNodes,
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
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> result;
    {
        py::gil_scoped_release release;
        result = extractTypedTopKWithResidualTopUp(statePtrs, channelQuotas, maxPPRNodes, enableResidualTopUp);
    }
    return result;
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
                      std::vector<torch::Tensor>>())
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
          py::arg("channel_quotas"),
          py::arg("max_ppr_nodes"),
          py::arg("enable_residual_topup"));
}
