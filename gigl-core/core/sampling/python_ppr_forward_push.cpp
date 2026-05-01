// Python bindings for PPRForwardPushState.
//
// Pure C++ algorithm lives in ppr_forward_push.{h,cpp}; this file only handles
// type conversion between Python (pybind11) and C++ types, then delegates to
// the C++ implementation.

#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <unordered_map>

#include "ppr_forward_push.h"

namespace py = pybind11;

namespace gigl {

// pushResiduals: a wrapper is needed solely to release the GIL during the C++ push.
// pybind11/stl.h handles all type conversions automatically; the other methods use
// direct member function pointers for the same reason.
static void pushResidualsWrapper(PPRForwardPushState& state, const py::dict& fetchedByEtypeId) {
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
    {
        py::gil_scoped_release release;
        state.pushResiduals(neighborTensorsByEtypeId);
    }
}

} // namespace gigl

// TORCH_EXTENSION_NAME is set by PyTorch's build system to match the Python
// module name derived from this file's path (e.g. "ppr_forward_push").
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<gigl::PPRForwardPushState>(m, "PPRForwardPushState")
        .def(py::init<torch::Tensor,
                      int32_t,
                      double,
                      double,
                      std::vector<std::vector<int32_t>>,
                      std::vector<int32_t>,
                      std::vector<torch::Tensor>>())
        .def("drain_queue", &gigl::PPRForwardPushState::drainQueue)
        .def("push_residuals", gigl::pushResidualsWrapper)
        .def("extract_top_k", &gigl::PPRForwardPushState::extractTopK)
        .def("get_nodes_drained_per_iteration", &gigl::PPRForwardPushState::getNodesDrainedPerIteration);
}
