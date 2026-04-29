// Python bindings for PPRForwardPushState.
//
// Follows PyTorch's csrc convention: pure C++ algorithm lives in
// ppr_forward_push.{h,cpp}; this file only handles type conversion between
// Python (pybind11) and C++ types, then delegates to the C++ implementation.

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "ppr_forward_push.h"

namespace py = pybind11;

// drainQueue: C++ returns std::optional<map<etype_id, Tensor>>.
// Exposed to Python as: None (convergence) or dict[int, Tensor].
static py::object drainQueueWrapper(PPRForwardPushState& self) {
    auto result = self.drainQueue();
    if (!result) {
        return py::none();
    }
    py::dict d;
    for (auto& [eid, tensor] : *result) {
        d[py::int_(eid)] = tensor;
    }
    return d;
}

// pushResiduals: Python passes dict[int, tuple[Tensor, Tensor, Tensor]].
// Convert to C++ map before delegating.
static void pushResidualsWrapper(PPRForwardPushState& self, const py::dict& fetchedByEtypeId) {
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> cppMap;
    // Dict iteration touches Python objects — GIL must be held here.
    for (auto item : fetchedByEtypeId) {
        auto eid = item.first.cast<int32_t>();
        auto tup = item.second.cast<py::tuple>();
        cppMap[eid] = {tup[0].cast<torch::Tensor>(), tup[1].cast<torch::Tensor>(), tup[2].cast<torch::Tensor>()};
    }
    // C++ push only uses tensor accessor/data_ptr APIs — GIL-safe to release.
    // Releasing here lets the asyncio event loop process RPC completion callbacks
    // from other concurrent PPR coroutines while this push runs.
    {
        py::gil_scoped_release release;
        self.pushResiduals(cppMap);
    }
}

// extractTopK: C++ returns map<ntype_id, tuple<Tensor, Tensor, Tensor>>.
// Exposed to Python as dict[int, tuple[Tensor, Tensor, Tensor]].
static py::dict extractTopKWrapper(PPRForwardPushState& self, int32_t maxPprNodes) {
    auto result = self.extractTopK(maxPprNodes);
    py::dict d;
    for (auto& [nt, tup] : result) {
        d[py::int_(nt)] = py::make_tuple(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup));
    }
    return d;
}

// TORCH_EXTENSION_NAME is set by PyTorch's build system to match the Python
// module name derived from this file's path (e.g. "ppr_forward_push").
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<PPRForwardPushState>(m, "PPRForwardPushState")
        .def(py::init<torch::Tensor,
                      int32_t,
                      double,
                      double,
                      std::vector<std::vector<int32_t>>,
                      std::vector<int32_t>,
                      std::vector<torch::Tensor>>())
        .def("drain_queue", drainQueueWrapper)
        .def("push_residuals", pushResidualsWrapper)
        .def("extract_top_k", extractTopKWrapper)
        .def("get_nodes_drained_per_iteration", &PPRForwardPushState::getNodesDrainedPerIteration);
}
