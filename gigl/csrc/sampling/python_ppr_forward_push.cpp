// Python bindings for PPRForwardPushState.
//
// Follows PyTorch's csrc convention: pure C++ algorithm lives in
// ppr_forward_push.{h,cpp}; this file only handles type conversion between
// Python (pybind11) and C++ types, then delegates to the C++ implementation.

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "ppr_forward_push.h"

namespace py = pybind11;

// drain_queue: C++ returns std::optional<map<etype_id, Tensor>>.
// Exposed to Python as: None (convergence) or dict[int, Tensor].
static py::object drain_queue_wrapper(PPRForwardPushState& self) {
    auto result = self.drain_queue();
    if (!result) {
        return py::none();
    }
    py::dict d;
    for (auto& [eid, tensor] : *result) {
        d[py::int_(eid)] = tensor;
    }
    return d;
}

// push_residuals: Python passes dict[int, tuple[Tensor, Tensor, Tensor]].
// Convert to C++ map before delegating.
static void push_residuals_wrapper(PPRForwardPushState& self, py::dict fetched_by_etype_id) {
    std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> cpp_map;
    // Dict iteration touches Python objects — GIL must be held here.
    for (auto item : fetched_by_etype_id) {
        int32_t eid = item.first.cast<int32_t>();
        auto tup = item.second.cast<py::tuple>();
        cpp_map[eid] = {tup[0].cast<torch::Tensor>(), tup[1].cast<torch::Tensor>(), tup[2].cast<torch::Tensor>()};
    }
    // C++ push only uses tensor accessor/data_ptr APIs — GIL-safe to release.
    // Releasing here lets the asyncio event loop process RPC completion callbacks
    // from other concurrent PPR coroutines while this push runs.
    {
        py::gil_scoped_release release;
        self.push_residuals(cpp_map);
    }
}

// extract_top_k: C++ returns map<ntype_id, tuple<Tensor, Tensor, Tensor>>.
// Exposed to Python as dict[int, tuple[Tensor, Tensor, Tensor]].
static py::dict extract_top_k_wrapper(PPRForwardPushState& self, int32_t max_ppr_nodes) {
    auto result = self.extract_top_k(max_ppr_nodes);
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
        .def("drain_queue", drain_queue_wrapper)
        .def("push_residuals", push_residuals_wrapper)
        .def("extract_top_k", extract_top_k_wrapper)
        .def("get_nodes_drained_per_iteration", &PPRForwardPushState::get_nodes_drained_per_iteration);
}
