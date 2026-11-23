// Python bindings and the TorchScript operator for the RPC neighbor lookup.
//
// Importing this module registers torch.ops.gigl_core.rpc_lookup_neighbors, which a TorchScript
// function calls so torch RPC serves neighbor requests without the interpreter lock. The pybind
// functions register a partition's CSR and call the lookup directly.

#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdint>
#include <string>

#include "rpc_neighbor_lookup.h"

namespace py = pybind11;

TORCH_LIBRARY(gigl_core, m) {
    m.def("rpc_lookup_neighbors(int sampler_id, str edge_type, Tensor ids, int req_num) -> (Tensor, Tensor, Tensor)",
          [](int64_t samplerId, const std::string& edgeType, const torch::Tensor& ids, int64_t reqNum) {
              return gigl::lookupNeighbors(samplerId, edgeType, ids, reqNum);
          });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "One-hop neighbor lookup served by a TorchScript RPC without the interpreter lock.";

    m.def("register_csr",
          &gigl::registerCsr,
          py::arg("sampler_id"),
          py::arg("edge_type"),
          py::arg("indptr"),
          py::arg("indices"),
          py::arg("edge_ids"),
          py::arg("compressed_id_offset"),
          py::arg("with_edge"));
    m.def("unregister_csr", &gigl::unregisterCsr, py::arg("sampler_id"));
    m.def("set_random_seed", &gigl::setRandomSeed, py::arg("seed"));
    m.def("lookup_neighbors",
          &gigl::lookupNeighbors,
          py::arg("sampler_id"),
          py::arg("edge_type"),
          py::arg("ids"),
          py::arg("req_num"),
          py::call_guard<py::gil_scoped_release>());
}
