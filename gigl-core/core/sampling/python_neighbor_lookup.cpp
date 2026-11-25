// Python bindings and the TorchScript operator for the native neighbor lookup.
//
// Importing this module registers torch.ops.gigl_core.lookup_neighbors, which a TorchScript
// function calls so torch RPC serves remote neighbor requests without the interpreter lock, and which Python
// calls for the local partition (a torch operator call releases the lock). The pybind
// functions register a partition's CSR and seed the sub-sampling. The lookup itself is reached only
// through the torch operator; gigl.distributed.utils.native_neighbor_lookup.lookup_neighbors calls it
// from Python.

#include <pybind11/stl.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <cstdint>
#include <string>

#include "neighbor_lookup.h"

namespace py = pybind11;

// FRAGMENT so that other gigl-core extension modules can add operators to the `gigl_core` namespace (only one
// TORCH_LIBRARY per namespace may exist in a process).
TORCH_LIBRARY_FRAGMENT(gigl_core, m) {
    m.def("lookup_neighbors(int sampler_id, str edge_type, Tensor ids, int req_num) -> (Tensor, Tensor, Tensor?)",
          [](int64_t samplerId, const std::string& edgeType, const torch::Tensor& ids, int64_t reqNum) {
              return gigl::lookupNeighbors(samplerId, edgeType, ids, reqNum);
          });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() =
        "One-hop neighbor lookup over a registered CSR, for local and remote requests, without the interpreter lock.";

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
}
