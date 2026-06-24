// Python bindings for the generic distributed-neighbor-loader collation kernel.
//
// Pure C++ logic lives in collate_core.{h,cpp}; this file handles only type
// conversion between Python (pybind11) and C++ types, then delegates.

#include <pybind11/stl.h>
#include <torch/extension.h>

#include "collate_core.h"

namespace py = pybind11;

// TORCH_EXTENSION_NAME is set by the build system to match the Python module
// name derived from this file's path (here: "collate_core").
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Generic collation kernel for distributed neighbor loaders.";
    m.def("ping", &gigl::collation::ping, "Scaffold sentinel; returns 0.");
}
