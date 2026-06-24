// Python bindings for the generic distributed-neighbor-loader collation kernel.
//
// Pure C++ logic lives in collate_core.{h,cpp}; this file handles only type
// conversion between Python (pybind11) and C++ types, then delegates. The GIL is
// released around the pure-C++ call (no Python objects are touched there).

#include <pybind11/stl.h>
#include <torch/extension.h>

#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "collate_core.h"

namespace py = pybind11;

namespace {

// Convert a Python 3-tuple (or list) into an EdgeTypeArray. GIL must be held.
gigl::collation::EdgeTypeArray toEdgeTypeArray(const py::handle& obj) {
    auto seq = obj.cast<py::sequence>();
    return gigl::collation::EdgeTypeArray{
        seq[0].cast<std::string>(), seq[1].cast<std::string>(), seq[2].cast<std::string>()};
}

py::tuple fromEdgeTypeArray(const gigl::collation::EdgeTypeArray& e) {
    return py::make_tuple(e[0], e[1], e[2]);
}

// Build the homogeneous component dict (str -> Tensor | None). GIL must be held.
py::dict homogeneousToDict(const gigl::collation::HomogeneousCollateResult& r) {
    py::dict out;
    out["node"] = r.node;
    out["edge_index"] = r.edgeIndex;
    out["edge"] = r.eid ? py::cast(*r.eid) : py::none();
    out["x"] = r.x ? py::cast(*r.x) : py::none();
    out["edge_attr"] = r.edgeAttr ? py::cast(*r.edgeAttr) : py::none();
    out["batch"] = r.batch ? py::cast(*r.batch) : py::none();
    out["num_sampled_nodes"] = r.numSampledNodes ? py::cast(*r.numSampledNodes) : py::none();
    out["num_sampled_edges"] = r.numSampledEdges ? py::cast(*r.numSampledEdges) : py::none();
    return out;
}

py::dict collateHomogeneousBinding(const torch::Tensor& ids,
                                   const torch::Tensor& rows,
                                   const torch::Tensor& cols,
                                   const std::optional<torch::Tensor>& eids,
                                   const std::optional<torch::Tensor>& nfeats,
                                   const std::optional<torch::Tensor>& efeats,
                                   const std::optional<torch::Tensor>& batch,
                                   const std::optional<torch::Tensor>& numSampledNodes,
                                   const std::optional<torch::Tensor>& numSampledEdges) {
    gigl::collation::HomogeneousCollateResult result;
    {
        // Tensor ops only — GIL-safe to release. Inputs must not be mutated by Python
        // while released (the caller passes freshly-parsed message tensors).
        py::gil_scoped_release release;
        result = gigl::collation::collateHomogeneous(
            ids, rows, cols, eids, nfeats, efeats, batch, numSampledNodes, numSampledEdges);
    }
    return homogeneousToDict(result);
}

gigl::collation::HeterogeneousCollateResult collateHeterogeneousBinding(
    const py::dict& msg,
    const std::vector<std::string>& nodeTypes,
    const py::dict& edgeTypeStrToRev,
    const py::list& reversedEdgeTypes,
    const std::string& inputType,
    bool hasBatch,
    int64_t batchSize) {
    // Convert Python containers to C++ types (GIL held here).
    std::unordered_map<std::string, torch::Tensor> msgCpp;
    for (auto item : msg) {
        msgCpp[item.first.cast<std::string>()] = item.second.cast<torch::Tensor>();
    }
    std::vector<std::pair<std::string, gigl::collation::EdgeTypeArray>> mapCpp;
    for (auto item : edgeTypeStrToRev) {
        mapCpp.emplace_back(item.first.cast<std::string>(), toEdgeTypeArray(item.second));
    }
    std::vector<gigl::collation::EdgeTypeArray> revCpp;
    for (auto item : reversedEdgeTypes) {
        revCpp.push_back(toEdgeTypeArray(item));
    }
    gigl::collation::HeterogeneousCollateResult result;
    {
        py::gil_scoped_release release;
        result = gigl::collation::collateHeterogeneous(
            msgCpp, nodeTypes, mapCpp, revCpp, inputType, hasBatch, batchSize);
    }
    return result;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Generic collation kernel for distributed neighbor loaders.";

    py::class_<gigl::collation::HeterogeneousCollateResult>(m, "CollateHeteroResult")
        .def_property_readonly("node",
            [](const gigl::collation::HeterogeneousCollateResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.node) d[py::str(k)] = v;
                return d;
            })
        .def_property_readonly("edge_index",
            [](const gigl::collation::HeterogeneousCollateResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.edgeIndex) d[fromEdgeTypeArray(k)] = v;
                return d;
            })
        .def_property_readonly("edge",
            [](const gigl::collation::HeterogeneousCollateResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.edge) d[fromEdgeTypeArray(k)] = v;
                return d;
            })
        .def_property_readonly("x",
            [](const gigl::collation::HeterogeneousCollateResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.x) d[py::str(k)] = v;
                return d;
            })
        .def_property_readonly("edge_attr",
            [](const gigl::collation::HeterogeneousCollateResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.edgeAttr) d[fromEdgeTypeArray(k)] = v;
                return d;
            })
        .def_property_readonly("batch",
            [](const gigl::collation::HeterogeneousCollateResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.batch) d[py::str(k)] = v;
                return d;
            })
        .def_property_readonly("num_sampled_nodes",
            [](const gigl::collation::HeterogeneousCollateResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.numSampledNodes) d[py::str(k)] = v;
                return d;
            })
        .def_property_readonly("num_sampled_edges",
            [](const gigl::collation::HeterogeneousCollateResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.numSampledEdges) d[fromEdgeTypeArray(k)] = v;
                return d;
            });

    m.def("collate_homogeneous", &collateHomogeneousBinding,
          py::arg("ids"), py::arg("rows"), py::arg("cols"), py::arg("eids"),
          py::arg("nfeats"), py::arg("efeats"), py::arg("batch"),
          py::arg("num_sampled_nodes"), py::arg("num_sampled_edges"),
          "Collate a homogeneous sampler message into component tensors.");

    m.def("collate_heterogeneous", &collateHeterogeneousBinding,
          py::arg("msg"), py::arg("node_types"), py::arg("edge_type_str_to_rev"),
          py::arg("reversed_edge_types"), py::arg("input_type"), py::arg("has_batch"),
          py::arg("batch_size"),
          "Collate a heterogeneous sampler message into component tensors.");
}
