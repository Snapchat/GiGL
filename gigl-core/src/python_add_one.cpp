#include "add_one.h"

#include <torch/extension.h>

#include <pybind11/pybind11.h>

PYBIND11_MODULE(_core, m) {
    m.doc() = "GiGL core pybind11 extension module.";
    m.def("add_one",
          &addOne,
          "Return a new tensor equal to the input with one added to each element. Requires a CPU tensor.");
}
