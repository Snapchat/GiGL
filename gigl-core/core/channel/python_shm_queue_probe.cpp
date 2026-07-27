// Python bindings for ShmQueueProbe.
//
// The probe itself lives in shm_queue_probe.{h,cpp}; this file only exposes it to Python.

#include <pybind11/pybind11.h>

#include "shm_queue_probe.h"

namespace py = pybind11;

namespace gigl {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Read-only queue-depth probe for GraphLearn-for-PyTorch shared-memory channels.";

    py::class_<ShmQueueProbe>(m, "ShmQueueProbe")
        // std::system_error derives from std::runtime_error, so pybind11 surfaces it as RuntimeError.
        // That matches what the previous ctypes-based qsize() raised, so callers need no changes.
        .def(py::init<int>(),
             py::arg("shmid"),
             "Attach to a GLT ShmQueue segment by its System V shmid. Raises RuntimeError if the "
             "segment cannot be attached.")
        // Bound as "qsize" to match the Python queue/channel vocabulary the callers already use;
        // the C++ name is queueSize() to satisfy the repo's camelBack method convention.
        // No py::call_guard<py::gil_scoped_release>: the body is two relaxed loads, so releasing and
        // reacquiring the GIL would cost far more than the work it guards.
        .def("qsize", &ShmQueueProbe::queueSize, "Approximate number of messages currently in the channel.")
        .def_property_readonly("shmid", &ShmQueueProbe::shmId, "The System V shmid this probe is attached to.");
}

} // namespace gigl
