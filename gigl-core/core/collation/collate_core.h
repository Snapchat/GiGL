#pragma once

// Generic collation kernel for distributed neighbor loaders.
//
// Reproduces the per-batch tensor wrangling that a distributed neighbor loader
// performs when turning a flat sampler message (per-type id/feature/edge tensors)
// into the component tensors of a graph batch: per-type dict assembly, the
// edge-direction row/col swap, empty-edge filling, and per-hop count padding.
//
// All input tensors are assumed to already reside on the target device; this
// kernel never issues device transfers. Pure C++ lives here and in collate_core.cpp;
// python_collate_core.cpp handles only Python<->C++ type conversion.

#include <torch/torch.h>

namespace gigl {
namespace collation {

// Sentinel used by the scaffold import test; removed once real entry points land.
int ping();

}  // namespace collation
}  // namespace gigl
