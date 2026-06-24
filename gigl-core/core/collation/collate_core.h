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

// Right-pad a 1-D per-hop count tensor with zeros to `targetLen` on its own device.
// Mirrors torch.nn.functional.pad(t, (0, targetLen - t.size(0))).
// Precondition: counts.dim() == 1 and counts.size(0) <= targetLen.
torch::Tensor padCount(const torch::Tensor& counts, int64_t targetLen);

// Build a length-`targetLen` zero count vector with the given options (dtype/device).
// Mirrors torch.tensor([0]*targetLen, device=...).
torch::Tensor zeroCount(int64_t targetLen, const torch::TensorOptions& options);

}  // namespace collation
}  // namespace gigl
