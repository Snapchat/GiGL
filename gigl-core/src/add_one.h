#pragma once

#include <torch/extension.h>

at::Tensor addOne(const at::Tensor& input);
