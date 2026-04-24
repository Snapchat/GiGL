#include "add_one.h"

#include <torch/extension.h>

at::Tensor addOne(const at::Tensor& input) {
    TORCH_CHECK(input.device().is_cpu(), "add_one requires a CPU tensor");
    return input + 1;
}
