#include <gtest/gtest.h>
#include <torch/torch.h>

#include "collation/collate_core.h"

namespace {

TEST(PadCount, PadsShortTensorWithZeros) {
    auto t = torch::tensor({2, 3}, torch::kInt64);
    auto out = gigl::collation::padCount(t, 4);
    auto expected = torch::tensor({2, 3, 0, 0}, torch::kInt64);
    EXPECT_TRUE(torch::equal(out, expected));
}

TEST(PadCount, ExactLengthIsUnchanged) {
    auto t = torch::tensor({1, 2, 3}, torch::kInt64);
    auto out = gigl::collation::padCount(t, 3);
    EXPECT_TRUE(torch::equal(out, t));
}

TEST(ZeroCount, BuildsZeroVectorOfTargetLength) {
    auto opts = torch::TensorOptions().dtype(torch::kInt64);
    auto out = gigl::collation::zeroCount(3, opts);
    auto expected = torch::tensor({0, 0, 0}, torch::kInt64);
    EXPECT_TRUE(torch::equal(out, expected));
}

}  // namespace
