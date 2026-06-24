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

TEST(CollateHomogeneous, StacksEdgeIndexFromProvidedRowCol) {
    // Caller passes rows/cols already in stack order (row first). The GLT homogeneous
    // path passes cols as row and rows as col (dist_loader.py:446); here we verify the
    // kernel stacks [rowsArg, colsArg] verbatim so the caller controls the swap.
    auto ids = torch::tensor({10, 11, 12}, torch::kInt64);
    auto rowsArg = torch::tensor({0, 1}, torch::kInt64);
    auto colsArg = torch::tensor({1, 2}, torch::kInt64);
    auto res = gigl::collation::collateHomogeneous(
        ids, rowsArg, colsArg,
        /*eids=*/c10::nullopt, /*nfeats=*/c10::nullopt, /*efeats=*/c10::nullopt,
        /*batch=*/c10::nullopt, /*numSampledNodes=*/c10::nullopt,
        /*numSampledEdges=*/c10::nullopt);
    auto expectedEdgeIndex = torch::stack({rowsArg, colsArg});
    EXPECT_TRUE(torch::equal(res.edgeIndex, expectedEdgeIndex));
    EXPECT_TRUE(torch::equal(res.node, ids));
    EXPECT_FALSE(res.eid.has_value());
    EXPECT_FALSE(res.numSampledNodes.has_value());
}

}  // namespace
