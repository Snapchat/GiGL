#include <gtest/gtest.h>
#include "sampling/ppr_forward_push.h"

// Builds a single-edge-type, single-node-type PPRForwardPushState.
static PPRForwardPushState makeState(
    std::vector<int64_t> seeds,
    double alpha,
    double requeueThresholdFactor,
    std::vector<int32_t> degrees) {
    return PPRForwardPushState(
        torch::tensor(seeds, torch::kLong),
        /*seedNodeTypeId=*/0,
        alpha,
        requeueThresholdFactor,
        /*nodeTypeToEdgeTypeIds=*/{{0}},
        /*edgeTypeToDstNtypeId=*/{0},
        {torch::tensor(degrees, torch::kInt)});
}

// After construction, drainQueue() returns the seed node under etype 0.
TEST(PPRForwardPush, DrainQueueReturnsSeedNodeInitially) {
    auto state = makeState({0}, 0.15, 1e-6, {1});
    auto result = state.drainQueue();
    ASSERT_TRUE(result.has_value());
    ASSERT_NE(result->find(0), result->end());
    EXPECT_EQ(result->at(0).size(0), 1);
    EXPECT_EQ(result->at(0)[0].item<int64_t>(), 0);
}

// After convergence (sink node absorbs all residual), drainQueue() returns nullopt.
TEST(PPRForwardPush, DrainQueueReturnsNulloptAfterConvergence) {
    auto state = makeState({0}, 0.15, 1e-6, {0});  // node 0 is a sink (degree 0)
    state.drainQueue();
    state.pushResiduals({});
    EXPECT_FALSE(state.drainQueue().has_value());
}

// A sink seed node absorbs its full residual as PPR score (= alpha).
TEST(PPRForwardPush, PprScoreAbsorbsAlpha) {
    const double alpha = 0.15;
    auto state = makeState({0}, alpha, 1e-6, {0});
    state.drainQueue();
    state.pushResiduals({});
    auto topk = state.extractTopK(10);
    ASSERT_NE(topk.find(0), topk.end());
    const auto& [ids, weights, counts] = topk.at(0);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_NEAR(weights[0].item<float>(), static_cast<float>(alpha), 1e-5f);
}

// Node 0 (degree 1) pushes (1-alpha)*alpha residual to node 1 (sink).
TEST(PPRForwardPush, ResidualDistributedToNeighbor) {
    const double alpha = 0.15;
    auto state = makeState({0}, alpha, 1e-6, {1, 0});

    // Iteration 1: seed node 0 → neighbor node 1.
    state.drainQueue();
    state.pushResiduals({{0,
        {torch::tensor(std::vector<int64_t>{0}, torch::kLong),
         torch::tensor(std::vector<int64_t>{1}, torch::kLong),
         torch::tensor(std::vector<int64_t>{1}, torch::kLong)}}});

    // Iteration 2: node 1 is a sink; absorbs its residual, no further push.
    state.drainQueue();
    state.pushResiduals({});

    EXPECT_FALSE(state.drainQueue().has_value());

    auto topk = state.extractTopK(10);
    ASSERT_NE(topk.find(0), topk.end());
    const auto& [ids, weights, counts] = topk.at(0);
    ASSERT_EQ(counts[0].item<int64_t>(), 2);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_EQ(ids[1].item<int64_t>(), 1);
    EXPECT_NEAR(weights[0].item<float>(), static_cast<float>(alpha), 1e-5f);
    EXPECT_NEAR(weights[1].item<float>(), static_cast<float>((1.0 - alpha) * alpha), 1e-5f);
}

// Two seeds both push residual to node 2; the neighbor-lookup request deduplicates
// to one entry, but getNodesDrainedPerIteration counts both seed queues.
TEST(PPRForwardPush, DeduplicatesNodesAcrossSeeds) {
    auto state = makeState({0, 1}, 0.15, 1e-6, {1, 1, 0});

    state.drainQueue();
    state.pushResiduals({{0,
        {torch::tensor(std::vector<int64_t>{0, 1}, torch::kLong),
         torch::tensor(std::vector<int64_t>{2, 2}, torch::kLong),
         torch::tensor(std::vector<int64_t>{1, 1}, torch::kLong)}}});

    auto iter2 = state.drainQueue();
    ASSERT_TRUE(iter2.has_value());
    ASSERT_NE(iter2->find(0), iter2->end());
    EXPECT_EQ(iter2->at(0).size(0), 1);               // node 2 deduplicated in lookup
    EXPECT_EQ(state.getNodesDrainedPerIteration()[1], 2);  // but drained from 2 seed queues
}

// extractTopK respects the maxPprNodes limit.
TEST(PPRForwardPush, ExtractTopKLimitsResults) {
    auto state = makeState({0}, 0.15, 1e-6, {1, 0});

    state.drainQueue();
    state.pushResiduals({{0,
        {torch::tensor(std::vector<int64_t>{0}, torch::kLong),
         torch::tensor(std::vector<int64_t>{1}, torch::kLong),
         torch::tensor(std::vector<int64_t>{1}, torch::kLong)}}});
    state.drainQueue();
    state.pushResiduals({});

    auto topk1 = state.extractTopK(1);
    ASSERT_NE(topk1.find(0), topk1.end());
    EXPECT_EQ(std::get<2>(topk1.at(0))[0].item<int64_t>(), 1);

    auto topk10 = state.extractTopK(10);
    ASSERT_NE(topk10.find(0), topk10.end());
    EXPECT_EQ(std::get<2>(topk10.at(0))[0].item<int64_t>(), 2);
}
