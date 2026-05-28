#include <gtest/gtest.h>
#include "sampling/ppr_forward_push.h"

using gigl::PPRForwardPush;

using FetchedByEtypeId = std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>;

// Builds a single-edge-type, single-node-type PPRForwardPush.
static PPRForwardPush makeState(
    const std::vector<int64_t>& seeds,
    double alpha,
    double requeueThresholdFactor,
    const std::vector<int32_t>& degrees) {
    return PPRForwardPush(
        torch::tensor(seeds, torch::kLong),
        /*seedNodeTypeId=*/0,
        alpha,
        requeueThresholdFactor,
        /*nodeTypeToEdgeTypeIds=*/{{0}},
        /*edgeTypeToDstNtypeId=*/{0},
        {torch::tensor(degrees, torch::kInt)});
}

// Convenience wrapper: build the fetchedByEtypeId argument for pushResiduals
// from flat vectors, keeping test call sites readable.  Empty weights select
// uniform-residual mode.
static FetchedByEtypeId
makeFetched(int32_t edgeTypeId,
            const std::vector<int64_t>& nodeIds,
            const std::vector<int64_t>& flatNeighborIds,
            const std::vector<int64_t>& counts,
            const std::vector<double>& flatWeights = {}) {
    auto weightsTensor =
        flatWeights.empty() ? torch::empty({0}, torch::kDouble) : torch::tensor(flatWeights, torch::kDouble);
    return {{edgeTypeId,
             {torch::tensor(nodeIds, torch::kLong),
              torch::tensor(flatNeighborIds, torch::kLong),
              torch::tensor(counts, torch::kLong),
              weightsTensor}}};
}

// After construction, drainQueue() returns the seed node under etype 0.
TEST(PPRForwardPush, DrainQueueReturnsSeedNodeInitially) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{1});
    auto result = state.drainQueue();
    ASSERT_TRUE(result.has_value());
    const auto& nodeMap = result.value();
    ASSERT_NE(nodeMap.find(0), nodeMap.end());
    EXPECT_EQ(nodeMap.at(0).size(0), 1);
    EXPECT_EQ(nodeMap.at(0)[0].item<int64_t>(), 0);
}

// After convergence (sink node absorbs all residual), drainQueue() returns nullopt.
TEST(PPRForwardPush, DrainQueueReturnsNulloptAfterConvergence) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{0});
    state.drainQueue();
    state.pushResiduals({});
    EXPECT_FALSE(state.drainQueue().has_value());
}

// A sink seed node absorbs its full residual as PPR score (= alpha).
TEST(PPRForwardPush, PprScoreAbsorbsAlpha) {
    const double alpha = 0.15;
    auto state = makeState(/*seeds=*/{0}, alpha, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{0});
    state.drainQueue();
    state.pushResiduals({});
    auto topk = state.extractTopK(10);
    ASSERT_NE(topk.find(0), topk.end());
    const auto& [ids, weights, counts] = topk.at(0);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_NEAR(weights[0].item<float>(), static_cast<float>(alpha), 1e-5F);
}

// Node 0 (degree 1) pushes (1-alpha)*alpha residual to node 1 (sink).
TEST(PPRForwardPush, ResidualDistributedToNeighbor) {
    const double alpha = 0.15;
    auto state = makeState(/*seeds=*/{0}, alpha, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{1, 0});

    // Iteration 1: seed node 0 → neighbor node 1.
    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1}, /*counts=*/{1}));

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
    EXPECT_NEAR(weights[0].item<float>(), static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weights[1].item<float>(), static_cast<float>((1.0 - alpha) * alpha), 1e-5F);
}

// In weighted mode, zero-weight edges must not enqueue a zero-residual neighbor.
TEST(PPRForwardPush, WeightedResidualSkipsZeroWeightNeighbor) {
    const double alpha = 0.5;
    auto state = makeState(/*seeds=*/{0}, alpha, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{2, 0, 0});

    state.drainQueue();
    state.pushResiduals(makeFetched(
        /*edgeTypeId=*/0,
        /*nodeIds=*/{0},
        /*flatNeighborIds=*/{1, 2},
        /*counts=*/{2},
        /*flatWeights=*/{0.0, 2.0}));

    auto iter2 = state.drainQueue();
    ASSERT_TRUE(iter2.has_value());
    const auto& iter2Map = iter2.value();
    ASSERT_NE(iter2Map.find(0), iter2Map.end());
    ASSERT_EQ(iter2Map.at(0).size(0), 1);
    EXPECT_EQ(iter2Map.at(0)[0].item<int64_t>(), 2);

    state.pushResiduals({});
    EXPECT_FALSE(state.drainQueue().has_value());

    auto topk = state.extractTopK(10);
    ASSERT_NE(topk.find(0), topk.end());
    const auto& [ids, weights, counts] = topk.at(0);
    ASSERT_EQ(counts[0].item<int64_t>(), 2);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_EQ(ids[1].item<int64_t>(), 2);
    EXPECT_NEAR(weights[0].item<float>(), static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weights[1].item<float>(), static_cast<float>((1.0 - alpha) * alpha), 1e-5F);
}

// Two seeds (0 and 1) both push residual to sink node 2.  The neighbor-lookup
// request must deduplicate to one entry for node 2, yet both seeds must still
// accumulate a PPR score for it.
TEST(PPRForwardPush, DeduplicatesNodesAcrossSeeds) {
    auto state = makeState(/*seeds=*/{0, 1}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{1, 1, 0});

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0, 1}, /*flatNeighborIds=*/{2, 2}, /*counts=*/{1, 1}));

    auto iter2 = state.drainQueue();
    ASSERT_TRUE(iter2.has_value());
    const auto& iter2Map = iter2.value();
    ASSERT_NE(iter2Map.find(0), iter2Map.end());
    EXPECT_EQ(iter2Map.at(0).size(0), 1);  // node 2 deduplicated in the lookup request

    state.pushResiduals({});
    EXPECT_FALSE(state.drainQueue().has_value());

    auto topk = state.extractTopK(10);
    ASSERT_NE(topk.find(0), topk.end());
    const auto& [ids, weights, counts] = topk.at(0);
    // Each seed (batch indices 0 and 1) should have 2 nodes in its top-k.
    EXPECT_EQ(counts[0].item<int64_t>(), 2);  // seed 0: nodes {0, 2}
    EXPECT_EQ(counts[1].item<int64_t>(), 2);  // seed 1: nodes {1, 2}
    // The flat id layout is [seed0_top1, seed0_top2, seed1_top1, seed1_top2].
    // Within each seed the highest scorer comes first, so seed-node beats node 2.
    EXPECT_EQ(ids[1].item<int64_t>(), 2);  // seed 0's second node is node 2
    EXPECT_EQ(ids[3].item<int64_t>(), 2);  // seed 1's second node is node 2
}

// extractTopK respects the maxPprNodes limit.
TEST(PPRForwardPush, ExtractTopKLimitsResults) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{1, 0});

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1}, /*counts=*/{1}));
    state.drainQueue();
    state.pushResiduals({});

    auto topk1 = state.extractTopK(1);
    ASSERT_NE(topk1.find(0), topk1.end());
    EXPECT_EQ(std::get<2>(topk1.at(0))[0].item<int64_t>(), 1);

    auto topk10 = state.extractTopK(10);
    ASSERT_NE(topk10.find(0), topk10.end());
    EXPECT_EQ(std::get<2>(topk10.at(0))[0].item<int64_t>(), 2);
}
