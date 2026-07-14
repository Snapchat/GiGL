#include <gtest/gtest.h>
#include "sampling/ppr_forward_push.h"

#include <unordered_map>
#include <unordered_set>

using gigl::drainTypedPPRChannelQueues;
using gigl::extractTypedTopKWithResidualTopUp;
using gigl::PPRForwardPush;

// Builds a single-edge-type, single-node-type PPRForwardPush.
static PPRForwardPush makeState(const std::vector<int64_t>& seeds,
                                double alpha,
                                double requeueThresholdFactor,
                                const std::vector<int32_t>& degrees) {
    return PPRForwardPush(torch::tensor(seeds, torch::kLong),
                          /*seedNodeTypeId=*/0,
                          alpha,
                          requeueThresholdFactor,
                          /*nodeTypeToEdgeTypeIds=*/{{0}},
                          /*edgeTypeToDstNtypeId=*/{0},
                          {torch::tensor(degrees, torch::kInt)});
}

// Convenience wrapper: build the fetchedByEtypeId argument for pushResiduals
// from flat vectors, keeping test call sites readable.
static std::unordered_map<int32_t, std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>> makeFetched(
    int32_t edgeTypeId,
    const std::vector<int64_t>& nodeIds,
    const std::vector<int64_t>& flatNeighborIds,
    const std::vector<int64_t>& counts) {
    return {{edgeTypeId,
             {torch::tensor(nodeIds, torch::kLong),
              torch::tensor(flatNeighborIds, torch::kLong),
              torch::tensor(counts, torch::kLong)}}};
}

static std::unordered_set<int64_t> tensorValues(const torch::Tensor& values) {
    std::unordered_set<int64_t> result;
    auto accessor = values.accessor<int64_t, 1>();
    for (int64_t index = 0; index < values.size(0); ++index) {
        result.insert(accessor[index]);
    }
    return result;
}

static std::vector<int64_t> tensorToInt64Vector(const torch::Tensor& values) {
    std::vector<int64_t> result;
    auto accessor = values.accessor<int64_t, 1>();
    result.reserve(static_cast<size_t>(values.size(0)));
    for (int64_t index = 0; index < values.size(0); ++index) {
        result.push_back(accessor[index]);
    }
    return result;
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
    auto topk = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/false);
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

    auto topk = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/false);
    ASSERT_NE(topk.find(0), topk.end());
    const auto& [ids, weights, counts] = topk.at(0);
    ASSERT_EQ(counts[0].item<int64_t>(), 2);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_EQ(ids[1].item<int64_t>(), 1);
    EXPECT_NEAR(weights[0].item<float>(), static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weights[1].item<float>(), static_cast<float>((1.0 - alpha) * alpha), 1e-5F);
}

// Once a (node, edge type) neighbor list is fetched, it should be cached for the
// rest of the PPR state. In a cycle, revisiting node 0 should therefore require
// no second lookup for node 0.
TEST(PPRForwardPush, NeighborCacheAvoidsRefetchingPreviouslyFetchedNode) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1}, /*counts=*/{1}));

    auto iter2 = state.drainQueue();
    ASSERT_TRUE(iter2.has_value());
    ASSERT_NE(iter2->find(0), iter2->end());
    ASSERT_EQ(iter2->at(0).size(0), 1);
    EXPECT_EQ(iter2->at(0)[0].item<int64_t>(), 1);
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{1}, /*flatNeighborIds=*/{0}, /*counts=*/{1}));

    auto iter3 = state.drainQueue();
    ASSERT_TRUE(iter3.has_value());
    EXPECT_TRUE(iter3->empty());
}

TEST(PPRForwardPush, DrainTypedPPRChannelQueuesUnionsChannelFrontiers) {
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});
    auto channel1 = makeState(/*seeds=*/{1}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});

    auto drained = drainTypedPPRChannelQueues(std::vector<PPRForwardPush*>{&channel0, &channel1},
                                              /*fetchIterationCounts=*/{0, 0},
                                              /*maxFetchIterations=*/-1);

    EXPECT_EQ(drained.activeChannelIndices, std::vector<int32_t>({0, 1}));
    EXPECT_EQ(drained.fetchChannelIndices, std::vector<int32_t>({0, 1}));
    ASSERT_EQ(drained.edgeTypeIdsByFetchChannel.size(), 2);
    EXPECT_EQ(drained.edgeTypeIdsByFetchChannel[0], std::vector<int32_t>({0}));
    EXPECT_EQ(drained.edgeTypeIdsByFetchChannel[1], std::vector<int32_t>({0}));

    ASSERT_NE(drained.unionNodesByEdgeTypeId.find(0), drained.unionNodesByEdgeTypeId.end());
    EXPECT_EQ(tensorValues(drained.unionNodesByEdgeTypeId.at(0)), std::unordered_set<int64_t>({0, 1}));
}

TEST(PPRForwardPush, DrainTypedPPRChannelQueuesHonorsPerChannelFetchBudget) {
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});
    auto channel1 = makeState(/*seeds=*/{1}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});

    auto drained = drainTypedPPRChannelQueues(std::vector<PPRForwardPush*>{&channel0, &channel1},
                                              /*fetchIterationCounts=*/{1, 0},
                                              /*maxFetchIterations=*/1);

    EXPECT_EQ(drained.activeChannelIndices, std::vector<int32_t>({0, 1}));
    EXPECT_EQ(drained.fetchChannelIndices, std::vector<int32_t>({1}));
    ASSERT_EQ(drained.edgeTypeIdsByFetchChannel.size(), 1);
    EXPECT_EQ(drained.edgeTypeIdsByFetchChannel[0], std::vector<int32_t>({0}));

    ASSERT_NE(drained.unionNodesByEdgeTypeId.find(0), drained.unionNodesByEdgeTypeId.end());
    EXPECT_EQ(tensorValues(drained.unionNodesByEdgeTypeId.at(0)), std::unordered_set<int64_t>({1}));
}

TEST(PPRForwardPush, ExtractTypedTopKWithResidualTopUpMergesChannelsInCpp) {
    std::vector<int32_t> degrees(21, 1);
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1.0, degrees);
    auto channel1 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1.0, degrees);

    channel0.drainQueue();
    channel0.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{10}, /*counts=*/{1}));
    channel1.drainQueue();
    channel1.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{20}, /*counts=*/{1}));

    auto result = extractTypedTopKWithResidualTopUp(std::vector<PPRForwardPush*>{&channel0, &channel1},
                                                    /*channelQuotas=*/{2, 1},
                                                    /*maxPPRNodes=*/3,
                                                    /*enableResidualTopUp=*/true);

    ASSERT_NE(result.find(0), result.end());
    const auto& [ids, features, counts] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(ids), std::vector<int64_t>({0, 10, 20}));
    EXPECT_EQ(tensorToInt64Vector(counts), std::vector<int64_t>({3}));
    ASSERT_EQ(features.size(0), 3);
    ASSERT_EQ(features.size(1), 5);

    auto featureAccessor = features.accessor<double, 2>();
    EXPECT_NEAR(featureAccessor[0][0], 1.0, 1e-9);
    EXPECT_NEAR(featureAccessor[0][1], 1.0, 1e-9);
    EXPECT_NEAR(featureAccessor[0][2], 1.0, 1e-9);
    EXPECT_NEAR(featureAccessor[0][3], 1.0, 1e-9);
    EXPECT_NEAR(featureAccessor[0][4], 1.0, 1e-9);

    EXPECT_NEAR(featureAccessor[1][0], 0.5, 1e-9);
    EXPECT_NEAR(featureAccessor[1][1], 0.5, 1e-9);
    EXPECT_NEAR(featureAccessor[1][2], 0.0, 1e-9);
    EXPECT_NEAR(featureAccessor[1][3], 1.0, 1e-9);
    EXPECT_NEAR(featureAccessor[1][4], 0.0, 1e-9);

    EXPECT_NEAR(featureAccessor[2][0], 0.5, 1e-9);
    EXPECT_NEAR(featureAccessor[2][1], 0.0, 1e-9);
    EXPECT_NEAR(featureAccessor[2][2], 0.5, 1e-9);
    EXPECT_NEAR(featureAccessor[2][3], 0.0, 1e-9);
    EXPECT_NEAR(featureAccessor[2][4], 1.0, 1e-9);
}

TEST(PPRForwardPush, ExtractTypedTopKWithResidualTopUpCanDisableTopUp) {
    std::vector<int32_t> degrees(21, 1);
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1.0, degrees);
    auto channel1 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1.0, degrees);

    channel0.drainQueue();
    channel0.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{10}, /*counts=*/{1}));
    channel1.drainQueue();
    channel1.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{20}, /*counts=*/{1}));

    auto result = extractTypedTopKWithResidualTopUp(std::vector<PPRForwardPush*>{&channel0, &channel1},
                                                    /*channelQuotas=*/{2, 1},
                                                    /*maxPPRNodes=*/3,
                                                    /*enableResidualTopUp=*/false);

    ASSERT_NE(result.find(0), result.end());
    const auto& [ids, features, counts] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(ids), std::vector<int64_t>({0}));
    EXPECT_EQ(tensorToInt64Vector(counts), std::vector<int64_t>({1}));
    ASSERT_EQ(features.size(0), 1);
    ASSERT_EQ(features.size(1), 5);
}

// Two seeds (0 and 1) both push residual to sink node 2.  The neighbor-lookup
// request must deduplicate to one entry for node 2, yet both seeds must still
// accumulate a PPR score for it.
TEST(PPRForwardPush, DeduplicatesNodesAcrossSeeds) {
    auto state = makeState(/*seeds=*/{0, 1}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{1, 1, 0});

    state.drainQueue();
    state.pushResiduals(
        makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0, 1}, /*flatNeighborIds=*/{2, 2}, /*counts=*/{1, 1}));

    auto iter2 = state.drainQueue();
    ASSERT_TRUE(iter2.has_value());
    const auto& iter2Map = iter2.value();
    ASSERT_NE(iter2Map.find(0), iter2Map.end());
    EXPECT_EQ(iter2Map.at(0).size(0), 1); // node 2 deduplicated in the lookup request

    state.pushResiduals({});
    EXPECT_FALSE(state.drainQueue().has_value());

    auto topk = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/false);
    ASSERT_NE(topk.find(0), topk.end());
    const auto& [ids, weights, counts] = topk.at(0);
    // Each seed (batch indices 0 and 1) should have 2 nodes in its top-k.
    EXPECT_EQ(counts[0].item<int64_t>(), 2); // seed 0: nodes {0, 2}
    EXPECT_EQ(counts[1].item<int64_t>(), 2); // seed 1: nodes {1, 2}
    // The flat id layout is [seed0_top1, seed0_top2, seed1_top1, seed1_top2].
    // Within each seed the highest scorer comes first, so seed-node beats node 2.
    EXPECT_EQ(ids[1].item<int64_t>(), 2); // seed 0's second node is node 2
    EXPECT_EQ(ids[3].item<int64_t>(), 2); // seed 1's second node is node 2
}

// extractTopKWithResidualTopUp respects the maxPPRNodes limit when residual top-up is disabled.
TEST(PPRForwardPush, ExtractTopKWithResidualTopUpLimitsResultsWithoutResidualTopUp) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{1, 0});

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1}, /*counts=*/{1}));
    state.drainQueue();
    state.pushResiduals({});

    auto topk1 = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/1, /*enableResidualTopUp=*/false);
    ASSERT_NE(topk1.find(0), topk1.end());
    EXPECT_EQ(std::get<2>(topk1.at(0))[0].item<int64_t>(), 1);

    auto topk10 = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/false);
    ASSERT_NE(topk10.find(0), topk10.end());
    EXPECT_EQ(std::get<2>(topk10.at(0))[0].item<int64_t>(), 2);
}

// Residual top-up includes discovered nodes whose residual never crossed the
// requeue threshold, while enableResidualTopUp=false keeps finalized-PPR-only behavior.
TEST(PPRForwardPush, ExtractTopKWithResidualTopUpIncludesUnpushedResiduals) {
    const double alpha = 0.5;
    auto state = makeState(/*seeds=*/{0}, alpha, /*requeueThresholdFactor=*/1.0, /*degrees=*/{2, 1, 1});

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1, 2}, /*counts=*/{2}));
    EXPECT_FALSE(state.drainQueue().has_value());

    auto topk = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/false);
    ASSERT_NE(topk.find(0), topk.end());
    EXPECT_EQ(std::get<2>(topk.at(0))[0].item<int64_t>(), 1);
    EXPECT_EQ(std::get<0>(topk.at(0))[0].item<int64_t>(), 0);

    auto topkWithResiduals = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/3, /*enableResidualTopUp=*/true);
    ASSERT_NE(topkWithResiduals.find(0), topkWithResiduals.end());
    const auto& [ids, weights, counts] = topkWithResiduals.at(0);
    ASSERT_EQ(counts[0].item<int64_t>(), 3);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_NEAR(weights[0].item<float>(), static_cast<float>(alpha), 1e-5F);

    std::unordered_map<int64_t, float> residualWeights;
    residualWeights[ids[1].item<int64_t>()] = weights[1].item<float>();
    residualWeights[ids[2].item<int64_t>()] = weights[2].item<float>();
    ASSERT_NE(residualWeights.find(1), residualWeights.end());
    ASSERT_NE(residualWeights.find(2), residualWeights.end());
    EXPECT_NEAR(residualWeights[1], static_cast<float>((1.0 - alpha) * alpha / 2.0), 1e-5F);
    EXPECT_NEAR(residualWeights[2], static_cast<float>((1.0 - alpha) * alpha / 2.0), 1e-5F);
}

// Residual top-up cannot displace finalized PPR nodes from the selected set,
// but the final emitted sequence should still be sorted by emitted score.
TEST(PPRForwardPush, ExtractTopKWithResidualTopUpSortsSelectedResultsByScore) {
    const double alpha = 0.5;
    auto state = makeState(/*seeds=*/{0}, alpha, /*requeueThresholdFactor=*/0.1, /*degrees=*/{2, 2, 1, 0});

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1, 2}, /*counts=*/{2}));

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{2}, /*flatNeighborIds=*/{3}, /*counts=*/{1}));

    state.drainQueue();
    state.pushResiduals({});
    EXPECT_FALSE(state.drainQueue().has_value());

    auto topkWithResiduals = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/4, /*enableResidualTopUp=*/true);
    ASSERT_NE(topkWithResiduals.find(0), topkWithResiduals.end());
    const auto& [ids, weights, counts] = topkWithResiduals.at(0);
    ASSERT_EQ(counts[0].item<int64_t>(), 4);

    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_EQ(ids[3].item<int64_t>(), 3);
    for (int64_t index = 1; index < counts[0].item<int64_t>(); ++index) {
        EXPECT_GE(weights[index - 1].item<float>(), weights[index].item<float>());
    }

    std::unordered_map<int64_t, float> weightsByNodeId;
    for (int64_t index = 0; index < counts[0].item<int64_t>(); ++index) {
        weightsByNodeId[ids[index].item<int64_t>()] = weights[index].item<float>();
    }
    EXPECT_NEAR(weightsByNodeId[0], static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weightsByNodeId[1], static_cast<float>((1.0 - alpha) * alpha / 2.0), 1e-5F);
    EXPECT_NEAR(weightsByNodeId[2], static_cast<float>((1.0 - alpha) * alpha / 2.0), 1e-5F);
    EXPECT_NEAR(weightsByNodeId[3], static_cast<float>((1.0 - alpha) * (1.0 - alpha) * alpha / 2.0), 1e-5F);
}

// maxPPRNodes is the total output cap across finalized PPR and residual top-up
// candidates.
TEST(PPRForwardPush, ExtractTopKWithResidualTopUpUsesMaxPPRNodesAsTotalCap) {
    const double alpha = 0.5;
    auto state = makeState(/*seeds=*/{0}, alpha, /*requeueThresholdFactor=*/1.0, /*degrees=*/{2, 1, 1});

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1, 2}, /*counts=*/{2}));
    EXPECT_FALSE(state.drainQueue().has_value());

    auto cappedTopk = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/2, /*enableResidualTopUp=*/true);
    ASSERT_NE(cappedTopk.find(0), cappedTopk.end());
    const auto& [ids, weights, counts] = cappedTopk.at(0);

    ASSERT_EQ(counts[0].item<int64_t>(), 2);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_NEAR(weights[0].item<float>(), static_cast<float>(alpha), 1e-5F);
}

TEST(PPRForwardPush, ExtractTopKWithResidualTopUpRejectsNegativeMaxPPRNodes) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{0});
    EXPECT_THROW(state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/-1, /*enableResidualTopUp=*/true), c10::Error);
}
