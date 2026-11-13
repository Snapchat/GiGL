#include <gtest/gtest.h>
#include "sampling/ppr_forward_push.h"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>

using gigl::drainTypedPPRChannelQueues;
using gigl::extractOriginalEdgesFromPPRCaches;
using gigl::extractTypedTopKWithResidualTopUp;
using gigl::NeighborFetchMap;
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
static NeighborFetchMap makeFetched(int32_t edgeTypeId,
                                    const std::vector<int64_t>& nodeIds,
                                    const std::vector<int64_t>& flatNeighborIds,
                                    const std::vector<int64_t>& counts) {
    return {{edgeTypeId,
             {torch::tensor(nodeIds, torch::kLong),
              torch::tensor(flatNeighborIds, torch::kLong),
              torch::tensor(counts, torch::kLong),
              std::nullopt}}};
}

static NeighborFetchMap makeFetchedWithEdgeIds(int32_t edgeTypeId,
                                               const std::vector<int64_t>& nodeIds,
                                               const std::vector<int64_t>& flatNeighborIds,
                                               const std::vector<int64_t>& counts,
                                               const std::vector<int64_t>& edgeIds) {
    return {{edgeTypeId,
             {torch::tensor(nodeIds, torch::kLong),
              torch::tensor(flatNeighborIds, torch::kLong),
              torch::tensor(counts, torch::kLong),
              torch::tensor(edgeIds, torch::kLong)}}};
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

static void expectTypedPPRFeatureRowNear(const torch::Tensor& features,
                                         int64_t rowIndex,
                                         double bestScore,
                                         double hopProximity,
                                         const std::vector<double>& channelScores,
                                         const std::vector<double>& channelHopProximities,
                                         const std::vector<double>& channelPresence) {
    SCOPED_TRACE(::testing::Message() << "typed PPR feature row " << rowIndex);
    ASSERT_EQ(features.dim(), 2);
    ASSERT_GE(rowIndex, 0);
    ASSERT_LT(rowIndex, features.size(0));
    ASSERT_EQ(channelScores.size(), channelHopProximities.size());
    ASSERT_EQ(channelScores.size(), channelPresence.size());

    const auto numChannels = static_cast<int64_t>(channelScores.size());
    ASSERT_EQ(features.size(1), 2 + (3 * numChannels));

    auto featureAccessor = features.accessor<double, 2>();
    EXPECT_NEAR(featureAccessor[rowIndex][0], bestScore, 1e-9);
    EXPECT_NEAR(featureAccessor[rowIndex][1], hopProximity, 1e-9);
    for (int64_t channelIndex = 0; channelIndex < numChannels; ++channelIndex) {
        const int64_t channelBaseIndex = 2 + (3 * channelIndex);
        EXPECT_NEAR(featureAccessor[rowIndex][channelBaseIndex], channelScores[channelIndex], 1e-9);
        EXPECT_NEAR(featureAccessor[rowIndex][channelBaseIndex + 1], channelHopProximities[channelIndex], 1e-9);
        EXPECT_NEAR(featureAccessor[rowIndex][channelBaseIndex + 2], channelPresence[channelIndex], 1e-9);
    }
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
    const auto& [ids, weights, counts, isExpanded] = topk.at(0);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    ASSERT_EQ(weights.size(1), 2);
    EXPECT_NEAR(weights[0][0].item<float>(), static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weights[0][1].item<float>(), 1.0F, 1e-5F);
    EXPECT_EQ(isExpanded[0].item<int64_t>(), 1);
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
    const auto& [ids, weights, counts, isExpanded] = topk.at(0);
    ASSERT_EQ(counts[0].item<int64_t>(), 2);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_EQ(ids[1].item<int64_t>(), 1);
    ASSERT_EQ(weights.size(1), 2);
    EXPECT_NEAR(weights[0][0].item<float>(), static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weights[0][1].item<float>(), 1.0F, 1e-5F);
    EXPECT_NEAR(weights[1][0].item<float>(), static_cast<float>((1.0 - alpha) * alpha), 1e-5F);
    EXPECT_NEAR(weights[1][1].item<float>(), 0.5F, 1e-5F);
}

TEST(PPRForwardPush, HopProximityIsScopedPerSeed) {
    auto state = makeState(/*seeds=*/{0, 1}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1.0, /*degrees=*/{1, 0});

    state.drainQueue();
    state.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0, 1}, /*flatNeighborIds=*/{1}, /*counts=*/{1, 0}));
    state.drainQueue();
    state.pushResiduals({});

    auto topk = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/false);
    ASSERT_NE(topk.find(0), topk.end());
    const auto& [ids, weights, counts, isExpanded] = topk.at(0);

    EXPECT_EQ(tensorToInt64Vector(ids), std::vector<int64_t>({0, 1, 1}));
    EXPECT_EQ(tensorToInt64Vector(counts), std::vector<int64_t>({2, 1}));
    ASSERT_EQ(weights.size(1), 2);
    EXPECT_NEAR(weights[0][1].item<float>(), 1.0F, 1e-5F); // seed 0 anchor
    EXPECT_NEAR(weights[1][1].item<float>(), 0.5F, 1e-5F); // seed 0 reaches node 1 at 1-hop
    EXPECT_NEAR(weights[2][1].item<float>(), 1.0F, 1e-5F); // seed 1 anchor remains isolated
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

TEST(PPRForwardPush, ExtractOriginalEdgesFiltersToSelectedEndpoints) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{3, 0, 1});

    state.drainQueue();
    state.pushResiduals(
        makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0, 2}, /*flatNeighborIds=*/{0, 1, 2, 0}, /*counts=*/{3, 1}));

    auto result = extractOriginalEdgesFromPPRCaches(std::vector<const PPRForwardPush*>{&state},
                                                    {{0, torch::tensor({0, 1}, torch::kLong)}},
                                                    /*includeEdgeIds=*/false);

    ASSERT_NE(result.find(0), result.end());
    const auto& [rows, cols, edgeIds] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(rows), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(tensorToInt64Vector(cols), std::vector<int64_t>({0, 1}));
    EXPECT_FALSE(edgeIds.has_value());
}

TEST(PPRForwardPush, ExtractOriginalEdgesPreservesEdgeIds) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{3, 0, 1});

    state.drainQueue();
    state.pushResiduals(makeFetchedWithEdgeIds(/*edgeTypeId=*/0,
                                               /*nodeIds=*/{0, 2},
                                               /*flatNeighborIds=*/{0, 1, 2, 0},
                                               /*counts=*/{3, 1},
                                               /*edgeIds=*/{10, 11, 12, 13}));

    auto result = extractOriginalEdgesFromPPRCaches(std::vector<const PPRForwardPush*>{&state},
                                                    {{0, torch::tensor({0, 1}, torch::kLong)}},
                                                    /*includeEdgeIds=*/true);

    ASSERT_NE(result.find(0), result.end());
    const auto& [rows, cols, edgeIds] = result.at(0);
    ASSERT_TRUE(edgeIds.has_value());
    EXPECT_EQ(tensorToInt64Vector(rows), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(tensorToInt64Vector(cols), std::vector<int64_t>({0, 1}));
    EXPECT_EQ(tensorToInt64Vector(edgeIds.value()), std::vector<int64_t>({10, 11}));
}

TEST(PPRForwardPush, ExtractOriginalEdgesDeduplicatesDuplicateEdgesAcrossStates) {
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 0});
    auto channel1 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 0});

    channel0.drainQueue();
    channel0.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1}, /*counts=*/{1}));
    channel1.drainQueue();
    channel1.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1}, /*counts=*/{1}));

    auto result = extractOriginalEdgesFromPPRCaches(std::vector<const PPRForwardPush*>{&channel0, &channel1},
                                                    {{0, torch::tensor({0, 1}, torch::kLong)}},
                                                    /*includeEdgeIds=*/false);

    ASSERT_NE(result.find(0), result.end());
    const auto& [rows, cols, edgeIds] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(rows), std::vector<int64_t>({0}));
    EXPECT_EQ(tensorToInt64Vector(cols), std::vector<int64_t>({1}));
    EXPECT_FALSE(edgeIds.has_value());
}

TEST(PPRForwardPush, ExtractOriginalEdgesKeepsDifferentNeighborsFromSameRowAcrossStates) {
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{2, 0, 0});
    auto channel1 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{2, 0, 0});

    channel0.drainQueue();
    channel0.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1}, /*counts=*/{1}));
    channel1.drainQueue();
    channel1.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{2}, /*counts=*/{1}));

    auto result = extractOriginalEdgesFromPPRCaches(std::vector<const PPRForwardPush*>{&channel0, &channel1},
                                                    {{0, torch::tensor({0, 1, 2}, torch::kLong)}},
                                                    /*includeEdgeIds=*/false);

    ASSERT_NE(result.find(0), result.end());
    const auto& [rows, cols, edgeIds] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(rows), std::vector<int64_t>({0, 0}));
    EXPECT_EQ(tensorToInt64Vector(cols), std::vector<int64_t>({1, 2}));
    EXPECT_FALSE(edgeIds.has_value());
}

TEST(PPRForwardPush, ExtractOriginalEdgesDeduplicatesStructurallyWhenEdgeIdsAreAvailable) {
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{2, 0});
    auto channel1 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{2, 0});
    auto channel2 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{2, 0});

    channel0.drainQueue();
    channel0.pushResiduals(makeFetchedWithEdgeIds(/*edgeTypeId=*/0,
                                                  /*nodeIds=*/{0},
                                                  /*flatNeighborIds=*/{1},
                                                  /*counts=*/{1},
                                                  /*edgeIds=*/{10}));
    channel1.drainQueue();
    channel1.pushResiduals(makeFetchedWithEdgeIds(/*edgeTypeId=*/0,
                                                  /*nodeIds=*/{0},
                                                  /*flatNeighborIds=*/{1},
                                                  /*counts=*/{1},
                                                  /*edgeIds=*/{10}));
    channel2.drainQueue();
    channel2.pushResiduals(makeFetchedWithEdgeIds(/*edgeTypeId=*/0,
                                                  /*nodeIds=*/{0},
                                                  /*flatNeighborIds=*/{1},
                                                  /*counts=*/{1},
                                                  /*edgeIds=*/{11}));

    auto result = extractOriginalEdgesFromPPRCaches(std::vector<const PPRForwardPush*>{&channel0, &channel1, &channel2},
                                                    {{0, torch::tensor({0, 1}, torch::kLong)}},
                                                    /*includeEdgeIds=*/true);

    ASSERT_NE(result.find(0), result.end());
    const auto& [rows, cols, edgeIds] = result.at(0);
    ASSERT_TRUE(edgeIds.has_value());
    EXPECT_EQ(tensorToInt64Vector(rows), std::vector<int64_t>({0}));
    EXPECT_EQ(tensorToInt64Vector(cols), std::vector<int64_t>({1}));
    EXPECT_EQ(tensorToInt64Vector(edgeIds.value()), std::vector<int64_t>({10}));
}

TEST(PPRForwardPush, DrainTypedPPRChannelQueuesUnionsChannelFrontiers) {
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});
    auto channel1 = makeState(/*seeds=*/{1}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});

    auto queueDrainResult = drainTypedPPRChannelQueues(std::vector<PPRForwardPush*>{&channel0, &channel1},
                                                       /*fetchIterationCounts=*/{0, 0},
                                                       /*maxFetchIterations=*/-1);

    EXPECT_EQ(queueDrainResult.drainedChannelIndices, std::vector<int32_t>({0, 1}));
    EXPECT_EQ(queueDrainResult.fetchChannelIndices, std::vector<int32_t>({0, 1}));
    ASSERT_EQ(queueDrainResult.edgeTypeIdsByFetchChannel.size(), 2);
    EXPECT_EQ(queueDrainResult.edgeTypeIdsByFetchChannel[0], std::vector<int32_t>({0}));
    EXPECT_EQ(queueDrainResult.edgeTypeIdsByFetchChannel[1], std::vector<int32_t>({0}));

    ASSERT_NE(queueDrainResult.unionedNodeIdsByEdgeTypeId.find(0), queueDrainResult.unionedNodeIdsByEdgeTypeId.end());
    EXPECT_EQ(tensorValues(queueDrainResult.unionedNodeIdsByEdgeTypeId.at(0)), std::unordered_set<int64_t>({0, 1}));
}

TEST(PPRForwardPush, DrainTypedPPRChannelQueuesHonorsPerChannelFetchBudget) {
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});
    auto channel1 = makeState(/*seeds=*/{1}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});

    auto queueDrainResult = drainTypedPPRChannelQueues(std::vector<PPRForwardPush*>{&channel0, &channel1},
                                                       /*fetchIterationCounts=*/{1, 0},
                                                       /*maxFetchIterations=*/1);

    EXPECT_EQ(queueDrainResult.drainedChannelIndices, std::vector<int32_t>({0, 1}));
    EXPECT_EQ(queueDrainResult.fetchChannelIndices, std::vector<int32_t>({1}));
    ASSERT_EQ(queueDrainResult.edgeTypeIdsByFetchChannel.size(), 1);
    EXPECT_EQ(queueDrainResult.edgeTypeIdsByFetchChannel[0], std::vector<int32_t>({0}));

    ASSERT_NE(queueDrainResult.unionedNodeIdsByEdgeTypeId.find(0), queueDrainResult.unionedNodeIdsByEdgeTypeId.end());
    EXPECT_EQ(tensorValues(queueDrainResult.unionedNodeIdsByEdgeTypeId.at(0)), std::unordered_set<int64_t>({1}));
}

TEST(PPRForwardPush, ExtractTypedTopKWithResidualTopUpValidatesInputs) {
    auto channel = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1});

    EXPECT_THROW(extractTypedTopKWithResidualTopUp(std::vector<PPRForwardPush*>{}, std::vector<int32_t>{}, false),
                 c10::Error);
    EXPECT_THROW(
        extractTypedTopKWithResidualTopUp(std::vector<PPRForwardPush*>{&channel}, std::vector<int32_t>{}, false),
        c10::Error);
    EXPECT_THROW(
        extractTypedTopKWithResidualTopUp(std::vector<PPRForwardPush*>{&channel}, std::vector<int32_t>{-1}, false),
        c10::Error);
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
                                                    /*channelTargetCounts=*/{2, 1},
                                                    /*enableResidualTopUp=*/true);

    ASSERT_NE(result.find(0), result.end());
    const auto& [ids, features, counts, isExpanded] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(ids), std::vector<int64_t>({0, 10, 20}));
    EXPECT_EQ(tensorToInt64Vector(counts), std::vector<int64_t>({3}));
    ASSERT_EQ(features.size(0), 3);
    ASSERT_EQ(features.size(1), 8);

    expectTypedPPRFeatureRowNear(features,
                                 /*rowIndex=*/0,
                                 /*bestScore=*/0.5,
                                 /*hopProximity=*/1.0,
                                 /*channelScores=*/{0.5, 0.5},
                                 /*channelHopProximities=*/{1.0, 1.0},
                                 /*channelPresence=*/{1.0, 1.0});
    expectTypedPPRFeatureRowNear(features,
                                 /*rowIndex=*/1,
                                 /*bestScore=*/0.25,
                                 /*hopProximity=*/0.5,
                                 /*channelScores=*/{0.25, 0.0},
                                 /*channelHopProximities=*/{0.5, 0.0},
                                 /*channelPresence=*/{1.0, 0.0});
    expectTypedPPRFeatureRowNear(features,
                                 /*rowIndex=*/2,
                                 /*bestScore=*/0.25,
                                 /*hopProximity=*/0.5,
                                 /*channelScores=*/{0.0, 0.25},
                                 /*channelHopProximities=*/{0.0, 0.5},
                                 /*channelPresence=*/{0.0, 1.0});
}

TEST(PPRForwardPush, ExtractTypedTopKWithResidualTopUpUsesTargetsForResidualRows) {
    std::vector<int32_t> degrees(21, 1);
    auto channel0 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1.0, degrees);
    auto channel1 = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1.0, degrees);

    channel0.drainQueue();
    channel0.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{10}, /*counts=*/{1}));
    channel1.drainQueue();
    channel1.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{20}, /*counts=*/{1}));

    auto result = extractTypedTopKWithResidualTopUp(std::vector<PPRForwardPush*>{&channel0, &channel1},
                                                    /*channelTargetCounts=*/{1, 1},
                                                    /*enableResidualTopUp=*/true);

    ASSERT_NE(result.find(0), result.end());
    const auto& [ids, features, counts, isExpanded] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(ids), std::vector<int64_t>({0, 20}));
    EXPECT_EQ(tensorToInt64Vector(counts), std::vector<int64_t>({2}));
    ASSERT_EQ(features.size(0), 2);
    ASSERT_EQ(features.size(1), 8);

    expectTypedPPRFeatureRowNear(features,
                                 /*rowIndex=*/0,
                                 /*bestScore=*/0.5,
                                 /*hopProximity=*/1.0,
                                 /*channelScores=*/{0.5, 0.5},
                                 /*channelHopProximities=*/{1.0, 1.0},
                                 /*channelPresence=*/{1.0, 1.0});
    expectTypedPPRFeatureRowNear(features,
                                 /*rowIndex=*/1,
                                 /*bestScore=*/0.25,
                                 /*hopProximity=*/0.5,
                                 /*channelScores=*/{0.0, 0.25},
                                 /*channelHopProximities=*/{0.0, 0.5},
                                 /*channelPresence=*/{0.0, 1.0});
}

TEST(PPRForwardPush, ExtractTypedTopKWithResidualTopUpEmitsResidualAwareBaseRows) {
    auto channel = makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-9, /*degrees=*/{1, 1});

    channel.drainQueue();
    channel.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{0}, /*flatNeighborIds=*/{1}, /*counts=*/{1}));
    channel.drainQueue();
    channel.pushResiduals(makeFetched(/*edgeTypeId=*/0, /*nodeIds=*/{1}, /*flatNeighborIds=*/{0}, /*counts=*/{1}));

    auto result = extractTypedTopKWithResidualTopUp(std::vector<PPRForwardPush*>{&channel},
                                                    /*channelTargetCounts=*/{2},
                                                    /*enableResidualTopUp=*/true);

    ASSERT_NE(result.find(0), result.end());
    const auto& [ids, features, counts, isExpanded] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(ids), std::vector<int64_t>({0, 1}));
    EXPECT_EQ(tensorToInt64Vector(counts), std::vector<int64_t>({2}));
    ASSERT_EQ(features.size(0), 2);
    ASSERT_EQ(features.size(1), 5);

    expectTypedPPRFeatureRowNear(features,
                                 /*rowIndex=*/0,
                                 /*bestScore=*/0.625,
                                 /*hopProximity=*/1.0,
                                 /*channelScores=*/{0.625},
                                 /*channelHopProximities=*/{1.0},
                                 /*channelPresence=*/{1.0});
    expectTypedPPRFeatureRowNear(features,
                                 /*rowIndex=*/1,
                                 /*bestScore=*/0.25,
                                 /*hopProximity=*/0.5,
                                 /*channelScores=*/{0.25},
                                 /*channelHopProximities=*/{0.5},
                                 /*channelPresence=*/{1.0});
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
                                                    /*channelTargetCounts=*/{2, 1},
                                                    /*enableResidualTopUp=*/false);

    ASSERT_NE(result.find(0), result.end());
    const auto& [ids, features, counts, isExpanded] = result.at(0);
    EXPECT_EQ(tensorToInt64Vector(ids), std::vector<int64_t>({0}));
    EXPECT_EQ(tensorToInt64Vector(counts), std::vector<int64_t>({1}));
    ASSERT_EQ(features.size(0), 1);
    ASSERT_EQ(features.size(1), 8);
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
    const auto& [ids, weights, counts, isExpanded] = topk.at(0);
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
    EXPECT_EQ(topk1.at(0).validCounts[0].item<int64_t>(), 1);

    auto topk10 = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/false);
    ASSERT_NE(topk10.find(0), topk10.end());
    EXPECT_EQ(topk10.at(0).validCounts[0].item<int64_t>(), 2);
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
    EXPECT_EQ(topk.at(0).validCounts[0].item<int64_t>(), 1);
    EXPECT_EQ(topk.at(0).ids[0].item<int64_t>(), 0);

    auto topkWithResiduals = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/3, /*enableResidualTopUp=*/true);
    ASSERT_NE(topkWithResiduals.find(0), topkWithResiduals.end());
    const auto& [ids, weights, counts, isExpanded] = topkWithResiduals.at(0);
    ASSERT_EQ(counts[0].item<int64_t>(), 3);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    ASSERT_EQ(weights.size(1), 2);
    EXPECT_NEAR(weights[0][0].item<float>(), static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weights[0][1].item<float>(), 1.0F, 1e-5F);

    std::unordered_map<int64_t, float> residualWeights;
    residualWeights[ids[1].item<int64_t>()] = weights[1][0].item<float>();
    residualWeights[ids[2].item<int64_t>()] = weights[2][0].item<float>();
    ASSERT_NE(residualWeights.find(1), residualWeights.end());
    ASSERT_NE(residualWeights.find(2), residualWeights.end());
    EXPECT_NEAR(residualWeights[1], static_cast<float>((1.0 - alpha) * alpha / 2.0), 1e-5F);
    EXPECT_NEAR(residualWeights[2], static_cast<float>((1.0 - alpha) * alpha / 2.0), 1e-5F);

    // Node 0 was pushed (expanded); nodes 1 and 2 are unpushed residual top-ups.
    EXPECT_EQ(isExpanded[0].item<int64_t>(), 1);
    std::unordered_map<int64_t, int64_t> residualExpanded;
    residualExpanded[ids[1].item<int64_t>()] = isExpanded[1].item<int64_t>();
    residualExpanded[ids[2].item<int64_t>()] = isExpanded[2].item<int64_t>();
    EXPECT_EQ(residualExpanded[1], 0);
    EXPECT_EQ(residualExpanded[2], 0);
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
    const auto& [ids, weights, counts, isExpanded] = topkWithResiduals.at(0);
    ASSERT_EQ(counts[0].item<int64_t>(), 4);

    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    EXPECT_EQ(ids[3].item<int64_t>(), 3);
    for (int64_t index = 1; index < counts[0].item<int64_t>(); ++index) {
        EXPECT_GE(weights[index - 1][0].item<float>(), weights[index][0].item<float>());
    }

    std::unordered_map<int64_t, float> weightsByNodeId;
    std::unordered_map<int64_t, float> proximitiesByNodeId;
    for (int64_t index = 0; index < counts[0].item<int64_t>(); ++index) {
        weightsByNodeId[ids[index].item<int64_t>()] = weights[index][0].item<float>();
        proximitiesByNodeId[ids[index].item<int64_t>()] = weights[index][1].item<float>();
    }
    EXPECT_NEAR(weightsByNodeId[0], static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weightsByNodeId[1], static_cast<float>((1.0 - alpha) * alpha / 2.0), 1e-5F);
    EXPECT_NEAR(weightsByNodeId[2], static_cast<float>((1.0 - alpha) * alpha / 2.0), 1e-5F);
    EXPECT_NEAR(weightsByNodeId[3], static_cast<float>((1.0 - alpha) * (1.0 - alpha) * alpha / 2.0), 1e-5F);
    EXPECT_NEAR(proximitiesByNodeId[0], 1.0F, 1e-5F);
    EXPECT_NEAR(proximitiesByNodeId[1], 0.5F, 1e-5F);
    EXPECT_NEAR(proximitiesByNodeId[2], 0.5F, 1e-5F);
    EXPECT_NEAR(proximitiesByNodeId[3], 1.0F / 3.0F, 1e-5F);
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
    const auto& [ids, weights, counts, isExpanded] = cappedTopk.at(0);

    ASSERT_EQ(counts[0].item<int64_t>(), 2);
    EXPECT_EQ(ids[0].item<int64_t>(), 0);
    ASSERT_EQ(weights.size(1), 2);
    EXPECT_NEAR(weights[0][0].item<float>(), static_cast<float>(alpha), 1e-5F);
    EXPECT_NEAR(weights[0][1].item<float>(), 1.0F, 1e-5F);
}

TEST(PPRForwardPush, ExtractTopKWithResidualTopUpRejectsNegativeMaxPPRNodes) {
    auto state = makeState(/*seeds=*/{0}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{0});
    EXPECT_THROW(state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/-1, /*enableResidualTopUp=*/true), c10::Error);
}

// edge type id -> source node id -> neighbor ids.
using Adjacency = std::map<int32_t, std::map<int64_t, std::vector<int64_t>>>;

// Runs drainQueue / pushResiduals to convergence, serving neighbor lookups from
// the given adjacency. Nodes missing from the adjacency have no neighbors.
static void runToConvergence(PPRForwardPush& state, const Adjacency& adjacency) {
    for (int iteration = 0; iteration < 10000; ++iteration) {
        auto toFetch = state.drainQueue();
        if (!toFetch.has_value()) {
            return;
        }
        NeighborFetchMap fetched;
        for (const auto& [edgeTypeId, nodeIdsTensor] : *toFetch) {
            std::vector<int64_t> nodeIds = tensorToInt64Vector(nodeIdsTensor);
            std::vector<int64_t> flatNeighborIds;
            std::vector<int64_t> counts;
            for (int64_t nodeId : nodeIds) {
                std::vector<int64_t> neighbors;
                auto edgeTypeIter = adjacency.find(edgeTypeId);
                if (edgeTypeIter != adjacency.end() && edgeTypeIter->second.count(nodeId) > 0) {
                    neighbors = edgeTypeIter->second.at(nodeId);
                }
                flatNeighborIds.insert(flatNeighborIds.end(), neighbors.begin(), neighbors.end());
                counts.push_back(static_cast<int64_t>(neighbors.size()));
            }
            fetched.merge(makeFetched(edgeTypeId, nodeIds, flatNeighborIds, counts));
        }
        state.pushResiduals(fetched);
    }
    FAIL() << "PPR did not converge";
}

// Single-seed scores for one node type, keyed by node id.
static std::map<int64_t, double> scoresByNode(PPRForwardPush& state,
                                              int32_t nodeTypeId,
                                              int32_t maxPPRNodes,
                                              bool enableResidualTopUp) {
    auto topk = state.extractTopKWithResidualTopUp(maxPPRNodes, enableResidualTopUp);
    std::map<int64_t, double> scores;
    auto typeIter = topk.find(nodeTypeId);
    if (typeIter == topk.end()) {
        return scores;
    }
    auto ids = tensorToInt64Vector(typeIter->second.ids);
    auto weights = typeIter->second.weights.accessor<double, 2>();
    for (size_t row = 0; row < ids.size(); ++row) {
        scores[ids[row]] = weights[row][0];
    }
    return scores;
}

// With alpha the restart probability, sum(ppr) + sum(residual) / alpha equals 1
// for a seed whose reachable nodes all have outgoing edges.
static double pprMassInvariant(PPRForwardPush& state, double alpha) {
    double pprMass = 0.0;
    for (const auto& [nodeId, score] : scoresByNode(state, 0, 1000000, false)) {
        pprMass += score;
    }
    double totalMass = 0.0;
    for (const auto& [nodeId, score] : scoresByNode(state, 0, 1000000, true)) {
        totalMass += score;
    }
    return pprMass + (totalMass - pprMass) / alpha;
}

static PPRForwardPush makeStateWithDegrees(const torch::Tensor& degrees) {
    return PPRForwardPush(torch::tensor({0}, torch::kLong),
                          /*seedNodeTypeId=*/0,
                          /*alpha=*/0.5,
                          /*requeueThresholdFactor=*/1e-6,
                          /*nodeTypeToEdgeTypeIds=*/{{0}},
                          /*edgeTypeToDstNtypeId=*/{0},
                          {degrees});
}

TEST(PPRForwardPush, Int64DegreeTensorMatchesInt32) {
    const Adjacency adjacency = {{0, {{0, {1, 2}}, {1, {2}}}}};
    auto int32State = makeStateWithDegrees(torch::tensor({2, 1, 0}, torch::kInt));
    auto int64State = makeStateWithDegrees(torch::tensor({2, 1, 0}, torch::kLong));
    runToConvergence(int32State, adjacency);
    runToConvergence(int64State, adjacency);
    EXPECT_EQ(scoresByNode(int32State, 0, 10, true), scoresByNode(int64State, 0, 10, true));
}

// A node type without degree data has degree 0, so every discovered node is
// queued and then absorbs its residual.
TEST(PPRForwardPush, EmptyDegreeTensorQueuesEveryDiscoveredNode) {
    const double alpha = 0.5;
    auto state = makeStateWithDegrees(torch::empty({0}, torch::kInt));
    runToConvergence(state, {{0, {{0, {1}}}}});
    auto scores = scoresByNode(state, 0, 10, false);
    ASSERT_EQ(scores.size(), 2U);
    EXPECT_NEAR(scores[0], alpha, 1e-12);
    EXPECT_NEAR(scores[1], (1.0 - alpha) * alpha, 1e-12);
}

// Source type 0 reaches destination type 1 through two edge types. The residual
// is split by total neighbor count across both edge types.
TEST(PPRForwardPush, HeterogeneousDestinationReachedThroughTwoEdgeTypes) {
    const double alpha = 0.5;
    PPRForwardPush state(torch::tensor({0}, torch::kLong),
                         /*seedNodeTypeId=*/0,
                         alpha,
                         /*requeueThresholdFactor=*/1e-6,
                         /*nodeTypeToEdgeTypeIds=*/{{0, 1}, {}},
                         /*edgeTypeToDstNtypeId=*/{1, 1},
                         {torch::tensor({1}, torch::kInt), torch::empty({0}, torch::kInt)});
    runToConvergence(state, {{0, {{0, {5}}}}, {1, {{0, {5, 6}}}}});

    auto sourceScores = scoresByNode(state, 0, 10, false);
    ASSERT_EQ(sourceScores.size(), 1U);
    EXPECT_NEAR(sourceScores[0], alpha, 1e-12);

    auto destinationScores = scoresByNode(state, 1, 10, false);
    ASSERT_EQ(destinationScores.size(), 2U);
    const double pushed = (1.0 - alpha) * alpha;
    EXPECT_NEAR(destinationScores[5], pushed * 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(destinationScores[6], pushed / 3.0, 1e-12);
}

// A self-loop re-queues the node being pushed.
TEST(PPRForwardPush, SelfLoopRequeuesDrainedNode) {
    const double alpha = 0.5;
    auto state = makeStateWithDegrees(torch::tensor({1}, torch::kInt));
    runToConvergence(state, {{0, {{0, {0}}}}});
    EXPECT_NEAR(pprMassInvariant(state, alpha), 1.0, 1e-9);
    EXPECT_NEAR(scoresByNode(state, 0, 10, false).at(0), 1.0, 1e-5);
}

// Nodes 0 and 1 are both in one drain snapshot and each re-queues both.
TEST(PPRForwardPush, NodesInSameSnapshotRequeueEachOther) {
    const double alpha = 0.5;
    auto state = makeStateWithDegrees(torch::tensor({2, 2}, torch::kInt));
    runToConvergence(state, {{0, {{0, {0, 1}}, {1, {0, 1}}}}});
    EXPECT_NEAR(pprMassInvariant(state, alpha), 1.0, 1e-9);
    auto scores = scoresByNode(state, 0, 10, false);
    EXPECT_NEAR(scores[0] + scores[1], 1.0, 1e-5);
}

// A star with thousands of leaves grows the per-seed tables well past their
// initial size.
TEST(PPRForwardPush, LargeTraversalConservesMass) {
    const double alpha = 0.15;
    const int64_t numLeaves = 5000;
    std::vector<int64_t> leaves;
    std::vector<int32_t> degrees(numLeaves + 1, 1);
    degrees[0] = static_cast<int32_t>(numLeaves);
    Adjacency adjacency;
    for (int64_t leaf = 1; leaf <= numLeaves; ++leaf) {
        leaves.push_back(leaf);
        adjacency[0][leaf] = {0};
    }
    adjacency[0][0] = leaves;
    auto state = makeState(/*seeds=*/{0}, alpha, /*requeueThresholdFactor=*/1e-6, degrees);
    runToConvergence(state, adjacency);
    EXPECT_NEAR(pprMassInvariant(state, alpha), 1.0, 1e-9);
    EXPECT_EQ(scoresByNode(state, 0, 100000, true).size(), static_cast<size_t>(numLeaves + 1));
}

// Equal scores are ordered by ascending node id, so the cutoff keeps the lowest ids.
TEST(PPRForwardPush, TopKBreaksScoreTiesByNodeId) {
    auto state =
        makeState(/*seeds=*/{0}, /*alpha=*/0.5, /*requeueThresholdFactor=*/1e-6, /*degrees=*/{3, 0, 0, 0, 0, 0, 0, 0});
    runToConvergence(state, {{0, {{0, {7, 3, 5}}}}});
    auto topk = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/3, /*enableResidualTopUp=*/false);
    EXPECT_EQ(tensorToInt64Vector(topk.at(0).ids), std::vector<int64_t>({0, 3, 5}));
}

// Reordering neighbor lists does not change the output.
TEST(PPRForwardPush, OutputIsIndependentOfNeighborOrder) {
    const std::vector<int32_t> degrees = {3, 3, 2, 2, 1, 1};
    const Adjacency adjacency = {{0, {{0, {1, 2, 3}}, {1, {0, 3, 4}}, {2, {1, 5}}, {3, {0, 4}}, {4, {5}}, {5, {2}}}}};
    Adjacency reversed = adjacency;
    for (auto& [nodeId, neighbors] : reversed[0]) {
        std::reverse(neighbors.begin(), neighbors.end());
    }
    auto state = makeState(/*seeds=*/{0, 1}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-4, degrees);
    auto reversedState = makeState(/*seeds=*/{0, 1}, /*alpha=*/0.15, /*requeueThresholdFactor=*/1e-4, degrees);
    runToConvergence(state, adjacency);
    runToConvergence(reversedState, reversed);

    auto topk = state.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/true);
    auto reversedTopk = reversedState.extractTopKWithResidualTopUp(/*maxPPRNodes=*/10, /*enableResidualTopUp=*/true);
    EXPECT_TRUE(torch::equal(topk.at(0).ids, reversedTopk.at(0).ids));
    EXPECT_TRUE(torch::equal(topk.at(0).weights, reversedTopk.at(0).weights));
}

// Degree tensors must be int32 or int64; any other dtype is rejected at construction.
TEST(PPRForwardPush, ConstructorRejectsUnsupportedDegreeDtype) {
    EXPECT_THROW(PPRForwardPush(torch::tensor(std::vector<int64_t>{0}, torch::kLong),
                                /*seedNodeTypeId=*/0,
                                /*alpha=*/0.15,
                                /*requeueThresholdFactor=*/1e-6,
                                /*nodeTypeToEdgeTypeIds=*/{{0}},
                                /*edgeTypeToDstNtypeId=*/{0},
                                {torch::tensor(std::vector<float>{1.0F}, torch::kFloat)}),
                 c10::Error);
}
