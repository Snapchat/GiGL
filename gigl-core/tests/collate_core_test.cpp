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

TEST(CollateHeterogeneous, EdgeDirInKeysUnderProvidedRevEtype) {
    using Arr = std::array<std::string, 3>;
    std::unordered_map<std::string, torch::Tensor> msg;
    msg["u.ids"] = torch::tensor({1, 2}, torch::kInt64);
    msg["i.ids"] = torch::tensor({3}, torch::kInt64);
    // For edge_dir=="in", GLT keys the message under reverse_edge_type(etype) string
    // and maps it back to `etype`. We pass that mapping directly.
    msg["i__rev_to__u.rows"] = torch::tensor({0}, torch::kInt64);
    msg["i__rev_to__u.cols"] = torch::tensor({0}, torch::kInt64);
    std::vector<std::string> nodeTypes = {"u", "i"};
    std::vector<std::pair<std::string, Arr>> edgeTypeStrToRev = {
        {"i__rev_to__u", Arr{"u", "to", "i"}},  // maps back to the original outward type
    };
    std::vector<Arr> reversedEdgeTypes = {Arr{"u", "to", "i"}};
    auto res = gigl::collation::collateHeterogeneous(
        msg, nodeTypes, edgeTypeStrToRev, reversedEdgeTypes, "u", false, /*batchSize=*/0);
    auto key = Arr{"u", "to", "i"};
    auto expected = torch::stack({msg["i__rev_to__u.cols"], msg["i__rev_to__u.rows"]});
    EXPECT_TRUE(torch::equal(res.edgeIndex.at(key), expected));
}

TEST(CollateHeterogeneous, BatchFallbackSlicesAnchorIdsWhenBatchKeyAbsent) {
    using Arr = std::array<std::string, 3>;
    // NODE sampling produced no ".batch" key (GLT writes it only when output.batch is
    // not None, dist_neighbor_sampler.py:781-783). The kernel must fall back to
    // node[inputType][:batchSize] (dist_loader.py:397-399).
    std::unordered_map<std::string, torch::Tensor> msg;
    msg["u.ids"] = torch::tensor({100, 101, 102, 103}, torch::kInt64);
    msg["i.ids"] = torch::tensor({200}, torch::kInt64);
    std::vector<std::string> nodeTypes = {"u", "i"};
    std::vector<std::pair<std::string, Arr>> edgeTypeStrToRev = {};
    std::vector<Arr> reversedEdgeTypes = {};
    auto res = gigl::collation::collateHeterogeneous(
        msg, nodeTypes, edgeTypeStrToRev, reversedEdgeTypes,
        /*inputType=*/"u", /*hasBatch=*/true, /*batchSize=*/2);
    // batch == first 2 anchor ids.
    EXPECT_TRUE(torch::equal(res.batch.at("u"), torch::tensor({100, 101}, torch::kInt64)));
}

TEST(CollateHeterogeneous, BuildsDictsSwapsEdgesPadsAndFillsEmpty) {
    using Arr = std::array<std::string, 3>;
    // Node types "u","i"; one sampled edge type str "u__to__i" reversed to ("i","rev_to","u")
    // (edge_dir=="out" convention: store under reverse_edge_type). A second reversed edge type
    // ("u","rev_to2","i") has no sampled edges and must be filled with [2,0] + zero counts.
    std::unordered_map<std::string, torch::Tensor> msg;
    msg["u.ids"] = torch::tensor({100, 101}, torch::kInt64);
    msg["i.ids"] = torch::tensor({200, 201, 202}, torch::kInt64);
    msg["u.num_sampled_nodes"] = torch::tensor({2}, torch::kInt64);
    msg["i.num_sampled_nodes"] = torch::tensor({3}, torch::kInt64);
    msg["u__to__i.rows"] = torch::tensor({0, 1}, torch::kInt64);   // src local ids (u)
    msg["u__to__i.cols"] = torch::tensor({0, 2}, torch::kInt64);   // dst local ids (i)
    msg["u__to__i.num_sampled_edges"] = torch::tensor({2}, torch::kInt64);

    std::vector<std::string> nodeTypes = {"u", "i"};
    std::vector<std::pair<std::string, Arr>> edgeTypeStrToRev = {
        {"u__to__i", Arr{"i", "rev_to", "u"}},
    };
    std::vector<Arr> reversedEdgeTypes = {Arr{"i", "rev_to", "u"}, Arr{"u", "rev_to2", "i"}};

    auto res = gigl::collation::collateHeterogeneous(
        msg, nodeTypes, edgeTypeStrToRev, reversedEdgeTypes,
        /*inputType=*/"u", /*hasBatch=*/false, /*batchSize=*/0);

    // Edge index for the sampled type: row=cols (swap), col=rows -> stack([cols, rows]).
    auto sampled = Arr{"i", "rev_to", "u"};
    auto absent = Arr{"u", "rev_to2", "i"};
    auto expectedSampled = torch::stack({msg["u__to__i.cols"], msg["u__to__i.rows"]});
    EXPECT_TRUE(torch::equal(res.edgeIndex.at(sampled), expectedSampled));
    // Absent edge type filled with [2,0].
    EXPECT_EQ(res.edgeIndex.at(absent).size(0), 2);
    EXPECT_EQ(res.edgeIndex.at(absent).size(1), 0);
    // num_sampled_edges: num_hops = max len = 1; sampled stays [2]-> [2]; absent -> [0].
    EXPECT_TRUE(torch::equal(res.numSampledEdges.at(sampled), torch::tensor({2}, torch::kInt64)));
    EXPECT_TRUE(torch::equal(res.numSampledEdges.at(absent), torch::tensor({0}, torch::kInt64)));
    // num_sampled_nodes padded to num_hops+1 == 2.
    EXPECT_TRUE(torch::equal(res.numSampledNodes.at("u"), torch::tensor({2, 0}, torch::kInt64)));
    EXPECT_TRUE(torch::equal(res.numSampledNodes.at("i"), torch::tensor({3, 0}, torch::kInt64)));
    // Nodes.
    EXPECT_TRUE(torch::equal(res.node.at("u"), msg["u.ids"]));
    EXPECT_TRUE(torch::equal(res.node.at("i"), msg["i.ids"]));
}

}  // namespace
