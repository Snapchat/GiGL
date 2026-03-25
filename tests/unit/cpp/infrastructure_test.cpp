// Placeholder C++ unit test.
//
// This file exists to verify that the GoogleTest infrastructure compiles and
// runs end-to-end.  Replace or supplement it with tests for actual GiGL C++
// code (e.g. PPRForwardPushState) as those components are added.

#include <gtest/gtest.h>

// A trivial sanity-check test — if this fails, something is very wrong with
// the build environment itself.
TEST(PlaceholderTest, BasicArithmetic) {
    EXPECT_EQ(1 + 1, 2);
    EXPECT_NE(1 + 1, 3);
}
