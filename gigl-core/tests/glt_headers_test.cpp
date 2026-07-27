// Drift alarm for the GraphLearn-for-PyTorch (GLT) headers gigl-core compiles against.
//
// Two things are asserted here, and they fail for different reasons:
//
//  1. That GLT's headers are reachable at all. This is the only compilation unit that exists purely
//     to exercise the `glt_headers` CMake target, the GLT_PIN.cmake acquisition, and the include
//     path. Without it, breaking the build plumbing would go unnoticed until something else happened
//     to need a GLT header.
//
//  2. That `graphlearn_torch::ShmQueueMeta`'s layout is what gigl-core's ShmQueueProbe assumes.
//     The probe reads that struct out of a shared-memory segment created by the *installed GLT
//     wheel*. If the pinned headers and the installed wheel ever describe different layouts, queue
//     sizes come back silently wrong -- no crash, no exception, just bad metrics. The offsets below
//     turn that into a build failure.
//
// The offsets are obtained the same way the probe obtains them: from pointers-to-member the compiler
// derives from GLT's header. Nothing here hard-codes a byte offset as an *input* -- 32 and 40 are
// assertions about what the compiler computed, which is exactly the check we want.

#include <gtest/gtest.h>

#include "graphlearn_torch/include/shm_queue.h"

#include <cstddef>

namespace {

using MetaField = size_t graphlearn_torch::ShmQueueMeta::*;

// Access control is not applied inside an explicit template instantiation ([temp.spec]/6), so naming
// GLT's private members in the instantiations below is legal -- this is not `#define private public`
// and not undefined behaviour. The private member must be named ONLY there: spelling
// `&ShmQueueMeta::write_block_id_` at a call site is an ordinary expression, where access control
// does apply, and will not compile.
template<class Tag, MetaField member>
struct StealField {
    friend MetaField gltField(Tag /*unused*/) {
        return member;
    }
};

struct WriteBlockIdTag {
    friend MetaField gltField(WriteBlockIdTag /*unused*/);
};
struct ReadBlockIdTag {
    friend MetaField gltField(ReadBlockIdTag /*unused*/);
};

template struct StealField<WriteBlockIdTag, &graphlearn_torch::ShmQueueMeta::write_block_id_>;
template struct StealField<ReadBlockIdTag, &graphlearn_torch::ShmQueueMeta::read_block_id_>;

// Byte offset of a size_t member within ShmQueueMeta, computed from a pointer-to-member.
std::ptrdiff_t fieldOffset(MetaField member) {
    graphlearn_torch::ShmQueueMeta meta{};
    const auto* base = reinterpret_cast<const char*>(&meta);
    return reinterpret_cast<const char*>(&(meta.*member)) - base;
}

// ShmQueueMeta is POD (size_t and sem_t members only) with no #ifdef WITH_CUDA in its header, so its
// layout does not vary with our compile flags. A change here means the pinned GLT commit moved.
//
// Linux-only: the total size folds in sizeof(sem_t), which is platform ABI rather than anything the
// pinned GLT header controls (32 bytes on glibc, 4 on Darwin). Asserting it unconditionally would make
// a macOS test run fail for a portability difference while reporting it as header drift. The counter
// offsets below are portable and are the part the probe actually depends on.
#ifdef __linux__
TEST(GltHeadersTest, ShmQueueMetaSizeMatchesPin) {
    EXPECT_EQ(sizeof(graphlearn_torch::ShmQueueMeta), 128u);
}
#endif

TEST(GltHeadersTest, QueueCounterOffsetsMatchPin) {
    EXPECT_EQ(fieldOffset(gltField(WriteBlockIdTag{})), 32);
    EXPECT_EQ(fieldOffset(gltField(ReadBlockIdTag{})), 40);
}

// The counters must be adjacent and ordered write-then-read. ShmQueueProbe subtracts one from the
// other, so a swap would invert the sign and the clamp would mask it as a permanent zero.
TEST(GltHeadersTest, WriteCounterPrecedesReadCounter) {
    const std::ptrdiff_t writeOffset = fieldOffset(gltField(WriteBlockIdTag{}));
    const std::ptrdiff_t readOffset = fieldOffset(gltField(ReadBlockIdTag{}));
    EXPECT_LT(writeOffset, readOffset);
    EXPECT_EQ(readOffset - writeOffset, static_cast<std::ptrdiff_t>(sizeof(size_t)));
}

} // namespace
