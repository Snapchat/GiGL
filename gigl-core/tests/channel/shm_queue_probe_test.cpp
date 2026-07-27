// Tests for gigl::ShmQueueProbe against a real graphlearn_torch::ShmQueue.
//
// These drive an actual queue (GLT's csrc/shm_queue.cc is linked in via the glt_shm_queue target)
// rather than asserting on a hand-built ShmQueueMeta. Hand-building one would require either reusing
// the probe's own private-access mechanism -- making the test tautological -- or re-hard-coding the
// byte offsets the probe exists to avoid.

#include <gtest/gtest.h>

#include "graphlearn_torch/include/shm_queue.h"

#include "channel/shm_queue_probe.h"

#include <sys/shm.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <vector>

namespace gigl {
namespace {

// Small enough to keep the segment cheap, large enough to exercise multiple blocks.
constexpr size_t kMaxBlockNum = 8;
constexpr size_t kMaxBufSize = 64 * 1024;

// Enqueues a distinctly-sized payload so successive messages are not accidentally interchangeable.
void enqueuePayload(graphlearn_torch::ShmQueue& queue, size_t sizeBytes) {
    const std::vector<uint8_t> payload(sizeBytes, static_cast<uint8_t>(sizeBytes & 0xFFu));
    queue.Enqueue(payload.data(), payload.size());
}

TEST(ShmQueueProbeTest, ReportsZeroForFreshQueue) {
    graphlearn_torch::ShmQueue queue(kMaxBlockNum, kMaxBufSize);
    const ShmQueueProbe probe(queue.ShmId());

    EXPECT_EQ(probe.queueSize(), 0u);
    EXPECT_TRUE(queue.Empty());
}

TEST(ShmQueueProbeTest, TracksEnqueueAndDequeue) {
    graphlearn_torch::ShmQueue queue(kMaxBlockNum, kMaxBufSize);
    const ShmQueueProbe probe(queue.ShmId());

    enqueuePayload(queue, 16);
    EXPECT_EQ(probe.queueSize(), 1u);
    EXPECT_FALSE(queue.Empty());

    enqueuePayload(queue, 32);
    enqueuePayload(queue, 48);
    EXPECT_EQ(probe.queueSize(), 3u);

    {
        // ShmData releases its block on destruction, so scope each dequeue.
        graphlearn_torch::ShmData first = queue.Dequeue();
        EXPECT_EQ(first.Length(), 16u);
    }
    EXPECT_EQ(probe.queueSize(), 2u);

    {
        graphlearn_torch::ShmData second = queue.Dequeue();
        EXPECT_EQ(second.Length(), 32u);
    }
    {
        graphlearn_torch::ShmData third = queue.Dequeue();
        EXPECT_EQ(third.Length(), 48u);
    }
    EXPECT_EQ(probe.queueSize(), 0u);
    EXPECT_TRUE(queue.Empty());
}

TEST(ShmQueueProbeTest, ExposesTheShmidItAttachedTo) {
    graphlearn_torch::ShmQueue queue(kMaxBlockNum, kMaxBufSize);
    const ShmQueueProbe probe(queue.ShmId());

    EXPECT_EQ(probe.shmId(), queue.ShmId());
}

// A probe must not disturb the segment it observes: it attaches read-only and detaches without
// shmctl(IPC_RMID). If it ever destroyed the segment, a probe created afterwards would fail to attach.
TEST(ShmQueueProbeTest, DestructionLeavesTheSegmentUsable) {
    graphlearn_torch::ShmQueue queue(kMaxBlockNum, kMaxBufSize);
    enqueuePayload(queue, 24);

    {
        const ShmQueueProbe firstProbe(queue.ShmId());
        EXPECT_EQ(firstProbe.queueSize(), 1u);
    } // firstProbe detaches here

    const ShmQueueProbe secondProbe(queue.ShmId());
    EXPECT_EQ(secondProbe.queueSize(), 1u);
}

// Multiple live probes on one queue are independent attachments observing the same state.
TEST(ShmQueueProbeTest, ConcurrentProbesAgree) {
    graphlearn_torch::ShmQueue queue(kMaxBlockNum, kMaxBufSize);
    const ShmQueueProbe probeA(queue.ShmId());
    const ShmQueueProbe probeB(queue.ShmId());

    enqueuePayload(queue, 8);
    enqueuePayload(queue, 8);

    EXPECT_EQ(probeA.queueSize(), 2u);
    EXPECT_EQ(probeB.queueSize(), 2u);
}

// An unusable shmid must surface as an exception, not a segfault. shmat returns (void*)-1 on failure;
// storing that sentinel would make queueSize() dereference address -1 and the destructor pass it to
// shmdt. -1 is never a valid shmid.
TEST(ShmQueueProbeTest, ThrowsOnInvalidShmid) {
    EXPECT_THROW(ShmQueueProbe(-1), std::system_error);
}

TEST(ShmQueueProbeTest, InvalidShmidErrorMentionsTheShmid) {
    try {
        ShmQueueProbe probe(-1);
        FAIL() << "expected ShmQueueProbe(-1) to throw";
    } catch (const std::system_error& error) {
        // The shmid is load-bearing in this message: it is the only handle an operator has for
        // correlating the failure with a channel.
        EXPECT_NE(std::string(error.what()).find("-1"), std::string::npos);
    }
}

// GLT's Dequeue checks `read >= write` without a lock before bumping read_block_id_ with
// __sync_fetch_and_add, so concurrent consumers can push read past write. Subtracting size_t values in
// that window wraps to ~1.8e19 rather than going negative, so the probe clamps.
//
// Reproducing the race directly would be flaky, so the counters are forced into the overshoot state.
// This is the one place a test reaches into GLT's private state, using the same explicit-instantiation
// mechanism as the probe (legal: access checks are not applied inside explicit instantiations).
namespace force_overshoot {

using MetaField = size_t graphlearn_torch::ShmQueueMeta::*;

template<class Tag, MetaField member>
struct StealField {
    friend MetaField gltField(Tag /*unused*/) {
        return member;
    }
};

struct ReadBlockIdTag {
    friend MetaField gltField(ReadBlockIdTag /*unused*/);
};

template struct StealField<ReadBlockIdTag, &graphlearn_torch::ShmQueueMeta::read_block_id_>;

} // namespace force_overshoot

TEST(ShmQueueProbeTest, ClampsWhenReadCounterOvershootsWriteCounter) {
    graphlearn_torch::ShmQueue queue(kMaxBlockNum, kMaxBufSize);
    enqueuePayload(queue, 8);

    const ShmQueueProbe probe(queue.ShmId());
    ASSERT_EQ(probe.queueSize(), 1u);

    // Attach read-write so the counter can be driven into the overshoot state the probe must tolerate.
    void* attached = shmat(queue.ShmId(), nullptr, 0);
    ASSERT_NE(attached, reinterpret_cast<void*>(-1));
    auto* meta = reinterpret_cast<graphlearn_torch::ShmQueueMeta*>(attached);

    // read = write + 1, i.e. one more dequeue than enqueue was observed.
    // gltField is a friend defined inside a class template, so it is reachable only via
    // argument-dependent lookup -- it must be called unqualified, with the tag supplying the namespace.
    using force_overshoot::ReadBlockIdTag;
    meta->*gltField(ReadBlockIdTag{}) = 2;
    EXPECT_EQ(probe.queueSize(), 0u) << "unsigned subtraction wrapped instead of clamping";

    shmdt(attached);
}

} // namespace
} // namespace gigl
