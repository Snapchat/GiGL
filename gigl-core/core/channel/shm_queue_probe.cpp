#include "shm_queue_probe.h"

#include <sys/shm.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>

namespace gigl {

namespace {

using MetaField = size_t graphlearn_torch::ShmQueueMeta::*;

// GLT's enqueue/dequeue counters are private, with `friend class ShmQueue` only -- friendship is not
// inherited, so subclassing buys nothing, and `ShmQueue::meta_` is private too.
//
// Access control is not applied inside an explicit template instantiation ([temp.spec]/6), so naming
// those private members in the instantiations below is legal standard C++ -- not `#define private
// public`, not undefined behaviour. The private member may be named ONLY inside the instantiation:
// writing `&ShmQueueMeta::write_block_id_` at a call site is an ordinary expression, where access
// control does apply, and will not compile.
//
// The payoff over reading raw byte offsets (which is what this code replaced) is that the offsets come
// from the compiler reading GLT's header. There is no hard-coded 32 or 40 here, so a field inserted
// upstream ahead of these counters shifts our reads automatically instead of silently corrupting them.
// gigl-core/tests/glt_headers_test.cpp additionally pins the expected layout so such a move is loud.
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

} // namespace

ShmQueueProbe::ShmQueueProbe(int shmid) : _shmId(shmid) {
    // SHM_RDONLY: this probe only ever reads. GLT creates the segment with mode 0666
    // (shmget(IPC_PRIVATE, ..., 0666 | IPC_CREAT | IPC_EXCL)), so read permission is granted.
    void* attached = shmat(shmid, nullptr, SHM_RDONLY);
    // POSIX specifies shmat's failure return as (void*)-1. Comparing via intptr_t rather than casting
    // -1 up to a pointer keeps clang-tidy's performance-no-int-to-ptr happy: an integer-to-pointer cast
    // defeats the compiler's pointer provenance tracking, whereas pointer-to-integer does not.
    if (reinterpret_cast<std::intptr_t>(attached) == -1) {
        // Capture errno before anything else can clobber it.
        const int attachErrno = errno;
        throw std::system_error(
            attachErrno, std::generic_category(), "shmat failed for shmid=" + std::to_string(shmid));
    }
    // ShmQueueMeta sits at offset 0 of the segment; GLT's own ShmQueue constructors cast the shmat
    // result directly the same way.
    _meta = reinterpret_cast<graphlearn_torch::ShmQueueMeta*>(attached);
}

ShmQueueProbe::~ShmQueueProbe() {
    if (_meta != nullptr) {
        // Detach only. Never shmctl(IPC_RMID) -- the queue's owner controls the segment's lifetime,
        // and destroying it here would break every other process attached to it.
        shmdt(_meta);
    }
}

size_t ShmQueueProbe::queueSize() const {
    // Relaxed atomic loads rather than plain ones. Other processes mutate these words concurrently:
    // write_block_id_ via a plain `++` under a process-shared semaphore we do not hold
    // (shm_queue.cc:85-86), read_block_id_ via __sync_fetch_and_add (:102).
    //
    // To be precise about what this does and does not buy: because GLT's write side is a *non-atomic*
    // store, this pairing is still formally a data race under the C++ memory model, and using an
    // atomic load does not make it race-free. What it does is stop the compiler from eliding or
    // splitting our side of it, and document the intent. On the platforms gigl-core ships for
    // (x86-64/aarch64, naturally-aligned 8-byte words) it compiles to the same load a plain read would
    // and cannot observe a torn value.
    //
    // That is acceptable only because the result is a monitoring gauge: a stale-by-microseconds depth
    // is indistinguishable from a fresh one for this purpose. Do not build control flow on it. Making
    // this genuinely race-free requires a change on GLT's side -- either atomic counters or a public
    // accessor that takes the queue's semaphore -- which is tracked as the upstream `Size()` work.
    const size_t written = __atomic_load_n(&(_meta->*gltField(WriteBlockIdTag{})), __ATOMIC_RELAXED);
    const size_t read = __atomic_load_n(&(_meta->*gltField(ReadBlockIdTag{})), __ATOMIC_RELAXED);

    // `read` can transiently exceed `written`: GLT's Dequeue checks `read >= write` without holding a
    // lock, then bumps read_block_id_ with __sync_fetch_and_add, so concurrent consumers can overshoot.
    // Subtracting unsigned values in that window would wrap to ~1.8e19 instead of underflowing to a
    // small negative, so clamp rather than trusting the ordering.
    return written > read ? written - read : 0;
}

} // namespace gigl
