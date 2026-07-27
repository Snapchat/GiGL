#ifndef GIGL_CORE_CHANNEL_SHM_QUEUE_PROBE_H_
#define GIGL_CORE_CHANNEL_SHM_QUEUE_PROBE_H_

#include "graphlearn_torch/include/shm_queue.h"

#include <cstddef>

namespace gigl {

/// Read-only view onto the metadata block of a GLT ``ShmQueue``, addressed by its System V shmid.
///
/// GraphLearn-for-PyTorch exposes no way to ask a shared-memory channel how many messages it is
/// holding: ``ShmQueue::Empty()`` is a bool, and the enqueue/dequeue counters that would answer the
/// question are private. This class reads those counters directly out of the segment, using field
/// offsets the compiler derives from GLT's own header rather than hard-coded constants.
///
/// Attaches once on construction and detaches on destruction, so ``queueSize()`` is two loads --
/// cheap enough to call on the per-batch monitoring path. Attaches read-only and never issues
/// ``shmctl(IPC_RMID)``, so a probe cannot disturb or destroy a queue it does not own.
///
/// The returned depth is an instantaneous approximation. Producers and consumers mutate the counters
/// concurrently and this probe holds none of GLT's locks, so treat the value as a metric, not as a
/// basis for control flow.
///
/// Not copyable, and not safe to share across a fork: a shared-memory attachment belongs to the
/// process that made it. Construct one per process from the same shmid instead.
///
/// Example:
///     >>> from gigl_core import ShmQueueProbe
///     >>> probe = ShmQueueProbe(sample_queue.__getstate__())  # __getstate__ returns the shmid
///     >>> probe.qsize()
///     0
class ShmQueueProbe {
public:
    /// Attaches to an existing GLT ``ShmQueue`` segment.
    ///
    /// \param shmid System V shared-memory id of a live GLT ``ShmQueue``. Obtainable in Python via
    ///     ``SampleQueue.__getstate__()``, which GLT defines to return the raw shmid.
    /// \throws std::system_error if the segment cannot be attached (bad or stale shmid, permissions).
    explicit ShmQueueProbe(int shmid);

    ~ShmQueueProbe();

    ShmQueueProbe(const ShmQueueProbe&) = delete;
    ShmQueueProbe& operator=(const ShmQueueProbe&) = delete;
    ShmQueueProbe(ShmQueueProbe&&) = delete;
    ShmQueueProbe& operator=(ShmQueueProbe&&) = delete;

    /// Messages currently enqueued: total enqueued minus total dequeued, clamped at 0.
    ///
    /// \return Approximate queue depth.
    size_t queueSize() const;

    /// The shmid this probe is attached to.
    int shmId() const {
        return _shmId;
    }

private:
    int _shmId;
    graphlearn_torch::ShmQueueMeta* _meta = nullptr;
};

} // namespace gigl

#endif // GIGL_CORE_CHANNEL_SHM_QUEUE_PROBE_H_
