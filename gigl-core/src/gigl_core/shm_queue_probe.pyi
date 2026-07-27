class ShmQueueProbe:
    """Read-only queue-depth probe for a GraphLearn-for-PyTorch shared-memory channel."""

    def __init__(self, shmid: int) -> None:
        """Attaches to a GLT ``ShmQueue`` segment.

        Args:
            shmid: System V shared-memory id of a live GLT ``ShmQueue``. In Python this is what
                ``graphlearn_torch``'s ``SampleQueue.__getstate__()`` returns.

        Raises:
            RuntimeError: If the segment cannot be attached (bad or stale shmid, or permissions). The
                message includes the shmid and the underlying ``errno`` description. This mirrors what
                the previous ``ctypes``-based implementation raised, so callers need no changes.
        """
        ...

    def qsize(self) -> int:
        """Returns the approximate number of messages currently in the channel.

        The value is instantaneous and unsynchronised: producers and consumers mutate the underlying
        counters concurrently and the probe holds none of GLT's locks.

        Treat it as a metric, not as a basis for control flow.
        """
        ...

    @property
    def shmid(self) -> int:
        """The System V shmid this probe is attached to."""
        ...
