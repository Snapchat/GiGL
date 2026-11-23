"""Wire format for forwarding a sampling-worker exception to the training loop.

A GLT ``SampleMessage`` is a ``dict[str, torch.Tensor]``. We reserve one ``#``-prefixed
key (matching GLT's ``#IS_HETERO`` / ``#META.`` convention) to carry a UTF-8 traceback
encoded as a uint8 tensor, so a failed sampling coroutine can surface as a fast, explained
error on the consumer instead of a silent hang.
"""

from typing import Final

import torch
from graphlearn_torch.channel import SampleMessage

SAMPLING_ERROR_KEY: Final[str] = "#SAMPLING_ERROR"


def encode_sampling_error(traceback_str: str) -> torch.Tensor:
    """Encode a traceback string as a writable 1-D uint8 tensor.

    Uses ``bytearray`` so the backing buffer is writable (``torch.frombuffer`` warns on
    read-only buffers and raises on empty ones); a single sentinel byte represents the
    empty string so the tensor is never zero-length.

    Args:
        traceback_str: The traceback text to transport.

    Returns:
        A 1-D ``torch.uint8`` tensor holding the UTF-8 bytes.
    """
    raw = traceback_str.encode("utf-8")
    if not raw:
        raw = b"\x00"
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8)


def raise_if_sampling_error(msg: SampleMessage) -> None:
    """Raise ``RuntimeError`` with the embedded traceback if ``msg`` is a poison pill.

    No-op when the reserved key is absent.

    Args:
        msg: A received ``SampleMessage``.

    Raises:
        RuntimeError: If ``msg`` carries a sampling-error payload.
    """
    if SAMPLING_ERROR_KEY in msg:
        decoded = bytes(msg[SAMPLING_ERROR_KEY].cpu().numpy()).decode(
            "utf-8", errors="replace"
        )
        raise RuntimeError(
            "A sampling worker failed while producing this batch:\n" + decoded
        )


def raise_if_message_exceeds_channel(msg: SampleMessage, channel: object) -> None:
    """Raise ``RuntimeError`` if ``msg`` can never fit in ``channel``'s shared-memory buffer.

    GLT's ``ShmQueue`` rejects a message at least as large as its buffer with ``message is too large!``,
    but it throws while still holding its allocation lock, so every later ``send`` on that channel,
    including a poison pill, blocks forever. Checking first keeps the channel usable so the error reaches
    the consumer. The check only catches a message larger than the whole buffer; a message that is smaller
    but cannot be placed in the ring at its current offset is still rejected by GLT.

    The buffer size is read from the channel's ``max_message_bytes`` attribute, which
    ``BaseDistLoader.create_colocated_channel`` sets. Channels without it are not checked.

    Args:
        msg: The message about to be sent.
        channel: The channel it will be sent on.

    Raises:
        RuntimeError: If the message's tensor data is at least ``channel.max_message_bytes``.
    """
    max_message_bytes = getattr(channel, "max_message_bytes", None)
    if max_message_bytes is None:
        return
    message_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in msg.values()
    )
    if message_bytes >= max_message_bytes:
        raise RuntimeError(
            f"A sampled batch of {message_bytes} bytes cannot fit in the sampling channel, whose shared "
            f"memory holds {max_message_bytes} bytes. Increase the channel size or reduce the batch size, "
            "the number of sampled neighbors, or max_ppr_nodes."
        )
