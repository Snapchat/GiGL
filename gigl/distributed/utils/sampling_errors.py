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
