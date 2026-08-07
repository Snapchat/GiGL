"""Utilities for node feature quantization in GiGL."""

from typing import Final

SUPPORTED_QUANTIZATION_BITS: Final[tuple[int, ...]] = (1, 2, 4, 8)
