"""Lossless narrow-integer packing of feature columns, the lossless half of feature compression.

Many columns of a typical feature set are categorical ids (``nn.Embedding`` indices) that need one
to four bytes, not four. ``FeatureCodec`` packs a set of such columns into one ``uint8`` row of
fixed width. Rows are stored in the packed sidecar that feature quantization already uses
(``FeatureQuantizationMetadata``): GLT's feature store, RPC, stitch, shared-memory channel and
host-to-device copy move opaque byte rows unchanged, and ``decode`` rebuilds the float32 columns on
the consumer's device.

Layout: columns are grouped by storage kind and the groups are laid out from widest to
narrowest (4-byte, 2-byte, 1-byte) so every group starts at an offset aligned to its element
size and can be read with ``Tensor.view(dtype)`` on a byte slice. The row is zero-padded to a
multiple of four bytes so that row strides stay aligned for any row subset.

Real-valued columns can be stored as ``float32`` (exact) or ``bfloat16`` (lossy), and every id kind is exact for
the values its range holds; ``encode`` raises on any value a column's kind cannot represent.
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Final, Optional

import torch

# kind -> (storage dtype, bias, inclusive range of representable column values).
# float32 cannot represent 2**31 - 1, so the int32 upper bound is the largest float32 below 2**31.
# ``uint8_bias1`` stores ``value + 1``, so ids in ``[-1, 254]`` (an OOV id of -1 is common) fit one byte.
_KINDS: Final[dict[str, tuple[torch.dtype, int, tuple[int, int]]]] = {
    "float32": (torch.float32, 0, (0, 0)),
    "int32": (torch.int32, 0, (-(2**31), 2**31 - 128)),
    "bfloat16": (torch.bfloat16, 0, (0, 0)),
    "int16": (torch.int16, 0, (-(2**15), 2**15 - 1)),
    "int8": (torch.int8, 0, (-(2**7), 2**7 - 1)),
    "uint8": (torch.uint8, 0, (0, 2**8 - 1)),
    "uint8_bias1": (torch.uint8, 1, (-1, 2**8 - 2)),
}
_DENSE_KINDS: Final[tuple[str, ...]] = ("float32", "bfloat16")
# Widest first, so each group's byte offset is a multiple of its element size.
_KIND_ORDER: Final[tuple[str, ...]] = tuple(_KINDS)
# Narrowest first, so the first kind that holds a range is the cheapest.
_INTEGER_KINDS_NARROWEST_FIRST: Final[tuple[str, ...]] = (
    "int8",
    "uint8",
    "uint8_bias1",
    "int16",
    "int32",
)
_CHUNK_ROWS: Final[int] = 1 << 16


def kind_holding_domain(low: int, high: int) -> Optional[str]:
    """The narrowest integer storage kind that holds every integer in ``[low, high]``, if any."""
    for kind in _INTEGER_KINDS_NARROWEST_FIRST:
        kind_low, kind_high = _KINDS[kind][2]
        if kind_low <= low and high <= kind_high:
            return kind
    return None


@dataclass(frozen=True)
class _Group:
    kind: str
    dtype: torch.dtype
    bias: int
    byte_offset: int
    columns: tuple[int, ...]
    # (first original column, first column within the group, length) of each maximal run of
    # columns that are consecutive both in the original matrix and in the group.
    runs: tuple[tuple[int, int, int], ...]

    @property
    def num_bytes(self) -> int:
        return len(self.columns) * self.dtype.itemsize


@dataclass(frozen=True)
class FeatureCodec:
    """Packing of a float32 ``[N, D]`` matrix into ``uint8`` ``[N, row_bytes]`` rows.

    Args:
        column_dtypes: Storage kind of every original column: ``float32``, ``bfloat16`` (real-valued
            columns, lossy), or ``int32`` / ``int16`` / ``int8`` / ``uint8`` / ``uint8_bias1``
            (integer ids).
    """

    column_dtypes: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = set(self.column_dtypes) - set(_KINDS)
        if unknown:
            raise ValueError(f"Unknown column kinds {sorted(unknown)}")

    @property
    def num_columns(self) -> int:
        return len(self.column_dtypes)

    @cached_property
    def _groups(self) -> tuple[_Group, ...]:
        groups: list[_Group] = []
        offset = 0
        for kind in _KIND_ORDER:
            columns = tuple(
                column
                for column, column_kind in enumerate(self.column_dtypes)
                if column_kind == kind
            )
            if not columns:
                continue
            dtype, bias, _ = _KINDS[kind]
            runs: list[tuple[int, int, int]] = []
            for position, column in enumerate(columns):
                if runs and runs[-1][0] + runs[-1][2] == column:
                    runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
                else:
                    runs.append((column, position, 1))
            group = _Group(kind, dtype, bias, offset, columns, tuple(runs))
            groups.append(group)
            offset += group.num_bytes
        return tuple(groups)

    @cached_property
    def row_bytes(self) -> int:
        """Bytes per packed row, padded up to a multiple of four."""
        used = sum(group.num_bytes for group in self._groups)
        return (used + 3) // 4 * 4

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """Packs float32 ``[N, D]`` features into ``uint8`` ``[N, row_bytes]`` rows.

        Raises:
            ValueError: If the width is not ``D``, or an id column holds a non-integral or
                out-of-range value (including NaN).
        """
        if features.dim() != 2 or features.shape[1] != self.num_columns:
            raise ValueError(
                f"Expected features of shape [N, {self.num_columns}], got {tuple(features.shape)}"
            )
        features = features.to(torch.float32)
        out = torch.zeros(
            features.shape[0], self.row_bytes, dtype=torch.uint8, device=features.device
        )
        for start in range(0, features.shape[0], _CHUNK_ROWS):
            chunk = features[start : start + _CHUNK_ROWS]
            for group in self._groups:
                index = torch.tensor(
                    group.columns, dtype=torch.long, device=chunk.device
                )
                values = chunk.index_select(1, index)
                if group.kind not in _DENSE_KINDS:
                    self._check_ids(values, group)
                    if group.bias:
                        values = values + group.bias
                stored = values.to(group.dtype).contiguous().view(torch.uint8)
                out[
                    start : start + chunk.shape[0],
                    group.byte_offset : group.byte_offset + group.num_bytes,
                ] = stored
        return out

    @staticmethod
    def _check_ids(values: torch.Tensor, group: _Group) -> None:
        low, high = _KINDS[group.kind][2]
        ok = (values >= low) & (values <= high) & (values == values.floor())
        if not bool(ok.all()):
            bad_column = group.columns[int((~ok).any(dim=0).nonzero()[0])]
            raise ValueError(
                f"Column {bad_column} holds a value that is not an integer in [{low}, {high}] "
                f"(kind {group.kind}); the packed layout cannot represent it"
            )

    def decode(self, rows: torch.Tensor) -> torch.Tensor:
        """Unpacks ``uint8`` ``[N, row_bytes]`` rows (on any device) to float32 ``[N, D]``."""
        if (
            rows.dtype != torch.uint8
            or rows.dim() != 2
            or rows.shape[1] != self.row_bytes
        ):
            raise ValueError(
                f"Expected uint8 rows of shape [N, {self.row_bytes}], got "
                f"{rows.dtype} {tuple(rows.shape)}"
            )
        if rows.stride(0) % 4 or rows.stride(1) != 1 or rows.storage_offset() % 4:
            # ``view(dtype)`` needs unit column stride and element-aligned strides and offsets; a copy restores them.
            rows = rows.clone(memory_format=torch.contiguous_format)
        out = torch.empty(
            rows.shape[0], self.num_columns, dtype=torch.float32, device=rows.device
        )
        for group in self._groups:
            part = rows[
                :, group.byte_offset : group.byte_offset + group.num_bytes
            ].view(group.dtype)
            part = part.to(torch.float32)
            if group.bias:
                part = part - group.bias
            for first_column, first_position, length in group.runs:
                out[:, first_column : first_column + length] = part[
                    :, first_position : first_position + length
                ]
        return out
