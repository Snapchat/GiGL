"""Lossless narrow-integer packing of feature columns, the lossless half of feature compression.

Many columns of a typical feature set are categorical ids (``nn.Embedding`` indices) that need one
to four bytes, not four. ``FeatureCodec`` packs a set of such columns into one ``uint8`` row of
fixed width. Rows are stored in the packed sidecar that feature quantization already uses
(``FeatureQuantizationMetadata``): GLT's feature store, RPC, stitch, shared-memory channel and
host-to-device copy move opaque byte rows unchanged, and ``decode`` rebuilds the float32 columns on
the consumer's device.

Storage kinds, one per column. ``encode`` raises on a value its column's kind cannot represent:

* ``float32``: exact real values. Also the kind of a column that preprocessing already quantized.
* ``bfloat16``: real values, lossy.
* ``int8``: ids in ``[-128, 127]``.
* ``uint8``: ids in ``[0, 255]``.
* ``uint8_bias1``: ids in ``[-1, 254]``, stored as ``id + 1``. Fits an out-of-vocabulary id of -1.
* ``int16``: ids in ``[-32768, 32767]``.
* ``int32``: ids in ``[-2**31, 2**31 - 128]``. The upper bound is the largest float32 below ``2**31``.
* ``bit``: exactly ``0`` or ``1`` (for example a one-hot column), stored in one bit.

Layout: columns are grouped by storage kind and the groups are laid out from widest to
narrowest (4-byte, 2-byte, 1-byte) so every group starts at an offset aligned to its element
size and can be read with ``Tensor.view(dtype)`` on a byte slice. The ``bit`` group comes last: its
columns are packed eight per byte, column ``j`` of the group in bit ``j % 8`` of byte ``j // 8``
(least significant bit first), and the last byte is zero-padded. The row is zero-padded to a
multiple of four bytes so that row strides stay aligned for any row subset.

Example:
    Declare the codec of a node type with one real-valued column, three id columns and one 0/1
    column. Column 0 is a float column that preprocessing already quantized, so it stays
    ``float32``.

    >>> import torch
    >>> from gigl.common.utils.feature_quantization.lossless import FeatureCodec
    >>> codec = FeatureCodec(("float32", "int8", "uint8_bias1", "int32", "bit"))
    >>> features = torch.tensor(
    ...     [[0.5, -3.0, -1.0, 100000.0, 1.0], [2.25, 127.0, 254.0, -7.0, 0.0]]
    ... )

    ``encode`` packs each row to ``uint8``: 4 + 4 + 1 + 1 + 1 bytes, padded to 12.

    >>> rows = codec.encode(features)
    >>> rows.dtype, tuple(rows.shape)
    (torch.uint8, (2, 12))

    ``decode`` returns float32 columns equal to the input.

    >>> decoded = codec.decode(rows)
    >>> decoded.dtype
    torch.float32
    >>> bool(torch.equal(decoded, features))
    True

    A value outside a column's range is rejected instead of silently wrapping.

    >>> codec.encode(torch.tensor([[0.0, 128.0, 0.0, 0.0, 0.0]]))
    Traceback (most recent call last):
        ...
    ValueError: Column 1 holds a value that is not an integer in [-128, 127] (kind int8); the packed layout cannot represent it
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Final, Optional

import torch
from jaxtyping import Float, Float32, Shaped, UInt8


@dataclass(frozen=True)
class _StorageKind:
    dtype: torch.dtype
    # Added to a column value before the cast to ``dtype`` and subtracted after decoding.
    bias: int
    # Inclusive range of representable column values; (0, 0) and unused for the dense kinds.
    value_range: tuple[int, int]


_KINDS: Final[dict[str, _StorageKind]] = {
    "float32": _StorageKind(torch.float32, 0, (0, 0)),
    "int32": _StorageKind(torch.int32, 0, (-(2**31), 2**31 - 128)),
    "bfloat16": _StorageKind(torch.bfloat16, 0, (0, 0)),
    "int16": _StorageKind(torch.int16, 0, (-(2**15), 2**15 - 1)),
    "int8": _StorageKind(torch.int8, 0, (-(2**7), 2**7 - 1)),
    "uint8": _StorageKind(torch.uint8, 0, (0, 2**8 - 1)),
    "uint8_bias1": _StorageKind(torch.uint8, 1, (-1, 2**8 - 2)),
    # Eight columns per byte; ``dtype`` is the type of one packed byte.
    "bit": _StorageKind(torch.uint8, 0, (0, 1)),
}
_DENSE_KINDS: Final[tuple[str, ...]] = ("float32", "bfloat16")
# Widest first (dict order), so each group's byte offset is a multiple of its element size.
_KIND_ORDER: Final[tuple[str, ...]] = tuple(_KINDS)
# Narrowest first, so the first kind that holds a range is the cheapest.
_INTEGER_KINDS_NARROWEST_FIRST: Final[tuple[str, ...]] = (
    "bit",
    "int8",
    "uint8",
    "uint8_bias1",
    "int16",
    "int32",
)
# A chunk and its temporaries should fit in CPU cache. A 65536-row chunk of a 512-column float32
# matrix is 128 MiB, which made every scan step stream through main memory; a sweep showed
# about 2x faster scans at 1 thread with chunks of this size.
_CHUNK_BYTES: Final[int] = 8 << 20


def chunk_rows(num_columns: int, element_size: int) -> int:
    """Rows per chunk so that one chunk of the input is about ``_CHUNK_BYTES``."""
    return max(1, _CHUNK_BYTES // max(1, num_columns * element_size))


def kind_holding_domain(low: int, high: int) -> Optional[str]:
    """The narrowest integer storage kind that holds every integer in ``[low, high]``, if any."""
    for kind in _INTEGER_KINDS_NARROWEST_FIRST:
        kind_low, kind_high = _KINDS[kind].value_range
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
        if self.kind == "bit":
            return (len(self.columns) + 7) // 8
        return len(self.columns) * self.dtype.itemsize


@dataclass(frozen=True)
class FeatureCodec:
    """Packing of a float32 ``[N, D]`` matrix into ``uint8`` ``[N, row_bytes]`` rows.

    Args:
        column_dtypes: Storage kind of every original column; the module docstring lists the kinds.
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
            storage_kind = _KINDS[kind]
            runs: list[tuple[int, int, int]] = []
            for position, column in enumerate(columns):
                if runs and runs[-1][0] + runs[-1][2] == column:
                    runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
                else:
                    runs.append((column, position, 1))
            group = _Group(
                kind,
                storage_kind.dtype,
                storage_kind.bias,
                offset,
                columns,
                tuple(runs),
            )
            groups.append(group)
            offset += group.num_bytes
        return tuple(groups)

    @cached_property
    def row_bytes(self) -> int:
        """Bytes per packed row, padded up to a multiple of four."""
        used = sum(group.num_bytes for group in self._groups)
        return (used + 3) // 4 * 4

    def encode(
        self, features: Float[torch.Tensor, "n _num_columns"]
    ) -> UInt8[torch.Tensor, "n {self.row_bytes}"]:
        """Packs float32 ``[N, D]`` features into ``uint8`` ``[N, row_bytes]`` rows.

        Raises:
            ValueError: If the width is not ``D``, or an id column holds a non-integral or
                out-of-range value (including NaN).
        """
        if features.dim() != 2 or features.shape[1] != self.num_columns:
            raise ValueError(
                f"Expected features of shape [N, {self.num_columns}], got {tuple(features.shape)}"
            )
        out = torch.zeros(
            features.shape[0], self.row_bytes, dtype=torch.uint8, device=features.device
        )
        rows_per_chunk = chunk_rows(features.shape[1], features.element_size())
        for start in range(0, features.shape[0], rows_per_chunk):
            chunk = features[start : start + rows_per_chunk]
            for group in self._groups:
                index = torch.tensor(
                    group.columns, dtype=torch.long, device=chunk.device
                )
                values = chunk.index_select(1, index)
                if group.kind in _DENSE_KINDS:
                    values = values.to(torch.float32)
                else:
                    # Check before any narrowing cast so float64 ids are not rounded into range.
                    self._check_ids(values, group)
                    if group.bias:
                        values = values + group.bias
                if group.kind == "bit":
                    stored = self._pack_bits(values)
                else:
                    stored = values.to(group.dtype).contiguous().view(torch.uint8)
                out[
                    start : start + chunk.shape[0],
                    group.byte_offset : group.byte_offset + group.num_bytes,
                ] = stored
        return out

    @staticmethod
    def _pack_bits(
        values: Float[torch.Tensor, "n columns"],
    ) -> UInt8[torch.Tensor, "n packed_bytes"]:
        """Packs 0/1 columns eight per byte, column ``j`` in bit ``j % 8`` of byte ``j // 8``."""
        num_rows, num_columns = values.shape
        num_bytes = (num_columns + 7) // 8
        padded = torch.zeros(
            num_rows, num_bytes * 8, dtype=torch.uint8, device=values.device
        )
        padded[:, :num_columns] = values.to(torch.uint8)
        shifts = torch.arange(8, dtype=torch.uint8, device=values.device)
        return (padded.view(num_rows, num_bytes, 8) << shifts).sum(
            dim=-1, dtype=torch.uint8
        )

    @staticmethod
    def _unpack_bits(
        packed: UInt8[torch.Tensor, "n packed_bytes"], num_columns: int
    ) -> Float32[torch.Tensor, "n {num_columns}"]:
        """Inverse of ``_pack_bits``."""
        shifts = torch.arange(8, dtype=torch.uint8, device=packed.device)
        bits = (packed.unsqueeze(-1) >> shifts) & 1
        return bits.reshape(packed.shape[0], packed.shape[1] * 8)[:, :num_columns].to(
            torch.float32
        )

    @staticmethod
    def _check_ids(values: torch.Tensor, group: _Group) -> None:
        low, high = _KINDS[group.kind].value_range
        ok = (values >= low) & (values <= high) & (values == values.floor())
        if not bool(ok.all()):
            bad_column = group.columns[int((~ok).any(dim=0).nonzero()[0])]
            raise ValueError(
                f"Column {bad_column} holds a value that is not an integer in [{low}, {high}] "
                f"(kind {group.kind}); the packed layout cannot represent it"
            )

    def decode(
        self, rows: Shaped[torch.Tensor, "n _row_bytes"]
    ) -> Float32[torch.Tensor, "n {self.num_columns}"]:
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
            part = rows[:, group.byte_offset : group.byte_offset + group.num_bytes]
            if group.kind == "bit":
                part = self._unpack_bits(part, len(group.columns))
            else:
                part = part.view(group.dtype).to(torch.float32)
            if group.bias:
                part = part - group.bias
            for first_column, first_position, length in group.runs:
                out[:, first_column : first_column + length] = part[
                    :, first_position : first_position + length
                ]
        return out
