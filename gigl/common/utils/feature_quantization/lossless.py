"""Lossless narrow-integer packing of feature columns, the lossless half of feature compression.

Many columns of a typical feature set are categorical ids (``nn.Embedding`` indices) that need one
to four bytes, not four. ``FeatureCodec`` packs a set of such columns into one ``uint8`` row of
fixed width. Rows are stored in the packed sidecar that feature quantization already uses
(``FeatureQuantizationMetadata``): GLT's feature store, RPC, stitch, shared-memory channel and
host-to-device copy move opaque byte rows unchanged, and ``decode`` rebuilds the float32 columns on
the consumer's device.

Each column has one encoding, which says how the codec stores that column. ``encode`` raises on a value its column's encoding cannot represent:

* ``float32``: exact real values. Also the encoding of a column that preprocessing already quantized.
* ``bfloat16``: real values, lossy.
* ``int8``: ids in ``[-128, 127]``.
* ``uint8``: ids in ``[0, 255]``.
* ``uint8_bias1``: ids in ``[-1, 254]``, stored as ``id + 1``. Fits an out-of-vocabulary id of -1.
* ``int16``: ids in ``[-32768, 32767]``.
* ``int32``: ids in ``[-2**31, 2**31 - 128]``. The upper bound is the largest float32 below ``2**31``.
* ``bit``: exactly ``0`` or ``1`` (for example a one-hot column), stored in one bit.

Layout: columns are grouped by encoding and the groups are laid out from widest to
narrowest (4-byte, 2-byte, 1-byte) so every group starts at an offset aligned to its element
size and can be read with ``Tensor.view(dtype)`` on a byte slice. The ``bit`` group comes last: its
columns are packed eight per byte, column ``j`` of the group in bit ``j % 8`` of byte ``j // 8``
(least significant bit first), and the last byte is zero-padded. The row is zero-padded to a
multiple of four bytes so that row strides stay aligned for any row subset.

Example:
    Declare the codec of a node type with five columns: an ``int16`` id, an ``int8`` id, a
    ``uint8_bias1`` id, an ``int32`` id and a 0/1 ``bit`` column.

    >>> import torch
    >>> from gigl.common.utils.feature_quantization.lossless import FeatureCodec
    >>> codec = FeatureCodec(("int16", "int8", "uint8_bias1", "int32", "bit"))
    >>> features = torch.tensor(
    ...     [[300.0, -3.0, -1.0, 100000.0, 1.0], [-2.0, 127.0, 254.0, -7.0, 0.0]]
    ... )

    In the declared column order the columns need 2, 1, 1, 4 bytes and 1 bit. ``encode`` lays the
    groups out widest first: ``int32`` 4 bytes, ``int16`` 2, ``int8`` 1, ``uint8_bias1`` 1, then one
    byte for the ``bit`` group. That is 9 bytes, padded to 12, and each row is stored as ``uint8``.

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
    ValueError: Column 1 holds a value that is not an integer in [-128, 127] (encoding int8); the packed layout cannot represent it
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Final, NamedTuple, Optional

import torch
from jaxtyping import Float, Float32, Shaped, UInt8


@dataclass(frozen=True)
class _ColumnEncoding:
    dtype: torch.dtype
    # Added to a column value before the cast to ``dtype`` and subtracted after decoding.
    bias: int
    # Inclusive range of representable column values; (0, 0) and unused for the dense encodings.
    value_range: tuple[int, int]


_ENCODINGS: Final[dict[str, _ColumnEncoding]] = {
    "float32": _ColumnEncoding(torch.float32, 0, (0, 0)),
    "int32": _ColumnEncoding(torch.int32, 0, (-(2**31), 2**31 - 128)),
    "bfloat16": _ColumnEncoding(torch.bfloat16, 0, (0, 0)),
    "int16": _ColumnEncoding(torch.int16, 0, (-(2**15), 2**15 - 1)),
    "int8": _ColumnEncoding(torch.int8, 0, (-(2**7), 2**7 - 1)),
    "uint8": _ColumnEncoding(torch.uint8, 0, (0, 2**8 - 1)),
    "uint8_bias1": _ColumnEncoding(torch.uint8, 1, (-1, 2**8 - 2)),
    # Eight columns per byte; ``dtype`` is the type of one packed byte.
    "bit": _ColumnEncoding(torch.uint8, 0, (0, 1)),
}
_DENSE_ENCODINGS: Final[tuple[str, ...]] = ("float32", "bfloat16")
# Widest first (dict order), so each group's byte offset is a multiple of its element size.
_ENCODING_ORDER: Final[tuple[str, ...]] = tuple(_ENCODINGS)
# Narrowest first, so the first encoding that holds a range is the cheapest.
_INTEGER_ENCODINGS_NARROWEST_FIRST: Final[tuple[str, ...]] = (
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


def narrowest_integer_encoding(low: int, high: int) -> Optional[str]:
    """The narrowest integer encoding that holds every integer in ``[low, high]``, if any."""
    for encoding in _INTEGER_ENCODINGS_NARROWEST_FIRST:
        encoding_low, encoding_high = _ENCODINGS[encoding].value_range
        if encoding_low <= low and high <= encoding_high:
            return encoding
    return None


class _ColumnRun(NamedTuple):
    """Columns that are next to each other both in the original matrix and in their group.

    Decode copies a run as one slice, not one column at a time.
    """

    # Original column index of the first column of the run.
    first_column: int
    # Position of that column within its group.
    first_position: int
    # Number of columns in the run.
    length: int


@dataclass(frozen=True)
class _Group:
    encoding: str
    dtype: torch.dtype
    bias: int
    byte_offset: int
    columns: tuple[int, ...]
    runs: tuple[_ColumnRun, ...]

    @property
    def num_bytes(self) -> int:
        if self.encoding == "bit":
            return (len(self.columns) + 7) // 8
        return len(self.columns) * self.dtype.itemsize


@dataclass(frozen=True)
class FeatureCodec:
    """Packing of a float32 ``[N, D]`` matrix into ``uint8`` ``[N, row_bytes]`` rows.

    Args:
        column_encodings: Encoding of every original column; the module docstring lists the encodings.
    """

    column_encodings: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = set(self.column_encodings) - set(_ENCODINGS)
        if unknown:
            raise ValueError(f"Unknown column encodings {sorted(unknown)}")

    @property
    def num_columns(self) -> int:
        return len(self.column_encodings)

    @cached_property
    def _groups(self) -> tuple[_Group, ...]:
        groups: list[_Group] = []
        offset = 0
        for encoding in _ENCODING_ORDER:
            columns = tuple(
                column
                for column, column_encoding in enumerate(self.column_encodings)
                if column_encoding == encoding
            )
            if not columns:
                continue
            encoding_spec = _ENCODINGS[encoding]
            # Split the group's columns into runs. Positions within a group always go up by one, so a column
            # extends the last run if it comes right after that run's last original column.
            # For example, group columns (2, 3, 4, 7) give two runs: original columns 2-4 at group
            # positions 0-2, and original column 7 at group position 3.
            runs: list[_ColumnRun] = []
            for position, column in enumerate(columns):
                if runs and runs[-1].first_column + runs[-1].length == column:
                    runs[-1] = runs[-1]._replace(length=runs[-1].length + 1)
                else:
                    runs.append(
                        _ColumnRun(
                            first_column=column, first_position=position, length=1
                        )
                    )
            group = _Group(
                encoding,
                encoding_spec.dtype,
                encoding_spec.bias,
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
        self,
        features: Float[torch.Tensor, "n _num_columns"],
        *,
        validate: bool = True,
    ) -> UInt8[torch.Tensor, "n {self.row_bytes}"]:
        """Packs float32 ``[N, D]`` features into ``uint8`` ``[N, row_bytes]`` rows.

        Args:
            features: The ``[N, D]`` matrix to pack.
            validate: Whether to check that every id column holds integers its encoding can store.
                Pass ``False`` only when the statistics this codec was built from cover ``features``, because
                they already prove that every value fits.

        Raises:
            ValueError: If the width is not ``D``, if a ``float32`` or ``bfloat16`` column is declared
                and ``features`` is not float32 (the cast would silently lose precision), or if ``validate``
                is set and an id column holds a non-integral or out-of-range value (including NaN).
        """
        if features.dim() != 2 or features.shape[1] != self.num_columns:
            raise ValueError(
                f"Expected features of shape [N, {self.num_columns}], got {tuple(features.shape)}"
            )
        if features.dtype != torch.float32 and any(
            group.encoding in _DENSE_ENCODINGS for group in self._groups
        ):
            raise ValueError(
                f"Dense encodings require float32 features, got {features.dtype}; "
                "casting would silently lose precision"
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
                if group.encoding in _DENSE_ENCODINGS:
                    values = values.to(torch.float32)
                else:
                    if validate:
                        # Check before any narrowing cast so float64 ids are not rounded into range.
                        self._check_ids(values, group)
                    if group.bias:
                        values = values + group.bias
                if group.encoding == "bit":
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
        low, high = _ENCODINGS[group.encoding].value_range
        ok = (values >= low) & (values <= high) & (values == values.floor())
        if not bool(ok.all()):
            bad_column = group.columns[int((~ok).any(dim=0).nonzero()[0])]
            raise ValueError(
                f"Column {bad_column} holds a value that is not an integer in [{low}, {high}] "
                f"(encoding {group.encoding}); the packed layout cannot represent it"
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
            if group.encoding == "bit":
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
