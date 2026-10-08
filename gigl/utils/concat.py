from collections import deque
from typing import Any, Callable, Optional, Sequence, Union

import torch

from gigl.utils.share_memory import allocate_preshared

ChunkField = Union[int, tuple[int, ...]]


def concatenate_chunks(
    chunks: list[tuple[Any, ...]],
    fields: Sequence[Optional[ChunkField]],
    *,
    axis: int = 0,
    convert: Optional[Callable[[Any], torch.Tensor]] = None,
    preshare: bool = False,
) -> dict[ChunkField, torch.Tensor]:
    """Concatenates fields of a list of chunks into preallocated tensors.

    For each field ``i`` this gives the same result as ``torch.cat([chunk[i] for chunk in chunks], dim=axis)``.
    For a tuple of fields it gives ``torch.stack`` of those concatenations, so ``(0, 1)`` builds a
    ``(2, num_rows)`` edge index from source and destination id fields.

    The input list is consumed. Each output is allocated once and each chunk is released after it is copied,
    so peak memory stays near the size of the outputs plus one chunk, instead of the inputs plus a full copy.

    Chunks with zero rows are skipped, and outputs take their shape from the first chunk that has rows.
    This matches ``torch.cat``, which ignores the legacy 1-D empty tensors a rank with nothing to send can return.

    Example:
        >>> chunks = [(torch.tensor([0, 1]), torch.tensor([5, 6])), (torch.tensor([2]), torch.tensor([7]))]
        >>> concatenate_chunks(chunks, [(0, 1)])[(0, 1)]
        tensor([[0, 1, 2],
                [5, 6, 7]])

    Args:
        chunks: Non-empty list of chunks. A chunk is a tuple of tensors with the same size along ``axis``.
            Each field has the same dtype and the same size on every other dim in every chunk.
        fields: Fields to concatenate. An int is the index of a field in each chunk. A tuple of ints stacks
            those fields along a new leading dim. ``None`` entries are skipped, so callers can pass the
            index of an optional field as is.
        axis: The dim to concatenate along.
        convert: Maps each chunk field to a torch tensor when it is copied, e.g. from a TensorFlow tensor.
            Fields must already be torch tensors when this is ``None``.
        preshare: Allocate CPU outputs with :func:`gigl.utils.share_memory.allocate_preshared`, so a later
            ``share_memory_()`` does not copy them.
            Otherwise outputs are allocated with ``torch.empty``.

    Returns:
        A dict from each non-``None`` entry of ``fields`` to its concatenated tensor.

    Raises:
        ValueError: If ``chunks`` is empty, or a field's dtype or shape off ``axis`` differs across chunks.
    """
    if not chunks:
        raise ValueError("Expected at least one chunk to concatenate.")
    to_torch = convert if convert is not None else (lambda tensor: tensor)

    def num_rows_in(chunk: tuple[Any, ...]) -> int:
        return int(chunk[0].shape[axis])

    total_rows = sum(num_rows_in(chunk) for chunk in chunks)
    pending_chunks = deque(chunk for chunk in chunks if num_rows_in(chunk) > 0)
    # If every chunk is empty, shape the outputs from the first one.
    first_chunk = pending_chunks[0] if pending_chunks else chunks[0]
    # Callers may hold other references to the list, so clear it to leave the deque as the only owner.
    chunks.clear()

    outputs: dict[ChunkField, torch.Tensor] = {}
    # (index of the field in each chunk, the slice of an output it is copied into)
    destinations: list[tuple[int, torch.Tensor]] = []
    for field in fields:
        if field is None:
            continue
        field_indices = (field,) if isinstance(field, int) else field
        template = to_torch(first_chunk[field_indices[0]])
        shape = list(template.shape)
        shape[axis] = total_rows
        if not isinstance(field, int):
            shape.insert(0, len(field_indices))
        if preshare and template.device.type == "cpu":
            output = allocate_preshared(tuple(shape), template.dtype)
        else:
            output = torch.empty(shape, dtype=template.dtype, device=template.device)
        if isinstance(field, int):
            destinations.append((field, output))
        else:
            destinations.extend(
                (field_index, output[row])
                for row, field_index in enumerate(field_indices)
            )
        outputs[field] = output
        del template
    del first_chunk

    next_row = 0
    while pending_chunks:
        chunk = pending_chunks.popleft()
        chunk_num_rows = num_rows_in(chunk)
        for field_index, destination in destinations:
            tensor = to_torch(chunk[field_index])
            # copy_ would silently cast or broadcast where torch.cat would promote or fail.
            if tensor.dtype != destination.dtype or _shape_off_axis(
                tensor, axis
            ) != _shape_off_axis(destination, axis):
                raise ValueError(
                    f"Chunk field {field_index} has dtype {tensor.dtype} and shape {tuple(tensor.shape)}, expected "
                    f"dtype {destination.dtype} and shape {tuple(destination.shape)} on every dim except {axis}."
                )
            destination.narrow(axis, next_row, chunk_num_rows).copy_(tensor)
            del tensor
        next_row += chunk_num_rows
        del chunk

    return outputs


def _shape_off_axis(tensor: torch.Tensor, axis: int) -> tuple[int, ...]:
    shape = list(tensor.shape)
    del shape[axis]
    return tuple(shape)
