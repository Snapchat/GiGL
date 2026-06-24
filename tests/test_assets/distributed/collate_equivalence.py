"""Reusable equivalence assertions for collated neighbor-loader batches.

These helpers compare two collated PyG ``Data``/``HeteroData`` objects field-for-field
so callers can assert that two collation implementations produce identical output.

They are shared by the collate-equivalence unit tests across implementations.
"""

from typing import Union

import torch
from torch_geometric.data import Data, HeteroData

from tests.test_assets.distributed.utils import assert_tensor_equality


def assert_label_dict_equal(
    actual: dict[int, torch.Tensor],
    expected: dict[int, torch.Tensor],
    *,
    name: str = "label_dict",
) -> None:
    """Assert two ragged anchor->label-index dicts are identical.

    Enforces the three label traps:
    key-set equality (every anchor index present),
    per-key tensor equality including empty tensors,
    and matching device.

    Args:
        actual: Anchor-local-id to local-label-index tensor produced by one implementation.
        expected: The oracle dict to compare against.
        name: Label used in assertion messages (e.g. "y_positive").

    Raises:
        AssertionError: If key sets differ, a value tensor differs (incl. empties), or devices differ.
    """
    assert set(actual.keys()) == set(expected.keys()), (
        f"{name}: anchor key sets differ. "
        f"actual={sorted(actual.keys())} expected={sorted(expected.keys())}"
    )
    for anchor_id in expected:
        actual_value = actual[anchor_id]
        expected_value = expected[anchor_id]
        assert actual_value.device == expected_value.device, (
            f"{name}[{anchor_id}]: device mismatch "
            f"{actual_value.device} != {expected_value.device}"
        )
        # EXACT comparison (dim=None), including empty tensors. The oracle produces
        # local indices via torch.nonzero(mask)[:, 0], which is ASCENDING and preserves
        # duplicate multiplicity when an anchor's padded label row repeats a global id
        # across columns. Order AND multiplicity are contractual, so implementations
        # must reproduce them exactly — a dim=0 sort here would mask a wrong-order impl
        # and a dropped/added duplicate.
        # assert_tensor_equality with dim=None routes to torch.testing.assert_close,
        # which compares empty long tensors as equal (shape [0], dtype int64).
        assert_tensor_equality(actual_value, expected_value)


def _assert_optional_tensor_equal(
    actual_obj: Union[Data, HeteroData],
    expected_obj: Union[Data, HeteroData],
    attr: str,
    *,
    sort_dim: Union[int, None] = None,
) -> None:
    """Assert an optional tensor attribute matches, including matched-absence.

    Both objects must agree on whether ``attr`` is present.
    When present on both, the tensors must be equal.

    Args:
        actual_obj: Object produced by the implementation under test.
        expected_obj: Oracle object.
        attr: Attribute name to compare (e.g. "x", "edge_attr", "num_sampled_nodes").
        sort_dim: Optional dimension to sort over before comparing.

    Raises:
        AssertionError: If presence differs or values differ.
    """
    has_actual = hasattr(actual_obj, attr) and getattr(actual_obj, attr) is not None
    has_expected = (
        hasattr(expected_obj, attr) and getattr(expected_obj, attr) is not None
    )
    assert has_actual == has_expected, (
        f"attribute '{attr}' presence differs: actual={has_actual} expected={has_expected}"
    )
    if has_expected:
        assert_tensor_equality(
            getattr(actual_obj, attr), getattr(expected_obj, attr), dim=sort_dim
        )


def _assert_label_field_equal(
    actual_obj: Union[Data, HeteroData],
    expected_obj: Union[Data, HeteroData],
    field: str,
) -> None:
    """Assert a label field (``y_positive``/``y_negative``) matches.

    Handles matched-absence, the single-supervision-edge-type form
    (``dict[int, Tensor]``) and the multiple-supervision form
    (``dict[EdgeType, dict[int, Tensor]]``).

    Args:
        actual_obj: Object produced by the implementation under test.
        expected_obj: Oracle object.
        field: Either "y_positive" or "y_negative".

    Raises:
        AssertionError: If presence, edge-type key sets, or label dicts differ.
    """
    has_actual = hasattr(actual_obj, field)
    has_expected = hasattr(expected_obj, field)
    assert has_actual == has_expected, (
        f"label field '{field}' presence differs: actual={has_actual} expected={has_expected}"
    )
    if not has_expected:
        return
    actual_value = getattr(actual_obj, field)
    expected_value = getattr(expected_obj, field)
    # Distinguish nested (multiple edge types) from flat (single edge type) by
    # inspecting a sample value: nested maps EdgeType -> dict.
    expected_is_nested = len(expected_value) > 0 and isinstance(
        next(iter(expected_value.values())), dict
    )
    actual_is_nested = len(actual_value) > 0 and isinstance(
        next(iter(actual_value.values())), dict
    )
    # Catch a flat-vs-nested structural divergence cleanly (e.g. an impl that
    # collapses multi-supervision into a single flat dict, or vice versa). Both
    # non-empty: shapes must agree. (If either is empty, key-set equality below
    # handles it — an empty dict is shape-agnostic.)
    if len(expected_value) > 0 and len(actual_value) > 0:
        assert actual_is_nested == expected_is_nested, (
            f"{field}: label-dict nesting differs "
            f"(actual nested={actual_is_nested}, expected nested={expected_is_nested})"
        )
    if expected_is_nested:
        assert set(actual_value.keys()) == set(expected_value.keys()), (
            f"{field}: edge-type key sets differ "
            f"{set(actual_value.keys())} != {set(expected_value.keys())}"
        )
        for edge_type in expected_value:
            assert_label_dict_equal(
                actual_value[edge_type],
                expected_value[edge_type],
                name=f"{field}[{edge_type}]",
            )
    else:
        assert_label_dict_equal(actual_value, expected_value, name=field)


def _assert_homogeneous_equal(actual: Data, expected: Data) -> None:
    """Assert two homogeneous ``Data`` batches are field-for-field identical.

    Args:
        actual: ``Data`` produced by the implementation under test.
        expected: Oracle ``Data``.

    Raises:
        AssertionError: On any field mismatch.
    """
    # EXACT comparison (dim=None), NOT sorted. The collate path is deterministic
    # for a fixed sampler seed + input; implementations only change the label remap,
    # so `node` (the local->global map) must be bit-identical across impls. Sorting
    # `node` here would be a FALSE-PASS HOLE: it would let two impls with different
    # local orderings pass while their `y_positive` keys (local anchor ids) and `coo`
    # indices silently point at different global nodes. Keep dim=None.
    assert_tensor_equality(actual.node, expected.node)
    _assert_optional_tensor_equal(actual, expected, "x")
    _assert_optional_tensor_equal(actual, expected, "edge_attr")
    _assert_optional_tensor_equal(actual, expected, "num_sampled_nodes")
    _assert_optional_tensor_equal(actual, expected, "num_sampled_edges")
    _assert_optional_tensor_equal(actual, expected, "batch")

    assert getattr(actual, "batch_size", None) == getattr(expected, "batch_size", None), (
        f"batch_size differs: {getattr(actual, 'batch_size', None)} != "
        f"{getattr(expected, 'batch_size', None)}"
    )

    # Edge connectivity via coo() -> (dst, src, *rest). Because `node` is already
    # asserted bit-identical above, the local edge-index tensors share the same
    # local id space across impls, so compare the *local* endpoints exactly. We do
    # NOT pre-sort: edge order is deterministic. If a future impl is shown to
    # legitimately reorder edges (justify in code review), relax to a paired
    # (src,dst) co-sort that keeps endpoints aligned — never an independent
    # per-endpoint sort, which would decouple src from dst.
    actual_dst, actual_src, *_ = actual.coo()
    expected_dst, expected_src, *_ = expected.coo()
    assert_tensor_equality(actual_src, expected_src)
    assert_tensor_equality(actual_dst, expected_dst)

    _assert_label_field_equal(actual, expected, "y_positive")
    _assert_label_field_equal(actual, expected, "y_negative")


def _assert_heterogeneous_equal(actual: HeteroData, expected: HeteroData) -> None:
    raise NotImplementedError("Implemented in Task 2.")


def assert_collated_equal(
    actual: Union[Data, HeteroData],
    expected: Union[Data, HeteroData],
) -> None:
    """Assert two collated batches are field-for-field identical.

    Compares ``node`` maps, ``coo()`` connectivity, ``x``, ``edge_attr``,
    ``num_sampled_nodes``/``num_sampled_edges``, ``batch``/``batch_size``,
    and the ragged ``y_positive``/``y_negative`` label dicts (including empties).

    Both arguments must be the same concrete type (``Data`` or ``HeteroData``).

    Args:
        actual: Batch produced by the implementation under test.
        expected: Oracle batch (the Python ``_collate_fn`` output).

    Raises:
        AssertionError: On type mismatch or any field mismatch.
    """
    assert type(actual) is type(expected), (
        f"collated batch type differs: {type(actual)} != {type(expected)}"
    )
    if isinstance(expected, Data):
        _assert_homogeneous_equal(actual, expected)
    else:
        _assert_heterogeneous_equal(actual, expected)
