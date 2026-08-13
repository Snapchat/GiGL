"""Graph topology types for range-partitioned distributed graphs.

The conversion logic in :class:`OffsetTopology` is adapted from
GraphLearn-for-PyTorch's ``Topology`` / ``coo_to_csr`` implementation
(https://github.com/alibaba/graphlearn-for-pytorch), which is licensed under
the Apache License, Version 2.0. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Stock GLT ``Topology`` sizes the CSR/CSC index pointer from the maximum id in
the compressed dimension. Under range partitioning the compressed dimension of
each partition covers a contiguous slice ``[offset, offset + num_nodes)`` of
the global node-id space, so a globally-sized index pointer carries an all-zero
prefix of ``offset`` entries. :class:`OffsetTopology` instead rebases the
compressed dimension onto ``[0, num_nodes)`` so the index pointer is sized to
the local partition, while the adjacent dimension (the neighbor ids), edge ids,
and edge weights all stay in the global id space.
"""

from collections.abc import Mapping
from typing import Literal, Optional, Tuple, Union

import torch
from graphlearn_torch.data import Graph, Topology
from graphlearn_torch.typing import TensorDataType
from graphlearn_torch.utils import convert_to_tensor, ptr2ind


class OffsetTopology(Topology):
    """CSR/CSC topology whose compressed dimension is rebased by a partition offset.

    The compressed-dimension ids of the input COO edge index must all lie in
    ``[offset, offset + num_nodes)``; they are stored rebased to
    ``[0, num_nodes)`` so ``indptr`` has exactly ``num_nodes + 1`` entries.

    The adjacent dimension (``indices``), ``edge_ids``, and ``edge_weights``
    keep their global values.

    Lookups into ``indptr`` therefore require offset-rebased ids; values read
    from ``indices``/``edge_ids``/``edge_weights`` are global.

    Only COO input is supported.

    Within each compressed row, edges keep their input order (a stable sort on
    the compressed dimension only). Stock GLT ``Topology`` additionally orders
    neighbors within each row; GLT APIs that rely on that ordering (e.g. strict
    negative sampling) are not supported with this topology.

    Args:
        edge_index (a 2D torch.Tensor or numpy.ndarray, or a tuple): The COO
            edge index, in the order of first row and then column.
        edge_ids (torch.Tensor or numpy.ndarray, optional): The global edge ids
            for graph edges. If set to ``None``, it will be aranged by the edge
            size. (default: ``None``)
        edge_weights (torch.Tensor or numpy.ndarray, optional): The edge
            weights for graph edges. (default: ``None``)
        input_layout (str): Must be ``'COO'``. (default: ``'COO'``)
        layout ('CSR' or 'CSC'): The target edge layout representation for the
            output. (default: ``'CSR'``)
        offset (int): The first global node id owned by this partition in the
            compressed dimension.
        num_nodes (int): The number of nodes owned by this partition in the
            compressed dimension.

    Raises:
        ValueError: If ``input_layout`` is not ``'COO'``, if ``offset`` or
            ``num_nodes`` is negative, or if any compressed-dimension id falls
            outside ``[offset, offset + num_nodes)``.
    """

    def __init__(
        self,
        edge_index: Union[TensorDataType, Tuple[TensorDataType, TensorDataType]],
        edge_ids: Optional[TensorDataType] = None,
        edge_weights: Optional[TensorDataType] = None,
        input_layout: str = "COO",
        layout: Literal["CSR", "CSC"] = "CSR",
        *,
        offset: int,
        num_nodes: int,
    ):
        # Intentionally does not call Topology.__init__: the base constructor
        # runs the globally-sized conversion this class exists to replace. All
        # attributes the base class reads (_layout, _indptr, _indices,
        # _edge_ids, _edge_weights) are set here.
        if str(input_layout).upper() != "COO":
            raise ValueError(
                f"OffsetTopology only supports input_layout='COO', got {input_layout}."
            )
        if layout not in ("CSR", "CSC"):
            raise ValueError(f"layout must be 'CSR' or 'CSC', got {layout}.")
        if offset < 0 or num_nodes < 0:
            raise ValueError(
                f"offset and num_nodes must be non-negative, got offset={offset}, "
                f"num_nodes={num_nodes}."
            )

        edge_index = convert_to_tensor(edge_index, dtype=torch.int64)
        row, col = edge_index[0], edge_index[1]
        if row.numel() != col.numel():
            raise ValueError(
                f"Row and column must have the same number of entries, got "
                f"{row.numel()} rows and {col.numel()} columns."
            )
        num_edges = row.numel()

        edge_ids = convert_to_tensor(edge_ids, dtype=torch.int64)
        if edge_ids is None:
            edge_ids = torch.arange(num_edges, dtype=torch.int64, device=row.device)
        elif edge_ids.numel() != num_edges:
            raise ValueError(f"Expected {num_edges} edge ids, got {edge_ids.numel()}.")

        edge_weights = convert_to_tensor(edge_weights, dtype=torch.float)
        if edge_weights is not None and edge_weights.numel() != num_edges:
            raise ValueError(
                f"Expected {num_edges} edge weights, got {edge_weights.numel()}."
            )

        compressed, adjacent = (row, col) if layout == "CSR" else (col, row)

        if num_edges == 0:
            indptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
            indices = adjacent
        else:
            compressed_min = int(compressed.min().item())
            compressed_max = int(compressed.max().item())
            if compressed_min < offset or compressed_max >= offset + num_nodes:
                raise ValueError(
                    f"Compressed-dimension ids must lie in [{offset}, "
                    f"{offset + num_nodes}), got range "
                    f"[{compressed_min}, {compressed_max}]."
                )
            local_compressed = compressed - offset
            # One stable sort on the rebased compressed dimension; the same
            # permutation realigns neighbors, edge ids, and edge weights so the
            # per-edge pairing is preserved.
            permutation = torch.argsort(local_compressed, stable=True)
            indices = adjacent[permutation]
            edge_ids = edge_ids[permutation]
            if edge_weights is not None:
                edge_weights = edge_weights[permutation]
            counts = torch.bincount(local_compressed, minlength=num_nodes)
            indptr = torch.zeros(num_nodes + 1, dtype=torch.int64)
            indptr[1:] = torch.cumsum(counts, dim=0)

        self._layout = layout
        self._indptr = indptr
        self._indices = indices
        self._edge_ids = edge_ids
        self._edge_weights = edge_weights
        self.offset = offset

    def to_coo(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Convert to COO format with global ids in both dimensions.

        Returns:
            row indice tensor, column indice tensor, edge id tensor, edge weight tensor
        """
        compressed = ptr2ind(self._indptr) + self.offset
        if self._layout == "CSR":
            return compressed, self._indices, self._edge_ids, self._edge_weights
        else:
            return self._indices, compressed, self._edge_ids, self._edge_weights

    def to_csr(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Not supported: the stored index pointer is rebased to the local
        partition, so exposing it through the global-id CSR API would leak
        local-based pointers to callers expecting global sizing.
        """
        raise NotImplementedError(
            "OffsetTopology does not support to_csr(); its index pointer is "
            "rebased to the local partition node range. Use to_coo() for "
            "global ids."
        )

    def to_csc(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Not supported: the stored index pointer is rebased to the local
        partition, so exposing it through the global-id CSC API would leak
        local-based pointers to callers expecting global sizing.
        """
        raise NotImplementedError(
            "OffsetTopology does not support to_csc(); its index pointer is "
            "rebased to the local partition node range. Use to_coo() for "
            "global ids."
        )


def contains_offset_topology(
    graph: Optional[Union[Graph, Mapping]],
) -> bool:
    """Returns True if any topology held by ``graph`` is an :class:`OffsetTopology`.

    Args:
        graph: A GLT ``Graph``, a mapping of edge type to ``Graph``, or ``None``.
    """
    if graph is None:
        return False
    if isinstance(graph, Graph):
        return isinstance(graph.topo, OffsetTopology)
    return any(
        isinstance(edge_graph.topo, OffsetTopology) for edge_graph in graph.values()
    )
