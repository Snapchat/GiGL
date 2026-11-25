"""The native neighbor lookup over a CSR whose row pointers are int32 or int64.

GiGL stores a compact CSR's ``indptr`` as int32 when the edge count fits (``0003-glt-int32-indptr.patch``). The native
lookup must read both, with the same results as GLT's compiled sampler on the same CSR, including when the CSR starts
at a compressed id offset.
"""

import torch
from absl.testing import absltest
from graphlearn_torch import py_graphlearn_torch as pywrap
from parameterized import param, parameterized

from gigl.distributed.utils.native_neighbor_lookup import (
    lookup_local,
    register_partition_csr,
)
from gigl.scripts.verify_glt_patches import build_cpu_csr_graph
from gigl.utils.csr import to_local_compressed_ids
from tests.test_assets.test_case import TestCase

_SAMPLER_ID = 11
_NUM_ROWS = 40
_NUM_EDGES = 300
_COMPRESSED_ID_OFFSET = 25


def _random_csr(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """An int64 CSR with ``_NUM_ROWS`` rows, some of them empty, and distinct edge ids in no particular order.

    Returns:
        The row pointers, the column ids and the edge ids.
    """
    generator = torch.Generator().manual_seed(seed)
    # Squaring the draws skews the rows toward low ids, so the last rows are empty and the degrees vary.
    rows = (torch.rand(_NUM_EDGES, generator=generator) ** 2 * (_NUM_ROWS - 5)).long()
    rows, _ = torch.sort(rows)
    columns = torch.randint(0, 1000, (_NUM_EDGES,), generator=generator)
    row_degrees = torch.bincount(rows, minlength=_NUM_ROWS)
    row_pointers = torch.zeros(_NUM_ROWS + 1, dtype=torch.int64)
    torch.cumsum(row_degrees, dim=0, out=row_pointers[1:])
    edge_ids = torch.randperm(_NUM_EDGES, generator=generator)
    return row_pointers, columns, edge_ids


class NativeLookupIndptrDtypeTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        import gigl_core.neighbor_lookup as native

        self.native = native

    def tearDown(self) -> None:
        self.native.unregister_csr(_SAMPLER_ID)
        super().tearDown()

    @parameterized.expand(
        [
            param(
                f"{indptr_name}_indptr_{indices_name}_indices_{offset_name}",
                indptr_dtype=indptr_dtype,
                indices_dtype=indices_dtype,
                compressed_id_offset=compressed_id_offset,
            )
            for indptr_name, indptr_dtype in (
                ("int32", torch.int32),
                ("int64", torch.int64),
            )
            for indices_name, indices_dtype in (
                ("int32", torch.int32),
                ("int64", torch.int64),
            )
            for offset_name, compressed_id_offset in (
                ("offset_zero", 0),
                ("offset_nonzero", _COMPRESSED_ID_OFFSET),
            )
        ]
    )
    def test_the_lookup_matches_glt_sampler(
        self,
        _name: str,
        indptr_dtype: torch.dtype,
        indices_dtype: torch.dtype,
        compressed_id_offset: int,
    ) -> None:
        row_pointers, columns, edge_ids = _random_csr(seed=3)
        row_pointers = row_pointers.to(indptr_dtype)
        columns = columns.to(indices_dtype)
        # The sampler borrows the graph's storage, so the graph must outlive it.
        graph = build_cpu_csr_graph(row_pointers, columns, edge_ids)
        glt_sampler = pywrap.CPURandomSampler(graph.graph_handler)
        self.native.register_csr(
            _SAMPLER_ID,
            "homogeneous_edge",
            row_pointers,
            columns,
            edge_ids,
            compressed_id_offset,
            True,
        )
        max_degree = int(torch.diff(row_pointers.long()).max())
        # Ids below, inside (including the empty last rows) and past the rows the CSR holds.
        global_ids = torch.cat(
            [
                torch.arange(
                    compressed_id_offset - 3, compressed_id_offset + _NUM_ROWS + 3
                ),
                torch.tensor([compressed_id_offset + 7] * 2),
            ]
        ).clamp(min=0)
        local_ids = to_local_compressed_ids(global_ids, compressed_id_offset, _NUM_ROWS)
        # A fan-out above every degree copies each row in stored order, so GLT's draws are not involved.
        for requested_neighbors in (max_degree + 1, -1):
            expected_neighbors, expected_counts, expected_edge_ids = (
                glt_sampler.sample_with_edge(local_ids, requested_neighbors)
            )
            output = lookup_local(_SAMPLER_ID, None, global_ids, requested_neighbors)
            self.assertEqual(output.nbr_num.tolist(), expected_counts.tolist())
            self.assertEqual(output.nbr.tolist(), expected_neighbors.tolist())
            self.assertEqual(output.edge.tolist(), expected_edge_ids.tolist())
        # The comparison is not vacuous: rows in range have neighbors, and ids outside the rows have none.
        self.assertGreater(int(output.nbr_num.sum()), 0)
        outside_rows = (global_ids < compressed_id_offset) | (
            global_ids >= compressed_id_offset + _NUM_ROWS
        )
        self.assertTrue(bool(outside_rows.any()))
        self.assertEqual(int(output.nbr_num[outside_rows].sum()), 0)

    @parameterized.expand(
        [param("int32", dtype=torch.int32), param("int64", dtype=torch.int64)]
    )
    def test_a_long_row_is_sampled_from_its_own_neighbors(
        self, _name: str, dtype: torch.dtype
    ) -> None:
        row_pointers, columns, edge_ids = _random_csr(seed=5)
        self.native.register_csr(
            _SAMPLER_ID,
            "homogeneous_edge",
            row_pointers.to(dtype),
            columns,
            edge_ids,
            _COMPRESSED_ID_OFFSET,
            True,
        )
        degrees = torch.diff(row_pointers)
        long_row = int(degrees.argmax())
        requested_neighbors = int(degrees[long_row]) - 1
        output = lookup_local(
            _SAMPLER_ID,
            None,
            torch.tensor([_COMPRESSED_ID_OFFSET + long_row]),
            requested_neighbors,
        )
        row_start, row_end = (
            int(row_pointers[long_row]),
            int(row_pointers[long_row + 1]),
        )
        self.assertEqual(output.nbr_num.tolist(), [requested_neighbors])
        # Each sampled (neighbor, edge id) pair is one of the row's stored edges.
        stored_edges = set(
            zip(
                columns[row_start:row_end].tolist(),
                edge_ids[row_start:row_end].tolist(),
            )
        )
        self.assertTrue(
            set(zip(output.nbr.tolist(), output.edge.tolist())) <= stored_edges
        )

    @parameterized.expand(
        [param("int32", dtype=torch.int32), param("int64", dtype=torch.int64)]
    )
    def test_a_partition_with_this_indptr_dtype_registers_for_native_lookups(
        self, _name: str, dtype: torch.dtype
    ) -> None:
        row_pointers, columns, edge_ids = _random_csr(seed=7)
        graph = build_cpu_csr_graph(row_pointers.to(dtype), columns, edge_ids)
        self.assertTrue(
            register_partition_csr(
                _SAMPLER_ID,
                graph,
                with_edge=True,
                with_weight=False,
                device=torch.device("cpu"),
                seed=None,
            )
        )
        output = lookup_local(_SAMPLER_ID, None, torch.tensor([0]), -1)
        self.assertEqual(int(output.nbr_num[0]), int(row_pointers[1] - row_pointers[0]))

    def test_an_indptr_of_another_dtype_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            self.native.register_csr(
                _SAMPLER_ID,
                "homogeneous_edge",
                torch.tensor([0, 1], dtype=torch.int16),
                torch.tensor([4]),
                None,
                0,
                False,
            )


if __name__ == "__main__":
    absltest.main()
