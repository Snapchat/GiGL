"""Runs seeded CPU inducer batches and saves every output tensor.

Executed in a child process by ``glt_inducer_impl_test.py`` because ``GIGL_GLT_INDUCER_IMPL`` is read
once per process. Usage: ``python -m tests.unit.utils.glt_inducer_scenarios <output.pt>``.
"""

import sys

import torch
from graphlearn_torch import py_graphlearn_torch as pywrap

_INT64_MIN = -(2**63)
_SPECIAL_IDS = [
    0,
    -1,
    1,
    2**62,
    2**62 - 1,
    2**62 + 5,
    _INT64_MIN,
    _INT64_MIN + 1,
    2**63 - 1,
]
_HETERO_EDGE_TYPES = [("a", "to", "b"), ("b", "to", "a"), ("a", "to", "a")]


def _id_pool(generator: torch.Generator, size: int) -> torch.Tensor:
    """Small pool so ids repeat across seeds, neighbors and hops, plus the extreme ids."""
    pool = torch.randint(0, 10**9, (size,), generator=generator, dtype=torch.int64)
    pool[: len(_SPECIAL_IDS)] = torch.tensor(_SPECIAL_IDS, dtype=torch.int64)
    return pool


def _draw(generator: torch.Generator, pool: torch.Tensor, count: int) -> torch.Tensor:
    return pool[torch.randint(0, pool.numel(), (count,), generator=generator)]


def _neighbors_for(
    generator: torch.Generator, pool: torch.Tensor, num_srcs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Neighbors with duplicates and some sources with zero neighbors."""
    nbrs_num = torch.randint(0, 5, (num_srcs,), generator=generator, dtype=torch.int64)
    nbrs_num[torch.rand(num_srcs, generator=generator) < 0.3] = 0
    return _draw(generator, pool, int(nbrs_num.sum())), nbrs_num


def _homo_batches(
    generator: torch.Generator, pool: torch.Tensor, outputs: list
) -> None:
    inducer = pywrap.CPUInducer(64)
    # One inducer for every batch: init_node must fully reset the previous batch.
    for _ in range(30):
        seeds = _draw(
            generator, pool, int(torch.randint(0, 40, (1,), generator=generator))
        )
        srcs = inducer.init_node(seeds)
        outputs.append(srcs)
        for _ in range(3):
            nbrs, nbrs_num = _neighbors_for(generator, pool, srcs.numel())
            nodes, rows, cols = inducer.induce_next(srcs, nbrs, nbrs_num)
            outputs.extend([nodes, rows[: nbrs.numel()], cols[: nbrs.numel()]])
            # Next hop expands the newly found nodes plus a repeat of the previous sources.
            srcs = torch.cat([nodes, srcs[: srcs.numel() // 2]])


def _hetero_batches(
    generator: torch.Generator, pool: torch.Tensor, outputs: list
) -> None:
    inducer = pywrap.CPUHeteroInducer({"a": 64, "b": 64})
    for _ in range(30):
        seeds = {
            "a": _draw(
                generator, pool, int(torch.randint(1, 30, (1,), generator=generator))
            ),
            "b": _draw(
                generator, pool, int(torch.randint(0, 30, (1,), generator=generator))
            ),
        }
        srcs = inducer.init_node(seeds)
        for key in sorted(srcs):
            outputs.append(srcs[key])
        for _ in range(3):
            hetero_nbrs = {}
            for edge_type in _HETERO_EDGE_TYPES:
                src = srcs.get(edge_type[0])
                if src is None or src.numel() == 0:
                    continue
                nbrs, nbrs_num = _neighbors_for(generator, pool, src.numel())
                if nbrs.numel() > 0:
                    hetero_nbrs[edge_type] = (src, nbrs, nbrs_num)
            if not hetero_nbrs:
                break
            nodes, rows, cols = inducer.induce_next(hetero_nbrs)
            for key in sorted(nodes):
                outputs.append(nodes[key])
            for key in sorted(rows):
                outputs.extend([rows[key], cols[key]])
            srcs = nodes


def _stress(generator: torch.Generator, outputs: list) -> None:
    pool = torch.randint(0, 2**62, (700_000,), generator=generator, dtype=torch.int64)
    inducer = pywrap.CPUInducer(1000)
    srcs = inducer.init_node(_draw(generator, pool, 100_000))
    outputs.append(srcs)
    nbrs_num = torch.full((srcs.numel(),), 10, dtype=torch.int64)
    nbrs = _draw(generator, pool, 10 * srcs.numel())
    nodes, rows, cols = inducer.induce_next(srcs, nbrs, nbrs_num)
    outputs.extend([nodes, rows, cols])


def main(output_path: str) -> None:
    generator = torch.Generator().manual_seed(0)
    pool = _id_pool(generator, 60)
    outputs: list[torch.Tensor] = []
    _homo_batches(generator, pool, outputs)
    _hetero_batches(generator, pool, outputs)
    _stress(generator, outputs)
    torch.save({"impl": pywrap.inducer_impl(), "outputs": outputs}, output_path)


if __name__ == "__main__":
    main(sys.argv[1])
