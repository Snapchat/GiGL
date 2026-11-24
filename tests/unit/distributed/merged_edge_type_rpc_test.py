"""Tests for fetching several edge types' neighbors with one RPC per remote partition.

The worker functions run one process per partition, the way ``remote_hop_routing_test`` does, so frontiers
really route to a remote partition. Each check runs with the native neighbor lookup on, where a merged request is a
TorchScript RPC over the C++ lookup, and off, where it is GLT's Python RPC callee.
"""

import contextlib
import os
import signal
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Union
from unittest import mock

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.channel import ShmChannel
from graphlearn_torch.distributed import (
    barrier,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from graphlearn_torch.sampler import NeighborOutput
from graphlearn_torch.typing import EdgeType, NodeType
from parameterized import param, parameterized
from torch_geometric.data import HeteroData

from gigl.distributed import base_sampler
from gigl.distributed.base_sampler import (
    BaseDistNeighborSampler,
    _split_requests_by_byte_budget,
)
from gigl.distributed.constants import NATIVE_NEIGHBOR_LOOKUP_ENV
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.dist_ppr_sampler import DistPPRNeighborSampler
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.sampler_options import PPRSamplerOptions
from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.sampling_errors import (
    SAMPLING_ERROR_KEY,
    raise_if_sampling_error,
)
from gigl.types.graph import GraphPartitionData, PartitionOutput
from tests.test_assets.distributed.test_dataset import (
    STORY_TO_USER,
    USER,
    USER_TO_STORY,
    create_heterogeneous_dataset,
    create_homogeneous_dataset,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

_NUM_NODES_PER_TYPE = 8
_NUM_NEIGHBORS = 5
_USER_BUYS_ITEM: EdgeType = ("user", "buys", "item")
_ITEM_BOUGHT_BY_USER: EdgeType = ("item", "bought_by", "user")
# Only even users, plus user 1, have these edges, so most odd users have no follows edges on partition 1.
_USER_FOLLOWS_USER: EdgeType = ("user", "follows", "user")
_EDGE_TYPES = [_USER_BUYS_ITEM, _ITEM_BOUGHT_BY_USER, _USER_FOLLOWS_USER]
_TEST_TIMEOUT_SECONDS = 120
_UNKNOWN_SAMPLER_ID = 10_000


def _global_edge_indices() -> dict[EdgeType, torch.Tensor]:
    """The full graph. Every node ``i`` has at most ``_NUM_NEIGHBORS`` neighbors, so sampling is exhaustive."""
    users = torch.arange(_NUM_NODES_PER_TYPE)
    even_users = users[users % 2 == 0]
    followers = torch.cat([even_users, torch.tensor([1])])
    return {
        _USER_BUYS_ITEM: torch.stack(
            [
                torch.cat([users, users]),
                torch.cat(
                    [
                        (users + 1) % _NUM_NODES_PER_TYPE,
                        (users + 3) % _NUM_NODES_PER_TYPE,
                    ]
                ),
            ]
        ),
        _ITEM_BOUGHT_BY_USER: torch.stack([users, (users + 2) % _NUM_NODES_PER_TYPE]),
        _USER_FOLLOWS_USER: torch.stack(
            [followers, (followers + 2) % _NUM_NODES_PER_TYPE]
        ),
    }


def _partition_dataset(rank: int) -> DistDataset:
    """One of two partitions; node ``i`` of every type, and its out-edges, live on partition ``i % 2``."""
    node_partition_book = {
        node_type: torch.arange(_NUM_NODES_PER_TYPE) % 2
        for node_type in ("user", "item")
    }
    edge_partition_book: dict[EdgeType, torch.Tensor] = {}
    partitioned_edge_index: dict[EdgeType, GraphPartitionData] = {}
    for edge_type, edge_index in _global_edge_indices().items():
        edge_ids = torch.arange(edge_index.size(1))
        owned = edge_index[0] % 2 == rank
        edge_partition_book[edge_type] = edge_index[0] % 2
        partitioned_edge_index[edge_type] = GraphPartitionData(
            edge_index=edge_index[:, owned], edge_ids=edge_ids[owned]
        )
    partition_output = PartitionOutput(
        node_partition_book=node_partition_book,
        edge_partition_book=edge_partition_book,
        partitioned_edge_index=partitioned_edge_index,
        partitioned_node_features=None,
        partitioned_edge_features=None,
        partitioned_positive_labels=None,
        partitioned_negative_labels=None,
        partitioned_node_labels=None,
    )
    dataset = DistDataset(rank=rank, world_size=2, edge_dir="out")
    dataset.build(partition_output=partition_output)
    return dataset


def _frontiers(rank: int) -> dict[str, dict[EdgeType, torch.Tensor]]:
    """Frontiers per scenario for one rank. Odd ids are on partition 1, even ids on partition 0."""
    empty = torch.tensor([], dtype=torch.long)
    return {
        "mixed_local_and_remote": {
            _USER_BUYS_ITEM: torch.tensor([0, 1, 2, 5]),
            _ITEM_BOUGHT_BY_USER: torch.tensor([3, 4, 7]),
            _USER_FOLLOWS_USER: torch.tensor([0, 2, 6]),
        },
        "one_edge_type_empty": {
            _USER_BUYS_ITEM: torch.tensor([1, 2, 6]),
            _ITEM_BOUGHT_BY_USER: empty,
            _USER_FOLLOWS_USER: torch.tensor([4, 5]),
        },
        "all_edge_types_empty": {edge_type: empty for edge_type in _EDGE_TYPES},
        # Rank 0 asks partition 1 about users 3, 5 and 7, which have no follows edges, so it answers with
        # zero neighbors for them.
        "sources_without_edges_on_remote_partition": {
            _USER_BUYS_ITEM: torch.tensor([3 - 3 * rank, 5 - 5 * rank]),
            _ITEM_BOUGHT_BY_USER: torch.tensor([1 - rank]),
            _USER_FOLLOWS_USER: torch.tensor([3, 5, 7])
            if rank == 0
            else torch.tensor([0]),
        },
        "only_own_partition": {
            _USER_BUYS_ITEM: torch.tensor([rank, rank + 2]),
            _ITEM_BOUGHT_BY_USER: torch.tensor([rank + 4]),
            _USER_FOLLOWS_USER: torch.tensor([rank + 6]),
        },
    }


def _output_as_lists(output: Any) -> Optional[tuple[list[int], list[int], list[int]]]:
    """Neighbors, counts and edge ids of a hop result as lists; ``None`` for GLT's empty-result placeholder."""
    if not isinstance(output, NeighborOutput):
        assert output.numel() == 0
        return None
    assert output.edge is not None
    return output.nbr.tolist(), output.nbr_num.tolist(), output.edge.tolist()


# The functions a hop calls to send a request to another partition: GLT's Python callee, and the native lookup's
# TorchScript RPCs for one edge type and for several.
_PYTHON_REQUEST_FUNCTIONS = ("rpc_request_async",)
_NATIVE_REQUEST_FUNCTIONS = ("lookup_remote", "lookup_remote_multi")


class _RpcCounter:
    """Records which request function a hop calls, once per RPC it sends."""

    def __init__(self) -> None:
        self.function_names: list[str] = []

    @property
    def num_requests(self) -> int:
        return len(self.function_names)


def _counted(
    function_name: str, counter: _RpcCounter, real_function: Callable[..., Any]
) -> Callable[..., Any]:
    def counted(*args: Any, **kwargs: Any) -> Any:
        counter.function_names.append(function_name)
        return real_function(*args, **kwargs)

    return counted


@contextmanager
def _counting_requests() -> Iterator[_RpcCounter]:
    """Counts the RPCs sent through any of the request functions in ``base_sampler`` while the block runs."""
    counter = _RpcCounter()
    with contextlib.ExitStack() as stack:
        for function_name in _PYTHON_REQUEST_FUNCTIONS + _NATIVE_REQUEST_FUNCTIONS:
            counted = _counted(
                function_name, counter, getattr(base_sampler, function_name)
            )
            stack.enter_context(mock.patch.object(base_sampler, function_name, counted))
        yield counter


def _init_rpc(rank: int, master_port: int) -> None:
    init_worker_group(world_size=2, rank=rank)
    init_rpc(master_addr="127.0.0.1", master_port=master_port, num_rpc_threads=4)


def _build_sampler(
    rank: int, native: bool, rank_cannot_serve: Optional[int] = None
) -> BaseDistNeighborSampler:
    """A started sampler on the CPU with the native neighbor lookup switched on or off for its construction.

    Constructing a sampler is a collective, so every rank must build the same samplers in the same order. With
    ``rank_cannot_serve`` set, that rank fails to register its CSR, as a rank with a GPU sampler would.
    """
    environment = {NATIVE_NEIGHBOR_LOOKUP_ENV: "on" if native else "off"}
    cannot_serve = mock.patch.object(
        base_sampler, "register_partition_csr", return_value=False
    )
    with mock.patch.dict(os.environ, environment):
        if rank == rank_cannot_serve:
            cannot_serve.start()
        try:
            sampler = BaseDistNeighborSampler(
                data=_partition_dataset(rank),
                num_neighbors={
                    edge_type: [_NUM_NEIGHBORS] for edge_type in _EDGE_TYPES
                },
                with_edge=True,
                concurrency=1,
                device=torch.device("cpu"),
            )
        finally:
            if rank == rank_cannot_serve:
                cannot_serve.stop()
    sampler.start_loop()
    return sampler


def _start_sampler(
    rank: int,
    master_port: int,
    native: bool,
    rank_cannot_serve: Optional[int] = None,
) -> BaseDistNeighborSampler:
    _init_rpc(rank, master_port)
    sampler = _build_sampler(rank, native, rank_cannot_serve)
    barrier()
    return sampler


def _run_with_timeout(function: Callable[[], None]) -> None:
    """Runs ``function``, failing instead of hanging if it takes more than the test timeout."""

    def on_timeout(*_: Any) -> None:
        raise TimeoutError(f"Still running after {_TEST_TIMEOUT_SECONDS} seconds.")

    signal.signal(signal.SIGALRM, on_timeout)
    signal.alarm(_TEST_TIMEOUT_SECONDS)
    try:
        function()
    finally:
        signal.alarm(0)


def _check_merged_matches_unmerged(
    rank: int,
    master_port: int,
    native: bool,
    max_rpc_bytes: int,
    expected_requests: dict[str, Optional[int]],
    rank_cannot_serve: Optional[int] = None,
) -> None:
    """Merged and unmerged hops agree for every scenario, and the merged hop sends the expected RPCs.

    ``native`` is the setting of ``GIGL_NATIVE_NEIGHBOR_LOOKUP``. The hops must use the native lookup only when it is
    on and every rank can serve it.
    """

    def check() -> None:
        sampler = _start_sampler(rank, master_port, native, rank_cannot_serve)
        uses_native_lookup = native and rank_cannot_serve is None
        assert sampler._native_neighbor_lookup == uses_native_lookup
        expected_functions = set(_PYTHON_REQUEST_FUNCTIONS)
        if uses_native_lookup:
            expected_functions = set(_NATIVE_REQUEST_FUNCTIONS)
        for scenario, frontier in _frontiers(rank).items():
            with _counting_requests() as unmerged_counter:
                unmerged = {
                    edge_type: sampler.run_task(
                        sampler._sample_one_hop(ids, _NUM_NEIGHBORS, edge_type)
                    )
                    for edge_type, ids in frontier.items()
                }
            with _counting_requests() as merged_counter:
                merged = sampler.run_task(
                    sampler._sample_one_hop_multi_etype(
                        frontier, _NUM_NEIGHBORS, max_rpc_bytes
                    )
                )
            assert set(merged) == set(unmerged), scenario
            for edge_type in frontier:
                assert _output_as_lists(merged[edge_type]) == _output_as_lists(
                    unmerged[edge_type]
                ), f"{scenario} {edge_type}"
            expected = expected_requests[scenario]
            if expected is None:
                expected = unmerged_counter.num_requests
            assert merged_counter.num_requests == expected, (
                f"{scenario}: {merged_counter.num_requests} merged vs "
                f"{unmerged_counter.num_requests} unmerged requests"
            )
            assert set(merged_counter.function_names) <= expected_functions, scenario
        barrier()
        sampler.shutdown_loop()
        shutdown_rpc()

    _run_with_timeout(check)


def _check_native_merged_matches_python_merged(
    rank: int, master_port: int, max_rpc_bytes: int
) -> None:
    """A merged hop over the native lookup returns the same neighbors, counts and edge ids as over GLT's callee.

    Every node has at most ``_NUM_NEIGHBORS`` neighbors, so both paths return every neighbor in stored order.
    """

    def check() -> None:
        _init_rpc(rank, master_port)
        native_sampler = _build_sampler(rank, native=True)
        python_sampler = _build_sampler(rank, native=False)
        barrier()
        assert native_sampler._native_neighbor_lookup
        assert not python_sampler._native_neighbor_lookup
        for scenario, frontier in _frontiers(rank).items():
            outputs = {}
            counters = {}
            for name, sampler in (
                ("native", native_sampler),
                ("python", python_sampler),
            ):
                with _counting_requests() as counters[name]:
                    outputs[name] = sampler.run_task(
                        sampler._sample_one_hop_multi_etype(
                            frontier, _NUM_NEIGHBORS, max_rpc_bytes
                        )
                    )
            assert set(outputs["native"]) == set(outputs["python"]), scenario
            for edge_type in frontier:
                assert _output_as_lists(
                    outputs["native"][edge_type]
                ) == _output_as_lists(outputs["python"][edge_type]), (
                    f"{scenario} {edge_type}"
                )
            # The same number of RPCs, each over its own transport.
            assert counters["native"].num_requests == counters["python"].num_requests
            assert set(counters["native"].function_names) <= {
                "lookup_remote_multi",
                "lookup_remote",
            }
            assert set(counters["python"].function_names) <= {"rpc_request_async"}
        barrier()
        native_sampler.shutdown_loop()
        python_sampler.shutdown_loop()
        shutdown_rpc()

    _run_with_timeout(check)


def _to_unknown_sampler(
    real_function: Callable[..., Any],
) -> Callable[..., Any]:
    """Wraps a request function so it addresses a sampler id that GLT's callee table and the native registry lack."""

    def request_unknown_sampler(
        to_worker: Any, _sampler_id: int, *args: Any, **kwargs: Any
    ) -> Any:
        return real_function(to_worker, _UNKNOWN_SAMPLER_ID, *args, **kwargs)

    return request_unknown_sampler


def _check_failed_remote_call_raises(
    rank: int, master_port: int, merged: bool, native: bool
) -> None:
    """A remote call that fails on the server raises on the caller instead of hanging."""

    def check() -> None:
        sampler = _start_sampler(rank, master_port, native)
        assert sampler._native_neighbor_lookup == native
        frontier = _frontiers(rank)["mixed_local_and_remote"]
        with contextlib.ExitStack() as stack:
            for function_name in _PYTHON_REQUEST_FUNCTIONS + _NATIVE_REQUEST_FUNCTIONS:
                stack.enter_context(
                    mock.patch.object(
                        base_sampler,
                        function_name,
                        _to_unknown_sampler(getattr(base_sampler, function_name)),
                    )
                )
            if merged:
                coroutine = sampler._sample_one_hop_multi_etype(
                    frontier, _NUM_NEIGHBORS
                )
            else:
                coroutine = sampler._sample_one_hop(
                    frontier[_USER_BUYS_ITEM], _NUM_NEIGHBORS, _USER_BUYS_ITEM
                )
            raised = False
            try:
                sampler.run_task(coroutine)
            except Exception:
                raised = True
            assert raised, "a failed remote call did not raise"
        barrier()
        sampler.shutdown_loop()
        shutdown_rpc()

    _run_with_timeout(check)


def _check_oversized_batch_reaches_consumer(_: int, master_port: int) -> None:
    """A batch too big for the channel reaches the consumer as an error, and the channel stays usable."""

    def check() -> None:
        init_worker_group(world_size=1, rank=0)
        init_rpc(master_addr="127.0.0.1", master_port=master_port, num_rpc_threads=2)
        channel = ShmChannel(capacity=4, shm_size="1MB")
        channel.max_message_bytes = 1024 * 1024
        sampler = BaseDistNeighborSampler(
            data=create_heterogeneous_dataset(
                edge_indices={USER_TO_STORY: torch.tensor([[0, 1], [0, 1]])}
            ),
            num_neighbors={USER_TO_STORY: [1]},
            channel=channel,
            concurrency=1,
        )
        sampler.start_loop()

        async def sample() -> None:
            return None

        async def collate_oversized_batch(_: Any) -> dict[str, torch.Tensor]:
            return {"batch": torch.zeros(2 * 1024 * 1024, dtype=torch.uint8)}

        with mock.patch.object(sampler, "_collate_fn", collate_oversized_batch):
            sampler.run_task(sampler._send_adapter(sample))
        message = channel.recv(timeout_ms=30_000)
        assert SAMPLING_ERROR_KEY in message
        raised = False
        try:
            raise_if_sampling_error(message)
        except RuntimeError as error:
            raised = "cannot fit in the sampling channel" in str(error)
        assert raised
        sampler.shutdown_loop()
        shutdown_rpc()

    _run_with_timeout(check)


def _fetch_as_lists(
    fetched: dict[int, Any],
) -> dict[int, tuple[list[int], list[int], list[int], Optional[list[int]]]]:
    return {
        edge_type_id: (
            tensors.node_ids.tolist(),
            tensors.flat_neighbor_ids.tolist(),
            tensors.counts.tolist(),
            None if tensors.edge_ids is None else tensors.edge_ids.tolist(),
        )
        for edge_type_id, tensors in fetched.items()
    }


def _check_ppr_fetch_merged_matches_unmerged(
    rank: int, master_port: int, native: bool
) -> None:
    """The PPR fetch returns the same neighbors, counts and edge ids merged and unmerged."""

    def check() -> None:
        _init_rpc(rank, master_port)
        dataset = _partition_dataset(rank)
        degree_tensors = {
            node_type: torch.ones(_NUM_NODES_PER_TYPE, dtype=torch.int32)
            for node_type in ("user", "item")
        }
        environment = {NATIVE_NEIGHBOR_LOOKUP_ENV: "on" if native else "off"}
        with mock.patch.dict(os.environ, environment):
            samplers = [
                DistPPRNeighborSampler(
                    data=dataset,
                    num_neighbors={
                        edge_type: [_NUM_NEIGHBORS] for edge_type in _EDGE_TYPES
                    },
                    with_edge=True,
                    concurrency=1,
                    device=torch.device("cpu"),
                    degree_tensors=degree_tensors,
                    num_neighbors_per_hop=_NUM_NEIGHBORS,
                    include_sampled_edges=True,
                    merge_edge_type_rpcs=merge,
                    # 8 bytes forces one request per edge type, so splitting is exercised too.
                    merge_edge_type_rpc_max_bytes=8 if merge else 512 * 1024 * 1024,
                )
                for merge in (False, True)
            ]
        for sampler in samplers:
            assert sampler._native_neighbor_lookup == native
            sampler.start_loop()
        barrier()
        unmerged_sampler, merged_sampler = samplers
        for scenario, frontier in _frontiers(rank).items():
            # As in a real fetch round, only edge types with a non-empty frontier are requested.
            nodes_by_edge_type_id = {
                unmerged_sampler._etype_to_etype_id[edge_type]: ids
                for edge_type, ids in frontier.items()
                if ids.numel() > 0
            }
            if not nodes_by_edge_type_id:
                continue
            results = [
                _fetch_as_lists(
                    sampler.run_task(
                        sampler._batch_fetch_neighbors(nodes_by_edge_type_id)
                    )
                )
                for sampler in samplers
            ]
            assert results[0] == results[1], scenario
        barrier()
        for sampler in samplers:
            sampler.shutdown_loop()
        shutdown_rpc()

    _run_with_timeout(check)


def _flatten_batch(datum: Any) -> dict[str, torch.Tensor]:
    """Every tensor of a loader batch, keyed by ``store.attribute`` (``attribute`` for homogeneous data)."""
    if isinstance(datum, HeteroData):
        return {
            f"{store}.{key}": value
            for store, store_dict in datum.to_dict().items()
            for key, value in store_dict.items()
            if isinstance(value, torch.Tensor)
        }
    return {
        key: value
        for key, value in datum.to_dict().items()
        if isinstance(value, torch.Tensor)
    }


def _run_ppr_loader(_: int, hetero: bool, typed: bool) -> None:
    """On one partition, the PPR loader's batches do not depend on ``merge_edge_type_rpcs``.

    Batches arrive in no fixed order and forward push scores differ in the last bits between runs, so batches
    are matched by seed and scores by tolerance; every other tensor must be equal.
    """
    create_test_process_group()
    if hetero:
        dataset = create_heterogeneous_dataset(
            edge_indices={
                USER_TO_STORY: torch.tensor([[0, 0, 1, 2, 3], [0, 1, 1, 2, 0]]),
                STORY_TO_USER: torch.tensor([[0, 1, 1, 2], [1, 2, 3, 0]]),
            },
            edge_dir="out",
        )
        input_nodes: Union[torch.Tensor, tuple[NodeType, torch.Tensor]] = (
            USER,
            torch.arange(4),
        )
        typed_channel_ratios: Optional[dict] = (
            {USER_TO_STORY: 0.5, STORY_TO_USER: 0.5} if typed else None
        )
        seed_key = f"{USER}.batch"
    else:
        dataset = create_homogeneous_dataset(
            edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
        )
        input_nodes = torch.arange(3)
        typed_channel_ratios = None
        seed_key = "batch"
    batches_by_flag: dict[bool, dict[int, dict[str, torch.Tensor]]] = {}
    for merge_edge_type_rpcs in (False, True):
        loader = DistNeighborLoader(
            dataset=dataset,
            input_nodes=input_nodes,
            num_neighbors=[],
            sampler_options=PPRSamplerOptions(
                eps=1e-6,
                max_ppr_nodes=3,
                typed_channel_ratios=typed_channel_ratios,
                include_sampled_edges=hetero,
                merge_edge_type_rpcs=merge_edge_type_rpcs,
            ),
            pin_memory_device=torch.device("cpu"),
            batch_size=1,
        )
        batches_by_flag[merge_edge_type_rpcs] = {
            int(flat[seed_key][0]): flat
            for flat in (_flatten_batch(datum) for datum in loader)
        }
    unmerged, merged = batches_by_flag[False], batches_by_flag[True]
    assert len(unmerged) == len(input_nodes[1] if hetero else input_nodes)
    assert set(unmerged) == set(merged)
    for seed, unmerged_batch in unmerged.items():
        assert set(unmerged_batch) == set(merged[seed]), seed
        for key, unmerged_tensor in unmerged_batch.items():
            if unmerged_tensor.is_floating_point():
                torch.testing.assert_close(
                    merged[seed][key], unmerged_tensor, atol=1e-5, rtol=0
                )
            else:
                assert torch.equal(merged[seed][key], unmerged_tensor), (seed, key)
    shutdown_rpc()


class SplitRequestsByByteBudgetTest(TestCase):
    @parameterized.expand(
        [
            param(
                "requests_that_fit_the_budget_are_one_group",
                sizes=[2, 3, 1],
                num_nbr=2,
                with_edge=False,
                max_bytes=10_000,
                expected=[[0, 1, 2]],
            ),
            param(
                "a_request_that_crosses_the_budget_starts_a_new_group",
                # 8 bytes per id; with no neighbors the response is one count per id, also 8 bytes.
                sizes=[2, 1, 3],
                num_nbr=0,
                with_edge=False,
                max_bytes=24,
                expected=[[0, 1], [2]],
            ),
            param(
                "the_response_estimate_splits_even_when_the_requests_fit",
                # One id answers with 1 count + 4 neighbors = 5 int64s = 40 bytes.
                sizes=[1, 1, 1],
                num_nbr=4,
                with_edge=False,
                max_bytes=80,
                expected=[[0, 1], [2]],
            ),
            param(
                "edge_ids_double_the_neighbors_in_the_response_estimate",
                # One id answers with 1 + 4 * 2 = 9 int64s = 72 bytes.
                sizes=[1, 1],
                num_nbr=4,
                with_edge=True,
                max_bytes=100,
                expected=[[0], [1]],
            ),
            param(
                "a_request_over_the_budget_is_its_own_group",
                sizes=[1, 100, 1],
                num_nbr=0,
                with_edge=False,
                max_bytes=64,
                expected=[[0], [1], [2]],
            ),
            param(
                "all_neighbors_has_no_bound_so_each_request_is_its_own_group",
                sizes=[1, 1, 1],
                num_nbr=-1,
                with_edge=False,
                max_bytes=10_000,
                expected=[[0], [1], [2]],
            ),
            param(
                "no_requests_give_no_groups",
                sizes=[],
                num_nbr=2,
                with_edge=False,
                max_bytes=10,
                expected=[],
            ),
        ]
    )
    def test_requests_are_grouped_in_order_within_the_byte_budget(
        self,
        _name: str,
        sizes: list[int],
        num_nbr: int,
        with_edge: bool,
        max_bytes: int,
        expected: list[list[int]],
    ) -> None:
        request_ids = [torch.zeros(size, dtype=torch.long) for size in sizes]
        self.assertEqual(
            _split_requests_by_byte_budget(request_ids, num_nbr, with_edge, max_bytes),
            expected,
        )


class MergedHopTest(TestCase):
    """``_sample_one_hop_multi_etype`` on two partitions."""

    @parameterized.expand(
        [
            param("native_lookup", native=True),
            param("python_callee", native=False),
        ]
    )
    def test_a_merged_hop_matches_per_edge_type_hops_with_one_rpc_per_remote_partition(
        self, _name: str, native: bool
    ) -> None:
        # A scenario with any remote id sends one merged RPC to the other partition, and none otherwise.
        expected_requests = {
            "mixed_local_and_remote": 1,
            "one_edge_type_empty": 1,
            "all_edge_types_empty": 0,
            "sources_without_edges_on_remote_partition": 1,
            "only_own_partition": 0,
        }
        mp.spawn(
            _check_merged_matches_unmerged,
            args=(get_free_port(), native, 512 * 1024 * 1024, expected_requests),
            nprocs=2,
        )

    @parameterized.expand(
        [
            param("native_lookup", native=True),
            param("python_callee", native=False),
        ]
    )
    def test_a_tiny_byte_budget_splits_into_per_edge_type_rpcs_with_the_same_results(
        self, _name: str, native: bool
    ) -> None:
        # With a one-byte budget every edge type is its own request, so the RPC count equals the unmerged one.
        expected_requests = {scenario: None for scenario in _frontiers(0)}
        mp.spawn(
            _check_merged_matches_unmerged,
            args=(get_free_port(), native, 1, expected_requests),
            nprocs=2,
        )

    @parameterized.expand(
        [
            param("default_budget", max_rpc_bytes=512 * 1024 * 1024),
            param("one_request_per_edge_type", max_rpc_bytes=1),
        ]
    )
    def test_the_native_merged_hop_returns_the_same_neighbors_and_edge_ids_as_the_python_one(
        self, _name: str, max_rpc_bytes: int
    ) -> None:
        mp.spawn(
            _check_native_merged_matches_python_merged,
            args=(get_free_port(), max_rpc_bytes),
            nprocs=2,
        )

    @parameterized.expand(
        [
            param("rank_0_cannot_serve", rank_cannot_serve=0),
            param("rank_1_cannot_serve", rank_cannot_serve=1),
        ]
    )
    def test_a_merged_hop_falls_back_to_the_python_callee_when_a_rank_is_not_registered(
        self, _name: str, rank_cannot_serve: int
    ) -> None:
        # One rank failing to register turns the native lookup off on every rank, so the merged hop is the
        # Python callee, with the same results and RPC counts as before.
        expected_requests = {
            "mixed_local_and_remote": 1,
            "one_edge_type_empty": 1,
            "all_edge_types_empty": 0,
            "sources_without_edges_on_remote_partition": 1,
            "only_own_partition": 0,
        }
        mp.spawn(
            _check_merged_matches_unmerged,
            args=(
                get_free_port(),
                True,
                512 * 1024 * 1024,
                expected_requests,
                rank_cannot_serve,
            ),
            nprocs=2,
        )

    @parameterized.expand(
        [
            param("merged_hop_native", merged=True, native=True),
            param("merged_hop_python", merged=True, native=False),
            param("per_edge_type_hop_native", merged=False, native=True),
            param("per_edge_type_hop_python", merged=False, native=False),
        ]
    )
    def test_a_failed_remote_call_raises_on_the_caller(
        self, _name: str, merged: bool, native: bool
    ) -> None:
        mp.spawn(
            _check_failed_remote_call_raises,
            args=(get_free_port(), merged, native),
            nprocs=2,
        )


class PprMergedFetchTest(TestCase):
    @parameterized.expand(
        [
            param("native_lookup", native=True),
            param("python_callee", native=False),
        ]
    )
    def test_the_ppr_fetch_returns_the_same_tensors_merged_and_unmerged(
        self, _name: str, native: bool
    ) -> None:
        mp.spawn(
            _check_ppr_fetch_merged_matches_unmerged,
            args=(get_free_port(), native),
            nprocs=2,
        )

    @parameterized.expand(
        [
            param("homogeneous_graph", hetero=False, typed=False),
            param(
                "heterogeneous_graph_with_untyped_channels", hetero=True, typed=False
            ),
            param("heterogeneous_graph_with_typed_channels", hetero=True, typed=True),
        ]
    )
    def test_the_ppr_loader_returns_the_same_batches_with_merging_on(
        self, _name: str, hetero: bool, typed: bool
    ) -> None:
        mp.spawn(_run_ppr_loader, args=(hetero, typed), nprocs=1)


class OversizedBatchTest(TestCase):
    def test_a_batch_larger_than_the_channel_reaches_the_consumer_as_an_error(
        self,
    ) -> None:
        mp.spawn(
            _check_oversized_batch_reaches_consumer, args=(get_free_port(),), nprocs=1
        )


if __name__ == "__main__":
    absltest.main()
