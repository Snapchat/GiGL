"""Tests for ``gigl.distributed.utils.rpc_init``.

The file tests four layers:

1. Parsing the env value (``SamplingRpcChannelsFromEnvTest``).
2. GLT drift: the GLT source the copy was made from has not changed (``GltInitRpcDriftTest``).
3. Call-site wiring, with TensorPipe stopped before an agent starts (``CallSitesTest``).
4. Real two-process parity with GLT (``InitRpcMatchesGltTest``).
"""

import hashlib
import inspect
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any
from unittest import mock

import graphlearn_torch.distributed.rpc as glt_rpc
import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import (
    dist_context,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from graphlearn_torch.distributed.dist_context import DistRole
from torch.distributed import rpc

from gigl.distributed.constants import SAMPLING_RPC_CHANNELS_ENV
from gigl.distributed.dataset_factory import _build_dataset_process
from gigl.distributed.dist_sampling_producer import DistSamplingProducer
from gigl.distributed.utils.init_neighbor_loader_worker import (
    init_neighbor_loader_worker,
)
from gigl.distributed.utils.networking import get_free_port
from gigl.distributed.utils.rpc_init import (
    init_rpc_from_env,
    sampling_rpc_channels_from_env,
)
from tests.test_assets.test_case import TestCase

# sha256 of the concatenated inspect.getsource of graphlearn_torch.distributed.rpc's init_rpc,
# shutdown_rpc and rpc_is_initialized, which gigl.distributed.utils.rpc_init.init_rpc_with_options
# was copied from. Update it only after re-comparing the copy with the new GLT source.
_GLT_RPC_SOURCE_SHA256 = (
    "9d140e60ba6b86e4219655b727f434a7d0ca7aacdb8a06f82bc8f6610e0cedab"
)
_WORLD_SIZE = 2
_GROUP = "rpc_init_test"
# mpt_uv opens 16 extra connections per agent, each served by its own TP_UV_loop thread.
_MPT_UV_LANE_THREADS = 16
_NUM_RPC_THREADS = 4
_RPC_TIMEOUT_SECONDS = 90.0


class _StopInit(Exception):
    """Raised in place of ``torch.distributed.rpc.init_rpc``.

    A test sees the options that would have been used without starting an agent or opening a port.
    """


class _Echo(glt_rpc.RpcCalleeBase):
    """A GLT RPC callee that returns its input plus one, to prove a registered callee works across ranks."""

    def call(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 1


def _count_uv_threads() -> int:
    # TensorPipe names its libuv I/O threads "TP_UV_loop". A thread can exit between listing
    # /proc/self/task and opening its comm file, so a missing file is skipped.
    count = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                if f.read().strip() == "TP_UV_loop":
                    count += 1
        except OSError:
            continue
    return count


def _init_with_glt(**kwargs: Any) -> None:
    init_rpc(**kwargs)


def _init_with_gigl(**kwargs: Any) -> None:
    init_rpc_from_env(**kwargs)


def _run_rank(
    rank: int,
    port: int,
    init_function: Callable[..., None],
    result_queue: Any,
) -> None:
    """Runs one rank of a two-process RPC job and puts its observable state on ``result_queue``."""
    init_worker_group(world_size=_WORLD_SIZE, rank=rank, group_name=_GROUP)

    # Init: the number of TP_UV_loop threads this init added shows which channels TensorPipe set up.
    uv_threads_before = _count_uv_threads()
    init_function(
        master_addr="127.0.0.1",
        master_port=port,
        num_rpc_threads=_NUM_RPC_THREADS,
        rpc_timeout=_RPC_TIMEOUT_SECONDS,
    )
    peer = f"{_GROUP}_{1 - rank}"
    state: dict[str, Any] = {
        "initialized": glt_rpc.rpc_is_initialized(),
        "worker_names": glt_rpc.get_rpc_worker_names(),
        "group_worker_names": glt_rpc.get_rpc_current_group_worker_names(),
        "master_addr": glt_rpc._rpc_master_addr,
        "master_port": glt_rpc._rpc_master_port,
        "uv_threads": _count_uv_threads() - uv_threads_before,
        "thread_pool_size": rpc.api._get_current_rpc_agent().get_debug_info()[
            "agent.thread_pool_size"
        ],
        "rpc_timeout": rpc.get_rpc_timeout(),
    }

    # RPC checks: a request to the peer, an all_gather, a barrier and a global request.
    callee_id = glt_rpc.rpc_register(_Echo())
    state["callee_id"] = callee_id
    state["echo"] = glt_rpc.rpc_request(
        peer, callee_id, args=(torch.arange(4),)
    ).tolist()
    state["all_gather"] = glt_rpc.all_gather(rank)
    glt_rpc.barrier()
    state["global_request"] = glt_rpc.rpc_global_request(
        DistRole.WORKER, 1 - rank, torch.add, args=(1, 2)
    )

    # A second init while initialized returns without raising.
    init_function(master_addr="127.0.0.1", master_port=port)

    # Shutdown leaves the module state "None" (shut down), not "False" (never initialized).
    shutdown_rpc()
    state["after_shutdown"] = glt_rpc.rpc_is_initialized()

    # Re-init after shutdown raises.
    try:
        init_function(master_addr="127.0.0.1", master_port=port)
        state["reinit_error"] = ""
    except RuntimeError as e:
        state["reinit_error"] = str(e)
    result_queue.put((rank, state))


def _run_two_ranks(
    init_function: Callable[..., None], env: dict[str, str]
) -> dict[int, dict[str, Any]]:
    """Spawns two ranks that run ``_run_rank`` with ``env`` and returns their states by rank."""
    result_queue = mp.get_context("spawn").SimpleQueue()
    with mock.patch.dict(os.environ):
        os.environ.pop(SAMPLING_RPC_CHANNELS_ENV, None)
        os.environ.update(env)
        mp.spawn(
            fn=_run_rank,
            args=(get_free_port(), init_function, result_queue),
            nprocs=_WORLD_SIZE,
        )
    return dict(result_queue.get() for _ in range(_WORLD_SIZE))


@contextmanager
def _record_tensorpipe_options() -> Iterator[list[dict[str, Any]]]:
    """Yields the kwargs of every ``TensorPipeRpcBackendOptions`` built, and stops init before an agent starts."""
    recorded: list[dict[str, Any]] = []
    real_options = rpc.TensorPipeRpcBackendOptions

    def record_options(**kwargs: Any) -> Any:
        recorded.append(kwargs)
        return real_options(**kwargs)

    with (
        mock.patch.object(rpc, "TensorPipeRpcBackendOptions", record_options),
        mock.patch.object(rpc, "init_rpc", side_effect=_StopInit),
    ):
        yield recorded


class GltInitRpcDriftTest(TestCase):
    """The GLT source that ``init_rpc_with_options`` was copied from has not changed."""

    def test_glt_init_rpc_source_is_unchanged(self) -> None:
        source = "".join(
            inspect.getsource(function)
            for function in (
                glt_rpc.init_rpc,
                glt_rpc.shutdown_rpc,
                glt_rpc.rpc_is_initialized,
            )
        )
        digest = hashlib.sha256(source.encode()).hexdigest()
        self.assertEqual(
            digest,
            _GLT_RPC_SOURCE_SHA256,
            "graphlearn_torch's init_rpc, shutdown_rpc or rpc_is_initialized changed. Re-diff "
            "them against gigl.distributed.utils.rpc_init.init_rpc_with_options and port the "
            f"change. After a deliberate GLT upgrade, set _GLT_RPC_SOURCE_SHA256 to {digest}.",
        )


class CallSitesTest(TestCase):
    """Every call site that initializes RPC through ``init_rpc_from_env`` reads the channels variable."""

    def setUp(self) -> None:
        super().setUp()
        glt_rpc._rpc_inited = False
        self.addCleanup(setattr, glt_rpc, "_rpc_inited", False)
        self.addCleanup(
            setattr, dist_context, "_dist_context", dist_context._dist_context
        )

    def _call_init_rpc_from_env(self) -> None:
        init_worker_group(world_size=1, rank=0, group_name=_GROUP)
        init_rpc_from_env(master_addr="127.0.0.1", master_port=1)

    def _call_dataset_building_process(self) -> None:
        _build_dataset_process(
            process_number_on_current_machine=1,
            output_dict={},
            serialized_graph_metadata=mock.Mock(),
            master_ip_address="127.0.0.1",
            master_dataset_building_ports=(1, 2),
            node_rank=0,
            node_world_size=1,
            sample_edge_direction="out",
            should_load_tensors_in_parallel=False,
            partitioner_class=None,
            node_tf_dataset_options=mock.Mock(),
            edge_tf_dataset_options=mock.Mock(),
        )

    def _call_loader_worker(self) -> None:
        # The worker pins the CPU affinity and thread counts of its process; keep the test
        # process unchanged.
        with (
            mock.patch("psutil.Process"),
            mock.patch("torch.set_num_threads"),
            mock.patch("torch.set_num_interop_threads"),
        ):
            init_neighbor_loader_worker.__wrapped__(
                master_ip_address="127.0.0.1",
                local_process_rank=0,
                local_process_world_size=1,
                rank=0,
                world_size=1,
                master_worker_port=1,
                device=torch.device("cpu"),
                should_use_cpu_workers=True,
                num_cpu_threads=1,
            )

    def _call_sampling_producer_init(self) -> None:
        # init() reads the variable in the parent process before it spawns any worker, so an
        # object that skipped __init__ is enough.
        producer = object.__new__(DistSamplingProducer)
        producer.init()

    def test_invalid_channels_raise_at_each_call_site(self) -> None:
        # "mpt_uv" alone is invalid because it lacks "basic".
        for call in (
            self._call_init_rpc_from_env,
            self._call_dataset_building_process,
            self._call_loader_worker,
            self._call_sampling_producer_init,
        ):
            with self.subTest(call=call.__name__):
                with mock.patch.dict(os.environ, {SAMPLING_RPC_CHANNELS_ENV: "mpt_uv"}):
                    with self.assertRaises(ValueError):
                        call()

    def test_selected_channels_reach_tensorpipe_options(self) -> None:
        for call in (
            self._call_init_rpc_from_env,
            self._call_dataset_building_process,
            self._call_loader_worker,
        ):
            with self.subTest(call=call.__name__):
                with (
                    _record_tensorpipe_options() as recorded,
                    mock.patch.dict(
                        os.environ, {SAMPLING_RPC_CHANNELS_ENV: "cma,basic"}
                    ),
                ):
                    with self.assertRaises(_StopInit):
                        call()
                self.assertEqual(recorded[0]["_channels"], ["cma", "basic"])
                self.assertEqual(recorded[0]["_transports"], ["uv"])


class InitRpcMatchesGltTest(TestCase):
    """Two real processes that use ``init_rpc_from_env`` behave like ones that use GLT's ``init_rpc``."""

    def test_glt_channels_match_glt(self) -> None:
        glt_states = _run_two_ranks(_init_with_glt, {})
        gigl_states = _run_two_ranks(
            _init_with_gigl, {SAMPLING_RPC_CHANNELS_ENV: "mpt_uv,basic"}
        )

        for rank in range(_WORLD_SIZE):
            # The two implementations agree on every observable value, thread count included.
            self.assertEqual(gigl_states[rank], glt_states[rank])
            # The expected values are checked once, so a regression shared by both is caught.
            expected = glt_states[rank]
            self.assertTrue(expected["initialized"])
            self.assertEqual(
                expected["worker_names"][DistRole.WORKER],
                [f"{_GROUP}_0", f"{_GROUP}_1"],
            )
            self.assertEqual(
                expected["all_gather"], {f"{_GROUP}_0": 0, f"{_GROUP}_1": 1}
            )
            self.assertEqual(expected["echo"], [1, 2, 3, 4])
            self.assertEqual(expected["global_request"], 3)
            self.assertIsNone(expected["after_shutdown"])
            self.assertIn("re-init", expected["reinit_error"])
            self.assertEqual(expected["thread_pool_size"], str(_NUM_RPC_THREADS))
            self.assertEqual(expected["rpc_timeout"], _RPC_TIMEOUT_SECONDS)

    def test_basic_channel_skips_mpt_uv_threads(self) -> None:
        mpt_uv_states = _run_two_ranks(
            _init_with_gigl, {SAMPLING_RPC_CHANNELS_ENV: "mpt_uv,basic"}
        )
        basic_states = _run_two_ranks(
            _init_with_gigl, {SAMPLING_RPC_CHANNELS_ENV: "basic"}
        )

        for rank in range(_WORLD_SIZE):
            mpt_uv_state = dict(mpt_uv_states[rank])
            basic_state = dict(basic_states[rank])
            # Without the mpt_uv channel the agent has the lane threads fewer, and nothing else differs.
            self.assertEqual(
                mpt_uv_state.pop("uv_threads") - basic_state.pop("uv_threads"),
                _MPT_UV_LANE_THREADS,
            )
            self.assertEqual(basic_state, mpt_uv_state)


class SamplingRpcChannelsFromEnvTest(TestCase):
    """Parsing of the channels variable: defaults, whitespace trimming and invalid values."""

    def _parse(self, env: dict[str, str]) -> tuple[str, ...]:
        with mock.patch.dict(os.environ, env):
            return sampling_rpc_channels_from_env()

    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(SAMPLING_RPC_CHANNELS_ENV, None)

    def test_unset_gives_basic(self) -> None:
        self.assertEqual(self._parse({}), ("basic",))

    def test_valid_values_are_parsed_with_whitespace_trimmed(self) -> None:
        env = {SAMPLING_RPC_CHANNELS_ENV: "cma, mpt_uv ,basic"}
        self.assertEqual(self._parse(env), ("cma", "mpt_uv", "basic"))

    def test_invalid_values_are_rejected(self) -> None:
        for channels in (
            "",  # empty item
            "cuda_basic",  # CUDA channel
            "basic,cuda_ipc",  # CUDA channel
            "xth",  # unknown name
            "cma",  # missing basic
            "mpt_uv",  # missing basic
            "mpt_uv,cma",  # missing basic
            "basic,,cma",  # empty item
            "basic,basic",  # repeat
            "basic,mpt_uv,basic",  # repeat
        ):
            with self.subTest(channels=channels):
                with self.assertRaises(ValueError):
                    self._parse({SAMPLING_RPC_CHANNELS_ENV: channels})


if __name__ == "__main__":
    absltest.main()
