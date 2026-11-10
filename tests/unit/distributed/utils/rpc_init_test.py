import hashlib
import inspect
import os
from typing import Any
from unittest import mock

import graphlearn_torch.distributed.rpc as glt_rpc
import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import init_rpc, init_worker_group, shutdown_rpc
from graphlearn_torch.distributed.dist_context import DistRole

from gigl.distributed.utils.networking import get_free_port
from gigl.distributed.utils.rpc_init import (
    SAMPLING_RPC_CHANNELS_ENV,
    SAMPLING_RPC_TRANSPORTS_ENV,
    init_rpc_from_env,
    sampling_rpc_backend_from_env,
)
from tests.test_assets.test_case import TestCase

# sha256 of inspect.getsource(graphlearn_torch.distributed.rpc.init_rpc) that
# gigl.distributed.utils.rpc_init.init_rpc_with_options was copied from.
_GLT_INIT_RPC_SOURCE_SHA256 = (
    "7f16c76c2f3a99535997baff879229801e0839cd473aa01ec8c9a8716bffb949"
)
_WORLD_SIZE = 2
_GROUP = "rpc_init_test"
_MPT_UV_LANE_THREADS = 16


class _Echo(glt_rpc.RpcCalleeBase):
    def call(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 1


def _count_uv_threads() -> int:
    count = 0
    for tid in os.listdir("/proc/self/task"):
        with open(f"/proc/self/task/{tid}/comm") as f:
            if f.read().strip() == "TP_UV_loop":
                count += 1
    return count


def _run_rank(rank: int, port: int, implementation: str, result_queue: Any) -> None:
    init_worker_group(world_size=_WORLD_SIZE, rank=rank, group_name=_GROUP)
    uv_threads_before = _count_uv_threads()
    if implementation == "glt":
        init_rpc(master_addr="127.0.0.1", master_port=port, num_rpc_threads=4)
    else:
        init_rpc_from_env(master_addr="127.0.0.1", master_port=port, num_rpc_threads=4)
    uv_threads = _count_uv_threads() - uv_threads_before
    peer = f"{_GROUP}_{1 - rank}"
    state = {
        "initialized": glt_rpc.rpc_is_initialized(),
        "worker_names": glt_rpc.get_rpc_worker_names(),
        "group_worker_names": glt_rpc.get_rpc_current_group_worker_names(),
        "master_addr": glt_rpc._rpc_master_addr,
        "master_port": glt_rpc._rpc_master_port,
        "uv_threads": uv_threads,
    }
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
    shutdown_rpc()
    state["after_shutdown"] = glt_rpc.rpc_is_initialized()
    try:
        init_rpc(master_addr="127.0.0.1", master_port=port)
        state["reinit_error"] = ""
    except RuntimeError as e:
        state["reinit_error"] = str(e)
    result_queue.put((rank, state))


def _run_two_ranks(implementation: str, env: dict[str, str]) -> dict[int, Any]:
    result_queue = mp.get_context("spawn").SimpleQueue()
    with mock.patch.dict(os.environ, env):
        mp.spawn(
            fn=_run_rank,
            args=(get_free_port(), implementation, result_queue),
            nprocs=_WORLD_SIZE,
        )
    results = dict(result_queue.get() for _ in range(_WORLD_SIZE))
    return results


class GltInitRpcDriftTest(TestCase):
    def test_glt_init_rpc_source_is_unchanged(self) -> None:
        digest = hashlib.sha256(
            inspect.getsource(glt_rpc.init_rpc).encode()
        ).hexdigest()
        self.assertEqual(
            digest,
            _GLT_INIT_RPC_SOURCE_SHA256,
            "graphlearn_torch's init_rpc changed. Re-diff it against "
            "gigl.distributed.utils.rpc_init.init_rpc_with_options, port the change, then "
            "update _GLT_INIT_RPC_SOURCE_SHA256.",
        )


class InitRpcParityTest(TestCase):
    def test_default_matches_glt_and_basic_channel_has_fewer_io_threads(self) -> None:
        glt_default = _run_two_ranks("glt", {})
        gigl_default = _run_two_ranks("gigl", {})
        gigl_basic = _run_two_ranks("gigl", {SAMPLING_RPC_CHANNELS_ENV: "basic"})

        for rank in range(_WORLD_SIZE):
            expected = glt_default[rank]
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
            for actual in (gigl_default[rank], gigl_basic[rank]):
                for key in expected:
                    if key != "uv_threads":
                        self.assertEqual(actual[key], expected[key], msg=key)
            self.assertEqual(
                gigl_default[rank]["uv_threads"], glt_default[rank]["uv_threads"]
            )
            self.assertEqual(
                gigl_default[rank]["uv_threads"] - gigl_basic[rank]["uv_threads"],
                _MPT_UV_LANE_THREADS,
            )


class SamplingRpcBackendFromEnvTest(TestCase):
    def _parse(self, env: dict[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
        with mock.patch.dict(os.environ, env):
            return sampling_rpc_backend_from_env()

    def setUp(self) -> None:
        super().setUp()
        for name in (SAMPLING_RPC_CHANNELS_ENV, SAMPLING_RPC_TRANSPORTS_ENV):
            patcher = mock.patch.dict(os.environ)
            patcher.start()
            self.addCleanup(patcher.stop)
            os.environ.pop(name, None)

    def test_unset_gives_glt_defaults(self) -> None:
        self.assertEqual(self._parse({}), (("uv",), ("mpt_uv", "basic")))

    def test_valid_values_are_parsed_with_whitespace_trimmed(self) -> None:
        env = {
            SAMPLING_RPC_CHANNELS_ENV: "cma, mpt_uv ,basic",
            SAMPLING_RPC_TRANSPORTS_ENV: "uv",
        }
        self.assertEqual(self._parse(env), (("uv",), ("cma", "mpt_uv", "basic")))

    def test_invalid_values_are_rejected(self) -> None:
        for channels in (
            "",
            "cuda_basic",
            "basic,cuda_ipc",
            "xth",
            "cma",
            "basic,,cma",
        ):
            with self.subTest(channels=channels):
                with self.assertRaises(ValueError):
                    self._parse({SAMPLING_RPC_CHANNELS_ENV: channels})
        for transports in ("", "shm", "ibv", "uv,shm"):
            with self.subTest(transports=transports):
                with self.assertRaises(ValueError):
                    self._parse({SAMPLING_RPC_TRANSPORTS_ENV: transports})


if __name__ == "__main__":
    absltest.main()
