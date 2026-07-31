import asyncio
import os
import time
from collections.abc import Mapping
from threading import BrokenBarrierError
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import torch
from graphlearn_torch.channel import ChannelBase
from graphlearn_torch.distributed import DistDataset, MpDistSamplingWorkerOptions
from graphlearn_torch.sampler import NeighborOutput, NodeSamplerInput, SamplingConfig

from gigl.distributed.base_sampler import (
    BaseDistNeighborSampler,
    SampleLoopInputs,
    _SamplingTimingRecorder,
    _stable_unique_preserve_order,
)
from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler
from gigl.distributed.dist_sampling_producer import (
    DistSamplingProducer,
    SamplingPortLease,
    SamplingWorkerRpcSpec,
    SamplingWorkerStatus,
    close_sampling_port_lease_with_retries,
    resolve_isolated_sampling_worker_rpc_specs,
    validate_isolated_sampling_group_readiness,
)
from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
)
from gigl.distributed.sampler_options import SamplerOptions
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_ITEM = NodeType("item")
_BUYS = Relation("buys")
_CLICKS = Relation("clicks")
_FRIEND = Relation("friend")
_USER_BUYS_ITEM = EdgeType(_USER, _BUYS, _ITEM)
_USER_CLICKS_ITEM = EdgeType(_USER, _CLICKS, _ITEM)
_USER_FRIEND_USER = EdgeType(_USER, _FRIEND, _USER)


def _send_isolated_worker_error(*args) -> None:
    rpc_spec = args[-2]
    status_connection = args[-1]
    assert isinstance(rpc_spec, SamplingWorkerRpcSpec)
    status_connection.send(
        SamplingWorkerStatus(
            state="ERROR",
            worker_index=rpc_spec.worker_index,
            group_name=rpc_spec.group_name,
            rank=rpc_spec.rank,
            master_port=rpc_spec.master_port,
            pid=os.getpid(),
            phase="injected_failure",
            elapsed_seconds=0.0,
            error="injected isolated worker failure",
        )
    )
    status_connection.close()


def _hang_isolated_worker(*args) -> None:
    time.sleep(60)


class SamplingTimingRecorderTest(TestCase):
    def test_invalid_interval_fails_in_parent_before_worker_start(self) -> None:
        for value in ("invalid", "-1"):
            with self.subTest(value=value):
                with (
                    patch.dict(
                        os.environ,
                        {"GIGL_SAMPLER_TIMING_LOG_EVERY_N": value},
                    ),
                    self.assertRaisesRegex(ValueError, "must be"),
                ):
                    DistSamplingProducer.init(cast(DistSamplingProducer, object()))

    def test_emits_complete_windows_and_resets_stage_totals(self) -> None:
        recorder = _SamplingTimingRecorder(log_every_n=2)
        with (
            patch(
                "gigl.distributed.base_sampler.time.perf_counter",
                side_effect=[10.0, 14.0, 14.0, 20.0, 20.0],
            ),
            patch(
                "gigl.distributed.base_sampler.time.thread_time",
                side_effect=[4.0, 7.0, 7.0, 10.0, 10.0],
            ),
        ):
            recorder.begin_loop_observation()
            recorder.record_admission(0.01)
            self.assertIsNone(
                recorder.record_completed(
                    sample_await_seconds=0.1,
                    collate_seconds=0.02,
                    channel_send_seconds=0.03,
                )
            )
            recorder.record_admission(0.02)
            payload = recorder.record_completed(
                sample_await_seconds=0.2,
                collate_seconds=0.04,
                channel_send_seconds=0.06,
            )

            self.assertIsNotNone(payload)
            assert payload is not None
            self.assertEqual(payload["completed_batches"], 2)
            self.assertEqual(payload["window_batches"], 2)
            self.assertEqual(payload["admission_events"], 2)
            self.assertEqual(payload["admission_blocked_s"], 0.03)
            self.assertEqual(payload["sample_await_s"], 0.3)
            self.assertEqual(payload["collate_s"], 0.06)
            self.assertEqual(payload["channel_send_blocked_s"], 0.09)
            self.assertEqual(payload["loop_wall_s"], 4.0)
            self.assertEqual(payload["loop_thread_cpu_s"], 3.0)
            self.assertEqual(payload["loop_thread_busy_fraction"], 0.75)
            self.assertEqual(payload["admission_blocked_ms_per_event"], 15.0)
            self.assertEqual(payload["sample_await_ms_per_batch"], 150.0)
            self.assertEqual(payload["collate_ms_per_batch"], 30.0)
            self.assertEqual(payload["channel_send_blocked_ms_per_batch"], 45.0)

            recorder.record_admission(0.04)
            self.assertIsNone(
                recorder.record_completed(
                    sample_await_seconds=0.4,
                    collate_seconds=0.08,
                    channel_send_seconds=0.12,
                )
            )
            recorder.record_admission(0.06)
            next_payload = recorder.record_completed(
                sample_await_seconds=0.6,
                collate_seconds=0.12,
                channel_send_seconds=0.18,
            )
        self.assertIsNotNone(next_payload)
        assert next_payload is not None
        self.assertEqual(next_payload["completed_batches"], 4)
        self.assertEqual(next_payload["admission_blocked_s"], 0.1)
        self.assertEqual(next_payload["sample_await_s"], 1.0)
        self.assertEqual(next_payload["loop_wall_s"], 6.0)
        self.assertEqual(next_payload["loop_thread_cpu_s"], 3.0)
        self.assertEqual(next_payload["loop_thread_busy_fraction"], 0.5)

    def test_busy_fraction_is_clamped_for_skewed_clocks(self) -> None:
        recorder = _SamplingTimingRecorder(log_every_n=1)
        with (
            patch(
                "gigl.distributed.base_sampler.time.perf_counter",
                side_effect=[10.0, 11.0, 11.0],
            ),
            patch(
                "gigl.distributed.base_sampler.time.thread_time",
                side_effect=[4.0, 6.0, 6.0],
            ),
        ):
            recorder.begin_loop_observation()
            payload = recorder.record_completed(
                sample_await_seconds=0.1,
                collate_seconds=0.02,
                channel_send_seconds=0.03,
            )

        assert payload is not None
        self.assertEqual(payload["loop_thread_busy_fraction"], 1.0)


class IsolatedSamplingWorkerRpcSpecTest(TestCase):
    @staticmethod
    def _build_lifecycle_test_producer(*, rpc_timeout: float) -> DistSamplingProducer:
        producer = object.__new__(DistSamplingProducer)
        producer.sampling_config = SimpleNamespace(seed=None, shuffle=False)
        producer.worker_options = SimpleNamespace(
            worker_concurrency=1,
            rpc_timeout=rpc_timeout,
        )
        producer.num_workers = 1
        producer.data = object()
        producer.sampler_input = object()
        producer.output_channel = object()
        producer.sampling_completed_worker_count = object()
        producer._sampler_options = object()
        producer._degree_tensors = None
        producer._isolated_rpc_specs = (
            SamplingWorkerRpcSpec(
                worker_index=0,
                group_name="isolated_test_group",
                world_size=1,
                rank=0,
                master_port=20000,
            ),
        )
        producer._task_queues = []
        producer._workers = []
        producer._isolated_status_connections = []
        producer._isolated_ready_workers = set()
        producer._isolated_barrier = None
        producer._isolated_resources_closed = False
        producer._isolated_cleanup_complete = False
        producer._isolated_port_lease = None
        producer._shutdown = False
        producer._get_seeds_indexes = Mock(return_value=[torch.tensor([0])])
        return producer

    def test_resolves_one_complete_group_per_worker(self) -> None:
        specs = resolve_isolated_sampling_worker_rpc_specs(
            parent_world_size=65,
            parent_rank=13,
            parent_group_name="inference_local_rank_0",
            data_num_partitions=65,
            data_partition_idx=13,
            num_workers=4,
            master_ports=[20000, 20001, 20002, 20003],
        )

        self.assertEqual(len(specs), 4)
        self.assertEqual([spec.worker_index for spec in specs], [0, 1, 2, 3])
        self.assertEqual([spec.world_size for spec in specs], [65] * 4)
        self.assertEqual([spec.rank for spec in specs], [13] * 4)
        self.assertEqual(
            [spec.master_port for spec in specs], [20000, 20001, 20002, 20003]
        )
        self.assertEqual(len({spec.group_name for spec in specs}), 4)

    def test_requires_one_parent_per_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "one parent per data partition"):
            resolve_isolated_sampling_worker_rpc_specs(
                parent_world_size=65,
                parent_rank=13,
                parent_group_name="group",
                data_num_partitions=64,
                data_partition_idx=13,
                num_workers=2,
                master_ports=[20000, 20001],
            )

    def test_requires_parent_rank_to_match_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "parent rank to match"):
            resolve_isolated_sampling_worker_rpc_specs(
                parent_world_size=65,
                parent_rank=13,
                parent_group_name="group",
                data_num_partitions=65,
                data_partition_idx=12,
                num_workers=2,
                master_ports=[20000, 20001],
            )

    def test_rejects_wrong_or_duplicate_ports(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2"):
            resolve_isolated_sampling_worker_rpc_specs(
                parent_world_size=2,
                parent_rank=0,
                parent_group_name="group",
                data_num_partitions=2,
                data_partition_idx=0,
                num_workers=2,
                master_ports=[20000],
            )
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            resolve_isolated_sampling_worker_rpc_specs(
                parent_world_size=2,
                parent_rank=0,
                parent_group_name="group",
                data_num_partitions=2,
                data_partition_idx=0,
                num_workers=2,
                master_ports=[20000, 20000],
            )

    def test_readiness_requires_exact_group_identity_and_partition_map(self) -> None:
        spec = SamplingWorkerRpcSpec(
            worker_index=2,
            group_name="group_2",
            world_size=2,
            rank=0,
            master_port=21002,
        )
        valid = {
            "group_2_0": {
                "worker_index": 2,
                "rank": 0,
                "partition": 0,
                "port": 21002,
            },
            "group_2_1": {
                "worker_index": 2,
                "rank": 1,
                "partition": 1,
                "port": 21002,
            },
        }
        validate_isolated_sampling_group_readiness(valid, spec)

        invalid_cases = [
            (
                "worker names",
                {
                    "wrong_0": valid["group_2_0"],
                    "wrong_1": valid["group_2_1"],
                },
            ),
            (
                "identity payloads",
                {
                    **valid,
                    "group_2_1": {**valid["group_2_1"], "worker_index": 3},
                },
            ),
            (
                "identity payloads",
                {
                    **valid,
                    "group_2_1": {**valid["group_2_1"], "port": 21003},
                },
            ),
            (
                "rank/partition coverage",
                {
                    **valid,
                    "group_2_1": {**valid["group_2_1"], "partition": 0},
                },
            ),
            (
                "identity payloads",
                {
                    "group_2_0": valid["group_2_1"],
                    "group_2_1": valid["group_2_0"],
                },
            ),
        ]
        for expected_error, readiness in invalid_cases:
            with self.subTest(expected_error=expected_error):
                with self.assertRaisesRegex(RuntimeError, expected_error):
                    validate_isolated_sampling_group_readiness(readiness, spec)

    def test_structured_startup_error_reaps_child_and_is_idempotent(self) -> None:
        fork_context = torch.multiprocessing.get_context("fork")
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)

        with (
            patch(
                "gigl.distributed.dist_sampling_producer.mp.get_context",
                return_value=fork_context,
            ),
            patch(
                "gigl.distributed.dist_sampling_producer._sampling_worker_loop",
                _send_isolated_worker_error,
            ),
            patch(
                "gigl.distributed.dist_sampling_producer.MP_STATUS_CHECK_INTERVAL",
                0.1,
            ),
            self.assertRaisesRegex(RuntimeError, "injected isolated worker failure"),
        ):
            producer.init()

        self.assertTrue(producer._isolated_resources_closed)
        self.assertTrue(all(not worker.is_alive() for worker in producer._workers))
        producer.shutdown()

    def test_startup_timeout_kills_unresponsive_child(self) -> None:
        fork_context = torch.multiprocessing.get_context("fork")
        producer = self._build_lifecycle_test_producer(rpc_timeout=0.2)

        with (
            patch(
                "gigl.distributed.dist_sampling_producer.mp.get_context",
                return_value=fork_context,
            ),
            patch(
                "gigl.distributed.dist_sampling_producer._sampling_worker_loop",
                _hang_isolated_worker,
            ),
            patch(
                "gigl.distributed.dist_sampling_producer.MP_STATUS_CHECK_INTERVAL",
                0.1,
            ),
            self.assertRaisesRegex(TimeoutError, "timed out waiting"),
        ):
            producer.init()

        self.assertTrue(producer._isolated_resources_closed)
        self.assertTrue(all(not worker.is_alive() for worker in producer._workers))
        producer.shutdown()

    def test_status_eof_before_ready_is_explicit(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        worker = Mock()
        worker.exitcode = None
        connection = Mock()
        connection.poll.return_value = True
        connection.recv.side_effect = EOFError
        producer._workers = [worker]
        producer._isolated_status_connections = [connection]

        with self.assertRaisesRegex(RuntimeError, "closed its status pipe"):
            producer._poll_isolated_worker_statuses()

    def test_ready_status_requires_exact_parent_child_identity(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        worker = Mock()
        worker.pid = 1234
        worker.exitcode = None
        connection = Mock()
        connection.poll.side_effect = [True, False]
        connection.recv.return_value = SamplingWorkerStatus(
            state="READY",
            worker_index=0,
            group_name="wrong_group",
            rank=0,
            master_port=20000,
            pid=1234,
            phase="group_readiness",
            elapsed_seconds=0.1,
        )
        producer._workers = [worker]
        producer._isolated_status_connections = [connection]

        with self.assertRaisesRegex(RuntimeError, "invalid identity"):
            producer._poll_isolated_worker_statuses()

    def test_ready_status_accepts_exact_parent_child_identity(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        worker = Mock(pid=1234, exitcode=None)
        connection = Mock()
        connection.poll.side_effect = [True, False]
        connection.recv.return_value = SamplingWorkerStatus(
            state="READY",
            worker_index=0,
            group_name="isolated_test_group",
            rank=0,
            master_port=20000,
            pid=1234,
            phase="group_readiness",
            elapsed_seconds=0.1,
        )
        producer._workers = [worker]
        producer._isolated_status_connections = [connection]

        producer._poll_isolated_worker_statuses()

        self.assertEqual(producer._isolated_ready_workers, {0})

    def test_process_start_failure_closes_all_parent_resources(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        port_lease = Mock()
        producer._isolated_port_lease = port_lease

        with (
            patch(
                "multiprocessing.process.BaseProcess.start",
                side_effect=OSError("injected Process.start failure"),
            ),
            self.assertRaisesRegex(OSError, "injected Process.start failure"),
        ):
            producer.init()

        self.assertEqual(len(producer._workers), 1)
        self.assertIsNone(producer._workers[0].pid)
        self.assertTrue(producer._isolated_resources_closed)
        port_lease.release_reservation.assert_called_once_with(20000)
        port_lease.close.assert_called_once_with()

    def test_reservation_release_failure_closes_all_parent_resources(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        fake_context = Mock()
        fake_context.Barrier.return_value = Mock()
        task_queue = Mock()
        fake_context.Queue.return_value = task_queue
        parent_connection = Mock()
        child_connection = Mock()
        fake_context.Pipe.return_value = (parent_connection, child_connection)
        worker = Mock(pid=None)
        fake_context.Process.return_value = worker
        port_lease = Mock()
        port_lease._closed = False
        port_lease.release_reservation.side_effect = OSError(
            "injected reservation release failure"
        )

        def close_lease() -> None:
            port_lease._closed = True

        port_lease.close.side_effect = close_lease
        producer._isolated_port_lease = port_lease

        with (
            patch(
                "gigl.distributed.dist_sampling_producer.mp.get_context",
                return_value=fake_context,
            ),
            self.assertRaisesRegex(OSError, "reservation release failure"),
        ):
            producer.init()

        worker.start.assert_not_called()
        child_connection.close.assert_called_once_with()
        parent_connection.close.assert_called_once_with()
        task_queue.close.assert_called_once_with()
        port_lease.close.assert_called_once_with()
        self.assertTrue(producer._isolated_cleanup_complete)

    def test_port_lease_close_attempts_all_and_retries_failures(self) -> None:
        first_reservation = Mock()
        first_reservation.close.side_effect = [OSError("socket close failed"), None]
        second_reservation = Mock()
        lease = SamplingPortLease(
            ports=(20000, 20001),
            lock_file_descriptors=(10, 11),
            reservations={
                20000: first_reservation,
                20001: second_reservation,
            },
        )

        with patch(
            "gigl.distributed.dist_sampling_producer.os.close",
            side_effect=[OSError("fd close failed"), None, None],
        ) as close_fd:
            lease.close()
            self.assertFalse(lease._closed)
            self.assertEqual(set(lease.reservations), {20000})
            self.assertEqual(lease.lock_file_descriptors, (10,))

            lease.close()

        self.assertTrue(lease._closed)
        self.assertEqual(lease.reservations, {})
        self.assertEqual(lease.lock_file_descriptors, ())
        self.assertEqual(first_reservation.close.call_count, 2)
        second_reservation.close.assert_called_once_with()
        self.assertEqual(
            [call.args[0] for call in close_fd.call_args_list], [10, 11, 10]
        )

    def test_pre_owner_lease_cleanup_retries_transient_close_failure(self) -> None:
        lease = SamplingPortLease(
            ports=(20000,),
            lock_file_descriptors=(10,),
        )

        with patch(
            "gigl.distributed.dist_sampling_producer.os.close",
            side_effect=[OSError("transient fd close failure"), None],
        ) as close_fd:
            closed = close_sampling_port_lease_with_retries(
                lease,
                context="injected pre-owner failure",
            )

        self.assertTrue(closed)
        self.assertTrue(lease._closed)
        self.assertEqual(close_fd.call_count, 2)

    def test_parent_resource_close_attempts_all_and_retries_failures(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        first_connection = Mock()
        first_connection.close.side_effect = [OSError("pipe close failed"), None]
        second_connection = Mock()
        first_queue = Mock()
        first_queue.close.side_effect = [OSError("queue close failed"), None]
        second_queue = Mock()
        port_lease = SamplingPortLease((20000,))
        producer._isolated_status_connections = [
            first_connection,
            second_connection,
        ]
        producer._task_queues = [first_queue, second_queue]
        producer._isolated_port_lease = port_lease

        producer._close_isolated_resources()

        self.assertFalse(producer._isolated_resources_closed)
        self.assertEqual(producer._isolated_status_connections, [first_connection])
        self.assertEqual(producer._task_queues, [first_queue])
        self.assertTrue(port_lease._closed)

        producer._close_isolated_resources()

        self.assertTrue(producer._isolated_resources_closed)
        self.assertEqual(producer._isolated_status_connections, [])
        self.assertEqual(producer._task_queues, [])
        self.assertEqual(first_connection.close.call_count, 2)
        second_connection.close.assert_called_once_with()
        self.assertEqual(first_queue.close.call_count, 2)
        second_queue.close.assert_called_once_with()

    def test_process_cleanup_continues_after_worker_operation_error(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        first_worker = Mock(pid=101)
        first_worker.join.side_effect = [OSError("join failed"), None, None]
        first_worker.is_alive.return_value = False
        second_worker = Mock(pid=102)
        second_worker.is_alive.return_value = False
        producer._workers = [first_worker, second_worker]
        producer._isolated_port_lease = SamplingPortLease((20000,))

        with self.assertRaisesRegex(RuntimeError, "join failed"):
            producer._cleanup_isolated_workers(graceful=False)

        self.assertEqual(first_worker.join.call_count, 3)
        self.assertEqual(second_worker.join.call_count, 3)
        self.assertTrue(producer._isolated_resources_closed)

    def test_process_cleanup_retries_live_survivor_after_resources_close(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        worker = Mock(pid=101)
        worker.is_alive.side_effect = [True, True, True, False, False, False]
        producer._workers = [worker]

        with self.assertRaisesRegex(RuntimeError, "live child PIDs remain"):
            producer._cleanup_isolated_workers(graceful=False)

        self.assertTrue(producer._isolated_resources_closed)
        self.assertFalse(producer._isolated_cleanup_complete)

        producer._cleanup_isolated_workers(graceful=False)

        self.assertTrue(producer._isolated_cleanup_complete)
        self.assertEqual(worker.join.call_count, 6)

    @staticmethod
    def _configure_two_worker_fake_context(
        producer: DistSamplingProducer,
    ) -> tuple[Mock, list[str], list[Mock], Mock]:
        producer.num_workers = 2
        producer._isolated_rpc_specs = tuple(
            SamplingWorkerRpcSpec(
                worker_index=index,
                group_name=f"isolated_test_group_{index}",
                world_size=1,
                rank=0,
                master_port=20000 + index,
            )
            for index in range(2)
        )
        producer._get_seeds_indexes = Mock(
            return_value=[torch.tensor([0]), torch.tensor([1])]
        )
        producer._isolated_port_lease = SamplingPortLease((20000, 20001))
        events: list[str] = []
        workers: list[Mock] = []
        fake_context = Mock()
        barrier = Mock()
        fake_context.Barrier.return_value = barrier
        fake_context.Queue.side_effect = [Mock(), Mock()]
        fake_context.Pipe.side_effect = [
            (Mock(), Mock()),
            (Mock(), Mock()),
        ]

        def create_process(**_) -> Mock:
            index = len(workers)
            worker = Mock(pid=200 + index)
            worker.exitcode = None
            worker.is_alive.return_value = False
            worker.start.side_effect = lambda: events.append(f"start_{index}")
            workers.append(worker)
            return worker

        fake_context.Process.side_effect = create_process
        return fake_context, events, workers, barrier

    def test_isolated_workers_start_only_after_previous_ready(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        fake_context, events, _, _ = self._configure_two_worker_fake_context(producer)

        def wait_ready(index: int) -> None:
            events.append(f"ready_{index}")

        with (
            patch(
                "gigl.distributed.dist_sampling_producer.mp.get_context",
                return_value=fake_context,
            ),
            patch.object(
                producer,
                "_wait_for_isolated_worker_ready",
                side_effect=wait_ready,
            ),
            patch.object(
                producer,
                "_wait_for_isolated_workers_at_barrier",
                side_effect=lambda: events.append("barrier"),
            ),
        ):
            producer.init()

        self.assertEqual(
            events,
            ["start_0", "ready_0", "start_1", "ready_1", "barrier"],
        )
        producer.shutdown()

    def test_later_group_failure_reaps_ready_prefix(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        fake_context, events, workers, barrier = (
            self._configure_two_worker_fake_context(producer)
        )

        def wait_ready(index: int) -> None:
            events.append(f"ready_{index}")
            if index == 1:
                raise RuntimeError("injected later-group failure")

        with (
            patch(
                "gigl.distributed.dist_sampling_producer.mp.get_context",
                return_value=fake_context,
            ),
            patch.object(
                producer,
                "_wait_for_isolated_worker_ready",
                side_effect=wait_ready,
            ),
            self.assertRaisesRegex(RuntimeError, "injected later-group failure"),
        ):
            producer.init()

        self.assertEqual(events, ["start_0", "ready_0", "start_1", "ready_1"])
        barrier.abort.assert_called_once_with()
        self.assertTrue(all(worker.join.called for worker in workers))
        self.assertTrue(producer._isolated_resources_closed)

    def test_final_barrier_failure_reaps_all_ready_workers(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        fake_context, events, workers, barrier = (
            self._configure_two_worker_fake_context(producer)
        )

        with (
            patch(
                "gigl.distributed.dist_sampling_producer.mp.get_context",
                return_value=fake_context,
            ),
            patch.object(
                producer,
                "_wait_for_isolated_worker_ready",
                side_effect=lambda index: events.append(f"ready_{index}"),
            ),
            patch.object(
                producer,
                "_wait_for_isolated_workers_at_barrier",
                side_effect=BrokenBarrierError("injected final barrier failure"),
            ),
            self.assertRaisesRegex(
                BrokenBarrierError, "injected final barrier failure"
            ),
        ):
            producer.init()

        self.assertEqual(events, ["start_0", "ready_0", "start_1", "ready_1"])
        barrier.abort.assert_called_once_with()
        self.assertTrue(all(worker.join.called for worker in workers))
        self.assertTrue(producer._isolated_resources_closed)

    def test_cleanup_aborts_barrier_before_closing_resources(self) -> None:
        producer = self._build_lifecycle_test_producer(rpc_timeout=1.0)
        barrier = Mock()
        producer._isolated_barrier = barrier

        producer._cleanup_isolated_workers(graceful=False)

        barrier.abort.assert_called_once_with()
        self.assertTrue(producer._isolated_resources_closed)

    def test_isolated_mode_rejects_all_to_all(self) -> None:
        worker_options = SimpleNamespace(
            use_all2all=True,
            rpc_timeout=1.0,
        )
        port_lease = Mock()

        def initialize_base(producer, *_args) -> None:
            producer.num_workers = 1

        with (
            patch(
                "gigl.distributed.dist_sampling_producer.DistMpSamplingProducer.__init__",
                new=initialize_base,
            ),
            self.assertRaisesRegex(ValueError, "does not yet support use_all2all"),
        ):
            DistSamplingProducer(
                data=cast(DistDataset, object()),
                sampler_input=cast(NodeSamplerInput, object()),
                sampling_config=cast(SamplingConfig, object()),
                worker_options=cast(MpDistSamplingWorkerOptions, worker_options),
                channel=cast(ChannelBase, object()),
                sampler_options=cast(SamplerOptions, object()),
                isolated_rpc_specs=(
                    SamplingWorkerRpcSpec(
                        worker_index=0,
                        group_name="group",
                        world_size=1,
                        rank=0,
                        master_port=21000,
                    ),
                ),
                isolated_port_lease=port_lease,
            )
        port_lease.close.assert_called_once_with()


def _build_sampler_input(
    num_nodes: int = 4,
) -> ABLPNodeSamplerInput:
    """Builds a simple ABLPNodeSamplerInput for testing with two edge types."""
    node = torch.arange(num_nodes)
    positive_label_by_edge_types = {
        _USER_BUYS_ITEM: torch.arange(100, 100 + num_nodes),
        _USER_CLICKS_ITEM: torch.arange(200, 200 + num_nodes),
    }
    negative_label_by_edge_types = {
        _USER_BUYS_ITEM: torch.arange(300, 300 + num_nodes),
        _USER_CLICKS_ITEM: torch.arange(400, 400 + num_nodes),
    }
    return ABLPNodeSamplerInput(
        node=node,
        input_type=_USER,
        positive_label_by_edge_types=positive_label_by_edge_types,
        negative_label_by_edge_types=negative_label_by_edge_types,
    )


class TestABLPNodeSamplerInput(TestCase):
    def test_construction_and_properties(self) -> None:
        node = torch.tensor([10, 20, 30])
        positive_labels = {_USER_BUYS_ITEM: torch.tensor([1, 2, 3])}
        negative_labels = {_USER_CLICKS_ITEM: torch.tensor([4, 5, 6])}

        sampler_input = ABLPNodeSamplerInput(
            node=node,
            input_type=_USER,
            positive_label_by_edge_types=positive_labels,
            negative_label_by_edge_types=negative_labels,
        )

        self.assert_tensor_equality(sampler_input.node, node)
        self.assertEqual(sampler_input.input_type, _USER)
        self.assertEqual(
            set(sampler_input.positive_label_by_edge_types.keys()),
            {_USER_BUYS_ITEM},
        )
        self.assert_tensor_equality(
            sampler_input.positive_label_by_edge_types[_USER_BUYS_ITEM],
            positive_labels[_USER_BUYS_ITEM],
        )
        self.assertEqual(
            set(sampler_input.negative_label_by_edge_types.keys()),
            {_USER_CLICKS_ITEM},
        )
        self.assert_tensor_equality(
            sampler_input.negative_label_by_edge_types[_USER_CLICKS_ITEM],
            negative_labels[_USER_CLICKS_ITEM],
        )

    def test_len(self) -> None:
        for num_nodes in (1, 4, 10):
            sampler_input = _build_sampler_input(num_nodes=num_nodes)
            self.assertEqual(len(sampler_input), num_nodes)

    def test_getitem_with_tensor_index(self) -> None:
        sampler_input = _build_sampler_input(num_nodes=4)
        index = torch.tensor([0, 2])
        sliced = sampler_input[index]

        self.assertIsInstance(sliced, ABLPNodeSamplerInput)
        self.assert_tensor_equality(sliced.node, torch.tensor([0, 2]))
        self.assertEqual(sliced.input_type, _USER)
        self.assert_tensor_equality(
            sliced.positive_label_by_edge_types[_USER_BUYS_ITEM],
            torch.tensor([100, 102]),
        )
        self.assert_tensor_equality(
            sliced.positive_label_by_edge_types[_USER_CLICKS_ITEM],
            torch.tensor([200, 202]),
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_BUYS_ITEM],
            torch.tensor([300, 302]),
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_CLICKS_ITEM],
            torch.tensor([400, 402]),
        )

    def test_getitem_with_list_index(self) -> None:
        sampler_input = _build_sampler_input(num_nodes=4)
        sliced = sampler_input[[1]]

        self.assertIsInstance(sliced, ABLPNodeSamplerInput)
        self.assertTrue(torch.equal(sliced.node, torch.tensor([1])))
        self.assert_tensor_equality(
            sliced.positive_label_by_edge_types[_USER_BUYS_ITEM], torch.tensor([101])
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_CLICKS_ITEM], torch.tensor([401])
        )

    def test_share_memory(self) -> None:
        sampler_input = _build_sampler_input(num_nodes=3)
        result = sampler_input.share_memory()

        self.assertIs(result, sampler_input)
        self.assertTrue(sampler_input.node.is_shared())
        self.assertTrue(
            sampler_input.positive_label_by_edge_types[_USER_BUYS_ITEM].is_shared()
        )
        self.assertTrue(
            sampler_input.positive_label_by_edge_types[_USER_CLICKS_ITEM].is_shared()
        )
        self.assertTrue(
            sampler_input.negative_label_by_edge_types[_USER_BUYS_ITEM].is_shared()
        )
        self.assertTrue(
            sampler_input.negative_label_by_edge_types[_USER_CLICKS_ITEM].is_shared()
        )


class _ZeroFanoutInducer:
    def init_node(
        self, nodes: dict[NodeType, torch.Tensor]
    ) -> dict[NodeType, torch.Tensor]:
        return nodes

    def induce_next(
        self, nbr_dict: dict[EdgeType, list[torch.Tensor]]
    ) -> tuple[
        dict[NodeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
    ]:
        sampled_edge_type = next(iter(nbr_dict))
        return (
            {_USER: torch.tensor([3])},
            {sampled_edge_type: torch.tensor([0])},
            {sampled_edge_type: torch.tensor([0])},
        )


class DistNeighborSamplerZeroFanoutTest(TestCase):
    @staticmethod
    async def _run_heterogeneous_sampler(
        *, edge_dir: str, positive_fanout: int
    ) -> AsyncMock:
        zero_edge_type = EdgeType(_USER, Relation("zero"), _USER)
        sampled_edge_type = EdgeType(_USER, Relation("sampled"), _USER)
        sampler = DistNeighborSampler.__new__(DistNeighborSampler)
        sampler.max_input_size = 0
        sampler.device = torch.device("cpu")
        sampler.dist_graph = SimpleNamespace(data_cls="hetero")
        sampler._acquire_inducer = Mock(return_value=_ZeroFanoutInducer())
        sampler.inducer_pool = Mock()
        sampler.with_edge = False
        sampler.num_hops = 1
        sampler.edge_types = [zero_edge_type, sampled_edge_type]
        sampler.num_neighbors = {
            zero_edge_type: [0],
            sampled_edge_type: [positive_fanout],
        }
        sampler.edge_dir = edge_dir
        sampler._loop = asyncio.get_running_loop()
        sample_one_hop = AsyncMock(
            return_value=NeighborOutput(
                nbr=torch.tensor([3]),
                nbr_num=torch.tensor([1, 0]),
                edge=None,
            )
        )
        sampler._sample_one_hop = sample_one_hop

        await sampler._sample_from_nodes(
            NodeSamplerInput(node=torch.tensor([1, 2]), input_type=_USER)
        )
        return sample_one_hop

    def test_heterogeneous_sampler_skips_only_exact_zero_fanout(self) -> None:
        for edge_dir in ("in", "out"):
            for positive_fanout in (-1, 2):
                with self.subTest(edge_dir=edge_dir, positive_fanout=positive_fanout):
                    sample_one_hop = asyncio.run(
                        self._run_heterogeneous_sampler(
                            edge_dir=edge_dir,
                            positive_fanout=positive_fanout,
                        )
                    )
                    self.assertEqual(sample_one_hop.await_count, 1)
                    await_args = sample_one_hop.await_args
                    assert await_args is not None
                    _, observed_fanout, _ = await_args.args
                    self.assertEqual(observed_fanout, positive_fanout)

    def test_homogeneous_sampler_stops_before_zero_fanout_rpc(self) -> None:
        async def run_sampler() -> tuple[AsyncMock, torch.Tensor, torch.Tensor]:
            sampler = DistNeighborSampler.__new__(DistNeighborSampler)
            sampler.max_input_size = 0
            sampler.device = torch.device("cpu")
            sampler.dist_graph = SimpleNamespace(data_cls="homo")
            inducer = Mock()
            inducer.init_node.return_value = torch.tensor([1, 2])
            sampler._acquire_inducer = Mock(return_value=inducer)
            sampler.inducer_pool = Mock()
            sampler.with_edge = False
            sampler.num_neighbors = [0, 2]
            sample_one_hop = AsyncMock()
            sampler._sample_one_hop = sample_one_hop

            output = await sampler._sample_from_nodes(
                NodeSamplerInput(node=torch.tensor([1, 2]), input_type=None)
            )
            return sample_one_hop, output.row, output.col

        sample_one_hop, rows, columns = asyncio.run(run_sampler())
        sample_one_hop.assert_not_awaited()
        self.assertEqual(rows.numel(), 0)
        self.assertEqual(columns.numel(), 0)


def _build_sampler_stub(edge_dir: str = "out") -> BaseDistNeighborSampler:
    """Build a minimal BaseGiGLSampler stub for testing shared utilities."""
    sampler = BaseDistNeighborSampler.__new__(BaseDistNeighborSampler)
    sampler.device = torch.device("cpu")
    sampler.edge_dir = edge_dir
    return sampler


class TestBaseGiGLSamplerPreparation(TestCase):
    def test_stable_unique_preserves_first_occurrence_order(self) -> None:
        self.assert_tensor_equality(
            _stable_unique_preserve_order(torch.tensor([7, 3, 7, 5, 3, 9])),
            torch.tensor([7, 3, 5, 9]),
        )

    def test_stable_unique_requires_one_dimensional_tensor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected a 1-D tensor"):
            _stable_unique_preserve_order(torch.tensor([[1, 2], [3, 4]]))

    def test_prepare_ablp_inputs_dedupes_same_type_seeds_and_keeps_anchors_first(
        self,
    ) -> None:
        sampler = _build_sampler_stub(edge_dir="out")
        positive_labels = {_USER_FRIEND_USER: torch.tensor([11, 12, -1, 13])}
        negative_labels = {_USER_FRIEND_USER: torch.tensor([13, 14, 10, -1])}
        sampler_input = ABLPNodeSamplerInput(
            node=torch.tensor([10, 11, 10]),
            input_type=_USER,
            positive_label_by_edge_types=positive_labels,
            negative_label_by_edge_types=negative_labels,
        )

        sample_loop_inputs = sampler._prepare_ablp_inputs(
            inputs=sampler_input,
            input_seeds=sampler_input.node,
            input_type=_USER,
        )

        nodes_to_sample = sample_loop_inputs.nodes_to_sample
        assert isinstance(nodes_to_sample, Mapping)
        self.assertEqual(set(nodes_to_sample.keys()), {_USER})
        self.assert_tensor_equality(
            nodes_to_sample[_USER],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
            torch.tensor([10, 11, 12, 13, 14]),
        )
        self.assert_tensor_equality(
            sample_loop_inputs.metadata[
                f"{POSITIVE_LABEL_METADATA_KEY}{str(tuple(_USER_FRIEND_USER))}"
            ],
            positive_labels[_USER_FRIEND_USER],
        )
        self.assert_tensor_equality(
            sample_loop_inputs.metadata[
                f"{NEGATIVE_LABEL_METADATA_KEY}{str(tuple(_USER_FRIEND_USER))}"
            ],
            negative_labels[_USER_FRIEND_USER],
        )

    def test_prepare_ablp_inputs_dedupes_cross_type_supervision_nodes(self) -> None:
        sampler = _build_sampler_stub(edge_dir="out")
        sampler_input = ABLPNodeSamplerInput(
            node=torch.tensor([4, 5]),
            input_type=_USER,
            positive_label_by_edge_types={
                _USER_BUYS_ITEM: torch.tensor([20, 21, 20, -1])
            },
            negative_label_by_edge_types={
                _USER_BUYS_ITEM: torch.tensor([21, 22, -1, 20])
            },
        )

        sample_loop_inputs = sampler._prepare_ablp_inputs(
            inputs=sampler_input,
            input_seeds=sampler_input.node,
            input_type=_USER,
        )

        nodes_to_sample = sample_loop_inputs.nodes_to_sample
        assert isinstance(nodes_to_sample, Mapping)
        self.assertEqual(set(nodes_to_sample.keys()), {_USER, _ITEM})
        self.assert_tensor_equality(
            nodes_to_sample[_USER],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
            torch.tensor([4, 5]),
        )
        self.assert_tensor_equality(
            nodes_to_sample[_ITEM],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
            torch.tensor([20, 21, 22]),
        )

    def test_prepare_sample_loop_inputs_homogeneous(self) -> None:
        """Standard NodeSamplerInput with no input_type returns a tensor."""
        sampler = _build_sampler_stub()
        inputs = NodeSamplerInput(
            node=torch.tensor([10, 20, 30]),
            input_type=None,
        )

        result = sampler._prepare_sample_loop_inputs(inputs)

        self.assertIsInstance(result, SampleLoopInputs)
        assert isinstance(result.nodes_to_sample, torch.Tensor)
        self.assert_tensor_equality(result.nodes_to_sample, torch.tensor([10, 20, 30]))
        self.assertEqual(result.metadata, {})

    def test_prepare_sample_loop_inputs_heterogeneous(self) -> None:
        """Standard NodeSamplerInput with input_type returns a dict."""
        sampler = _build_sampler_stub()
        inputs = NodeSamplerInput(
            node=torch.tensor([1, 2]),
            input_type=_USER,
        )

        result = sampler._prepare_sample_loop_inputs(inputs)

        self.assertIsInstance(result, SampleLoopInputs)
        assert isinstance(result.nodes_to_sample, Mapping)
        self.assertEqual(set(result.nodes_to_sample.keys()), {_USER})
        self.assert_tensor_equality(result.nodes_to_sample[_USER], torch.tensor([1, 2]))  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        self.assertEqual(result.metadata, {})
