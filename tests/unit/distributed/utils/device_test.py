from unittest.mock import patch

import torch
import torch.distributed as dist
from parameterized import param, parameterized

from gigl.distributed.utils.device import (
    get_available_device,
    get_device_from_process_group,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase


class TestGetAvailableDevice(TestCase):
    """Tests for get_available_device."""

    @parameterized.expand(
        [
            param("rank_0", local_process_rank=0),
            param("rank_3", local_process_rank=3),
        ]
    )
    def test_returns_cpu_when_cuda_unavailable(
        self, _name: str, local_process_rank: int
    ) -> None:
        with patch(
            "gigl.distributed.utils.device.torch.cuda.is_available", return_value=False
        ):
            device = get_available_device(local_process_rank)
        self.assertEqual(device, torch.device("cpu"))

    @parameterized.expand(
        [
            param(
                "rank_0_of_1_gpu",
                local_process_rank=0,
                device_count=1,
                expected_index=0,
            ),
            param(
                "rank_0_of_4_gpus",
                local_process_rank=0,
                device_count=4,
                expected_index=0,
            ),
            param(
                "rank_1_of_4_gpus",
                local_process_rank=1,
                device_count=4,
                expected_index=1,
            ),
            param(
                "rank_5_of_4_gpus_round_robin",
                local_process_rank=5,
                device_count=4,
                expected_index=1,
            ),
            param(
                "rank_8_of_4_gpus_round_robin",
                local_process_rank=8,
                device_count=4,
                expected_index=0,
            ),
        ]
    )
    def test_returns_correct_cuda_device(
        self,
        _name: str,
        local_process_rank: int,
        device_count: int,
        expected_index: int,
    ) -> None:
        with (
            patch(
                "gigl.distributed.utils.device.torch.cuda.is_available",
                return_value=True,
            ),
            patch(
                "gigl.distributed.utils.device.torch.cuda.device_count",
                return_value=device_count,
            ),
        ):
            device = get_available_device(local_process_rank)
        self.assertEqual(device, torch.device(f"cuda:{expected_index}"))


class TestGetDeviceFromProcessGroup(TestCase):
    """Tests for get_device_from_process_group."""

    def tearDown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_raises_when_distributed_not_initialized(self) -> None:
        with self.assertRaises(ValueError):
            get_device_from_process_group()

    def test_returns_cpu_with_gloo_backend(self) -> None:
        # Test process group is created with gloo backend
        create_test_process_group()
        device = get_device_from_process_group()
        self.assertEqual(device, torch.device("cpu"))

    def test_returns_cuda_with_nccl_backend(self) -> None:
        with (
            patch(
                "gigl.distributed.utils.device.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                "gigl.distributed.utils.device.torch.distributed.get_backend",
                return_value="nccl",
            ),
        ):
            device = get_device_from_process_group()
        self.assertEqual(device, torch.device("cuda"))
