import os
from unittest.mock import patch

from absl.testing import absltest

from gigl.distributed.dist_sampling_producer import sampling_mode_log_lines
from gigl.distributed.utils.partition_routing import GROUPED_ROUTING_ENV
from tests.test_assets.test_case import TestCase


class SamplingModeLogTest(TestCase):
    def test_lines_report_the_routing_mode_and_the_worker(self) -> None:
        with patch.dict(os.environ, {GROUPED_ROUTING_ENV: "1"}):
            enabled = sampling_mode_log_lines(3)
        with patch.dict(os.environ, {GROUPED_ROUTING_ENV: "0"}):
            disabled = sampling_mode_log_lines(3)
        self.assertEqual(enabled[1], "GROUPED_ROUTING enabled=1 worker=3")
        self.assertEqual(disabled[1], "GROUPED_ROUTING enabled=0 worker=3")
        self.assertRegex(enabled[0], r"^INDUCER_IMPL impl=(std|flat) worker=3$")


if __name__ == "__main__":
    absltest.main()
