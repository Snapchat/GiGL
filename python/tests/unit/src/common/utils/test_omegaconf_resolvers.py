import unittest
from datetime import datetime
from unittest.mock import patch
from gigl.src.common.utils.omegaconf_resolvers import register_resolvers

import yaml
from omegaconf import OmegaConf


class TestNowResolver(unittest.TestCase):
    """Test cases for the 'now' OmegaConf resolver."""

    def setUp(self):
        """Set up test fixtures with a fixed datetime for consistent testing."""
        # Use a fixed datetime for predictable testing
        register_resolvers()
        self.test_datetime = datetime(
            year=2023, month=12, day=15, hour=14, minute=30, second=22
        )  # 2023-12-15 14:30:22

    @patch("gigl.src.common.utils.omegaconf_resolvers.datetime")
    def test_now_resolver(self, mock_datetime):
        mock_datetime.now.return_value = self.test_datetime

        # Example YAML configuration using the unified 'now' resolver
        yaml_config = """
        experiment:
            name: "exp_${now:%Y%m%d_%H%M%S}"
            start_time: "${now:%Y-%m-%d %H:%M:%S}"
            log_file: "logs/run_${now:%H-%M-%S}.log"
            timestamp: "${now:}"  # Uses default format
            short_date: "${now:%m-%d}"

            # Explicit named parameter syntax (user-friendly!)
            tomorrow: "${now:%Y-%m-%d,days+1}"
            yesterday: "${now:%Y-%m-%d,days-1}"
            tomorrow_plus_5_hours_30_min_15_sec: "${now:%Y-%m-%d %H:%M:%S,days+1,hours+5,minutes+30,seconds+15}"
        """

        expected_name = "exp_20231215_143022"
        expected_start_time = "2023-12-15 14:30:22"
        expected_log_file = "logs/run_14-30-22.log"
        expected_timestamp = "20231215_143022"
        expected_short_date = "12-15"
        expected_tomorrow = "2023-12-16"
        expected_yesterday = "2023-12-14"
        expected_tomorrow_plus_5_hours_30_min_15_sec = "2023-12-16 20:00:37"

        config_dict = yaml.safe_load(yaml_config)
        config = OmegaConf.create(config_dict)
        self.assertEqual(config.experiment.name, expected_name)
        self.assertEqual(config.experiment.start_time, expected_start_time)
        self.assertEqual(config.experiment.log_file, expected_log_file)
        self.assertEqual(config.experiment.timestamp, expected_timestamp)
        self.assertEqual(config.experiment.short_date, expected_short_date)
        self.assertEqual(config.experiment.tomorrow, expected_tomorrow)
        self.assertEqual(config.experiment.yesterday, expected_yesterday)
        self.assertEqual(
            config.experiment.tomorrow_plus_5_hours_30_min_15_sec,
            expected_tomorrow_plus_5_hours_30_min_15_sec,
        )


if __name__ == "__main__":
    # Run the tests
    unittest.main()
