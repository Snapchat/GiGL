import subprocess
from datetime import datetime
from unittest.mock import MagicMock, patch

import yaml
from absl.testing import absltest
from omegaconf import OmegaConf

from gigl.common.omegaconf_resolvers import (
    register_resolvers,
    set_gigl_resolver_values,
)
from tests.test_assets.test_case import TestCase


class TestNowResolver(TestCase):
    def setUp(self):
        register_resolvers()
        # Use a fixed datetime for predictable testing
        self.test_datetime = datetime(
            year=2023, month=12, day=15, hour=14, minute=30, second=22
        )  # 2023-12-15 14:30:22

    @patch("gigl.common.omegaconf_resolvers.datetime")
    def test_now_resolver(self, mock_datetime):
        mock_datetime.now.return_value = self.test_datetime

        # Example YAML configuration using the unified 'now' resolver
        yaml_config = """
        experiment:
            name: "exp_${now:%Y%m%d_%H%M%S}"
            start_time: ${now:%Y-%m-%d %H:%M:%S}
            log_file: logs/run_${now:%H-%M-%S}.log
            timestamp: "${now:}"  # Uses default format
            short_date: "${now:%m-%d}"

            tomorrow: "${now:%Y-%m-%d, days+1}"
            yesterday: "${now:%Y-%m-%d, days-1}"
            tomorrow_plus_5_hours_30_min_15_sec: "${now:%Y-%m-%d %H:%M:%S,hours+5,days+1,minutes+30,seconds+15}"
            next_week: "${now:%Y-%m-%d, weeks+1}"
            multiple_args: "${now:%Y%m%d, days-15}:${now:%Y%m%d, days-1}"
        """

        expected_name = "exp_20231215_143022"
        expected_start_time = "2023-12-15 14:30:22"
        expected_log_file = "logs/run_14-30-22.log"
        expected_timestamp = "20231215_143022"
        expected_short_date = "12-15"
        expected_tomorrow = "2023-12-16"
        expected_yesterday = "2023-12-14"
        expected_tomorrow_plus_5_hours_30_min_15_sec = "2023-12-16 20:00:37"
        expected_next_week = "2023-12-22"
        expected_multiple_args = "20231130:20231214"

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
        self.assertEqual(config.experiment.next_week, expected_next_week)
        self.assertEqual(config.experiment.multiple_args, expected_multiple_args)

    def test_now_resolver_with_invalid_format(self):
        yaml_config = """
        experiment:
            name: "exp_${now: hours+1, weirdargument-1}"
        """
        with self.assertRaises(ValueError):
            OmegaConf.create(yaml_config).experiment.name

    @patch("gigl.common.omegaconf_resolvers.subprocess.run")
    def test_git_hash_resolver_success(self, mock_subprocess_run):
        # Mock successful git command
        mock_result = MagicMock()
        mocked_hash = "abc123def456789012345678901234567890abcd"
        mock_result.stdout = mocked_hash
        mock_subprocess_run.return_value = mock_result

        yaml_config = """
        experiment:
            commit: "${git_hash:}"
            model_version: "model_${git_hash:}"
        """

        config = OmegaConf.create(yaml.safe_load(yaml_config))
        # Verify subprocess was called with the correct git command

        self.assertEqual(config.experiment.commit, mocked_hash)
        self.assertEqual(config.experiment.model_version, f"model_{mocked_hash}")
        mock_subprocess_run.assert_called()

    @patch("gigl.common.omegaconf_resolvers.subprocess.run")
    def test_git_hash_resolver_command_not_found(self, mock_subprocess_run):
        # Command not found throws a 127 exit code.
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            127, ["git", "rev-parse", "HEAD"], "command not found: git"
        )
        yaml_config = """
        experiment:
            commit: "${git_hash:}"
        """

        # Should return empty string when git is not available
        self.assertEqual(OmegaConf.create(yaml_config).experiment.commit, "")

    @patch("gigl.common.omegaconf_resolvers.subprocess.run")
    def test_git_hash_resolver_not_git_repo(self, mock_subprocess_run):
        # When calling git rev-parse HEAD on a non-git directory, it will return a
        # CalledProcessError with exit code 128.
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            128,
            ["git", "rev-parse", "HEAD"],
            "fatal: not a git repository (or any of the parent directories): .git",
        )

        yaml_config = """
        experiment:
            commit: "${git_hash:}"
        """

        self.assertEqual(OmegaConf.create(yaml_config).experiment.commit, "")


class TestGiglResolver(TestCase):
    """Tests for the ``gigl`` OmegaConf resolver.

    The resolver pulls values from a module-level dict populated by
    ``set_gigl_resolver_values(...)``. When a key is unset it falls
    back to the literal placeholder ``"${gigl:<key>}"`` so a first-pass
    YAML load is lossless and a caller can re-resolve later once
    runtime context is known.
    """

    def setUp(self):
        register_resolvers()
        # Reset the resolver dict between tests so values from prior cases
        # do not leak — set_gigl_resolver_values clears before populating.
        set_gigl_resolver_values({})

    def test_gigl_resolver_returns_value_when_set(self):
        set_gigl_resolver_values({"foo": "bar"})
        cfg = OmegaConf.create({"v": "${gigl:foo}"})
        self.assertEqual(cfg.v, "bar")

    def test_gigl_resolver_returns_placeholder_when_unset(self):
        cfg = OmegaConf.create({"v": "${gigl:foo}"})
        self.assertEqual(cfg.v, "${gigl:foo}")

    def test_gigl_resolver_round_trips_unset_keys_through_yaml(self):
        # Models the first-pass YAML load that ProtoUtils does — the
        # placeholder must survive a YAML→OmegaConf round trip so a
        # caller can re-resolve it later with values set.
        yaml_config = """
        custom:
            command: "${gigl:task_config_uri}"
            args:
              - "--component=${gigl:component}"
              - "--applied_task_identifier=${gigl:applied_task_identifier}"
        """
        cfg = OmegaConf.create(yaml.safe_load(yaml_config))
        self.assertEqual(cfg.custom.command, "${gigl:task_config_uri}")
        self.assertEqual(cfg.custom.args[0], "--component=${gigl:component}")
        self.assertEqual(
            cfg.custom.args[1],
            "--applied_task_identifier=${gigl:applied_task_identifier}",
        )

    def test_set_gigl_resolver_values_overwrites_prior_values(self):
        set_gigl_resolver_values({"a": "1", "b": "2"})
        set_gigl_resolver_values({"a": "10"})
        cfg = OmegaConf.create({"a": "${gigl:a}", "b": "${gigl:b}"})
        self.assertEqual(cfg.a, "10")
        # ``b`` was not in the second call, so it must fall back to the
        # placeholder rather than retaining the prior call's "2" value.
        self.assertEqual(cfg.b, "${gigl:b}")


if __name__ == "__main__":
    absltest.main()
