from absl.testing import absltest

from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.runner import (
    _assert_required_flags,
    _get_parser,
    _parse_additional_job_args,
    _parse_env_vars,
    _parse_labels,
)
from gigl.src.common.constants.components import GiGLComponents
from tests.test_assets.test_case import TestCase

logger = Logger()


class KFPRunnerTest(TestCase):
    def test_parse_additional_job_args(
        self,
    ):
        args = [
            "subgraph_sampler.additional_spark35_jar_file_uris=gs://path/to/jar",
            "subgraph_sampler.arg_2=value=10.243,123",
            "split_generator.some_other_arg=value",
        ]

        expected_parsed_args = {
            GiGLComponents.SubgraphSampler: {
                "additional_spark35_jar_file_uris": "gs://path/to/jar",
                "arg_2": "value=10.243,123",
            },
            GiGLComponents.SplitGenerator: {
                "some_other_arg": "value",
            },
        }
        parsed_args = _parse_additional_job_args(args)
        self.assertEqual(parsed_args, expected_parsed_args)

    def test_parse_labels(self):
        args = ["gigl-integration-test=true", "user=me"]
        expected_parsed_args = {"gigl-integration-test": "true", "user": "me"}
        parsed_args = _parse_labels(args)
        self.assertEqual(parsed_args, expected_parsed_args)

    def test_parse_args_from_cli(self):
        parser = _get_parser()
        args = parser.parse_args(
            [
                "--action=run",  # required arg - not tested here
                "--additional_job_args=subgraph_sampler.additional_spark35_jar_file_uris=gs://path/to/jar",
                "--additional_job_args=subgraph_sampler.arg_2=value=10.243,123",
                "--additional_job_args=split_generator.some_other_arg=value",
                "--run_labels=gigl-integration-test=true",
                "--run_labels=user=me",
            ]
        )
        parsed_args = _parse_additional_job_args(args.additional_job_args)
        parsed_labels = _parse_labels(args.run_labels)
        expected_parsed_args = {
            GiGLComponents.SubgraphSampler: {
                "additional_spark35_jar_file_uris": "gs://path/to/jar",
                "arg_2": "value=10.243,123",
            },
            GiGLComponents.SplitGenerator: {
                "some_other_arg": "value",
            },
        }
        expected_parsed_labels = {"gigl-integration-test": "true", "user": "me"}
        self.assertEqual(parsed_args, expected_parsed_args)
        self.assertEqual(parsed_labels, expected_parsed_labels)

    def test_assert_required_flags_missing_value(self):
        """Test that _assert_required_flags raises ValueError when a required flag has no value."""
        parser = _get_parser()
        # Parse with RUN action but task_config_uri not provided (will be None)
        args = parser.parse_args(
            [
                "--action=run",
                "--resource_config_uri=gs://bucket/resource_config.yaml",
                "--task_config_uri=",
            ]
        )
        with self.assertRaises(ValueError):
            _assert_required_flags(args)

    def test_assert_required_flags_none_value(self):
        """Test that _assert_required_flags raises ValueError when a required flag has no value."""
        parser = _get_parser()
        # Parse with RUN action but task_config_uri not provided (will be None)
        args = parser.parse_args(
            [
                "--action=run",
                "--resource_config_uri=gs://bucket/resource_config.yaml",
            ]
        )
        with self.assertRaises(ValueError):
            _assert_required_flags(args)

    def test_assert_required_flags_success(self):
        """Test that _assert_required_flags succeeds when all required flags are present."""
        parser = _get_parser()
        # Parse with RUN action and all required flags
        args = parser.parse_args(
            [
                "--action=run",
                "--task_config_uri=gs://bucket/task_config.yaml",
                "--resource_config_uri=gs://bucket/resource_config.yaml",
            ]
        )

        # Should not raise any exception
        _assert_required_flags(args)

    def test_parse_env_vars_single(self):
        parsed = _parse_env_vars(["FOO=bar"])
        self.assertEqual(parsed, {"FOO": "bar"})

    def test_parse_env_vars_multiple(self):
        parsed = _parse_env_vars(["FOO=bar", "BAZ=qux"])
        self.assertEqual(parsed, {"FOO": "bar", "BAZ": "qux"})

    def test_parse_env_vars_value_contains_equals(self):
        # split("=", 1) means only the first '=' delimits key/value; the rest is value.
        parsed = _parse_env_vars(["URL=https://example.com/?q=1&r=2"])
        self.assertEqual(parsed, {"URL": "https://example.com/?q=1&r=2"})

    def test_parse_env_vars_empty_value(self):
        parsed = _parse_env_vars(["FOO="])
        self.assertEqual(parsed, {"FOO": ""})

    def test_parse_env_vars_empty_list(self):
        self.assertEqual(_parse_env_vars([]), {})

    def test_parse_env_vars_malformed_raises(self):
        # No '=' in the entry — split("=", 1) returns a single-element list and the
        # tuple unpack raises ValueError, mirroring _parse_labels semantics.
        with self.assertRaises(ValueError):
            _parse_env_vars(["NOT_A_VALID_ENTRY"])

    def test_parse_env_vars_duplicate_keys_last_wins(self):
        parsed = _parse_env_vars(["FOO=first", "FOO=second"])
        self.assertEqual(parsed, {"FOO": "second"})

    def test_assert_required_flags_rejects_env_vars_with_run_no_compile(self):
        """--env_vars must not be combined with --action=run_no_compile."""
        parser = _get_parser()
        args = parser.parse_args(
            [
                "--action=run_no_compile",
                "--task_config_uri=gs://bucket/task_config.yaml",
                "--resource_config_uri=gs://bucket/resource_config.yaml",
                "--compiled_pipeline_path=gs://bucket/pipeline.yaml",
                "--env_vars=FOO=bar",
            ]
        )
        with self.assertRaises(ValueError):
            _assert_required_flags(args)

    def test_assert_required_flags_allows_env_vars_with_run(self):
        """--env_vars is valid for --action=run."""
        parser = _get_parser()
        args = parser.parse_args(
            [
                "--action=run",
                "--task_config_uri=gs://bucket/task_config.yaml",
                "--resource_config_uri=gs://bucket/resource_config.yaml",
                "--env_vars=FOO=bar",
            ]
        )
        _assert_required_flags(args)

    def test_assert_required_flags_allows_env_vars_with_compile(self):
        """--env_vars is valid for --action=compile."""
        parser = _get_parser()
        args = parser.parse_args(
            [
                "--action=compile",
                "--compiled_pipeline_path=gs://bucket/pipeline.yaml",
                "--env_vars=FOO=bar",
            ]
        )
        _assert_required_flags(args)


if __name__ == "__main__":
    absltest.main()
