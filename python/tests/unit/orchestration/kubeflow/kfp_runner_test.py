import unittest

from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.runner import (
    _assert_required_flags,
    _get_parser,
    _parse_additional_job_args,
    _parse_labels,
)
from gigl.src.common.constants.components import GiGLComponents

logger = Logger()


class KFPRunnerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
