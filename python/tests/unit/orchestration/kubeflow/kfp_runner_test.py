import unittest

from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.runner import (
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


if __name__ == "__main__":
    unittest.main()
