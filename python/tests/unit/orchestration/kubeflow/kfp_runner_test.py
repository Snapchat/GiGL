import unittest

from gigl.common.logger import Logger
from gigl.orchestration.kubeflow.runner import _parse_additional_job_args
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


if __name__ == "__main__":
    unittest.main()
