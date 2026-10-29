from gigl.src.common.utils.spark_job_manager import (
    SparkJobManager,
    build_split_generator_cluster_name,
    build_subgraph_sampler_cluster_name,
)
from tests.test_assets.test_case import TestCase


class ClusterNameTest(TestCase):
    def test_subgraph_sampler_name_replaces_underscores_and_prefixes(self) -> None:
        self.assertEqual(
            build_subgraph_sampler_cluster_name("cora_nalp_test_on_20261026_183748"),
            "sgs-cora-nalp-test-on-20261026-183748",
        )

    def test_split_generator_name(self) -> None:
        self.assertEqual(build_split_generator_cluster_name("a_b"), "split-a-b")

    def test_cuts_to_50_and_fixes_trailing_dash(self) -> None:
        name = SparkJobManager.get_sanitized_dataproc_cluster_name("a" * 49 + "_x")

        self.assertEqual(len(name), 50)
        self.assertEqual(name, "a" * 49 + "z")
