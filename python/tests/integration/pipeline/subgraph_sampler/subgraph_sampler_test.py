import tempfile
import unittest
from collections import defaultdict
from itertools import chain
from typing import Dict, Iterable, List, Set, Tuple

from gigl.common import LocalUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    EdgeUsageType,
    NodeType,
    Relation,
)
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.graph_data_types import (
    EdgePbWrapper,
    NodePbWrapper,
)
from gigl.src.common.types.pb_wrappers.graph_data_types_utils import (
    get_dehydrated_node_pb_wrappers_from_edge_wrapper,
)
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    TOY_GRAPH_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from snapchat.research.gbml import (
    gbml_config_pb2,
    graph_schema_pb2,
    subgraph_sampling_strategy_pb2,
)
from snapchat.research.gbml.training_samples_schema_pb2 import (
    NodeAnchorBasedLinkPredictionSample,
    RootedNodeNeighborhood,
)
from tests.integration.pipeline.subgraph_sampler.utils import (
    EdgeMetadataInfo,
    ExpectedGraphFromPreprocessor,
    bidirectionalize_edge_type_to_edge_to_features_map,
    bidirectionalize_feasible_adjacency_list_map,
    compile_and_run_sgs_pipeline_locally,
    overwrite_subgraph_sampler_downloaded_assets_paths_to_local,
    overwrite_subgraph_sampler_output_paths_to_local,
    read_output_nablp_samples_from_subgraph_sampler,
    read_output_node_based_task_samples_from_subgraph_sampler,
    reconstruct_graph_information_from_preprocessor_output,
)
from tests.integration.pipeline.utils import (
    get_gcs_assets_dir_from_frozen_gbml_config_uri,
)

logger = Logger()

TEST_NUM_HOPS = 2
TEST_NUM_NEIGHBORS_TO_SAMPLE = 10
TEST_NUM_POSITIVE_SAMPLES = 3
TEST_NUM_USER_DEFINED_POSITIVE_SAMPLES = 2
TEST_NUM_USER_DEFINED_NEGATIVE_SAMPLES = 2


class SubgraphSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def __compile_and_run_sgs_pipeline_locally(
        self,
        gbml_config_uri: Uri,
        subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
        use_spark35: bool = False,
    ) -> Tuple[ExpectedGraphFromPreprocessor, GbmlConfigPbWrapper]:
        """
        Helper function to compile and run an SGS pipeline locally using the provided GbmlConfig and SubgraphSamplerConfig.
        Also returns the ExpectedGraphFromPreprocessor object reconstructed from the preprocessor output since this is
        needed for all validity checks.
        Also return the updated GbmlConfigPbWrapper after overwriting the assets and output paths to local.
        """

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=gbml_config_uri
            )
        )

        # Pre-load files locally since SGS test is unable to read from gcs.
        gcs_assets_dir = get_gcs_assets_dir_from_frozen_gbml_config_uri(
            gbml_config_uri=gbml_config_uri
        )
        tmp_local_assets_download_path = LocalUri(f"{tempfile.mkdtemp()}/")
        file_loader = FileLoader()
        file_loader.load_directory(
            dir_uri_src=gcs_assets_dir, dir_uri_dst=tmp_local_assets_download_path
        )

        overwrite_subgraph_sampler_downloaded_assets_paths_to_local(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            gcs_dir=gcs_assets_dir,
            local_dir=tmp_local_assets_download_path,
        )

        frozen_gbml_config_uri = overwrite_subgraph_sampler_output_paths_to_local(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
        )

        frozen_gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=frozen_gbml_config_uri
            )
        )

        logger.info(
            f"Running SGS pipeline using GbmlConfig at {frozen_gbml_config_uri}"
        )
        resource_config_uri = UriFactory.create_uri(
            uri=get_resource_config().get_resource_config_uri
        )
        assert isinstance(resource_config_uri, LocalUri)

        compile_and_run_sgs_pipeline_locally(
            frozen_gbml_config_uri=frozen_gbml_config_uri,
            resource_config_uri=resource_config_uri,
            use_spark35=use_spark35,
        )

        # We load the expected graph obtained from Data Preprocessor output which we fed into SGS below
        expected_graph_from_preprocessor = (
            reconstruct_graph_information_from_preprocessor_output(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper
            )
        )

        # Since SGS may bidirectionalize the graph (adding new feasible edges from the previous output), we need to update the expected graph assets accordingly.
        if (
            not gbml_config_pb_wrapper.shared_config.is_graph_directed
            and not gbml_config_pb_wrapper.graph_metadata_pb_wrapper.is_heterogeneous
        ):
            # If the graph is homogeneous, and it is not directed, then SGS will bidirectionalize the main edges. This means that for any edge
            # (src=A, dst=B, edge_type=C, features=D) in the preprocessed data, SGS will add (src=B, dst=A, edge_type=C, features=D)
            # when constructing rooted neighborhood or training samples.

            # We add these flipped edges to the feasible edge set.
            expected_graph_from_preprocessor.main_edge_info.feasible_adjacency_list_map = bidirectionalize_feasible_adjacency_list_map(
                src_node_to_edge_map=expected_graph_from_preprocessor.main_edge_info.feasible_adjacency_list_map,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            )

            # We also add these flipped edges to the edge type to edge to features map.
            expected_graph_from_preprocessor.main_edge_info.edge_type_to_edge_to_features_map = bidirectionalize_edge_type_to_edge_to_features_map(
                edge_type_to_edge_to_features_map=expected_graph_from_preprocessor.main_edge_info.edge_type_to_edge_to_features_map,
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            )

        return expected_graph_from_preprocessor, frozen_gbml_config_pb_wrapper

    def __run_and_check_node_anchor_based_link_prediction_task_sgs_validity(
        self,
        gbml_config_uri: Uri,
        subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
        should_check_user_defined_labels: bool = False,
        use_spark35: bool = False,
    ) -> Tuple[
        ExpectedGraphFromPreprocessor,
        Dict[NodeType, List[RootedNodeNeighborhood]],
        List[NodeAnchorBasedLinkPredictionSample],
    ]:
        """
        Run the SGS pipeline for a NABLP task using the provided GbmlConfig and check the validity of the output.
        These validity checks should pass for *all* graphs.  The checks test:
            - The number of training samples outputted by SGS is greater than 0.
            - The number of rooted neighborhood samples outputted by SGS is greater than 0.
            - The number of rooted neighborhood samples outputted by SGS is equal to the number of nodes outputted by the preprocessor.
            - The number of training samples outputted by SGS is less than or equal to the number of nodes outputted by the preprocessor.
            - All subgraph samples are valid (i.e. no duplicate or inconsistent nodes or edges, correct number of features, root node included)
        """
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=gbml_config_uri
            )
        )

        is_subgraph_sampling_strategy_provided = subgraph_sampler_config_pb.HasField(
            "subgraph_sampling_strategy"
        )

        # TODO(nshah-sc): this should be removed once SGS properly outputs the right # of RNNs for graphs where is_graph_directed = True.
        # Currently, we will only allow this test to run on graphs where is_graph_directed = False.  SGS will bidirectionalize these graphs.
        # i.e. if the preprocessed data only has (src, dst, edge_type), then SGS will add (dst, src, edge_type) as well.
        assert not gbml_config_pb_wrapper.shared_config.is_graph_directed, (
            "This test is currently buggy for is_graph_directed=True. In particular, we can surprisingly encounter duplicated RNNs.",
        )

        (
            expected_graph_from_preprocessor,
            updated_gbml_config_pb_wrapper,
        ) = self.__compile_and_run_sgs_pipeline_locally(
            gbml_config_uri=gbml_config_uri,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            use_spark35=use_spark35,
        )

        gbml_config_pb_wrapper = updated_gbml_config_pb_wrapper

        # We read the output that the SGS component produced
        (
            rooted_node_neighborhood_samples,
            training_samples,
        ) = read_output_nablp_samples_from_subgraph_sampler(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper
        )

        if is_subgraph_sampling_strategy_provided:
            # Check that the number of types of rooted node neighborhood samples matches the number of types of root nodes in the subgraph sampling strategy
            actual_num_node_types_produced = len(
                rooted_node_neighborhood_samples.keys()
            )
            expected_num_node_types_produced = len(
                subgraph_sampler_config_pb.subgraph_sampling_strategy.message_passing_paths.paths
            )
            self.assertEqual(
                actual_num_node_types_produced,
                expected_num_node_types_produced,
                f"Subgraph Sampler produced RNNs for {actual_num_node_types_produced} node types, but expected {expected_num_node_types_produced}.",
            )

            # Check that the root node types in rooted node neighborhoods match the root node types in subgraph sampling strategy.
            sampling_strategy_root_node_types = [
                NodeType(path.root_node_type)
                for path in subgraph_sampler_config_pb.subgraph_sampling_strategy.message_passing_paths.paths
            ]
            for (
                rooted_node_neighborhood_root_node_type
            ) in rooted_node_neighborhood_samples.keys():
                self.assertIn(
                    rooted_node_neighborhood_root_node_type,
                    sampling_strategy_root_node_types,
                    f"Subgraph Sampler produced RNNs for node type {rooted_node_neighborhood_root_node_type}, which does not match any root node in the SubgraphSamplingStrategy.",
                )

            # Check that the correct number of rooted node types for RNNs were produced
            self.assertEqual(
                len(rooted_node_neighborhood_samples.keys()),
                len(sampling_strategy_root_node_types),
            )

        # Check that the number of rooted node neighborhoods produced is equal to the number of nodes of that type encountered in the preprocessor output.
        for (
            node_type,
            rooted_samples_for_node_type,
        ) in rooted_node_neighborhood_samples.items():
            self.assertEqual(
                len(rooted_samples_for_node_type),
                len(
                    expected_graph_from_preprocessor.node_type_to_node_to_features_map[
                        node_type
                    ]
                ),
                f"Found {len(rooted_samples_for_node_type)} rooted samples for node type {node_type} from SGS output, but we have {len(expected_graph_from_preprocessor.node_type_to_node_to_features_map[node_type])} nodes with type {node_type} from preprocessor output",
            )

        # Check that the number of training samples produced is greater than 0.
        num_training_samples = len(training_samples)
        self.assertGreater(
            num_training_samples,
            0,
            "Missing node-anchor training samples from SGS output",
        )

        root_node_type_to_num_training_samples: Dict[NodeType, int] = defaultdict(
            lambda: 0
        )
        for training_sample in training_samples:
            root_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                CondensedNodeType(training_sample.root_node.condensed_node_type)
            ]
            root_node_type_to_num_training_samples[root_node_type] += 1

        # Check that the number of training samples produced is less than or equal to the number of nodes of the src node type produced by the preprocessor.
        supervision_edge_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
        )
        training_sample_root_node_types = [
            edge_type.src_node_type for edge_type in supervision_edge_types
        ]
        for supervision_edge_type in supervision_edge_types:
            src_node_type = supervision_edge_type.src_node_type
            self.assertLessEqual(
                root_node_type_to_num_training_samples[src_node_type],
                len(
                    expected_graph_from_preprocessor.node_type_to_node_to_features_map[
                        src_node_type
                    ]
                ),
                f"Found {num_training_samples} training samples from src node type {src_node_type}, but expected at most {len(expected_graph_from_preprocessor.node_type_to_node_to_features_map[src_node_type])} training samples, since there are only {len(expected_graph_from_preprocessor.node_type_to_node_to_features_map[src_node_type])} nodes of type {node_type}.",
            )

        # Check all training samples has root node type in src nodes of supervision edge types
        for training_sample in training_samples:
            root_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                CondensedNodeType(training_sample.root_node.condensed_node_type)
            ]
            self.assertIn(
                root_node_type,
                training_sample_root_node_types,
                f"Root node type {root_node_type} in training samples not found as source node of any supervision edge types: {supervision_edge_types}",
            )

        self.__check_subgraph_samples_for_invalid_nodes_or_edges(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            expected_graph_from_preprocessor=expected_graph_from_preprocessor,
            rooted_node_neighborhood_sample_pbs=chain(
                *rooted_node_neighborhood_samples.values()
            ),
            training_sample_pbs=training_samples,
        )

        # Check that pos_edges and neg_edges are well-specified in both UDL and non-UDL NABLP settings
        self.__check_nablp_samples_for_positive_and_negative_edges(
            training_samples=training_samples,
            expected_graph_from_preprocessor=expected_graph_from_preprocessor,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            should_check_user_defined_labels=should_check_user_defined_labels,
        )

        return (
            expected_graph_from_preprocessor,
            rooted_node_neighborhood_samples,
            training_samples,
        )

    def __check_nablp_samples_for_positive_and_negative_edges(
        self,
        training_samples: Iterable[NodeAnchorBasedLinkPredictionSample],
        expected_graph_from_preprocessor: ExpectedGraphFromPreprocessor,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
        should_check_user_defined_labels: bool = False,
    ):
        """
        Checks that the positive and negative edges in NABLP training samples are well-specified.
        If should_check_user_defined_labels is True:
            - we check that the user-defined labels from NABLP samples match the expected user-defined labels from the preprocessor output.
            - we check that the number of features in each user-defined label edge match the expected number of features from the preprocessor output.
        If should_check_user_defined_labels is False:
            - we check that the edges in the NABLP samples are well-specified in the expected graph edges preprocessor output.
        For all positive and negative edges, check that the edge type is a valid supervision edge type.
        """

        def __check_supervision_edge_dst_node_exists_in_preprocessor_output(
            edge_pb: graph_schema_pb2.Edge,
            gbml_config_pb_wrapper: GbmlConfigPbWrapper,
            edge_usage_type: EdgeUsageType,
            expected_graph_from_preprocessor: ExpectedGraphFromPreprocessor,
        ):
            edge_pbw = EdgePbWrapper(pb=edge_pb)
            _, dst_node_pbw = get_dehydrated_node_pb_wrappers_from_edge_wrapper(
                edge_pb_wrapper=edge_pbw,
                graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
            )
            dst_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                dst_node_pbw.condensed_node_type
            ]
            self.assertIn(
                dst_node_pbw,
                expected_graph_from_preprocessor.node_type_to_node_to_features_map[
                    dst_node_type
                ],
                f"Found a {edge_usage_type.value} edge {edge_pbw} in training samples supervision, but didn't find the target node in the graph.",
            )

        def __check_user_defined_labels(
            edge_pbs: Iterable[graph_schema_pb2.Edge],
            edge_metadata_info: EdgeMetadataInfo,
            edge_usage_type: EdgeUsageType,
        ):
            for edge_pb in edge_pbs:
                edge_pbw = EdgePbWrapper(pb=edge_pb).dehydrate()
                edge_features = edge_pb.feature_values
                src_node_pbw, _ = get_dehydrated_node_pb_wrappers_from_edge_wrapper(
                    edge_pb_wrapper=edge_pbw,
                    graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
                )
                edge_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                    edge_pbw.condensed_edge_type
                ]
                self.assertIn(
                    src_node_pbw,
                    edge_metadata_info.feasible_adjacency_list_map,
                    f"Found a {edge_usage_type.value} edge {edge_pbw} in training samples supervision, but wasn't expecting any user-defined labels from source node {src_node_pbw}.",
                )
                self.assertIn(
                    edge_pbw,
                    edge_metadata_info.feasible_adjacency_list_map[src_node_pbw],
                    f"Found a {edge_usage_type.value} edge {edge_pbw} in training samples supervision, but wasn't expecting it from source node {src_node_pbw}'s preprocessor output.",
                )
                actual_num_edge_features = len(edge_features)
                expected_num_edge_features = len(
                    edge_metadata_info.edge_type_to_edge_to_features_map[edge_type][
                        edge_pbw
                    ]
                )
                self.assertEqual(
                    actual_num_edge_features,
                    expected_num_edge_features,
                    f"Found a {edge_usage_type.value} edge {edge_pbw} in training samples supervision with {actual_num_edge_features} features, but expected {expected_num_edge_features} from preprocessor output.",
                )

        def __check_valid_supervision_edge_type(
            edge_pbs: Iterable[graph_schema_pb2.Edge],
            gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        ):
            """
            Checks if the edge types in the provided edges are part of the supervision edge types.

            This method iterates over the provided edges, dehydrates them to get the edge type, and checks if this edge type is
            part of the supervision edge types defined in the gbml configuration. If an edge type is not part of the supervision
            edge types, an AssertionError is raised.

            Args:
                edge_pbs (Iterable[graph_schema_pb2.Edge]): Iterable of edges to check.
                gbml_config_pb_wrapper (GbmlConfigPbWrapper): Wrapper for the gbml configuration protobuf.

            Raises:
                AssertionError: If an edge type is not part of the supervision edge types.
            """
            supervision_edge_types = (
                gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
            )
            condensed_edge_type_to_edge_type_map = (
                gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map
            )
            for edge_pb in edge_pbs:
                edge_pbw = EdgePbWrapper(pb=edge_pb).dehydrate()
                edge_type = condensed_edge_type_to_edge_type_map[
                    edge_pbw.condensed_edge_type
                ]
                self.assertIn(
                    edge_type,
                    supervision_edge_types,
                    f"Found an edge {edge_pbw} in training samples supervision, but the edge type {edge_type} is not part of the supervision edge types: {supervision_edge_types}.",
                )

        def __check_num_positive_edges(
            training_samples: Iterable[NodeAnchorBasedLinkPredictionSample],
        ):
            """
            Checks that the number of positive edges match the number of positive samples specified in the subgraph sampler config.
            Positives can be user provided, so we don't check against the expected graph from preprocessor.
            """
            num_positive_samples = subgraph_sampler_config_pb.num_positive_samples
            for training_sample in training_samples:
                self.assertLessEqual(
                    len(training_sample.pos_edges),
                    num_positive_samples,
                    f"Found {len(training_sample.pos_edges)} positive edges in training samples, but expected {num_positive_samples}.",
                )

        __check_num_positive_edges(
            training_samples=training_samples,
        )

        main_edge_info = expected_graph_from_preprocessor.main_edge_info

        if should_check_user_defined_labels:
            logger.info(f"Checking pos and neg edges for UDL NABLP setting.")
            for sample_pb in training_samples:
                __check_user_defined_labels(
                    edge_pbs=sample_pb.pos_edges,
                    edge_metadata_info=expected_graph_from_preprocessor.pos_edge_info,
                    edge_usage_type=EdgeUsageType.POSITIVE,
                )
                __check_user_defined_labels(
                    edge_pbs=chain(sample_pb.neg_edges, sample_pb.hard_neg_edges),
                    edge_metadata_info=expected_graph_from_preprocessor.neg_edge_info,
                    edge_usage_type=EdgeUsageType.NEGATIVE,
                )
        else:
            logger.info(f"Checking pos and neg edges for standard NABLP setting.")
            for sample_pb in training_samples:
                for edge_pb in sample_pb.pos_edges:
                    edge_pbw = EdgePbWrapper(pb=edge_pb).dehydrate()
                    src_node_pbw, _ = get_dehydrated_node_pb_wrappers_from_edge_wrapper(
                        edge_pb_wrapper=edge_pbw,
                        graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
                    )
                    self.assertIn(
                        src_node_pbw,
                        main_edge_info.feasible_adjacency_list_map,
                        f"Found an edge {edge_pbw} in training samples supervision, but not expecting any edges from source node {src_node_pbw} in preprocessor output.",
                    )
                    self.assertIn(
                        edge_pbw,
                        main_edge_info.feasible_adjacency_list_map[src_node_pbw],
                        f"Found an edge {edge_pbw} in training samples supervision, but wasn't expecting it from source node {src_node_pbw}'s preprocessor output.",
                    )

        # Check that the destination node of each edge in the supervision exists as a valid node in the graph from preprocessor output
        # Also check that the edge type for positives/negatives provided in training samples is valid supervision edge type
        for sample_pb in training_samples:
            for edge_usage_type, edge_pbs in [
                (EdgeUsageType.POSITIVE, sample_pb.pos_edges),
                (
                    EdgeUsageType.NEGATIVE,
                    chain(sample_pb.neg_edges, sample_pb.hard_neg_edges),
                ),
            ]:
                __check_valid_supervision_edge_type(
                    edge_pbs=edge_pbs,
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                )
                for edge_pb in edge_pbs:
                    __check_supervision_edge_dst_node_exists_in_preprocessor_output(
                        edge_pb=edge_pb,
                        gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                        edge_usage_type=edge_usage_type,
                        expected_graph_from_preprocessor=expected_graph_from_preprocessor,
                    )

    def __check_subgraph_samples_for_invalid_nodes_or_edges(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        expected_graph_from_preprocessor: ExpectedGraphFromPreprocessor,
        rooted_node_neighborhood_sample_pbs: Iterable[RootedNodeNeighborhood] = list(),
        training_sample_pbs: Iterable[NodeAnchorBasedLinkPredictionSample] = list(),
    ):
        """
        Conducts a few key checks on validity of RNN and NABLP samples produced by SGS:
            - Ensure we don't have duplication of node PBs in each sample.
            - Ensure that the number of features in each node is as expected.
            - Ensure we don't have duplication of edge PBs in each sample.
            - Ensure that the number of features in each edge is as expected.
            - Ensure that the edge is in the feasible set of edges for the source node.
            - Ensure that every edge's adjacent nodes are contained in the neighborhood.
            - Ensure that the sample root node is contained in the neighborhood.

        Also, specifically for NABLP samples:
            - Ensure that the nodes in the supervision edges are contained in the neighborhood.
        """

        for sample_pb in chain(
            rooted_node_neighborhood_sample_pbs, training_sample_pbs
        ):
            assert isinstance(sample_pb, RootedNodeNeighborhood) or isinstance(
                sample_pb, NodeAnchorBasedLinkPredictionSample
            ), f"Expected a {RootedNodeNeighborhood.__name__} or {NodeAnchorBasedLinkPredictionSample.__name__}, but got {type(sample_pb)}."
            # Ensure we don't have duplication of node PBs in each sample.
            # Also ensure that the number of features in each node is as expected.
            sample_nodes: Set[NodePbWrapper] = set()
            for node_pb in sample_pb.neighborhood.nodes:
                node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                    CondensedNodeType(node_pb.condensed_node_type)
                ]
                node_pb_wrapper = NodePbWrapper(pb=node_pb).dehydrate()
                self.assertTrue(
                    node_pb_wrapper not in sample_nodes,
                    f"Found a duplicate node {node_pb_wrapper}; this should not happen.",
                )
                sample_nodes.add(node_pb_wrapper)
                self.assertEqual(
                    len(node_pb.feature_values),
                    len(
                        expected_graph_from_preprocessor.node_type_to_node_to_features_map[
                            node_type
                        ][
                            node_pb_wrapper
                        ]
                    ),
                    f"Node {node_pb_wrapper} in SGS output has misaligned number of features as in Data Preprocessor: {len(node_pb.feature_values)} instead of {len(expected_graph_from_preprocessor.node_type_to_node_to_features_map[node_type][node_pb_wrapper])} ",
                )

            # Ensure we don't have duplication of edge PBs in each sample.
            # Also ensure that the number of features in each edge is as expected.
            # Also ensure that the edge is in the feasible set of edges for the source node.
            # Also ensure that every edge in the neighborhood has its adjacent nodes in the neighborhood.
            sample_edges: Set[EdgePbWrapper] = set()
            for edge_pb in sample_pb.neighborhood.edges:
                edge_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                    CondensedEdgeType(edge_pb.condensed_edge_type)
                ]
                edge_pb_wrapper = EdgePbWrapper(pb=edge_pb).dehydrate()
                self.assertTrue(
                    edge_pb_wrapper not in sample_edges,
                    f"Found a duplicate edge {edge_pb_wrapper}; this should not happen.",
                )
                sample_edges.add(edge_pb_wrapper)
                self.assertEqual(
                    len(edge_pb.feature_values),
                    len(
                        expected_graph_from_preprocessor.main_edge_info.edge_type_to_edge_to_features_map[
                            edge_type
                        ][
                            edge_pb_wrapper
                        ]
                    ),
                    f"Edge {edge_pb_wrapper} in SGS output has misaligned number of features as in Data Preprocessor: {len(edge_pb.feature_values)} instead of {expected_graph_from_preprocessor.main_edge_info.edge_type_to_edge_to_features_map[edge_type][edge_pb_wrapper]} ",
                )
                (
                    src_node_pb_wrapper,
                    dst_node_pb_wrapper,
                ) = get_dehydrated_node_pb_wrappers_from_edge_wrapper(
                    edge_pb_wrapper=edge_pb_wrapper,
                    graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
                )

                # Make sure any edge (src=A, dst=B, edge_type=C, features=D) encountered in an SGS sample exists in the feasible edge set from running
                # SGS on this preprocessed data.
                self.assertIn(
                    src_node_pb_wrapper,
                    expected_graph_from_preprocessor.main_edge_info.feasible_adjacency_list_map,
                    f"Source node {src_node_pb_wrapper} not found in adjacency list constructed from preprocessor output, but encountered in SGS output",
                )
                self.assertIn(
                    edge_pb_wrapper,
                    expected_graph_from_preprocessor.main_edge_info.feasible_adjacency_list_map[
                        src_node_pb_wrapper
                    ],
                    f"Edge {edge_pb_wrapper} not found in adjacency list map for source node {src_node_pb_wrapper}",
                )

                # Make sure that any edge (src=A, dst=B) has its constituent nodes A and B in the neighborhood.
                self.assertIn(
                    src_node_pb_wrapper,
                    sample_nodes,
                    f"Found an edge {edge_pb_wrapper} with source node {src_node_pb_wrapper} not in the neighborhood.",
                )
                self.assertIn(
                    dst_node_pb_wrapper,
                    sample_nodes,
                    f"Found an edge {edge_pb_wrapper} with source node {src_node_pb_wrapper} not in the neighborhood.",
                )

            # Check that the sample root node is contained in the neighborhood.
            self.assertIn(
                NodePbWrapper(pb=sample_pb.root_node).dehydrate(),
                sample_nodes,
                f"Root node {sample_pb.root_node} not found in neighborhood.",
            )

        # For training samples, check that the nodes in the supervision edges are contained in the neighborhood.
        for training_sample_pb in training_sample_pbs:
            training_sample_neighborhood_nodes: Set[NodePbWrapper] = set(
                [
                    NodePbWrapper(pb=node_pb).dehydrate()
                    for node_pb in training_sample_pb.neighborhood.nodes
                ]
            )
            for edge_usage_type, edge_pbs in [
                (EdgeUsageType.POSITIVE, training_sample_pb.pos_edges),
                (
                    EdgeUsageType.NEGATIVE,
                    chain(
                        training_sample_pb.neg_edges, training_sample_pb.hard_neg_edges
                    ),
                ),
            ]:
                for edge_pb in edge_pbs:
                    edge_pbw = EdgePbWrapper(pb=edge_pb).dehydrate()
                    (
                        src_node_pbw,
                        dst_node_pbw,
                    ) = get_dehydrated_node_pb_wrappers_from_edge_wrapper(
                        edge_pb_wrapper=edge_pbw,
                        graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
                    )
                    self.assertIn(
                        src_node_pbw,
                        training_sample_neighborhood_nodes,
                        f"Found an edge {edge_pbw} in training samples {edge_usage_type} supervision, but couldn't find source node {src_node_pbw} inside the sample's neighborhood.",
                    )
                    self.assertIn(
                        dst_node_pbw,
                        training_sample_neighborhood_nodes,
                        f"Found an edge {edge_pbw} in training samples {edge_usage_type} supervision, but couldn't find destination node {dst_node_pbw} inside the sample's neighborhood.",
                    )

    def __build_input_graph_dst_node_to_src_node_type_to_in_edges_count_map(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        expected_graph_from_preprocessor: ExpectedGraphFromPreprocessor,
    ) -> Dict[NodePbWrapper, Dict[NodeType, int]]:
        """
        Builds a map counting the in-edge for each dst node from input graph, per src node type.
        Used to check the number of inbound edges for each node in the rooted subgraph samples
        """
        input_graph_dst_node_to_src_node_type_to_in_edges_count_map: Dict[
            NodePbWrapper, Dict[NodeType, int]
        ] = defaultdict(lambda: defaultdict(lambda: 0))
        for (
            src_node,
            edges,
        ) in (
            expected_graph_from_preprocessor.main_edge_info.feasible_adjacency_list_map.items()
        ):
            for edge in edges:
                (
                    condensed_src_node_type,
                    condensed_dst_node_type,
                ) = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_condensed_node_types[
                    edge.condensed_edge_type
                ]
                dst_node = NodePbWrapper(
                    graph_schema_pb2.Node(
                        node_id=edge.dst_node_id,
                        condensed_node_type=condensed_dst_node_type,
                    )
                )
                src_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                    condensed_src_node_type
                ]
                input_graph_dst_node_to_src_node_type_to_in_edges_count_map[dst_node][
                    src_node_type
                ] += 1
        return input_graph_dst_node_to_src_node_type_to_in_edges_count_map

    def __build_root_node_type_to_dst_node_type_to_sampling_op_map(
        self,
        subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
        is_subgraph_sampling_strategy_provided: bool = False,
    ) -> Dict[
        NodeType, Dict[NodeType, List[subgraph_sampling_strategy_pb2.SamplingOp]]
    ]:
        """
        Builds a map of samplingOps for each path in the subgraph sampling strategy. For each root node type, maps to the dst node type to the sampling ops.
        For each rooted node neighborhood sample, we can check that each dst node has the correct number of in-edges based on the sampling ops.
        """
        # map subgraph sampling strategy root node to dst node to sampling op
        root_node_type_to_dst_node_type_to_sampling_ops_map: Dict[
            NodeType, Dict[NodeType, List[subgraph_sampling_strategy_pb2.SamplingOp]]
        ] = defaultdict(lambda: defaultdict(list))
        if is_subgraph_sampling_strategy_provided:
            for (
                path
            ) in (
                subgraph_sampler_config_pb.subgraph_sampling_strategy.message_passing_paths.paths
            ):
                root_node_type = NodeType(path.root_node_type)
                for sampling_op in path.sampling_ops:
                    edge_type = sampling_op.edge_type
                    dst_node_type = edge_type.dst_node_type
                    root_node_type_to_dst_node_type_to_sampling_ops_map[root_node_type][
                        NodeType(dst_node_type)
                    ].append(sampling_op)
        return root_node_type_to_dst_node_type_to_sampling_ops_map

    def __build_sample_node_to_in_edge_map(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        rooted_node_neighborhood_samples: List[RootedNodeNeighborhood],
    ) -> Dict[NodePbWrapper, List[EdgePbWrapper]]:
        """
        Builds a map of NodePbWrapper to inbound EdgePbWrappers for each sample.
        """
        sample_node_to_in_edge_map = defaultdict(list)
        for sample in rooted_node_neighborhood_samples:
            for edge_pb in sample.neighborhood.edges:
                edge_pb_wrapper = EdgePbWrapper(pb=edge_pb).dehydrate()
                (
                    _,
                    dst_node_pb_wrapper,
                ) = get_dehydrated_node_pb_wrappers_from_edge_wrapper(
                    edge_pb_wrapper=edge_pb_wrapper,
                    graph_metadata_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
                )
                sample_node_to_in_edge_map[dst_node_pb_wrapper].append(edge_pb_wrapper)
        return sample_node_to_in_edge_map

    def __check_rooted_node_neighborhood_samples_for_neighbor_sampling_cap(
        self,
        rooted_node_neighborhood_samples: Iterable[RootedNodeNeighborhood],
        subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        expected_graph_from_preprocessor: ExpectedGraphFromPreprocessor,
        should_check_exact_number_of_in_edges: bool = False,
    ):
        """
        Check that the number of inbound edges for each node in the rooted subgraph samples is less than or equal to the # neighbors to sample specified in the SGS configuration.
        This check only applies to certain configurations, where it is impossible to exhibit such behavior given the underlying input graph structure and SGS configuration.
        For example, consider running a 2-hop SGS with up to 10 neighbors sampled per hop.  If this is run on a homogeneous cycle graph, where every node has at most 2 neighbors,
        then it is impossible for any node to have more than 4 inbound edges in the subgraph samples.

        Args:
            rooted_node_neighborhood_samples (Iterable[RootedNodeNeighborhood]): The rooted node neighborhood samples to check.
            subgraph_sampler_config_pb (gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig):
            gbml_config_pb_wrapper (GbmlConfigPbWrapper):
            expected_graph_from_preprocessor (ExpectedGraphFromPreprocessor): The expected graph from the preprocessor.
            should_check_exact_number_of_in_edges (bool, optional): A flag indicating if the exact number of inbound edges should be checked. Defaults to False.
                                                                **Only set to True when the SamplingOp DAG does not sampling the same node type more than once.
        """

        is_subgraph_sampling_strategy_provided = subgraph_sampler_config_pb.HasField(
            "subgraph_sampling_strategy"
        )

        #  Builds a map counting the in-edge for each dst node from input graph, per src node type.
        input_graph_dst_node_to_src_node_type_to_in_edges_count_map = (
            self.__build_input_graph_dst_node_to_src_node_type_to_in_edges_count_map(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                expected_graph_from_preprocessor=expected_graph_from_preprocessor,
            )
        )

        #  Builds a map of samplingOps for each path in the subgraph sampling strategy.
        #   For each root node type, maps to the dst node type to the sampling ops.
        root_node_type_to_dst_node_type_to_sampling_op_map = self.__build_root_node_type_to_dst_node_type_to_sampling_op_map(
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            is_subgraph_sampling_strategy_provided=is_subgraph_sampling_strategy_provided,
        )

        def __check_exact_number_of_in_edges_for_dst_node(
            dst_node_pb_wrapper: NodePbWrapper,
            sampling_ops: List[subgraph_sampling_strategy_pb2.SamplingOp],
            condensed_edge_type_to_in_edges_count: Dict[CondensedEdgeType, int],
        ):
            for sampling_op in sampling_ops:
                sampling_op_edge_type = EdgeType(
                    src_node_type=NodeType(sampling_op.edge_type.src_node_type),
                    dst_node_type=NodeType(sampling_op.edge_type.dst_node_type),
                    relation=Relation(sampling_op.edge_type.relation),
                )
                sampling_op_src_node_type = sampling_op_edge_type.src_node_type
                num_nodes_to_sample = min(
                    sampling_op.random_uniform.num_nodes_to_sample,
                    input_graph_dst_node_to_src_node_type_to_in_edges_count_map[
                        dst_node_pb_wrapper
                    ][sampling_op_src_node_type],
                )
                # check that the number of in-edges match exactly to the smaller number of (in edges in the input graph, the number of nodes to sample)
                self.assertEqual(
                    condensed_edge_type_to_in_edges_count[
                        gbml_config_pb_wrapper.graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[
                            sampling_op_edge_type
                        ]
                    ],  # number of returned in-edges for the dst node
                    num_nodes_to_sample,  # number of expected in-edges for the dst node
                    f"Found {len(in_edges)} inbound edges for node {dst_node_pb_wrapper.node_id} {dst_node_pb_wrapper}, but expected {num_nodes_to_sample} edges given SGS configuration. Sample: {sample}",
                )

        def __check_number_of_in_edges_for_dst_node(
            dst_node_pb_wrapper: NodePbWrapper,
            sampling_ops: List[subgraph_sampling_strategy_pb2.SamplingOp],
            condensed_edge_type_to_in_edges_count: Dict[CondensedEdgeType, int],
        ):
            condensed_edge_type_to_max_num_nodes_to_sample: Dict[
                CondensedEdgeType, int
            ] = defaultdict(lambda: 0)
            for sampling_op in sampling_ops:
                edge_type = EdgeType(
                    src_node_type=NodeType(sampling_op.edge_type.src_node_type),
                    dst_node_type=NodeType(sampling_op.edge_type.dst_node_type),
                    relation=Relation(sampling_op.edge_type.relation),
                )
                condensed_edge_type = CondensedEdgeType(
                    gbml_config_pb_wrapper.graph_metadata_pb_wrapper.edge_type_to_condensed_edge_type_map[
                        edge_type
                    ]
                )
                condensed_edge_type_to_max_num_nodes_to_sample[
                    condensed_edge_type
                ] += sampling_op.random_uniform.num_nodes_to_sample
            # check that the number of in-edges for each dst node is less than or equal to the added number of nodes to sample for the edge type for all samplingOps
            for (
                condensed_edge_type,
                max_num_nodes_to_sample,
            ) in condensed_edge_type_to_max_num_nodes_to_sample.items():
                self.assertLessEqual(
                    condensed_edge_type_to_in_edges_count[condensed_edge_type],
                    max_num_nodes_to_sample,
                    f"Found {condensed_edge_type_to_in_edges_count[condensed_edge_type]} inbound edges for node {dst_node_pb_wrapper.node_id} {dst_node_pb_wrapper}, but expected at most {max_num_nodes_to_sample} edges given SGS configuration. Sample: {sample}",
                )
            # check that the number of in-edges for each dst node is less than or equal to the number of in-edges in the input graph
            for (
                condensed_edge_type,
                in_edges_count,
            ) in condensed_edge_type_to_in_edges_count.items():
                self.assertLessEqual(
                    in_edges_count,
                    input_graph_dst_node_to_src_node_type_to_in_edges_count_map[
                        dst_node_pb_wrapper
                    ][
                        gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                            condensed_edge_type
                        ].src_node_type
                    ],
                    f"Found {in_edges_count} inbound edges for node {dst_node_pb_wrapper.node_id} {dst_node_pb_wrapper}, but expected at most {input_graph_dst_node_to_src_node_type_to_in_edges_count_map[dst_node_pb_wrapper][gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[condensed_edge_type].src_node_type]} edges given SGS configuration. Sample: {sample}",
                )

        def __build_condensed_edge_type_to_in_edges_count_map(
            in_edges: List[EdgePbWrapper],
        ):
            condensed_edge_type_to_in_edges_count: Dict[
                CondensedEdgeType, int
            ] = defaultdict(lambda: 0)
            for edge_pb_wrapper in in_edges:
                condensed_edge_type_to_in_edges_count[
                    edge_pb_wrapper.condensed_edge_type
                ] += 1
            return condensed_edge_type_to_in_edges_count

        # check per RNN sample
        for sample in rooted_node_neighborhood_samples:
            root_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                CondensedNodeType(sample.root_node.condensed_node_type)
            ]

            sample_node_to_in_edge_map = self.__build_sample_node_to_in_edge_map(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                rooted_node_neighborhood_samples=[sample],
            )

            if is_subgraph_sampling_strategy_provided:
                # NOTE:
                # - test will only work when the same node type is not repeated in the samplingDAG, as we check exact numbers of in-edges
                # - num nodes that are sampled on the in-edge for each dst node should be the num_nodes_to_sample defined in samplingOp
                #    or the number of in-edges in the graph, whichever is smaller
                # - We assume that there are no self loops
                # NOTE: in the test we assume RandomUniform sampling
                for (
                    dst_node_pb_wrapper,
                    in_edges,
                ) in sample_node_to_in_edge_map.items():
                    dst_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                        dst_node_pb_wrapper.condensed_node_type
                    ]
                    sampling_ops = root_node_type_to_dst_node_type_to_sampling_op_map[
                        root_node_type
                    ][dst_node_type]
                    condensed_edge_type_to_in_edges_count = (
                        __build_condensed_edge_type_to_in_edges_count_map(
                            in_edges=in_edges
                        )
                    )

                    if (
                        should_check_exact_number_of_in_edges
                    ):  # When checking for exact number of in-edges, SamplingOp of the same edge type can not be repeated in the samplingDAG
                        __check_exact_number_of_in_edges_for_dst_node(
                            dst_node_pb_wrapper=dst_node_pb_wrapper,
                            sampling_ops=sampling_ops,
                            condensed_edge_type_to_in_edges_count=condensed_edge_type_to_in_edges_count,
                        )
                    else:
                        __check_number_of_in_edges_for_dst_node(
                            dst_node_pb_wrapper=dst_node_pb_wrapper,
                            sampling_ops=sampling_ops,
                            condensed_edge_type_to_in_edges_count=condensed_edge_type_to_in_edges_count,
                        )

            else:
                # NOTE: we use this check for homogeneous graph, also for cases where subgraph_sampling_strategy is not provided
                # Check that no node has more than the expected number of edges specified in sampling.
                for dst_node_pb_wrapper, in_edges in sample_node_to_in_edge_map.items():
                    self.assertLessEqual(
                        len(in_edges),
                        subgraph_sampler_config_pb.num_neighbors_to_sample,
                        f"Found {len(in_edges)} inbound edges for node {dst_node_pb_wrapper}, but expected at most {subgraph_sampler_config_pb.num_neighbors_to_sample} edges given SGS configuration.",
                    )

    def __check_nodes_do_not_have_nablp_training_samples(
        self,
        training_samples: List[NodeAnchorBasedLinkPredictionSample],
        nodes_to_check: List[NodePbWrapper],
    ):
        """
        Check that the provided nodes do not appear in the training samples.
        """

        nodes_with_training_samples: Set[NodePbWrapper] = set(
            [
                NodePbWrapper(pb=training_sample.root_node).dehydrate()
                for training_sample in training_samples
            ]
        )
        for node_pb_wrapper in nodes_to_check:
            self.assertNotIn(
                node_pb_wrapper,
                nodes_with_training_samples,
                f"Found node {node_pb_wrapper} in training samples, but expected it to not be included.",
            )

    def __run_and_check_nablp_sgs_on_homogeneous_toy_graph(
        self,
        mocked_dataset_info: MockedDatasetInfo,
        subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
        isolated_node_ids: List[int] = [],
        should_check_user_defined_labels: bool = False,
    ):
        """
        Helper function to bundle certain checks run on the homogeneous toy graph for NABLP tasks.
        """

        task_name = mocked_dataset_info.name
        artifact_metadata = get_mocked_dataset_artifact_metadata()[task_name]
        toy_graph_gbml_config_uri = artifact_metadata.frozen_gbml_config_uri

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=toy_graph_gbml_config_uri
            )
        )

        # Run SGS with a customized SubgraphSamplerConfig proto and check the validity of the output
        (
            expected_graph_from_preprocessor,
            rooted_node_neighborhood_samples,
            training_samples,
        ) = self.__run_and_check_node_anchor_based_link_prediction_task_sgs_validity(
            gbml_config_uri=toy_graph_gbml_config_uri,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            should_check_user_defined_labels=should_check_user_defined_labels,
        )

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=toy_graph_gbml_config_uri
            )
        )

        # Check that we have fewer training samples than nodes of type src_node_type since isolated nodes exist in the preprocessor output.
        # This check only applies when isolated nodes are not included in training.
        supervision_edge_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_pb.node_anchor_based_link_prediction_task_metadata.supervision_edge_types
        )
        if (
            not gbml_config_pb_wrapper.shared_config.should_include_isolated_nodes_in_training
        ):
            for supervision_edge_type in supervision_edge_types:
                src_node_type = NodeType(supervision_edge_type.src_node_type)
                self.assertLess(
                    len(training_samples),
                    len(
                        expected_graph_from_preprocessor.node_type_to_node_to_features_map[
                            src_node_type
                        ]
                    ),
                    f"Found {len(training_samples)} training samples, but expected fewer than {len(expected_graph_from_preprocessor.node_type_to_node_to_features_map[src_node_type])}, since there are only {len(expected_graph_from_preprocessor.node_type_to_node_to_features_map[src_node_type])} nodes of type {src_node_type}, and some are isolated.",
                )

                # Check that specific nodes which are known to be isolated are not associated with training samples.
                condensed_src_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                    src_node_type
                ]
                self.__check_nodes_do_not_have_nablp_training_samples(
                    training_samples=training_samples,
                    nodes_to_check=[
                        NodePbWrapper(
                            pb=graph_schema_pb2.Node(
                                node_id=node_id,
                                condensed_node_type=condensed_src_node_type,
                            )
                        )
                        for node_id in isolated_node_ids
                    ],
                )

        # Check that rooted samples for src_node_type (also = to dst_node_type for homogeneous setting) don't have too many neighbors.
        self.__check_rooted_node_neighborhood_samples_for_neighbor_sampling_cap(
            rooted_node_neighborhood_samples=rooted_node_neighborhood_samples[
                src_node_type
            ],
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            expected_graph_from_preprocessor=expected_graph_from_preprocessor,
        )

    def test_nablp_sgs_on_homogeneous_toy_graph(
        self,
    ):
        subgraph_sampler_config_pb = (
            gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig(
                num_hops=TEST_NUM_HOPS,
                num_neighbors_to_sample=TEST_NUM_NEIGHBORS_TO_SAMPLE,
                num_positive_samples=TEST_NUM_POSITIVE_SAMPLES,
            )
        )

        self.__run_and_check_nablp_sgs_on_homogeneous_toy_graph(
            mocked_dataset_info=TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            isolated_node_ids=[13, 14, 23, 24, 25],
            # these are isolated node IDs in the homogeneous toy graph
        )

    def test_nablp_sgs_on_homogeneous_toy_graph_with_udl(
        self,
    ):
        subgraph_sampler_config_pb = gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig(
            num_hops=TEST_NUM_HOPS,
            num_neighbors_to_sample=TEST_NUM_NEIGHBORS_TO_SAMPLE,
            num_positive_samples=TEST_NUM_POSITIVE_SAMPLES,
            num_user_defined_positive_samples=TEST_NUM_USER_DEFINED_POSITIVE_SAMPLES,
            num_user_defined_negative_samples=TEST_NUM_USER_DEFINED_NEGATIVE_SAMPLES,
        )
        self.__run_and_check_nablp_sgs_on_homogeneous_toy_graph(
            mocked_dataset_info=TOY_GRAPH_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            isolated_node_ids=[13, 14, 24, 25],
            should_check_user_defined_labels=True,
            # these are isolated node IDs without user-defined +/- edges in the homogeneous toy graph
        )

    def __run_and_check_nablp_sgs_on_heterogeneous_toy_graph(
        self,
        mocked_dataset_info: MockedDatasetInfo,
        subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
    ):
        """
        Helper function to bundle certain checks run on the heterogeneous toy graph for NABLP tasks.
        """

        task_name = mocked_dataset_info.name
        artifact_metadata = get_mocked_dataset_artifact_metadata()[task_name]
        toy_graph_gbml_config_uri = artifact_metadata.frozen_gbml_config_uri

        # Run SGS with a customized SubgraphSamplerConfig proto and check the validity of the output
        (
            expected_graph_from_preprocessor,
            rooted_node_neighborhood_samples,
            training_samples,
        ) = self.__run_and_check_node_anchor_based_link_prediction_task_sgs_validity(
            gbml_config_uri=toy_graph_gbml_config_uri,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            use_spark35=True,  # Only spark35 implementation supports heterogeneous graph sampling
        )

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=toy_graph_gbml_config_uri
            )
        )

        # Check that we have fewer training samples than nodes since isolated nodes exist in the preprocessor output.
        supervision_edge_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_pb.node_anchor_based_link_prediction_task_metadata.supervision_edge_types
        )
        assert (
            len(supervision_edge_types) == 1
        ), "This graph only has 1 supervision edge type."
        src_node_type, dst_node_type = NodeType(
            supervision_edge_types[0].src_node_type
        ), NodeType(supervision_edge_types[0].dst_node_type)

        # Check that we have fewer training samples than nodes of type src_node_type since isolated nodes exist in the preprocessor output.
        # This check only applies when isolated nodes are not included in training.
        if (
            not gbml_config_pb_wrapper.shared_config.should_include_isolated_nodes_in_training
        ):
            self.assertLess(
                len(training_samples),
                len(
                    expected_graph_from_preprocessor.node_type_to_node_to_features_map[
                        src_node_type
                    ]
                ),
                f"Found {len(training_samples)} training samples, but expected fewer than {len(expected_graph_from_preprocessor.node_type_to_node_to_features_map[src_node_type])}, since there are only {len(expected_graph_from_preprocessor.node_type_to_node_to_features_map[src_node_type])} nodes of type {src_node_type}, and some are isolated.",
            )

            # Check that specific nodes which are known to be isolated are not associated with training samples.
            condensed_src_node_type = gbml_config_pb_wrapper.graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                src_node_type
            ]
            self.__check_nodes_do_not_have_nablp_training_samples(
                training_samples=training_samples,
                nodes_to_check=[
                    NodePbWrapper(
                        pb=graph_schema_pb2.Node(
                            node_id=node_id, condensed_node_type=condensed_src_node_type
                        )
                    )
                    for node_id in [8]
                ],
            )

        # Check that rooted samples for src_node_type and dst_node_type don't have too many neighbors.
        self.__check_rooted_node_neighborhood_samples_for_neighbor_sampling_cap(
            rooted_node_neighborhood_samples=rooted_node_neighborhood_samples[
                src_node_type
            ],
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            expected_graph_from_preprocessor=expected_graph_from_preprocessor,
            should_check_exact_number_of_in_edges=True,
        )
        self.__check_rooted_node_neighborhood_samples_for_neighbor_sampling_cap(
            rooted_node_neighborhood_samples=rooted_node_neighborhood_samples[
                dst_node_type
            ],
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            expected_graph_from_preprocessor=expected_graph_from_preprocessor,
            should_check_exact_number_of_in_edges=True,
        )

    def test_nablp_sgs_on_heterogeneous_toy_graph(
        self,
    ):
        # See python/gigl/src/mocking/mocking_assets/bipartite_toy_graph_data.yaml for graph def
        # and python/gigl/src/mocking/mocking_assets/bipartite_toy_graph_data.png for visualization
        subgraph_sampling_strategy_pb = subgraph_sampling_strategy_pb2.SubgraphSamplingStrategy(
            message_passing_paths=subgraph_sampling_strategy_pb2.MessagePassingPathStrategy(
                paths=[
                    subgraph_sampling_strategy_pb2.MessagePassingPath(
                        root_node_type="user",
                        sampling_ops=[
                            subgraph_sampling_strategy_pb2.SamplingOp(
                                op_name="sample_stories_from_user",
                                edge_type=graph_schema_pb2.EdgeType(
                                    src_node_type="story",
                                    relation="to",
                                    dst_node_type="user",
                                ),
                                input_op_names=[],
                                random_uniform=subgraph_sampling_strategy_pb2.RandomUniform(
                                    num_nodes_to_sample=TEST_NUM_NEIGHBORS_TO_SAMPLE,
                                ),
                            ),
                            subgraph_sampling_strategy_pb2.SamplingOp(
                                op_name="sample_users_from_story",
                                edge_type=graph_schema_pb2.EdgeType(
                                    src_node_type="user",
                                    relation="to",
                                    dst_node_type="story",
                                ),
                                input_op_names=["sample_stories_from_user"],
                                random_uniform=subgraph_sampling_strategy_pb2.RandomUniform(
                                    num_nodes_to_sample=TEST_NUM_NEIGHBORS_TO_SAMPLE,
                                ),
                            ),
                        ],
                    ),
                    subgraph_sampling_strategy_pb2.MessagePassingPath(
                        root_node_type="story",
                        sampling_ops=[
                            subgraph_sampling_strategy_pb2.SamplingOp(
                                op_name="sample_users_from_story",
                                edge_type=graph_schema_pb2.EdgeType(
                                    src_node_type="user",
                                    relation="to",
                                    dst_node_type="story",
                                ),
                                input_op_names=[],
                                random_uniform=subgraph_sampling_strategy_pb2.RandomUniform(
                                    num_nodes_to_sample=TEST_NUM_NEIGHBORS_TO_SAMPLE,
                                ),
                            ),
                            subgraph_sampling_strategy_pb2.SamplingOp(
                                op_name="sample_stories_from_user",
                                edge_type=graph_schema_pb2.EdgeType(
                                    src_node_type="story",
                                    relation="to",
                                    dst_node_type="user",
                                ),
                                input_op_names=["sample_users_from_story"],
                                random_uniform=subgraph_sampling_strategy_pb2.RandomUniform(
                                    num_nodes_to_sample=TEST_NUM_NEIGHBORS_TO_SAMPLE,
                                ),
                            ),
                        ],
                    ),
                ]
            )
        )
        subgraph_sampler_config_pb = (
            gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig(
                num_positive_samples=TEST_NUM_POSITIVE_SAMPLES,
                subgraph_sampling_strategy=subgraph_sampling_strategy_pb,
                graph_db_config=gbml_config_pb2.GbmlConfig.GraphDBConfig(
                    graph_db_args={
                        "use_local_sampler": "true",
                    }
                ),
            )
        )
        self.__run_and_check_nablp_sgs_on_heterogeneous_toy_graph(
            mocked_dataset_info=HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
        )
        print("test_nablp_sgs_on_heterogeneous_toy_graph ran successfully.")

    def __run_and_check_node_based_task_sgs_validity(
        self,
        gbml_config_uri: Uri,
        subgraph_sampler_config_pb: gbml_config_pb2.GbmlConfig.DatasetConfig.SubgraphSamplerConfig,
    ) -> Tuple[
        ExpectedGraphFromPreprocessor,
        List[RootedNodeNeighborhood],
        List[RootedNodeNeighborhood],
    ]:
        """
        Run the SGS pipeline for a Node-Based Task using the provided GbmlConfig and check the validity of the output.
        These validity checks should pass for *all* graphs.  The checks test:
            - The number of labeled rooted samples outputted by SGS is greater than 0.
            - The number of labeled + unlabeled samples outputted by SGS is equal to the number of nodes outputted by the preprocessor.
            - All subgraph samples are valid (i.e. no duplicate or inconsistent nodes or edges, correct number of features, root node included).
        """

        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=gbml_config_uri
            )
        )

        # TODO(nshah-sc): this should be removed once SGS properly outputs the right # of RNNs for graphs where is_graph_directed = True.
        # Currently, we will only allow this test to run on graphs where is_graph_directed = False.  SGS will bidirectionalize these graphs.
        # i.e. if the preprocessed data only has (src, dst, edge_type), then SGS will add (dst, src, edge_type) as well.
        assert not gbml_config_pb_wrapper.shared_config.is_graph_directed, (
            "This test is currently buggy for is_graph_directed=True. In particular, we can surprisingly encounter duplicated RNNs.",
        )

        (
            expected_graph_from_preprocessor,
            updated_gbml_config_wrapper,
        ) = self.__compile_and_run_sgs_pipeline_locally(
            gbml_config_uri=gbml_config_uri,
            subgraph_sampler_config_pb=subgraph_sampler_config_pb,
        )
        gbml_config_pb_wrapper = updated_gbml_config_wrapper

        # We read the output that the SGS component produced
        (
            labeled_rooted_node_neighborhood_samples,
            unlabeled_rooted_node_neighborhood_samples,
        ) = read_output_node_based_task_samples_from_subgraph_sampler(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper
        )

        supervision_node_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.task_metadata_pb.node_based_task_metadata.supervision_node_types
        )
        assert (
            len(supervision_node_types) == 1
        ), "Only 1 supervision node type is supported for this test."
        supervision_node_type = NodeType(supervision_node_types[0])

        # Check that the number of labeled rooted node neighborhoods produced is greater than 0.
        self.assertGreater(
            len(labeled_rooted_node_neighborhood_samples),
            0,
            "Missing labeled rooted neighborhood samples from SGS output",
        )

        # Check that the number of rooted node neighborhoods produced is equal to the number of nodes of that type encountered in the preprocessor output.
        total_rooted_node_neighborhood_samples = len(
            labeled_rooted_node_neighborhood_samples
        ) + len(unlabeled_rooted_node_neighborhood_samples)
        expected_nodes_of_supervision_node_type = len(
            expected_graph_from_preprocessor.node_type_to_node_to_features_map[
                supervision_node_type
            ]
        )
        self.assertEquals(
            total_rooted_node_neighborhood_samples,
            expected_nodes_of_supervision_node_type,
            f"Found {total_rooted_node_neighborhood_samples} rooted samples from SGS output, but found {expected_nodes_of_supervision_node_type} nodes from Data Preprocessor output",
        )
        self.__check_subgraph_samples_for_invalid_nodes_or_edges(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            expected_graph_from_preprocessor=expected_graph_from_preprocessor,
            rooted_node_neighborhood_sample_pbs=chain(
                labeled_rooted_node_neighborhood_samples,
                unlabeled_rooted_node_neighborhood_samples,
            ),
        )

        return (
            expected_graph_from_preprocessor,
            labeled_rooted_node_neighborhood_samples,
            unlabeled_rooted_node_neighborhood_samples,
        )
