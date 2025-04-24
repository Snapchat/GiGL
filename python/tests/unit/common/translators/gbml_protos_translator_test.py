import unittest
from typing import cast

from gigl.common.logger import Logger
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.graph_builder.pyg_graph_data import PygGraphData
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.model import GraphBackend
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from tests.test_assets.celeb_test_graph.assets import (
    get_celeb_expected_pyg_graph,
    get_celeb_graph_metadata_pb2,
    get_celeb_khop_subgraph_for_node1,
)

logger = Logger()


class GbmlProtosTranslatorTest(unittest.TestCase):
    def test_parsing_graph_data_from_KHopSubgraph(self):
        _, khop_subgraph = get_celeb_khop_subgraph_for_node1()
        graph_metadata = get_celeb_graph_metadata_pb2()

        graph_builder = GraphBuilderFactory.get_graph_builder(
            backend_name=GraphBackend.PYG
        )
        graph_data = cast(
            PygGraphData,
            GbmlProtosTranslator.graph_data_from_GraphPb(
                samples=[khop_subgraph],
                graph_metadata_pb_wrapper=GraphMetadataPbWrapper(graph_metadata),
                builder=graph_builder,
            ),
        )

        # We build expected graph:
        expected_graph_data = get_celeb_expected_pyg_graph()

        self.assertEqual(graph_data, expected_graph_data)
