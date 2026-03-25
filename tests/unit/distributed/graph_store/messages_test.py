from gigl.distributed.graph_store.messages import (
    FetchABLPInputRequest,
    FetchNodesRequest,
)
from gigl.distributed.graph_store.sharding import ServerSlice
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from tests.test_assets.test_case import TestCase


class TestFetchNodesRequest(TestCase):
    def test_validate_accepts_rank_world_size(self) -> None:
        request = FetchNodesRequest(rank=0, world_size=2)
        request.validate()

    def test_validate_accepts_server_slice(self) -> None:
        request = FetchNodesRequest(
            server_slice=ServerSlice(
                server_rank=0,
                start_num=0,
                start_den=2,
                end_num=1,
                end_den=2,
            )
        )
        request.validate()

    def test_validate_rejects_partial_rank_world_size(self) -> None:
        with self.assertRaises(ValueError):
            FetchNodesRequest(rank=0).validate()

        with self.assertRaises(ValueError):
            FetchNodesRequest(world_size=2).validate()

    def test_validate_rejects_mixed_sharding_modes(self) -> None:
        with self.assertRaises(ValueError):
            FetchNodesRequest(
                rank=0,
                world_size=2,
                server_slice=ServerSlice(
                    server_rank=0,
                    start_num=0,
                    start_den=2,
                    end_num=1,
                    end_den=2,
                ),
            ).validate()


class TestFetchABLPRequest(TestCase):
    def test_validate_accepts_rank_world_size(self) -> None:
        request = FetchABLPInputRequest(
            split="train",
            rank=0,
            world_size=2,
            node_type=NodeType("user"),
            supervision_edge_type=EdgeType(
                src_node_type=NodeType("user"),
                relation=Relation("to"),
                dst_node_type=NodeType("story"),
            ),
        )
        request.validate()

    def test_validate_accepts_server_slice(self) -> None:
        request = FetchABLPInputRequest(
            split="train",
            node_type=NodeType("user"),
            supervision_edge_type=EdgeType(
                src_node_type=NodeType("user"),
                relation=Relation("to"),
                dst_node_type=NodeType("story"),
            ),
            server_slice=ServerSlice(
                server_rank=0,
                start_num=0,
                start_den=2,
                end_num=1,
                end_den=2,
            ),
        )
        request.validate()

    def test_validate_rejects_partial_rank_world_size(self) -> None:
        with self.assertRaises(ValueError):
            FetchABLPInputRequest(
                split="train",
                rank=0,
                node_type=NodeType("user"),
                supervision_edge_type=EdgeType(
                    src_node_type=NodeType("user"),
                    relation=Relation("to"),
                    dst_node_type=NodeType("story"),
                ),
            ).validate()

        with self.assertRaises(ValueError):
            FetchABLPInputRequest(
                split="train",
                world_size=2,
                node_type=NodeType("user"),
                supervision_edge_type=EdgeType(
                    src_node_type=NodeType("user"),
                    relation=Relation("to"),
                    dst_node_type=NodeType("story"),
                ),
            ).validate()

    def test_validate_rejects_mixed_sharding_modes(self) -> None:
        with self.assertRaises(ValueError):
            FetchABLPInputRequest(
                split="train",
                rank=0,
                world_size=2,
                node_type=NodeType("user"),
                supervision_edge_type=EdgeType(
                    src_node_type=NodeType("user"),
                    relation=Relation("to"),
                    dst_node_type=NodeType("story"),
                ),
                server_slice=ServerSlice(
                    server_rank=0,
                    start_num=0,
                    start_den=2,
                    end_num=1,
                    end_den=2,
                ),
            ).validate()
