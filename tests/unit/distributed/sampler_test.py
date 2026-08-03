import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import torch
from graphlearn_torch.sampler import NeighborOutput, NodeSamplerInput

from gigl.distributed.base_sampler import (
    BaseDistNeighborSampler,
    SampleLoopInputs,
    _stable_unique_preserve_order,
)
from gigl.distributed.dist_neighbor_sampler import DistNeighborSampler
from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_ITEM = NodeType("item")
_BUYS = Relation("buys")
_CLICKS = Relation("clicks")
_FRIEND = Relation("friend")
_USER_BUYS_ITEM = EdgeType(_USER, _BUYS, _ITEM)
_USER_CLICKS_ITEM = EdgeType(_USER, _CLICKS, _ITEM)
_USER_FRIEND_USER = EdgeType(_USER, _FRIEND, _USER)


def _build_sampler_input(
    num_nodes: int = 4,
) -> ABLPNodeSamplerInput:
    """Builds a simple ABLPNodeSamplerInput for testing with two edge types."""
    node = torch.arange(num_nodes)
    positive_label_by_edge_types = {
        _USER_BUYS_ITEM: torch.arange(100, 100 + num_nodes),
        _USER_CLICKS_ITEM: torch.arange(200, 200 + num_nodes),
    }
    negative_label_by_edge_types = {
        _USER_BUYS_ITEM: torch.arange(300, 300 + num_nodes),
        _USER_CLICKS_ITEM: torch.arange(400, 400 + num_nodes),
    }
    return ABLPNodeSamplerInput(
        node=node,
        input_type=_USER,
        positive_label_by_edge_types=positive_label_by_edge_types,
        negative_label_by_edge_types=negative_label_by_edge_types,
    )


class TestABLPNodeSamplerInput(TestCase):
    def test_construction_and_properties(self) -> None:
        node = torch.tensor([10, 20, 30])
        positive_labels = {_USER_BUYS_ITEM: torch.tensor([1, 2, 3])}
        negative_labels = {_USER_CLICKS_ITEM: torch.tensor([4, 5, 6])}

        sampler_input = ABLPNodeSamplerInput(
            node=node,
            input_type=_USER,
            positive_label_by_edge_types=positive_labels,
            negative_label_by_edge_types=negative_labels,
        )

        self.assert_tensor_equality(sampler_input.node, node)
        self.assertEqual(sampler_input.input_type, _USER)
        self.assertEqual(
            set(sampler_input.positive_label_by_edge_types.keys()),
            {_USER_BUYS_ITEM},
        )
        self.assert_tensor_equality(
            sampler_input.positive_label_by_edge_types[_USER_BUYS_ITEM],
            positive_labels[_USER_BUYS_ITEM],
        )
        self.assertEqual(
            set(sampler_input.negative_label_by_edge_types.keys()),
            {_USER_CLICKS_ITEM},
        )
        self.assert_tensor_equality(
            sampler_input.negative_label_by_edge_types[_USER_CLICKS_ITEM],
            negative_labels[_USER_CLICKS_ITEM],
        )

    def test_len(self) -> None:
        for num_nodes in (1, 4, 10):
            sampler_input = _build_sampler_input(num_nodes=num_nodes)
            self.assertEqual(len(sampler_input), num_nodes)

    def test_getitem_with_tensor_index(self) -> None:
        sampler_input = _build_sampler_input(num_nodes=4)
        index = torch.tensor([0, 2])
        sliced = sampler_input[index]

        self.assertIsInstance(sliced, ABLPNodeSamplerInput)
        self.assert_tensor_equality(sliced.node, torch.tensor([0, 2]))
        self.assertEqual(sliced.input_type, _USER)
        self.assert_tensor_equality(
            sliced.positive_label_by_edge_types[_USER_BUYS_ITEM],
            torch.tensor([100, 102]),
        )
        self.assert_tensor_equality(
            sliced.positive_label_by_edge_types[_USER_CLICKS_ITEM],
            torch.tensor([200, 202]),
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_BUYS_ITEM],
            torch.tensor([300, 302]),
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_CLICKS_ITEM],
            torch.tensor([400, 402]),
        )

    def test_getitem_with_list_index(self) -> None:
        sampler_input = _build_sampler_input(num_nodes=4)
        sliced = sampler_input[[1]]

        self.assertIsInstance(sliced, ABLPNodeSamplerInput)
        self.assertTrue(torch.equal(sliced.node, torch.tensor([1])))
        self.assert_tensor_equality(
            sliced.positive_label_by_edge_types[_USER_BUYS_ITEM], torch.tensor([101])
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_CLICKS_ITEM], torch.tensor([401])
        )

    def test_share_memory(self) -> None:
        sampler_input = _build_sampler_input(num_nodes=3)
        result = sampler_input.share_memory()

        self.assertIs(result, sampler_input)
        self.assertTrue(sampler_input.node.is_shared())
        self.assertTrue(
            sampler_input.positive_label_by_edge_types[_USER_BUYS_ITEM].is_shared()
        )
        self.assertTrue(
            sampler_input.positive_label_by_edge_types[_USER_CLICKS_ITEM].is_shared()
        )
        self.assertTrue(
            sampler_input.negative_label_by_edge_types[_USER_BUYS_ITEM].is_shared()
        )
        self.assertTrue(
            sampler_input.negative_label_by_edge_types[_USER_CLICKS_ITEM].is_shared()
        )


def _build_sampler_stub(edge_dir: str = "out") -> BaseDistNeighborSampler:
    """Build a minimal BaseGiGLSampler stub for testing shared utilities."""
    sampler = BaseDistNeighborSampler.__new__(BaseDistNeighborSampler)
    sampler.device = torch.device("cpu")
    sampler.edge_dir = edge_dir
    return sampler


class TestBaseGiGLSamplerPreparation(TestCase):
    def test_stable_unique_preserves_first_occurrence_order(self) -> None:
        self.assert_tensor_equality(
            _stable_unique_preserve_order(torch.tensor([7, 3, 7, 5, 3, 9])),
            torch.tensor([7, 3, 5, 9]),
        )

    def test_stable_unique_requires_one_dimensional_tensor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected a 1-D tensor"):
            _stable_unique_preserve_order(torch.tensor([[1, 2], [3, 4]]))

    def test_prepare_ablp_inputs_dedupes_same_type_seeds_and_keeps_anchors_first(
        self,
    ) -> None:
        sampler = _build_sampler_stub(edge_dir="out")
        positive_labels = {_USER_FRIEND_USER: torch.tensor([11, 12, -1, 13])}
        negative_labels = {_USER_FRIEND_USER: torch.tensor([13, 14, 10, -1])}
        sampler_input = ABLPNodeSamplerInput(
            node=torch.tensor([10, 11, 10]),
            input_type=_USER,
            positive_label_by_edge_types=positive_labels,
            negative_label_by_edge_types=negative_labels,
        )

        sample_loop_inputs = sampler._prepare_ablp_inputs(
            inputs=sampler_input,
            input_seeds=sampler_input.node,
            input_type=_USER,
        )

        nodes_to_sample = sample_loop_inputs.nodes_to_sample
        assert isinstance(nodes_to_sample, Mapping)
        self.assertEqual(set(nodes_to_sample.keys()), {_USER})
        self.assert_tensor_equality(
            nodes_to_sample[_USER],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
            torch.tensor([10, 11, 12, 13, 14]),
        )
        self.assert_tensor_equality(
            sample_loop_inputs.metadata[
                f"{POSITIVE_LABEL_METADATA_KEY}{str(tuple(_USER_FRIEND_USER))}"
            ],
            positive_labels[_USER_FRIEND_USER],
        )
        self.assert_tensor_equality(
            sample_loop_inputs.metadata[
                f"{NEGATIVE_LABEL_METADATA_KEY}{str(tuple(_USER_FRIEND_USER))}"
            ],
            negative_labels[_USER_FRIEND_USER],
        )

    def test_prepare_ablp_inputs_dedupes_cross_type_supervision_nodes(self) -> None:
        sampler = _build_sampler_stub(edge_dir="out")
        sampler_input = ABLPNodeSamplerInput(
            node=torch.tensor([4, 5]),
            input_type=_USER,
            positive_label_by_edge_types={
                _USER_BUYS_ITEM: torch.tensor([20, 21, 20, -1])
            },
            negative_label_by_edge_types={
                _USER_BUYS_ITEM: torch.tensor([21, 22, -1, 20])
            },
        )

        sample_loop_inputs = sampler._prepare_ablp_inputs(
            inputs=sampler_input,
            input_seeds=sampler_input.node,
            input_type=_USER,
        )

        nodes_to_sample = sample_loop_inputs.nodes_to_sample
        assert isinstance(nodes_to_sample, Mapping)
        self.assertEqual(set(nodes_to_sample.keys()), {_USER, _ITEM})
        self.assert_tensor_equality(
            nodes_to_sample[_USER],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
            torch.tensor([4, 5]),
        )
        self.assert_tensor_equality(
            nodes_to_sample[_ITEM],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
            torch.tensor([20, 21, 22]),
        )

    def test_prepare_sample_loop_inputs_homogeneous(self) -> None:
        """Standard NodeSamplerInput with no input_type returns a tensor."""
        sampler = _build_sampler_stub()
        inputs = NodeSamplerInput(
            node=torch.tensor([10, 20, 30]),
            input_type=None,
        )

        result = sampler._prepare_sample_loop_inputs(inputs)

        self.assertIsInstance(result, SampleLoopInputs)
        assert isinstance(result.nodes_to_sample, torch.Tensor)
        self.assert_tensor_equality(result.nodes_to_sample, torch.tensor([10, 20, 30]))
        self.assertEqual(result.metadata, {})

    def test_prepare_sample_loop_inputs_heterogeneous(self) -> None:
        """Standard NodeSamplerInput with input_type returns a dict."""
        sampler = _build_sampler_stub()
        inputs = NodeSamplerInput(
            node=torch.tensor([1, 2]),
            input_type=_USER,
        )

        result = sampler._prepare_sample_loop_inputs(inputs)

        self.assertIsInstance(result, SampleLoopInputs)
        assert isinstance(result.nodes_to_sample, Mapping)
        self.assertEqual(set(result.nodes_to_sample.keys()), {_USER})
        self.assert_tensor_equality(result.nodes_to_sample[_USER], torch.tensor([1, 2]))  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
        self.assertEqual(result.metadata, {})


class _ZeroFanoutInducer:
    def init_node(
        self, nodes: dict[NodeType, torch.Tensor]
    ) -> dict[NodeType, torch.Tensor]:
        return nodes

    def induce_next(
        self, neighbors: dict[EdgeType, list[torch.Tensor]]
    ) -> tuple[
        dict[NodeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
        dict[EdgeType, torch.Tensor],
    ]:
        edge_type = next(iter(neighbors))
        return (
            {_USER: torch.tensor([3])},
            {edge_type: torch.tensor([0])},
            {edge_type: torch.tensor([0])},
        )


class TestDistNeighborSamplerZeroFanout(TestCase):
    @staticmethod
    async def _run_heterogeneous_sampler(
        *, edge_direction: str, positive_fanout: int
    ) -> AsyncMock:
        zero_edge_type = EdgeType(_USER, Relation("zero"), _USER)
        sampled_edge_type = EdgeType(_USER, Relation("sampled"), _USER)
        sampler = DistNeighborSampler.__new__(DistNeighborSampler)
        sampler.max_input_size = 0
        sampler.device = torch.device("cpu")
        sampler.dist_graph = SimpleNamespace(data_cls="hetero")
        sampler._acquire_inducer = Mock(return_value=_ZeroFanoutInducer())
        sampler.inducer_pool = Mock()
        sampler.with_edge = False
        sampler.num_hops = 1
        sampler.edge_types = [zero_edge_type, sampled_edge_type]
        sampler.num_neighbors = {
            zero_edge_type: [0],
            sampled_edge_type: [positive_fanout],
        }
        sampler.edge_dir = edge_direction
        sampler._loop = asyncio.get_running_loop()
        sample_one_hop = AsyncMock(
            return_value=NeighborOutput(
                nbr=torch.tensor([3]),
                nbr_num=torch.tensor([1, 0]),
                edge=None,
            )
        )
        sampler._sample_one_hop = sample_one_hop

        await sampler._sample_from_nodes(
            NodeSamplerInput(node=torch.tensor([1, 2]), input_type=_USER)
        )
        return sample_one_hop

    def test_heterogeneous_sampler_skips_only_exact_zero_fanout(self) -> None:
        for edge_direction in ("in", "out"):
            for positive_fanout in (-1, 2):
                with self.subTest(
                    edge_direction=edge_direction,
                    positive_fanout=positive_fanout,
                ):
                    sample_one_hop = asyncio.run(
                        self._run_heterogeneous_sampler(
                            edge_direction=edge_direction,
                            positive_fanout=positive_fanout,
                        )
                    )
                    self.assertEqual(sample_one_hop.await_count, 1)
                    await_args = sample_one_hop.await_args
                    assert await_args is not None
                    _, observed_fanout, _ = await_args.args
                    self.assertEqual(observed_fanout, positive_fanout)

    def test_homogeneous_sampler_stops_before_zero_fanout_rpc(self) -> None:
        async def run_sampler() -> tuple[AsyncMock, torch.Tensor, torch.Tensor]:
            sampler = DistNeighborSampler.__new__(DistNeighborSampler)
            sampler.max_input_size = 0
            sampler.device = torch.device("cpu")
            sampler.dist_graph = SimpleNamespace(data_cls="homo")
            inducer = Mock()
            inducer.init_node.return_value = torch.tensor([1, 2])
            sampler._acquire_inducer = Mock(return_value=inducer)
            sampler.inducer_pool = Mock()
            sampler.with_edge = False
            sampler.num_neighbors = [0, 2]
            sample_one_hop = AsyncMock()
            sampler._sample_one_hop = sample_one_hop

            output = await sampler._sample_from_nodes(
                NodeSamplerInput(node=torch.tensor([1, 2]), input_type=None)
            )
            return sample_one_hop, output.row, output.col

        sample_one_hop, rows, columns = asyncio.run(run_sampler())
        sample_one_hop.assert_not_awaited()
        self.assertEqual(rows.numel(), 0)
        self.assertEqual(columns.numel(), 0)
