import torch

from gigl.distributed.dist_neighbor_sampler import (
    DistNeighborSampler,
    _stable_unique_preserve_order,
)
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


def _build_sampler_stub(edge_dir: str = "out") -> DistNeighborSampler:
    sampler = DistNeighborSampler.__new__(DistNeighborSampler)
    sampler.device = torch.device("cpu")
    sampler.edge_dir = edge_dir
    return sampler


class TestDistNeighborSamplerAblpPreparation(TestCase):
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

        self.assertEqual(set(sample_loop_inputs.nodes_to_sample.keys()), {_USER})
        self.assert_tensor_equality(
            sample_loop_inputs.nodes_to_sample[_USER],
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

        self.assertEqual(set(sample_loop_inputs.nodes_to_sample.keys()), {_USER, _ITEM})
        self.assert_tensor_equality(
            sample_loop_inputs.nodes_to_sample[_USER],
            torch.tensor([4, 5]),
        )
        self.assert_tensor_equality(
            sample_loop_inputs.nodes_to_sample[_ITEM],
            torch.tensor([20, 21, 22]),
        )
