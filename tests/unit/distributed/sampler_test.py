import asyncio
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import torch
from graphlearn_torch.data import Feature
from graphlearn_torch.distributed.rpc import RpcDataPartitionRouter
from graphlearn_torch.sampler import (
    HeteroSamplerOutput,
    NodeSamplerInput,
    SamplerOutput,
)

from gigl.distributed.base_sampler import (
    BaseDistNeighborSampler,
    SampleLoopInputs,
    _stable_unique_preserve_order,
)
from gigl.distributed.sampler import (
    NEGATIVE_LABEL_METADATA_KEY,
    POSITIVE_LABEL_METADATA_KEY,
    ABLPNodeSamplerInput,
)
from gigl.distributed.utils.fused_feature import (
    FusedDistFeature,
    build_fused_dist_feature,
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
        _USER_BUYS_ITEM: torch.arange(100, 100 + num_nodes).unsqueeze(1),
        _USER_CLICKS_ITEM: torch.arange(200, 200 + num_nodes).unsqueeze(1),
    }
    negative_label_by_edge_types = {
        _USER_BUYS_ITEM: torch.arange(300, 300 + num_nodes).unsqueeze(1),
        _USER_CLICKS_ITEM: torch.arange(400, 400 + num_nodes).unsqueeze(1),
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
        positive_labels = {_USER_BUYS_ITEM: torch.tensor([[1], [2], [3]])}
        negative_labels = {_USER_CLICKS_ITEM: torch.tensor([[4], [5], [6]])}

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
            torch.tensor([[100], [102]]),
        )
        self.assert_tensor_equality(
            sliced.positive_label_by_edge_types[_USER_CLICKS_ITEM],
            torch.tensor([[200], [202]]),
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_BUYS_ITEM],
            torch.tensor([[300], [302]]),
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_CLICKS_ITEM],
            torch.tensor([[400], [402]]),
        )

    def test_getitem_with_list_index(self) -> None:
        sampler_input = _build_sampler_input(num_nodes=4)
        sliced = sampler_input[[1]]

        self.assertIsInstance(sliced, ABLPNodeSamplerInput)
        self.assertTrue(torch.equal(sliced.node, torch.tensor([1])))
        self.assert_tensor_equality(
            sliced.positive_label_by_edge_types[_USER_BUYS_ITEM], torch.tensor([[101]])
        )
        self.assert_tensor_equality(
            sliced.negative_label_by_edge_types[_USER_CLICKS_ITEM],
            torch.tensor([[401]]),
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
        positive_labels = {
            _USER_FRIEND_USER: torch.tensor([[11, 12], [-1, 13], [-1, -1]])
        }
        negative_labels = {
            _USER_FRIEND_USER: torch.tensor([[13, 14], [10, -1], [-1, -1]])
        }
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
                _USER_BUYS_ITEM: torch.tensor([[20, 21], [20, -1]])
            },
            negative_label_by_edge_types={
                _USER_BUYS_ITEM: torch.tensor([[21, 22], [-1, 20]])
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


class _CountingFusedDistFeature(FusedDistFeature):
    """Records the node types a batch requests."""

    requested_types: list[Optional[str]]

    def async_get(self, ids, input_type=None):  # type: ignore[override]
        self.requested_types.append(input_type)
        return super().async_get(ids, input_type)


def _fused_node_lookup(
    features: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
) -> FusedDistFeature:
    """A single-partition fused lookup over the given per-type features and labels."""
    owner = torch.zeros(4, dtype=torch.long)

    def store(values: torch.Tensor, dtype: Optional[torch.dtype]) -> Feature:
        return Feature(values, id2index=torch.arange(4), with_gpu=False, dtype=dtype)

    with patch("gigl.distributed.utils.fused_feature.rpc_register"):
        lookup = build_fused_dist_feature(
            1,
            0,
            {key: store(value, None) for key, value in features.items()},
            None,
            {key: owner for key in features.keys() | labels.keys()},
            Mock(spec=RpcDataPartitionRouter),
            labels={key: store(value, torch.int64) for key, value in labels.items()},
        )
    assert lookup is not None
    lookup.__class__ = _CountingFusedDistFeature
    lookup.requested_types = []
    return lookup


class TestCollateFnNodeLabels(TestCase):
    def _collate(
        self,
        lookup: FusedDistFeature,
        output: "SamplerOutput | HeteroSamplerOutput",
        is_hetero: bool,
    ) -> dict[str, torch.Tensor]:
        sampler = _build_sampler_stub()
        sampler.dist_graph = SimpleNamespace(data_cls="hetero" if is_hetero else "homo")
        sampler.with_edge = False
        sampler.edge_types = []
        sampler.dist_node_feature = lookup
        sampler.dist_edge_feature = None
        return asyncio.run(sampler._collate_fn(output))

    def test_labels_are_kept_for_the_seed_type_only(self) -> None:
        # "item" has features and labels, "tag" has labels only; the seeds are "user".
        lookup = _fused_node_lookup(
            features={"user": torch.rand(4, 2), "item": torch.rand(4, 2)},
            labels={
                "user": torch.tensor([[1], [2], [3], [4]]),
                "item": torch.tensor([[5], [6], [7], [8]]),
                "tag": torch.tensor([[9], [9], [9], [9]]),
            },
        )
        nodes = torch.tensor([2, 0])
        output = HeteroSamplerOutput(
            node={"user": nodes, "item": nodes, "tag": nodes},
            row={},
            col={},
            edge=None,
            input_type="user",
        )

        result = self._collate(lookup, output, is_hetero=True)

        self.assertEqual(result["user.nlabels"].tolist(), [3, 1])
        self.assertEqual(result["user.nfeats"].shape, (2, 2))
        self.assertEqual(result["item.nfeats"].shape, (2, 2))
        self.assertNotIn("item.nlabels", result)
        self.assertNotIn("tag.nlabels", result)
        self.assertNotIn("tag.nfeats", result)
        self.assertCountEqual(lookup.requested_types, ["user", "item"])

    def test_a_label_only_seed_type_is_requested(self) -> None:
        lookup = _fused_node_lookup(
            features={}, labels={"user": torch.tensor([[1, 5], [2, 6], [3, 7], [4, 8]])}
        )
        output = HeteroSamplerOutput(
            node={"user": torch.tensor([3, 1])},
            row={},
            col={},
            edge=None,
            input_type="user",
        )

        result = self._collate(lookup, output, is_hetero=True)

        self.assertEqual(result["user.nlabels"].tolist(), [[4, 8], [2, 6]])
        self.assertNotIn("user.nfeats", result)

    def test_homogeneous_labels_are_squeezed_to_one_dimension(self) -> None:
        lookup = _fused_node_lookup(
            features={}, labels={"user": torch.tensor([[1], [2], [3], [4]])}
        )
        # A homogeneous store is keyed by a single FusedFeatureStore rather than by type.
        lookup.local_feature = lookup.local_feature["user"]
        lookup.feature_pb = lookup.feature_pb["user"]
        lookup.data_cls = "homo"
        output = SamplerOutput(
            node=torch.tensor([3, 1]),
            row=torch.tensor([0]),
            col=torch.tensor([1]),
            edge=None,
        )

        result = self._collate(lookup, output, is_hetero=False)

        self.assertEqual(result["nlabels"].tolist(), [4, 2])
