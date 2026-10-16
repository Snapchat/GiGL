from typing import Optional, cast

import numpy as np
import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.data import Feature
from graphlearn_torch.distributed import shutdown_rpc
from torch_geometric.data import Data, HeteroData

import gigl.distributed.utils
from gigl.common.utils.feature_quantization.lossless import FeatureCodec
from gigl.common.utils.feature_quantization.numpy_ops import quantize_ndarray
from gigl.common.utils.feature_quantization.torch_ops import dequantize_torch_tensor
from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils.lossless_compression import (
    compress_lossless_columns,
    lossless_column_stats,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import (
    DEFAULT_HOMOGENEOUS_EDGE_TYPE,
    DEFAULT_HOMOGENEOUS_NODE_TYPE,
    FeatureInfo,
    FeaturePartitionData,
    FeatureQuantizationMetadata,
    GraphPartitionData,
    LoadedGraphTensors,
    PartitionOutput,
)
from tests.test_assets.distributed.run_distributed_dataset import (
    run_distributed_dataset,
)
from tests.test_assets.distributed.utils import (
    assert_tensor_equality,
    create_test_process_group,
)
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)
_AUTHOR, _PAPER, _TERM = NodeType("author"), NodeType("paper"), NodeType("term")
_AUTHOR_TO_PAPER = EdgeType(_AUTHOR, Relation("to"), _PAPER)

_NUM_NODES = 40


def _id_features(
    num_rows: int, id_ranges: dict[int, tuple[int, int]], num_columns: int, seed: int
) -> torch.Tensor:
    """Gaussian columns except the ``id_ranges`` columns, which hold ids with both range ends present."""
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_rows, num_columns, generator=generator)
    for column, (low, high) in id_ranges.items():
        ids = torch.randint(low, high + 1, (num_rows,), generator=generator).float()
        ids[0], ids[1] = low, high
        features[:, column] = ids
    return features


def _compress(
    features: Optional[torch.Tensor],
    declared: FeatureCodec,
    quantized: Optional[torch.Tensor] = None,
    metadata: Optional[FeatureQuantizationMetadata] = None,
):
    stats = lossless_column_stats(features, metadata, declared)
    return compress_lossless_columns(features, quantized, metadata, declared, stats)


def _ring_edges() -> torch.Tensor:
    nodes = torch.arange(_NUM_NODES)
    return torch.stack(
        [
            nodes.repeat(2),
            torch.cat([(nodes + 1) % _NUM_NODES, (nodes + 7) % _NUM_NODES]),
        ]
    )


# GLT loaders must run in their own process, so each scenario is a module-level function run via mp.spawn.
def _assert_node_batches_equal_expected(
    _: int, dataset: DistDataset, expected: torch.Tensor
) -> None:
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        batch_size=4,
        pin_memory_device=torch.device("cpu"),
    )
    nodes_seen = 0
    for batch in loader:
        assert isinstance(batch, Data)
        assert batch.x.dtype == torch.float32
        assert_tensor_equality(batch.x, expected[batch.node])
        nodes_seen += batch.node.shape[0]
    assert nodes_seen > 0
    shutdown_rpc()


def _assert_heterogeneous_batches_equal_expected(
    _: int, dataset: DistDataset, expected: dict[str, torch.Tensor]
) -> None:
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=(_USER, torch.arange(5)),
        batch_size=2,
        pin_memory_device=torch.device("cpu"),
    )
    seen: set[str] = set()
    for batch in loader:
        assert isinstance(batch, HeteroData)
        for node_type in batch.node_types:
            assert_tensor_equality(
                batch[node_type].x, expected[node_type][batch[node_type].node]
            )
            seen.add(node_type)
    assert seen == {_USER, _STORY}
    shutdown_rpc()


def _assert_incoming_edge_batches_equal_expected(
    _: int, dataset: DistDataset, expected_edge_features: torch.Tensor
) -> None:
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[1],
        input_nodes=(_STORY, torch.arange(5)),
        batch_size=5,
        pin_memory_device=torch.device("cpu"),
    )
    batch = next(iter(loader))
    assert isinstance(batch, HeteroData)
    # Edge i joins user i to story i, so a sampled edge ends at the story whose features it carries.
    destinations = batch[_STORY].node[batch[_USER_TO_STORY].edge_index[1]]
    assert destinations.numel() == 5
    assert_tensor_equality(
        batch[_USER_TO_STORY].edge_attr, expected_edge_features[destinations]
    )
    shutdown_rpc()


def _assert_unsampled_compressed_type_is_a_full_width_placeholder(
    _: int, dataset: DistDataset, expected_user_features: torch.Tensor, story_dim: int
) -> None:
    create_test_process_group()
    loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[0],
        input_nodes=(_USER, torch.arange(5)),
        batch_size=5,
        pin_memory_device=torch.device("cpu"),
    )
    batch = next(iter(loader))
    assert isinstance(batch, HeteroData)
    assert_tensor_equality(batch[_USER].x, expected_user_features[batch[_USER].node])
    # No story node was sampled, but a model still expects an (empty) story feature matrix of its full width.
    assert batch[_STORY].x.shape == (0, story_dim)
    shutdown_rpc()


def _assert_ablp_batch_equal_expected(
    _: int,
    dataset: DistDataset,
    expected_node_features: torch.Tensor,
    expected_edge_features: dict[tuple[int, int], torch.Tensor],
) -> None:
    create_test_process_group()
    loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2],
        input_nodes=torch.tensor([0]),
        batch_size=1,
        pin_memory_device=torch.device("cpu"),
    )
    batch_count = 0
    for batch in loader:
        assert isinstance(batch, Data)
        assert_tensor_equality(batch.x, expected_node_features[batch.node])
        for local_edge_index, edge_feature in zip(batch.edge_index.T, batch.edge_attr):
            source, destination = batch.node[local_edge_index]
            # Outward sampling reverses the batch edge for message passing while
            # preserving the original edge's feature payload.
            assert_tensor_equality(
                edge_feature,
                expected_edge_features[(destination.item(), source.item())],
            )
        batch_count += 1
    assert batch_count == 1
    shutdown_rpc()


def _unpacked_rows(dataset: DistDataset, key: Optional[object]) -> torch.Tensor:
    features = dataset.node_features
    if isinstance(features, dict):
        feature = features[key]  # ty: ignore[invalid-argument-type]
    else:
        feature = features
    assert isinstance(feature, Feature)
    feature.lazy_init_with_ipc_handle()
    return feature.feature_tensor


class LosslessCompressionLoaderTest(TestCase):
    """Loaders return the uncompressed float32 rows for datasets whose columns were compressed at build time."""

    def test_node_loader_mixes_quantized_and_lossless_columns(self) -> None:
        # Logical columns: 0 and 3 are 2-bit quantized by preprocessing, 1 and 4 are ids, 2 and 5 are real.
        features = _id_features(_NUM_NODES, {1: (-1, 70), 4: (0, 5000)}, 6, seed=0)
        features[:, [0, 3]] = (torch.arange(_NUM_NODES) % 4).float().unsqueeze(1)
        quantized_metadata = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=6,
            quantized_feature_indices=(0, 3),
            clip_min=0.0,
            clip_max=3.0,
        )
        quantized = torch.from_numpy(
            quantize_ndarray(
                features[:, [0, 3]].numpy().astype(np.float32),
                bits=2,
                clip_min=0.0,
                clip_max=3.0,
            )
        )
        declared = FeatureCodec(
            ("float32", "int8", "float32", "float32", "int16", "float32")
        )

        raw, packed, metadata = _compress(
            features[:, [1, 2, 4, 5]], declared, quantized, quantized_metadata
        )

        assert raw is not None and packed is not None and metadata is not None
        dataset = DistDataset(
            rank=0,
            world_size=1,
            edge_dir="out",
            node_quantization_metadata=metadata,
        )
        dataset.build(
            partition_output=PartitionOutput(
                node_partition_book=torch.zeros(_NUM_NODES),
                edge_partition_book=torch.zeros(_ring_edges().shape[1]),
                partitioned_edge_index=GraphPartitionData(
                    edge_index=_ring_edges(), edge_ids=None
                ),
                partitioned_node_features=FeaturePartitionData(
                    feats=raw, ids=torch.arange(_NUM_NODES)
                ),
                partitioned_node_quantized_features=FeaturePartitionData(
                    feats=packed, ids=torch.arange(_NUM_NODES)
                ),
                partitioned_edge_features=None,
                partitioned_positive_labels=None,
                partitioned_negative_labels=None,
                partitioned_node_labels=None,
            )
        )
        # The raw store holds only the two real columns, but the dataset describes all six logical columns.
        self.assertEqual(
            dataset.node_feature_info, FeatureInfo(dim=6, dtype=torch.float32)
        )
        expected = features.clone()
        expected[:, [0, 3]] = dequantize_torch_tensor(quantized, quantized_metadata)
        mp.spawn(fn=_assert_node_batches_equal_expected, args=(dataset, expected))

    def test_heterogeneous_loader_compresses_only_the_declared_node_type(self) -> None:
        expected = {
            _USER: _id_features(5, {2: (-1, 70), 3: (0, 1247)}, 5, seed=1),
            _STORY: _id_features(5, {}, 3, seed=2),
        }
        raw, packed, metadata = _compress(
            expected[_USER],
            FeatureCodec(("float32", "float32", "int8", "int16", "float32")),
        )
        assert raw is not None and packed is not None and metadata is not None
        dataset = DistDataset(
            rank=0,
            world_size=1,
            edge_dir="out",
            node_quantization_metadata={_USER: metadata},
        )
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        dataset.build(
            partition_output=PartitionOutput(
                node_partition_book={_USER: torch.zeros(5), _STORY: torch.zeros(5)},
                edge_partition_book={
                    _USER_TO_STORY: torch.zeros(5),
                    EdgeType(_STORY, Relation("to"), _USER): torch.zeros(5),
                },
                partitioned_edge_index={
                    _USER_TO_STORY: GraphPartitionData(
                        edge_index=edge_index, edge_ids=None
                    ),
                    EdgeType(_STORY, Relation("to"), _USER): GraphPartitionData(
                        edge_index=edge_index, edge_ids=None
                    ),
                },
                partitioned_node_features={
                    _USER: FeaturePartitionData(feats=raw, ids=torch.arange(5)),
                    _STORY: FeaturePartitionData(
                        feats=expected[_STORY], ids=torch.arange(5)
                    ),
                },
                partitioned_node_quantized_features={
                    _USER: FeaturePartitionData(feats=packed, ids=torch.arange(5))
                },
                partitioned_edge_features=None,
                partitioned_positive_labels=None,
                partitioned_negative_labels=None,
                partitioned_node_labels=None,
            )
        )
        mp.spawn(
            fn=_assert_heterogeneous_batches_equal_expected,
            args=(dataset, expected),
        )

    def test_fully_compressed_homogeneous_node_features_need_no_raw_store(self) -> None:
        features = _id_features(
            _NUM_NODES, {0: (-1, 70), 1: (0, 5000), 2: (-1, 200)}, 3, seed=5
        )
        declared = FeatureCodec(("int8", "int16", "uint8_bias1"))
        raw, packed, metadata = _compress(features, declared)
        assert raw is None and packed is not None and metadata is not None
        dataset = DistDataset(
            rank=0,
            world_size=1,
            edge_dir="out",
            node_quantization_metadata=metadata,
        )
        dataset.build(
            partition_output=PartitionOutput(
                node_partition_book=torch.zeros(_NUM_NODES),
                edge_partition_book=torch.zeros(_ring_edges().shape[1]),
                partitioned_edge_index=GraphPartitionData(
                    edge_index=_ring_edges(), edge_ids=None
                ),
                partitioned_node_features=None,
                partitioned_node_quantized_features=FeaturePartitionData(
                    feats=packed, ids=torch.arange(_NUM_NODES)
                ),
                partitioned_edge_features=None,
                partitioned_positive_labels=None,
                partitioned_negative_labels=None,
                partitioned_node_labels=None,
            )
        )
        self.assertIsNone(dataset.node_features)
        # The dataset still describes the logical float32 features.
        self.assertEqual(
            dataset.node_feature_info, FeatureInfo(dim=3, dtype=torch.float32)
        )
        mp.spawn(fn=_assert_node_batches_equal_expected, args=(dataset, features))

    def test_unsampled_fully_compressed_node_type_keeps_its_logical_width(self) -> None:
        user_features = _id_features(5, {}, 5, seed=6)
        story_features = _id_features(
            5, {0: (0, 100), 1: (0, 100), 2: (0, 100)}, 3, seed=7
        )
        raw, packed, story_metadata = _compress(
            story_features, FeatureCodec(("int8", "int8", "int8"))
        )
        assert raw is None and packed is not None and story_metadata is not None
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        story_to_user = EdgeType(_STORY, Relation("to"), _USER)
        dataset = DistDataset(
            rank=0,
            world_size=1,
            edge_dir="out",
            node_quantization_metadata={_STORY: story_metadata},
        )
        dataset.build(
            partition_output=PartitionOutput(
                node_partition_book={_USER: torch.zeros(5), _STORY: torch.zeros(5)},
                edge_partition_book={
                    _USER_TO_STORY: torch.zeros(5),
                    story_to_user: torch.zeros(5),
                },
                partitioned_edge_index={
                    _USER_TO_STORY: GraphPartitionData(
                        edge_index=edge_index, edge_ids=None
                    ),
                    story_to_user: GraphPartitionData(
                        edge_index=edge_index, edge_ids=None
                    ),
                },
                partitioned_node_features={
                    _USER: FeaturePartitionData(
                        feats=user_features, ids=torch.arange(5)
                    )
                },
                partitioned_node_quantized_features={
                    _STORY: FeaturePartitionData(feats=packed, ids=torch.arange(5))
                },
                partitioned_edge_features=None,
                partitioned_positive_labels=None,
                partitioned_negative_labels=None,
                partitioned_node_labels=None,
            )
        )
        self.assertEqual(
            dataset.node_feature_info,
            {
                _USER: FeatureInfo(dim=5, dtype=torch.float32),
                _STORY: FeatureInfo(dim=3, dtype=torch.float32),
            },
        )
        mp.spawn(
            fn=_assert_unsampled_compressed_type_is_a_full_width_placeholder,
            args=(dataset, user_features, 3),
        )

    def test_incoming_edge_loader_returns_compressed_edge_features(self) -> None:
        edge_features = _id_features(5, {0: (-1, 70), 2: (0, 1247)}, 3, seed=3)
        raw, packed, metadata = _compress(
            edge_features, FeatureCodec(("int8", "float32", "int16"))
        )
        assert raw is not None and packed is not None and metadata is not None
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        dataset = DistDataset(
            rank=0,
            world_size=1,
            edge_dir="in",
            edge_quantization_metadata={_USER_TO_STORY: metadata},
        )
        dataset.build(
            partition_output=PartitionOutput(
                node_partition_book={_USER: torch.zeros(5), _STORY: torch.zeros(5)},
                edge_partition_book={_USER_TO_STORY: torch.zeros(5)},
                partitioned_edge_index={
                    _USER_TO_STORY: GraphPartitionData(
                        edge_index=edge_index, edge_ids=None
                    )
                },
                partitioned_node_features=None,
                partitioned_edge_features={
                    _USER_TO_STORY: FeaturePartitionData(feats=raw, ids=torch.arange(5))
                },
                partitioned_edge_quantized_features={
                    _USER_TO_STORY: FeaturePartitionData(
                        feats=packed, ids=torch.arange(5)
                    )
                },
                partitioned_positive_labels=None,
                partitioned_negative_labels=None,
                partitioned_node_labels=None,
            )
        )
        mp.spawn(
            fn=_assert_incoming_edge_batches_equal_expected,
            args=(dataset, edge_features),
        )

    def test_ablp_loader_returns_compressed_node_and_edge_features(self) -> None:
        node_features = _id_features(3, {0: (-1, 70), 2: (0, 1247)}, 4, seed=4)
        edge_features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        node_raw, node_packed, node_metadata = _compress(
            node_features, FeatureCodec(("int8", "float32", "int16", "float32"))
        )
        edge_raw, edge_packed, edge_metadata = _compress(
            edge_features, FeatureCodec(("int8", "int8"))
        )
        assert (
            node_raw is not None
            and node_packed is not None
            and node_metadata is not None
        )
        assert (
            edge_raw is None and edge_packed is not None and edge_metadata is not None
        )
        loaded_graph_tensors = LoadedGraphTensors(
            node_ids=torch.arange(3),
            node_features=node_raw,
            node_quantized_features=node_packed,
            node_labels=None,
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_features=None,
            edge_quantized_features=edge_packed,
            positive_label=torch.tensor([[0], [1]]),
            negative_label=torch.tensor([[0], [2]]),
        )
        loaded_graph_tensors.treat_labels_as_edges(edge_dir="out")
        edge_index = cast(dict[EdgeType, torch.Tensor], loaded_graph_tensors.edge_index)
        keyed_node_features = cast(
            dict[NodeType, torch.Tensor], loaded_graph_tensors.node_features
        )
        keyed_node_packed = cast(
            dict[NodeType, torch.Tensor], loaded_graph_tensors.node_quantized_features
        )
        keyed_edge_packed = cast(
            dict[EdgeType, torch.Tensor], loaded_graph_tensors.edge_quantized_features
        )
        dataset = DistDataset(
            rank=0,
            world_size=1,
            edge_dir="out",
            node_quantization_metadata=node_metadata,
            edge_quantization_metadata=edge_metadata,
        )
        dataset.build(
            partition_output=PartitionOutput(
                node_partition_book={DEFAULT_HOMOGENEOUS_NODE_TYPE: torch.zeros(3)},
                edge_partition_book={
                    edge_type: torch.zeros(3) for edge_type in edge_index
                },
                partitioned_edge_index={
                    edge_type: GraphPartitionData(
                        edge_tensor, torch.arange(edge_tensor.size(1))
                    )
                    for edge_type, edge_tensor in edge_index.items()
                },
                partitioned_node_features={
                    DEFAULT_HOMOGENEOUS_NODE_TYPE: FeaturePartitionData(
                        feats=keyed_node_features[DEFAULT_HOMOGENEOUS_NODE_TYPE],
                        ids=torch.arange(3),
                    )
                },
                partitioned_node_quantized_features={
                    DEFAULT_HOMOGENEOUS_NODE_TYPE: FeaturePartitionData(
                        feats=keyed_node_packed[DEFAULT_HOMOGENEOUS_NODE_TYPE],
                        ids=torch.arange(3),
                    )
                },
                partitioned_edge_quantized_features={
                    DEFAULT_HOMOGENEOUS_EDGE_TYPE: FeaturePartitionData(
                        feats=keyed_edge_packed[DEFAULT_HOMOGENEOUS_EDGE_TYPE],
                        ids=torch.arange(2),
                    )
                },
                partitioned_edge_features=None,
                partitioned_negative_labels=None,
                partitioned_positive_labels=None,
                partitioned_node_labels=None,
            )
        )
        mp.spawn(
            fn=_assert_ablp_batch_equal_expected,
            args=(
                dataset,
                node_features,
                {(0, 1): edge_features[0], (1, 0): edge_features[1]},
            ),
        )


class BuildDatasetLosslessCompressionTest(TestCase):
    """``build_dataset`` compresses declared columns of real partitioned data after checking them on every rank."""

    def _build(self, info, **kwargs) -> DistDataset:
        return run_distributed_dataset(
            rank=0,
            world_size=1,
            mocked_dataset_info=info,
            _port=gigl.distributed.utils.get_free_port(),
            **kwargs,
        )

    def test_homogeneous_node_features_declared_as_ids_are_compressed_and_decode_exactly(
        self,
    ) -> None:
        info = CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO
        reference = self._build(info)
        assert isinstance(reference.node_feature_info, FeatureInfo)
        dim = reference.node_feature_info.dim

        dataset = self._build(info, node_lossless_codecs=FeatureCodec(("int8",) * dim))

        metadata = dataset.node_quantization_metadata
        assert isinstance(metadata, FeatureQuantizationMetadata)
        self.assertEqual(metadata.lossless_feature_indices, tuple(range(dim)))
        self.assertIsNone(dataset.node_features)
        packed = dataset.node_quantized_features
        assert isinstance(packed, Feature)
        packed.lazy_init_with_ipc_handle()
        # One byte per binary column, padded to a multiple of four.
        self.assertEqual(packed.feature_tensor.shape[1], (dim + 3) // 4 * 4)
        self.assertTrue(
            torch.equal(
                dequantize_torch_tensor(packed.feature_tensor, metadata),
                _unpacked_rows(reference, None),
            )
        )

    def test_heterogeneous_node_features_fall_back_per_node_type(
        self,
    ) -> None:
        info = DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO
        reference = self._build(info)
        # author features are 0/1 (334 columns) and paper features are small counts (4231 columns);
        # term features (50 columns) are real valued, so declaring them as ids must leave them alone.
        node_codecs = {
            _AUTHOR: FeatureCodec(("int8",) * 334),
            _PAPER: FeatureCodec(("uint8",) * 4231),
            _TERM: FeatureCodec(("int8",) * 50),
        }

        dataset = self._build(info, node_lossless_codecs=node_codecs)

        node_metadata = cast(
            dict[NodeType, FeatureQuantizationMetadata],
            dataset.node_quantization_metadata,
        )
        self.assertEqual(set(node_metadata), {_AUTHOR, _PAPER})
        node_store = cast(dict[NodeType, Feature], dataset.node_quantized_features)
        for node_type in (_AUTHOR, _PAPER):
            node_store[node_type].lazy_init_with_ipc_handle()
            self.assertTrue(
                torch.equal(
                    dequantize_torch_tensor(
                        node_store[node_type].feature_tensor, node_metadata[node_type]
                    ),
                    _unpacked_rows(reference, node_type),
                )
            )
        self.assertEqual(_unpacked_rows(dataset, _TERM).dtype, torch.float32)
        # Compressed node types no longer keep a raw float matrix; the logical feature widths are unchanged.
        self.assertEqual(
            {
                node_type: node_metadata[node_type].feature_dim
                for node_type in node_metadata
            },
            {_AUTHOR: 334, _PAPER: 4231},
        )

    def test_heterogeneous_edge_features_declared_as_ids_are_compressed_and_decode_exactly(
        self,
    ) -> None:
        info = DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO
        reference = self._build(info)
        edge_feature_info = cast(
            dict[EdgeType, FeatureInfo], reference.edge_feature_info
        )
        edge_codecs = {
            _AUTHOR_TO_PAPER: FeatureCodec(
                ("uint8",) * edge_feature_info[_AUTHOR_TO_PAPER].dim
            )
        }

        dataset = self._build(info, edge_lossless_codecs=edge_codecs)

        edge_metadata = cast(
            dict[EdgeType, FeatureQuantizationMetadata],
            dataset.edge_quantization_metadata,
        )
        self.assertEqual(set(edge_metadata), {_AUTHOR_TO_PAPER})
        edge_store = cast(dict[EdgeType, Feature], dataset.edge_quantized_features)
        edge_store[_AUTHOR_TO_PAPER].lazy_init_with_ipc_handle()
        reference_edges = cast(dict[EdgeType, Feature], reference.edge_features)
        reference_edges[_AUTHOR_TO_PAPER].lazy_init_with_ipc_handle()
        self.assertTrue(
            torch.equal(
                dequantize_torch_tensor(
                    edge_store[_AUTHOR_TO_PAPER].feature_tensor,
                    edge_metadata[_AUTHOR_TO_PAPER],
                ),
                reference_edges[_AUTHOR_TO_PAPER].feature_tensor,
            )
        )
        # The dataset still describes every edge type's logical float32 features.
        self.assertEqual(dataset.edge_feature_info, reference.edge_feature_info)
        # Edge types that were not declared keep their raw features.
        dataset_edges = cast(dict[EdgeType, Feature], dataset.edge_features)
        self.assertEqual(set(dataset_edges), set(reference_edges) - {_AUTHOR_TO_PAPER})

    def test_edge_compression_excludes_the_sampling_weight_feature(self) -> None:
        info = DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO
        # Feature f0 of author-to-paper edges is the sampling weight; f1..f4 are small counts.
        weights = {_AUTHOR_TO_PAPER: "f0"}
        reference = self._build(info, weight_edge_feat_name=weights)

        dataset = self._build(
            info,
            weight_edge_feat_name=weights,
            edge_lossless_codecs={_AUTHOR_TO_PAPER: FeatureCodec(("uint8",) * 4)},
        )

        edge_metadata = cast(
            dict[EdgeType, FeatureQuantizationMetadata],
            dataset.edge_quantization_metadata,
        )
        self.assertEqual(edge_metadata[_AUTHOR_TO_PAPER].feature_dim, 4)
        edge_store = cast(dict[EdgeType, Feature], dataset.edge_quantized_features)
        edge_store[_AUTHOR_TO_PAPER].lazy_init_with_ipc_handle()
        reference_edges = cast(dict[EdgeType, Feature], reference.edge_features)
        reference_edges[_AUTHOR_TO_PAPER].lazy_init_with_ipc_handle()
        self.assertTrue(
            torch.equal(
                dequantize_torch_tensor(
                    edge_store[_AUTHOR_TO_PAPER].feature_tensor,
                    edge_metadata[_AUTHOR_TO_PAPER],
                ),
                reference_edges[_AUTHOR_TO_PAPER].feature_tensor,
            )
        )
        self.assertEqual(dataset.edge_feature_info, reference.edge_feature_info)
        self.assertTrue(dataset.has_edge_weights)
        weights_of = cast(dict[EdgeType, torch.Tensor], dataset.edge_weights)
        reference_weights = cast(dict[EdgeType, torch.Tensor], reference.edge_weights)
        self.assertTrue(
            torch.equal(
                weights_of[_AUTHOR_TO_PAPER], reference_weights[_AUTHOR_TO_PAPER]
            )
        )

    def test_strict_compression_raises_when_declared_ids_do_not_fit(self) -> None:
        with self.assertRaises(mp.ProcessRaisedException):
            self._build(
                DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
                node_lossless_codecs={_TERM: FeatureCodec(("int8",) * 50)},
                strict_lossless_compression=True,
            )


if __name__ == "__main__":
    absltest.main()
