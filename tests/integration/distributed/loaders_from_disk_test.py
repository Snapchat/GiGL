"""Integration tests that run both loaders end to end on mocked datasets read from disk/GCS.

These live in integration (not unit) because they read the mocked Cora and DBLP datasets
from storage and exercise DistNeighborLoader and DistABLPLoader on the full datasets.
Exact-value checks live in unit tests with hand-built graphs.

Each test loads its dataset once in the parent process, then spawns one process that
iterates a DistNeighborLoader followed by a DistABLPLoader (when the dataset has link
labels) on that same dataset, shutting down the first loader before creating the second (RPC cannot be re-initialized after shutdown_rpc).
"""

from collections.abc import Mapping
from typing import Optional

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.distributed import shutdown_rpc
from torch_geometric.data import Data, HeteroData

from gigl.distributed.dataset_factory import build_dataset
from gigl.distributed.dist_ablp_neighborloader import DistABLPLoader
from gigl.distributed.dist_dataset import DistDataset
from gigl.distributed.distributed_neighborloader import DistNeighborLoader
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO,
    CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
    DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import is_label_edge_type, to_homogeneous
from gigl.utils.data_splitters import (
    DistNodeAnchorLinkSplitter,
    DistNodeSplitter,
    NodeAnchorLinkSplitter,
    NodeSplitter,
)
from tests.test_assets.distributed.utils import create_test_process_group
from tests.test_assets.test_case import TestCase

_DBLP_NUM_AUTHORS = 4057
_CLASSIFICATION_BATCH_SIZE = 32


def _load_dataset(
    mocked_dataset_info: MockedDatasetInfo,
    splitter: NodeAnchorLinkSplitter | NodeSplitter,
    supervision_edge_types: Optional[list[EdgeType]] = None,
    ssl_positive_label_percentage: Optional[float] = None,
) -> DistDataset:
    """Builds a DistDataset from the mocked dataset's frozen config on disk."""
    mocked_info = get_mocked_dataset_artifact_metadata()[mocked_dataset_info.name]
    gbml_config_pb_wrapper = GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        gbml_config_uri=mocked_info.frozen_gbml_config_uri
    )
    serialized_graph_metadata = convert_pb_to_serialized_graph_metadata(
        preprocessed_metadata_pb_wrapper=gbml_config_pb_wrapper.preprocessed_metadata_pb_wrapper,
        graph_metadata_pb_wrapper=gbml_config_pb_wrapper.graph_metadata_pb_wrapper,
        tfrecord_uri_pattern=".*.tfrecord(.gz)?$",
    )
    return build_dataset(
        serialized_graph_metadata=serialized_graph_metadata,
        sample_edge_direction="in",
        _ssl_positive_label_percentage=ssl_positive_label_percentage,
        splitter=splitter,
    )


# GLT requires loaders to run in a separate process. These functions take local_rank as the
# first argument because they are launched with mp.spawn(nprocs=1).
def _run_cora_user_defined_loaders(_, dataset: DistDataset) -> None:
    create_test_process_group()
    assert isinstance(dataset.node_ids, Mapping)
    all_node_ids = to_homogeneous(dataset.node_ids)

    neighbor_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=all_node_ids,
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in neighbor_loader:
        assert isinstance(datum, Data), (
            f"DistNeighborLoader: expected Data, got {type(datum)}"
        )
        count += 1
    assert count == all_node_ids.size(0), (
        f"DistNeighborLoader: expected {all_node_ids.size(0)} batches, but got {count}."
    )
    neighbor_loader.shutdown()

    assert dataset.train_node_ids is not None, "Train node ids must exist."
    train_node_ids = to_homogeneous(dataset.train_node_ids)
    ablp_loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=train_node_ids,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in ablp_loader:
        assert isinstance(datum, Data), (
            f"DistABLPLoader: expected Data, got {type(datum)}"
        )
        assert hasattr(datum, "y_positive"), "DistABLPLoader: missing y_positive"
        assert isinstance(datum.y_positive, dict)
        assert hasattr(datum, "y_negative"), "DistABLPLoader: missing y_negative"
        assert isinstance(datum.y_negative, dict)
        assert datum.y_positive.keys() == datum.y_negative.keys(), (
            "DistABLPLoader: y_positive and y_negative keys differ"
        )
        count += 1
    assert count == train_node_ids.numel(), (
        f"DistABLPLoader: expected {train_node_ids.numel()} batches, but got {count}."
    )
    shutdown_rpc()


def _run_dblp_loaders(
    _, dataset: DistDataset, supervision_edge_types: list[EdgeType]
) -> None:
    assert len(supervision_edge_types) == 1, (
        "TODO (mkolodner-sc): Support multiple supervision edge types in dataloading"
    )
    supervision_edge_type = supervision_edge_types[0]
    anchor_node_type = supervision_edge_type.src_node_type
    supervision_node_type = supervision_edge_type.dst_node_type
    create_test_process_group()
    assert isinstance(dataset.node_ids, Mapping)

    author_node_ids = dataset.node_ids[NodeType("author")]  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
    neighbor_loader = DistNeighborLoader(
        dataset=dataset,
        input_nodes=(NodeType("author"), author_node_ids),
        num_neighbors=[2, 2],
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in neighbor_loader:
        assert isinstance(datum, HeteroData), (
            f"DistNeighborLoader: expected HeteroData, got {type(datum)}"
        )
        count += 1
    assert count == _DBLP_NUM_AUTHORS, (
        f"DistNeighborLoader: expected {_DBLP_NUM_AUTHORS} batches, but got {count}."
    )
    neighbor_loader.shutdown()

    assert isinstance(dataset.train_node_ids, dict)
    assert isinstance(dataset.graph, dict)
    # Label edge types must not be specified in the fanout (they are injected
    # internally and never sampled), so build num_neighbors over message-passing
    # edges only.
    num_neighbors = {
        edge_type: [2, 2]
        for edge_type in dataset.graph.keys()
        if not is_label_edge_type(edge_type)
    }
    train_anchor_ids = dataset.train_node_ids[anchor_node_type]  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
    ablp_loader = DistABLPLoader(
        dataset=dataset,
        num_neighbors=num_neighbors,
        input_nodes=(anchor_node_type, train_anchor_ids),
        supervision_edge_type=supervision_edge_type,
        pin_memory_device=torch.device("cpu"),
    )
    count = 0
    for datum in ablp_loader:
        assert isinstance(datum, HeteroData), (
            f"DistABLPLoader: expected HeteroData, got {type(datum)}"
        )
        assert hasattr(datum, "y_positive"), "DistABLPLoader: missing y_positive"
        assert isinstance(datum.y_positive, dict)
        assert not hasattr(datum, "y_negative"), "DistABLPLoader: unexpected y_negative"
        for local_anchor_node_id, local_positive_nodes in datum.y_positive.items():
            assert local_anchor_node_id < len(datum[anchor_node_type].batch), (
                "DistABLPLoader: anchor id out of range"
            )
            assert torch.all(
                local_positive_nodes < len(datum[supervision_node_type].node)
            ), "DistABLPLoader: positive label id out of range"
        count += 1
    assert count == train_anchor_ids.size(0), (
        f"DistABLPLoader: expected {train_anchor_ids.size(0)} batches, but got {count}."
    )
    shutdown_rpc()


def _run_cora_classification_loader(_, dataset: DistDataset, batch_size: int) -> None:
    create_test_process_group()
    neighbor_loader = DistNeighborLoader(
        dataset=dataset,
        num_neighbors=[2, 2],
        input_nodes=to_homogeneous(dataset.train_node_ids),
        pin_memory_device=torch.device("cpu"),
        batch_size=batch_size,
    )
    for datum in neighbor_loader:
        assert isinstance(datum, Data), (
            f"DistNeighborLoader: expected Data for homogeneous datasets, got {type(datum)}"
        )
        assert hasattr(datum, "y"), (
            "DistNeighborLoader: node labels should be present for node classification"
        )
        assert datum.y.size(0) == datum.node.size(0), (
            f"DistNeighborLoader: got {datum.y.size(0)} labels and {datum.node.size(0)} nodes"
        )
    shutdown_rpc()


class LoadersFromDiskTest(TestCase):
    def tearDown(self) -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        super().tearDown()

    def test_cora_user_defined_labels_neighbor_and_ablp_loaders(self) -> None:
        create_test_process_group()
        dataset = _load_dataset(
            CORA_USER_DEFINED_NODE_ANCHOR_MOCKED_DATASET_INFO,
            DistNodeAnchorLinkSplitter(
                sampling_direction="in", should_convert_labels_to_edges=True
            ),
        )
        mp.spawn(fn=_run_cora_user_defined_loaders, args=(dataset,))

    def test_dblp_neighbor_and_ablp_loaders(self) -> None:
        create_test_process_group()
        mocked_info = get_mocked_dataset_artifact_metadata()[
            DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO.name
        ]
        gbml_config_pb_wrapper = (
            GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
                gbml_config_uri=mocked_info.frozen_gbml_config_uri
            )
        )
        supervision_edge_types = (
            gbml_config_pb_wrapper.task_metadata_pb_wrapper.get_supervision_edge_types()
        )
        dataset = _load_dataset(
            DBLP_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
            DistNodeAnchorLinkSplitter(
                sampling_direction="in",
                supervision_edge_types=supervision_edge_types,
                should_convert_labels_to_edges=True,
            ),
            ssl_positive_label_percentage=0.1,
        )
        mp.spawn(fn=_run_dblp_loaders, args=(dataset, supervision_edge_types))

    def test_cora_node_classification_neighbor_loader(self) -> None:
        create_test_process_group()
        dataset = _load_dataset(
            CORA_NODE_CLASSIFICATION_MOCKED_DATASET_INFO, DistNodeSplitter()
        )
        mp.spawn(
            fn=_run_cora_classification_loader,
            args=(dataset, _CLASSIFICATION_BATCH_SIZE),
        )


if __name__ == "__main__":
    absltest.main()
