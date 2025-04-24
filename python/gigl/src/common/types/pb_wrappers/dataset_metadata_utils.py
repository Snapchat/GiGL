from __future__ import annotations

import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, TypeVar, Union, cast

import google.protobuf.message
import torch

from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.graph_builder_factory import GraphBuilderFactory
from gigl.src.common.types.graph_data import NodeType
from gigl.src.common.types.model import GraphBackend
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.training.v1.lib.data_loaders.common import (
    DataloaderConfig,
    DataloaderTypes,
)
from gigl.src.training.v1.lib.data_loaders.node_anchor_based_link_prediction_data_loader import (
    NodeAnchorBasedLinkPredictionBatch,
)
from gigl.src.training.v1.lib.data_loaders.rooted_node_neighborhood_data_loader import (
    RootedNodeNeighborhoodBatch,
)
from gigl.src.training.v1.lib.data_loaders.supervised_node_classification_data_loader import (
    SupervisedNodeClassificationBatch,
)
from gigl.src.training.v1.lib.data_loaders.tf_records_iterable_dataset import (
    TfRecordsIterableDataset,
)
from snapchat.research.gbml import dataset_metadata_pb2

logger = Logger()
T = TypeVar("T", bound=google.protobuf.message.Message)


# TODO: (svij-sc) Refactor this function to move inside `data_loading`
# Also, have `get_default_data_loader` take `uri_prefix` and then `get_default_data_loader` can call
# `_get_tfrecord_uris` - would promote much better clarity
def _get_tfrecord_uris(uri_prefix: Uri) -> List[Uri]:
    gcs_utils = GcsUtils(get_resource_config().project)
    uris: List[Uri]
    if isinstance(uri_prefix, GcsUri):
        uris = cast(
            List[Uri],
            gcs_utils.list_uris_with_gcs_path_pattern(
                gcs_path=uri_prefix, suffix=".tfrecord"
            ),
        )
    elif isinstance(uri_prefix, LocalUri):
        logger.info(f"We will be globing: {uri_prefix.uri}*.tfrecord")
        uris = cast(
            List[Uri],
            [LocalUri(path) for path in glob.glob(uri_prefix.uri + "*.tfrecord")],
        )
    else:
        raise TypeError("Only uri_prefix of GcsUri and LocalUri is supported for now")
    return uris


# TODO: (svij-sc) This function should move inside `data_loading` too
def read_training_sample_protos_from_tfrecords(
    uri_prefix: Uri, proto_cls: Type[T]
) -> List[T]:
    def parse_training_sample_pb(byte_str: bytes) -> T:
        pb = proto_cls()
        pb.ParseFromString(byte_str)
        return pb

    tf_record_uris = _get_tfrecord_uris(uri_prefix=uri_prefix)
    logger.info(f"Looking at following tf records: {tf_record_uris}")
    dataset = TfRecordsIterableDataset(
        tf_record_uris=tf_record_uris,
        process_raw_sample_fn=parse_training_sample_pb,
    )
    sample_pbs = list(dataset)
    return sample_pbs


@dataclass
class Dataloaders:
    train_main: Optional[torch.utils.data.DataLoader] = None
    val_main: Optional[torch.utils.data.DataLoader] = None
    test_main: Optional[torch.utils.data.DataLoader] = None
    train_random_negative: Optional[torch.utils.data.DataLoader] = None
    val_random_negative: Optional[torch.utils.data.DataLoader] = None
    test_random_negative: Optional[torch.utils.data.DataLoader] = None


class SupervisedNodeClassificationDatasetDataloaders:
    def __init__(
        self,
        batch_size_map: Dict[DataloaderTypes, int],
        num_workers_map: Dict[DataloaderTypes, int],
    ):
        self.dataloaders: Dict[DataloaderTypes, torch.utils.data.DataLoader] = {}
        self._batch_size_map = batch_size_map
        self._num_workers_map = num_workers_map

    def _get_data_loader_configs(
        self,
        uris_prefix_map: Dict[DataloaderTypes, str],
        device: torch.device,
        should_loop: bool = True,
    ) -> Dict[DataloaderTypes, DataloaderConfig]:
        data_loader_types = list(uris_prefix_map.keys())
        dataloader_configs: Dict[DataloaderTypes, DataloaderConfig] = {}
        for data_loader_type in data_loader_types:
            dataloader_configs[data_loader_type] = DataloaderConfig(
                uris=_get_tfrecord_uris(
                    UriFactory.create_uri(uris_prefix_map[data_loader_type])
                ),
                batch_size=self._batch_size_map[data_loader_type],
                num_workers=self._num_workers_map[data_loader_type],
                should_loop=should_loop,
                pin_memory=device.type != "cpu",
                seed=42,
            )
        return dataloader_configs

    def _load_dataloaders_from_config(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        configs: Dict[DataloaderTypes, DataloaderConfig],
        graph_builder: GraphBuilder,
    ) -> Dict[DataloaderTypes, torch.utils.data.DataLoader]:
        dataloaders: Dict[DataloaderTypes, torch.utils.data.DataLoader] = {}
        for data_loader_type, config in configs.items():
            if data_loader_type in self.dataloaders:
                dataloaders[data_loader_type] = self.dataloaders[data_loader_type]
                continue
            self.dataloaders[
                data_loader_type
            ] = SupervisedNodeClassificationBatch.get_default_data_loader(
                gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                graph_builder=graph_builder,
                config=config,
            )
            dataloaders[data_loader_type] = self.dataloaders[data_loader_type]
        return dataloaders

    def _get_uri_prefix_map(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        data_loader_types: List[DataloaderTypes],
    ) -> Dict[DataloaderTypes, str]:
        dataset_pb: dataset_metadata_pb2.SupervisedNodeClassificationDataset = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper.dataset_metadata_pb.supervised_node_classification_dataset
        )
        uri_map: Dict[DataloaderTypes, str] = {
            DataloaderTypes.train_main: dataset_pb.train_data_uri,
            DataloaderTypes.val_main: dataset_pb.val_data_uri,
            DataloaderTypes.test_main: dataset_pb.test_data_uri,
        }

        target_uri_map: Dict[DataloaderTypes, str] = {
            data_loader_type: uri_map[data_loader_type]
            for data_loader_type in data_loader_types
        }

        return target_uri_map

    def _get_data_loaders(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_backend: GraphBackend,
        device: torch.device,
        data_loader_types: List[DataloaderTypes],
        should_loop: bool = True,
    ) -> Dict[DataloaderTypes, torch.utils.data.DataLoader]:
        graph_builder = GraphBuilderFactory.get_graph_builder(
            backend_name=graph_backend
        )
        uris_prefix_map = self._get_uri_prefix_map(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            data_loader_types=data_loader_types,
        )
        configs = self._get_data_loader_configs(
            uris_prefix_map=uris_prefix_map,
            device=device,
            should_loop=should_loop,
        )
        dataloaders = self._load_dataloaders_from_config(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            configs=configs,
            graph_builder=graph_builder,
        )
        return dataloaders

    def get_training_dataloaders(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_backend: GraphBackend,
        device: torch.device,
    ) -> Dataloaders:
        data_loader_types = [DataloaderTypes.train_main, DataloaderTypes.val_main]
        dataloaders = self._get_data_loaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=graph_backend,
            device=device,
            data_loader_types=data_loader_types,
            should_loop=False,
        )
        assert (
            DataloaderTypes.train_main in dataloaders
            and DataloaderTypes.val_main in dataloaders
        )
        return Dataloaders(
            train_main=dataloaders[DataloaderTypes.train_main],
            val_main=dataloaders[DataloaderTypes.val_main],
        )

    def get_test_dataloaders(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_backend: GraphBackend,
        device: torch.device,
    ) -> Dataloaders:
        data_loader_types = [DataloaderTypes.test_main]
        dataloaders = self._get_data_loaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=graph_backend,
            device=device,
            data_loader_types=data_loader_types,
            should_loop=False,
        )
        assert DataloaderTypes.test_main in dataloaders
        return Dataloaders(
            test_main=dataloaders[DataloaderTypes.test_main],
        )

    def cleanup_dataloaders(self) -> None:
        while self.dataloaders:
            _, dataloader = self.dataloaders.popitem()
            del dataloader


class NodeAnchorBasedLinkPredictionDatasetDataloaders:
    def __init__(
        self,
        batch_size_map: Dict[DataloaderTypes, int],
        num_workers_map: Dict[DataloaderTypes, int],
    ):
        self.dataloaders: Dict[DataloaderTypes, torch.utils.data.DataLoader] = {}
        self._batch_size_map = batch_size_map
        self._num_workers_map = num_workers_map

    def _get_data_loader_configs(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        uris_prefix_map: Dict[DataloaderTypes, Union[str, Dict[str, str]]],
        device: torch.device,
        should_loop: bool = True,
    ) -> Dict[DataloaderTypes, DataloaderConfig]:
        data_loader_types = list(uris_prefix_map.keys())
        dataloader_configs: Dict[DataloaderTypes, DataloaderConfig] = {}

        seed_map = {
            DataloaderTypes.train_main: 42,
            DataloaderTypes.val_main: 84,
            DataloaderTypes.test_main: 168,
            DataloaderTypes.train_random_negative: 42,
            DataloaderTypes.val_random_negative: 84,
            DataloaderTypes.test_random_negative: 168,
        }

        for data_loader_type in data_loader_types:
            uris: Union[List[Uri], Dict[NodeType, List[Uri]]]
            if isinstance(uris_prefix_map[data_loader_type], str):
                uris_prefix: str = uris_prefix_map[data_loader_type]  # type: ignore
                uris = _get_tfrecord_uris(UriFactory.create_uri(uris_prefix))
            else:
                task_metadata_pb_wrapper = (
                    gbml_config_pb_wrapper.task_metadata_pb_wrapper
                )
                uris = {}
                node_type_to_uris_prefix: Dict[str, str] = uris_prefix_map[  # type: ignore
                    data_loader_type
                ]
                for (
                    supervision_dst_node_type_str
                ) in task_metadata_pb_wrapper.get_supervision_edge_node_types(
                    should_include_src_nodes=False,
                    should_include_dst_nodes=True,
                ):
                    supervision_dst_node_type = NodeType(supervision_dst_node_type_str)
                    uris[supervision_dst_node_type] = _get_tfrecord_uris(
                        UriFactory.create_uri(
                            node_type_to_uris_prefix[supervision_dst_node_type_str]
                        )
                    )
            dataloader_configs[data_loader_type] = DataloaderConfig(
                uris=uris,
                batch_size=self._batch_size_map[data_loader_type],
                num_workers=self._num_workers_map[data_loader_type],
                should_loop=should_loop,
                pin_memory=device.type != "cpu",
                seed=seed_map[data_loader_type],
            )
        return dataloader_configs

    def _load_dataloaders_from_config(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        configs: Dict[DataloaderTypes, DataloaderConfig],
        graph_builder: GraphBuilder,
    ) -> Dict[DataloaderTypes, torch.utils.data.DataLoader]:
        dataloaders: Dict[DataloaderTypes, torch.utils.data.DataLoader] = {}
        for data_loader_type, config in configs.items():
            if data_loader_type in self.dataloaders:
                dataloaders[data_loader_type] = self.dataloaders[data_loader_type]
                continue
            # If we hae a list of uris, we are getting a main batch dataloader
            if isinstance(config.uris, list):
                self.dataloaders[
                    data_loader_type
                ] = NodeAnchorBasedLinkPredictionBatch.get_default_data_loader(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                    graph_builder=graph_builder,
                    config=config,
                )
            # If we have a dictionary of uris, we are getting a rooted node neighborhood dataloader
            else:
                self.dataloaders[
                    data_loader_type
                ] = RootedNodeNeighborhoodBatch.get_default_data_loader(
                    gbml_config_pb_wrapper=gbml_config_pb_wrapper,
                    graph_builder=graph_builder,
                    config=config,
                )
            dataloaders[data_loader_type] = self.dataloaders[data_loader_type]
        return dataloaders

    def _get_uri_prefix_map(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        data_loader_types: List[DataloaderTypes],
    ) -> Dict[DataloaderTypes, Union[str, Dict[str, str]]]:
        dataset_pb: dataset_metadata_pb2.NodeAnchorBasedLinkPredictionDataset = (
            gbml_config_pb_wrapper.dataset_metadata_pb_wrapper.dataset_metadata_pb.node_anchor_based_link_prediction_dataset
        )
        uri_map: Dict[DataloaderTypes, Union[str, Dict[str, str]]] = {
            DataloaderTypes.train_main: dataset_pb.train_main_data_uri,
            DataloaderTypes.val_main: dataset_pb.val_main_data_uri,
            DataloaderTypes.train_random_negative: dict(
                dataset_pb.train_node_type_to_random_negative_data_uri
            ),
            DataloaderTypes.val_random_negative: dict(
                dataset_pb.val_node_type_to_random_negative_data_uri
            ),
            DataloaderTypes.test_main: dataset_pb.test_main_data_uri,
            DataloaderTypes.test_random_negative: dict(
                dataset_pb.test_node_type_to_random_negative_data_uri
            ),
        }

        target_uri_map: Dict[DataloaderTypes, Union[str, Dict[str, str]]] = {
            data_loader_type: uri_map[data_loader_type]
            for data_loader_type in data_loader_types
        }

        return target_uri_map

    def _get_data_loaders(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_backend: GraphBackend,
        device: torch.device,
        data_loader_types: List[DataloaderTypes],
        should_loop: bool = True,
    ) -> Dict[DataloaderTypes, torch.utils.data.DataLoader]:
        graph_builder = GraphBuilderFactory.get_graph_builder(
            backend_name=graph_backend
        )
        uris_prefix_map = self._get_uri_prefix_map(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            data_loader_types=data_loader_types,
        )
        configs = self._get_data_loader_configs(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            uris_prefix_map=uris_prefix_map,
            device=device,
            should_loop=should_loop,
        )
        dataloaders = self._load_dataloaders_from_config(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            configs=configs,
            graph_builder=graph_builder,
        )
        return dataloaders

    def get_training_dataloaders(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_backend: GraphBackend,
        device: torch.device,
    ) -> Dataloaders:
        data_loader_types = [
            DataloaderTypes.train_main,
            DataloaderTypes.val_main,
            DataloaderTypes.train_random_negative,
            DataloaderTypes.val_random_negative,
        ]
        dataloaders = self._get_data_loaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=graph_backend,
            device=device,
            data_loader_types=data_loader_types,
            should_loop=True,
        )
        for data_loader_type in data_loader_types:
            assert data_loader_type in dataloaders

        return Dataloaders(
            train_main=dataloaders[DataloaderTypes.train_main],
            val_main=dataloaders[DataloaderTypes.val_main],
            train_random_negative=dataloaders[DataloaderTypes.train_random_negative],
            val_random_negative=dataloaders[DataloaderTypes.val_random_negative],
        )

    def get_test_dataloaders(
        self,
        gbml_config_pb_wrapper: GbmlConfigPbWrapper,
        graph_backend: GraphBackend,
        device: torch.device,
    ) -> Dataloaders:
        data_loader_types = [
            DataloaderTypes.test_main,
            DataloaderTypes.test_random_negative,
        ]
        dataloaders = self._get_data_loaders(
            gbml_config_pb_wrapper=gbml_config_pb_wrapper,
            graph_backend=graph_backend,
            device=device,
            data_loader_types=data_loader_types,
            should_loop=True,
        )
        for data_loader_type in data_loader_types:
            assert data_loader_type in dataloaders

        return Dataloaders(
            test_main=dataloaders[DataloaderTypes.test_main],
            test_random_negative=dataloaders[DataloaderTypes.test_random_negative],
        )

    def cleanup_dataloaders(self) -> None:
        while self.dataloaders:
            _, dataloader = self.dataloaders.popitem()
            del dataloader
