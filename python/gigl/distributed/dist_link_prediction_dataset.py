# Originally taken from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_dataset.py

import gc
import time
from collections.abc import Mapping
from multiprocessing.reduction import ForkingPickler
from typing import Literal, Optional, Tuple, Union

import graphlearn_torch as glt
import torch
from graphlearn_torch.data import Feature, Graph
from graphlearn_torch.partition import PartitionBook, RangePartitionBook
from graphlearn_torch.utils import id2idx

from gigl.common.logger import Logger
from gigl.distributed.utils.partition_book import get_ids_on_rank
from gigl.src.common.types.graph_data import (  # TODO (mkolodner-sc): Change to use torch_geometric.typing
    EdgeType,
    NodeType,
)
from gigl.types.graph import (
    FeatureInfo,
    FeaturePartitionData,
    GraphPartitionData,
    PartitionOutput,
)
from gigl.utils.data_splitters import NodeAnchorLinkSplitter, NodeSplitter
from gigl.utils.share_memory import share_memory

logger = Logger()


class DistDataset(glt.distributed.DistDataset):
    """
    This class is inherited from GraphLearn-for-PyTorch's DistDataset class. We override the __init__ functionality to support positive and
    negative edges and labels. We also override the share_ipc function to correctly serialize these new fields. We additionally introduce
    a `build` function for storing the partitioned inside of this class. We assume data in this class is only in the CPU RAM, and do not support
    data on GPU memory, thus simplifying the logic and tooling required compared to the base DistDataset class.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        edge_dir: Literal["in", "out"],
        graph_partition: Optional[Union[Graph, dict[EdgeType, Graph]]] = None,
        node_feature_partition: Optional[
            Union[Feature, dict[NodeType, Feature]]
        ] = None,
        edge_feature_partition: Optional[
            Union[Feature, dict[EdgeType, Feature]]
        ] = None,
        node_labels: Optional[Union[Feature, dict[NodeType, Feature]]] = None,
        node_partition_book: Optional[
            Union[PartitionBook, dict[NodeType, PartitionBook]]
        ] = None,
        edge_partition_book: Optional[
            Union[PartitionBook, dict[EdgeType, PartitionBook]]
        ] = None,
        positive_edge_label: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = None,
        negative_edge_label: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = None,
        node_ids: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]] = None,
        num_train: Optional[Union[int, dict[NodeType, int]]] = None,
        num_val: Optional[Union[int, dict[NodeType, int]]] = None,
        num_test: Optional[Union[int, dict[NodeType, int]]] = None,
        node_feature_info: Optional[
            Union[FeatureInfo, dict[NodeType, FeatureInfo]]
        ] = None,
        edge_feature_info: Optional[
            Union[FeatureInfo, dict[EdgeType, FeatureInfo]]
        ] = None,
    ) -> None:
        """
        Initializes the fields of the DistDataset class. This function is called upon each serialization of the DistDataset instance.
        Args:
            rank (int): Rank of the current process
            world_size (int): World size of the current process
            edge_dir (Literal["in", "out"]): Edge direction of the provied graph
        The below arguments are only expected to be provided when re-serializing an instance of the DistDataset class after build() has been called
            graph_partition (Optional[Union[Graph, dict[EdgeType, Graph]]]): Partitioned Graph Data
            node_feature_partition (Optional[Union[Feature, dict[NodeType, Feature]]]): Partitioned Node Feature Data
            edge_feature_partition (Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]): Partitioned Edge Feature Data
            node_labels (Optional[Union[Feature, dict[NodeType, Feature]]]): The labels of each node on the current machine
            node_partition_book (Optional[Union[PartitionBook, dict[NodeType, PartitionBook]]]): Node Partition Book
            edge_partition_book (Optional[Union[PartitionBook, dict[EdgeType, PartitionBook]]]): Edge Partition Book
            positive_edge_label (Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]): Positive Edge Label Tensor
            negative_edge_label (Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]): Negative Edge Label Tensor
            node_ids (Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]): Node IDs on the current machine
            num_train:  Optional[Union[int, dict[NodeType, int]]]): Number of training nodes on the current machine. Will be a dict if heterogeneous.
            num_val: (Optional[Union[int, dict[NodeType, int]]]): Number of validation nodes on the current machine. Will be a dict if heterogeneous.
            num_test: (Optional[Union[int, dict[NodeType, int]]]): Number of test nodes on the current machine. Will be a dict if heterogeneous.
            node_feature_info: Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]: Dimension of node features and its data type, will be a dict if heterogeneous.
                Note this will be None in the homogeneous case if the data has no node features, or will only contain node types with node features in the heterogeneous case.
            edge_feature_info: Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]: Dimension of edge features and its data type, will be a dict if heterogeneous.
                Note this will be None in the homogeneous case if the data has no edge features, or will only contain edge types with edge features in the heterogeneous case.
        """
        self._rank: int = rank
        self._world_size: int = world_size
        self._edge_dir: Literal["in", "out"] = edge_dir

        super().__init__(
            num_partitions=world_size,
            partition_idx=rank,
            graph_partition=graph_partition,
            node_feature_partition=node_feature_partition,
            edge_feature_partition=edge_feature_partition,
            whole_node_labels=node_labels,
            node_pb=node_partition_book,
            edge_pb=edge_partition_book,
            edge_dir=edge_dir,
        )
        self._positive_edge_label: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = positive_edge_label
        self._negative_edge_label: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = negative_edge_label

        self._node_ids: Optional[
            Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        ] = node_ids

        self._num_train = num_train
        self._num_val = num_val
        self._num_test = num_test

        # These fields are added so we can extract the node and edge feature dimensions and data type in the dataloader without having to lazily initialize the features.
        self._node_feature_info = node_feature_info
        self._edge_feature_info = edge_feature_info

    # TODO (mkolodner-sc): Modify so that we don't need to rely on GLT's base variable naming (i.e. partition_idx, num_partitions) in favor of more clear
    # naming (i.e. rank, world_size).

    @property
    def partition_idx(self) -> int:
        return self._rank

    @partition_idx.setter
    def partition_idx(self, new_partition_idx: int):
        self._rank = new_partition_idx

    @property
    def num_partitions(self) -> int:
        return self._world_size

    @num_partitions.setter
    def num_partitions(self, new_num_partitions: int):
        self._world_size = new_num_partitions

    @property
    def edge_dir(self) -> Literal["in", "out"]:
        return self._edge_dir

    @edge_dir.setter
    def edge_dir(self, new_edge_dir: Literal["in", "out"]):
        self._edge_dir = new_edge_dir

    @property
    def graph(self) -> Optional[Union[Graph, dict[EdgeType, Graph]]]:
        return self._graph

    @graph.setter
    def graph(self, new_graph: Optional[Union[Graph, dict[EdgeType, Graph]]]):
        self._graph = new_graph

    @property
    def node_features(self) -> Optional[Union[Feature, dict[NodeType, Feature]]]:
        """
        During serializiation, the initialized `Feature` type does not immediately contain the feature and id2index tensors. These
        fields are initially set to None, and are only populated when we retrieve the size, retrieve the shape, or index into one of these tensors.
        This can also be done manually with the feature.lazy_init_with_ipc_handle() function.
        """
        return self._node_features

    @node_features.setter
    def node_features(
        self, new_node_features: Optional[Union[Feature, dict[NodeType, Feature]]]
    ):
        self._node_features = new_node_features

    @property
    def edge_features(self) -> Optional[Union[Feature, dict[EdgeType, Feature]]]:
        """
        During serializiation, the initialized `Feature` type does not immediately contain the feature and id2index tensors. These
        fields are initially set to None, and are only populated when we retrieve the size, retrieve the shape, or index into one of these tensors.
        This can also be done manually with the feature.lazy_init_with_ipc_handle() function.
        """
        return self._edge_features

    @edge_features.setter
    def edge_features(
        self, new_edge_features: Optional[Union[Feature, dict[EdgeType, Feature]]]
    ):
        self._edge_features = new_edge_features

    @property
    def node_pb(
        self,
    ) -> Optional[Union[PartitionBook, dict[NodeType, PartitionBook]]]:
        return self._node_partition_book

    @node_pb.setter
    def node_pb(
        self,
        new_node_pb: Optional[Union[PartitionBook, dict[NodeType, PartitionBook]]],
    ):
        self._node_partition_book = new_node_pb

    @property
    def edge_pb(
        self,
    ) -> Optional[Union[PartitionBook, dict[EdgeType, PartitionBook]]]:
        return self._edge_partition_book

    @edge_pb.setter
    def edge_pb(
        self,
        new_edge_pb: Optional[Union[PartitionBook, dict[EdgeType, PartitionBook]]],
    ):
        self._edge_partition_book = new_edge_pb

    @property
    def positive_edge_label(
        self,
    ) -> Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]:
        return self._positive_edge_label

    @property
    def negative_edge_label(
        self,
    ) -> Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]:
        return self._negative_edge_label

    @property
    def node_ids(self) -> Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]:
        """
        Node ids local to the current machine.

        May be None if the dataset is not built.
        Will be a torch.Tensor if the dataset is homogeneous,
        or a dict[NodeType, torch.Tensor] if the dataset is heterogeneous.

        Note: If the dataset has been split, then the form of this tensor will be:
            [train_node_ids, val_node_ids, test_node_ids, remaining_node_ids]
        e.g. if we have 10 nodes, and we split them into
        train = [0, 1, 2, 3], val = [3, 4], test = [5, 6, 7], then the node_ids will be:
            [0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9]
        Note that we *de-dupe* the nodes which are in splits (after all the splits there is just [8, 9])
        but we *don't* de-dupe per split, e.g. we have [3, 3] in the node_ids tensor.
        """
        return self._node_ids

    @property
    def node_labels(
        self,
    ) -> Optional[Union[Feature, dict[NodeType, Feature]]]:
        return self._node_labels

    @node_labels.setter
    def node_labels(
        self,
        new_node_labels: Optional[Union[Feature, dict[NodeType, Feature]]],
    ):
        self._node_labels = new_node_labels

    @property
    def node_feature_info(
        self,
    ) -> Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]:
        """
        Contains information about the dimension and dtype for the node features in the graph
        """
        return self._node_feature_info

    @property
    def edge_feature_info(
        self,
    ) -> Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]:
        """
        Contains information about the dimension and dtype for the edge features in the graph
        """
        return self._edge_feature_info

    @property
    def train_node_ids(
        self,
    ) -> Optional[Union[torch.Tensor, Mapping[NodeType, torch.Tensor]]]:
        if self._num_train is None:
            return None
        elif isinstance(self._num_train, int) and isinstance(
            self._node_ids, torch.Tensor
        ):
            return self._node_ids[: self._num_train]
        elif isinstance(self._num_train, Mapping) and isinstance(
            self._node_ids, Mapping
        ):
            node_ids = {}
            for node_type, num_train in self._num_train.items():
                node_ids[node_type] = self._node_ids[node_type][:num_train]
            return node_ids
        else:
            raise ValueError(
                f"We have num_train as {type(self._num_train)} and node_ids as {type(self._node_ids)}, and don't know how to deal with them! If you are using the constructor make sure all data is either homogeneous or heterogeneous. If you are using `build()` this is likely a bug, please report it."
            )

    @property
    def val_node_ids(
        self,
    ) -> Optional[Union[torch.Tensor, Mapping[NodeType, torch.Tensor]]]:
        if self._num_val is None:
            return None
        if self._num_train is None:
            raise ValueError(
                "num_train must be set if num_val is set. If you are using the constructor make sure all data is either homogeneous or heterogeneous. If you are using `build()` this is likely a bug, please report it."
            )
        elif (
            isinstance(self._num_train, int)
            and isinstance(self._num_val, int)
            and isinstance(self._node_ids, torch.Tensor)
        ):
            idx = slice(self._num_train, self._num_train + self._num_val)
            return self._node_ids[idx]
        elif (
            isinstance(self._num_train, Mapping)
            and isinstance(self._num_val, Mapping)
            and isinstance(self._node_ids, Mapping)
        ):
            node_ids = {}
            for node_type, num_val in self._num_val.items():
                idx = slice(
                    self._num_train[node_type], self._num_train[node_type] + num_val
                )
                node_ids[node_type] = self._node_ids[node_type][idx]
            return node_ids
        else:
            raise ValueError(
                f"We have num_val as {type(self._num_val)} and node_ids as {type(self._node_ids)}, and don't know how to deal with them! If you are using the constructor make sure all data is either homogeneous or heterogeneous. If you are using `build()` this is likely a bug, please report it."
            )

    @property
    def test_node_ids(
        self,
    ) -> Optional[Union[torch.Tensor, Mapping[NodeType, torch.Tensor]]]:
        if self._num_test is None:
            return None
        if self._num_train is None or self._num_val is None:
            raise ValueError(
                "num_train and num_val must be set if num_test is set. If you are using the constructor make sure all data is either homogeneous or heterogeneous. If you are using `build()` this is likely a bug, please report it."
            )
        elif (
            isinstance(self._num_train, int)
            and isinstance(self._num_val, int)
            and isinstance(self._num_test, int)
            and isinstance(self._node_ids, torch.Tensor)
        ):
            idx = slice(
                self._num_train + self._num_val,
                self._num_train + self._num_val + self._num_test,
            )
            return self._node_ids[idx]
        elif (
            isinstance(self._num_train, Mapping)
            and isinstance(self._num_val, Mapping)
            and isinstance(self._num_test, Mapping)
            and isinstance(self._node_ids, Mapping)
        ):
            node_ids = {}
            for node_type, num_test in self._num_test.items():
                idx = slice(
                    self._num_train[node_type] + self._num_val[node_type],
                    self._num_train[node_type] + self._num_val[node_type] + num_test,
                )
                node_ids[node_type] = self._node_ids[node_type][idx]
            return node_ids
        else:
            raise ValueError(
                f"We have num_val as {type(self._num_val)} and node_ids as {type(self._node_ids)}, and don't know how to deal with them! If you are using the constructor make sure all data is either homogeneous or heterogeneous. If you are using `build()` this is likely a bug, please report it."
            )

    def load(self, *args, **kwargs):
        raise NotImplementedError(
            f"load() is not supported for the {type(self)} class. Please use build() instead."
        )

    def _split_and_initialize_node_ids(
        self,
        splitter: Optional[Union[NodeSplitter, NodeAnchorLinkSplitter]],
        partitioned_edge_index: Optional[
            Union[GraphPartitionData, dict[EdgeType, GraphPartitionData]]
        ],
    ) -> None:
        """
        Handles data splitting and computes node IDs and optional split node ids for the current machine. Note that this method assumes
        that the node partition book is set in the dataset class prior to calling this method.

        This method:
        - Performs data splitting for training, validation, and testing
        - Gets node IDs on the current machine from partition books
        - If splits are provided, organizes them into train/val/test and appends non-split nodes

        Args:
            splitter(Optional[Union[NodeSplitter, NodeAnchorLinkSplitter]]): The splitter to use for data splitting
            partitioned_edge_index(Optional[Union[GraphPartitionData, dict[EdgeType, GraphPartitionData]]]): The partitioned edge index. Only required if we are splitting on edges.
        """
        assert (
            self._node_partition_book is not None
        ), "Node partition book must be set prior to calling this method."

        node_ids_on_machine: Union[torch.Tensor, dict[NodeType, torch.Tensor]] = (
            {
                node_type: get_ids_on_rank(partition_book, rank=self._rank)
                for node_type, partition_book in self._node_partition_book.items()
            }
            if isinstance(self._node_partition_book, Mapping)
            else get_ids_on_rank(self._node_partition_book, rank=self._rank)
        )

        # Handle data splitting
        splits = None
        if isinstance(splitter, NodeAnchorLinkSplitter):
            split_start = time.time()
            assert partitioned_edge_index is not None
            edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]] = (
                partitioned_edge_index.edge_index
                if isinstance(partitioned_edge_index, GraphPartitionData)
                else {
                    edge_type: graph_partition_data.edge_index
                    for edge_type, graph_partition_data in partitioned_edge_index.items()
                }
            )
            logger.info("Starting splitting edges...")
            splits = splitter(edge_index=edge_index)
            logger.info(
                f"Finished splitting edges in {time.time() - split_start:.2f} seconds."
            )
        elif isinstance(splitter, NodeSplitter):
            split_start = time.time()
            logger.info("Starting splitting nodes...")
            # Every node is required to have a label, so we split among all ids on the current machine.
            splits = splitter(node_ids=node_ids_on_machine)
            logger.info(
                f"Finished splitting edges in {time.time() - split_start:.2f} seconds."
            )

        # We compute the node ids on the current machine, which will be used as input to the DistNeighborLoader.
        # If the nodes were split, then we set the total number of nodes in each split here.
        # Additionally, we append any node ids, for a given node type, that were *not* split to the end of "node ids"
        # so that all node ids on a given machine are included in the dataset.
        # This is done with `_append_non_split_node_ids`.
        # An example here is if we have:
        #   train_nodes: [1, 2, 3]
        #   val_nodes: [3, 4]  # Note dupes are ok!
        #   test_nodes: [5, 6]
        #   node_ids_on_machine: [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # We would then append [7, 8] as they are not in any split.
        # We do all of this as if a user provides labels, they may be for some subset of edges
        # on a given machine, but we still want to store all node ids for the given machine.
        # TODO(kmonte): We may not need to store all node ids (either for all types - if we split, or the "extras" as described above).
        # Look into this and see if we can remove this.

        # For tensor based partitioning, the partition_book will be a torch.Tensor under-the-hood. We need to check if this is a torch.Tensor
        # here, as it will not be recognized by `isinstance` as a `PartitionBook` since torch.Tensor doesn't directly inherit from `PartitionBook`.
        if isinstance(node_ids_on_machine, torch.Tensor):
            if splits is not None:
                logger.info("Using node ids that we got from the splitter.")
                if not isinstance(splits, tuple):
                    if len(splits) == 1:
                        logger.warning(
                            f"Got splits as a mapping, which is intended for heterogeneous graphs. We recieved the node types: {splits.keys()}. Since we only got one key, we will use it as the node type."
                        )
                        train_nodes, val_nodes, test_nodes = next(iter(splits.values()))
                    else:
                        raise ValueError(
                            f"Got splits as a mapping, which is intended for heterogeneous graphs. We recieved the node types: {splits.keys()}. Please use a splitter that returns a tuple of tensors."
                        )
                else:
                    train_nodes, val_nodes, test_nodes = splits
                self._num_train = train_nodes.numel()
                self._num_val = val_nodes.numel()
                self._num_test = test_nodes.numel()
                self._node_ids = _append_non_split_node_ids(
                    train_nodes, val_nodes, test_nodes, node_ids_on_machine
                )
                # do gc to save memory.
                del train_nodes, val_nodes, test_nodes, node_ids_on_machine
                gc.collect()
            else:
                logger.info(
                    "Node ids will be all nodes on this machine, derived from the partition book."
                )
                self._node_ids = node_ids_on_machine
        else:
            node_ids_by_node_type: dict[NodeType, torch.Tensor] = {}
            num_train_by_node_type: dict[NodeType, int] = {}
            num_val_by_node_type: dict[NodeType, int] = {}
            num_test_by_node_type: dict[NodeType, int] = {}
            if splits is not None and isinstance(splits, tuple):
                node_types = (
                    node_ids_on_machine.keys()
                    if isinstance(node_ids_on_machine, Mapping)
                    else []
                )
                raise ValueError(
                    f"Got splits as a tuple, which is intended for homogeneous graphs. We recieved the node types: {node_types}. Please use a splitter that returns a mapping of tensors."
                )
            for (
                node_type,
                node_ids_on_machine_per_node_type,
            ) in node_ids_on_machine.items():
                if splits is None or node_type not in splits:
                    logger.info(f"Did not split for node type {node_type}.")
                    node_ids_by_node_type[node_type] = node_ids_on_machine_per_node_type
                elif splits is not None:
                    logger.info(
                        f"Using node ids that we got from the splitter for node type {node_type}."
                    )
                    train_nodes, val_nodes, test_nodes = splits[node_type]
                    num_train_by_node_type[node_type] = train_nodes.numel()
                    num_val_by_node_type[node_type] = val_nodes.numel()
                    num_test_by_node_type[node_type] = test_nodes.numel()
                    node_ids_by_node_type[node_type] = _append_non_split_node_ids(
                        train_nodes,
                        val_nodes,
                        test_nodes,
                        node_ids_on_machine_per_node_type,
                    )
                    # do gc to save memory.
                    del (
                        train_nodes,
                        val_nodes,
                        test_nodes,
                        node_ids_on_machine_per_node_type,
                    )
                    gc.collect()
                else:
                    raise ValueError(f"We should not get here, whoops!")
            self._node_ids = node_ids_by_node_type
            if splits is not None:
                self._num_train = num_train_by_node_type
                self._num_val = num_val_by_node_type
                self._num_test = num_test_by_node_type

    def _initialize_graph(
        self,
        partitioned_edge_index: Optional[
            Union[GraphPartitionData, dict[EdgeType, GraphPartitionData]]
        ],
    ) -> None:
        """
        Initializes the graph structure with edge index and edge IDs from partition output.

        Args:
            partitioned_edge_index(Optional[Union[GraphPartitionData, dict[EdgeType, GraphPartitionData]]]): The partitioned graph data
        """

        # Edge Index refers to the [2, num_edges] tensor representing pairs of nodes connecting each edge
        # Edge IDs refers to the [num_edges] tensor representing the unique integer assigned to each edge
        edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        edge_ids: Union[Optional[torch.Tensor], dict[EdgeType, Optional[torch.Tensor]]]
        if isinstance(partitioned_edge_index, GraphPartitionData):
            edge_index = partitioned_edge_index.edge_index
            edge_ids = partitioned_edge_index.edge_ids
        elif isinstance(partitioned_edge_index, Mapping):
            edge_index = {
                edge_type: graph_partition_data.edge_index
                for edge_type, graph_partition_data in partitioned_edge_index.items()
            }
            edge_ids = {
                edge_type: graph_partition_data.edge_ids
                for edge_type, graph_partition_data in partitioned_edge_index.items()
            }
        else:
            raise ValueError(
                f"Got an unexpected type for partitioned edge index: {type(partitioned_edge_index)}"
            )

        self.init_graph(
            edge_index=edge_index,
            edge_ids=edge_ids,
            graph_mode="CPU",
            directed=True,
        )

    def _initialize_node_features(
        self,
        partitioned_node_features: Optional[
            Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]
        ],
    ) -> None:
        """
        Initializes node features in the dataset class. Requires that node partition book is set prior to calling this function.

        Args:
            partitioned_node_features(Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]]): The partitioned graph data containing node features
        """
        if isinstance(partitioned_node_features, FeaturePartitionData):
            assert isinstance(self._node_partition_book, (torch.Tensor, PartitionBook))
            node_features = partitioned_node_features.feats
            node_feature_ids = partitioned_node_features.ids
            if isinstance(self._node_partition_book, RangePartitionBook):
                node_id2idx = self._node_partition_book.id2index
            else:
                node_id2idx = id2idx(node_feature_ids)

            self.init_node_features(
                node_feature_data=node_features,
                id2idx=node_id2idx,
                with_gpu=False,
            )
            self._node_feature_info = FeatureInfo(
                dim=node_features.size(1),
                dtype=node_features.dtype,
            )
        elif (
            isinstance(partitioned_node_features, Mapping)
            and len(partitioned_node_features) > 0
        ):
            assert isinstance(
                self._node_partition_book, Mapping
            ), f"Found heterogeneous partitioned node features, but no corresponding heterogeneous node partition book. Got node partition book of type {type(self._node_partition_book)}"
            assert (
                len(self._node_partition_book) > 0
            ), f"Expected at least one node type in node partition book, got {self._node_partition_book.keys()}."

            # Partitioned node features can be empty when there are no features but are
            # labels, so we only populate the dictionary with non-empty features.
            node_type_to_node_features = {
                node_type: feature_partition_data.feats
                for node_type, feature_partition_data in partitioned_node_features.items()
            }
            node_type_to_node_feature_ids = {
                node_type: feature_partition_data.ids
                for node_type, feature_partition_data in partitioned_node_features.items()
            }
            node_type_to_id2idx = {}
            for (
                node_type,
                node_partition_book,
            ) in self._node_partition_book.items():
                if node_type in node_type_to_node_features:
                    if isinstance(node_partition_book, RangePartitionBook):
                        node_type_to_id2idx[node_type] = node_partition_book.id2index
                    else:
                        node_type_to_id2idx[node_type] = id2idx(
                            node_type_to_node_feature_ids[node_type]
                        )
            self.init_node_features(
                node_feature_data=node_type_to_node_features,
                id2idx=node_type_to_id2idx,
                with_gpu=False,
            )
            self._node_feature_info = {
                node_type: FeatureInfo(
                    dim=node_features.size(1),
                    dtype=node_features.dtype,
                )
                for node_type, node_features in node_type_to_node_features.items()
            }

    def _initialize_node_labels(
        self,
        partitioned_node_labels: Optional[
            Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]
        ],
    ) -> None:
        """
        Initializes node labels in the dataset class. Requires that node partition book is set prior to calling this function.

        Args:
            partitioned_node_labels(Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]]): The partitioned graph data containing node labels
        """
        if isinstance(partitioned_node_labels, FeaturePartitionData):
            assert isinstance(self._node_partition_book, (torch.Tensor, PartitionBook))
            node_labels = partitioned_node_labels.feats
            node_label_ids = partitioned_node_labels.ids
            if isinstance(self._node_partition_book, RangePartitionBook):
                node_id2idx = self._node_partition_book.id2index
            else:
                node_id2idx = id2idx(node_label_ids)

            self.init_node_labels(
                node_label_data=node_labels,
                id2idx=node_id2idx,
            )

        elif isinstance(partitioned_node_labels, Mapping):
            assert isinstance(
                self._node_partition_book, Mapping
            ), f"Found heterogeneous partitioned node labels, but no corresponding heterogeneous node partition book. Got node partition book of type {type(self._node_partition_book)}"
            assert (
                len(self._node_partition_book) > 0
            ), f"Expected at least one node type in node partition book, got {self._node_partition_book.keys()}."
            node_type_to_node_labels = {
                node_type: node_labels.feats
                for node_type, node_labels in partitioned_node_labels.items()
            }
            node_type_to_node_label_ids = {
                node_type: node_labels.ids
                for node_type, node_labels in partitioned_node_labels.items()
            }
            node_type_to_id2idx = {}
            for (
                node_type,
                node_partition_book,
            ) in self._node_partition_book.items():
                if node_type in node_type_to_node_labels:
                    if isinstance(node_partition_book, RangePartitionBook):
                        node_type_to_id2idx[node_type] = node_partition_book.id2index
                    else:
                        node_type_to_id2idx[node_type] = id2idx(
                            node_type_to_node_label_ids[node_type]
                        )
            self.init_node_labels(
                node_label_data=node_type_to_node_labels,
                id2idx=node_type_to_id2idx,
            )

    def _initialize_edge_features(
        self,
        partitioned_edge_features: Optional[
            Union[FeaturePartitionData, dict[EdgeType, FeaturePartitionData]]
        ],
    ) -> None:
        """
        Initializes edge features in the dataset class. Requires that edge partition book is set prior to calling this function.

        Args:
            partitioned_edge_features(Optional[Union[FeaturePartitionData, dict[EdgeType, FeaturePartitionData]]]): The partitioned graph data containing edge features
        """
        if isinstance(partitioned_edge_features, FeaturePartitionData):
            assert isinstance(self._edge_partition_book, (torch.Tensor, PartitionBook))
            edge_features = partitioned_edge_features.feats
            edge_feature_ids = partitioned_edge_features.ids
            if isinstance(self._edge_partition_book, RangePartitionBook):
                edge_id2idx = self._edge_partition_book.id2index
            else:
                edge_id2idx = id2idx(edge_feature_ids)
            self.init_edge_features(
                edge_feature_data=edge_features,
                id2idx=edge_id2idx,
                with_gpu=False,
            )
            self._edge_feature_info = FeatureInfo(
                dim=edge_features.size(1),
                dtype=edge_features.dtype,
            )
        elif (
            isinstance(partitioned_edge_features, Mapping)
            and len(partitioned_edge_features) > 0
        ):
            assert isinstance(
                self._edge_partition_book, Mapping
            ), f"Found heterogeneous partitioned edge features, but no corresponding heterogeneous edge partition book. Got edge partition book of type {type(self._edge_partition_book)}"
            assert (
                len(self._edge_partition_book) > 0
            ), f"Expected at least one edge type in edge partition book, got {self._edge_partition_book.keys()}."
            edge_type_to_edge_features = {
                edge_type: feature_partition_data.feats
                for edge_type, feature_partition_data in partitioned_edge_features.items()
            }
            edge_type_to_edge_feature_ids = {
                edge_type: feature_partition_data.ids
                for edge_type, feature_partition_data in partitioned_edge_features.items()
            }
            edge_type_to_id2idx = {}
            for (
                edge_type,
                edge_partition_book,
            ) in self._edge_partition_book.items():
                if isinstance(edge_partition_book, RangePartitionBook):
                    edge_type_to_id2idx[edge_type] = edge_partition_book.id2index
                # Not all edge types may have partitioned features.
                elif edge_type in edge_type_to_edge_feature_ids:
                    edge_type_to_id2idx[edge_type] = id2idx(
                        edge_type_to_edge_feature_ids[edge_type]
                    )
            self.init_edge_features(
                edge_feature_data=edge_type_to_edge_features,
                id2idx=edge_type_to_id2idx,
                with_gpu=False,
            )
            self._edge_feature_info = {
                edge_type: FeatureInfo(
                    dim=edge_features.size(1),
                    dtype=edge_features.dtype,
                )
                for edge_type, edge_features in edge_type_to_edge_features.items()
            }

    def build(
        self,
        partition_output: PartitionOutput,
        splitter: Optional[Union[NodeSplitter, NodeAnchorLinkSplitter]] = None,
    ) -> None:
        """
        Provided some partition graph information, this method stores these tensors inside of the class for
        subsequent live subgraph sampling using a GraphLearn-for-PyTorch NeighborLoader.

        Note that this method will all the fields from the provided partition_output:
        We do this to decrease the peak memory usage during the build process by removing these intermediate assets.

        Args:
            partition_output (PartitionOutput): Partitioned Graph to be stored in the DistDataset class
            splitter (Optional[Union[NodeSplitter, NodeAnchorLinkSplitter]]): A function that takes in an edge index or node and returns:
                                                            * a tuple of train, val, and test node ids, if heterogeneous
                                                            * a dict[NodeType, tuple[train, val, test]] of node ids, if homogeneous
                                               Optional as not all datasets need to be split on, e.g. if we're doing inference.
        """
        logger.info(
            f"Rank {self._rank} starting building dataset class from partitioned graph ..."
        )

        start_time = time.time()

        self._node_partition_book = partition_output.node_partition_book
        self._edge_partition_book = partition_output.edge_partition_book

        partition_output.node_partition_book = None
        partition_output.edge_partition_book = None
        gc.collect()

        # Handle data splitting and compute node IDs
        self._split_and_initialize_node_ids(
            splitter=splitter,
            partitioned_edge_index=partition_output.partitioned_edge_index,
        )

        # Initialize Graph and get edge data for splitting
        self._initialize_graph(
            partitioned_edge_index=partition_output.partitioned_edge_index
        )
        partition_output.partitioned_edge_index = None
        gc.collect()

        self._initialize_node_features(
            partitioned_node_features=partition_output.partitioned_node_features
        )
        partition_output.partitioned_node_features = None
        gc.collect()

        self._initialize_node_labels(
            partitioned_node_labels=partition_output.partitioned_node_labels
        )
        partition_output.partitioned_node_labels = None
        gc.collect()

        self._initialize_edge_features(
            partitioned_edge_features=partition_output.partitioned_edge_features
        )
        partition_output.partitioned_edge_features = None
        gc.collect()

        self._positive_edge_label = partition_output.partitioned_positive_labels
        self._negative_edge_label = partition_output.partitioned_negative_labels

        partition_output.partitioned_positive_labels = None
        partition_output.partitioned_negative_labels = None
        gc.collect()

        logger.info(
            f"Rank {self._rank} finished building dataset class from partitioned graph in {time.time() - start_time:.2f} seconds. Waiting for other ranks to finish ..."
        )

    def share_ipc(
        self,
    ) -> Tuple[
        int,
        int,
        Literal["in", "out"],
        Optional[Union[Graph, dict[EdgeType, Graph]]],
        Optional[Union[Feature, dict[NodeType, Feature]]],
        Optional[Union[Feature, dict[EdgeType, Feature]]],
        Optional[Union[Feature, dict[NodeType, Feature]]],
        Optional[Union[PartitionBook, dict[NodeType, PartitionBook]]],
        Optional[Union[PartitionBook, dict[EdgeType, PartitionBook]]],
        Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],
        Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],
        Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]],
        Optional[Union[int, dict[NodeType, int]]],
        Optional[Union[int, dict[NodeType, int]]],
        Optional[Union[int, dict[NodeType, int]]],
        Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]],
        Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]],
    ]:
        """
        Serializes the member variables of the DistDatasetClass
        Returns:
            int: Rank on current machine
            int: World size across all machines
            Literal["in", "out"]: Graph Edge Direction
            Optional[Union[Graph, dict[EdgeType, Graph]]]: Partitioned Graph Data
            Optional[Union[Feature, dict[NodeType, Feature]]]: Partitioned Node Feature Data
            Optional[Union[Feature, dict[EdgeType, Feature]]]: Partitioned Edge Feature Data
            Optional[Union[Feature, dict[NodeType, Feature]]]: Node labels on the current machine. Will be a dict if heterogeneous.
            Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]: Node Partition Book Tensor
            Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]: Edge Partition Book Tensor
            Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]: Positive Edge Label Tensor
            Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]: Negative Edge Label Tensor
            Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]: Node Ids
            Optional[Union[int, dict[NodeType, int]]]: Number of training nodes on the current machine. Will be a dict if heterogeneous.
            Optional[Union[int, dict[NodeType, int]]]: Number of validation nodes on the current machine. Will be a dict if heterogeneous.
            Optional[Union[int, dict[NodeType, int]]]: Number of test nodes on the current machine. Will be a dict if heterogeneous.
            Optional[Union[FeatureInfo, dict[NodeType, FeatureInfo]]]: Node feature dim and its data type, will be a dict if heterogeneous
            Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]: Edge feature dim and its data type, will be a dict if heterogeneous
        """
        # TODO (mkolodner-sc): Investigate moving share_memory calls to the build() function

        share_memory(entity=self._node_partition_book)
        share_memory(entity=self._edge_partition_book)
        share_memory(entity=self._positive_edge_label)
        share_memory(entity=self._negative_edge_label)
        share_memory(entity=self._node_ids)
        ipc_handle = (
            self._rank,
            self._world_size,
            self._edge_dir,
            self._graph,
            self._node_features,
            self._edge_features,
            self._node_labels,
            self._node_partition_book,
            self._edge_partition_book,
            self._positive_edge_label,  # Additional field unique to DistDataset class
            self._negative_edge_label,  # Additional field unique to DistDataset class
            self._node_ids,  # Additional field unique to DistDataset class
            self._num_train,  # Additional field unique to DistDataset class
            self._num_val,  # Additional field unique to DistDataset class
            self._num_test,  # Additional field unique to DistDataset class
            self._node_feature_info,  # Additional field unique to DistDataset class
            self._edge_feature_info,  # Additional field unique to DistDataset class
        )
        return ipc_handle


def _append_non_split_node_ids(
    train_node_ids: torch.Tensor,
    val_node_ids: torch.Tensor,
    test_node_ids: torch.Tensor,
    node_ids_on_machine: torch.Tensor,
) -> torch.Tensor:
    """Given some node ids that that are in splits, and the node ids on a machine, concats the node ids on the machine that were not in a split onto the splits.

    Ex: _append_non_split_node_ids([2], [3], [4], [0, 1, 2, 3, 4, 5, 6]) -> [2, 3, 4, 0, 1, 5, 6]
    """
    # Do this as the splits may be empty, and without it we see errors like:
    # RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.
    node_ids_to_get_max = [node_ids_on_machine]
    if train_node_ids.numel():
        node_ids_to_get_max.append(train_node_ids)
    if val_node_ids.numel():
        node_ids_to_get_max.append(val_node_ids)
    if test_node_ids.numel():
        node_ids_to_get_max.append(test_node_ids)
    max_node_id = int(max(n.max().item() for n in node_ids_to_get_max)) + 1

    def clamped_bin_count(
        tensor: torch.Tensor, max_node_id: int, dtype=torch.uint8
    ) -> torch.Tensor:
        """
        Counts the number of occurrences of each value in the input tensor.

        We clamp the counts to avoid overflow to 0,
        Without clamp, and if we have 255 nodes in a split,
        we will asume we have no nodes in that bucket, which is incorrect.

        Args:
            tensor: The input tensor to count the occurrences of each value.
            max_node_id: The maximum value in the input tensor.
            dtype: The data type of the output tensor.

        Returns:
            A tensor of the same shape as the input tensor, where each element is the number of occurrences of the corresponding value in the input tensor.
        """
        return (
            torch.bincount(tensor, minlength=max_node_id)
            .clamp(max=torch.iinfo(dtype).max)
            .to(dtype)
        )

    def add_clamped_counts(
        counts: torch.Tensor, to_add: torch.Tensor, max_node_id: int
    ) -> torch.Tensor:
        """
        Adds the counts of the input tensor to the counts tensor.
        We clamp the counts to avoid overflow to 0,
        Without clamp, and if we have 255 nodes in a split,
        we will asume we have no nodes in that bucket, which is incorrect.

        Args:
            counts: The tensor to add the counts to.
            to_add: The tensor to add the counts from.
            max_node_id: The maximum value in the input tensor.

        Returns:
            A tensor of the same shape as the input tensor, where each element is the number of occurrences of the corresponding value in the input tensor.
        """
        counts.add_(clamped_bin_count(to_add, max_node_id)).clamp(max=255)
        return counts

    split_counts = clamped_bin_count(train_node_ids, max_node_id)
    add_clamped_counts(split_counts, val_node_ids, max_node_id)
    add_clamped_counts(split_counts, test_node_ids, max_node_id)

    # Count all instances of node ids, then subtract the counts of the node ids in the split from the ones in the machines.
    # Since splits are not guaranteed to be unique, we check where the count is greater than zero.
    node_id_indices_not_in_split = (
        clamped_bin_count(node_ids_on_machine, max_node_id, dtype=torch.int32).sub_(
            split_counts
        )
        > 0
    )
    # Then convert the indices to the original node ids
    node_ids_not_in_split = torch.nonzero(node_id_indices_not_in_split).squeeze(dim=1)
    logger.info(
        f"We found {node_ids_not_in_split.numel()} nodes that are not in the split."
    )
    if node_ids_not_in_split.numel() == 0:
        logger.info("Found no nodes that are not in the splits.")
        return torch.cat([train_node_ids, val_node_ids, test_node_ids])
    else:
        return torch.cat(
            [train_node_ids, val_node_ids, test_node_ids, node_ids_not_in_split]
        )


## Pickling Registration
# The serialization function (share_ipc) first pushes all member variable tensors
# to the shared memory, and then packages all references to the tensors in one ipc
# handle and sends the handle to another process. The deserialization function
# (from_ipc_handle) calls the class constructor with the ipc_handle. Therefore, the
# order of variables in the ipc_handle needs to be the same with the constructor
# interface.

# Since we add the self.positive_label and self.negative_label fields to the dataset class and remove several unused fields for link prediction task
# and cpu-only sampling, we override the `share_ipc` function to handle our custom member variables.


def _rebuild_distributed_dataset(
    ipc_handle: Tuple[
        int,  # Rank on current machine
        int,  # World size across machines
        Literal["in", "out"],  # Edge Direction
        Optional[Union[Graph, dict[EdgeType, Graph]]],  # Partitioned Graph Data
        Optional[
            Union[Feature, dict[NodeType, Feature]]
        ],  # Partitioned Node Feature Data
        Optional[
            Union[Feature, dict[EdgeType, Feature]]
        ],  # Partitioned Edge Feature Data
        Optional[Union[Feature, dict[NodeType, Feature]]],  # Node Labels
        Optional[
            Union[PartitionBook, dict[NodeType, PartitionBook]]
        ],  # Node Partition Book Tensor
        Optional[
            Union[PartitionBook, dict[EdgeType, PartitionBook]]
        ],  # Edge Partition Book Tensor
        Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ],  # Positive Edge Label Tensor
        Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ],  # Negative Edge Label Tensor
        Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]],  # Node Ids
        Optional[Union[int, dict[NodeType, int]]],  # Number of training nodes
        Optional[Union[int, dict[NodeType, int]]],  # Number of val nodes
        Optional[Union[int, dict[NodeType, int]]],  # Number of test nodes
        Optional[
            Union[FeatureInfo, dict[NodeType, FeatureInfo]]
        ],  # Node feature dim and its data type
        Optional[
            Union[FeatureInfo, dict[EdgeType, FeatureInfo]]
        ],  # Edge feature dim and its data type
    ]
):
    dataset = DistDataset.from_ipc_handle(ipc_handle)
    return dataset


def _reduce_distributed_dataset(dataset: DistDataset):
    ipc_handle = dataset.share_ipc()
    return (_rebuild_distributed_dataset, (ipc_handle,))


# Register custom serialization for DistDataset with multiprocessing's ForkingPickler.
# This enables DistDataset objects to be safely passed between processes by using
# IPC handles instead of trying to pickle the underlying shared memory directly,
# which would fail or cause data corruption.
ForkingPickler.register(DistDataset, _reduce_distributed_dataset)


# TODO (mkolodner-sc): Deprecate this class in favor of DistDataset
class DistLinkPredictionDataset(DistDataset):
    def __init__(self, *args, **kwargs):
        logger.warning(
            "DistLinkPredictionDataset is deprecated. Please use DistDataset instead."
        )
        super().__init__(*args, **kwargs)


# Register custom serialization for DistLinkPredictionDataset with multiprocessing's ForkingPickler.
# This enables DistLinkPredictionDataset objects to be safely passed between processes by using
# IPC handles instead of trying to pickle the underlying shared memory directly,
# which would fail or cause data corruption.
ForkingPickler.register(DistLinkPredictionDataset, _reduce_distributed_dataset)
