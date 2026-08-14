# Originally taken from https://github.com/alibaba/graphlearn-for-pytorch/blob/main/graphlearn_torch/python/distributed/dist_dataset.py

import gc
import time
from collections.abc import Mapping
from multiprocessing.reduction import ForkingPickler
from typing import Literal, Optional, Tuple, TypeVar, Union, overload

import graphlearn_torch as glt
import torch
from graphlearn_torch.data import Feature, Graph, Topology
from graphlearn_torch.partition import PartitionBook, RangePartitionBook
from graphlearn_torch.typing import TensorDataType
from graphlearn_torch.utils import id2idx

from gigl.common.logger import Logger
from gigl.distributed.utils.degree import compute_and_broadcast_degree_tensor
from gigl.distributed.utils.partition_book import get_ids_on_rank
from gigl.distributed.utils.topology import OffsetTopology
from gigl.src.common.types.graph_data import (  # TODO (mkolodner-sc): Change to use torch_geometric.typing
    EdgeType,
    NodeType,
)
from gigl.types.graph import (
    FeatureInfo,
    FeaturePartitionData,
    FeatureQuantizationMetadata,
    GraphPartitionData,
    PartitionOutput,
    is_label_edge_type,
)
from gigl.utils.data_splitters import (
    NodeAnchorLinkSplitter,
    NodeSplitter,
)
from gigl.utils.share_memory import share_memory

logger = Logger()

_EntityType = TypeVar("_EntityType", NodeType, EdgeType)


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
        node_quantized_feature_partition: Optional[
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
        node_quantization_metadata: Optional[
            Union[
                FeatureQuantizationMetadata,
                dict[NodeType, FeatureQuantizationMetadata],
            ]
        ] = None,
        edge_feature_info: Optional[
            Union[FeatureInfo, dict[EdgeType, FeatureInfo]]
        ] = None,
        degree_tensor: Optional[
            Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        ] = None,
        max_labels_per_anchor_node: Optional[int] = None,
        edge_weights: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = None,
    ) -> None:
        """
        Initializes the fields of the DistDataset class. This function is called upon each serialization of the DistDataset instance.
        Args:
            rank (int): Rank of the current process
            world_size (int): World size of the current process
            edge_dir (Literal["in", "out"]): Edge direction of the provied graph
            node_quantization_metadata: Metadata for packed node features.
                May be provided during initial construction or IPC rebuild.
        The below arguments are only expected to be provided when re-serializing an instance of the DistDataset class after build() has been called
            graph_partition (Optional[Union[Graph, dict[EdgeType, Graph]]]): Partitioned Graph Data
            node_feature_partition (Optional[Union[Feature, dict[NodeType, Feature]]]): Partitioned Node Feature Data
            node_quantized_feature_partition (Optional[Union[Feature, dict[NodeType, Feature]]]): Partitioned packed uint8 node feature data
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
            degree_tensor: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]: Pre-computed degree tensor. Lazily computed on first access via the degree_tensor property.
            max_labels_per_anchor_node (Optional[int]): Optional cap for how many
                labels to materialize per anchor node for ABLP label fetching.
            edge_weights: Per-edge sampling weights for this rank's partition.
                Automatically set during ``build()``; do not pass this manually. (default: ``None``)
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
        # GLT's sampler routes seed ids through ``dataset.id_select`` before
        # every local or remote one-hop lookup. Sampling workers rebuild this
        # dataset from its ipc handle (which carries no id_select), so the
        # translation must be installed here in __init__ — unconditionally —
        # to survive every rebuild. Whether it actually translates is decided
        # at call time from the graph topology type.
        self.id_select = self._select_ids_for_partition
        self._positive_edge_label: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = positive_edge_label
        self._negative_edge_label: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = negative_edge_label

        self._node_ids: Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]] = (
            node_ids
        )

        self._num_train = num_train
        self._num_val = num_val
        self._num_test = num_test

        # These fields are added so we can extract the node and edge feature dimensions and data type in the dataloader without having to lazily initialize the features.
        self._node_feature_info = node_feature_info
        self._edge_feature_info = edge_feature_info

        self._node_quantized_features = node_quantized_feature_partition
        self._node_quantization_metadata = node_quantization_metadata

        self._degree_tensor: Optional[
            Union[torch.Tensor, dict[NodeType, torch.Tensor]]
        ] = degree_tensor
        self._max_labels_per_anchor_node = max_labels_per_anchor_node
        self._edge_weights: Optional[
            Union[torch.Tensor, dict[EdgeType, torch.Tensor]]
        ] = edge_weights

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
    def node_quantized_features(
        self,
    ) -> Optional[Union[Feature, dict[NodeType, Feature]]]:
        return self._node_quantized_features

    @node_quantized_features.setter
    def node_quantized_features(
        self,
        new_node_quantized_features: Optional[Union[Feature, dict[NodeType, Feature]]],
    ):
        self._node_quantized_features = new_node_quantized_features

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
    def node_quantization_metadata(
        self,
    ) -> Optional[
        Union[FeatureQuantizationMetadata, dict[NodeType, FeatureQuantizationMetadata]]
    ]:
        return self._node_quantization_metadata

    @property
    def edge_feature_info(
        self,
    ) -> Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]:
        """
        Contains information about the dimension and dtype for the edge features in the graph
        """
        return self._edge_feature_info

    @property
    def degree_tensor(
        self,
    ) -> Union[torch.Tensor, dict[NodeType, torch.Tensor]]:
        """
        Lazily compute and return degree tensors for the graph.

        On first access, computes node degrees from the graph partition and uses
        all-reduce to aggregate across all machines. For heterogeneous graphs,
        degrees are summed across all incident edge types per anchor node type
        before the all-reduce, so the per-edge-type tensor is never stored.
        Requires torch.distributed to be initialized.

        Over-counting correction (for processes sharing the same data on the same
        machine) is handled automatically by detecting the distributed topology.

        The result is cached for subsequent accesses.

        Returns:
            Union[torch.Tensor, dict[NodeType, torch.Tensor]]: Degree tensor for
                homogeneous graphs, or total degree tensors keyed by node type
                for heterogeneous graphs.

        Raises:
            RuntimeError: If torch.distributed is not initialized.
            ValueError: If the dataset graph is None or topology is unavailable.
        """
        if self._degree_tensor is None:
            if self.graph is None:
                raise ValueError("Dataset graph is None. Cannot compute degrees.")

            self._degree_tensor = compute_and_broadcast_degree_tensor(
                self.graph, self._edge_dir
            )
        return self._degree_tensor

    @property
    def has_edge_weights(self) -> bool:
        """True if edge weights were registered during dataset construction via
        ``DistPartitioner.register_edge_weights()``.
        """
        return self._edge_weights is not None

    @property
    def edge_weights(
        self,
    ) -> Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]:
        """Per-edge sampling weights for this rank's partition, or ``None`` if not registered.

        The tensors are the topology-owned weight storage, ordered to match
        each topology's edge order (``topo.edge_ids``), not the partition
        input order. The per-edge pairing between edge id and weight is
        preserved.
        """
        return self._edge_weights

    @property
    def max_labels_per_anchor_node(self) -> Optional[int]:
        return self._max_labels_per_anchor_node

    @max_labels_per_anchor_node.setter
    def max_labels_per_anchor_node(
        self, new_max_labels_per_anchor_node: Optional[int]
    ) -> None:
        self._max_labels_per_anchor_node = new_max_labels_per_anchor_node

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

    def _initialize_node_ids(
        self,
        node_ids_on_machine: Union[torch.Tensor, dict[NodeType, torch.Tensor]],
        splits: Optional[
            Union[
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            ]
        ],
    ) -> None:
        """
        This method:
        - Sets the node ID tensor on the current machine, derived from the node partition book
        - Sets the train, validation, and testing node IDs if splits are provided

        Args:
            node_ids_on_machine(Union[torch.Tensor, dict[NodeType, torch.Tensor]]): The node ids on the current machine
            splits(Optional[Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Mapping[NodeType, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]]): The splits to use for data splitting.
        """

        # If the nodes are split, then we set the total number of nodes in each split here.
        # Additionally, we append any node ids, for a given node type, that were *not* split to the end of "node ids"
        # so that all node ids on a given machine are included in the dataset in self._node_ids.
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
                self._num_train = (
                    train_nodes.numel()  # ty: ignore[unresolved-attribute]
                )
                self._num_val = val_nodes.numel()  # ty: ignore[unresolved-attribute]
                self._num_test = test_nodes.numel()  # ty: ignore[unresolved-attribute]
                self._node_ids = _append_non_split_node_ids(
                    train_nodes,  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                    val_nodes,  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                    test_nodes,  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                    node_ids_on_machine,
                )
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
                else:
                    raise ValueError(f"We should not get here, whoops!")
            self._node_ids = node_ids_by_node_type
            if splits is not None:
                self._num_train = num_train_by_node_type
                self._num_val = num_val_by_node_type
                self._num_test = num_test_by_node_type

    def _message_passing_topologies_are_rebased(self) -> bool:
        """True if this dataset's sampler-reachable topologies are offset-rebased.

        Message-passing edge types are rebased all-or-nothing at build time
        (a partition-range violation raises), so one non-label
        ``OffsetTopology`` implies all sampler-reachable topologies are
        rebased. Label edge types are excluded: they are never sampled and may
        be rebased or global independently, per topology.
        """
        graph = self.graph
        if isinstance(graph, Graph):
            return isinstance(graph.topo, OffsetTopology)
        if isinstance(graph, Mapping):
            return any(
                isinstance(edge_graph.topo, OffsetTopology)
                for edge_type, edge_graph in graph.items()
                if not is_label_edge_type(edge_type)
            )
        return False

    def _select_ids_for_partition(
        self,
        srcs: torch.Tensor,
        p_mask: torch.Tensor,
        node_pb: Optional[PartitionBook] = None,
    ) -> torch.Tensor:
        """Selects the ids in ``srcs`` that belong to one partition, in the id
        space that partition's topology is indexed by.

        Installed as ``self.id_select``; GLT's sampler calls it once per
        partition before local sampling or the RPC to that partition.

        With global topologies this is a plain masked selection (GLT's
        default). With rebased (``OffsetTopology``) graphs, each selected id is
        additionally translated by its own partition's lower bound, derived
        per element from the ``RangePartitionBook`` bounds (partition 0's
        lower bound is 0), so both the local and the remote sampling branches
        index the owning partition's locally-sized index pointer correctly.

        Args:
            srcs: 1D tensor of global seed node ids.
            p_mask: Boolean mask over ``srcs`` selecting the ids that belong to
                one partition.
            node_pb: The partition book of the seeds' node type.

        Returns:
            The selected ids, partition-local when translation is active,
            global otherwise.
        """
        selected = torch.masked_select(srcs, p_mask)
        if (
            isinstance(node_pb, RangePartitionBook)
            and self._message_passing_topologies_are_rebased()
        ):
            bounds = node_pb.partition_bounds.to(selected.device)
            partition_ids = torch.searchsorted(bounds, selected, right=True)
            lower_bounds = torch.where(
                partition_ids == 0,
                torch.zeros_like(selected),
                bounds[(partition_ids - 1).clamp(min=0)],
            )
            return selected - lower_bounds
        return selected

    def _build_graph_for_partition(
        self,
        edge_index: torch.Tensor,
        edge_ids: Optional[torch.Tensor],
        edge_weights: Optional[torch.Tensor],
        node_partition_book: PartitionBook,
        layout: Literal["CSR", "CSC"],
        is_label_edge: bool,
    ) -> Graph:
        """Builds one edge type's ``Graph``, locally sized when range-partitioned.

        When the compressed-dimension node type (rows for CSR, columns for CSC)
        is range-partitioned, the topology is an ``OffsetTopology`` rebased
        onto this rank's node range ``[lower, upper)``: its index pointer has
        ``upper - lower + 1`` entries instead of spanning the global node-id
        space, while neighbor ids, edge ids, and edge weights stay global.
        Otherwise the stock globally-sized GLT ``Topology`` is built.

        Message-passing edge types must be partition-aligned; a violation
        raises so a graph never mixes rebased and global sampler-reachable
        topologies (the ``id_select`` translation is all-or-nothing per
        compressed node type).

        Label edge types are not sampler-reachable and are decided per type:
        partition-aligned label edges are rebased, others fall back to a global
        topology. Their consumer (``_get_padded_labels``) reads the
        per-topology offset, so both outcomes are correct.

        Args:
            edge_index: COO edge index for this edge type.
            edge_ids: Global edge ids, or None to arange them.
            edge_weights: Per-edge sampling weights, or None.
            node_partition_book: Partition book of the compressed-dimension
                node type.
            layout: Target layout, "CSR" for edge_dir "out", "CSC" for "in".
            is_label_edge: Whether this edge type carries supervision labels.

        Returns:
            A CPU ``Graph``. Native (C++) initialization is deferred to first
            use: this process never samples from the graph it builds, and
            sampling workers rebuild the ``Graph`` from its ipc handle
            (dropping any native state built here) before initializing their
            own copy on demand. Initializing eagerly would only spend an
            edge-scale transient (the native CPU init runs ``unique`` over the
            indices).

        Raises:
            ValueError: If a message-passing edge type's compressed-dimension
                ids fall outside this rank's range-partition bounds.
        """
        topology: Topology
        if (
            isinstance(node_partition_book, RangePartitionBook)
            and not self.graph_caching
        ):
            bounds = node_partition_book.partition_bounds
            lower = int(bounds[self._rank - 1].item()) if self._rank > 0 else 0
            upper = int(bounds[self._rank].item())
            compressed = edge_index[0] if layout == "CSR" else edge_index[1]
            is_partition_aligned = compressed.numel() == 0 or (
                int(compressed.min().item()) >= lower
                and int(compressed.max().item()) < upper
            )
            if is_partition_aligned:
                topology = OffsetTopology(
                    edge_index=edge_index,
                    edge_ids=edge_ids,
                    edge_weights=edge_weights,
                    input_layout="COO",
                    layout=layout,
                    offset=lower,
                    num_nodes=upper - lower,
                )
            elif is_label_edge:
                # Label tensors partitioned by the side opposite the compressed
                # dimension land here; they are read by anchor id, not sampled,
                # so a globally-sized topology is valid for them.
                topology = Topology(
                    edge_index=edge_index,
                    edge_ids=edge_ids,
                    edge_weights=edge_weights,
                    input_layout="COO",
                    layout=layout,
                )
            else:
                raise ValueError(
                    f"Rank {self._rank} received message-passing edges whose "
                    f"compressed-dimension ids fall outside its range partition "
                    f"[{lower}, {upper}). Range-partitioned graphs must be "
                    f"partition-aligned on the compressed dimension."
                )
        else:
            topology = Topology(
                edge_index=edge_index,
                edge_ids=edge_ids,
                edge_weights=edge_weights,
                input_layout="COO",
                layout=layout,
            )
        return Graph(topology, "CPU")

    def _initialize_graph(
        self,
        partitioned_edge_index: Union[
            GraphPartitionData, dict[EdgeType, GraphPartitionData]
        ],
        node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]],
    ) -> None:
        """Initializes the graph structure from partition output.

        Builds one topology per edge type with edge index, edge IDs, and
        optional edge weights. Range-partitioned edge types get an
        ``OffsetTopology`` whose index pointer covers only this rank's node
        range; others get the stock globally-sized GLT ``Topology``.

        Heterogeneous inputs are consumed destructively: each edge type's COO
        tensors are popped from ``partitioned_edge_index`` and released as soon
        as its topology owns the data, reducing the build's peak memory.

        For heterogeneous graphs with weights registered on only some edge
        types, a warning is logged and unweighted edge types fall back to
        uniform sampling.

        Args:
            partitioned_edge_index: Partitioned graph data per edge type (heterogeneous)
                or a single partition (homogeneous).
            node_partition_book: Node partition book(s) from the partition
                output. Passed explicitly because ``build()`` initializes the
                graph before assigning ``self._node_partition_book``.

        Raises:
            ValueError: If a message-passing edge type violates this rank's
                range-partition bounds.
        """

        # Edge Index refers to the [2, num_edges] tensor representing pairs of nodes connecting each edge
        # Edge IDs refers to the [num_edges] tensor representing the unique integer assigned to each edge
        layout: Literal["CSR", "CSC"] = "CSR" if self._edge_dir == "out" else "CSC"

        if isinstance(partitioned_edge_index, GraphPartitionData):
            assert not isinstance(node_partition_book, Mapping), (
                "Found homogeneous graph data with a heterogeneous node partition book."
            )
            homogeneous_graph = self._build_graph_for_partition(
                edge_index=partitioned_edge_index.edge_index,
                edge_ids=partitioned_edge_index.edge_ids,
                edge_weights=partitioned_edge_index.weights,
                node_partition_book=node_partition_book,
                layout=layout,
                is_label_edge=False,
            )
            self.graph = homogeneous_graph
            # The topology owns the weights, realigned to its edge order;
            # referencing them here avoids retaining a second full copy in
            # partition order for the lifetime of the dataset.
            self._edge_weights = homogeneous_graph.topo.edge_weights
            self._directed = True
            logger.info("Initialized homogeneous graph to dataset")
            return

        assert isinstance(node_partition_book, Mapping), (
            "Found heterogeneous graph data with a homogeneous node partition book."
        )
        weights_by_type = {
            edge_type: graph_partition_data.weights
            for edge_type, graph_partition_data in partitioned_edge_index.items()
            if graph_partition_data.weights is not None
        }
        if weights_by_type:
            missing = set(partitioned_edge_index.keys()) - set(weights_by_type.keys())
            if missing:
                logger.warning(
                    f"Edge weights are registered for {set(weights_by_type.keys())} but "
                    f"not for {missing}. Filling missing edge types with uniform weights "
                    f"(all 1s) so GLT does not segfault on partial-weight heterogeneous graphs."
                )
                missing_sorted = sorted(
                    missing,
                    key=lambda et: partitioned_edge_index[et].edge_index.size(1),
                    reverse=True,
                )
                largest_et = missing_sorted[0]
                max_n_edges = partitioned_edge_index[largest_et].edge_index.size(1)
                base_ones = torch.ones(max_n_edges)
                weights_by_type[largest_et] = base_ones
                for edge_type in missing_sorted[1:]:
                    n_edges = partitioned_edge_index[edge_type].edge_index.size(1)
                    weights_by_type[edge_type] = base_ones[:n_edges]
        edge_weights: Optional[dict[EdgeType, torch.Tensor]] = (
            weights_by_type if weights_by_type else None
        )

        edge_types = list(partitioned_edge_index.keys())
        graph: dict[EdgeType, Graph] = {}
        for edge_type in edge_types:
            # Pop so each edge type's COO tensors are released once its
            # topology owns the data.
            graph_partition_data = partitioned_edge_index.pop(edge_type)
            # The compressed dimension is the source side for CSR ("out") and
            # the destination side for CSC ("in").
            compressed_node_type = edge_type[0] if layout == "CSR" else edge_type[2]
            graph[edge_type] = self._build_graph_for_partition(
                edge_index=graph_partition_data.edge_index,
                edge_ids=graph_partition_data.edge_ids,
                edge_weights=edge_weights[edge_type]
                if edge_weights is not None
                else None,
                node_partition_book=node_partition_book[compressed_node_type],
                layout=layout,
                is_label_edge=is_label_edge_type(edge_type),
            )
            del graph_partition_data
        self.graph = graph
        # Reference the topology-owned weight tensors (realigned to each
        # topology's edge order) instead of retaining the partition-order
        # inputs, so weights are stored once per edge type.
        topology_weights = {
            edge_type: edge_graph.topo.edge_weights
            for edge_type, edge_graph in graph.items()
            if edge_graph.topo.edge_weights is not None
        }
        self._edge_weights = topology_weights if topology_weights else None
        self._directed = True
        logger.info(
            f"Initialized heterogeneous graph to dataset with edge types: {edge_types}"
        )

    def _initialize_node_features(
        self,
        node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]],
        partitioned_node_features: Optional[
            Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]
        ],
    ) -> None:
        """
        Initializes node features in the dataset class

        Args:
            node_partition_book(Union[PartitionBook, dict[NodeType, PartitionBook]]): The partition book for nodes
            partitioned_node_features(Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]]):
                The partitioned graph data containing node features.
        """

        node_features, node_feature_id_to_index = _prepare_feature_data(
            partition_book=node_partition_book,
            partitioned_data=partitioned_node_features,
        )

        if node_features is None or node_feature_id_to_index is None:
            logger.info("Found no node features to initialize")
            return

        self.init_node_features(
            node_feature_data=node_features,
            id2idx=node_feature_id_to_index,
            with_gpu=False,
        )

        if isinstance(node_features, Mapping):
            self._node_feature_info = {}
            for node_type, node_features_per_node_type in node_features.items():
                # We cannot make isinstance checks with NodeType, so we check
                # if it is not an edge type, since it must be one of the two.
                assert not isinstance(node_type, EdgeType)
                self._node_feature_info[node_type] = FeatureInfo(
                    dim=node_features_per_node_type.size(1),  # ty: ignore[unresolved-attribute] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                    dtype=node_features_per_node_type.dtype,  # ty: ignore[unresolved-attribute] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                )
            logger.info(
                f"Initialized node features for heterogeneous graph to dataset with node types: {node_features.keys()}"
            )
        else:
            self._node_feature_info = FeatureInfo(
                dim=node_features.size(1),
                dtype=node_features.dtype,
            )
            logger.info("Initialized node features for homogeneous graph to dataset")

    def _initialize_node_quantized_features(
        self,
        node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]],
        partitioned_node_quantized_features: Optional[
            Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]
        ],
    ) -> None:
        """
        Initializes packed uint8 node features in a separate feature store.
        """

        node_quantized_features, node_quantized_feature_id_to_index = (
            _prepare_feature_data(
                partition_book=node_partition_book,
                partitioned_data=partitioned_node_quantized_features,
            )
        )

        if (
            node_quantized_features is None
            or node_quantized_feature_id_to_index is None
        ):
            logger.info("Found no node quantized features to initialize")
            return

        # GLT only exposes init_node_features for the standard node feature
        # store. Calling it here would overwrite self._node_features, so build
        # this sidecar Feature store directly, mirroring GLT's construction:
        # https://github.com/alibaba/graphlearn-for-pytorch/blob/88ff111ac0d9e45c6c9d2d18cfc5883dca07e9f9/graphlearn_torch/python/data/dataset.py#L236

        if isinstance(node_quantized_features, Mapping):
            assert isinstance(node_quantized_feature_id_to_index, Mapping)
            self._node_quantized_features = {
                node_type: Feature(
                    feature_tensor=features_per_node_type,
                    id2index=node_quantized_feature_id_to_index[node_type],  # ty: ignore[invalid-argument-type] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                    with_gpu=False,
                    dtype=torch.uint8,
                )
                for node_type, features_per_node_type in node_quantized_features.items()
            }
            logger.info(
                f"Initialized node quantized features for heterogeneous graph to dataset with node types: {node_quantized_features.keys()}"
            )
        else:
            assert not isinstance(node_quantized_feature_id_to_index, Mapping)
            self._node_quantized_features = Feature(
                feature_tensor=node_quantized_features,
                id2index=node_quantized_feature_id_to_index,
                with_gpu=False,
                dtype=torch.uint8,
            )
            logger.info(
                "Initialized node quantized features for homogeneous graph to dataset"
            )

    def _initialize_node_labels(
        self,
        node_partition_book: Union[PartitionBook, dict[NodeType, PartitionBook]],
        partitioned_node_labels: Optional[
            Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]
        ],
    ) -> None:
        """
        Initializes node labels in the dataset class

        Args:
            node_partition_book(Union[PartitionBook, dict[NodeType, PartitionBook]]): The partition book for nodes
            partitioned_node_labels(Optional[Union[FeaturePartitionData, dict[NodeType, FeaturePartitionData]]]):
                The partitioned graph data containing node labels
        """
        node_labels, node_label_id_to_index = _prepare_feature_data(
            partition_book=node_partition_book,
            partitioned_data=partitioned_node_labels,
        )

        if node_labels is None or node_label_id_to_index is None:
            logger.info("Found no node labels to initialize")
            return

        self.init_node_labels(
            node_label_data=node_labels,
            id2idx=node_label_id_to_index,
        )

        if isinstance(node_labels, Mapping):
            logger.info(
                f"Initialized node labels for heterogeneous graph to dataset with node types: {node_labels.keys()}"
            )
        else:
            logger.info("Initialized node labels for homogeneous graph to dataset")

    def _initialize_edge_features(
        self,
        edge_partition_book: Union[PartitionBook, dict[EdgeType, PartitionBook]],
        partitioned_edge_features: Optional[
            Union[FeaturePartitionData, dict[EdgeType, FeaturePartitionData]]
        ],
    ) -> None:
        """
        Initializes edge features in the dataset class. Can be None if therea re no edge features

        Args:
            edge_partition_book(Union[PartitionBook, dict[EdgeType, PartitionBook]]): The partition book for edges
            partitioned_edge_features(Optional[Union[FeaturePartitionData, dict[EdgeType, FeaturePartitionData]]]): The partitioned graph data containing edge features
        """
        edge_features, edge_feature_id_to_index = _prepare_feature_data(
            partition_book=edge_partition_book,
            partitioned_data=partitioned_edge_features,
        )

        if edge_features is None or edge_feature_id_to_index is None:
            logger.info("Found no edge features to initialize")
            return

        self.init_edge_features(
            edge_feature_data=edge_features,
            id2idx=edge_feature_id_to_index,
            with_gpu=False,
        )

        if isinstance(edge_features, Mapping):
            self._edge_feature_info = {}
            for edge_type, edge_features_per_edge_type in edge_features.items():
                assert isinstance(edge_type, EdgeType)
                self._edge_feature_info[edge_type] = FeatureInfo(
                    dim=edge_features_per_edge_type.size(1),  # ty: ignore[unresolved-attribute] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                    dtype=edge_features_per_edge_type.dtype,  # ty: ignore[unresolved-attribute] TODO(ty-torch-keyed-access): fix ty false positives for torch-backed keyed container access.
                )
            logger.info(
                f"Initialized edge features for heterogeneous graph to dataset with edge types: {edge_features.keys()}"
            )
        else:
            self._edge_feature_info = FeatureInfo(
                dim=edge_features.size(1),
                dtype=edge_features.dtype,
            )
            logger.info(f"Initialized edge features for homogeneous graph to dataset")

    def build(
        self,
        partition_output: PartitionOutput,
        splitter: Optional[Union[NodeSplitter, NodeAnchorLinkSplitter]] = None,
    ) -> None:
        """
        Provided some partition graph information, this method stores these tensors inside of the class for
        subsequent live subgraph sampling using a GraphLearn-for-PyTorch NeighborLoader.

        Note that this method will remove all the fields from the provided partition_output:
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

        assert partition_output.partitioned_edge_index is not None, (
            "Edge index must be present in the partition output"
        )

        # We compute the node ids on the current machine, which will be used as input to the neighbor loaders.
        node_ids_on_machine: Union[torch.Tensor, dict[NodeType, torch.Tensor]] = (
            {
                node_type: get_ids_on_rank(partition_book, rank=self._rank)
                for node_type, partition_book in partition_output.node_partition_book.items()
            }
            if isinstance(partition_output.node_partition_book, Mapping)
            else get_ids_on_rank(partition_output.node_partition_book, rank=self._rank)
        )

        # Handle data splitting
        splits = None
        if isinstance(splitter, NodeAnchorLinkSplitter):
            split_start = time.time()
            assert partition_output.partitioned_edge_index is not None
            edge_index: Union[torch.Tensor, dict[EdgeType, torch.Tensor]] = (
                partition_output.partitioned_edge_index.edge_index
                if isinstance(
                    partition_output.partitioned_edge_index, GraphPartitionData
                )
                else {
                    edge_type: graph_partition_data.edge_index
                    for edge_type, graph_partition_data in partition_output.partitioned_edge_index.items()
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

        # Handle data splitting and compute node IDs
        self._initialize_node_ids(
            node_ids_on_machine=node_ids_on_machine,
            splits=splits,
        )

        del node_ids_on_machine, splits
        gc.collect()

        # Initialize Graph and get edge data for splitting. The node partition
        # books are passed explicitly: they are not assigned to the dataset
        # until later in build().
        self._initialize_graph(
            partitioned_edge_index=partition_output.partitioned_edge_index,
            node_partition_book=partition_output.node_partition_book,
        )
        partition_output.partitioned_edge_index = None
        gc.collect()

        self._initialize_node_features(
            node_partition_book=partition_output.node_partition_book,
            partitioned_node_features=partition_output.partitioned_node_features,
        )
        partition_output.partitioned_node_features = None
        gc.collect()

        self._initialize_node_quantized_features(
            node_partition_book=partition_output.node_partition_book,
            partitioned_node_quantized_features=partition_output.partitioned_node_quantized_features,
        )
        partition_output.partitioned_node_quantized_features = None
        gc.collect()

        self._initialize_node_labels(
            node_partition_book=partition_output.node_partition_book,
            partitioned_node_labels=partition_output.partitioned_node_labels,
        )
        partition_output.partitioned_node_labels = None
        gc.collect()

        self._initialize_edge_features(
            edge_partition_book=partition_output.edge_partition_book,
            partitioned_edge_features=partition_output.partitioned_edge_features,
        )
        partition_output.partitioned_edge_features = None
        gc.collect()

        self._node_partition_book = partition_output.node_partition_book
        self._edge_partition_book = partition_output.edge_partition_book

        self._positive_edge_label = partition_output.partitioned_positive_labels
        self._negative_edge_label = partition_output.partitioned_negative_labels

        partition_output.node_partition_book = None
        partition_output.edge_partition_book = None

        partition_output.partitioned_positive_labels = None
        partition_output.partitioned_negative_labels = None

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
        Optional[
            Union[
                FeatureQuantizationMetadata,
                dict[NodeType, FeatureQuantizationMetadata],
            ]
        ],
        Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]],
        Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]],
        Optional[int],
        Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],
    ]:
        """
        Serializes the member variables of the DistDatasetClass
        Returns:
            int: Rank on current machine
            int: World size across all machines
            Literal["in", "out"]: Graph Edge Direction
            Optional[Union[Graph, dict[EdgeType, Graph]]]: Partitioned Graph Data
            Optional[Union[Feature, dict[NodeType, Feature]]]: Partitioned Node Feature Data
            Optional[Union[Feature, dict[NodeType, Feature]]]: Partitioned packed uint8 node feature data
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
            Optional[Union[FeatureQuantizationMetadata, dict[NodeType, FeatureQuantizationMetadata]]]: Node quantization metadata.
            Optional[Union[FeatureInfo, dict[EdgeType, FeatureInfo]]]: Edge feature dim and its data type, will be a dict if heterogeneous
            Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]]: Degree tensors
            Optional[int]: Optional per-anchor label cap for ABLP label fetching
            Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]]: Per-edge sampling weights for this rank's partition
        """
        # TODO (mkolodner-sc): Investigate moving share_memory calls to the build() function

        share_memory(entity=self._node_partition_book)
        share_memory(entity=self._edge_partition_book)
        share_memory(entity=self._positive_edge_label)
        share_memory(entity=self._negative_edge_label)
        share_memory(entity=self._node_ids)
        share_memory(entity=self._degree_tensor)
        share_memory(entity=self._edge_weights)
        ipc_handle = (
            self._rank,
            self._world_size,
            self._edge_dir,
            self._graph,
            self._node_features,
            self._node_quantized_features,
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
            self._node_quantization_metadata,  # Additional field unique to DistDataset class
            self._edge_feature_info,  # Additional field unique to DistDataset class
            self._degree_tensor,  # Additional field unique to DistDataset class
            self._max_labels_per_anchor_node,  # Additional field unique to DistDataset class
            self._edge_weights,  # Additional field unique to DistDataset class
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


@overload
def _prepare_feature_data(
    partition_book: PartitionBook,
    partitioned_data: None,
) -> Tuple[None, None]: ...


@overload
def _prepare_feature_data(
    partition_book: PartitionBook,
    partitioned_data: FeaturePartitionData,
) -> Tuple[torch.Tensor, TensorDataType]: ...


@overload
def _prepare_feature_data(
    partition_book: dict[_EntityType, PartitionBook],
    partitioned_data: dict[_EntityType, FeaturePartitionData],
) -> Tuple[
    dict[_EntityType, torch.Tensor],
    dict[_EntityType, TensorDataType],
]: ...


def _prepare_feature_data(
    partition_book: Union[PartitionBook, dict[_EntityType, PartitionBook]],
    partitioned_data: Optional[
        Union[
            FeaturePartitionData,
            dict[_EntityType, FeaturePartitionData],
        ]
    ],
) -> Tuple[
    Optional[
        Union[
            torch.Tensor,
            dict[_EntityType, torch.Tensor],
        ]
    ],
    Optional[
        Union[
            TensorDataType,
            dict[_EntityType, TensorDataType],
        ]
    ],
]:
    """
    Utility function to prepare feature/label data for initialization.

    This function handles the common data preparation pattern shared by node features,
    node labels, and edge features initialization. It extracts features, feature IDs,
    and computes id2idx mappings for both homogeneous and heterogeneous cases.

    Args:
        partition_book (Union[PartitionBook, dict[_EntityType, PartitionBook]]): The partition book for the data type
        partitioned_data (Optional[Union[FeaturePartitionData, dict[_EntityType, FeaturePartitionData]]]): The partitioned data containing features/labels
    Returns:
        Tuple[
            Optional[Union[torch.Tensor, dict[_EntityType, torch.Tensor]]]:
                Partitioned features or labels
            Optional[Union[TensorDataType, dict[_EntityType, TensorDataType]]]:
                Global id to local index tensor for features or labels
        ]
    """
    if isinstance(partitioned_data, FeaturePartitionData):
        # Homogeneous case
        assert isinstance(partition_book, (torch.Tensor, PartitionBook))
        features = partitioned_data.feats
        feature_ids = partitioned_data.ids

        if isinstance(partition_book, RangePartitionBook):
            id_to_index = partition_book.id2index
        else:
            id_to_index = id2idx(feature_ids)

        return features, id_to_index

    elif isinstance(partitioned_data, Mapping):
        assert len(partitioned_data) > 0, (
            f"Expected at least one entity type in partitioned data, but got no entities. In heterogeneous settings, \
            please make sure you are registering entities with non-empty fields i.e. not an empty dictionary."
        )
        # Heterogeneous case
        assert isinstance(partition_book, Mapping), (
            f"Found heterogeneous partitioned data, but no corresponding heterogeneous partition book. \
            Got partition book of type {type(partition_book)}."
        )
        assert len(partition_book) > 0, (
            f"Expected at least one entity type in partition book, but got no entities. In heterogeneous settings, \
            please make sure you are registering entities with non-empty fields i.e. not an empty dictionary."
        )

        # Extract features and IDs by type
        features_per_entity_type: dict[_EntityType, torch.Tensor] = {}
        id_to_index_per_entity_type: dict[_EntityType, TensorDataType] = {}
        for entity_key, partition_book_instance in partition_book.items():
            if entity_key in partitioned_data:
                features_per_entity_type[entity_key] = partitioned_data[
                    entity_key
                ].feats
                if isinstance(partition_book_instance, RangePartitionBook):
                    id_to_index_per_entity_type[entity_key] = (
                        partition_book_instance.id2index
                    )
                else:
                    id_to_index_per_entity_type[entity_key] = id2idx(
                        partitioned_data[entity_key].ids
                    )
        return features_per_entity_type, id_to_index_per_entity_type
    else:
        return None, None


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
            Union[Feature, dict[NodeType, Feature]]
        ],  # Partitioned packed uint8 node feature data
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
            Union[
                FeatureQuantizationMetadata,
                dict[NodeType, FeatureQuantizationMetadata],
            ]
        ],  # Node quantization metadata
        Optional[
            Union[FeatureInfo, dict[EdgeType, FeatureInfo]]
        ],  # Edge feature dim and its data type
        Optional[Union[torch.Tensor, dict[NodeType, torch.Tensor]]],  # Degree tensors
        Optional[int],  # Optional per-anchor label cap for ABLP label fetching
        Optional[Union[torch.Tensor, dict[EdgeType, torch.Tensor]]],  # edge_weights
    ],
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
