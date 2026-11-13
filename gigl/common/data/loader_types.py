"""Types shared by the TFRecord and Parquet loaders."""

from dataclasses import dataclass, field
from typing import NamedTuple, Optional, Sequence, Tuple, Union

import torch
from jaxtyping import Int64, Shaped, UInt8

from gigl.common import Uri
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict


class LoadedEntityTensors(NamedTuple):
    ids: Union[Int64[torch.Tensor, "entities"], Int64[torch.Tensor, "2 entities"]]
    features: Optional[Shaped[torch.Tensor, "entities feature_dim"]]
    packed_features: Optional[UInt8[torch.Tensor, "entities packed_feature_dim"]]
    labels: Optional[Shaped[torch.Tensor, "entities labels"]]


@dataclass(frozen=True)
class SerializedTFRecordInfo:
    """
    Stores information pertaining to how a single entity (node, edge, positive label, negative label) and single node/edge type in the heterogeneous case is serialized on disk.
    This field is used as input to TFRecordDataLoader.load_as_torch_tensors() or, when parquet_uri_prefix is set,
    ParquetDataLoader.load_as_torch_tensors() for loading torch tensors.
    """

    # Uri Prefix for stored TfRecords
    tfrecord_uri_prefix: Uri
    # Feature names to load for the current entity
    feature_keys: Sequence[str]
    # A dict of feature name -> FeatureSpec (eg. FixedLenFeature, VarlenFeature, SparseFeature, RaggedFeature).
    # If entity keys are not present, we insert them during tensor loading. For example, if the FeatureSpecDict
    # doesn't have the "node_id" identifier, we populate the feature_spce with a FixedLenFeature with shape=[], dtype=tf.int64.
    # Note that entity label keys should also be included in the feature_spec if they are present.
    feature_spec: FeatureSpecDict
    # Feature dimension of current entity
    feature_dim: int
    # Entity ID Key for current entity. If this is a Node Entity, this must be a string. If this is an edge entity, this must be a Tuple[str, str] for the source and destination ids.
    entity_key: Union[str, Tuple[str, str]]
    # Packed uint8 feature name to load for the current node entity.
    packed_feature_key: Optional[str] = None
    # Number of packed uint8 columns for the current node entity.
    packed_feature_dim: int = 0
    # Name of the label columns for the current entity, defaults to an empty list.
    label_keys: Sequence[str] = field(default_factory=list)
    # The regex pattern to match the TFRecord files at the specified prefix
    tfrecord_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$"
    # Uri Prefix for stored Parquet files. When set, the entity is loaded with ParquetDataLoader and
    # tfrecord_uri_prefix is ignored.
    parquet_uri_prefix: Optional[Uri] = None

    @property
    def data_uri_prefix(self) -> Uri:
        """The Uri Prefix the entity is loaded from, Parquet if it is set and TFRecord otherwise."""
        if self.parquet_uri_prefix is not None:
            return self.parquet_uri_prefix
        return self.tfrecord_uri_prefix

    @property
    def is_node_entity(self) -> bool:
        """
        Returns whether this serialized entity contains node or edge information by checking the type of entity_key
        """
        return isinstance(self.entity_key, str)


@dataclass(frozen=True)
class TFDatasetOptions:
    """
    Options for tuning the loading of a tf.data.Dataset. Note that this dataclass is tied to TFRecord loading specifically for the `load_as_torch_tensors` function.

    Choosing between interleave or not is not straightforward.
    We've found that interleave is faster for large numbers (>100) of small (<20M) files.
    Though this is highly variable, you should do your own benchmarks to find the best settings for your use case.

    Deterministic processing is much (100%!) slower for larger (>10M entities) datasets, but has very little impact on smaller datasets.

    Args:
        batch_size (int): How large each batch should be while processing the data.
        file_buffer_size (int): The size of the buffer to use when reading files.
        deterministic (bool): Whether to use deterministic processing, if False then the order of elements can be non-deterministic.
        use_interleave (bool): Whether to use tf.data.Dataset.interleave to read files in parallel, if not set then `num_parallel_file_reads` will be used.
        num_parallel_file_reads (int): The number of files to read in parallel if `use_interleave` is False.
        ram_budget_multiplier (float): The multiplier of the total system memory to set as the tf.data RAM budget.
        log_every_n_batch (int): Frequency that we should log information while looping through the dataset
    """

    batch_size: int = 10_000
    file_buffer_size: int = 100 * 1024 * 1024
    deterministic: bool = False
    use_interleave: bool = True
    num_parallel_file_reads: int = 64
    ram_budget_multiplier: float = 0.5
    log_every_n_batch: int = 1000


def empty_entity_tensors(info: SerializedTFRecordInfo) -> LoadedEntityTensors:
    """Returns the tensors of an entity with no rows, shaped as a load of ``info`` with rows would be.

    Args:
        info (SerializedTFRecordInfo): How the entity is stored.

    Returns:
        LoadedEntityTensors: Zero-row tensors:

            - ids: int64 ``[0]`` for nodes, ``[2, 0]`` for edges.
            - features: ``[0, feature_dim]``, or None without ``feature_keys``.
            - packed_features: uint8 ``[0, packed_feature_dim]``, or None without ``packed_feature_key``.
            - labels: ``[0, len(label_keys)]``, or None without ``label_keys``.
    """
    if info.is_node_entity:
        ids = torch.empty(0, dtype=torch.int64)
    else:
        ids = torch.empty(2, 0, dtype=torch.int64)
    features: Optional[torch.Tensor] = None
    if info.feature_keys:
        features = torch.empty(0, info.feature_dim)
    labels: Optional[torch.Tensor] = None
    if info.label_keys:
        labels = torch.empty(0, len(info.label_keys))
    packed_features: Optional[torch.Tensor] = None
    if info.packed_feature_key is not None:
        packed_features = torch.empty(0, info.packed_feature_dim, dtype=torch.uint8)
    return LoadedEntityTensors(
        ids=ids,
        features=features,
        packed_features=packed_features,
        labels=labels,
    )
