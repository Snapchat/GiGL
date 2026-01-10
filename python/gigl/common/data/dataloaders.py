import time
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import psutil
import tensorflow as tf
import torch

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.utils.decorator import tf_on_cpu
from gigl.src.common.types.features import FeatureTypes
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict

logger = Logger()


class LoadedEntityTensors(NamedTuple):
    ids: torch.Tensor
    features: Optional[torch.Tensor]
    labels: Optional[torch.Tensor]


@dataclass(frozen=True)
class SerializedTFRecordInfo:
    """
    Stores information pertaining to how a single entity (node, edge, positive label, negative label) and single node/edge type in the heterogeneous case is serialized on disk.
    This field is used as input to the TFRecordDataLoader.load_as_torch_tensor() function for loading torch tensors.
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
    entity_key: str | Tuple[str, str]
    # Name of the label columns for the current entity, defaults to an empty list.
    label_keys: Sequence[str] = field(default_factory=list)
    # The regex pattern to match the TFRecord files at the specified prefix
    tfrecord_uri_pattern: str = ".*-of-.*\.tfrecord(\.gz)?$"

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


def _get_labels_from_features(
    feature_and_label_tensor: tf.Tensor, label_dim: int
) -> tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
    """
    Given a combined tensor of features and labels, returns the features and labels separately.
    Args:
        feature_and_label_tensor (tf.Tensor): Tensor of features and labels
        label_dim (int): Dimension of the labels
    Returns:
        feature_tensor (Optional[tf.Tensor]): Tensor of features
        label_tensor (Optional[tf.Tensor]): Tensor of labels
    """

    if len(feature_and_label_tensor.shape) != 2:
        raise ValueError(
            f"Expected tensor to be 2D for extracting labels, but got shape {feature_and_label_tensor.shape}"
        )

    _, feature_and_label_dim = feature_and_label_tensor.shape

    if not (0 <= label_dim <= feature_and_label_dim):
        raise ValueError(
            f"Got invalid label dim {label_dim} for extracting labels which must inclusively be between 0 and {feature_and_label_dim} for tensor of shape {feature_and_label_tensor.shape}"
        )

    feature_dim = feature_and_label_dim - label_dim

    if feature_dim == 0:
        feature_tensor = None
    else:
        feature_tensor = feature_and_label_tensor[:, :feature_dim]

    if label_dim == 0:
        label_tensor = None
    else:
        label_tensor = feature_and_label_tensor[:, feature_dim:]

    return (
        feature_tensor,
        label_tensor,
    )


def _concatenate_features_by_names(
    feature_key_to_tf_tensor: dict[str, tf.Tensor],
    feature_keys: Sequence[str],
    label_keys: Sequence[str],
) -> tuple[Optional[tf.Tensor], Optional[tf.Tensor]]:
    """
    Concatenates feature tensors in the order specified by feature names.
    Also concatenates labels to the end of the feature list if they are present using the corresponding label key

    It is assumed that feature_names is a subset of the keys in feature_name_to_tf_tensor.

    Args:
        feature_key_to_tf_tensor (dict[str, tf.Tensor]): A dictionary mapping feature names to their corresponding tf tensors.
        feature_keys (Sequence[str]): A list of feature names specifying the order in which tensors should be concatenated.
        label_keys (Sequence[str]): Name of the label columns for the current entity.

    Returns:
        Tuple[
            Optional[tf.Tensor]: A concatenated tensor of the features in the specified order of feature_keys.
            Optional[tf.Tensor]: A concatenated tensor of the labels in the specified order of label_keys.
        ]
    """

    features: list[tf.Tensor] = []
    label_dim = 0

    feature_iterable = list(feature_keys)

    for label_key in label_keys:
        feature_iterable.append(label_key)

    for feature_key in feature_iterable:
        tensor = feature_key_to_tf_tensor[feature_key]

        # TODO(kmonte, xgao, zfan): We will need to add support for this if we're trying to scale up.
        # Features may be stored as int type. We cast it to float here and will need to subsequently convert
        # it back to int. Note that this is ok for small int values (less than 2^24, or ~16 million).
        # For large int values, we will need to round it when converting back
        # from float, as otherwise there will be precision loss.
        if tensor.dtype != tf.float32:
            tensor = tf.cast(tensor, tf.float32)

        # Reshape 1D tensor to column vector
        if len(tensor.shape) == 1:
            tensor = tf.expand_dims(tensor, axis=-1)

        # Calculate label dimension by summing dimensions of label tensors
        if feature_key in label_keys:
            label_dim += tensor.shape[-1]

        features.append(tensor)

    combined_feature_tensor = tf.concat(features, axis=1)

    return _get_labels_from_features(combined_feature_tensor, label_dim)


def _tf_tensor_to_torch_tensor(tf_tensor: tf.Tensor) -> torch.Tensor:
    """
    Converts a TensorFlow tensor to a PyTorch tensor using DLPack to ensure zero-copy conversion.

    Args:
        tf_tensor (tf.Tensor): The TensorFlow tensor to convert.

    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """
    return torch.utils.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(tf_tensor))


def _build_example_parser(
    *,
    feature_spec: FeatureSpecDict,
) -> Callable[[bytes], dict[str, tf.Tensor]]:
    # Wrapping this partial with tf.function gives us a speedup.
    # https://www.tensorflow.org/guide/function
    @tf.function
    def _parse_example(
        example_proto: bytes, spec: FeatureSpecDict
    ) -> dict[str, tf.Tensor]:
        return tf.io.parse_example(example_proto, spec)

    return partial(_parse_example, spec=feature_spec)


class TFRecordDataLoader:
    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._world_size = world_size

    def _partition_children_uris(
        self,
        uri: Uri,
        tfrecord_pattern: str,
    ) -> Sequence[Uri]:
        """
        Partition the children of `uri` evenly by world_size. The partitions differ in size by at most 1 file.

        As an implementation detail, the *leading* partitions may be larger.

        Ex:
        world_size: 4, files: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Partitions: [[0, 1, 2], [3, 4, 5], [6, 7], [8, 9]]

        Args:
            uri (Uri): The parent uri for whoms children should be partitioned.
            tfrecord_pattern (str): Regex pattern to match for loading serialized tfrecords from uri prefix

        Returns:
            list[Uri]: The list of file Uris for the current partition.
        """
        file_loader = FileLoader()
        uris = sorted(
            file_loader.list_children(uri, pattern=tfrecord_pattern),
            key=lambda uri: uri.uri,
        )
        if len(uris) == 0:
            logger.warning(f"Found no children for uri: {uri}")

        # Compute the number of fields per partition and the number of partitions which will be larger.
        files_per_partition, extra_partitions = divmod(len(uris), self._world_size)

        if self._rank < extra_partitions:
            start_index = self._rank * (files_per_partition + 1)
        else:
            extra_offset = extra_partitions * (files_per_partition + 1)
            offset_index = self._rank - extra_partitions
            start_index = offset_index * files_per_partition + extra_offset

        # Calculate the end index for the current partition
        end_index = (
            start_index + files_per_partition + 1
            if self._rank < extra_partitions
            else start_index + files_per_partition
        )

        logger.info(
            f"Loading files by partitions.\n"
            f"Total files: {len(uris)}\n"
            f"World size: {self._world_size}\n"
            f"Current partition: {self._rank}\n"
            f"Files in current partition: {end_index - start_index}\n"
        )
        if start_index >= end_index:
            logger.info(f"No files to load for rank: {self._rank}.")
        else:
            logger.info(
                f"Current partition start file uri: {uris[start_index]}\n"
                f"Current partition end file uri: {uris[end_index-1]}"
            )

        # Return the subset of file Uris for the current partition
        return uris[start_index:end_index]

    @staticmethod
    def _build_dataset_for_uris(
        uris: Sequence[Uri],
        feature_spec: FeatureSpecDict,
        opts: TFDatasetOptions = TFDatasetOptions(),
    ) -> tf.data.Dataset:
        """
        Builds a tf.data.Dataset to load tf.Examples serialized as TFRecord files into tf.Tensors. This function will
        automatically infer the compression type (if any) from the suffix of the files located at the TFRecord URI.

        Args:
            uris (Sequence[Uri]): The URIs of the TFRecord files to load.
            feature_spec (FeatureSpecDict): The feature spec to use when parsing the tf.Examples.
            opts (TFDatasetOptions): The options to use when building the dataset.
        Returns:
            tf.data.Dataset: The dataset to load the TFRecords
        """
        logger.info(f"Building dataset for with opts: {opts}")
        data_opts = tf.data.Options()
        data_opts.autotune.ram_budget = int(
            psutil.virtual_memory().total * opts.ram_budget_multiplier
        )
        logger.info(f"Setting RAM budget to {data_opts.autotune.ram_budget}")
        # TODO (mkolodner-sc): Throw error if we observe folder with mixed gz / tfrecord files
        compression_type = (
            "GZIP" if all([uri.uri.endswith(".gz") for uri in uris]) else None
        )
        if opts.use_interleave:
            # Using .batch on the interleaved dataset provides a huge speed up (60%).
            # Using map on the interleaved dataset provides another smaller speedup (5%)
            dataset = (
                tf.data.Dataset.from_tensor_slices([uri.uri for uri in uris])
                .interleave(
                    lambda uri: tf.data.TFRecordDataset(
                        uri,
                        compression_type=compression_type,
                        buffer_size=opts.file_buffer_size,
                    )
                    .batch(
                        opts.batch_size,
                        num_parallel_calls=tf.data.AUTOTUNE,
                        deterministic=opts.deterministic,
                    )
                    .prefetch(tf.data.AUTOTUNE),
                    cycle_length=tf.data.AUTOTUNE,
                    deterministic=opts.deterministic,
                    num_parallel_calls=tf.data.AUTOTUNE,
                )
                .with_options(data_opts)
            )
        else:
            dataset = tf.data.TFRecordDataset(
                [uri.uri for uri in uris],
                compression_type=compression_type,
                buffer_size=opts.file_buffer_size,
                num_parallel_reads=opts.num_parallel_file_reads,
            ).batch(
                opts.batch_size,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=opts.deterministic,
            )

        return dataset.map(
            _build_example_parser(feature_spec=feature_spec),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=opts.deterministic,
        ).prefetch(tf.data.AUTOTUNE)

    @tf_on_cpu
    def load_as_torch_tensors(
        self,
        serialized_tf_record_info: SerializedTFRecordInfo,
        tf_dataset_options: TFDatasetOptions = TFDatasetOptions(),
    ) -> LoadedEntityTensors:
        """
        Loads torch tensors from a set of TFRecord files.

        Args:
            serialized_tf_record_info (SerializedTFRecordInfo): Information for how TFRecord files are serialized on disk.
            tf_dataset_options (TFDatasetOptions): The options to use when building the dataset.
        Returns:
            LoadedEntityTensors: The (id_tensor, feature_tensor, label_tensor) for the loaded entities.
        """
        entity_key = serialized_tf_record_info.entity_key
        feature_keys = serialized_tf_record_info.feature_keys
        label_keys = serialized_tf_record_info.label_keys

        # We make a deep copy of the feature spec dict so that future modifications don't redirect to the input

        feature_spec_dict = deepcopy(serialized_tf_record_info.feature_spec)

        if serialized_tf_record_info.is_node_entity:
            assert isinstance(entity_key, str)
            id_concat_axis = 0
            proccess_id_tensor = lambda t: t[entity_key]
            entity_type = FeatureTypes.NODE

            # We manually inject the node id into the FeatureSpecDict so that the schema will include
            # node ids in the produced batch when reading serialized tfrecords.
            if entity_key not in feature_spec_dict:
                logger.info(
                    f"Injecting entity key {entity_key} into feature spec dictionary with value `tf.io.FixedLenFeature(shape=[], dtype=tf.int64)`"
                )
                feature_spec_dict[entity_key] = tf.io.FixedLenFeature(
                    shape=[], dtype=tf.int64
                )
        else:
            id_concat_axis = 1
            proccess_id_tensor = lambda t: tf.stack(
                [t[entity_key[0]], t[entity_key[1]]], axis=0
            )
            entity_type = FeatureTypes.EDGE

            # We manually inject the edge ids into the FeatureSpecDict so that the schema will include
            # edge ids in the produced batch when reading serialized tfrecords.
            if entity_key[0] not in feature_spec_dict:
                logger.info(
                    f"Injecting entity key {entity_key[0]} into feature spec dictionary with value `tf.io.FixedLenFeature(shape=[], dtype=tf.int64)`"
                )
                feature_spec_dict[entity_key[0]] = tf.io.FixedLenFeature(
                    shape=[], dtype=tf.int64
                )

            if entity_key[1] not in feature_spec_dict:
                logger.info(
                    f"Injecting entity key {entity_key[1]} into feature spec dictionary with value `tf.io.FixedLenFeature(shape=[], dtype=tf.int64)`"
                )
                feature_spec_dict[entity_key[1]] = tf.io.FixedLenFeature(
                    shape=[], dtype=tf.int64
                )

        uris = self._partition_children_uris(
            serialized_tf_record_info.tfrecord_uri_prefix,
            serialized_tf_record_info.tfrecord_uri_pattern,
        )
        if not uris:
            logger.info(
                f"No files to load for rank: {self._rank} and entity type: {entity_type.name}, returning empty tensors."
            )
            empty_entity = (
                torch.empty(0)
                if entity_type == FeatureTypes.NODE
                else torch.empty(2, 0)
            )
            if feature_keys:
                empty_feature = torch.empty(0, serialized_tf_record_info.feature_dim)
            else:
                empty_feature = None

            if label_keys:
                empty_label = torch.empty(0, len(label_keys))
            else:
                empty_label = None

            return LoadedEntityTensors(
                ids=empty_entity, features=empty_feature, labels=empty_label
            )

        dataset = TFRecordDataLoader._build_dataset_for_uris(
            uris=uris,
            feature_spec=feature_spec_dict,
            opts=tf_dataset_options,
        )

        start_time = time.perf_counter()
        num_entities_processed = 0
        id_tensors: list[torch.Tensor] = []
        feature_tensors: list[torch.Tensor] = []
        label_tensors: list[torch.Tensor] = []
        for idx, batch in enumerate(dataset):
            id_tensors.append(proccess_id_tensor(batch))
            if feature_keys or label_keys:
                feature_tensor, label_tensor = _concatenate_features_by_names(
                    batch, feature_keys, label_keys
                )
                if feature_tensor is not None:
                    feature_tensors.append(feature_tensor)
                if label_tensor is not None:
                    label_tensors.append(label_tensor)
            num_entities_processed += (
                id_tensors[-1].shape[0]
                if entity_type == FeatureTypes.NODE
                else id_tensors[-1].shape[1]
            )
            if (idx + 1) % tf_dataset_options.log_every_n_batch == 0:
                logger.info(
                    f"Processed {idx + 1:,} total batches with {num_entities_processed:,} {entity_type.name}"
                )
        end = time.perf_counter()
        logger.info(
            f"Processed {num_entities_processed:,} {entity_type.name} records in {end - start_time:.2f} seconds, {num_entities_processed / (end - start_time):,.2f} records per second"
        )
        start = time.perf_counter()
        id_tensor = _tf_tensor_to_torch_tensor(
            tf.concat(id_tensors, axis=id_concat_axis)
        )
        output_feature_tensor: Optional[torch.Tensor] = None
        output_label_tensor: Optional[torch.Tensor] = None
        if feature_tensors:
            output_feature_tensor = _tf_tensor_to_torch_tensor(
                tf.concat(feature_tensors, axis=0)
            )
        if label_tensors:
            output_label_tensor = _tf_tensor_to_torch_tensor(
                tf.concat(label_tensors, axis=0)
            )

        if output_feature_tensor is not None and output_label_tensor is not None:
            assert output_feature_tensor.size(0) == output_label_tensor.size(
                0
            ), f"Loaded {output_feature_tensor.size(0)} features and {output_label_tensor.size(0)} labels, but they must be the same."

        end = time.perf_counter()
        logger.info(
            f"Converted {num_entities_processed:,} {entity_type.name} to torch tensors in {end - start:.2f} seconds"
        )
        return LoadedEntityTensors(
            ids=id_tensor, features=output_feature_tensor, labels=output_label_tensor
        )
