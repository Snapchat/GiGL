import time
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import psutil
import tensorflow as tf
import torch
import tqdm

from gigl.common import Uri
from gigl.common.logger import Logger
from gigl.common.utils.decorator import tf_on_cpu
from gigl.src.common.types.features import FeatureTypes
from gigl.src.common.utils.file_loader import FileLoader
from gigl.src.data_preprocessor.lib.types import FeatureSpecDict

logger = Logger()


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
    # a dict of feature name -> FeatureSpec (eg. FixedLenFeature, VarlenFeature, SparseFeature, RaggedFeature). If entity keys are not present, we insert them during tensor loading
    feature_spec: FeatureSpecDict
    # Feature dimension of current entity
    feature_dim: int
    # Entity ID Key for current entity. If this is a Node Entity, this must be a string. If this is an edge entity, this must be a Tuple[str, str] for the source and destination ids.
    entity_key: Union[str, Tuple[str, str]]
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
    Options for tuning a tf.data.Dataset.

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
        ram_budget_multiplier (float): The multiplier of the total system memory to set as the tf.data RAM budget..
    """

    batch_size: int = 10_000
    file_buffer_size: int = 100 * 1024 * 1024
    deterministic: bool = False
    use_interleave: bool = True
    num_parallel_file_reads: int = 64
    ram_budget_multiplier: float = 0.5


def _concatenate_features_by_names(
    feature_key_to_tf_tensor: Dict[str, tf.Tensor],
    feature_keys: Sequence[str],
) -> tf.Tensor:
    """
    Concatenates feature tensors in the order specified by feature names.

    It is assumed that feature_names is a subset of the keys in feature_name_to_tf_tensor.

    Args:
        feature_key_to_tf_tensor (Dict[str, tf.Tensor]): A dictionary mapping feature names to their corresponding tf tensors.
        feature_keys (List[str]): A list of feature names specifying the order in which tensors should be concatenated.

    Returns:
        tf.Tensor: A concatenated tensor of the features in the specified order.
    """

    features: List[tf.Tensor] = []

    for feature_key in feature_keys:
        tensor = feature_key_to_tf_tensor[feature_key]

        # TODO(kmonte, xgao, zfan): We will need to add support for this if we're trying to scale up.
        # Some features (e.g., home city, last city, etc.) are vocabulary
        # ids and are stored as int type. We cast it to float here and convert
        # it back to int before feeding it to the feature embedding layer.
        # Note that this is ok for small int values (less than 2^24, or ~16 million).
        # For large int values, we will need to round it when converting back
        # from float, as otherwise there will be precision loss.
        if tensor.dtype != tf.float32:
            tensor = tf.cast(tensor, tf.float32)

        # Reshape 1D tensor to column vector
        if len(tensor.shape) == 1:
            tensor = tf.expand_dims(tensor, axis=-1)

        features.append(tensor)

    return tf.concat(features, axis=1)


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
) -> Callable[[bytes], Dict[str, tf.Tensor]]:
    # Wrapping this partial with tf.function gives us a speedup.
    # https://www.tensorflow.org/guide/function
    @tf.function
    def _parse_example(
        example_proto: bytes, spec: FeatureSpecDict
    ) -> Dict[str, tf.Tensor]:
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
            List[Uri]: The list of file Uris for the current partition.
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Loads torch tensors from a set of TFRecord files.

        Args:
            serialized_tf_record_info (SerializedTFRecordInfo): Information for how TFRecord files are serialized on disk.
            tf_dataset_options (TFDatasetOptions): The options to use when building the dataset.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The (id_tensor, feature_tensor) for the loaded entities.
        """
        entity_key = serialized_tf_record_info.entity_key
        feature_keys = serialized_tf_record_info.feature_keys

        # We make a deep copy of the feature spec dict so that future modifications don't redirect to the input

        feature_spec_dict = deepcopy(serialized_tf_record_info.feature_spec)

        if isinstance(entity_key, str):
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
            empty_feature = (
                torch.empty(0, serialized_tf_record_info.feature_dim)
                if feature_keys
                else None
            )
            return empty_entity, empty_feature

        dataset = TFRecordDataLoader._build_dataset_for_uris(
            uris=uris,
            feature_spec=feature_spec_dict,
            opts=tf_dataset_options,
        )

        start_time = time.perf_counter()
        num_entities_processed = 0
        id_tensors = []
        feature_tensors = []
        for batch in tqdm.tqdm(dataset):
            id_tensors.append(proccess_id_tensor(batch))
            if feature_keys:
                feature_tensors.append(
                    _concatenate_features_by_names(batch, feature_keys)
                )
            num_entities_processed += (
                id_tensors[-1].shape[0]
                if entity_type == FeatureTypes.NODE
                else id_tensors[-1].shape[1]
            )
        end = time.perf_counter()
        logger.info(
            f"Processed {num_entities_processed:,} {entity_type.name} records in {end - start_time:.2f} seconds, {num_entities_processed / (end - start_time):,.2f} records per second"
        )
        start = time.perf_counter()
        id_tensor = _tf_tensor_to_torch_tensor(
            tf.concat(id_tensors, axis=id_concat_axis)
        )
        feature_tensor = (
            _tf_tensor_to_torch_tensor(tf.concat(feature_tensors, axis=0))
            if feature_tensors
            else None
        )
        end = time.perf_counter()
        logger.info(
            f"Converted {num_entities_processed:,} {entity_type.name} to torch tensors in {end - start:.2f} seconds"
        )
        return id_tensor, feature_tensor
