from typing import Callable, Dict, Generic, Iterable, Iterator, List, TypeVar

import numpy as np
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_transform as tft
import torch.utils.data

from gigl.common import Uri
from gigl.src.common.types.graph_data import NodeType
from gigl.src.training.v1.lib.data_loaders.utils import (
    get_data_split_for_current_worker,
)

T = TypeVar("T")
UriType = TypeVar("UriType", bound=Uri)


# We are using Pytorch for training, no need to load tf tensors into gpu.
# Loading tf tensors into GPU can cause some issues whilst using pytorch dataloader
# with multiple workers.
# Disable all GPUS
tf.config.set_visible_devices([], "GPU")
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != "GPU"


class _TfRecordDatasetIterable(Generic[T]):
    def __init__(
        self,
        tf_record_numpy_iterator: Iterator[bytes],
        process_raw_sample_fn: Callable[[bytes], T],
    ):
        self.__tf_record_numpy_iterator: tf.data.TFRecordDataset = (
            tf_record_numpy_iterator
        )
        self.__process_raw_sample_fn = process_raw_sample_fn

    def __iter__(self):
        return self

    def __next__(self) -> T:
        raw_data = next(self.__tf_record_numpy_iterator)
        processed_sample = self.__process_raw_sample_fn(raw_data)
        return processed_sample


class TfRecordsIterableDataset(torch.utils.data.IterableDataset, Generic[T]):
    def __init__(
        self,
        tf_record_uris: List[UriType],
        process_raw_sample_fn: Callable[[bytes], T],
        seed: int = 42,
    ) -> None:
        """
        Args:
            tf_record_uris (List[UriType]): Holds all the uris for the dataset.
            Note: for now only uris supported are ones that `tf.data.TFRecordDataset`
            can load from default; i.e .GcsUri and LocalUri.
            We permute the file list based on a seed as a means of "shuffling" the data
            on a file-level (rather than sample-level, as would be possible in cases
            where the data fits in memory.
        """
        assert isinstance(tf_record_uris, list)
        self._tf_record_uris: np.ndarray = np.random.RandomState(seed).permutation(
            np.array([uri.uri for uri in tf_record_uris])
        )

        # Subsequently reference to the TFRecordDataset that is to be initialized too
        self.__process_raw_sample_fn: Callable[[bytes], T] = process_raw_sample_fn

    def __iter__(self) -> Iterator[T]:
        # Need to first split the work based on worker information
        current_workers_tf_record_uris_to_process = get_data_split_for_current_worker(
            self._tf_record_uris
        )
        raw_dataset = tf.data.TFRecordDataset(current_workers_tf_record_uris_to_process)
        return _TfRecordDatasetIterable(
            tf_record_numpy_iterator=raw_dataset.as_numpy_iterator(),
            process_raw_sample_fn=self.__process_raw_sample_fn,
        )


class LoopyIterableDataset(torch.utils.data.IterableDataset, Generic[T]):
    """
    Takes as input an IterableDataset and makes it "loopy," so that the dataset
    can be iterated over cyclically.
    """

    def __init__(self, iterable_dataset: torch.utils.data.IterableDataset[T]) -> None:
        self._iterable_dataset = iterable_dataset

    @staticmethod
    def _custom_cycle_impl(iterable: Iterable):
        """
        Create an infinitely-loopable Iterable from a finite one.  This implementation
        differs from itertools.cycle in memory overhead: https://github.com/pytorch/pytorch/issues/23900.
        """

        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def __iter__(self) -> Iterator[T]:
        return self._custom_cycle_impl(iterable=self._iterable_dataset)


class CombinedIterableDatasets(torch.utils.data.IterableDataset, Generic[T]):
    def __init__(
        self,
        iterable_dataset_map: Dict[
            NodeType,
            torch.utils.data.IterableDataset[T],
        ],
    ) -> None:
        self._iterable_dataset_map = iterable_dataset_map

    def __iter__(self) -> Iterator[Dict[NodeType, T]]:
        iterators = [
            (node_type, iter(dataset))
            for node_type, dataset in self._iterable_dataset_map.items()
        ]
        while True:
            combined_data: Dict[NodeType, T] = {}
            for node_type, iterator in iterators:
                samples = next(iterator)
                combined_data[node_type] = samples
            yield combined_data


def get_np_iterator_from_tfrecords(schema_path: Uri, tfrecord_files: List[str]):
    batch_size = 1
    schema = tfdv.load_schema_text(schema_path.uri)
    feature_spec = tft.tf_metadata.schema_utils.schema_as_feature_spec(
        schema
    ).feature_spec
    dataset = (
        tf.data.TFRecordDataset(tfrecord_files)
        .map(lambda record: tf.io.parse_example(record, feature_spec))
        .batch(batch_size)
        .as_numpy_iterator()
    )
    return dataset
