"""Fetching a raw feature matrix and its packed sidecar with one request per partition.

A node or edge type can store its columns in a raw float matrix, in a packed ``uint8`` sidecar (feature quantization),
or in both. Fetching two stores through two ``DistFeature`` objects costs two RPCs per remote partition for every
batch. ``FusedFeatureStore`` serves whichever stores a type has as one row of bytes (the float columns reinterpreted as
bytes, followed by the sidecar columns) and ``FusedDistFeature`` is a ``DistFeature`` over such stores, so one lookup is
one request per partition for every type. ``FusedDistFeature.async_get`` resolves to the raw matrix and the sidecar
rows, the latter ``None`` for a type without a sidecar.

All stores are CPU stores: GiGL builds every ``Feature`` with ``with_gpu=False``.
"""

from typing import Optional, Union

import torch
from graphlearn_torch.data import Feature
from graphlearn_torch.distributed.dist_feature import (
    DistFeature,
    RpcFeatureLookupCallee,
)
from graphlearn_torch.distributed.rpc import RpcDataPartitionRouter, rpc_register
from graphlearn_torch.partition import GLTPartitionBook, PartitionBook
from graphlearn_torch.typing import EdgeType, NodeType
from graphlearn_torch.utils import ensure_device, get_available_device
from jaxtyping import Float, Int64, UInt8


class FusedFeatureStore:
    """A raw float feature store and/or its packed sidecar, served to other partitions as one ``uint8`` row per entity.

    A served row holds the raw float columns reinterpreted as bytes, then the sidecar columns, then zero padding up to
    a multiple of the raw dtype's size.
    The padding keeps every row aligned, so the raw columns of a received row can be read as floats without a copy.

    A missing or zero-width store contributes no columns. Without a raw store the raw dtype is ``float32``, the dtype
    the raw matrix is later rebuilt as. Without a sidecar the row is the raw bytes alone, which need no padding. A row
    is never narrower than one raw element, so a type with no columns at all is served as one zero element.

    Both stores must be CPU stores.

    Example:
        A ``float32`` raw store with 3 columns (12 bytes) and a 2-column ``uint8`` sidecar make 14 bytes, padded to 16.
        Bytes 0-11 are the raw floats, bytes 12-13 the sidecar and bytes 14-15 zero padding.

        >>> raw = Feature(torch.ones(4, 3), id2index=torch.arange(4), with_gpu=False)
        >>> packed = Feature(
        ...     torch.full((4, 2), 7, dtype=torch.uint8),
        ...     id2index=torch.arange(4),
        ...     with_gpu=False,
        ...     dtype=torch.uint8,
        ... )
        >>> store = FusedFeatureStore(raw, packed)
        >>> rows = store.cpu_get(torch.tensor([0, 2]))
        >>> rows.shape
        torch.Size([2, 16])
        >>> rows[0, 12:].tolist()
        [7, 7, 0, 0]
        >>> raw_rows, packed_rows = store.split(rows)
        >>> raw_rows.shape, packed_rows.shape
        (torch.Size([2, 3]), torch.Size([2, 2]))

    Args:
        raw: Raw float feature store, or ``None`` if the type has none.
        packed: Packed ``uint8`` sidecar store with the same ids and id mapping as ``raw``, or ``None``.

    Raises:
        ValueError: If both stores are ``None``.
    """

    def __init__(self, raw: Optional[Feature], packed: Optional[Feature]) -> None:
        if raw is None and packed is None:
            raise ValueError("A fused feature store needs a raw store or a sidecar.")
        self._raw = raw
        self._packed = packed

    def lazy_init_with_ipc_handle(self) -> None:
        """Materializes the stores after an IPC transfer."""
        for store in (self._raw, self._packed):
            if store is not None:
                store.lazy_init_with_ipc_handle()

    @property
    def _raw_dtype(self) -> torch.dtype:
        if self._raw is None:
            return torch.float32
        self._raw.lazy_init_with_ipc_handle()
        return self._raw.feature_tensor.dtype

    @property
    def _raw_row_bytes(self) -> int:
        if self._raw is None:
            return 0
        return self._raw.shape[1] * self._raw_dtype.itemsize

    @property
    def _packed_dim(self) -> int:
        if self._packed is None:
            return 0
        return self._packed.shape[1]

    @property
    def _row_bytes(self) -> int:
        itemsize = self._raw_dtype.itemsize
        unpadded = self._raw_row_bytes + self._packed_dim
        # A row is at least one element wide, as a zero-width byte view cannot be read as floats.
        return max(itemsize, (unpadded + itemsize - 1) // itemsize * itemsize)

    def __getitem__(
        self, ids: Int64[torch.Tensor, "num_ids"]
    ) -> tuple[
        Float[torch.Tensor, "num_ids raw_dim"],
        Optional[UInt8[torch.Tensor, "num_ids packed_dim"]],
    ]:
        """Raw float rows and packed sidecar rows of ids that are local to this store.

        The raw rows have zero columns without a raw store and the sidecar rows are ``None`` without a sidecar.
        """
        if self._raw is None:
            raw_rows = torch.empty(len(ids), 0, dtype=torch.float32)
        else:
            raw_rows = self._raw[ids]
        if self._packed is None:
            packed_rows = None
        else:
            packed_rows = self._packed[ids]
        return raw_rows, packed_rows

    def cpu_get(
        self, ids: Int64[torch.Tensor, "num_ids"]
    ) -> UInt8[torch.Tensor, "num_ids row_bytes"]:
        """Served rows of ids from the CPU tensors, as ``[len(ids), row_bytes]`` bytes, for answering remote requests.

        Each store is gathered straight into its columns of the output, so every value is copied once.
        """
        rows = torch.empty(len(ids), self._row_bytes, dtype=torch.uint8)
        raw_rows, packed_rows = self.split(rows)
        for store, out in ((self._raw, raw_rows), (self._packed, packed_rows)):
            if store is None:
                continue
            store.lazy_init_with_ipc_handle()
            index = ids.to("cpu")
            if store.id2index is not None:
                index = store.id2index[index]
            torch.index_select(store.feature_tensor, 0, index, out=out)
        rows[:, self._raw_row_bytes + self._packed_dim :] = 0
        return rows

    def split(
        self, rows: UInt8[torch.Tensor, "num_rows row_bytes"]
    ) -> tuple[
        Float[torch.Tensor, "num_rows raw_dim"],
        UInt8[torch.Tensor, "num_rows packed_dim"],
    ]:
        """Views of served byte rows as the raw float matrix and the packed sidecar rows, without copying.

        The views have zero columns for a missing store.
        """
        raw_bytes = self._raw_row_bytes
        return (
            rows[:, :raw_bytes].view(self._raw_dtype),
            rows[:, raw_bytes : raw_bytes + self._packed_dim],
        )


class FusedDistFeature(DistFeature):
    """A ``DistFeature`` whose local stores are ``FusedFeatureStore`` objects.

    GLT's ``DistFeature`` looks up local rows and answers remote requests by indexing its local stores with ids, which
    ``FusedFeatureStore`` supports; only the constructor checks the store type, so it is repeated here without that
    check.

    ``async_get`` resolves to the raw float matrix and the packed sidecar rows, stitched straight from the local rows
    and the received byte rows. The sidecar rows are ``None`` for a type without a sidecar.

    Args:
        num_partitions: Number of data partitions.
        partition_idx: Data partition of the current process.
        local_feature: A fused store, or one per node/edge type for heterogeneous data.
        feature_pb: Partition book of the entities, or one per type for heterogeneous data.
        rpc_router: Router used to reach the other partitions.
        device: Device used for computing.
    """

    def __init__(
        self,
        num_partitions: int,
        partition_idx: int,
        local_feature: Union[
            FusedFeatureStore, dict[Union[NodeType, EdgeType], FusedFeatureStore]
        ],
        feature_pb: Union[
            PartitionBook, dict[Union[NodeType, EdgeType], PartitionBook]
        ],
        rpc_router: RpcDataPartitionRouter,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_partitions = num_partitions
        self.partition_idx = partition_idx
        self.device = get_available_device(device)
        ensure_device(self.device)

        self.local_feature = local_feature
        if isinstance(local_feature, FusedFeatureStore):
            self.data_cls = "homo"
            local_feature.lazy_init_with_ipc_handle()
        else:
            self.data_cls = "hetero"
            for store in local_feature.values():
                store.lazy_init_with_ipc_handle()

        self.feature_pb = feature_pb
        if isinstance(feature_pb, dict):
            assert self.data_cls == "hetero"
            for key, book in feature_pb.items():
                if not isinstance(book, PartitionBook):
                    feature_pb[key] = GLTPartitionBook(book)
        elif isinstance(feature_pb, torch.Tensor):
            assert self.data_cls == "homo"
            self.feature_pb = GLTPartitionBook(feature_pb)
        else:
            assert self.data_cls == "homo"

        self.rpc_router = rpc_router
        self.rpc_callee_id = rpc_register(RpcFeatureLookupCallee(self))

    def async_get(
        self,
        ids: Int64[torch.Tensor, "num_ids"],
        input_type: Optional[Union[NodeType, EdgeType]] = None,
    ) -> torch.futures.Future[
        tuple[
            Float[torch.Tensor, "num_ids raw_dim"],
            Optional[UInt8[torch.Tensor, "num_ids packed_dim"]],
        ]
    ]:
        """Looks up the raw float rows and packed sidecar rows of ids.

        Args:
            ids: Ids to look up.
            input_type: Node or edge type of the ids for heterogeneous data.

        Returns:
            A future of the ``[len(ids), raw_dim]`` raw matrix and the ``[len(ids), packed_dim]`` sidecar rows, or
            ``None`` for the sidecar rows if the type has no sidecar.
        """
        remote_future = self._remote_selecting_get(ids, input_type)
        local_rows, local_index = self._local_selecting_get(ids, input_type)
        result_future: torch.futures.Future[
            tuple[
                Float[torch.Tensor, "num_ids raw_dim"],
                Optional[UInt8[torch.Tensor, "num_ids packed_dim"]],
            ]
        ] = torch.futures.Future()

        def on_done(*_: object) -> None:
            try:
                result = self._stitch_fused(
                    ids, local_rows, local_index, remote_future.wait()
                )
            except Exception as e:
                # torch types set_exception as taking the result type, not an exception.
                result_future.set_exception(e)  # ty: ignore[invalid-argument-type]
            else:
                result_future.set_result(result)

        remote_future.add_done_callback(on_done)
        return result_future

    def _stitch_fused(
        self,
        ids: Int64[torch.Tensor, "num_ids"],
        local_rows: tuple[
            Float[torch.Tensor, "num_local raw_dim"],
            Optional[UInt8[torch.Tensor, "num_local packed_dim"]],
        ],
        local_index: Int64[torch.Tensor, "num_local"],
        remotes: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[
        Float[torch.Tensor, "num_ids raw_dim"],
        Optional[UInt8[torch.Tensor, "num_ids packed_dim"]],
    ]:
        """Puts the local rows and the received byte rows of each remote partition into the raw and sidecar outputs.

        The sidecar output is ``None``, and costs nothing, if the local rows have no sidecar.
        """
        local_raw_rows, local_packed_rows = local_rows
        if local_index.numel() == ids.numel():
            # Every id is local, so the local rows are already complete and in order.
            if local_packed_rows is None:
                return local_raw_rows.to(self.device), None
            return local_raw_rows.to(self.device), local_packed_rows.to(self.device)
        raw_output = torch.empty(
            ids.shape[0],
            local_raw_rows.shape[1],
            dtype=local_raw_rows.dtype,
            device=self.device,
        )
        packed_output = None
        if local_packed_rows is not None:
            packed_output = torch.empty(
                ids.shape[0],
                local_packed_rows.shape[1],
                dtype=local_packed_rows.dtype,
                device=self.device,
            )
        local_index = local_index.to(self.device)
        raw_output[local_index] = local_raw_rows.to(self.device)
        if packed_output is not None and local_packed_rows is not None:
            packed_output[local_index] = local_packed_rows.to(self.device)
        # Each remote partition answered with byte rows that hold every store the type has: the raw columns, then
        # the sidecar columns. The layout is the one of FusedFeatureStore.split, read here without its per-call lookups.
        raw_bytes = local_raw_rows.shape[1] * local_raw_rows.element_size()
        for remote_rows, remote_index in remotes:
            remote_rows = remote_rows.to(self.device)
            remote_index = remote_index.to(self.device)
            raw_output[remote_index] = remote_rows[:, :raw_bytes].view(
                local_raw_rows.dtype
            )
            if packed_output is not None:
                packed_output[remote_index] = remote_rows[
                    :, raw_bytes : raw_bytes + packed_output.shape[1]
                ]
        return raw_output, packed_output


def build_fused_dist_feature(
    num_partitions: int,
    partition_idx: int,
    raw_features: Optional[Union[Feature, dict[Union[NodeType, EdgeType], Feature]]],
    packed_features: Optional[Union[Feature, dict[Union[NodeType, EdgeType], Feature]]],
    feature_pb: Union[PartitionBook, dict[Union[NodeType, EdgeType], PartitionBook]],
    rpc_router: RpcDataPartitionRouter,
    device: Optional[torch.device] = None,
) -> Optional[FusedDistFeature]:
    """Builds a ``FusedDistFeature`` over every entity type that has a raw store, a packed sidecar, or both.

    Args:
        num_partitions: Number of data partitions.
        partition_idx: Data partition of the current process.
        raw_features: Raw float store, or one per type.
        packed_features: Packed sidecar store, or one per type.
        feature_pb: Partition book of the entities, or one per type.
        rpc_router: Router used to reach the other partitions.
        device: Device used for computing.

    Returns:
        ``None`` if there are no stores at all.
    """
    if raw_features is None and packed_features is None:
        return None
    local_feature: Union[
        FusedFeatureStore, dict[Union[NodeType, EdgeType], FusedFeatureStore]
    ]
    if isinstance(raw_features, Feature) or isinstance(packed_features, Feature):
        assert raw_features is None or isinstance(raw_features, Feature)
        assert packed_features is None or isinstance(packed_features, Feature)
        local_feature = FusedFeatureStore(raw_features, packed_features)
    else:
        raw_by_type = raw_features or {}
        packed_by_type = packed_features or {}
        local_feature = {
            key: FusedFeatureStore(raw_by_type.get(key), packed_by_type.get(key))
            for key in raw_by_type.keys() | packed_by_type.keys()
        }
        if not local_feature:
            return None
    return FusedDistFeature(
        num_partitions,
        partition_idx,
        local_feature,
        feature_pb,
        rpc_router,
        device,
    )
