from unittest.mock import Mock, patch

import torch
import torch.multiprocessing as mp
from absl.testing import absltest
from graphlearn_torch.data import Feature
from graphlearn_torch.distributed import (
    barrier,
    init_rpc,
    init_worker_group,
    shutdown_rpc,
)
from graphlearn_torch.distributed.dist_feature import DistFeature
from graphlearn_torch.distributed.rpc import (
    RpcDataPartitionRouter,
    rpc_sync_data_partitions,
)
from graphlearn_torch.utils import id2idx

from gigl.distributed.utils import get_free_port
from gigl.distributed.utils.fused_feature import (
    FusedDistFeature,
    FusedFeatureStore,
    build_fused_dist_feature,
)
from tests.test_assets.test_case import TestCase

_NUM_ENTITIES = 12
_RAW_COLUMNS = 3
_PACKED_COLUMNS = 5


class _CountingDistFeature(DistFeature):
    """Counts the lookups other partitions request from this one."""

    served_requests = 0

    def local_get(self, ids, input_type=None):
        self.served_requests += 1
        return super().local_get(ids, input_type)


class _CountingFusedDistFeature(FusedDistFeature):
    served_requests = 0

    def local_get(self, ids, input_type=None):
        self.served_requests += 1
        return super().local_get(ids, input_type)


def _features(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(_NUM_ENTITIES, _RAW_COLUMNS, generator=generator)
    packed = torch.randint(
        0, 256, (_NUM_ENTITIES, _PACKED_COLUMNS), generator=generator, dtype=torch.uint8
    )
    return raw, packed


def _local_stores(
    rank: int, raw: torch.Tensor, packed: torch.Tensor, owner: torch.Tensor
) -> tuple[Feature, Feature]:
    """This partition's rows of the global matrices, indexed by global id."""
    ids = torch.nonzero(owner == rank).flatten()
    return (
        Feature(raw[ids], id2index=id2idx(ids), with_gpu=False),
        Feature(packed[ids], id2index=id2idx(ids), with_gpu=False, dtype=torch.uint8),
    )


def _lookup_on_partition(
    rank: int, master_port: int, output: dict, is_heterogeneous: bool
) -> None:
    """Looks up every type through GLT ``DistFeature`` objects and through the fused feature.

    Heterogeneous data has a type with both stores, a raw-only type and a sidecar-only type.
    """
    init_worker_group(world_size=2, rank=rank)
    init_rpc(master_addr="127.0.0.1", master_port=master_port, num_rpc_threads=4)
    router = RpcDataPartitionRouter(rpc_sync_data_partitions(2, rank))
    owner = torch.arange(_NUM_ENTITIES) % 2
    types = ["user", "story", "ad"] if is_heterogeneous else [None]
    globals_by_type = {t: _features(seed) for seed, t in enumerate(types)}
    stores = {t: _local_stores(rank, *globals_by_type[t], owner) for t in types}
    # "story" has no sidecar and "ad" has no raw store.
    raw_stores = {t: stores[t][0] for t in types if t != "ad"}
    packed_stores = {t: stores[t][1] for t in types if t != "story"}

    def hetero_or_homo(per_type: dict):
        return per_type if is_heterogeneous else per_type[None]

    pb = hetero_or_homo({t: owner for t in types})
    separate_raw = _CountingDistFeature(
        2, rank, hetero_or_homo(raw_stores), pb, False, router
    )
    separate_packed = _CountingDistFeature(
        2, rank, hetero_or_homo(packed_stores), pb, False, router
    )
    fused = build_fused_dist_feature(
        2,
        rank,
        hetero_or_homo(raw_stores),
        hetero_or_homo(packed_stores),
        pb,
        router,
    )
    assert fused is not None
    # Count the requests the fused feature serves instead of those of a plain one.
    counting_fused = _CountingFusedDistFeature(2, rank, fused.local_feature, pb, router)

    # Ids from both partitions, repeated and unordered.
    ids = torch.tensor([5, 0, 7, 7, 2, 11, 4, 9])
    matches = True
    for t in types:
        expected_raw, expected_packed = globals_by_type[t]
        fused_raw, fused_packed = counting_fused.async_get(ids, t).wait()
        matches &= fused_raw.is_contiguous()
        if t in raw_stores:
            raw = separate_raw.async_get(ids, t).wait()
            matches &= torch.equal(raw, expected_raw[ids])
            matches &= torch.equal(fused_raw, expected_raw[ids])
        else:
            matches &= fused_raw.shape == (len(ids), 0)
            matches &= fused_raw.dtype == torch.float32
        if t in packed_stores:
            packed = separate_packed.async_get(ids, t).wait()
            assert fused_packed is not None
            matches &= fused_packed.is_contiguous()
            matches &= torch.equal(packed, expected_packed[ids])
            matches &= torch.equal(fused_packed, expected_packed[ids])
        else:
            matches &= fused_packed is None
    barrier()
    output[rank] = (
        matches,
        separate_raw.served_requests + separate_packed.served_requests,
        counting_fused.served_requests,
    )
    barrier()
    shutdown_rpc()


def _store(raw: torch.Tensor, packed: torch.Tensor) -> FusedFeatureStore:
    return FusedFeatureStore(
        Feature(raw, id2index=torch.arange(len(raw)), with_gpu=False),
        Feature(
            packed,
            id2index=torch.arange(len(packed)),
            with_gpu=False,
            dtype=torch.uint8,
        ),
    )


# Three float32 columns and five sidecar columns are 17 bytes, padded to a multiple of 4.
_SERVED_ROW_BYTES = 20


class FusedFeatureStoreTest(TestCase):
    def test_served_rows_split_back_into_the_raw_and_packed_matrices(self) -> None:
        raw, packed = _features(0)
        store = _store(raw, packed)
        ids = torch.tensor([3, 3, 0, 11])

        local_raw, local_packed = store[ids]
        served = store.cpu_get(ids)
        split_raw, split_packed = store.split(served)

        assert local_packed is not None
        self.assertTrue(torch.equal(local_raw, raw[ids]))
        self.assertTrue(torch.equal(local_packed, packed[ids]))
        self.assertEqual(served.dtype, torch.uint8)
        self.assertEqual(served.shape, (4, _SERVED_ROW_BYTES))
        self.assertTrue(
            torch.equal(served[:, -3:], torch.zeros(4, 3, dtype=torch.uint8))
        )
        self.assertTrue(torch.equal(split_raw, raw[ids]))
        self.assertTrue(torch.equal(split_packed, packed[ids]))

    def test_a_single_served_row_splits_like_many(self) -> None:
        raw, packed = _features(2)
        store = _store(raw, packed)
        split_raw, split_packed = store.split(store.cpu_get(torch.tensor([7])))
        self.assertTrue(torch.equal(split_raw, raw[7:8]))
        self.assertTrue(torch.equal(split_packed, packed[7:8]))

    def test_no_ids_gives_empty_rows_of_the_full_width(self) -> None:
        raw, packed = _features(1)
        store = _store(raw, packed)
        served = store.cpu_get(torch.empty(0, dtype=torch.long))
        split_raw, split_packed = store.split(served)
        self.assertEqual(served.shape, (0, _SERVED_ROW_BYTES))
        self.assertEqual(split_raw.shape, (0, _RAW_COLUMNS))
        self.assertEqual(split_packed.shape, (0, _PACKED_COLUMNS))

    def test_other_dtypes_widths_and_id_mappings_match_separate_lookups(self) -> None:
        for raw_dtype, raw_columns, packed_columns, has_id2index in (
            (torch.float64, 3, 5, True),
            (torch.bfloat16, 3, 1, False),
            (torch.float32, 1, 0, True),
        ):
            with self.subTest(raw_dtype=raw_dtype, raw_columns=raw_columns):
                raw = torch.randn(_NUM_ENTITIES, raw_columns).to(raw_dtype)
                packed = torch.randint(
                    0, 255, (_NUM_ENTITIES, packed_columns), dtype=torch.uint8
                )
                id2index = torch.randperm(_NUM_ENTITIES) if has_id2index else None
                store = FusedFeatureStore(
                    Feature(raw, id2index=id2index, with_gpu=False, dtype=raw_dtype),
                    Feature(
                        packed, id2index=id2index, with_gpu=False, dtype=torch.uint8
                    ),
                )
                for ids in (
                    torch.tensor([5, 1, 1]),
                    torch.tensor([2]),
                    torch.empty(0, dtype=torch.long),
                ):
                    assert store._raw is not None and store._packed is not None
                    expected_raw = store._raw.cpu_get(ids)
                    expected_packed = store._packed.cpu_get(ids)
                    served = store.cpu_get(ids)
                    split_raw, split_packed = store.split(served)
                    self.assertEqual(split_raw.dtype, raw_dtype)
                    self.assertTrue(torch.equal(split_raw, expected_raw))
                    self.assertTrue(torch.equal(split_packed, expected_packed))
                    self.assertEqual(served.shape[1] % raw_dtype.itemsize, 0)

    def test_raw_only_rows_are_the_raw_bytes_without_padding(self) -> None:
        raw, _ = _features(0)
        store = FusedFeatureStore(
            Feature(raw, id2index=torch.arange(len(raw)), with_gpu=False), None
        )
        ids = torch.tensor([4, 4, 1])

        local_raw, local_packed = store[ids]
        served = store.cpu_get(ids)
        split_raw, split_packed = store.split(served)

        self.assertIsNone(local_packed)
        self.assertTrue(torch.equal(local_raw, raw[ids]))
        self.assertEqual(served.shape, (3, _RAW_COLUMNS * 4))
        self.assertTrue(torch.equal(split_raw, raw[ids]))
        self.assertEqual(split_packed.shape, (3, 0))

    def test_sidecar_only_rows_have_a_float32_raw_matrix_without_columns(self) -> None:
        _, packed = _features(0)
        store = FusedFeatureStore(
            None,
            Feature(
                packed,
                id2index=torch.arange(len(packed)),
                with_gpu=False,
                dtype=torch.uint8,
            ),
        )
        ids = torch.tensor([4, 4, 1])

        local_raw, local_packed = store[ids]
        served = store.cpu_get(ids)
        split_raw, split_packed = store.split(served)

        self.assertEqual(local_raw.shape, (3, 0))
        self.assertEqual(local_raw.dtype, torch.float32)
        assert local_packed is not None
        self.assertTrue(torch.equal(local_packed, packed[ids]))
        # Five sidecar bytes padded to a multiple of the float32 size.
        self.assertEqual(served.shape, (3, 8))
        self.assertEqual(split_raw.shape, (3, 0))
        self.assertEqual(split_raw.dtype, torch.float32)
        self.assertTrue(torch.equal(split_packed, packed[ids]))

    def test_zero_width_raw_stores_give_empty_raw_rows(self) -> None:
        for raw_dtype in (torch.float32, torch.float64, torch.bfloat16):
            for packed_columns in (0, 5):
                for ids in (
                    torch.tensor([5, 1, 1]),
                    torch.tensor([2]),
                    torch.empty(0, dtype=torch.long),
                ):
                    with self.subTest(
                        raw_dtype=raw_dtype,
                        packed_columns=packed_columns,
                        num_ids=len(ids),
                    ):
                        packed = torch.randint(
                            0, 255, (_NUM_ENTITIES, packed_columns), dtype=torch.uint8
                        )
                        store = FusedFeatureStore(
                            Feature(
                                torch.empty(_NUM_ENTITIES, 0, dtype=raw_dtype),
                                id2index=torch.arange(_NUM_ENTITIES),
                                with_gpu=False,
                                dtype=raw_dtype,
                            ),
                            Feature(
                                packed,
                                id2index=torch.arange(_NUM_ENTITIES),
                                with_gpu=False,
                                dtype=torch.uint8,
                            )
                            if packed_columns
                            else None,
                        )

                        local_raw, local_packed = store[ids]
                        split_raw, split_packed = store.split(store.cpu_get(ids))

                        for raw_rows in (local_raw, split_raw):
                            self.assertEqual(raw_rows.shape, (len(ids), 0))
                            self.assertEqual(raw_rows.dtype, raw_dtype)
                        self.assertEqual(split_packed.shape, (len(ids), packed_columns))
                        if packed_columns:
                            self.assertTrue(torch.equal(split_packed, packed[ids]))
                            assert local_packed is not None
                            self.assertTrue(torch.equal(local_packed, packed[ids]))
                        else:
                            self.assertIsNone(local_packed)

    def test_a_store_needs_at_least_one_part(self) -> None:
        with self.assertRaises(ValueError):
            FusedFeatureStore(None, None)


class BuildFusedDistFeatureTest(TestCase):
    def test_zero_width_raw_stores_are_kept_with_no_columns(self) -> None:
        raw, packed = _features(0)
        no_raw_columns = Feature(
            torch.empty(_NUM_ENTITIES, 0),
            id2index=torch.arange(_NUM_ENTITIES),
            with_gpu=False,
        )
        sidecar = Feature(
            packed,
            id2index=torch.arange(_NUM_ENTITIES),
            with_gpu=False,
            dtype=torch.uint8,
        )
        owner = torch.zeros(_NUM_ENTITIES, dtype=torch.long)
        router = Mock(spec=RpcDataPartitionRouter)

        with patch("gigl.distributed.utils.fused_feature.rpc_register"):
            homogeneous = build_fused_dist_feature(
                1, 0, no_raw_columns, sidecar, owner, router
            )
            fused = build_fused_dist_feature(
                1,
                0,
                {
                    "user": no_raw_columns,
                    "story": Feature(
                        raw, id2index=torch.arange(_NUM_ENTITIES), with_gpu=False
                    ),
                },
                {"user": sidecar, "story": sidecar},
                {"user": owner, "story": owner},
                router,
            )
        assert homogeneous is not None
        assert isinstance(homogeneous.local_feature, FusedFeatureStore)
        self.assertIs(homogeneous.local_feature._raw, no_raw_columns)
        assert fused is not None
        assert isinstance(fused.local_feature, dict)
        self.assertEqual(set(fused.local_feature), {"user", "story"})
        self.assertIs(fused.local_feature["user"]._raw, no_raw_columns)
        self.assertIsNotNone(fused.local_feature["story"]._raw)

    def test_types_without_a_sidecar_are_fused_as_raw_only(self) -> None:
        raw, packed = _features(0)
        raw_feature = Feature(raw, id2index=torch.arange(_NUM_ENTITIES), with_gpu=False)
        sidecar = Feature(
            packed,
            id2index=torch.arange(_NUM_ENTITIES),
            with_gpu=False,
            dtype=torch.uint8,
        )
        owner = torch.zeros(_NUM_ENTITIES, dtype=torch.long)
        router = Mock(spec=RpcDataPartitionRouter)
        raw_by_type = {"user": raw_feature, "story": raw_feature}
        owner_by_type = {"user": owner, "story": owner}

        with patch("gigl.distributed.utils.fused_feature.rpc_register"):
            homogeneous = build_fused_dist_feature(
                1, 0, raw_feature, None, owner, router
            )
            fused = build_fused_dist_feature(
                1, 0, raw_by_type, {"story": sidecar}, owner_by_type, router
            )
            sidecar_only_types = build_fused_dist_feature(
                1, 0, None, {"story": sidecar}, owner_by_type, router
            )
        assert homogeneous is not None
        assert isinstance(homogeneous.local_feature, FusedFeatureStore)
        self.assertIsNone(homogeneous.local_feature._packed)
        assert fused is not None
        assert isinstance(fused.local_feature, dict)
        self.assertEqual(set(fused.local_feature), {"user", "story"})
        self.assertIsNone(fused.local_feature["user"]._packed)
        self.assertIsNotNone(fused.local_feature["story"]._packed)
        assert sidecar_only_types is not None
        assert isinstance(sidecar_only_types.local_feature, dict)
        self.assertEqual(set(sidecar_only_types.local_feature), {"story"})

    def test_no_stores_gives_no_fused_feature(self) -> None:
        owner = torch.zeros(_NUM_ENTITIES, dtype=torch.long)
        router = Mock(spec=RpcDataPartitionRouter)
        self.assertIsNone(build_fused_dist_feature(1, 0, None, None, owner, router))
        self.assertIsNone(
            build_fused_dist_feature(1, 0, {}, None, {"user": owner}, router)
        )


class FusedDistFeatureStitchTest(TestCase):
    def test_stitching_local_and_remote_rows_restores_the_requested_order(
        self,
    ) -> None:
        owner = torch.arange(_NUM_ENTITIES) % 2
        ids = torch.tensor([5, 0, 7, 7, 2, 11])
        for raw_dtype in (torch.float32, torch.float64, torch.bfloat16):
            for raw_columns, packed_columns in ((3, 5), (3, 0), (0, 5), (0, 0)):
                with self.subTest(
                    raw_dtype=raw_dtype,
                    raw_columns=raw_columns,
                    packed_columns=packed_columns,
                ):
                    raw = torch.randn(_NUM_ENTITIES, raw_columns).to(raw_dtype)
                    packed = torch.randint(
                        0, 255, (_NUM_ENTITIES, packed_columns), dtype=torch.uint8
                    )
                    store = FusedFeatureStore(
                        Feature(
                            raw,
                            id2index=torch.arange(_NUM_ENTITIES),
                            with_gpu=False,
                            dtype=raw_dtype,
                        ),
                        Feature(
                            packed,
                            id2index=torch.arange(_NUM_ENTITIES),
                            with_gpu=False,
                            dtype=torch.uint8,
                        )
                        if packed_columns
                        else None,
                    )
                    with patch("gigl.distributed.utils.fused_feature.rpc_register"):
                        feature = FusedDistFeature(
                            2, 0, store, owner, Mock(spec=RpcDataPartitionRouter)
                        )
                    local_positions = torch.nonzero(owner[ids] == 0).flatten()
                    remote_positions = torch.nonzero(owner[ids] == 1).flatten()

                    stitched_raw, stitched_packed = feature._stitch_fused(
                        ids,
                        store[ids[local_positions]],
                        local_positions,
                        [(store.cpu_get(ids[remote_positions]), remote_positions)],
                    )

                    self.assertEqual(stitched_raw.dtype, raw_dtype)
                    self.assertTrue(torch.equal(stitched_raw, raw[ids]))
                    if packed_columns:
                        assert stitched_packed is not None
                        self.assertTrue(torch.equal(stitched_packed, packed[ids]))
                    else:
                        self.assertIsNone(stitched_packed)


class FusedDistFeatureTest(TestCase):
    def _run(self, is_heterogeneous: bool) -> dict:
        output = mp.Manager().dict()
        mp.spawn(
            _lookup_on_partition,
            args=(get_free_port(), output, is_heterogeneous),
            nprocs=2,
        )
        return dict(output)

    def test_homogeneous_lookup_matches_separate_lookups_with_half_the_requests(
        self,
    ) -> None:
        for rank, (matches, separate_requests, fused_requests) in self._run(
            False
        ).items():
            self.assertTrue(matches, f"rank {rank} fetched different rows")
            self.assertEqual(separate_requests, 2)
            self.assertEqual(fused_requests, 1)

    def test_heterogeneous_lookup_needs_one_request_per_type_and_partition(
        self,
    ) -> None:
        for rank, (matches, separate_requests, fused_requests) in self._run(
            True
        ).items():
            self.assertTrue(matches, f"rank {rank} fetched different rows")
            # Three node types, each fetched once from the other partition. The separate lookups need one
            # request for each store a type has, which is 4 across the three types.
            self.assertEqual(separate_requests, 4)
            self.assertEqual(fused_requests, 3)


if __name__ == "__main__":
    absltest.main()
