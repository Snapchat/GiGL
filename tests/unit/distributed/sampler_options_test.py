from gigl.distributed.sampler_options import (
    CustomSamplerOptions,
    KHopNeighborSamplerOptions,
    resolve_sampler_options,
)
from tests.test_assets.test_case import TestCase


class ResolveSamplerOptionsTest(TestCase):
    def test_both_none_raises(self):
        with self.assertRaises(ValueError):
            resolve_sampler_options(num_neighbors=None, sampler_options=None)

    def test_num_neighbors_only(self):
        num_neighbors, options = resolve_sampler_options(
            num_neighbors=[2, 2], sampler_options=None
        )
        self.assertEqual(num_neighbors, [2, 2])
        assert isinstance(options, KHopNeighborSamplerOptions)
        self.assertEqual(options.num_neighbors, [2, 2])

    def test_khop_options_only(self):
        opts = KHopNeighborSamplerOptions(num_neighbors=[3, 3])
        num_neighbors, options = resolve_sampler_options(
            num_neighbors=None, sampler_options=opts
        )
        self.assertEqual(num_neighbors, [3, 3])
        self.assertIs(options, opts)

    def test_khop_options_matching_num_neighbors(self):
        opts = KHopNeighborSamplerOptions(num_neighbors=[2, 2])
        num_neighbors, options = resolve_sampler_options(
            num_neighbors=[2, 2], sampler_options=opts
        )
        self.assertEqual(num_neighbors, [2, 2])
        self.assertIs(options, opts)

    def test_khop_options_conflicting_num_neighbors_raises(self):
        opts = KHopNeighborSamplerOptions(num_neighbors=[3, 3])
        with self.assertRaises(ValueError):
            resolve_sampler_options(num_neighbors=[2, 2], sampler_options=opts)

    def test_custom_options_without_num_neighbors(self):
        opts = CustomSamplerOptions(class_path="my.module.MySampler")
        num_neighbors, options = resolve_sampler_options(
            num_neighbors=None, sampler_options=opts
        )
        self.assertEqual(num_neighbors, [])
        self.assertIs(options, opts)

    def test_custom_options_with_num_neighbors(self):
        opts = CustomSamplerOptions(class_path="my.module.MySampler")
        num_neighbors, options = resolve_sampler_options(
            num_neighbors=[2, 2], sampler_options=opts
        )
        self.assertEqual(num_neighbors, [2, 2])
        self.assertIs(options, opts)
