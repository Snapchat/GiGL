from gigl.distributed.sampler_options import (
    KHopNeighborSamplerOptions,
    resolve_sampler_options,
)
from tests.test_assets.test_case import TestCase


class ResolveSamplerOptionsTest(TestCase):
    def test_num_neighbors_only(self):
        options = resolve_sampler_options(num_neighbors=[2, 2], sampler_options=None)
        assert isinstance(options, KHopNeighborSamplerOptions)
        self.assertEqual(options.num_neighbors, [2, 2])

    def test_khop_options_matching_num_neighbors(self):
        opts = KHopNeighborSamplerOptions(num_neighbors=[2, 2])
        options = resolve_sampler_options(num_neighbors=[2, 2], sampler_options=opts)
        self.assertIs(options, opts)

    def test_khop_options_conflicting_num_neighbors_raises(self):
        opts = KHopNeighborSamplerOptions(num_neighbors=[3, 3])
        with self.assertRaises(ValueError):
            resolve_sampler_options(num_neighbors=[2, 2], sampler_options=opts)
