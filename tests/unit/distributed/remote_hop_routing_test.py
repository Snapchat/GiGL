import hashlib
import inspect

from absl.testing import absltest
from graphlearn_torch.distributed import DistNeighborSampler as GLTDistNeighborSampler
from graphlearn_torch.distributed.dist_feature import DistFeature

from tests.test_assets.test_case import TestCase

# sha256 of inspect.getsource of graphlearn_torch's DistNeighborSampler._sample_one_hop, which
# BaseDistNeighborSampler._sample_one_hop copies. To re-sync after a deliberate GLT upgrade, diff the two
# functions, port GLT's change into the copy, then set the hash to the one the failing test prints.
_GLT_SAMPLE_ONE_HOP_SHA256 = (
    "2158a4886c544e60380cc72e1011630dfd1f833ceb64bd50019f594f609fcb5d"
)

# sha256 of inspect.getsource of graphlearn_torch's DistFeature._remote_selecting_get, which
# FusedDistFeature._remote_selecting_get copies. To re-sync after a deliberate GLT upgrade, diff the two
# functions, port GLT's change into the copy, then set the hash to the one the failing test prints.
_GLT_REMOTE_SELECTING_GET_SHA256 = (
    "3dadb30e482876da4ba68760833e1543f883c6014472c89c725187f836a0d970"
)


class GltSampleOneHopDriftTest(TestCase):
    def test_glt_sample_one_hop_source_is_unchanged(self) -> None:
        source = inspect.getsource(GLTDistNeighborSampler._sample_one_hop)
        digest = hashlib.sha256(source.encode()).hexdigest()
        self.assertEqual(
            digest,
            _GLT_SAMPLE_ONE_HOP_SHA256,
            "graphlearn_torch's DistNeighborSampler._sample_one_hop changed. Re-diff it against "
            "gigl.distributed.base_sampler.BaseDistNeighborSampler._sample_one_hop and port the "
            f"change. After a deliberate GLT upgrade, set _GLT_SAMPLE_ONE_HOP_SHA256 to {digest}.",
        )


class GltRemoteSelectingGetDriftTest(TestCase):
    def test_glt_remote_selecting_get_source_is_unchanged(self) -> None:
        source = inspect.getsource(DistFeature._remote_selecting_get)
        digest = hashlib.sha256(source.encode()).hexdigest()
        self.assertEqual(
            digest,
            _GLT_REMOTE_SELECTING_GET_SHA256,
            "graphlearn_torch's DistFeature._remote_selecting_get changed. Re-diff it against "
            "gigl.distributed.utils.fused_feature.FusedDistFeature._remote_selecting_get and port "
            f"the change. After a deliberate GLT upgrade, set _GLT_REMOTE_SELECTING_GET_SHA256 to {digest}.",
        )


if __name__ == "__main__":
    absltest.main()
