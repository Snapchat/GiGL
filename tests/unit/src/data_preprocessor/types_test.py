from gigl.src.data_preprocessor.lib.types import FeatureQuantizationSpec
from tests.test_assets.test_case import TestCase


class FeatureQuantizationSpecTest(TestCase):
    def test_rejects_empty_feature_keys(self) -> None:
        with self.assertRaises(ValueError):
            FeatureQuantizationSpec(feature_keys=[], bits=2)

    def test_rejects_duplicate_feature_keys(self) -> None:
        with self.assertRaises(ValueError):
            FeatureQuantizationSpec(feature_keys=["feature", "feature"], bits=2)

    def test_rejects_unsupported_bit_width(self) -> None:
        with self.assertRaises(ValueError):
            FeatureQuantizationSpec(feature_keys=["feature"], bits=3)
