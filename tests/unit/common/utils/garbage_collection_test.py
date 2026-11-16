import gc

from absl.testing import absltest

from gigl.common.utils.garbage_collection import freeze_gc
from tests.test_assets.test_case import TestCase


class FreezeGcTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        gc.unfreeze()

    def tearDown(self) -> None:
        gc.unfreeze()
        super().tearDown()

    def test_freezes_inside_block_and_unfreezes_after(self) -> None:
        with freeze_gc():
            self.assertGreater(gc.get_freeze_count(), 0)
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_nested_use_keeps_outer_freeze(self) -> None:
        with freeze_gc():
            with freeze_gc():
                pass
            self.assertGreater(gc.get_freeze_count(), 0)
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_decorator_form(self) -> None:
        @freeze_gc()
        def frozen_count() -> int:
            return gc.get_freeze_count()

        self.assertGreater(frozen_count(), 0)
        self.assertEqual(gc.get_freeze_count(), 0)

    def test_unfreezes_after_exception(self) -> None:
        with self.assertRaises(ValueError):
            with freeze_gc():
                raise ValueError("boom")
        self.assertEqual(gc.get_freeze_count(), 0)


if __name__ == "__main__":
    absltest.main()
