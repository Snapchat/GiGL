from tests.test_assets.test_case import TestCase


class TestCollateCoreModule(TestCase):
    def test_module_imports_and_ping(self) -> None:
        from gigl_core import collate_core

        self.assertEqual(collate_core.ping(), 0)
