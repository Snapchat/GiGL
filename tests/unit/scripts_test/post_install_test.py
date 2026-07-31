from unittest.mock import patch

from gigl.scripts import post_install
from tests.test_assets.test_case import TestCase


class PostInstallTest(TestCase):
    def test_cli_propagates_nonzero_main_result(self) -> None:
        with patch.object(post_install, "main", return_value=7):
            with self.assertRaises(SystemExit) as context:
                post_install.cli()

        self.assertEqual(context.exception.code, 7)
