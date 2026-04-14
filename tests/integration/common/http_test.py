import tempfile

from gigl.common import HttpUri, LocalUri
from gigl.common.utils.http import HttpUtils
from tests.test_assets.test_case import TestCase


class HttpUtilsTest(TestCase):
    def test_download_file_from_http(self):
        http_utils = HttpUtils()
        http_path = HttpUri(
            "https://raw.githubusercontent.com/Snapchat/GiGL/refs/heads/main/LICENSE"
        )
        with tempfile.NamedTemporaryFile() as f:
            local_path = LocalUri(f.name)
            http_utils.download_files_from_http(
                {http_path: local_path},
            )

            # We assert that the licence file was downloaded correctly
            with open(local_path.uri) as file:
                read_text = file.read()
                self.assertTrue(
                    "MIT License" in read_text
                    and "Copyright" in read_text
                    and "Snap Inc." in read_text
                )
