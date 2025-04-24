import unittest

from gigl.common.types.uri.uri_factory import UriFactory


class UriTest(unittest.TestCase):
    def test_can_get_basename(self):
        file_name = "file.txt"
        gcs_uri_full = UriFactory.create_uri(f"gs://bucket/path/to/{file_name}")
        local_uri_full = UriFactory.create_uri(f"/path/to/{file_name}")
        http_uri_full = UriFactory.create_uri(f"http://abc.com/xyz/{file_name}")

        self.assertEqual(file_name, gcs_uri_full.get_basename())
        self.assertEqual(file_name, local_uri_full.get_basename())
        self.assertEqual(file_name, http_uri_full.get_basename())
