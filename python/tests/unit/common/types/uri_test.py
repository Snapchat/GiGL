import unittest
from pathlib import Path

from gigl.common.types.uri.gcs_uri import GcsUri
from gigl.common.types.uri.http_uri import HttpUri
from gigl.common.types.uri.local_uri import LocalUri
from gigl.common.types.uri.uri import Uri
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

    def test_join(self):
        joined: Uri
        with self.subTest("LocalUri"):
            joined = LocalUri.join("/foo/bar", "file.txt")
            self.assertEqual(joined, LocalUri("/foo/bar/file.txt"))
            self.assertIsInstance(joined, LocalUri)
        with self.subTest("HttpUri"):
            joined = HttpUri.join("http://abc.com/xyz", "foo")
            self.assertEqual(joined, HttpUri("http://abc.com/xyz/foo"))
            self.assertIsInstance(joined, HttpUri)
        with self.subTest("GcsUri"):
            joined = GcsUri.join("gs://bucket/", "file.txt")
            self.assertEqual(joined, GcsUri("gs://bucket/file.txt"))
            self.assertIsInstance(joined, GcsUri)
        with self.subTest("LocalUri with Path"):
            joined = LocalUri.join("/foo/bar", Path("file.text"))
            self.assertEqual(joined, LocalUri("/foo/bar/file.text"))

    def test_div_join(self):
        joined: Uri
        with self.subTest("LocalUri"):
            joined = LocalUri("/foo/bar") / "baz"
            self.assertEqual(joined, LocalUri("/foo/bar/baz"))
            self.assertIsInstance(joined, LocalUri)
        with self.subTest("HttpUri"):
            joined = HttpUri("http://abc.com/xyz") / "foo"
            self.assertEqual(joined, HttpUri("http://abc.com/xyz/foo"))
            self.assertIsInstance(joined, HttpUri)
        with self.subTest("GcsUri"):
            joined = GcsUri("gs://bucket/") / "file.txt"
            self.assertEqual(joined, GcsUri("gs://bucket/file.txt"))
            self.assertIsInstance(joined, GcsUri)
        with self.subTest("LocalUri with Path"):
            joined = LocalUri("/foo/bar") / Path("file.text")
            self.assertEqual(joined, LocalUri("/foo/bar/file.text"))
            self.assertIsInstance(joined, LocalUri)

    def test_div_join_invalid_type(self):
        with self.subTest("LocalUri / HttpUri"):
            with self.assertRaises(TypeError):
                LocalUri("/foo/bar") / HttpUri("http://abc.com/xyz")
        with self.subTest("LocalUri / GcsUri"):
            with self.assertRaises(TypeError):
                LocalUri("/foo/bar") / GcsUri("gs://bucket/path/to")
        with self.subTest("HttpUri / LocalUri"):
            with self.assertRaises(TypeError):
                HttpUri("http://abc.com/xyz") / LocalUri("/foo/bar")
        with self.subTest("HttpUri / GcsUri"):
            with self.assertRaises(TypeError):
                HttpUri("http://abc.com/xyz") / GcsUri("gs://bucket/path/to")
        with self.subTest("GcsUri / LocalUri"):
            with self.assertRaises(TypeError):
                GcsUri("gs://bucket/path/to") / LocalUri("/foo/bar")
        with self.subTest("GcsUri / HttpUri"):
            with self.assertRaises(TypeError):
                GcsUri("gs://bucket/path/to") / HttpUri("http://abc.com/xyz")


if __name__ == "__main__":
    unittest.main()
