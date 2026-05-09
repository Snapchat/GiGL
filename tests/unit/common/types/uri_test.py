from pathlib import Path

from absl.testing import absltest

from gigl.common.types.uri.gcs_uri import GcsUri
from gigl.common.types.uri.http_uri import HttpUri
from gigl.common.types.uri.local_uri import LocalUri
from gigl.common.types.uri.uri import Uri
from gigl.common.types.uri.uri_factory import UriFactory
from tests.test_assets.test_case import TestCase


class UriTest(TestCase):
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
        with self.subTest("str suffix"):
            relative_path_str = "file.txt"
            with self.subTest("LocalUri"):
                joined = LocalUri.join("/foo/bar", relative_path_str)
                self.assertEqual(joined, LocalUri("/foo/bar/file.txt"))
                self.assertIsInstance(joined, LocalUri)
            with self.subTest("HttpUri"):
                joined = HttpUri.join("http://abc.com/xyz", relative_path_str)
                self.assertEqual(joined, HttpUri("http://abc.com/xyz/file.txt"))
                self.assertIsInstance(joined, HttpUri)
            with self.subTest("GcsUri"):
                joined = GcsUri.join("gs://bucket/", relative_path_str)
                self.assertEqual(joined, GcsUri("gs://bucket/file.txt"))
                self.assertIsInstance(joined, GcsUri)
            with self.subTest("Uri"):
                joined = Uri.join(GcsUri("gs://bucket"), relative_path_str)
                self.assertEqual(joined, Uri("gs://bucket/file.txt"))
                self.assertIsInstance(joined, Uri)

        with self.subTest("Path suffix"):
            relative_path = Path("file.txt")
            with self.subTest("LocalUri"):
                joined = LocalUri.join("/foo/bar", relative_path)
                self.assertEqual(joined, LocalUri("/foo/bar/file.txt"))
                self.assertIsInstance(joined, LocalUri)
            with self.subTest("HttpUri"):
                joined = HttpUri.join("http://abc.com/xyz", relative_path)
                self.assertEqual(joined, HttpUri("http://abc.com/xyz/file.txt"))
                self.assertIsInstance(joined, HttpUri)
            with self.subTest("GcsUri"):
                joined = GcsUri.join("gs://bucket/", relative_path)
                self.assertEqual(joined, GcsUri("gs://bucket/file.txt"))
                self.assertIsInstance(joined, GcsUri)

        with self.subTest("LocalUri suffix"):
            relative_local_uri = LocalUri("file.txt")
            with self.subTest("LocalUri"):
                joined = LocalUri.join("/foo/bar", relative_local_uri)
                self.assertEqual(joined, LocalUri("/foo/bar/file.txt"))
                self.assertIsInstance(joined, LocalUri)

    def test_join_invalid_suffix(self):
        with self.subTest("relative LocalUri suffix with non-local join"):
            relative_local_uri = LocalUri("file.txt")
            with self.assertRaises(TypeError):
                HttpUri.join("http://abc.com/xyz", relative_local_uri)
            with self.assertRaises(TypeError):
                GcsUri.join("gs://bucket/path", relative_local_uri)
            with self.assertRaises(TypeError):
                Uri.join("foo", relative_local_uri)

        with self.subTest("mixed Uri first token"):
            with self.assertRaises(TypeError):
                LocalUri.join(GcsUri("gs://bucket/path"), "file.txt")
            with self.assertRaises(TypeError):
                GcsUri.join(HttpUri("http://abc.com/xyz"), "file.txt")
            with self.assertRaises(TypeError):
                HttpUri.join(LocalUri("/foo/bar"), "file.txt")

        with self.subTest("absolute LocalUri suffix"):
            absolute_local_uri = LocalUri("/other/file.txt")
            with self.assertRaises(TypeError):
                LocalUri.join("/foo/bar", absolute_local_uri)
            with self.assertRaises(TypeError):
                HttpUri.join("http://abc.com/xyz", absolute_local_uri)
            with self.assertRaises(TypeError):
                GcsUri.join("gs://bucket/path", absolute_local_uri)

        with self.subTest("HttpUri suffix"):
            http_uri = HttpUri("http://abc.com/file.txt")
            with self.assertRaises(TypeError):
                LocalUri.join("/foo/bar", http_uri)
            with self.assertRaises(TypeError):
                HttpUri.join("http://abc.com/xyz", http_uri)
            with self.assertRaises(TypeError):
                GcsUri.join("gs://bucket/path", http_uri)

        with self.subTest("GcsUri suffix"):
            gcs_uri = GcsUri("gs://bucket/file.txt")
            with self.assertRaises(TypeError):
                LocalUri.join("/foo/bar", gcs_uri)
            with self.assertRaises(TypeError):
                HttpUri.join("http://abc.com/xyz", gcs_uri)
            with self.assertRaises(TypeError):
                GcsUri.join("gs://bucket/path", gcs_uri)

    def test_join_rejects_relative_path_with_uri_separator(self):
        with self.assertRaises(TypeError):
            LocalUri.join("/foo/bar", "folder://file.txt")
        with self.assertRaises(TypeError):
            Uri.join("foo", "folder://file.txt")

    def test_base_uri_join_only_allows_concrete_uri_as_first_token(self):
        relative_local_uri = LocalUri("file.txt")

        self.assertEqual(
            Uri.join(LocalUri("/foo/bar"), "file.txt"),
            Uri("/foo/bar/file.txt"),
        )
        self.assertEqual(
            LocalUri.join("/foo/bar", relative_local_uri),
            LocalUri("/foo/bar/file.txt"),
        )
        with self.assertRaises(TypeError):
            Uri.join("/foo/bar", relative_local_uri)

    def test_uri_constructors_reject_invalid_remote_paths(self):
        with self.assertRaises(TypeError):
            GcsUri("file.txt")
        with self.assertRaises(TypeError):
            HttpUri("file.txt")

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
    absltest.main()
