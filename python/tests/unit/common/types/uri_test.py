import unittest
from pathlib import Path

from parameterized import param, parameterized

from gigl.common.types.uri.gcs_uri import GcsUri
from gigl.common.types.uri.http_uri import HttpUri
from gigl.common.types.uri.local_uri import LocalUri
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

    @parameterized.expand(
        [
            (LocalUri("/foo/bar"), "baz", LocalUri("/foo/bar/baz")),
            (HttpUri("http://abc.com/xyz"), "foo", HttpUri("http://abc.com/xyz/foo")),
            (GcsUri("gs://bucket/"), "file.txt", GcsUri("gs://bucket/file.txt")),
            (LocalUri("/foo/bar"), Path("file.text"), LocalUri("/foo/bar/file.text")),
        ]
    )
    def test_join(self, base, join_with, expected):
        joined = base.join(base, join_with)
        self.assertIsInstance(joined, type(base))
        self.assertEqual(expected, joined)

    @parameterized.expand(
        [
            (LocalUri("/foo/bar"), "baz", LocalUri("/foo/bar/baz")),
            (HttpUri("http://abc.com/xyz"), "foo", HttpUri("http://abc.com/xyz/foo")),
            (GcsUri("gs://bucket/"), "file.txt", GcsUri("gs://bucket/file.txt")),
            (LocalUri("/foo/bar"), Path("file.text"), LocalUri("/foo/bar/file.text")),
        ]
    )
    def test_div_join(self, base, join_with, expected):
        joined = base / join_with
        self.assertIsInstance(joined, type(base))
        self.assertEqual(expected, joined)

    @parameterized.expand(
        [
            param(
                "Local to",
                base=LocalUri("/foo/bar"),
                other=[HttpUri("http://abc.com/xyz"), GcsUri("gs://bucket/path/to")],
            ),
            param(
                "Http to",
                base=HttpUri("http://abc.com/xyz"),
                other=[LocalUri("/foo/bar"), GcsUri("gs://bucket/path/to")],
            ),
            param(
                "GCS to",
                base=GcsUri("gs://bucket/path/to"),
                other=[LocalUri("/foo/bar"), HttpUri("http://abc.com/xyz")],
            ),
        ]
    )
    def test_div_join_invalid_type(self, _, base, other):
        for o in other:
            with self.subTest(f"Try {type(base)} / {type(o)}"):
                with self.assertRaises(TypeError):
                    base / o


if __name__ == "__main__":
    unittest.main()
