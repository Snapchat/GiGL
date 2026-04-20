import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from gigl.analytics.data_analyzer.data_analyzer import _write_report
from tests.test_assets.test_case import TestCase

HTML = "<html><body>report</body></html>"


class WriteReportLocalTest(TestCase):
    def test_writes_to_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_report(HTML, tmpdir)
            report = Path(path)
            self.assertTrue(report.exists())
            self.assertEqual(report.read_text(), HTML)
            self.assertEqual(report.name, "report.html")

    def test_creates_missing_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "nested" / "path"
            path = _write_report(HTML, str(nested))
            self.assertTrue(Path(path).exists())


@patch("gigl.analytics.data_analyzer.data_analyzer.GcsUtils")
class WriteReportGcsTest(TestCase):
    def test_uploads_to_gcs(self, mock_gcs_cls: MagicMock) -> None:
        path = _write_report(HTML, "gs://my-bucket/output/")
        self.assertEqual(path, "gs://my-bucket/output/report.html")
        mock_gcs_cls.return_value.upload_from_string.assert_called_once()

    def test_handles_trailing_slash(self, mock_gcs_cls: MagicMock) -> None:
        path_with = _write_report(HTML, "gs://my-bucket/output/")
        path_without = _write_report(HTML, "gs://my-bucket/output")
        self.assertEqual(path_with, path_without)
