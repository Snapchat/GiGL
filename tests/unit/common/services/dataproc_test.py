from unittest.mock import patch

from gigl.common.services.dataproc import DataprocService
from tests.test_assets.test_case import TestCase


class DeleteClusterIfExistsTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        for client in ("ClusterControllerClient", "JobControllerClient"):
            patcher = patch(f"gigl.common.services.dataproc.dataproc_v1.{client}")
            patcher.start()
            self.addCleanup(patcher.stop)
        self.service = DataprocService(project_id="p", region="us-central1")

    def test_deletes_when_present(self) -> None:
        with (
            patch.object(self.service, "does_cluster_exist", return_value=True),
            patch.object(self.service, "delete_cluster") as delete_cluster,
        ):
            deleted = self.service.delete_cluster_if_exists("sgs-a")

        self.assertTrue(deleted)
        delete_cluster.assert_called_once_with("sgs-a")

    def test_skips_when_absent(self) -> None:
        with (
            patch.object(self.service, "does_cluster_exist", return_value=False),
            patch.object(self.service, "delete_cluster") as delete_cluster,
        ):
            deleted = self.service.delete_cluster_if_exists("sgs-a")

        self.assertFalse(deleted)
        delete_cluster.assert_not_called()
