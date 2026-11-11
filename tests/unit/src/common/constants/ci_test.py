from gigl.src.common.constants.ci import cloud_build_tag_for_github_run
from tests.test_assets.test_case import TestCase


class CiConstantsTest(TestCase):
    def test_cloud_build_tag_embeds_the_run_id(self) -> None:
        self.assertEqual(cloud_build_tag_for_github_run("123"), "gigl-ci-run-123")
