import yaml

from gigl.src.common.constants.ci import cloud_build_tag_for_github_run
from gigl.src.common.constants.local_fs import get_project_root_directory
from tests.test_assets.test_case import TestCase


class CiConstantsTest(TestCase):
    def test_cloud_build_tag_embeds_the_run_id(self) -> None:
        self.assertEqual(cloud_build_tag_for_github_run("123"), "gigl-ci-run-123")

    def test_cloud_build_config_tag_matches_the_helper(self) -> None:
        config_path = (
            get_project_root_directory()
            / ".github"
            / "cloud_builder"
            / "run_command_on_active_checkout.yaml"
        )
        with open(config_path) as config_file:
            config = yaml.safe_load(config_file)

        self.assertEqual(
            config["tags"], [cloud_build_tag_for_github_run("${_GITHUB_RUN_ID}")]
        )
        self.assertIn("_GITHUB_RUN_ID", config["substitutions"])
