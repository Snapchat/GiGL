import os
import tempfile
import textwrap
import unittest
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, patch

from gigl.common import LocalUri
from gigl.common.omegaconf_resolvers import register_resolvers
from gigl.common.utils.yaml_loader import load_resolved_yaml


@dataclass
class _SubConfig:
    name: str
    value: int
    enabled: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class _Complex_TestConfig:
    basic_config: _SubConfig
    description: str


class YamlLoaderTest(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        register_resolvers()
        self.temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        )

    def tearDown(self):
        self.temp_file.close()
        os.remove(self.temp_file.name)
        super().tearDown()

    def test_load_resolved_yaml_simple_config(self):
        """Test loading a simple YAML configuration."""

        contents = textwrap.dedent(
            """
        basic_config:
            name: "experiment_${now:%Y%m%d}"
            value: 42
            enabled: true
            tags:
                - "tag_${git_hash:}"
                - "${basic_config.value}" # resolves to 42
        description: "This is a test description"
        """
        )
        with self.temp_file:
            self.temp_file.write(contents)

        patch_commit_hash = "1234567890"
        patch_datetime = datetime(2023, 12, 15, 14, 30, 22)

        expected_result = _Complex_TestConfig(
            basic_config=_SubConfig(
                name=f"experiment_{patch_datetime.strftime('%Y%m%d')}",
                value=42,
                enabled=True,
                tags=[f"tag_{patch_commit_hash}", "42"],
            ),
            description="This is a test description",
        )
        with patch(
            "gigl.common.omegaconf_resolvers.subprocess.run"
        ) as mock_subprocess_run, patch(
            "gigl.common.omegaconf_resolvers.datetime"
        ) as mock_datetime:
            mock_result = MagicMock()
            mock_result.stdout = patch_commit_hash
            mock_subprocess_run.return_value = mock_result
            mock_datetime.now.return_value = patch_datetime

            uri = LocalUri(self.temp_file.name)
            result: _Complex_TestConfig = load_resolved_yaml(uri, _Complex_TestConfig)

        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
