import unittest
from typing import Optional, Type

from parameterized import param, parameterized

from gigl.common import Uri, UriFactory
from gigl.src.validation_check.config_validator import kfp_validation_checks

OFFLINE_SUBGRAPH_SAMPLING_RESOURCE_CONFIG_URI = UriFactory.create_uri(
    "deployment/configs/e2e_cicd_resource_config.yaml"
)
LIVE_SUBGRAPH_SAMPLING_RESOURCE_CONFIG_URI = UriFactory.create_uri(
    "deployment/configs/e2e_glt_resource_config.yaml"
)

OFFLINE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI = UriFactory.create_uri(
    "gigl/src/mocking/configs/e2e_node_anchor_based_link_prediction_template_gbml_config.yaml"
)
LIVE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI = UriFactory.create_uri(
    "examples/link_prediction/configs/e2e_hom_cora_sup_task_config.yaml"
)


class TestResourceConfigValidation(unittest.TestCase):
    """Test suite for resource config validation with different backends (live subgraph sampling, offline subgraph sampling)."""

    @parameterized.expand(
        [
            param(
                "Test that live subgraph sampling resource config with live subgraph sampling enabled passes",
                resource_config_uri=LIVE_SUBGRAPH_SAMPLING_RESOURCE_CONFIG_URI,
                task_config_uri=LIVE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI,
                expected_exception=None,
            ),
            param(
                "Test that live subgraph sampling resource config with offline subgraph sampling enabled fails",
                resource_config_uri=LIVE_SUBGRAPH_SAMPLING_RESOURCE_CONFIG_URI,
                task_config_uri=OFFLINE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI,
                expected_exception=AssertionError,
            ),
            param(
                "Test that offline subgraph sampling resource config with offline subgraph sampling enabled passes",
                resource_config_uri=OFFLINE_SUBGRAPH_SAMPLING_RESOURCE_CONFIG_URI,
                task_config_uri=OFFLINE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI,
                expected_exception=None,
            ),
            param(
                "Test that offline subgraph sampling resource config with live subgraph sampling enabled passes",
                resource_config_uri=OFFLINE_SUBGRAPH_SAMPLING_RESOURCE_CONFIG_URI,
                task_config_uri=LIVE_SUBGRAPH_SAMPLING_TASK_CONFIG_URI,
                expected_exception=None,
            ),
        ]
    )
    def test_resource_config_validation_with_real_configs(
        self,
        _,
        resource_config_uri: Uri,
        task_config_uri: Uri,
        expected_exception: Optional[Type[Exception]] = None,
    ) -> None:
        # Act & Assert
        if expected_exception is None:
            kfp_validation_checks(
                job_name="resource_config_validation_test",
                task_config_uri=task_config_uri,
                start_at="config_populator",
                resource_config_uri=resource_config_uri,
            )
        else:
            with self.assertRaises(expected_exception):
                kfp_validation_checks(
                    job_name="resource_config_validation_test",
                    task_config_uri=task_config_uri,
                    start_at="config_populator",
                    resource_config_uri=resource_config_uri,
                )


if __name__ == "__main__":
    unittest.main()
