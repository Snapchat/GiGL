from gigl.src.common.utils.time import new_test_run_id
from gigl.src.validation_check.libs.name_checks import (
    check_if_kfp_pipeline_job_name_valid,
)
from tests.test_assets.test_case import TestCase


class NewTestRunIdTest(TestCase):
    def test_ids_differ_between_calls(self) -> None:
        self.assertNotEqual(new_test_run_id("my_test"), new_test_run_id("my_test"))

    def test_id_is_valid_kfp_job_name(self) -> None:
        check_if_kfp_pipeline_job_name_valid(new_test_run_id("enumerator_test"))
