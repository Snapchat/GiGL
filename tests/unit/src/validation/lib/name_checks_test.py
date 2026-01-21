import unittest

from parameterized import param, parameterized

from gigl.src.validation_check.libs.name_checks import (
    check_if_kfp_pipeline_job_name_valid,
)


class TestStringChecks(unittest.TestCase):
    @parameterized.expand(
        [
            param("valid_job_name", "valid_job_name"),
            param("valid_job_name_with_numbers", "valid_job_name_123"),
        ]
    )
    def test_valid_job_names(self, name, job_name):
        try:
            check_if_kfp_pipeline_job_name_valid(job_name)
        except ValueError:
            self.fail(
                f"check_if_kfp_pipeline_job_name_valid raised ValueError unexpectedly for {job_name}"
            )

    @parameterized.expand(
        [
            param("empty_string", ""),
            param("starts_with_number", "1invalid-job-name"),
            param("contains_uppercase", "InvalidJobName"),
            param("contains_special_characters", "invalid@job#name"),
            param("too_long", "a" * 52),
            param("ends_with_dash", "invalid-job-name-"),
            param("ends_with_underscore", "invalid_job_name_"),
        ]
    )
    def test_invalid_job_names(self, name: str, job_name: str):
        with self.assertRaises(ValueError):
            check_if_kfp_pipeline_job_name_valid(job_name)


if __name__ == "__main__":
    unittest.main()
