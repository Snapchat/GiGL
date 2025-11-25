import unittest

from parameterized import param, parameterized

from gigl.common.services.vertex_ai import VertexAiJobConfig


class VertexAIServiceTest(unittest.TestCase):
    @parameterized.expand(
        [
            param(
                "g4 machine ; should default to hyperdisk-balanced",
                machine_type="g4-standard-8",
                expected_boot_disk_type="hyperdisk-balanced",
            ),
            param(
                "n1 machine ; should default to pd-ssd",
                machine_type="n1-standard-4",
                expected_boot_disk_type="pd-ssd",
            ),
        ]
    )
    def test_default_boot_disk_for_machine(
        self, _, machine_type, expected_boot_disk_type
    ):
        job_config = VertexAiJobConfig(
            job_name="job_name",
            container_uri="container_uri",
            command=["command"],
            machine_type=machine_type,
        )
        self.assertEqual(job_config.boot_disk_type, expected_boot_disk_type)


if __name__ == "__main__":
    unittest.main()
