import unittest

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.util import assert_that, equal_to

from gigl.src.common.types import AppliedTaskIdentifier
from gigl.src.common.utils.dataflow import init_beam_pipeline_options


class _ComputeLengthDoFn(beam.DoFn):
    def process(self, element):
        return [len(element)]


class _PrintDoFn(beam.DoFn):
    def process(self, element):
        print(element)
        return [element]


class DataflowUtilsTest(unittest.TestCase):
    def test_can_create_pipeline_config(self):
        NUM_WORKERS = 1
        MAX_NUM_WORKERS = 32
        MACHINE_TYPE = "n2-standard-48"
        DISK_SIZE_GB = 150

        options: PipelineOptions = init_beam_pipeline_options(
            applied_task_identifier=AppliedTaskIdentifier("test-applied-task"),
            job_name_suffix="test-job",
            num_workers=NUM_WORKERS,
            max_num_workers=MAX_NUM_WORKERS,
            machine_type=MACHINE_TYPE,
            disk_size_gb=DISK_SIZE_GB,
        )

        # Ensure pipeline runs
        with beam.Pipeline(options=options) as pipeline:
            lines = pipeline | beam.Create(
                [
                    "123",
                    "12345",
                    "12   6  10",
                    "",
                ]
            )
            lengths = lines | beam.ParDo(_ComputeLengthDoFn())

            assert_that(lengths, equal_to([3, 5, 10, 0]))

        # Ensure the pipeline options were propogated through
        parsed_options = options.get_all_options()
        self.assertEquals(parsed_options["num_workers"], NUM_WORKERS)
        self.assertEquals(parsed_options["max_num_workers"], MAX_NUM_WORKERS)
        self.assertEquals(parsed_options["machine_type"], MACHINE_TYPE)
        self.assertEquals(parsed_options["disk_size_gb"], DISK_SIZE_GB)
