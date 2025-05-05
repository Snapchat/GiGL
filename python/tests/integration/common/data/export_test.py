import unittest
import uuid

import torch

from gigl.common import GcsUri
from gigl.common.data.export import EmbeddingExporter, load_embeddings_to_bigquery
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.bq import BqUtils

logger = Logger()


class EmbeddingExportIntergrationTest(unittest.TestCase):
    def setUp(self):
        resource_config = get_resource_config()
        test_unique_name = f"GiGL-Intergration-Exporter-{uuid.uuid4().hex}"
        self.embedding_output_dir = GcsUri.join(
            resource_config.temp_assets_regional_bucket_path,
            test_unique_name,
            "embeddings",
        )
        self.embedding_output_bq_project = resource_config.project
        self.embedding_output_bq_dataset = resource_config.temp_assets_bq_dataset_name
        self.embedding_output_bq_table = test_unique_name

    def tearDown(self):
        gcs_utils = GcsUtils()
        gcs_utils.delete_files_in_bucket_dir(self.embedding_output_dir)
        bq_client = BqUtils()
        bq_export_table_path = bq_client.join_path(
            self.embedding_output_bq_project,
            self.embedding_output_bq_dataset,
            self.embedding_output_bq_table,
        )
        bq_client.delete_bq_table_if_exist(
            bq_table_path=bq_export_table_path,
        )

    def test_embedding_export(self):
        num_nodes = 1_000
        with EmbeddingExporter(export_dir=self.embedding_output_dir) as exporter:
            for i in torch.arange(num_nodes):
                exporter.add_embedding(
                    torch.tensor([i]), torch.ones(128, 1) * i, "node"
                )

        # We also want nested directories to be picked up.
        # e.g. if we have:
        # gs://MY BUCKET/embeddings/shard_0000.avro
        # gs://MY BUCKET/embeddings/nested/shard_0000.avro
        # The files under "nested" should be included.
        with EmbeddingExporter(
            export_dir=GcsUri.join(self.embedding_output_dir, "nested")
        ) as exporter:
            for i in torch.arange(num_nodes, num_nodes * 2):
                exporter.add_embedding(
                    torch.tensor([i]), torch.ones(128, 1) * i, "node"
                )
        bq_client = BqUtils()
        bq_export_table_path = bq_client.join_path(
            self.embedding_output_bq_project,
            self.embedding_output_bq_dataset,
            self.embedding_output_bq_table,
        )
        logger.info(
            f"Will try exporting to {self.embedding_output_dir} to BQ: {bq_export_table_path}"
        )
        load_job = load_embeddings_to_bigquery(
            gcs_folder=self.embedding_output_dir,
            project_id=self.embedding_output_bq_project,
            dataset_id=self.embedding_output_bq_dataset,
            table_id=self.embedding_output_bq_table,
        )

        # Check that data in BQ is as expected...
        self.assertEqual(
            load_job.output_rows,
            num_nodes * 2,
        )
        self.assertEqual(
            bq_client.count_number_of_rows_in_bq_table(bq_export_table_path),
            num_nodes * 2,
        )
