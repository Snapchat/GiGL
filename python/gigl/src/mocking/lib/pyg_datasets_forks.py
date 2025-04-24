"""
Our mocking logic uses public datasets like Cora and DBLP from PyG.  PyG datasets are
downloaded from public sources which may not be available or rate-limit us.  We thus 
override the dataset classes to download the datasets from GCS buckets to avoid issues. 
"""

from torch_geometric.data import extract_zip
from torch_geometric.datasets import DBLP, Planetoid

import gigl.env.dep_constants as dep_constants
from gigl.common import GcsUri, LocalUri
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.utils.file_loader import FileLoader

unprocessed_datasets_gcs_uri = GcsUri(
    f"gs://{dep_constants.GIGL_PUBLIC_BUCKET_NAME}/unprocessed_datasets/"
)


class DBLPFromGCS(DBLP):
    # The file from https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=1 was copied to the below GCS path.
    url = GcsUri.join(unprocessed_datasets_gcs_uri, "pyg/dblp/DBLP_processed.zip")

    def download(self):
        file_loader = FileLoader(project=get_resource_config().project)
        local_uri = LocalUri.join(self.raw_dir, "DBLP.zip")
        file_loader.load_file(file_uri_src=self.url, file_uri_dst=local_uri)
        extract_zip(local_uri.uri, self.raw_dir)


class CoraFromGCS(Planetoid):
    # The files from https://github.com/kimiyoung/planetoid/tree/master/data were copied to the below GCS path.
    url = GcsUri.join(unprocessed_datasets_gcs_uri, "pyg/planetoid/")

    def download(self):
        assert self.name == "Cora", "Only Cora dataset is supported"
        file_loader = FileLoader(project=get_resource_config().project)
        file_uri_srcs = [GcsUri.join(self.url, name) for name in self.raw_file_names]
        file_uri_dsts = [
            LocalUri.join(self.raw_dir, name) for name in self.raw_file_names
        ]
        file_loader.load_files(
            source_to_dest_file_uri_map=dict(zip(file_uri_srcs, file_uri_dsts))
        )
