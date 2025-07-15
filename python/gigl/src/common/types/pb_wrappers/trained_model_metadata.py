from dataclasses import dataclass
from typing import List

from gigl.common import Uri, UriFactory
from snapchat.research.gbml import trained_model_metadata_pb2


@dataclass
class TrainedModelMetadataPbWrapper:
    trained_model_metadata_pb: trained_model_metadata_pb2.TrainedModelMetadata

    def get_output_paths(self) -> List[Uri]:
        paths = [
            self.trained_model_metadata_pb.trained_model_uri,
            self.trained_model_metadata_pb.scripted_model_uri,
            self.trained_model_metadata_pb.eval_metrics_uri,
            self.trained_model_metadata_pb.tensorboard_logs_uri,
        ]
        uri_paths = [UriFactory.create_uri(uri=path) for path in paths if path]
        return uri_paths
