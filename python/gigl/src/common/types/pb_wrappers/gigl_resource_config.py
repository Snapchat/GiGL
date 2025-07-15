import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import gigl.src.common.constants.resource_config as resource_config_constants
from gigl.common import GcsUri, UriFactory
from gigl.common.logger import Logger
from gigl.src.common.constants.components import GiGLComponents
from snapchat.research.gbml.gigl_resource_config_pb2 import (
    DataflowResourceConfig,
    DataPreprocessorConfig,
    DistributedTrainerConfig,
    GiglResourceConfig,
    KFPResourceConfig,
    LocalResourceConfig,
    SharedResourceConfig,
    SparkResourceConfig,
    TrainerResourceConfig,
    VertexAiResourceConfig,
)

logger = Logger()

COMPONENT_TO_SHORTENED_COST_LABEL_MAP = {
    GiGLComponents.DataPreprocessor: "pre",
    GiGLComponents.SubgraphSampler: "sgs",
    GiGLComponents.SplitGenerator: "spl",
    GiGLComponents.Trainer: "tra",
    GiGLComponents.Inferencer: "inf",
    GiGLComponents.PostProcessor: "pos",
}

_TRAINER_CONFIG_FIELD = "trainer_config"
_VERTEX_AI_TRAINER_CONFIG = "vertex_ai_trainer_config"
_KFP_TRAINER_CONFIG = "kfp_trainer_config"
_LOCAL_TRAINER_CONFIG = "local_trainer_config"

_INFERENCER_CONFIG_FIELD = "inferencer_config"
_VERTEX_AI_INFERENCER_CONFIG = "vertex_ai_inferencer_config"
_DATAFLOW_INFERENCER_CONFIG = "dataflow_inferencer_config"
_LOCAL_INFERENCER_CONFIG = "local_inferencer_config"


@dataclass
class GiglResourceConfigWrapper:
    resource_config: GiglResourceConfig
    _loaded_shared_resource_config: Optional[SharedResourceConfig] = None
    _trainer_config: Optional[
        Union[VertexAiResourceConfig, KFPResourceConfig, LocalResourceConfig]
    ] = None
    _inference_config: Optional[
        Union[DataflowResourceConfig, VertexAiResourceConfig, LocalResourceConfig]
    ] = None

    _split_gen_config: Union[
        SparkResourceConfig, DataflowResourceConfig
    ] = None  # type: ignore

    @property
    def shared_resource_config(self) -> SharedResourceConfig:
        if self._loaded_shared_resource_config is None:
            field = self.resource_config.WhichOneof("shared_resource")

            if field is None:
                raise ValueError(
                    "A SharedResourceConfig or a SharedResourceConfig uri must be passed in."
                )

            oneof_field_value = getattr(self.resource_config, field)

            if field == "shared_resource_config":
                self._loaded_shared_resource_config = oneof_field_value
            elif field == "shared_resource_config_uri":
                self._loaded_shared_resource_config = (
                    self._load_shared_resource_config_from_uri(oneof_field_value)
                )
            else:
                raise ValueError(
                    "GiglResourceConfig must contain either a SharedResourceConfig or a SharedResourceConfig URI."
                )

        return self._loaded_shared_resource_config

    @property
    def get_resource_config_uri(self) -> str:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--resource_config_uri",
            type=str,
            required=False,
        )
        args, _ = parser.parse_known_args()

        resource_config_path = args.resource_config_uri or os.getenv(
            resource_config_constants.RESOURCE_CONFIG_OS_ENV
        )

        return str(resource_config_path)

    @staticmethod
    def _load_shared_resource_config_from_uri(uri: str):
        uri_object = UriFactory.create_uri(uri=uri)
        from gigl.common.utils.proto_utils import ProtoUtils

        proto_utils = ProtoUtils()
        shared_resource_config = proto_utils.read_proto_from_yaml(
            uri_object, proto_cls=SharedResourceConfig
        )

        return shared_resource_config

    def get_resource_labels(
        self,
        component: Optional[GiGLComponents] = None,
        replacement_key: str = "COMPONENT",
    ) -> Dict[str, str]:
        """
        Returns a dictionary of resource labels that can be used to tag resources in GCP.
        Users may also provide a custom suffix to replace in the resource labels which defaults to "COMPONENT".
        For example: If the resource labels are {"key": "value_COMPONENT"}, and the component is "DataPreprocessor",
        the returned resource labels will be {"key": "value_pre"}.

        Args:
            component (Optional[GiGLComponents]): The component to replace in the resource labels.
            replacement_key (str): The key to replace in the resource labels.

        Returns:
            Dict[str, str]: The resource labels with the component replaced by the shortened cost label.
        """
        labels = dict(self.shared_resource_config.resource_labels)

        def component_mapper(comp: Optional[GiGLComponents]) -> str:
            return (
                COMPONENT_TO_SHORTENED_COST_LABEL_MAP.get(comp, "na") if comp else "na"
            )

        replace_value = component_mapper(component)

        return {
            key: (
                value.replace(replacement_key, replace_value)
                if replacement_key in value
                else value
            )
            for key, value in labels.items()
        }

    def get_resource_labels_formatted_for_dataflow(
        self,
        component: Optional[GiGLComponents] = None,
    ) -> List[str]:
        labels: List[str] = []
        for key, val in self.get_resource_labels(component=component).items():
            labels.append(f"{key}={val}")
        return labels

    @property
    def project(self) -> str:
        """
        Returns the Cloud project name specified in the resource config.
        """
        return self.shared_resource_config.common_compute_config.project

    @property
    def service_account_email(self) -> str:
        """
        Returns the service account email specified in the resource config.
        """
        return (
            self.shared_resource_config.common_compute_config.gcp_service_account_email
        )

    @property
    def temp_assets_bucket_path(self) -> GcsUri:
        """
        Returns the GCS URI for the temporary assets bucket specified in the resource config.
        """
        return GcsUri(
            self.shared_resource_config.common_compute_config.temp_assets_bucket
        )

    @property
    def temp_assets_regional_bucket_path(self) -> GcsUri:
        """
        Returns the GCS URI for the regional temporary assets bucket specified in the resource config.
        """
        return GcsUri(
            self.shared_resource_config.common_compute_config.temp_regional_assets_bucket
        )

    @property
    def perm_assets_bucket_path(self) -> GcsUri:
        """
        Returns the GCS URI for the permanent assets bucket specified in the resource config.
        """
        return GcsUri(
            self.shared_resource_config.common_compute_config.perm_assets_bucket
        )

    @property
    def temp_assets_bq_dataset_name(self) -> str:
        """
        Returns the BigQuery dataset name for the temporary assets specified in the resource config.
        """
        return (
            self.shared_resource_config.common_compute_config.temp_assets_bq_dataset_name
        )

    @property
    def embedding_bq_dataset_name(self) -> str:
        """
        Returns the BigQuery dataset name for the embeddings table output by inferencer specified in the resource config.
        """
        return (
            self.shared_resource_config.common_compute_config.embedding_bq_dataset_name
        )

    @property
    def dataflow_runner(self) -> str:
        """
        Returns the Beam runner specified in the resource config. (e.g. Dataflow or DirectRunner)
        """
        return self.shared_resource_config.common_compute_config.dataflow_runner

    @property
    def region(self) -> str:
        """
        Returns the cloud region specified in the resource config. (e.g us-central1)
        """
        return self.shared_resource_config.common_compute_config.region

    @property
    def trainer_config(
        self,
    ) -> Union[VertexAiResourceConfig, KFPResourceConfig, LocalResourceConfig]:
        """
        Returns the trainer config specified in the resource config. (e.g. Vertex AI, KFP, Local)
        """

        if not self._trainer_config:
            # TODO: (svij) Marked for deprecation
            if self.resource_config.HasField("trainer_config"):
                logger.warning(
                    "Warning, GbmlConfig.trainer_config is deprecated. Please use trainer_resource_config instead."
                    + "Will try automatically casting trainer_config to trainer_resource_config."
                    + "The support for this casting may be removed without notice in the future."
                )

                deprecated_config: DistributedTrainerConfig = (
                    self.resource_config.trainer_config
                )
                _trainer_config: Union[
                    VertexAiResourceConfig, KFPResourceConfig, LocalResourceConfig
                ]
                if deprecated_config.WhichOneof(_TRAINER_CONFIG_FIELD) == _VERTEX_AI_TRAINER_CONFIG:  # type: ignore[arg-type]
                    logger.info(
                        f"Casting VertexAiTrainerConfig: ({deprecated_config.vertex_ai_trainer_config}) to VertexAiResourceConfig"
                    )
                    _trainer_config = VertexAiResourceConfig(
                        machine_type=deprecated_config.vertex_ai_trainer_config.machine_type,
                        gpu_type=deprecated_config.vertex_ai_trainer_config.gpu_type,
                        gpu_limit=deprecated_config.vertex_ai_trainer_config.gpu_limit,
                        num_replicas=deprecated_config.vertex_ai_trainer_config.num_replicas,
                    )
                elif deprecated_config.WhichOneof(_TRAINER_CONFIG_FIELD) == _KFP_TRAINER_CONFIG:  # type: ignore[arg-type]
                    logger.info(
                        f"Casting KFPTrainerConfig: ({deprecated_config.kfp_trainer_config}) to KFPResourceConfig"
                    )
                    _trainer_config = KFPResourceConfig(
                        cpu_request=deprecated_config.kfp_trainer_config.cpu_request,
                        memory_request=deprecated_config.kfp_trainer_config.memory_request,
                        gpu_type=deprecated_config.kfp_trainer_config.gpu_type,
                        gpu_limit=deprecated_config.kfp_trainer_config.gpu_limit,
                        num_replicas=deprecated_config.kfp_trainer_config.num_replicas,
                    )
                elif deprecated_config.WhichOneof(_TRAINER_CONFIG_FIELD) == _LOCAL_TRAINER_CONFIG:  # type: ignore[arg-type]
                    logger.info(
                        f"Casting LocalTrainerConfig: ({deprecated_config.local_trainer_config}) to LocalResourceConfig"
                    )
                    _trainer_config = LocalResourceConfig(
                        num_workers=deprecated_config.local_trainer_config.num_workers,
                    )
                else:
                    raise ValueError(
                        f"Invalid trainer_config type: {deprecated_config}"
                    )
            elif self.resource_config.HasField("trainer_resource_config"):
                config: TrainerResourceConfig = (
                    self.resource_config.trainer_resource_config
                )
                if config.WhichOneof(_TRAINER_CONFIG_FIELD) == _VERTEX_AI_TRAINER_CONFIG:  # type: ignore[arg-type]
                    _trainer_config = config.vertex_ai_trainer_config
                elif config.WhichOneof(_TRAINER_CONFIG_FIELD) == _KFP_TRAINER_CONFIG:  # type: ignore[arg-type]
                    _trainer_config = config.kfp_trainer_config
                elif config.WhichOneof(_TRAINER_CONFIG_FIELD) == _LOCAL_TRAINER_CONFIG:  # type: ignore[arg-type]
                    _trainer_config = config.local_trainer_config
                else:
                    raise ValueError(f"Invalid trainer_config type: {config}")
            else:
                raise ValueError(
                    f"Trainer config not found in resource config; neither trainer_config nor trainer_resource_config is set: {self.resource_config}"
                )
        return _trainer_config

    @property
    def inferencer_config(
        self,
    ) -> Union[DataflowResourceConfig, VertexAiResourceConfig, LocalResourceConfig]:
        """
        Returns the inferencer config specified in the resource config. (Dataflow)
        """
        if self._inference_config is None:
            # TODO: (svij) Marked for deprecation
            if self.resource_config.HasField("inferencer_config"):
                logger.warning(
                    "Warning, inferencer_config is deprecated. Please use inferencer_resource_config instead."
                    + "Will try automatically casting inferencer_config to inferencer_resource_config."
                    + "The support for this casting may be removed without notice in the future."
                )
                # self.resource_config.inferencer_config is a DataflowResourceConfig
                self._inference_config = self.resource_config.inferencer_config
            elif self.resource_config.HasField("inferencer_resource_config"):
                config = self.resource_config.inferencer_resource_config
                if config.WhichOneof(_INFERENCER_CONFIG_FIELD) == _DATAFLOW_INFERENCER_CONFIG:  # type: ignore[arg-type]
                    self._inference_config = config.dataflow_inferencer_config
                elif config.WhichOneof(_INFERENCER_CONFIG_FIELD) == _LOCAL_INFERENCER_CONFIG:  # type: ignore[arg-type]
                    self._inference_config = config.local_inferencer_config
                elif config.WhichOneof(_INFERENCER_CONFIG_FIELD) == _VERTEX_AI_INFERENCER_CONFIG:  # type: ignore[arg-type]
                    self._inference_config = config.vertex_ai_inferencer_config
                else:
                    raise ValueError("Invalid inferencer_config type")
            else:
                raise ValueError(
                    "Inferencer config not found in resource config; neither inferencer_config nor inferencer_resource_config is set."
                )

        return self._inference_config

    @property
    def preprocessor_config(self) -> DataPreprocessorConfig:
        """
        Returns the preprocessor config specified in the resource config. (Dataflow)
        """
        return self.resource_config.preprocessor_config

    @property
    def subgraph_sampler_config(self) -> SparkResourceConfig:
        """
        Returns the subgraph sampler config specified in the resource config. (Spark)
        """
        return self.resource_config.subgraph_sampler_config

    @property
    def split_generator_config(self) -> SparkResourceConfig:
        """
        Returns the split generator config specified in the resource config. (Spark)
        """
        return self.resource_config.split_generator_config
