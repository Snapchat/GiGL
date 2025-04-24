from __future__ import annotations

from dataclasses import dataclass, field
from distutils.util import strtobool
from typing import Optional

from gigl.common import Uri, UriFactory
from gigl.common.logger import Logger
from gigl.common.utils.proto_utils import ProtoUtils
from gigl.src.common.types.pb_wrappers.dataset_metadata import DatasetMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.flattened_graph_metadata import (
    FlattenedGraphMetadataPbWrapper,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.common.types.pb_wrappers.subgraph_sampling_strategy import (
    SubgraphSamplingStrategyPbWrapper,
)
from gigl.src.common.types.pb_wrappers.task_metadata import TaskMetadataPbWrapper
from gigl.src.common.types.pb_wrappers.trained_model_metadata import (
    TrainedModelMetadataPbWrapper,
)
from gigl.src.common.utils.file_loader import FileLoader
from snapchat.research.gbml import (
    dataset_metadata_pb2,
    flattened_graph_metadata_pb2,
    gbml_config_pb2,
    graph_schema_pb2,
    preprocessed_metadata_pb2,
    subgraph_sampling_strategy_pb2,
    trained_model_metadata_pb2,
)

logger = Logger()


@dataclass(frozen=True)
class GbmlConfigPbWrapper:
    """
    Wrapper class for GbmlConfig proto, also includes individual wrappers for the dataset_metadata_pb,
    graph_metadata_pb, preprocessed_metadata_pb, flattened_graph_metadata_pb, task_metadata_pb, and
    trained_model_metadata_pb.
    """

    gbml_config_pb: gbml_config_pb2.GbmlConfig

    _dataset_metadata_pb_wrapper: DatasetMetadataPbWrapper = field(init=False)
    _graph_metadata_pb_wrapper: GraphMetadataPbWrapper = field(init=False)
    _preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper = field(init=False)
    _flattened_graph_metadata_pb_wrapper: FlattenedGraphMetadataPbWrapper = field(
        init=False
    )
    _task_metadata_pb_wrapper: TaskMetadataPbWrapper = field(init=False)
    _trained_model_metadata_pb_wrapper: TrainedModelMetadataPbWrapper = field(
        init=False
    )
    _subgraph_sampling_strategy_pb_wrapper: SubgraphSamplingStrategyPbWrapper = field(
        init=False
    )

    def __post_init__(self):
        # Populate the _preprocessed_metadata_pb_wrapper field
        self.__load_preprocessed_metadata_pb_wrapper(
            uri=self.gbml_config_pb.shared_config.preprocessed_metadata_uri
        )
        # Populate the _dataset_metadata_pb_wrapper field
        self.__load_dataset_metadata_pb_wrapper(
            dataset_metadata_pb=self.gbml_config_pb.shared_config.dataset_metadata
        )
        # Populate the _graph_metadata_pb_wrapper field
        self.__load_graph_metadata_pb_wrapper(
            graph_metadata_pb=self.gbml_config_pb.graph_metadata
        )
        # Populate the _flattened_graph_metadata_pb_wrapper field
        self.__load_flattened_graph_metadata_pb_wrapper(
            flattened_graph_metadata_pb=self.gbml_config_pb.shared_config.flattened_graph_metadata
        )
        # Populate the _task_metadata_pb_wrapper field
        self.__load_task_metadata_pb_wrapper(
            task_metadata_pb=self.gbml_config_pb.task_metadata
        )
        # Populate the _trained_model_metadata_pb_wrapper field
        self.__load_trained_model_metadata_pb_wrapper(
            trained_model_metadata_pb=self.gbml_config_pb.shared_config.trained_model_metadata
        )

        # Populate the _subgraph_sampling_strategy_pb_wrapper field
        self.__load_subgraph_sampling_strategy_pb_wrapper(
            subgraph_sampling_strategy_pb=self.gbml_config_pb.dataset_config.subgraph_sampler_config.subgraph_sampling_strategy
        )

    def __load_preprocessed_metadata_pb_wrapper(self, uri: str) -> None:
        """
        Load preprocessed_metadata_pb when given a uri

        Args:
            uri (str): The path to preprocessed_metadata.yaml file.
        """
        preprocessed_metadata_pb_wrapper: Optional[PreprocessedMetadataPbWrapper] = None
        if not uri:
            logger.warning(
                "preprocessedMetadataUri is not set in the GbmlConfig. "
                "Please use ConfigPopulator to populate the preprocessedMetadata or designate it yourself."
            )
            return

        # Check the existence of preprocessed_metadata.yaml file, it's expected to not exist in Data Preprocessor
        # and expected to exist for all other components.
        file_loader = FileLoader()
        preprocessed_metadata_file_exists = file_loader.does_uri_exist(uri=uri)
        if not preprocessed_metadata_file_exists:
            logger.warning(
                f"preprocessedMetadataUri: {uri} does not exist. "
                "Something is wrong if you are seeing this warning outside data_preprocessor"
            )
        else:
            proto_utils = ProtoUtils()
            preprocessed_metadata_pb = proto_utils.read_proto_from_yaml(
                uri=UriFactory.create_uri(uri=uri),
                proto_cls=preprocessed_metadata_pb2.PreprocessedMetadata,
            )
            preprocessed_metadata_pb_wrapper = PreprocessedMetadataPbWrapper(
                preprocessed_metadata_pb=preprocessed_metadata_pb
            )

            object.__setattr__(
                self,
                "_preprocessed_metadata_pb_wrapper",
                preprocessed_metadata_pb_wrapper,
            )

    def __load_dataset_metadata_pb_wrapper(
        self, dataset_metadata_pb: dataset_metadata_pb2.DatasetMetadata
    ) -> None:
        """
        Load dataset_metadata_pb_wrapper with dataset_metadata_pb

        Args:
            dataset_metadata_pb (dataset_metadata_pb2.DatasetMetadata): The dataset metadata proto
        """
        dataset_metadata_pb_wrapper: DatasetMetadataPbWrapper = (
            DatasetMetadataPbWrapper(dataset_metadata_pb=dataset_metadata_pb)
        )
        object.__setattr__(
            self, "_dataset_metadata_pb_wrapper", dataset_metadata_pb_wrapper
        )

    def __load_graph_metadata_pb_wrapper(
        self, graph_metadata_pb: graph_schema_pb2.GraphMetadata
    ) -> None:
        """
        Load graph_metadata_pb_wrapper with graph_metadata_pb when condensed_node_type_map and
        condensed_edge_type_map exist in it

        Args:
            graph_metadata_pb (graph_schema_pb2.GraphMetadata): The graph metadata proto
        """
        if (
            graph_metadata_pb.condensed_edge_type_map
            and graph_metadata_pb.condensed_node_type_map
        ):
            graph_metadata_pb_wrapper: GraphMetadataPbWrapper = GraphMetadataPbWrapper(
                graph_metadata_pb=graph_metadata_pb
            )
            object.__setattr__(
                self, "_graph_metadata_pb_wrapper", graph_metadata_pb_wrapper
            )

    def __load_flattened_graph_metadata_pb_wrapper(
        self,
        flattened_graph_metadata_pb: flattened_graph_metadata_pb2.FlattenedGraphMetadata,
    ) -> None:
        """
        Load flattened_graph_metadata_pb_wrapper with flattened_graph_metadata_pb

        Args:
            flattened_graph_metadata_pb (flattened_graph_metadata_pb2.FlattenedGraphMetadata): The flattened graph metadata proto
        """
        flattened_graph_metadata_pb_wrapper: FlattenedGraphMetadataPbWrapper = (
            FlattenedGraphMetadataPbWrapper(
                flattened_graph_metadata_pb=flattened_graph_metadata_pb
            )
        )
        object.__setattr__(
            self,
            "_flattened_graph_metadata_pb_wrapper",
            flattened_graph_metadata_pb_wrapper,
        )

    def __load_task_metadata_pb_wrapper(
        self, task_metadata_pb: gbml_config_pb2.GbmlConfig.TaskMetadata
    ) -> None:
        """
        Load task_metadata_pb_wrapper with task_metadata_pb

        Args:
            task_metadata_pb (gbml_config_pb2.GbmlConfig.TaskMetadata): The task metadata proto
        """
        task_metadata_pb_wrapper: TaskMetadataPbWrapper = TaskMetadataPbWrapper(
            task_metadata_pb=task_metadata_pb
        )
        object.__setattr__(self, "_task_metadata_pb_wrapper", task_metadata_pb_wrapper)

    def __load_trained_model_metadata_pb_wrapper(
        self, trained_model_metadata_pb: trained_model_metadata_pb2.TrainedModelMetadata
    ) -> None:
        """
        Load trained_model_metadata_pb_wrapper with trained_model_metadata_pb

        Args:
            trained_model_metadata_pb (trained_model_metadata_pb.TrainedModelMetadata): The trained model metadata proto
        """
        trained_model_metadata_pb_wrapper: TrainedModelMetadataPbWrapper = (
            TrainedModelMetadataPbWrapper(
                trained_model_metadata_pb=trained_model_metadata_pb
            )
        )
        object.__setattr__(
            self,
            "_trained_model_metadata_pb_wrapper",
            trained_model_metadata_pb_wrapper,
        )

    def __load_subgraph_sampling_strategy_pb_wrapper(
        self,
        subgraph_sampling_strategy_pb: subgraph_sampling_strategy_pb2.SubgraphSamplingStrategy,
    ) -> None:
        """
        Load subgraph_sampling_strategy_pb_wrapper with subgraph_sampling_strategy_pb

        Args:
            subgraph_sampling_strategy_pb (subgraph_sampling_strategy_pb2.SubgraphSamplingStrategy): The subgraph sampling strategy proto
        """
        if subgraph_sampling_strategy_pb.WhichOneof("strategy") is not None:
            subgraph_sampling_strategy_pb_wrapper: SubgraphSamplingStrategyPbWrapper = (
                SubgraphSamplingStrategyPbWrapper(
                    subgraph_sampling_strategy_pb=subgraph_sampling_strategy_pb
                )
            )
            object.__setattr__(
                self,
                "_subgraph_sampling_strategy_pb_wrapper",
                subgraph_sampling_strategy_pb_wrapper,
            )
        else:
            object.__setattr__(
                self,
                "_subgraph_sampling_strategy_pb_wrapper",
                None,
            )

    @classmethod
    def get_gbml_config_pb_wrapper_from_uri(
        cls,
        gbml_config_uri: Uri,
    ) -> GbmlConfigPbWrapper:
        """
        Build and return a GbmlConfigPbWrapper from a uri of gbml_config

        Args:
            gbml_config_uri (Uri): The uri of gbml_config
        Returns:
            GbmlConfigPbWrapper: The GbmlConfigPbWrapper
        """
        proto_utils = ProtoUtils()
        gbml_config_pb: gbml_config_pb2.GbmlConfig = proto_utils.read_proto_from_yaml(
            uri=gbml_config_uri, proto_cls=gbml_config_pb2.GbmlConfig
        )
        return GbmlConfigPbWrapper(gbml_config_pb=gbml_config_pb)

    @property
    def dataset_metadata_pb_wrapper(self) -> DatasetMetadataPbWrapper:
        """
        Allows access to a dataset_metadata_pb_wrapper

        Returns:
            DatasetMetadataPbWrapper: The dataset metadata pb wrapper
        """
        return self._dataset_metadata_pb_wrapper

    @property
    def graph_metadata_pb_wrapper(self) -> GraphMetadataPbWrapper:
        """
        Allows access to a graph_metadata_pb_wrapper

        Returns:
            GraphMetadataPbWrapper: The graph metadata pb wrapper
        """
        return self._graph_metadata_pb_wrapper

    @property
    def subgraph_sampling_strategy_pb_wrapper(
        self,
    ) -> Optional[SubgraphSamplingStrategyPbWrapper]:
        """
        Allows access to a subgraph_sampling_strategy_pb_wrapper

        Returns:
            Optional[SubgraphSamplingStrategyPbWrapper]: The subgraph sampling strategy pb wrapper or none if it does not exist
        """
        return self._subgraph_sampling_strategy_pb_wrapper

    @property
    def task_metadata_pb_wrapper(self) -> TaskMetadataPbWrapper:
        """
        Allows access to a task_metadata_pb_wrapper

        Returns:
            TaskMetadataPbWrapper: The task metadata pb wrapper
        """
        return self._task_metadata_pb_wrapper

    @property
    def flattened_graph_metadata_pb_wrapper(self) -> FlattenedGraphMetadataPbWrapper:
        """
        Allows access to a flattened_graph_metadata_pb_wrapper

        Returns:
            FlattenedGraphMetadataPbWrapper: The flattened graph metadata pb wrapper
        """
        return self._flattened_graph_metadata_pb_wrapper

    @property
    def preprocessed_metadata_pb_wrapper(self) -> PreprocessedMetadataPbWrapper:
        """
        Allows access to a preprocessed_metadata_pb_wrapper

        Returns:
            PreprocessedMetadataPbWrapper: The preprocessed metadata pb wrapper
        """
        return self._preprocessed_metadata_pb_wrapper

    @property
    def trained_model_metadata_pb_wrapper(self) -> TrainedModelMetadataPbWrapper:
        """
        Allows access to a trained_model_metadata_pb_wrapper

        Returns:
            TrainedModelMetadataPbWrapper: The trained model metadata pb wrapper
        """
        return self._trained_model_metadata_pb_wrapper

    @property
    def graph_metadata(self) -> graph_schema_pb2.GraphMetadata:
        """
        Allows access to graph_metadata pb under GbmlConfig

        Returns:
            gbml_config_pb2.GraphMetadata: The graph metadata proto
        """
        return self.gbml_config_pb.graph_metadata

    @property
    def dataset_config(self) -> gbml_config_pb2.GbmlConfig.DatasetConfig:
        """
        Allows access to dataset_config pb under GbmlConfig

        Returns:
            gbml_config_pb2.DatasetConfig: The dataset config
        """
        return self.gbml_config_pb.dataset_config

    @property
    def inferencer_config(self) -> gbml_config_pb2.GbmlConfig.InferencerConfig:
        """
        Allows access to inferencer_config pb under GbmlConfig

        Returns:
            gbml_config_pb2.InferencerConfig: The inferencer config
        """
        return self.gbml_config_pb.inferencer_config

    @property
    def metrics_config(self) -> gbml_config_pb2.GbmlConfig.MetricsConfig:
        """
        Allows access to metrics_config pb under GbmlConfig

        Returns:
            gbml_config_pb2.MetricsConfig: The metrics config
        """
        return self.gbml_config_pb.metrics_config

    @property
    def profiler_config(self) -> gbml_config_pb2.GbmlConfig.ProfilerConfig:
        """
        Allows access to profiler_config pb under GbmlConfig

        Returns:
            gbml_config_pb2.ProfilerConfig: The profiler config
        """
        return self.gbml_config_pb.profiler_config

    @property
    def shared_config(self) -> gbml_config_pb2.GbmlConfig.SharedConfig:
        """
        Allows access to shared_config pb under GbmlConfig

        Returns:
            gbml_config_pb2.SharedConfig: The shared config
        """
        return self.gbml_config_pb.shared_config

    @property
    def trainer_config(self) -> gbml_config_pb2.GbmlConfig.TrainerConfig:
        """
        Allows access to trainer_config pb under GbmlConfig

        Returns:
            gbml_config_pb2.TrainerConfig: The trainer config
        """
        return self.gbml_config_pb.trainer_config

    @property
    def should_use_experimental_glt_backend(self) -> bool:
        """
        Allows access to should_use_glt_backend under GbmlConfig

        Returns:
            bool: Whether to use GLT as a backend for current run
        """

        return bool(
            strtobool(
                dict(self.gbml_config_pb.feature_flags).get(
                    "should_run_glt_backend", "False"
                )
            )
        )

    @property
    def should_populate_predictions_path(self) -> bool:
        """
        Allows access to should_populate_predictions_path under GbmlConfig

        This flag is a temporary workaround to populate the extra embeddings for the same entity type

        Returns:
            bool: Whether to populate predictions path in the InferenceOutput for each entity type
        """
        return bool(
            strtobool(
                dict(self.gbml_config_pb.feature_flags).get(
                    "should_populate_predictions_path", "False"
                )
            )
        )
