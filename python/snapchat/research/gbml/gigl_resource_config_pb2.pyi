"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _Component:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _ComponentEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_Component.ValueType], builtins.type):  # noqa: F821
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    Component_Unknown: _Component.ValueType  # 0
    Component_Config_Validator: _Component.ValueType  # 1
    Component_Config_Populator: _Component.ValueType  # 2
    Component_Data_Preprocessor: _Component.ValueType  # 3
    Component_Subgraph_Sampler: _Component.ValueType  # 4
    Component_Split_Generator: _Component.ValueType  # 5
    Component_Trainer: _Component.ValueType  # 6
    Component_Inferencer: _Component.ValueType  # 7

class Component(_Component, metaclass=_ComponentEnumTypeWrapper):
    """Enum for pipeline components"""

Component_Unknown: Component.ValueType  # 0
Component_Config_Validator: Component.ValueType  # 1
Component_Config_Populator: Component.ValueType  # 2
Component_Data_Preprocessor: Component.ValueType  # 3
Component_Subgraph_Sampler: Component.ValueType  # 4
Component_Split_Generator: Component.ValueType  # 5
Component_Trainer: Component.ValueType  # 6
Component_Inferencer: Component.ValueType  # 7
global___Component = Component

class SparkResourceConfig(google.protobuf.message.Message):
    """Configuration for Spark Components"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MACHINE_TYPE_FIELD_NUMBER: builtins.int
    NUM_LOCAL_SSDS_FIELD_NUMBER: builtins.int
    NUM_REPLICAS_FIELD_NUMBER: builtins.int
    machine_type: builtins.str
    """Machine type for Spark Resource"""
    num_local_ssds: builtins.int
    """Number of local SSDs"""
    num_replicas: builtins.int
    """Num workers for Spark Resource"""
    def __init__(
        self,
        *,
        machine_type: builtins.str = ...,
        num_local_ssds: builtins.int = ...,
        num_replicas: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["machine_type", b"machine_type", "num_local_ssds", b"num_local_ssds", "num_replicas", b"num_replicas"]) -> None: ...

global___SparkResourceConfig = SparkResourceConfig

class DataflowResourceConfig(google.protobuf.message.Message):
    """Configuration for Dataflow Components"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUM_WORKERS_FIELD_NUMBER: builtins.int
    MAX_NUM_WORKERS_FIELD_NUMBER: builtins.int
    MACHINE_TYPE_FIELD_NUMBER: builtins.int
    DISK_SIZE_GB_FIELD_NUMBER: builtins.int
    num_workers: builtins.int
    """Number of workers for Dataflow resources"""
    max_num_workers: builtins.int
    """Maximum number of workers for Dataflow resources"""
    machine_type: builtins.str
    """Machine type for Dataflow resources"""
    disk_size_gb: builtins.int
    """Disk size in GB for Dataflow resources"""
    def __init__(
        self,
        *,
        num_workers: builtins.int = ...,
        max_num_workers: builtins.int = ...,
        machine_type: builtins.str = ...,
        disk_size_gb: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["disk_size_gb", b"disk_size_gb", "machine_type", b"machine_type", "max_num_workers", b"max_num_workers", "num_workers", b"num_workers"]) -> None: ...

global___DataflowResourceConfig = DataflowResourceConfig

class DataPreprocessorConfig(google.protobuf.message.Message):
    """Configuration for Data Preprocessor"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    EDGE_PREPROCESSOR_CONFIG_FIELD_NUMBER: builtins.int
    NODE_PREPROCESSOR_CONFIG_FIELD_NUMBER: builtins.int
    @property
    def edge_preprocessor_config(self) -> global___DataflowResourceConfig: ...
    @property
    def node_preprocessor_config(self) -> global___DataflowResourceConfig: ...
    def __init__(
        self,
        *,
        edge_preprocessor_config: global___DataflowResourceConfig | None = ...,
        node_preprocessor_config: global___DataflowResourceConfig | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["edge_preprocessor_config", b"edge_preprocessor_config", "node_preprocessor_config", b"node_preprocessor_config"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["edge_preprocessor_config", b"edge_preprocessor_config", "node_preprocessor_config", b"node_preprocessor_config"]) -> None: ...

global___DataPreprocessorConfig = DataPreprocessorConfig

class VertexAiTrainerConfig(google.protobuf.message.Message):
    """(deprecated)
    Configuration for Vertex AI training resources
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MACHINE_TYPE_FIELD_NUMBER: builtins.int
    GPU_TYPE_FIELD_NUMBER: builtins.int
    GPU_LIMIT_FIELD_NUMBER: builtins.int
    NUM_REPLICAS_FIELD_NUMBER: builtins.int
    machine_type: builtins.str
    """Machine type for training job"""
    gpu_type: builtins.str
    """GPU type for training job. Must be set to 'ACCELERATOR_TYPE_UNSPECIFIED' for cpu training."""
    gpu_limit: builtins.int
    """GPU limit for training job. Must be set to 0 for cpu training."""
    num_replicas: builtins.int
    """Num workers for training job"""
    def __init__(
        self,
        *,
        machine_type: builtins.str = ...,
        gpu_type: builtins.str = ...,
        gpu_limit: builtins.int = ...,
        num_replicas: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["gpu_limit", b"gpu_limit", "gpu_type", b"gpu_type", "machine_type", b"machine_type", "num_replicas", b"num_replicas"]) -> None: ...

global___VertexAiTrainerConfig = VertexAiTrainerConfig

class KFPTrainerConfig(google.protobuf.message.Message):
    """(deprecated)
    Configuration for KFP training resources
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CPU_REQUEST_FIELD_NUMBER: builtins.int
    MEMORY_REQUEST_FIELD_NUMBER: builtins.int
    GPU_TYPE_FIELD_NUMBER: builtins.int
    GPU_LIMIT_FIELD_NUMBER: builtins.int
    NUM_REPLICAS_FIELD_NUMBER: builtins.int
    cpu_request: builtins.str
    """Num CPU requested for training job (str) which can be a number or a number followed by "m", which means 1/1000"""
    memory_request: builtins.str
    """Amount of Memory requested for training job (str) can either be a number or a number followed by one of "Ei", "Pi", "Ti", "Gi", "Mi", "Ki"."""
    gpu_type: builtins.str
    """GPU type for training job. Must be set to 'ACCELERATOR_TYPE_UNSPECIFIED' for cpu training."""
    gpu_limit: builtins.int
    """GPU limit for training job. Must be set to 0 for cpu training."""
    num_replicas: builtins.int
    """Number of replicas for training job"""
    def __init__(
        self,
        *,
        cpu_request: builtins.str = ...,
        memory_request: builtins.str = ...,
        gpu_type: builtins.str = ...,
        gpu_limit: builtins.int = ...,
        num_replicas: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["cpu_request", b"cpu_request", "gpu_limit", b"gpu_limit", "gpu_type", b"gpu_type", "memory_request", b"memory_request", "num_replicas", b"num_replicas"]) -> None: ...

global___KFPTrainerConfig = KFPTrainerConfig

class LocalTrainerConfig(google.protobuf.message.Message):
    """(deprecated)
    Configuration for Local Training
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUM_WORKERS_FIELD_NUMBER: builtins.int
    num_workers: builtins.int
    def __init__(
        self,
        *,
        num_workers: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["num_workers", b"num_workers"]) -> None: ...

global___LocalTrainerConfig = LocalTrainerConfig

class VertexAiResourceConfig(google.protobuf.message.Message):
    """Configuration for Vertex AI resources"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    MACHINE_TYPE_FIELD_NUMBER: builtins.int
    GPU_TYPE_FIELD_NUMBER: builtins.int
    GPU_LIMIT_FIELD_NUMBER: builtins.int
    NUM_REPLICAS_FIELD_NUMBER: builtins.int
    TIMEOUT_FIELD_NUMBER: builtins.int
    machine_type: builtins.str
    """Machine type for job"""
    gpu_type: builtins.str
    """GPU type for job. Must be set to 'ACCELERATOR_TYPE_UNSPECIFIED' for cpu."""
    gpu_limit: builtins.int
    """GPU limit for job. Must be set to 0 for cpu."""
    num_replicas: builtins.int
    """Num workers for job"""
    timeout: builtins.int
    """Timeout in seconds for the job. If unset or zero, will use the default @ google.cloud.aiplatform.CustomJob, which is 7 days: 
    https://github.com/googleapis/python-aiplatform/blob/58fbabdeeefd1ccf1a9d0c22eeb5606aeb9c2266/google/cloud/aiplatform/jobs.py#L2252-L2253
    """
    def __init__(
        self,
        *,
        machine_type: builtins.str = ...,
        gpu_type: builtins.str = ...,
        gpu_limit: builtins.int = ...,
        num_replicas: builtins.int = ...,
        timeout: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["gpu_limit", b"gpu_limit", "gpu_type", b"gpu_type", "machine_type", b"machine_type", "num_replicas", b"num_replicas", "timeout", b"timeout"]) -> None: ...

global___VertexAiResourceConfig = VertexAiResourceConfig

class KFPResourceConfig(google.protobuf.message.Message):
    """Configuration for KFP job resources"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    CPU_REQUEST_FIELD_NUMBER: builtins.int
    MEMORY_REQUEST_FIELD_NUMBER: builtins.int
    GPU_TYPE_FIELD_NUMBER: builtins.int
    GPU_LIMIT_FIELD_NUMBER: builtins.int
    NUM_REPLICAS_FIELD_NUMBER: builtins.int
    cpu_request: builtins.str
    """Num CPU requested for job (str) which can be a number or a number followed by "m", which means 1/1000"""
    memory_request: builtins.str
    """Amount of Memory requested for job (str) can either be a number or a number followed by one of "Ei", "Pi", "Ti", "Gi", "Mi", "Ki"."""
    gpu_type: builtins.str
    """GPU type for job. Must be set to 'ACCELERATOR_TYPE_UNSPECIFIED' for cpu."""
    gpu_limit: builtins.int
    """GPU limit for job. Must be set to 0 for cpu."""
    num_replicas: builtins.int
    """Number of replicas for job"""
    def __init__(
        self,
        *,
        cpu_request: builtins.str = ...,
        memory_request: builtins.str = ...,
        gpu_type: builtins.str = ...,
        gpu_limit: builtins.int = ...,
        num_replicas: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["cpu_request", b"cpu_request", "gpu_limit", b"gpu_limit", "gpu_type", b"gpu_type", "memory_request", b"memory_request", "num_replicas", b"num_replicas"]) -> None: ...

global___KFPResourceConfig = KFPResourceConfig

class LocalResourceConfig(google.protobuf.message.Message):
    """Configuration for Local Jobs"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NUM_WORKERS_FIELD_NUMBER: builtins.int
    num_workers: builtins.int
    def __init__(
        self,
        *,
        num_workers: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing_extensions.Literal["num_workers", b"num_workers"]) -> None: ...

global___LocalResourceConfig = LocalResourceConfig

class DistributedTrainerConfig(google.protobuf.message.Message):
    """(deprecated)
    Configuration for distributed training resources
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VERTEX_AI_TRAINER_CONFIG_FIELD_NUMBER: builtins.int
    KFP_TRAINER_CONFIG_FIELD_NUMBER: builtins.int
    LOCAL_TRAINER_CONFIG_FIELD_NUMBER: builtins.int
    @property
    def vertex_ai_trainer_config(self) -> global___VertexAiTrainerConfig: ...
    @property
    def kfp_trainer_config(self) -> global___KFPTrainerConfig: ...
    @property
    def local_trainer_config(self) -> global___LocalTrainerConfig: ...
    def __init__(
        self,
        *,
        vertex_ai_trainer_config: global___VertexAiTrainerConfig | None = ...,
        kfp_trainer_config: global___KFPTrainerConfig | None = ...,
        local_trainer_config: global___LocalTrainerConfig | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["kfp_trainer_config", b"kfp_trainer_config", "local_trainer_config", b"local_trainer_config", "trainer_config", b"trainer_config", "vertex_ai_trainer_config", b"vertex_ai_trainer_config"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["kfp_trainer_config", b"kfp_trainer_config", "local_trainer_config", b"local_trainer_config", "trainer_config", b"trainer_config", "vertex_ai_trainer_config", b"vertex_ai_trainer_config"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["trainer_config", b"trainer_config"]) -> typing_extensions.Literal["vertex_ai_trainer_config", "kfp_trainer_config", "local_trainer_config"] | None: ...

global___DistributedTrainerConfig = DistributedTrainerConfig

class TrainerResourceConfig(google.protobuf.message.Message):
    """Configuration for training resources"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VERTEX_AI_TRAINER_CONFIG_FIELD_NUMBER: builtins.int
    KFP_TRAINER_CONFIG_FIELD_NUMBER: builtins.int
    LOCAL_TRAINER_CONFIG_FIELD_NUMBER: builtins.int
    @property
    def vertex_ai_trainer_config(self) -> global___VertexAiResourceConfig: ...
    @property
    def kfp_trainer_config(self) -> global___KFPResourceConfig: ...
    @property
    def local_trainer_config(self) -> global___LocalResourceConfig: ...
    def __init__(
        self,
        *,
        vertex_ai_trainer_config: global___VertexAiResourceConfig | None = ...,
        kfp_trainer_config: global___KFPResourceConfig | None = ...,
        local_trainer_config: global___LocalResourceConfig | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["kfp_trainer_config", b"kfp_trainer_config", "local_trainer_config", b"local_trainer_config", "trainer_config", b"trainer_config", "vertex_ai_trainer_config", b"vertex_ai_trainer_config"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["kfp_trainer_config", b"kfp_trainer_config", "local_trainer_config", b"local_trainer_config", "trainer_config", b"trainer_config", "vertex_ai_trainer_config", b"vertex_ai_trainer_config"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["trainer_config", b"trainer_config"]) -> typing_extensions.Literal["vertex_ai_trainer_config", "kfp_trainer_config", "local_trainer_config"] | None: ...

global___TrainerResourceConfig = TrainerResourceConfig

class InferencerResourceConfig(google.protobuf.message.Message):
    """Configuration for distributed inference resources"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VERTEX_AI_INFERENCER_CONFIG_FIELD_NUMBER: builtins.int
    DATAFLOW_INFERENCER_CONFIG_FIELD_NUMBER: builtins.int
    LOCAL_INFERENCER_CONFIG_FIELD_NUMBER: builtins.int
    @property
    def vertex_ai_inferencer_config(self) -> global___VertexAiResourceConfig: ...
    @property
    def dataflow_inferencer_config(self) -> global___DataflowResourceConfig: ...
    @property
    def local_inferencer_config(self) -> global___LocalResourceConfig: ...
    def __init__(
        self,
        *,
        vertex_ai_inferencer_config: global___VertexAiResourceConfig | None = ...,
        dataflow_inferencer_config: global___DataflowResourceConfig | None = ...,
        local_inferencer_config: global___LocalResourceConfig | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["dataflow_inferencer_config", b"dataflow_inferencer_config", "inferencer_config", b"inferencer_config", "local_inferencer_config", b"local_inferencer_config", "vertex_ai_inferencer_config", b"vertex_ai_inferencer_config"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["dataflow_inferencer_config", b"dataflow_inferencer_config", "inferencer_config", b"inferencer_config", "local_inferencer_config", b"local_inferencer_config", "vertex_ai_inferencer_config", b"vertex_ai_inferencer_config"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["inferencer_config", b"inferencer_config"]) -> typing_extensions.Literal["vertex_ai_inferencer_config", "dataflow_inferencer_config", "local_inferencer_config"] | None: ...

global___InferencerResourceConfig = InferencerResourceConfig

class SharedResourceConfig(google.protobuf.message.Message):
    """Shared resources configuration"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    class CommonComputeConfig(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        PROJECT_FIELD_NUMBER: builtins.int
        REGION_FIELD_NUMBER: builtins.int
        TEMP_ASSETS_BUCKET_FIELD_NUMBER: builtins.int
        TEMP_REGIONAL_ASSETS_BUCKET_FIELD_NUMBER: builtins.int
        PERM_ASSETS_BUCKET_FIELD_NUMBER: builtins.int
        TEMP_ASSETS_BQ_DATASET_NAME_FIELD_NUMBER: builtins.int
        EMBEDDING_BQ_DATASET_NAME_FIELD_NUMBER: builtins.int
        GCP_SERVICE_ACCOUNT_EMAIL_FIELD_NUMBER: builtins.int
        DATAFLOW_RUNNER_FIELD_NUMBER: builtins.int
        project: builtins.str
        """GCP Project"""
        region: builtins.str
        """GCP Region where compute is to be scheduled"""
        temp_assets_bucket: builtins.str
        """GCS Bucket for where temporary assets are to be stored"""
        temp_regional_assets_bucket: builtins.str
        """Regional GCS Bucket used to store temporary assets"""
        perm_assets_bucket: builtins.str
        """Regional GCS Bucket that will store permanent assets like Trained Model"""
        temp_assets_bq_dataset_name: builtins.str
        """Path to BQ dataset used to store temporary assets"""
        embedding_bq_dataset_name: builtins.str
        """Path to BQ Dataset used to persist generated embeddings and predictions"""
        gcp_service_account_email: builtins.str
        """The GCP service account email being used to schedule compute on GCP"""
        dataflow_runner: builtins.str
        """The runner to use for Dataflow i.e DirectRunner or DataflowRunner"""
        def __init__(
            self,
            *,
            project: builtins.str = ...,
            region: builtins.str = ...,
            temp_assets_bucket: builtins.str = ...,
            temp_regional_assets_bucket: builtins.str = ...,
            perm_assets_bucket: builtins.str = ...,
            temp_assets_bq_dataset_name: builtins.str = ...,
            embedding_bq_dataset_name: builtins.str = ...,
            gcp_service_account_email: builtins.str = ...,
            dataflow_runner: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["dataflow_runner", b"dataflow_runner", "embedding_bq_dataset_name", b"embedding_bq_dataset_name", "gcp_service_account_email", b"gcp_service_account_email", "perm_assets_bucket", b"perm_assets_bucket", "project", b"project", "region", b"region", "temp_assets_bq_dataset_name", b"temp_assets_bq_dataset_name", "temp_assets_bucket", b"temp_assets_bucket", "temp_regional_assets_bucket", b"temp_regional_assets_bucket"]) -> None: ...

    class ResourceLabelsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing_extensions.Literal["key", b"key", "value", b"value"]) -> None: ...

    RESOURCE_LABELS_FIELD_NUMBER: builtins.int
    COMMON_COMPUTE_CONFIG_FIELD_NUMBER: builtins.int
    @property
    def resource_labels(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]: ...
    @property
    def common_compute_config(self) -> global___SharedResourceConfig.CommonComputeConfig: ...
    def __init__(
        self,
        *,
        resource_labels: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
        common_compute_config: global___SharedResourceConfig.CommonComputeConfig | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["common_compute_config", b"common_compute_config"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["common_compute_config", b"common_compute_config", "resource_labels", b"resource_labels"]) -> None: ...

global___SharedResourceConfig = SharedResourceConfig

class GiglResourceConfig(google.protobuf.message.Message):
    """GiGL resources configuration"""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    SHARED_RESOURCE_CONFIG_URI_FIELD_NUMBER: builtins.int
    SHARED_RESOURCE_CONFIG_FIELD_NUMBER: builtins.int
    PREPROCESSOR_CONFIG_FIELD_NUMBER: builtins.int
    SUBGRAPH_SAMPLER_CONFIG_FIELD_NUMBER: builtins.int
    SPLIT_GENERATOR_CONFIG_FIELD_NUMBER: builtins.int
    TRAINER_CONFIG_FIELD_NUMBER: builtins.int
    INFERENCER_CONFIG_FIELD_NUMBER: builtins.int
    TRAINER_RESOURCE_CONFIG_FIELD_NUMBER: builtins.int
    INFERENCER_RESOURCE_CONFIG_FIELD_NUMBER: builtins.int
    shared_resource_config_uri: builtins.str
    @property
    def shared_resource_config(self) -> global___SharedResourceConfig: ...
    @property
    def preprocessor_config(self) -> global___DataPreprocessorConfig:
        """Configuration for Data Preprocessor"""
    @property
    def subgraph_sampler_config(self) -> global___SparkResourceConfig:
        """Configuration for Spark subgraph sampler"""
    @property
    def split_generator_config(self) -> global___SparkResourceConfig:
        """Configuration for Spark split generator"""
    @property
    def trainer_config(self) -> global___DistributedTrainerConfig:
        """(deprecated)
        Configuration for trainer
        """
    @property
    def inferencer_config(self) -> global___DataflowResourceConfig:
        """(deprecated)
        Configuration for inferencer
        """
    @property
    def trainer_resource_config(self) -> global___TrainerResourceConfig:
        """Configuration for distributed trainer"""
    @property
    def inferencer_resource_config(self) -> global___InferencerResourceConfig:
        """Configuration for distributed inferencer"""
    def __init__(
        self,
        *,
        shared_resource_config_uri: builtins.str = ...,
        shared_resource_config: global___SharedResourceConfig | None = ...,
        preprocessor_config: global___DataPreprocessorConfig | None = ...,
        subgraph_sampler_config: global___SparkResourceConfig | None = ...,
        split_generator_config: global___SparkResourceConfig | None = ...,
        trainer_config: global___DistributedTrainerConfig | None = ...,
        inferencer_config: global___DataflowResourceConfig | None = ...,
        trainer_resource_config: global___TrainerResourceConfig | None = ...,
        inferencer_resource_config: global___InferencerResourceConfig | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal["inferencer_config", b"inferencer_config", "inferencer_resource_config", b"inferencer_resource_config", "preprocessor_config", b"preprocessor_config", "shared_resource", b"shared_resource", "shared_resource_config", b"shared_resource_config", "shared_resource_config_uri", b"shared_resource_config_uri", "split_generator_config", b"split_generator_config", "subgraph_sampler_config", b"subgraph_sampler_config", "trainer_config", b"trainer_config", "trainer_resource_config", b"trainer_resource_config"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing_extensions.Literal["inferencer_config", b"inferencer_config", "inferencer_resource_config", b"inferencer_resource_config", "preprocessor_config", b"preprocessor_config", "shared_resource", b"shared_resource", "shared_resource_config", b"shared_resource_config", "shared_resource_config_uri", b"shared_resource_config_uri", "split_generator_config", b"split_generator_config", "subgraph_sampler_config", b"subgraph_sampler_config", "trainer_config", b"trainer_config", "trainer_resource_config", b"trainer_resource_config"]) -> None: ...
    def WhichOneof(self, oneof_group: typing_extensions.Literal["shared_resource", b"shared_resource"]) -> typing_extensions.Literal["shared_resource_config_uri", "shared_resource_config"] | None: ...

global___GiglResourceConfig = GiglResourceConfig
