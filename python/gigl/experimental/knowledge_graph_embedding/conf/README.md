# Knowledge Graph Embedding Configuration

This directory contains Hydra configuration files for the heterogeneous graph sparse embedding system. The configuration is organized into modular components that can be mixed and matched to create different experimental setups.

## Overview

The configuration system uses [Hydra](https://hydra.cc/) for managing experiments and implements the `HeterogeneousGraphSparseEmbeddingConfig` dataclass structure. It allows you to:

- Override individual parameters from the command line
- Compose different configurations for experiments
- Maintain reproducible experimental setups
- Handle both raw BigQuery data and enumerated preprocessed data

## Main Configuration (`config.yaml`)

The main configuration file defines the default components for each experiment:

```yaml
defaults:
  - dataset: cora           # Default dataset configuration (mapped to GraphConfig)
  - run: default           # Runtime environment settings (RunConfig)
  - model: default         # Model architecture configuration (ModelConfig)
  - training: default      # Training process configuration (TrainConfig)
  - validation: default    # Validation configuration (EvaluationPhaseConfig)
  - testing: default       # Testing configuration (EvaluationPhaseConfig)

hydra:
  output_subdir: null      # Disable Hydra's default output subdirectory
  run:
    dir: .                 # Set output directory to current directory
```

## Configuration Components

### Dataset Configuration (`dataset/`) → `GraphConfig`

Defines graph data sources and metadata, implemented as the `GraphConfig` dataclass. Each dataset configuration specifies:

#### Core Parameters

- **`metadata`**: (`GraphMetadataPbWrapper`) Graph structure definition following GraphMetadata protobuf format
  - **`node_types`**: List of node types in the graph (e.g., `["paper"]`)
  - **`edge_types`**: List of edge relationships with source/destination node types and relations
  - **`condensed_edge_type_map`**: Mapping from integer IDs to edge type definitions for efficient lookup
  - **`condensed_node_type_map`**: Mapping from integer IDs to node type names for efficient lookup

#### Data Source Options

The configuration supports two mutually exclusive data source types:

##### Raw Graph Data (`raw_graph_data`) → `RawGraphData`

For BigQuery data sources before preprocessing:

- **`node_data`**: (`List[BigqueryNodeDataReference]`) List of BigQuery node data sources
  - **`_target_`**: `gigl.src.data_preprocessor.lib.ingest.bigquery.BigqueryNodeDataReference`
  - **`node_type`**: Type of nodes this source provides
  - **`reference_uri`**: BigQuery table URI for the data source
  - **`identifier`**: Column name for node identifiers
- **`edge_data`**: (`List[BigqueryEdgeDataReference]`) List of BigQuery edge data sources
  - **`_target_`**: `gigl.src.data_preprocessor.lib.ingest.bigquery.BigqueryEdgeDataReference`
  - **`edge_type`**: Edge type specification with `_target_`, `src_node_type`, `relation`, `dst_node_type`
  - **`reference_uri`**: BigQuery table URI for the edge data
  - **`src_identifier`**: Column name for source node identifiers
  - **`dst_identifier`**: Column name for destination node identifiers

##### Enumerated Graph Data (`enumerated_graph_data`) → `EnumeratedGraphData`

For preprocessed data with integer ID mappings:

- **`node_data`**: (`List[EnumeratorNodeTypeMetadata]`) Metadata for enumerated node types with ID mappings
- **`edge_data`**: (`List[EnumeratorEdgeTypeMetadata]`) Metadata for enumerated edge types with ID mappings

#### Available Datasets

- **`cora/cora.yaml`**: CORA citation network dataset with BigQuery raw data sources
- **`mag240/mag240_papers.yaml`**: MAG240M papers subset dataset with BigQuery raw data sources

### Model Configuration (`model/`) → `ModelConfig`

Defines the neural network architecture for knowledge graph embeddings, implemented as the `ModelConfig` dataclass.

#### Static Configuration Parameters

- **`node_embedding_dim`**: (`int`, default: `128`) Dimensionality of node embeddings. Higher dimensions can capture more complex relationships but require more memory and computation.

- **`embedding_similarity_type`**: (`SimilarityType`, default: `COSINE`) Method for computing similarity between embeddings:
  - `COSINE`: Cosine similarity (normalized dot product)
  - `DOT`: Dot product similarity

- **`src_operator`**: (`OperatorType`, default: `IDENTITY`) Transformation operator applied to source node embeddings before computing edge scores. Can be identity (no transformation) or learned operators.

- **`dst_operator`**: (`OperatorType`, default: `IDENTITY`) Transformation operator applied to destination node embeddings before computing edge scores.

#### Runtime-Populated Parameters

These parameters are automatically populated during configuration initialization and should not be set manually in YAML files:

- **`training_sampling`**: (`Optional[SamplingConfig]`) Sampling configuration used during training phase
- **`validation_sampling`**: (`Optional[SamplingConfig]`) Sampling configuration used during validation phase
- **`testing_sampling`**: (`Optional[SamplingConfig]`) Sampling configuration used during testing phase
- **`num_edge_types`**: (`Optional[int]`) Number of distinct edge types in the knowledge graph (populated from graph metadata)
- **`embeddings_config`**: (`Optional[List[torchrec.EmbeddingBagConfig]]`) TorchRec embedding configuration for sparse embeddings, including sharding strategies and optimization settings

### Training Configuration (`training/`) → `TrainConfig`

Controls the training process and optimization settings, implemented as the `TrainConfig` dataclass.

#### Core Training Parameters

- **`max_steps`**: (`Optional[int]`, default: `None`) Maximum number of training steps to perform. If None, training continues until early stopping or manual interruption.

#### Early Stopping (`early_stopping`) → `EarlyStoppingConfig`

- **`patience`**: (`Optional[int]`, default: `None`) Number of evaluation steps to wait for improvement before stopping training. Helps prevent overfitting by stopping when validation performance plateaus. If None, early stopping is disabled.

#### Training Data Loading (`dataloader`) → `DataloaderConfig`

- **`num_workers`**: (`int`, default: `1`) Number of worker processes for data loading. Higher values can improve data loading speed but use more memory and CPU cores. Setting to 0 uses the main process.
- **`pin_memory`**: (`bool`, default: `True`) Whether to pin loaded data tensors to GPU memory for faster host-to-device transfer. Should be True when using CUDA.

#### Sampling Strategy (`sampling`) → `SamplingConfig`

Negative sampling is crucial for contrastive learning in knowledge graph embeddings:

- **`negative_corruption_side`**: (`NegativeSamplingCorruptionType`, default: `DST`) Which side of the edge to corrupt for negative sampling:
  - `DST`: Corrupt the destination node
  - `SRC`: Corrupt the source node
- **`positive_edge_batch_size`**: (`int`, default: `1024`) Number of positive (true) edges to process in each batch. Controls memory usage and training stability.
- **`num_inbatch_negatives_per_edge`**: (`int`, default: `0`) Number of negative samples generated per positive edge using other edges in the same batch. Memory-efficient but may have limited diversity.
- **`num_random_negatives_per_edge`**: (`int`, default: `1024`) Number of negative samples generated per positive edge by randomly corrupting nodes. Provides high diversity but requires more computation.

#### Optimization (`optimizer`) → `OptimizerConfig`

Separate optimizers for sparse and dense parameters, as knowledge graph embedding models typically have both:

- **`sparse`**: (`OptimizerParamsConfig`) Settings for sparse embeddings (for nodes):
  - **`lr`**: (`float`, default: `0.01`) Learning rate for sparse parameters
  - **`weight_decay`**: (`float`, default: `0.001`) L2 regularization coefficient for sparse parameters
- **`dense`**: (`OptimizerParamsConfig`) Settings for dense model parameters (linear layers):
  - **`lr`**: (`float`, default: `0.01`) Learning rate for dense parameters
  - **`weight_decay`**: (`float`, default: `0.001`) L2 regularization coefficient for dense parameters

#### Distributed Training (`distributed`) → `DistributedConfig`

- **`num_processes_per_machine`**: (`int`, default: auto-detected GPU count or 1) Number of training processes to spawn per machine. Each process typically uses one GPU. Automatically adjusted based on available GPUs.
- **`storage_reservation_percentage`**: (`float`, default: `0.1`) Storage percentage buffer used by TorchRec to account for overhead on dense tensor and KJT storage.

#### Checkpointing (`checkpointing`) → `CheckpointingConfig`

- **`save_every`**: (`int`, default: `10000`) Save a checkpoint every N training steps. Allows recovery from failures and monitoring of training progress.
- **`should_save_async`**: (`bool`, default: `True`) Whether to save checkpoints asynchronously to avoid blocking training. Improves training efficiency but may use additional memory.
- **`load_from_path`**: (`Optional[str]`, default: `None`) Path to a checkpoint file to resume training from. If None, training starts from scratch.
- **`save_to_path`**: (`Optional[str]`, default: `None`) Directory path where checkpoints will be saved. If None, checkpoints are not saved.

#### Logging (`logging`) → `LoggingConfig`

- **`log_every`**: (`int`, default: `1`) Log training metrics every N steps. More frequent logging provides better monitoring but may slow down training.

### Validation Configuration (`validation/`) → `EvaluationPhaseConfig`

Controls validation during training to monitor model performance, implemented as the `EvaluationPhaseConfig` dataclass.

#### Evaluation Schedule

- **`step_frequency`**: (`Optional[int]`, default: `None`) How often to run evaluation during training (every N steps). If None, evaluation runs only at the end of training.
- **`num_batches`**: (`Optional[int]`, default: `None`) Maximum number of batches to evaluate. Useful for faster evaluation on large datasets by sampling a subset. If None, evaluates all data.

#### Evaluation Data Loading (`dataloader`) → `DataloaderConfig`

- **`num_workers`**: (`int`, default: `1`) Number of worker processes for data loading during evaluation
- **`pin_memory`**: (`bool`, default: `True`) Whether to pin loaded data tensors to GPU memory for faster transfer

#### Evaluation Sampling Strategy (`sampling`) → `SamplingConfig`

Uses the same sampling configuration structure as training. Should match or be compatible with training sampling for fair comparison:

- **`negative_corruption_side`**: (`NegativeSamplingCorruptionType`, default: `DST`) Which side to corrupt for negative sampling
- **`positive_edge_batch_size`**: (`int`, default: `1024`) Number of positive edges per evaluation batch
- **`num_random_negatives_per_edge`**: (`int`, default: `1024`) Random negative samples per positive edge
- **`num_inbatch_negatives_per_edge`**: (`int`, default: `0`) In-batch negative samples per positive edge

#### Evaluation Metrics

- **`hit_rates_at_k`**: (`List[int]`, default: `[1, 10, 100]`) List of k values for computing Hit@k (Hits at k) metrics. Hit@k measures if the correct answer appears in the top k predictions.

#### Validation Rules

The configuration includes automatic validation that ensures `max(hit_rates_at_k) <= (num_random_negatives_per_edge + num_inbatch_negatives_per_edge)`. This guarantees sufficient negative samples for meaningful ranking evaluation.

### Testing Configuration (`testing/`) → `EvaluationPhaseConfig`

Defines settings for final model evaluation on test data. Uses the same `EvaluationPhaseConfig` dataclass structure as validation configuration. See the [Validation Configuration](#validation-configuration-validation--evaluationphaseconfig) section for detailed parameter descriptions.

Key differences from validation:

- Typically runs only once at the end of training
- May use larger `num_batches` or no limit for comprehensive evaluation
- Results are used for final model assessment rather than training decisions

### Runtime Configuration (`run/`) → `RunConfig`

Controls the runtime execution environment and hardware settings, implemented as the `RunConfig` dataclass.

#### Parameters

- **`should_use_cuda`**: (`bool`, default: `True`) Whether to use CUDA (GPU) acceleration for training. If True, training will use available GPUs for faster computation. If False, training will run on CPU only. Automatically adjusted based on GPU availability during initialization.

### Experiment Configuration (`experiment/`)

High-level experiment configurations that override default settings for specific experimental setups. These files use Hydra's `@package _global_` directive to override parameters at the root level, creating complete experimental configurations.

#### Available Experiments

- **`cora.yaml`**: Experiment configuration for CORA dataset
  - Uses CORA dataset with 128-dim embeddings
  - Configured for 10,000 training steps with in-batch negative sampling
  - Optimized learning rates: 0.01 for sparse, 0.01 for dense parameters

- **`mag240_papers_inbatch.yaml`**: MAG240M experiment using in-batch negative sampling
  - 128-dim embeddings with COSINE similarity
  - 50,000 training steps with distributed training (4 processes)
  - Large batch sizes (8192 positive edges) with 1024 in-batch negatives
  - Higher learning rate for sparse parameters (0.1)

- **`mag240_papers_random.yaml`**: MAG240M experiment using random negative sampling
  - 128-dim embeddings with DOT product similarity
  - 1,000,000 training steps for extensive training
  - Uses 1024 random negatives instead of in-batch negatives
  - Larger batch sizes (16384 positive edges) for efficiency

Each experiment file demonstrates different configurations for:

- Embedding similarity functions (COSINE vs DOT)
- Negative sampling strategies (in-batch vs random)
- Training scales (steps, batch sizes, distributed settings)
- Optimization hyperparameters

## Usage Examples

The configuration system is accessed through the `HeterogeneousGraphSparseEmbeddingConfig.from_omegaconf()` method, which parses Hydra DictConfig objects into strongly-typed dataclass structures.

### Basic Usage

Run with default configuration:

```bash
python train.py --applied_task_identifier=my_experiment --resource_config_uri=path/to/resource_config.yaml
```

### Override Parameters

Override specific parameters from the command line using Hydra syntax:

```bash
python train.py --applied_task_identifier=my_experiment --resource_config_uri=path/to/resource_config.yaml \
  model.node_embedding_dim=256 training.max_steps=50000 training.optimizer.sparse.lr=0.001
```

### Use Different Experiment Configuration

```bash
python train.py --applied_task_identifier=my_experiment --resource_config_uri=path/to/resource_config.yaml \
  experiment=mag240_papers_inbatch
```

### Mix and Match Components

```bash
python train.py --applied_task_identifier=my_experiment --resource_config_uri=path/to/resource_config.yaml \
  dataset=mag240/mag240_papers model.embedding_similarity_type=DOT training.distributed.num_processes_per_machine=8
```

### Multirun Experiments

Run parameter sweeps using Hydra's multirun feature:

```bash
python train.py --applied_task_identifier=my_experiment --resource_config_uri=path/to/resource_config.yaml \
  -m model.node_embedding_dim=64,128,256 training.optimizer.sparse.lr=0.001,0.01,0.1
```

### Custom Modeling Configuration

You can also provide a custom modeling configuration file:

```bash
python train.py --applied_task_identifier=my_experiment --resource_config_uri=path/to/resource_config.yaml \
  --modeling_config_uri=path/to/custom_config.yaml
```

## Configuration Inheritance and Processing

The configuration system follows this processing hierarchy:

1. **Dataclass defaults**: Default values defined in the Python dataclasses (e.g., `ModelConfig`, `TrainConfig`)
2. **Base component configs**: Default YAML files (`model/default.yaml`, `training/default.yaml`, etc.)
3. **Dataset-specific configs**: Selected via `defaults.dataset` (e.g., `cora/cora.yaml`)
4. **Experiment overrides**: Applied via `experiment=<name>` using `@package _global_` directive
5. **Command-line overrides**: Specified with Hydra `key=value` syntax

### Configuration Parsing Process

1. **Hydra Configuration**: Hydra composes the final configuration from YAML files and command-line overrides
2. **Dataclass Conversion**: `HeterogeneousGraphSparseEmbeddingConfig.from_omegaconf()` converts the Hydra DictConfig to strongly-typed dataclasses
3. **Runtime Population**: Additional fields are populated automatically:
   - Graph metadata is parsed from protobuf format
   - Data references are instantiated using `hydra.utils.instantiate()`
   - Runtime parameters like `num_edge_types` are computed
   - GPU availability adjusts distributed training settings

### Validation and Error Handling

The system includes automatic validation:

- **Hit@k validation**: Ensures sufficient negative samples for evaluation metrics
- **GPU availability**: Automatically adjusts `num_processes_per_machine` based on available GPUs
- **Type checking**: Strongly-typed dataclasses catch configuration errors early
- **Required fields**: Missing required configuration (e.g., graph metadata) triggers clear error messages

Parameters specified later in the hierarchy override earlier ones, enabling flexible experimentation while maintaining sensible defaults and type safety.
