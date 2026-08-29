import math
from typing import Union

import gigl.experimental.knowledge_graph_embedding.lib.constants.bq as bq_constants
import gigl.experimental.knowledge_graph_embedding.lib.constants.gcs as gcs_constants
import torch
from gigl.experimental.knowledge_graph_embedding.lib.config import (
    HeterogeneousGraphSparseEmbeddingConfig,
)
from gigl.experimental.knowledge_graph_embedding.lib.data.edge_dataset import (
    AppliedTaskIdentifier,
)
from gigl.experimental.knowledge_graph_embedding.lib.data.node_batch import NodeBatch
from gigl.experimental.knowledge_graph_embedding.lib.model.heterogeneous_graph_model import (
    HeterogeneousGraphSparseEmbeddingModelAndLoss,
    ModelPhase,
)
from google.cloud import bigquery
from torchrec.distributed import DistributedModelParallel, TrainPipelineSparseDist

import gigl.src.data_preprocessor.lib.enumerate.queries as enumeration_queries
from gigl.common.data import export
from gigl.common.logger import Logger
from gigl.env.pipelines_config import get_resource_config
from gigl.src.common.types.graph_data import (
    CondensedEdgeType,
    CondensedNodeType,
    EdgeType,
    NodeType,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.common.utils.bq import BqUtils
from gigl.src.data_preprocessor.lib.enumerate.utils import EnumeratorNodeTypeMetadata

logger = Logger()


def infer_and_export_node_embeddings(
    is_src: bool,
    condensed_edge_type: CondensedEdgeType,
    edge_type: EdgeType,
    pipeline: TrainPipelineSparseDist,
    applied_task_identifier: AppliedTaskIdentifier,
    rank_prefix_str: str,
    rank: int,
    world_size: int,
    device: torch.device,
    kge_config: HeterogeneousGraphSparseEmbeddingConfig,
    graph_metadata: GraphMetadataPbWrapper,
    condensed_node_type_to_vocab_size_map: dict[CondensedNodeType, int],
):
    """Infer and export node embeddings for either source or destination nodes of a given edge type.

    This function handles the complete inference pipeline for node embeddings, including:
    - Setting the appropriate model phase (source or destination inference)
    - Determining the correct node type and vocabulary size
    - Creating the output directory path
    - Initializing the embedding exporter
    - Creating and processing the inference data loader
    - Batch processing and exporting embeddings to GCS

    The function is designed to work in a distributed training setup where each process
    handles a portion of the nodes based on its rank. The embeddings are sharded and
    written to separate files per rank to enable parallel processing.

    Args:
        is_src (bool): If True, process source nodes; if False, process destination nodes.
        condensed_edge_type (CondensedEdgeType): The condensed representation of the edge type.
        edge_type (EdgeType): The full edge type containing source and destination node types.
        pipeline (TrainPipelineSparseDist): The distributed training pipeline for inference.
        applied_task_identifier (AppliedTaskIdentifier): Identifier for the applied task.
        rank_prefix_str (str): A prefix string for logging, typically includes the rank of the process.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes in distributed training.
        device (torch.device): The device to run the inference on.
        kge_config (HeterogeneousGraphSparseEmbeddingConfig): The configuration for the KGE model.
        graph_metadata (GraphMetadataPbWrapper): Metadata about the graph, including edge types and node types.
        condensed_node_type_to_vocab_size_map (dict[CondensedNodeType, int]): A mapping from condensed node types to
            their vocabulary sizes.

    Returns:
        None: Embeddings are directly exported to GCS via the exporter.
    """
    # Determine node type string for logging and model phase selection
    node_type_str = "src" if is_src else "dst"
    phase = ModelPhase.INFERENCE_SRC if is_src else ModelPhase.INFERENCE_DST

    # Set the model phase for inference (source or destination)
    pipeline._model.module.set_phase(phase)
    logger.info(
        f"{rank_prefix_str} Set model phase to {pipeline._model.module.phase} for inference."
    )

    # Extract the condensed node types for both source and destination from the edge type
    (
        src_condensed_node_type,
        dst_condensed_node_type,
    ) = graph_metadata.condensed_edge_type_to_condensed_node_types[
        condensed_edge_type
    ]

    # Select the appropriate condensed node type based on whether we're processing src or dst
    condensed_node_type = (
        src_condensed_node_type if is_src else dst_condensed_node_type
    )

    # Get the vocabulary size for the selected node type
    vocab_size = condensed_node_type_to_vocab_size_map[condensed_node_type]

    # Get the actual node type (not condensed) for this inference
    node_type = edge_type.src_node_type if is_src else edge_type.dst_node_type

    # Determine the appropriate GCS output path based on node type
    if is_src:
        embedding_dir = gcs_constants.get_embedding_output_path_for_src_node(
            applied_task_identifier=applied_task_identifier,
            edge_type=edge_type,
        )
    else:
        embedding_dir = gcs_constants.get_embedding_output_path_for_dst_node(
            applied_task_identifier=applied_task_identifier,
            edge_type=edge_type,
        )

    # Initialize the embedding exporter with rank-specific file naming
    # This ensures each distributed process writes to its own file
    exporter = export.EmbeddingExporter(
        export_dir=embedding_dir,
        file_prefix=f"{rank}_of_{world_size}_embeddings_",
        min_shard_size_threshold_bytes=1_000_000_000,  # 1GB threshold for sharding
    )

    # Calculate the range of nodes this rank should process
    # Each rank gets approximately vocab_size / world_size nodes
    nodes_per_rank = math.ceil(vocab_size / world_size)
    dataset = range(
        rank * nodes_per_rank, min((rank + 1) * nodes_per_rank, vocab_size)
    )

    logger.info(
        f"Rank {rank_prefix_str} processing nodes {dataset} with length {len(dataset)} of {node_type_str} nodes of type {node_type}."
    )

    # Create the data loader for this rank's subset of nodes
    inference_loader = NodeBatch.build_data_loader(
        dataset=dataset,
        condensed_node_type=condensed_node_type,
        condensed_edge_type=condensed_edge_type,
        graph_metadata=graph_metadata,
        sampling_config=kge_config.training.sampling,
        dataloader_config=kge_config.training.dataloader,
        pin_memory=device.type == "cuda",  # Use pinned memory for GPU acceleration
    )
    inference_iter = iter(inference_loader)

    # Process inference batches until all nodes are processed
    while True:
        try:
            # Run inference on the next batch and get node IDs and their embeddings
            (
                node_ids,
                node_embeddings,
            ) = pipeline.progress(inference_iter)

            # Export the embeddings to the exporter (which will handle GCS upload)
            # Move tensors to CPU to free GPU memory
            exporter.add_embedding(
                id_batch=node_ids.cpu(),
                embedding_batch=node_embeddings.cpu(),
                embedding_type=node_type,
            )
        except StopIteration:
            # All batches processed - flush remaining embeddings and log completion
            exporter.flush_embeddings()
            logger.info(
                f"{rank_prefix_str} Finished inference for {node_type_str} nodes of type {node_type} "
                + f"(for edge type {edge_type.src_node_type}-{edge_type.relation}-{edge_type.dst_node_type}). "
                + f"Embeddings written to {embedding_dir}."
            )
            break


def infer_and_export_embeddings(
    applied_task_identifier: AppliedTaskIdentifier,
    rank_prefix_str: str,
    rank: int,
    world_size: int,
    device: torch.device,
    kge_config: HeterogeneousGraphSparseEmbeddingConfig,
    model_and_loss: Union[
        DistributedModelParallel, HeterogeneousGraphSparseEmbeddingModelAndLoss
    ],
    optimizer: torch.optim.Optimizer,
    graph_metadata: GraphMetadataPbWrapper,
    condensed_node_type_to_vocab_size_map: dict[CondensedNodeType, int],
):
    """Run inference to generate source and destination node embeddings for all edge types.

    This function iterates over each edge type in the graph metadata, infers embeddings for
    both source and destination nodes, and exports them to GCS. It operates within a
    distributed training setup where each process handles a portion of the node embeddings
    based on its rank. The embeddings are saved in a structured manner in GCS, with each
    process writing its portion to separate files.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): Identifier for the applied task.
        rank_prefix_str (str): A prefix string for logging, typically includes the rank of the process.
        rank (int): The rank of the current process in distributed training.
        world_size (int): The total number of processes in distributed training.
        device (torch.device): The device to run the inference on.
        kge_config (HeterogeneousGraphSparseEmbeddingConfig): The configuration for the KGE model.
        model_and_loss (Union[DistributedModelParallel, HeterogeneousGraphSparseEmbeddingModelAndLoss]): The model and loss function to use for inference.
        optimizer (torch.optim.Optimizer): The optimizer used during training, needed for pipeline initialization.
        graph_metadata (GraphMetadataPbWrapper): Metadata about the graph, including edge types and node types.
        condensed_node_type_to_vocab_size_map (dict[CondensedNodeType, int]): A mapping from condensed node types to
            their vocabulary sizes.

    Returns:
        None: Embeddings are directly exported to GCS for each edge type.
    """

    logger.info(
        f"{rank_prefix_str} Running inference to predict src and dst node embeddings for each edge type."
    )

    # Initialize the distributed training pipeline for inference
    pipeline = TrainPipelineSparseDist(
        model=model_and_loss, optimizer=optimizer, device=device
    )
    logger.info(f"{rank_prefix_str} Initialized TrainPipelineSparseDist for inference.")

    # Run inference in no_grad context to save memory and improve performance
    with torch.no_grad():
        # Set model to evaluation mode for inference
        pipeline._model.eval()

        # Process each edge type in the graph metadata
        for (
            condensed_edge_type,
            edge_type,
        ) in sorted(graph_metadata.condensed_edge_type_to_edge_type_map.items()):
            logger.info(
                f"""{rank_prefix_str} Running inference for edge type {edge_type} on
                src node type {edge_type.src_node_type} and dst node type {edge_type.dst_node_type}."""
            )

            # Process source nodes for this edge type
            infer_and_export_node_embeddings(
                is_src=True,
                condensed_edge_type=condensed_edge_type,
                edge_type=edge_type,
                pipeline=pipeline,
                applied_task_identifier=applied_task_identifier,
                rank_prefix_str=rank_prefix_str,
                rank=rank,
                world_size=world_size,
                device=device,
                kge_config=kge_config,
                graph_metadata=graph_metadata,
                condensed_node_type_to_vocab_size_map=condensed_node_type_to_vocab_size_map,
            )

            # Process destination nodes for this edge type
            infer_and_export_node_embeddings(
                is_src=False,
                condensed_edge_type=condensed_edge_type,
                edge_type=edge_type,
                pipeline=pipeline,
                applied_task_identifier=applied_task_identifier,
                rank_prefix_str=rank_prefix_str,
                rank=rank,
                world_size=world_size,
                device=device,
                kge_config=kge_config,
                graph_metadata=graph_metadata,
                condensed_node_type_to_vocab_size_map=condensed_node_type_to_vocab_size_map,
            )

    logger.info(f"Finished writing all embeddings.")


def upload_embeddings_to_bigquery(
    applied_task_identifier: AppliedTaskIdentifier,
    graph_metadata: GraphMetadataPbWrapper,
    enumerated_node_metadata: list[EnumeratorNodeTypeMetadata],
):
    """Upload node embeddings from GCS to BigQuery for all edge types.

    This function iterates over each edge type in the graph metadata and loads the
    previously inferred embeddings from GCS into BigQuery tables. It creates both
    enumerated and unenumerated embedding tables for source and destination nodes
    of each edge type.

    Args:
        applied_task_identifier (AppliedTaskIdentifier): Identifier for the applied task.
        graph_metadata (GraphMetadataPbWrapper): Metadata about the graph, including edge types and node types.
        enumerated_node_metadata (list[EnumeratorNodeTypeMetadata]): Metadata for enumerated node types, used to map
            node types to their corresponding BigQuery tables.

    Returns:
        None: Embeddings are uploaded to BigQuery tables.
    """

    node_type_to_enumerated_metadata_tables: dict[NodeType, str] = {
        node_type_metadata.enumerated_node_data_reference.node_type: node_type_metadata.bq_unique_node_ids_enumerated_table_name
        for node_type_metadata in enumerated_node_metadata
    }

    logger.info(f"Loading embeddings to BigQuery.")

    for edge_type in graph_metadata.edge_types:
        edge_type_src_node_embedding_dir = (
            gcs_constants.get_embedding_output_path_for_src_node(
                applied_task_identifier=applied_task_identifier,
                edge_type=edge_type,
            )
        )
        edge_type_dst_node_embedding_dir = (
            gcs_constants.get_embedding_output_path_for_dst_node(
                applied_task_identifier=applied_task_identifier,
                edge_type=edge_type,
            )
        )

        # Load src node embeddings to BigQuery.
        enum_src_node_embedding_table, unenum_src_node_embedding_table = (
            bq_constants.get_src_node_embedding_table_for_edge_type(
                applied_task_identifier=applied_task_identifier,
                edge_type=edge_type,
                is_enumerated=True,
            ),
            bq_constants.get_src_node_embedding_table_for_edge_type(
                applied_task_identifier=applied_task_identifier,
                edge_type=edge_type,
                is_enumerated=False,
            ),
        )
        project_id, dataset_id, table_id = BqUtils.parse_bq_table_path(
            enum_src_node_embedding_table
        )
        export.load_embeddings_to_bigquery(
            gcs_folder=edge_type_src_node_embedding_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id,
        )
        logger.info(
            f"Finished writing enumerated src node embeddings to BigQuery table `{enum_src_node_embedding_table}` for edge type {edge_type}."
        )
        unenumerate_embeddings_table(
            enumerated_embeddings_table=enum_src_node_embedding_table,
            embeddings_table_node_id_field=export._NODE_ID_KEY,
            unenumerated_embeddings_table=unenum_src_node_embedding_table,
            enumerator_mapping_table=node_type_to_enumerated_metadata_tables[
                edge_type.src_node_type
            ],
        )
        logger.info(
            f"Finished unenumerating src node embedings and wrote them to `{unenum_src_node_embedding_table}` using mapping `{node_type_to_enumerated_metadata_tables[edge_type.src_node_type]}`."
        )

        # Load dst node embeddings to BigQuery.
        enum_dst_node_embedding_table, unenum_dst_node_embedding_table = (
            bq_constants.get_dst_node_embedding_table_for_edge_type(
                applied_task_identifier=applied_task_identifier,
                edge_type=edge_type,
                is_enumerated=True,
            ),
            bq_constants.get_dst_node_embedding_table_for_edge_type(
                applied_task_identifier=applied_task_identifier,
                edge_type=edge_type,
                is_enumerated=False,
            ),
        )
        project_id, dataset_id, table_id = BqUtils.parse_bq_table_path(
            enum_dst_node_embedding_table
        )
        export.load_embeddings_to_bigquery(
            gcs_folder=edge_type_dst_node_embedding_dir,
            project_id=project_id,
            dataset_id=dataset_id,
            table_id=table_id,
        )
        logger.info(
            f"Finished writing enumerated dst node embeddings to BigQuery table `{enum_dst_node_embedding_table}` for edge type {edge_type}."
        )
        unenumerate_embeddings_table(
            enumerated_embeddings_table=enum_dst_node_embedding_table,
            embeddings_table_node_id_field=export._NODE_ID_KEY,
            unenumerated_embeddings_table=unenum_dst_node_embedding_table,
            enumerator_mapping_table=node_type_to_enumerated_metadata_tables[
                edge_type.dst_node_type
            ],
        )
        logger.info(
            f"Finished unenumerating dst node embeddings and wrote them to `{unenum_dst_node_embedding_table}` using mapping `{node_type_to_enumerated_metadata_tables[edge_type.dst_node_type]}`."
        )


def unenumerate_embeddings_table(
    enumerated_embeddings_table: str,
    embeddings_table_node_id_field: str,
    unenumerated_embeddings_table: str,
    enumerator_mapping_table: str,
):
    """Convert enumerated embeddings back to their original node IDs.

    This function transforms embeddings from an enumerated embeddings table to an
    unenumerated embeddings table by joining with a mapping table. The resulting
    table will have the original node IDs as keys and the embeddings as values.

    Args:
        enumerated_embeddings_table (str): The BigQuery table containing enumerated embeddings.
        embeddings_table_node_id_field (str): The field in the enumerated embeddings table
            that contains node IDs.
        unenumerated_embeddings_table (str): The destination BigQuery table for unenumerated
            embeddings.
        enumerator_mapping_table (str): The BigQuery table containing the mapping from
            enumerated to original node IDs.

    Returns:
        None: Results are written directly to the destination BigQuery table.
    """

    UNENUMERATION_QUERY = """
        SELECT
            mapping.{original_node_id_field},
            * EXCEPT({node_id_field}, {enumerated_int_id_field})
        FROM
            `{enumerated_assets_table}` enumerated_assets
        INNER JOIN
            `{mapping_table}` mapping
        ON
            mapping.int_id = enumerated_assets.{node_id_field}
        QUALIFY RANK() OVER (PARTITION BY mapping.{original_node_id_field} ORDER BY RAND()) = 1
    """

    bq_utils = BqUtils(project=get_resource_config().project)
    bq_utils.run_query(
        query=UNENUMERATION_QUERY.format(
            enumerated_assets_table=enumerated_embeddings_table,
            mapping_table=enumerator_mapping_table,
            node_id_field=embeddings_table_node_id_field,
            original_node_id_field=enumeration_queries.DEFAULT_ORIGINAL_NODE_ID_FIELD,
            enumerated_int_id_field=enumeration_queries.DEFAULT_ENUMERATED_NODE_ID_FIELD,
        ),
        labels=dict(),
        destination=unenumerated_embeddings_table,
        write_disposition=bigquery.job.WriteDisposition.WRITE_TRUNCATE,
    )
