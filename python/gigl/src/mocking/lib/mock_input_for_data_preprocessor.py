import json
import tempfile
from typing import Optional

import torch
from google.cloud import bigquery

from gigl.common import UriFactory
from gigl.common.logger import Logger
from gigl.src.common.types.graph_data import EdgeType, EdgeUsageType, NodeType
from gigl.src.common.utils.bq import BqUtils
from gigl.src.mocking.lib.constants import (
    get_example_task_edges_bq_table_path,
    get_example_task_nodes_bq_table_path,
)
from gigl.src.mocking.lib.feature_handling import get_feature_field_name
from gigl.src.mocking.lib.mocked_dataset_resources import MockedDatasetInfo

logger = Logger()


def _generate_bigquery_assets_for_single_node_type(
    mocked_dataset_info: MockedDatasetInfo,
    version: str,
    node_type: NodeType,
    num_nodes: int,
    num_node_features: int,
    node_feats: torch.Tensor,
    node_labels: Optional[torch.Tensor],
):
    # Upload node features table
    tfh = tempfile.NamedTemporaryFile(delete=False, mode="w")

    node_feature_column_names = [
        get_feature_field_name(n=n) for n in range(num_node_features)
    ]
    node_ids = torch.arange(num_nodes).reshape(-1, 1)
    node_labels = node_labels.reshape(-1, 1) if node_labels is not None else None

    with open(tfh.name, "w") as f:
        for i in range(num_nodes):
            node_id = node_ids[i]
            node_feat = node_feats[i]
            node_label = node_labels[i] if node_labels is not None else None

            row = {mocked_dataset_info.node_id_column_name: node_id.item()}
            for i, column_name in enumerate(node_feature_column_names):
                row.update({column_name: node_feat[i].item()})

            if node_label:
                row.update(
                    {mocked_dataset_info.node_label_column_name: node_label.item()}
                )

            f.write(f"{json.dumps(row)}\n")

    node_features_schema = [
        bigquery.SchemaField(mocked_dataset_info.node_id_column_name, "INTEGER"),
    ] + [
        bigquery.SchemaField(column_name, "FLOAT")
        for column_name in node_feature_column_names
    ]
    if node_labels is not None:
        node_features_schema += [
            bigquery.SchemaField(mocked_dataset_info.node_label_column_name, "INTEGER")
        ]

    nodes_bq_table = get_example_task_nodes_bq_table_path(
        task_name=mocked_dataset_info.name, version=version, node_type=node_type
    )

    bq_utils = BqUtils()
    bq_utils.load_file_to_bq(
        source_path=UriFactory.create_uri(tfh.name),
        bq_path=nodes_bq_table,
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=node_features_schema,
        ),
        retry=True,
    )
    tfh.close()
    logger.info(
        f"{mocked_dataset_info.name} node data loaded to BQ table {nodes_bq_table}"
    )


def _generate_bigquery_assets_for_single_edge_type(
    mocked_dataset_info: MockedDatasetInfo,
    version: str,
    edge_type: EdgeType,
    edge_index: torch.Tensor,
    num_edge_features: int,
    edge_feats: Optional[torch.Tensor],
    edge_usage_type: EdgeUsageType,
):
    # Upload graph edges table
    tfh = tempfile.NamedTemporaryFile(delete=False, mode="w")
    edge_feature_column_names = [
        get_feature_field_name(n=n) for n in range(num_edge_features)
    ]
    with open(tfh.name, "w") as f:
        for i, (src, dst) in enumerate(edge_index.T):
            row = {
                mocked_dataset_info.edge_src_column_name: src.item(),
                mocked_dataset_info.edge_dst_column_name: dst.item(),
            }

            if edge_feats is not None:
                edge_feat = edge_feats[i]
                for column_name, edge_feature in zip(
                    edge_feature_column_names, edge_feat
                ):
                    row.update({column_name: edge_feature.item()})

            f.write(f"{json.dumps(row)}\n")

    edge_features_schema = [
        bigquery.SchemaField(mocked_dataset_info.edge_src_column_name, "INTEGER"),
        bigquery.SchemaField(mocked_dataset_info.edge_dst_column_name, "INTEGER"),
    ]
    if edge_feats is not None:
        edge_features_schema += [
            bigquery.SchemaField(column_name, "FLOAT")
            for column_name in edge_feature_column_names
        ]

    edges_bq_table = get_example_task_edges_bq_table_path(
        task_name=mocked_dataset_info.name,
        version=version,
        edge_type=edge_type,
        edge_usage_type=edge_usage_type,
    )

    bq_utils = BqUtils()
    bq_utils.load_file_to_bq(
        source_path=UriFactory.create_uri(tfh.name),
        bq_path=edges_bq_table,
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema=edge_features_schema,
        ),
        retry=True,
    )
    tfh.close()
    logger.info(
        f"{mocked_dataset_info.name} edge data loaded to BQ table {edges_bq_table}"
    )


def generate_bigquery_assets(mocked_dataset_info: MockedDatasetInfo, version: str):
    """
    This generates a BQ table for each node type and edge type that exist in
    designated input.
    """

    node_types = mocked_dataset_info.node_types
    num_nodes_per_type = mocked_dataset_info.num_nodes
    num_node_features_per_type = mocked_dataset_info.num_node_features

    for node_type in node_types:
        num_nodes = num_nodes_per_type[node_type]
        num_node_features = num_node_features_per_type[node_type]
        node_feats = mocked_dataset_info.node_feats[node_type]
        node_labels = (
            None
            if mocked_dataset_info.node_labels is None
            else mocked_dataset_info.node_labels[node_type]
        )

        _generate_bigquery_assets_for_single_node_type(
            mocked_dataset_info=mocked_dataset_info,
            version=version,
            node_type=node_type,
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            node_feats=node_feats,
            node_labels=node_labels,
        )

    edge_types = mocked_dataset_info.edge_types
    edge_index_per_type = mocked_dataset_info.edge_index
    num_edge_features_per_type = mocked_dataset_info.num_edge_features
    for edge_type in edge_types:
        edge_index = edge_index_per_type[edge_type]
        num_edge_features = num_edge_features_per_type[edge_type]
        edge_feats = (
            mocked_dataset_info.edge_feats[edge_type]
            if mocked_dataset_info.edge_feats
            else None
        )
        _generate_bigquery_assets_for_single_edge_type(
            mocked_dataset_info=mocked_dataset_info,
            version=version,
            edge_type=edge_type,
            edge_index=edge_index,
            num_edge_features=num_edge_features,
            edge_feats=edge_feats,
            edge_usage_type=EdgeUsageType.MAIN,
        )

        if (
            mocked_dataset_info.user_defined_edge_index
            and edge_type in mocked_dataset_info.user_defined_edge_index
        ):
            for (
                edge_usage_type,
                user_def_edge_index,
            ) in mocked_dataset_info.user_defined_edge_index[edge_type].items():
                user_defined_edge_feats = (
                    mocked_dataset_info.user_defined_edge_feats[edge_type][
                        edge_usage_type
                    ]
                    if mocked_dataset_info.user_defined_edge_feats
                    and edge_type in mocked_dataset_info.user_defined_edge_feats
                    else None
                )
                num_user_def_edge_features = (
                    mocked_dataset_info.num_user_def_edge_features[edge_type][
                        edge_usage_type
                    ]
                )

                _generate_bigquery_assets_for_single_edge_type(
                    mocked_dataset_info=mocked_dataset_info,
                    version=version,
                    edge_type=edge_type,
                    edge_index=user_def_edge_index,
                    num_edge_features=num_user_def_edge_features,
                    edge_feats=user_defined_edge_feats,
                    edge_usage_type=edge_usage_type,
                )
