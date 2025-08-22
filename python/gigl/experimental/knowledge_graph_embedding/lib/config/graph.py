from dataclasses import dataclass
from typing import List, Optional

import yaml
from omegaconf import DictConfig, OmegaConf

from gigl.experimental.knowledge_graph_embedding.lib.config.hydra_utils import (
    build_hydra_dict_from_object,
)
from gigl.src.common.types.pb_wrappers.graph_metadata import GraphMetadataPbWrapper
from gigl.src.data_preprocessor.lib.enumerate.utils import (
    EnumeratorEdgeTypeMetadata,
    EnumeratorNodeTypeMetadata,
)
from gigl.src.data_preprocessor.lib.ingest.bigquery import (
    BigqueryEdgeDataReference,
    BigqueryNodeDataReference,
)


@dataclass
class RawGraphData:
    """
    Configuration for raw graph data references from BigQuery sources.

    Raw graph data refers to the original, unprocessed node and edge data stored
    in BigQuery tables before enumeration and preprocessing for model training.

    Attributes:
        node_data (List[BigqueryNodeDataReference]): List of BigQuery data references for node data tables.
            Each reference specifies the location and schema of node information.
        edge_data (List[BigqueryEdgeDataReference]): List of BigQuery data references for edge data tables.
            Each reference specifies the location and schema of relationship information.
    """

    node_data: List[BigqueryNodeDataReference]
    edge_data: List[BigqueryEdgeDataReference]


@dataclass
class EnumeratedGraphData:
    """
    Configuration for enumerated graph data after preprocessing.

    Enumerated graph data refers to preprocessed node and edge data where node and edge
    identifiers have been mapped to integer IDs, making them suitable for embedding lookups
    into tables during model training.

    Attributes:
        node_data (List[EnumeratorNodeTypeMetadata]): List of metadata for enumerated node types, containing mapping
            information from raw node IDs to integer IDs.
        edge_data (List[EnumeratorEdgeTypeMetadata]): List of metadata for enumerated edge types, containing mapping
            information from raw node ID-based edges to corresponding integer ID-based edges.
    """

    node_data: List[EnumeratorNodeTypeMetadata]
    edge_data: List[EnumeratorEdgeTypeMetadata]

    def generate_hydra_config_yaml(self) -> str:
        """
        Generate a Hydra-compatible YAML configuration string for enumerated graph data.

        Converts the enumerated node and edge data into a YAML format that can be used
        by Hydra for configuration management. Dynamically inserts '_target_' fields
        based on object types, handling dataclasses and namedtuples.

        Returns:
            str: A YAML-formatted string containing the Hydra configuration for
                enumerated graph data with proper '_target_' fields for instantiation.
        """
        hydra_node_data = [
            build_hydra_dict_from_object(node_entry) for node_entry in self.node_data
        ]
        hydra_edge_data = [
            build_hydra_dict_from_object(edge_entry) for edge_entry in self.edge_data
        ]

        config_dict = {
            "enumerated_graph_data": {
                "node_data": hydra_node_data,
                "edge_data": hydra_edge_data,
            }
        }

        yaml_string = yaml.safe_dump(
            config_dict, sort_keys=False, default_flow_style=False, indent=2
        )
        return yaml_string

    def to_dictconfig(self) -> DictConfig:
        """
        Convert enumerated graph data to an OmegaConf DictConfig object.

        Creates a Hydra-compatible configuration object from the enumerated node and edge data.
        This is useful for programmatic configuration management without writing to files.
        Dynamically inserts '_target_' fields based on object types.

        Returns:
            DictConfig: An OmegaConf DictConfig object containing the enumerated graph
                data configuration with proper '_target_' fields for Hydra instantiation.
        """
        hydra_node_data = [
            build_hydra_dict_from_object(node_entry) for node_entry in self.node_data
        ]
        hydra_edge_data = [
            build_hydra_dict_from_object(edge_entry) for edge_entry in self.edge_data
        ]

        config_dict = {
            "enumerated_graph_data": {
                "node_data": hydra_node_data,
                "edge_data": hydra_edge_data,
            }
        }

        # Directly create an OmegaConf DictConfig from the Python dictionary
        return OmegaConf.create(config_dict)


@dataclass
class GraphConfig:
    """
    Main graph configuration containing metadata and data references.

    This configuration encapsulates all information about the knowledge graph structure,
    including schema metadata and references to both raw and processed data sources.

    Attributes:
        metadata (GraphMetadataPbWrapper): Graph metadata wrapper containing schema information (node types,
            edge types, feature schemas) wrapped in a protocol buffer format.
        raw_graph_data (Optional[RawGraphData]): Optional reference to raw BigQuery data sources.
            Used during initial data ingestion and preprocessing. None if not applicable.
        enumerated_graph_data (Optional[EnumeratedGraphData]): Optional reference to preprocessed enumerated data.
            Used during model training when data has been preprocessed into integer IDs.
            None if not applicable.
    """

    metadata: GraphMetadataPbWrapper
    raw_graph_data: Optional[RawGraphData] = None
    enumerated_graph_data: Optional[EnumeratedGraphData] = None
