from dataclasses import dataclass
from typing import List, Optional

import yaml
from applied_tasks.knowledge_graph_embedding.lib.config.hydra_utils import (
    build_hydra_dict_from_object,
)
from omegaconf import DictConfig, OmegaConf

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
    node_data: List[BigqueryNodeDataReference]
    edge_data: List[BigqueryEdgeDataReference]


@dataclass
class EnumeratedGraphData:
    node_data: List[EnumeratorNodeTypeMetadata]
    edge_data: List[EnumeratorEdgeTypeMetadata]

    def generate_hydra_config_yaml(self) -> str:
        """
        Writes a Hydra-compatible YAML configuration file for enumerated graph data.
        Dynamically inserts '_target_' fields based on a mapping, handling dataclasses and namedtuples.

        Args:
            node_data: A list of EnumeratorNodeTypeMetadata instances.
            edge_data: A list of EnumeratorEdgeTypeMetadata instances.
            output_filepath: The path to the output YAML file.
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
        Writes a Hydra-compatible YAML configuration file for enumerated graph data.
        Dynamically inserts '_target_' fields based on a mapping, handling dataclasses and namedtuples.

        Args:
            node_data: A list of EnumeratorNodeTypeMetadata instances.
            edge_data: A list of EnumeratorEdgeTypeMetadata instances.
            output_filepath: The path to the output YAML file.
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
    metadata: GraphMetadataPbWrapper
    raw_graph_data: Optional[RawGraphData] = None
    enumerated_graph_data: Optional[EnumeratedGraphData] = None
