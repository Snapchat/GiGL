from typing import List

import torch
import yaml
from torch_geometric.data import HeteroData


# TODO: (svij) Investigate if we should consolidate this with creation of other mocked graphs, tests, etc.
def load_toy_graph(graph_config_path: str) -> HeteroData:
    with open(graph_config_path, "r") as f:
        graph_config: dict = yaml.safe_load(f)

    node_config = graph_config["graph"]["node_types"]
    edge_config = graph_config["graph"]["edge_types"]

    data = HeteroData()

    # Add node features
    for node_type in node_config:
        node_feats_list: List[str] = []
        for node in graph_config["nodes"][node_type]:
            features = node["features"]
            node_feats_list.append(features)
        data[node_type].x = torch.tensor(node_feats_list)

    # Add edge indices and edge features
    for edge_type in edge_config:
        src_type = edge_config[edge_type]["src_node_type"]
        dst_type = edge_config[edge_type]["dst_node_type"]
        rel_type = edge_config[edge_type]["relation_type"]

        edge_index_list = []
        for adj in graph_config["adj_list"][edge_type]:
            dst_list = adj["dst"]
            edge_index_list.extend([(adj["src"], dst) for dst in dst_list])
        edge_index = torch.tensor(edge_index_list).t().contiguous()
        data[(src_type, rel_type, dst_type)].edge_index = edge_index

        # Dummy edge features: edge_index.T * 0.1
        data[(src_type, rel_type, dst_type)].edge_attr = edge_index.t() * 0.1

    return data
