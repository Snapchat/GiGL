import argparse

import matplotlib.pyplot as plt
import networkx as nx
import yaml

"""
Usage:
( cd python && \
python -m tests.test_assets.dataset_mocking.visualization_test.visualize \
--config_path="./tests/test_assets/dataset_mocking/visualization_test/graph_config.yaml" \
&& cd ..;  )
"""


class GraphVisualizer:
    """
    Used to build and visualize graph which is user configured in a yaml file.
    """

    def __init__(self, graph_config_path: str):
        self._load_graph_config(graph_config_path)

    # load the graph config yaml file
    def _load_graph_config(self, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.node_type = config["graph"]["node_type"]
        self.adj_list = config["adj_list"]
        self.node_list = config["nodes"]

    # Use graph config params to build DiGraph which is to be visualized
    def _build_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()

        for node in self.node_list:
            graph.add_node(node["src"], label=node["src"])

        for edge in self.adj_list:
            print(edge)
            for dst_node in edge["dst"]:
                graph.add_edge(edge["src"], dst_node)

        return graph

    # Plot the graph nodes, node features, and edges
    # TODO: Adjustable params (arrowstyle, size, labels, etc.)
    def visualize_graph(self):
        graph = self._build_graph()

        pos = nx.spring_layout(graph)

        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_labels(graph, pos)

        # Draw edges
        nx.draw_networkx_edges(graph, pos, edge_color="gray")

        # Draw edge labels
        edge_labels = {(u, v): "" for u, v, w in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, pos, edge_labels)

        # Draw node features
        node_labels = {node["src"]: node["features"] for node in self.node_list}
        label_pos = {k: (v[0], v[1] - 0.05) for k, v in pos.items()}
        nx.draw_networkx_labels(graph, label_pos, labels=node_labels, font_size=8)

        plt.savefig("graph_vis.png", dpi=300)
        # plt.show(dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a graph from a YAML configuration file"
    )
    parser.add_argument(
        "--config_path", type=str, help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    graph_visualizer = GraphVisualizer(args.config_path)
    graph_visualizer.visualize_graph()
