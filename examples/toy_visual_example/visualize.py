from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.utils
from torch_geometric.data import HeteroData

from snapchat.research.gbml import training_samples_schema_pb2


class GraphVisualizer:
    """
    Used to build and visualize graph which is user configured in a yaml file.
    """

    # Fixed color palette — extend as needed
    fixed_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # yellow-green
        "#17becf",  # teal
    ]

    @staticmethod
    def assign_color(name: str) -> str:
        """Assign a color to a name based on hash and a fixed palette."""
        return GraphVisualizer.fixed_colors[
            hash(name) % len(GraphVisualizer.fixed_colors)
        ]

    @staticmethod
    def visualize_graph(data: HeteroData):
        g = torch_geometric.utils.to_networkx(data)

        node_colors = [GraphVisualizer.assign_color(node) for node in g.nodes()]

        # Generate a static layout
        pos = nx.spring_layout(g, seed=42)
        for node in g.nodes():
            g.nodes[node]["label"] = node
        nx.draw(
            g,
            pos,
            with_labels=True,
            node_color=node_colors,
            node_size=500,
            font_size=10,
            font_weight="bold",
        )
        plt.show()

    @staticmethod
    def plot_graph(
        pb: Union[
            training_samples_schema_pb2.RootedNodeNeighborhood,
            training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
        ]
    ):
        """
        Visualize the graph from the protobuf message.
        """
        output_graph = nx.DiGraph()
        nodes = {}

        for node in pb.neighborhood.nodes:
            node_id = node.node_id
            nodes[node_id] = {
                "condensed_node_type": node.condensed_node_type,
                "feature_values": node.feature_values,
            }
            output_graph.add_node(node_id)

        all_edges = list(pb.neighborhood.edges)
        if hasattr(pb, "pos_edges") and pb.pos_edges:
            all_edges.extend(pb.pos_edges)
        if hasattr(pb, "neg_edges") and pb.neg_edges:
            all_edges.extend(pb.neg_edges)

        for edge in all_edges:
            src_node_id = edge.src_node_id
            dst_node_id = edge.dst_node_id
            condensed_edge_type = edge.condensed_edge_type

            output_graph.add_edge(
                src_node_id,
                dst_node_id,
                condensed_edge_type=edge.condensed_edge_type,
            )

        edge_colors = []
        edge_widths = []
        for edge in output_graph.edges():
            color: str = "black"
            edge_width = 1.0
            if hasattr(pb, "pos_edges") and pb.pos_edges:
                for pos_edge in pb.pos_edges:
                    if (
                        pos_edge.src_node_id == edge[0]
                        and pos_edge.dst_node_id == edge[1]
                        and pos_edge.condensed_edge_type
                        == output_graph[edge[0]][edge[1]]["condensed_edge_type"]
                    ):
                        color = "red"
                        edge_width = 2.0
                        break
            if hasattr(pb, "neg_edges") and pb.neg_edges:
                for neg_edge in pb.neg_edges:
                    if (
                        neg_edge.src_node_id == edge[0]
                        and neg_edge.dst_node_id == edge[1]
                        and neg_edge.condensed_edge_type
                        == output_graph[edge[0]][edge[1]]["condensed_edge_type"]
                    ):
                        color = "blue"
                        edge_width = 2.0
                        break
            edge_colors.append(color)
            edge_widths.append(edge_width)

        node_colors = []
        node_border_colors = []
        node_border_widths = []
        for node_id in output_graph.nodes():
            if node_id == pb.root_node.node_id:
                node_border_colors.append("black")
                node_border_widths.append(4)
            else:
                node_border_colors.append("lightgrey")
                node_border_widths.append(1)
            node_color = GraphVisualizer.assign_color(node_id)
            node_colors.append(node_color)

        plt.clf()
        pos = nx.spring_layout(output_graph, seed=42)
        nx.draw(
            output_graph,
            pos,
            with_labels=True,
            node_color=node_colors,
            edge_color=edge_colors,
            width=edge_widths,
            edgecolors=node_border_colors,
            linewidths=node_border_widths,
            node_size=500,
            font_size=10,
            font_weight="bold",
        )
        plt.show()
        return plt
