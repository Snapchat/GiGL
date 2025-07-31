import math
import os
import pathlib
from difflib import unified_diff
from enum import Enum
from typing import Optional, Type, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import tensorflow as tf
import torch_geometric.utils
import yaml
from IPython.display import HTML, display
from torch_geometric.data import HeteroData

from gigl.common import Uri
from gigl.src.common.utils.file_loader import FileLoader
from snapchat.research.gbml import training_samples_schema_pb2

gigl_root_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent


def change_working_dir_to_gigl_root():
    """
    Can be used inside notebooks to change the working directory to the GIGL root directory.
    This is useful for ensuring that relative imports and file paths work correctly no matter where the notebook is located.
    """
    os.chdir(gigl_root_dir)
    print(f"Changed working directory to: {gigl_root_dir}")


CHARCOAL = "#36454F"
BLACK = "#000000"

class GraphVisualizerLayoutMode(Enum):
    HOMOGENEOUS = "homogeneous"
    BIPARTITE = "bipartite"


class GraphVisualizer:
    """
    Used to build and visualize graph which is user configured in a yaml file.
    """

    # Fixed node color palette — extend as needed
    node_colors = [
        "#64B5F6",  # blue
        "#E57373",  # red
        "#81C784",  # green
        "#FFD54F",  # yellow
        "#BA68C8",  # purple
        "#4DB6AC",  # teal
        "#F06292",  # pink
        "#A1887F",  # brown
        "#FFB74D",  # orange
    ]

    # Fixed edge color palette — best for white background
    edge_colors = [
        "#1565C0",  # medium blue
        "#43A047",  # vivid green
        "#E53935",  # vivid red
    ]

    @staticmethod
    def assign_node_color(name: str) -> str:
        """Assign a node color to a name based on hash and a fixed palette."""
        return GraphVisualizer.node_colors[
            hash(name) % len(GraphVisualizer.node_colors)
        ]

    @staticmethod
    def assign_edge_color(name: str) -> str:
        """Assign an edge color to a name based on hash and a fixed palette (optimized for white background)."""
        return GraphVisualizer.edge_colors[
            hash(name) % len(GraphVisualizer.edge_colors)
        ]

    @staticmethod
    def _create_type_grouped_layout(g, node_index_to_type, node_types, seed=42, layout_mode=GraphVisualizerLayoutMode.BIPARTITE):
        """Create a layout based on the specified mode (bipartite or homogeneous)."""

        if layout_mode == GraphVisualizerLayoutMode.HOMOGENEOUS:
            print("Using homogeneous layout")
            # For homogeneous graphs, use layouts that work well for general graph structure
            num_nodes = len(g.nodes())

            if num_nodes <= 30:
                # Small to medium graphs - use Kamada-Kawai (good for showing structure)
                try:
                    return nx.kamada_kawai_layout(g, scale=6)
                except:
                    # Fallback to spring layout if kamada_kawai fails
                    k = max(2.5, num_nodes / 8.0)
                    return nx.spring_layout(g, seed=seed, k=k, iterations=200, scale=8)
            else:
                # Large graphs - use spring layout with good parameters
                k = max(2.0, num_nodes / 10.0)
                return nx.spring_layout(g, seed=seed, k=k, iterations=150, scale=10)

        elif layout_mode == GraphVisualizerLayoutMode.BIPARTITE:
            # Group nodes by their types for bipartite/heterogeneous layout
            type_to_nodes = {}
            for node in g.nodes():
                node_type = node_index_to_type.get(node, "unknown")
                if node_type not in type_to_nodes:
                    type_to_nodes[node_type] = []
                type_to_nodes[node_type].append(node)

            num_types = len(type_to_nodes)

            if num_types == 1:
                # Single type - use circular layout with more spacing
                return nx.circular_layout(g, scale=6)
            elif num_types == 2:
                # Two types - use bipartite layout with more spacing
                types = list(type_to_nodes.keys())
                first_type_nodes = set(type_to_nodes[types[0]])
                return nx.bipartite_layout(g, first_type_nodes, scale=6)
            else:
                # Multiple types or fallback - use spring layout with much more spacing
                k = max(3.0, len(g.nodes()) / 5.0)  # Dynamic spacing based on node count
                return nx.spring_layout(g, seed=seed, k=k, iterations=200, scale=8)

    @staticmethod
    def visualize_graph(data: HeteroData, seed=42, layout_mode=GraphVisualizerLayoutMode.BIPARTITE):
        # Build a mapping from global node indices to node types BEFORE conversion
        node_index_to_type = {}
        current_index = 0

        # HeteroData stores nodes by type - we need to map the global indices
        # that NetworkX will use back to the original node types
        for node_type in data.node_types:
            if hasattr(data[node_type], 'num_nodes'):
                num_nodes = data[node_type].num_nodes
                for i in range(num_nodes):
                    node_index_to_type[current_index] = node_type
                    current_index += 1

        # Convert to NetworkX
        g = torch_geometric.utils.to_networkx(data)

        # Create node type to color mapping
        node_type_to_color = {}
        for node_type in data.node_types:
            node_type_to_color[node_type] = GraphVisualizer.assign_node_color(node_type)

        # Assign colors based on the mapping we built
        node_colors = []

        for node in g.nodes():
            node_type = node_index_to_type.get(node, "unknown")

            # Get color for this node type
            if node_type not in node_type_to_color:
                node_type_to_color[node_type] = GraphVisualizer.assign_node_color(node_type)

            node_colors.append(node_type_to_color[node_type])

        # Create a larger figure for better node spacing
        plt.figure(figsize=(10, 6))

        # Generate a layout based on the selected mode
        pos = GraphVisualizer._create_type_grouped_layout(g, node_index_to_type, data.node_types, seed, layout_mode)

        # Identify isolated nodes for special border styling
        isolated_nodes = [node for node in g.nodes() if g.degree(node) == 0]

        # Create border styling (thicker black border for isolated nodes)
        node_edge_colors = [BLACK if node in isolated_nodes else CHARCOAL for node in g.nodes()]
        node_line_widths = [3 if node in isolated_nodes else 1 for node in g.nodes()]

        # Create edge type to color mapping
        edge_type_to_color = {}
        edge_colors = []

        # Extract edge types from the original HeteroData
        for edge in g.edges():
            # Get node types for source and destination
            src_node_type = node_index_to_type.get(edge[0], "unknown")
            dst_node_type = node_index_to_type.get(edge[1], "unknown")

            # Create edge type identifier
            edge_type = f"{src_node_type} → {dst_node_type}"

            # Look for a more specific edge type in HeteroData if available
            if hasattr(data, 'edge_types') and data.edge_types:
                for et in data.edge_types:
                    if len(et) == 3:  # (src_type, relation, dst_type)
                        if et[0] == src_node_type and et[2] == dst_node_type:
                            edge_type = f"{et[0]} --{et[1]}--> {et[2]}"
                            break
                    elif isinstance(et, tuple) and len(et) == 2:  # Some formats might be (src, dst)
                        if et[0] == src_node_type and et[1] == dst_node_type:
                            edge_type = f"{et[0]} → {et[1]}"
                            break

            # Assign color to edge type
            if edge_type not in edge_type_to_color:
                edge_type_to_color[edge_type] = GraphVisualizer.assign_edge_color(edge_type)

            edge_colors.append(edge_type_to_color[edge_type])

        # Draw nodes first
        nx.draw_networkx_nodes(
            g,
            pos,
            node_color=node_colors if node_colors else 'lightblue',  # type: ignore
            edgecolors=node_edge_colors if node_edge_colors else CHARCOAL,  # type: ignore
            linewidths=node_line_widths if node_line_widths else 1,  # type: ignore
            node_size=500,
        )

        # Draw edges - straight for homogeneous, curved for bipartite
        if g.edges() and edge_colors:
            if layout_mode == GraphVisualizerLayoutMode.HOMOGENEOUS:
                # Straight edges for homogeneous graphs
                nx.draw_networkx_edges(
                    g,
                    pos,
                    edge_color=edge_colors,  # type: ignore
                    width=0.75,  # 75% of default edge width
                    alpha=0.9,   # Less transparent for cleaner look
                )
            else:
                # Curved edges for bipartite graphs to reduce overlap
                nx.draw_networkx_edges(
                    g,
                    pos,
                    edge_color=edge_colors,  # type: ignore
                    width=0.75,  # 75% of default edge width
                    alpha=0.8,   # Slightly transparent for better overlap visibility
                    connectionstyle="arc3,rad=0.1",  # Curved edges to reduce overlap
                )

        # Draw labels last so they appear on top
        nx.draw_networkx_labels(
            g,
            pos,
            font_size=10,
            font_weight="bold",
        )

        # Add a legend to show node type colors and edge types
        legend_elements = []

        # Add node types
        if len(node_type_to_color) > 1:
            for node_type in sorted(node_type_to_color.keys()):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                 markerfacecolor=node_type_to_color[node_type],
                                                 markersize=10, label=f'Node: {node_type}'))

        # Add isolated node indicator
        if isolated_nodes:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='black',
                                             markerfacecolor='white', markeredgewidth=3,
                                             markersize=10, label='Isolated nodes'))

        # Add edge types
        if edge_type_to_color:
            for edge_type in sorted(edge_type_to_color.keys()):
                legend_elements.append(plt.Line2D([0], [0], color=edge_type_to_color[edge_type],
                                                 linewidth=2, label=f'Edge: {edge_type}'))

        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.4, 1))

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
        for output_edge in output_graph.edges():
            color: str = "black"
            edge_width = 1.0
            if hasattr(pb, "pos_edges") and pb.pos_edges:
                for pos_edge in pb.pos_edges:
                    if (
                        pos_edge.src_node_id == output_edge[0]
                        and pos_edge.dst_node_id == output_edge[1]
                        and pos_edge.condensed_edge_type
                        == output_graph[output_edge[0]][output_edge[1]][
                            "condensed_edge_type"
                        ]
                    ):
                        color = "red"
                        edge_width = 2.0
                        break
            if hasattr(pb, "neg_edges") and pb.neg_edges:
                for neg_edge in pb.neg_edges:
                    if (
                        neg_edge.src_node_id == output_edge[0]
                        and neg_edge.dst_node_id == output_edge[1]
                        and neg_edge.condensed_edge_type
                        == output_graph[output_edge[0]][output_edge[1]][
                            "condensed_edge_type"
                        ]
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
            node_color = GraphVisualizer.assign_node_color(str(node_id))
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


def find_node_pb(
    tfrecord_uri_prefix: str,
    node_id: int,
    pb_type: Type[
        Union[
            training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
            training_samples_schema_pb2.RootedNodeNeighborhood,
        ]
    ],
) -> Optional[
    Union[
        training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
        training_samples_schema_pb2.RootedNodeNeighborhood,
    ]
]:
    uri = tfrecord_uri_prefix + "*.tfrecord"
    ds = tf.data.TFRecordDataset(tf.io.gfile.glob(uri)).as_numpy_iterator()
    pb: Optional[
        Union[
            training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
            training_samples_schema_pb2.RootedNodeNeighborhood,
        ]
    ] = None
    print(f"Searching for node {node_id} in {uri}")
    for bytestr in ds:
        try:
            if pb_type == training_samples_schema_pb2.RootedNodeNeighborhood:
                pb = training_samples_schema_pb2.RootedNodeNeighborhood()
            elif (
                pb_type
                == training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample
            ):
                pb = training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample()
            else:
                raise ValueError(f"Unsupported pb_type: {pb_type}")
            pb.ParseFromString(bytestr)
            if pb.root_node.node_id == node_id:
                break
        except StopIteration:
            break
    return pb


def sort_yaml_dict_recursively(obj: dict) -> dict:
    # We sort the json recursively as the GiGL proto serialization code does not guarantee order of original keys.
    # This is important for the diff to be stable and not show errors due to key/list order changes.
    if isinstance(obj, dict):
        return {k: sort_yaml_dict_recursively(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return [sort_yaml_dict_recursively(item) for item in obj]
    else:
        return obj


def show_colored_unified_diff(f1_lines, f2_lines, f1_name, f2_name):
    diff_lines = list(
        unified_diff(f1_lines, f2_lines, fromfile=f2_name, tofile=f1_name)
    )
    html_lines = []
    for line in diff_lines:
        if line.startswith("+") and not line.startswith("+++"):
            color = "#228B22"  # green
        elif line.startswith("-") and not line.startswith("---"):
            color = "#B22222"  # red
        elif line.startswith("@"):
            color = "#1E90FF"  # blue
        else:
            color = "#000000"  # black
        html_lines.append(
            f'<pre style="margin:0; color:{color}; background-color:white;">{line.rstrip()}</pre>'
        )
    display(HTML("".join(html_lines)))


def show_task_config_colored_unified_diff(
    f1_uri: Uri, f2_uri: Uri, f1_name: str, f2_name: str
):
    """
    Displays a colored unified diff of two task config files.
    Args:
        f1_uri (Uri): URI of the first file.
        f2_uri (Uri): URI of the second file.
    """
    file_loader = FileLoader()
    frozen_task_config_file_contents: str
    template_task_config_file_contents: str

    with open(file_loader.load_to_temp_file(file_uri_src=f1_uri).name, "r") as f:
        data = yaml.safe_load(f)
        # sort_keys by default
        frozen_task_config_file_contents = yaml.dump(sort_yaml_dict_recursively(data))

    with open(file_loader.load_to_temp_file(file_uri_src=f2_uri).name, "r") as f:
        data = yaml.safe_load(f)
        template_task_config_file_contents = yaml.dump(sort_yaml_dict_recursively(data))

    show_colored_unified_diff(
        template_task_config_file_contents.splitlines(),
        frozen_task_config_file_contents.splitlines(),
        f1_name=f1_name,
        f2_name=f2_name,
    )
