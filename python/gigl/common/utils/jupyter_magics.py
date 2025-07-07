import os
import pathlib
from difflib import unified_diff
from typing import Optional, Type, Union

import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import torch_geometric.utils
import yaml
from IPython.display import HTML, display
from torch_geometric.data import HeteroData

from gigl.common import Uri
from snapchat.research.gbml import training_samples_schema_pb2

gigl_root_dir = pathlib.Path(__file__).parent.parent.parent.parent.parent


def change_working_dir_to_gigl_root():
    """
    Can be used inside notebooks to change the working directory to the GIGL root directory.
    This is useful for ensuring that relative imports and file paths work correctly no matter where the notebook is located.
    """
    os.chdir(gigl_root_dir)
    print(f"Changed working directory to: {gigl_root_dir}")


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
            node_color = GraphVisualizer.assign_color(str(node_id))
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


from gigl.src.common.utils.file_loader import FileLoader


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
