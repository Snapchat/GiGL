# type: ignore
import hashlib
import os
import pathlib
from collections import defaultdict
from difflib import unified_diff
from enum import Enum
from typing import Optional, Type, Union

import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf
import torch_geometric.utils
import yaml
from gigl.common import Uri
from gigl.common.collections.frozen_dict import FrozenDict
from gigl.src.common.graph_builder.pyg_graph_builder import PygGraphBuilder
from gigl.src.common.translators.gbml_protos_translator import GbmlProtosTranslator
from gigl.src.common.types.graph_data import CondensedNodeType, EdgeType, Node, NodeType
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.utils.file_loader import FileLoader
from IPython.display import HTML, display
from snapchat.research.gbml import training_samples_schema_pb2
from torch_geometric.data import HeteroData

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


class PbVisualizerFromOutput(Enum):
    SGS = "sgs"
    SPLIT_TRAIN = "split_train"
    SPLIT_VAL = "split_val"
    SPLIT_TEST = "split_test"


class PbVisualizer:
    def __init__(self, frozen_task_config: GbmlConfigPbWrapper):
        self.frozen_task_config = frozen_task_config
        preprocessed_metadata = (
            frozen_task_config.preprocessed_metadata_pb_wrapper.preprocessed_metadata_pb
        )
        graph_metadata_pb_wrapper = frozen_task_config.graph_metadata_pb_wrapper

        from gigl.src.common.utils.bq import BqUtils

        bq_utils = BqUtils()

        # dict[tuple[condensed_node_type, enumerated_node_id], tuple[node_type, unenumerated_node_id]]
        self.enumerated_node_to_unenumerated_node_id_map: dict[
            tuple[int, int], tuple[str, int]
        ] = {}

        for (
            condensed_node_type,
            node_metadata,
        ) in preprocessed_metadata.condensed_node_type_to_preprocessed_metadata.items():
            node_type = graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                CondensedNodeType(condensed_node_type)
            ]
            result = bq_utils.run_query(
                query=f"""
                SELECT int_id, node_id FROM `{node_metadata.enumerated_node_ids_bq_table}`
                """,
                labels={},
            )
            for row in result:
                self.enumerated_node_to_unenumerated_node_id_map[
                    (condensed_node_type, row.int_id)
                ] = (node_type, row.node_id)

        # dict[tuple[node_type, unenumerated_node_id], tuple[condensed_node_type, enumerated_node_id]]
        self.unenumerated_node_id_to_enumerated_node_id_map: dict[
            tuple[str, int], tuple[int, int]
        ] = {}
        for (condensed_node_type, int_id), (
            node_type,
            node_id,
        ) in self.enumerated_node_to_unenumerated_node_id_map.items():
            self.unenumerated_node_id_to_enumerated_node_id_map[
                (node_type, node_id)
            ] = (CondensedNodeType(condensed_node_type), int_id)

    def plot_pb(
        self,
        pb: Union[
            training_samples_schema_pb2.RootedNodeNeighborhood,
            training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
        ],
        layout_mode: GraphVisualizerLayoutMode = GraphVisualizerLayoutMode.BIPARTITE,
    ):
        if not pb:
            print("No pb to plot")
            return
        builder = PygGraphBuilder()
        graph_metadata_pb_wrapper = self.frozen_task_config.graph_metadata_pb_wrapper
        graph_data = GbmlProtosTranslator.graph_data_from_GraphPb(
            samples=[pb.neighborhood],
            graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
            builder=builder,
        )

        # Extract positive edges if this is a NodeAnchorBasedLinkPredictionSample
        pos_edges: Optional[dict[tuple[str, str, str], list[tuple[int, int]]]] = None
        global_root_node: Optional[tuple[int, str]] = None
        if isinstance(
            pb, training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample
        ):
            pos_edges = defaultdict(list)
            for edge in pb.pos_edges:
                edge_type: EdgeType = (
                    graph_metadata_pb_wrapper.condensed_edge_type_to_edge_type_map[
                        edge.condensed_edge_type
                    ]
                )
                pos_edges[
                    (
                        edge_type.src_node_type,
                        edge_type.relation,
                        edge_type.dst_node_type,
                    )
                ].append((edge.src_node_id, edge.dst_node_id))
            global_root_node = (
                pb.root_node.node_id,
                graph_metadata_pb_wrapper.condensed_node_type_to_node_type_map[
                    pb.root_node.condensed_node_type
                ],
            )

        subgraph_node_to_unenumerated_node_id_map: dict[Node, Node] = {}
        for (
            node,
            global_node,
        ) in graph_data.subgraph_node_to_global_node_mapping.items():
            condensed_node_type = (
                graph_metadata_pb_wrapper.node_type_to_condensed_node_type_map[
                    global_node.type
                ]
            )
            (
                unenumerated_node_type,
                unenumerated_node_id,
            ) = self.enumerated_node_to_unenumerated_node_id_map[
                (condensed_node_type, global_node.id)
            ]
            subgraph_node_to_unenumerated_node_id_map[node] = Node(
                id=unenumerated_node_id,
                type=NodeType(unenumerated_node_type),
            )

        return GraphVisualizer.visualize_graph(
            data=graph_data.to_hetero_data(),
            layout_mode=layout_mode,
            subgraph_node_to_unenumerated_node_id_map=subgraph_node_to_unenumerated_node_id_map,
            pos_edges=pos_edges,
            global_root_node=global_root_node,
        )

    def find_node_pb(
        self,
        unenumerated_node_id: int,
        unenumerated_node_type: str,
        from_output: PbVisualizerFromOutput,
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
        tfrecord_uri_prefix: str
        if from_output == PbVisualizerFromOutput.SGS:
            flattened_graph_metadata = (
                self.frozen_task_config.shared_config.flattened_graph_metadata
            )
            assert hasattr(
                flattened_graph_metadata, "node_anchor_based_link_prediction_output"
            ), f"find_node_pb only supported for node_anchor_based_link_prediction, not {flattened_graph_metadata}"
            if (
                pb_type
                == training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample
            ):
                tfrecord_uri_prefix = (
                    flattened_graph_metadata.node_anchor_based_link_prediction_output.tfrecord_uri_prefix
                )
            elif pb_type == training_samples_schema_pb2.RootedNodeNeighborhood:
                tfrecord_uri_prefix = flattened_graph_metadata.node_anchor_based_link_prediction_output.node_type_to_random_negative_tfrecord_uri_prefix[
                    unenumerated_node_type
                ]
            else:
                raise ValueError(f"Unsupported pb_type: {pb_type}")
        else:
            assert hasattr(
                self.frozen_task_config.shared_config.dataset_metadata,
                "node_anchor_based_link_prediction_dataset",
            ), f"find_node_pb only supported for node_anchor_based_link_prediction, not {self.frozen_task_config.shared_config.dataset_metadata}"
            dataset = (
                self.frozen_task_config.shared_config.dataset_metadata.node_anchor_based_link_prediction_dataset
            )
            if (
                pb_type
                == training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample
            ):
                if from_output == PbVisualizerFromOutput.SPLIT_TRAIN:
                    tfrecord_uri_prefix = dataset.train_main_data_uri
                elif from_output == PbVisualizerFromOutput.SPLIT_VAL:
                    tfrecord_uri_prefix = dataset.val_main_data_uri
                elif from_output == PbVisualizerFromOutput.SPLIT_TEST:
                    tfrecord_uri_prefix = dataset.test_main_data_uri
                else:
                    raise ValueError(f"Unsupported from_output: {from_output}")
            elif pb_type == training_samples_schema_pb2.RootedNodeNeighborhood:
                if from_output == PbVisualizerFromOutput.SPLIT_TRAIN:
                    tfrecord_uri_prefix = (
                        dataset.train_node_type_to_random_negative_data_uri[
                            unenumerated_node_type
                        ]
                    )
                elif from_output == PbVisualizerFromOutput.SPLIT_VAL:
                    tfrecord_uri_prefix = (
                        dataset.val_node_type_to_random_negative_data_uri[
                            unenumerated_node_type
                        ]
                    )
                elif from_output == PbVisualizerFromOutput.SPLIT_TEST:
                    tfrecord_uri_prefix = (
                        dataset.test_node_type_to_random_negative_data_uri[
                            unenumerated_node_type
                        ]
                    )
                else:
                    raise ValueError(f"Unsupported from_output: {from_output}")
            else:
                raise ValueError(f"Unsupported pb_type: {pb_type}")
        uri = tfrecord_uri_prefix + "*.tfrecord"

        (
            search_node_type,
            search_node_id,
        ) = self.unenumerated_node_id_to_enumerated_node_id_map[
            (unenumerated_node_type, unenumerated_node_id)
        ]
        print(
            f"The node id {unenumerated_node_id}, type {unenumerated_node_type} maps to node id {search_node_id}, type {search_node_type}"
        )

        ds = tf.data.TFRecordDataset(tf.io.gfile.glob(uri)).as_numpy_iterator()
        pb: Optional[
            Union[
                training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
                training_samples_schema_pb2.RootedNodeNeighborhood,
            ]
        ] = None
        print(f" Looking for node {search_node_id} in {uri}")
        pb_output: Optional[
            Union[
                training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample,
                training_samples_schema_pb2.RootedNodeNeighborhood,
            ]
        ] = None
        for bytestr in ds:
            try:
                if pb_type == training_samples_schema_pb2.RootedNodeNeighborhood:
                    pb = training_samples_schema_pb2.RootedNodeNeighborhood()
                elif (
                    pb_type
                    == training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample
                ):
                    pb = (
                        training_samples_schema_pb2.NodeAnchorBasedLinkPredictionSample()
                    )
                else:
                    raise ValueError(f"Unsupported pb_type: {pb_type}")
                pb.ParseFromString(bytestr)
                if pb.root_node.node_id == search_node_id:
                    pb_output = pb
                    break
            except StopIteration:
                break

        return pb_output


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
        "#000000",  # black
    ]

    @staticmethod
    def assign_node_color(name: str) -> str:
        """Assign a node color to a name based on deterministic hash and a fixed palette."""
        # Use SHA256 for deterministic hashing
        hash_value = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16)
        return GraphVisualizer.node_colors[
            hash_value % len(GraphVisualizer.node_colors)
        ]

    @staticmethod
    def assign_edge_color(name: str) -> str:
        """Assign an edge color to a name based on deterministic hash and a fixed palette (optimized for white background)."""
        # Use SHA256 for deterministic hashing
        hash_value = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16)
        return GraphVisualizer.edge_colors[
            hash_value % len(GraphVisualizer.edge_colors)
        ]

    @staticmethod
    def _create_type_grouped_layout(
        g,
        node_index_to_type,
        node_types,
        seed=42,
        layout_mode=GraphVisualizerLayoutMode.BIPARTITE,
    ):
        """
        Warning: This is mostly just AI slop, but it serves the purpose for now.
        Create a layout based on the specified mode (bipartite or homogeneous).
        """

        # Handle empty graph case
        if len(g.nodes()) == 0:
            return {}

        if layout_mode == GraphVisualizerLayoutMode.HOMOGENEOUS:
            # For homogeneous graphs, use layouts that work well for general graph structure
            num_nodes = len(g.nodes())

            if num_nodes <= 30:
                # Small to medium graphs - use Kamada-Kawai (good for showing structure)
                try:
                    # Increase scale significantly to prevent node overlap (node_size=500)
                    return nx.kamada_kawai_layout(g, scale=15)
                except Exception as e:
                    print(
                        f"Kamada-Kawai layout failed: {e}, falling back to spring layout"
                    )
                    # Fallback to spring layout if kamada_kawai fails
                    # Increase k (ideal distance) and scale to prevent overlap
                    k = max(4.0, num_nodes / 3.0)
                    return nx.spring_layout(g, seed=seed, k=k, iterations=300, scale=15)
            else:
                # Large graphs - use spring layout with good parameters
                k = max(3.0, num_nodes / 6.0)
                return nx.spring_layout(g, seed=seed, k=k, iterations=250, scale=20)

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
                return nx.circular_layout(g, scale=15)
            elif num_types == 2:
                # Two types - use bipartite layout with more spacing
                types = list(type_to_nodes.keys())
                first_type_nodes = set(type_to_nodes[types[0]])
                return nx.bipartite_layout(g, first_type_nodes, scale=15)
            else:
                # Multiple types or fallback - use spring layout with much more spacing
                k = max(
                    4.0, len(g.nodes()) / 3.0
                )  # Dynamic spacing based on node count
                return nx.spring_layout(g, seed=seed, k=k, iterations=300, scale=15)

        else:
            raise ValueError(f"Invalid layout mode: {layout_mode}")

    @staticmethod
    def visualize_graph(
        data: HeteroData,
        seed=42,
        layout_mode=GraphVisualizerLayoutMode.BIPARTITE,
        subgraph_node_to_unenumerated_node_id_map: Optional[
            FrozenDict[Node, Node]
        ] = None,
        # pos_edges is a dictionary of edge type (src_node_type, relation, dst_node_type) to list of (src_node_id, dst_node_id) pairs
        pos_edges: Optional[dict[tuple[str, str, str], list[tuple[int, int]]]] = None,
        global_root_node: Optional[tuple[int, str]] = None,
    ):
        """
        Warning: This is mostly just AI slop, but it serves the purpose for now.
        Visualize a graph.

        Args:
            data: The HeteroData object to visualize.
            seed: The seed for the random number generator - fix it to ensure reproducibility in visualizations
            layout_mode: Either GraphVisualizerLayoutMode.HOMOGENEOUS or GraphVisualizerLayoutMode.BIPARTITE
            subgraph_node_to_global_node_mapping: A mapping from local node indices to global node indices.
        """

        # Build a mapping from global node indices to node types BEFORE conversion
        node_index_to_type = {}
        current_index = 0

        # HeteroData stores nodes by type - we need to map the global indices
        # that NetworkX will use back to the original node types
        for node_type in data.node_types:
            if hasattr(data[node_type], "num_nodes"):
                num_nodes = data[node_type].num_nodes
                for i in range(num_nodes):
                    node_index_to_type[current_index] = node_type
                    current_index += 1

        # Convert to NetworkX
        g = torch_geometric.utils.to_networkx(data)
        if subgraph_node_to_unenumerated_node_id_map:
            mapping = {}
            new_node_index_to_type = {}
            for node in g.nodes():
                node_type = node_index_to_type.get(node, "unknown")
                local_node = Node(type=node_type, id=node)
                unenumerated_node: Node = subgraph_node_to_unenumerated_node_id_map[
                    local_node
                ]
                mapping[node] = unenumerated_node.id
                # Preserve the node type information for the global node
                new_node_index_to_type[unenumerated_node.id] = unenumerated_node.type
            g = nx.relabel_nodes(g, mapping)
            # Update the node_index_to_type mapping to use global node types
            node_index_to_type = new_node_index_to_type  # type: ignore

        # Add positive edges to the graph if they don't already exist
        pos_edge_pairs = set()
        if pos_edges:
            for (
                src_node_type,
                relation,
                dst_node_type,
            ), edge_pairs in pos_edges.items():
                for src_id, dst_id in edge_pairs:
                    pos_edge_pairs.add((src_id, dst_id))

                    # Add nodes if they don't exist
                    if src_id not in g.nodes():
                        g.add_node(src_id)
                        node_index_to_type[src_id] = src_node_type
                    if dst_id not in g.nodes():
                        g.add_node(dst_id)
                        node_index_to_type[dst_id] = dst_node_type

                    # Add edge if it doesn't exist
                    if not g.has_edge(src_id, dst_id):
                        g.add_edge(src_id, dst_id)

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
                node_type_to_color[node_type] = GraphVisualizer.assign_node_color(
                    node_type
                )

            node_colors.append(node_type_to_color[node_type])

        # Create a larger figure for better node spacing
        plt.figure(figsize=(10, 6))

        # Generate a layout based on the selected mode
        # Get all unique node types actually present in the graph
        actual_node_types = set(node_index_to_type.values())
        pos = GraphVisualizer._create_type_grouped_layout(
            g, node_index_to_type, actual_node_types, seed, layout_mode
        )

        # Safety check: if pos is None or empty and we have nodes, create a fallback layout
        if pos is None or (len(g.nodes()) > 0 and not pos):
            print("Layout generation failed, using fallback spring layout")
            k = max(4.0, len(g.nodes()) / 3.0)
            pos = nx.spring_layout(g, seed=seed, k=k, iterations=300, scale=15)

        # Identify isolated nodes and root node for special styling
        isolated_nodes = [node for node in g.nodes() if g.degree(node) == 0]
        root_node_id = global_root_node[0] if global_root_node else None

        # Create border styling (red border for root node, thick black border for isolated nodes)
        node_edge_colors = []
        node_line_widths = []
        node_sizes = []

        for node in g.nodes():
            if node == root_node_id:
                node_edge_colors.append("#E53935")  # Red border for root node
                node_line_widths.append(4)  # Thick border for root node
                node_sizes.append(1000)  # Twice the size for root node
            elif node in isolated_nodes:
                node_edge_colors.append(BLACK)  # Black border for isolated nodes
                node_line_widths.append(3)
                node_sizes.append(500)  # Normal size
            else:
                node_edge_colors.append(CHARCOAL)  # Default border color
                node_line_widths.append(1)
                node_sizes.append(500)  # Normal size

        # Create edge type to color mapping
        edge_type_to_color = {}
        edge_colors = []

        # Extract edge types from the graph (now includes any added positive edges)
        for edge in g.edges():
            # Check if this is a positive edge
            is_positive_edge = (edge[0], edge[1]) in pos_edge_pairs

            if is_positive_edge:
                # Color positive edges red
                edge_colors.append("#E53935")  # Red color for positive edges
            else:
                # Get node types for source and destination
                src_node_type = node_index_to_type.get(edge[0], "unknown")
                dst_node_type = node_index_to_type.get(edge[1], "unknown")

                # Create edge type identifier
                edge_type = f"{src_node_type} → {dst_node_type}"

                # Look for a more specific edge type in HeteroData if available
                if hasattr(data, "edge_types") and data.edge_types:
                    for et in data.edge_types:
                        if len(et) == 3:  # (src_type, relation, dst_type)
                            if et[0] == src_node_type and et[2] == dst_node_type:
                                edge_type = f"{et[0]} --{et[1]}--> {et[2]}"
                                break
                        elif (
                            isinstance(et, tuple) and len(et) == 2
                        ):  # Some formats might be (src, dst)
                            if et[0] == src_node_type and et[1] == dst_node_type:
                                edge_type = f"{et[0]} → {et[1]}"
                                break

                # Assign color to edge type
                if edge_type not in edge_type_to_color:
                    edge_type_to_color[edge_type] = GraphVisualizer.assign_edge_color(
                        edge_type
                    )

                edge_colors.append(edge_type_to_color[edge_type])

        # Draw nodes first
        nx.draw_networkx_nodes(
            g,
            pos,
            node_color=node_colors if node_colors else "lightblue",  # type: ignore
            edgecolors=node_edge_colors if node_edge_colors else CHARCOAL,  # type: ignore
            linewidths=node_line_widths if node_line_widths else 1,  # type: ignore
            node_size=node_sizes if node_sizes else 500,  # type: ignore
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
                    alpha=0.9,  # Less transparent for cleaner look
                )
            else:
                # Curved edges for bipartite graphs to reduce overlap
                nx.draw_networkx_edges(
                    g,
                    pos,
                    edge_color=edge_colors,  # type: ignore
                    width=0.75,  # 75% of default edge width
                    alpha=0.8,  # Slightly transparent for better overlap visibility
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
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=node_type_to_color[node_type],
                        markersize=10,
                        label=f"Node: {node_type}",
                    )
                )

        # Add isolated node indicator
        if isolated_nodes:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markerfacecolor="white",
                    markeredgewidth=3,
                    markersize=10,
                    label="Isolated nodes",
                )
            )

        # Add root node indicator
        if global_root_node:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="#E53935",
                    markerfacecolor="white",
                    markeredgewidth=4,
                    markersize=15,  # Larger marker to represent the larger size
                    label="Root node",
                )
            )

        # Add positive edges to legend if they exist
        if pos_edges:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color="#E53935",
                    linewidth=3,
                    label="Positive edges",
                )
            )

        # Add edge types
        if edge_type_to_color:
            for edge_type in sorted(edge_type_to_color.keys()):
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=edge_type_to_color[edge_type],
                        linewidth=2,
                        label=f"Edge: {edge_type}",
                    )
                )

        if legend_elements:
            plt.legend(
                handles=legend_elements, loc="upper right", bbox_to_anchor=(1.4, 1)
            )

        plt.show()


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
