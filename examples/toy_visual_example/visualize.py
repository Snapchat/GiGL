import networkx as nx
from torch_geometric.data import HeteroData
import torch_geometric.utils


class GraphVisualizer:
    """
    Used to build and visualize graph which is user configured in a yaml file.
    """

    @staticmethod
    def visualize_graph(data: HeteroData):
        g = torch_geometric.utils.to_networkx(data)

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

        def assign_color(name: str) -> str:
            """Assign a color to a name based on hash and a fixed palette."""
            return fixed_colors[hash(name) % len(fixed_colors)]

        node_colors = [assign_color(node) for node in g.nodes()]

        # Generate a static layout
        pos = nx.spring_layout(g, seed=42)
        for node in g.nodes():
            g.nodes[node]['label'] = node
        nx.draw(g, pos, with_labels=True, node_color=node_colors, node_size=500, font_size=10, font_weight='bold')
