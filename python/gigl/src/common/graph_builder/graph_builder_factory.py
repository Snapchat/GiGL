from gigl.src.common.graph_builder.abstract_graph_builder import GraphBuilder
from gigl.src.common.graph_builder.pyg_graph_builder import PygGraphBuilder
from gigl.src.common.types.model import GraphBackend


class GraphBuilderFactory:
    """
    Instantiates a `GraphBuilder` object based on valid `GraphBackend` names
    """

    @classmethod
    def get_graph_builder(cls, backend_name: GraphBackend) -> GraphBuilder:
        if backend_name == GraphBackend.PYG:
            return PygGraphBuilder()
        else:
            raise ValueError(
                f"{backend_name} is not valid. backend_name can be one of {[gb.value for gb in GraphBackend]}"
            )
